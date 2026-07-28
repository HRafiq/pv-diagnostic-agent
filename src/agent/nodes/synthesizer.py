"""The synthesiser: turn measurements into a finding, or into an honest refusal.

The `Finding` model refuses to hold an unsettled answer that also shows a single
cause and a confidence. That validator is the last line, not the first: this
node strips those fields *before* construction when the answer is unsettled,
because the safest thing to do with a cause the interface must not show is to
drop it.

What this node will not do is the opposite — invent what is missing. An unsettled
answer needs two surviving causes and a resolving measurement; if the model
returned one cause and no test, no `Finding` is built and the loop records that
the synthesis was unusable. Manufacturing a second candidate to satisfy a
validator would be the same dishonesty in the other direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.agent.grounding import (
    GroundingReport,
    check_numeric_grounding,
    quotable_values,
)
from src.agent.llm import LLMClient, LLMResponse
from src.agent.nodes.prompts import (
    SYNTHESIZER_SYSTEM,
    evidence_digest,
    hypothesis_digest,
)
from src.agent.state import AgentState
from src.findings.models import CandidateCause, Finding
from src.tools import ToolResult

__all__ = ["SYNTHESIS_SCHEMA", "Synthesis", "ledger_of", "synthesize"]

_CATEGORIES = ["fault", "recoverable", "by_design", "not_the_plant"]


def _cause_vocabulary() -> list[str]:
    """The canonical cause names, read from the knowledge base.

    `cause` is a *label*, not an explanation. It was free text, and the model
    used it both ways — `shading` on one case, "A low-angle morning obstruction
    (trees, structure, or an adjacent row) is shading strings 1, 2 and…" on the
    next. Scoring compares it by exact equality against a canonical token
    (`eval/metrics.py`), so the prose form scored zero however right it was,
    while the rules engine — which picks from a fixed vocabulary — scored
    normally. The comparison between the two was decided by formatting.

    The explanation still has somewhere to live: `answer` and `summary` are
    free text and are where the reasoning belongs. This field says *which*
    cause, in the same words for both engines.

    Read from the knowledge base rather than restated here so the vocabulary
    cannot drift from the signatures the agent is shown.
    """
    from src.knowledge import load_knowledge

    return sorted(load_knowledge().signatures)


CAUSE_VOCABULARY = _cause_vocabulary()


SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "settled",
        "category",
        "cause",
        "confidence",
        "candidate_causes",
        "resolving_measurement",
        "title",
        "summary",
        "answer",
        "recommended_action",
        "energy_at_stake_kwh",
        "evidence",
    ],
    "properties": {
        "settled": {
            "type": "boolean",
            "description": "True only if one cause survived every measurement taken.",
        },
        "category": {
            "type": "string",
            "enum": [*_CATEGORIES, ""],
            "description": "Empty string when not settled.",
        },
        "cause": {
            "type": "string",
            "enum": [*CAUSE_VOCABULARY, ""],
            "description": (
                "Which cause, as one of the listed names. Empty string when not "
                "settled. The explanation goes in `answer`, not here."
            ),
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "0 when not settled; the field is not shown in that case.",
        },
        "candidate_causes": {
            "type": "array",
            "description": "Required when not settled: every cause still standing.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cause", "consequence_if_true"],
                "properties": {
                    "cause": {"type": "string", "enum": CAUSE_VOCABULARY},
                    "consequence_if_true": {
                        "type": "string",
                        "description": (
                            "What it means operationally if this one is true."
                        ),
                    },
                },
            },
        },
        "resolving_measurement": {
            "type": "string",
            "description": (
                "The cheap test that would settle it. Required when unsettled."
            ),
        },
        "title": {"type": "string", "description": "Under ten words, plain language."},
        "summary": {"type": "string", "description": "One to three sentences."},
        "answer": {
            "type": "string",
            "description": "The full explanation, quoting measured figures verbatim.",
        },
        "recommended_action": {
            "type": "string",
            "description": "Send someone, schedule something, or do nothing.",
        },
        "energy_at_stake_kwh": {
            "type": "number",
            "minimum": 0.0,
            "description": "Must be a figure a tool measured, or 0.",
        },
        "evidence": {
            "type": "array",
            "description": "Each claim and the measurement behind it.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "tool", "values_used"],
                "properties": {
                    "claim": {"type": "string"},
                    "tool": {"type": "string"},
                    "values_used": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class Synthesis:
    """The draft finding, plus what it can and cannot legitimately show."""

    settled: bool
    category: str | None
    cause: str | None
    confidence: float | None
    candidate_causes: list[CandidateCause]
    resolving_measurement: str | None
    title: str
    summary: str
    answer: str
    recommended_action: str
    energy_at_stake_kwh: float
    evidence: list[dict[str, Any]] = field(default_factory=list)
    grounding: GroundingReport = field(default_factory=GroundingReport)
    # Every number the model was shown when it wrote this draft. Carried on the
    # synthesis so the critic re-checks against the same material rather than
    # rebuilding it — the critic is not handed the knowledge text, and a
    # reconstruction that quietly omitted it would flag figures the synthesiser
    # had every right to quote.
    citable: tuple[float, ...] = ()
    response: LLMResponse | None = None
    build_error: str | None = None

    def to_finding(
        self,
        finding_id: str,
        detected_at: datetime,
        scope: str,
        investigation_id: str,
        lifecycle: str = "new",
    ) -> Finding:
        """Construct the `Finding`. Raises if the synthesis is not showable."""
        if self.build_error:
            raise ValueError(self.build_error)
        return Finding(
            id=finding_id,
            detected_at=detected_at,
            scope=scope,
            title=self.title,
            summary=self.summary,
            # An unsettled finding has no category to show; `not_the_plant` is
            # the honest bucket for "we cannot attribute this to the plant yet".
            category=self.category or "not_the_plant",
            lifecycle=lifecycle,
            settled=self.settled,
            cause=self.cause,
            confidence=self.confidence,
            candidate_causes=list(self.candidate_causes),
            resolving_measurement=self.resolving_measurement,
            energy_at_stake_kwh=max(0.0, self.energy_at_stake_kwh),
            energy_verified=self.settled,
            recommended_action=self.recommended_action,
            investigation_id=investigation_id,
        )


def ledger_of(results: list[ToolResult]) -> dict[str, float]:
    """The union of every tool's provenance ledger, namespaced by tool."""
    ledger: dict[str, float] = {}
    for result in results:
        for key, value in result.values.items():
            ledger[f"{result.tool}.{key}"] = value
    return ledger


def synthesize(
    client: LLMClient,
    state: AgentState,
    brief: str,
    results: list[ToolResult],
    errors: list[str],
    revision_request: str | None = None,
    knowledge: str = "",
) -> Synthesis:
    """Write the finding from what was measured."""
    parts = [
        brief,
        *(["", "WHAT IS KNOWN ABOUT THESE CAUSES", knowledge] if knowledge else []),
        "",
        "CANDIDATE CAUSES",
        hypothesis_digest(state),
        "",
        "EVERY MEASUREMENT TAKEN",
        evidence_digest(results, errors),
    ]
    if errors:
        parts += [
            "",
            "Measurements that could not be taken are listed above as FAILED. "
            "Treat those questions as unanswered, not as answered negatively.",
        ]
    if revision_request:
        parts += ["", "THE REVIEWER ASKED FOR", revision_request]

    response = client.complete(
        node="synthesizer",
        system=SYNTHESIZER_SYSTEM,
        user="\n".join(parts),
        schema=SYNTHESIS_SCHEMA,
    )
    payload = response.parsed or {}

    settled = bool(payload.get("settled", False))
    category = str(payload.get("category") or "").strip() or None
    cause = str(payload.get("cause") or "").strip() or None
    raw_confidence = payload.get("confidence")
    confidence = (
        float(raw_confidence) if isinstance(raw_confidence, int | float) else None
    )
    candidates = [
        CandidateCause(
            cause=str(item.get("cause", "")),
            consequence_if_true=str(item.get("consequence_if_true", "")),
        )
        for item in payload.get("candidate_causes", [])
        if str(item.get("cause", "")).strip()
    ]
    resolving = str(payload.get("resolving_measurement") or "").strip() or None

    build_error: str | None = None
    if settled:
        if category not in _CATEGORIES:
            build_error = (
                "the answer commits to a cause but names no valid category, so "
                "it cannot be filed or ranked"
            )
        if cause is None:
            build_error = "the answer claims to be settled but names no cause"
        if confidence is None:
            build_error = "the answer claims to be settled but carries no confidence"
        # A settled finding shows one answer. Anything left over would read as
        # hedging on a committed diagnosis.
        candidates = []
        resolving = None
    else:
        # The load-bearing strip. Whatever the model wrote, an unsettled finding
        # must not carry a cause or a confidence into the interface.
        cause = None
        confidence = None
        if len(candidates) < 2:
            build_error = (
                "the answer declines to commit but lists fewer than two "
                "surviving causes; with one survivor the answer is settled"
            )
        if not resolving:
            build_error = (
                "the answer declines to commit but names no measurement that "
                "would resolve it — an abstention without a next step is useless"
            )

    answer = str(payload.get("answer", ""))
    # `parts` is precisely what the model was shown and contains none of what it
    # wrote, so it is the honest quotation source.
    citable = tuple(quotable_values(*parts))
    grounding = check_numeric_grounding(
        "\n".join(
            [answer, str(payload.get("summary", "")), str(payload.get("title", ""))]
        ),
        ledger_of(results),
        quotable=citable,
    )

    return Synthesis(
        settled=settled,
        category=category,
        cause=cause,
        confidence=confidence,
        candidate_causes=candidates,
        resolving_measurement=resolving,
        title=str(payload.get("title", "")),
        summary=str(payload.get("summary", "")),
        answer=answer,
        recommended_action=str(payload.get("recommended_action", "")),
        energy_at_stake_kwh=float(payload.get("energy_at_stake_kwh") or 0.0),
        evidence=list(payload.get("evidence", [])),
        grounding=grounding,
        citable=citable,
        response=response,
        build_error=build_error,
    )
