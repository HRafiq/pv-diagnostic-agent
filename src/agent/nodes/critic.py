"""The critic. A structured verdict, and no way out through prose.

A critic that returns "looks good" is worse than no critic, because it launders
an unchecked answer as a reviewed one. So this node cannot return prose:
`CriticVerdict` has no free-text verdict field, and the loop acts on the
enumerated value.

The part worth defending is how much of the verdict is **not** the model's to
decide. Three things are computed here and then overridden onto whatever the
model returned:

1. **Unsupported claims** come from the deterministic grounding check. A model
   that inspects its own arithmetic and pronounces it sound is not a check. If
   a figure is not in a tool's ledger it is unsupported, and the critic cannot
   clear it by saying otherwise.
2. **Look-alikes counted as checked** must have had a tool run that actually
   discriminates them. Asserting "I considered clipping" without ever measuring
   whether output sits on a ceiling is box-ticking, and the checklist exists
   precisely to stop that.
3. **A verdict that cannot be constructed becomes `send_back`.** A critic that
   returns something the contract rejects has not reviewed the answer, and the
   honest consequence is another cycle rather than a silent accept.

Together these mean the critic can be *stricter* than the model wanted but never
looser, which is the only direction that is safe.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.agent.grounding import check_numeric_grounding
from src.agent.llm import LLMClient, LLMResponse
from src.agent.nodes.prompts import (
    CRITIC_SYSTEM,
    evidence_digest,
    hypothesis_digest,
)
from src.agent.nodes.synthesizer import Synthesis, ledger_of
from src.agent.state import LOOKALIKE_CHECKLIST, AgentState, CriticVerdict
from src.tools import REGISTRY, ToolResult

__all__ = [
    "CRITIC_SCHEMA",
    "Review",
    "lookalikes_measured",
    "review",
]


CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict",
        "hypotheses_considered",
        "hypotheses_excluded",
        "hypotheses_still_standing",
        "unsupported_claims",
        "lookalikes_considered",
        "revision_request",
    ],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["accept", "send_back", "not_enough_evidence"],
            "description": (
                "'not_enough_evidence' when two or more causes genuinely "
                "survive and no remaining measurement separates them."
            ),
        },
        "hypotheses_considered": {"type": "array", "items": {"type": "string"}},
        "hypotheses_excluded": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hypothesis", "excluded_by", "reasoning"],
                "properties": {
                    "hypothesis": {"type": "string"},
                    "excluded_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Names of the tools whose result did the excluding."
                        ),
                    },
                    "reasoning": {"type": "string"},
                },
            },
        },
        "hypotheses_still_standing": {"type": "array", "items": {"type": "string"}},
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Claims in the answer with no measurement behind them.",
        },
        "lookalikes_considered": {
            "type": "array",
            "items": {"type": "string", "enum": list(LOOKALIKE_CHECKLIST)},
            "description": (
                "Which of the standing look-alikes this investigation actually "
                "weighed. Naming one you did not measure will not count it."
            ),
        },
        "revision_request": {
            "type": "string",
            "description": (
                "Required on send_back. Must name a cause and a tool: 'exclude "
                "curtailment by checking the ceiling level against the "
                "inverter rating', not 'consider other causes'. Empty otherwise."
            ),
        },
    },
}


def lookalikes_measured(tools_called: list[str]) -> list[str]:
    """Look-alikes some tool that discriminates them actually measured.

    Derived from the tool registry, so it cannot be inflated by anything the
    model says — which is the whole guarantee the checklist is for.

    This used to be intersected with the look-alikes the critic *named*, on the
    reasoning that a model will happily claim all seven. That reasoning is
    right and the conclusion was wrong: a claim nobody can verify carries no
    information, so intersecting with it cannot remove a false positive — only
    a true one. It did, constantly. A reviewer looking at a string fault names
    the three look-alikes that bear on it, so one unnamed item made `accept`
    unreachable: ten reviews across the first four scored cases, zero accepts,
    every case to the cycle cap, and one correct answer rejected into a wrong
    abstention.

    The measurement is the fact. The claim is now ignored entirely.
    """
    measured: set[str] = set()
    for name in tools_called:
        spec = REGISTRY.get(name)
        if spec is not None:
            measured |= set(spec.discriminates)
    return [item for item in LOOKALIKE_CHECKLIST if item in measured]


def lookalikes_actually_checked(
    tools_called: list[str], claimed: list[str]
) -> list[str]:
    """Look-alikes the run both claimed to weigh and took a measurement about.

    The intersection, not the union. A look-alike counts as checked only if some
    tool that discriminates it was actually run — otherwise the checklist is a
    list of things the model said, and a model will happily say all seven.
    """
    measured: set[str] = set()
    for name in tools_called:
        spec = REGISTRY.get(name)
        if spec is not None:
            measured |= set(spec.discriminates)
    return [
        item for item in LOOKALIKE_CHECKLIST if item in measured and item in claimed
    ]


class Review:
    """The critic's verdict plus what it cost."""

    def __init__(self, verdict: CriticVerdict, response: LLMResponse | None) -> None:
        self.verdict = verdict
        self.response = response


def review(
    client: LLMClient,
    state: AgentState,
    brief: str,
    results: list[ToolResult],
    errors: list[str],
    synthesis: Synthesis,
) -> Review:
    """Review one draft answer and return a structured verdict."""
    draft = "\n".join(
        [
            f"settled: {synthesis.settled}",
            f"category: {synthesis.category or '(none)'}",
            f"cause: {synthesis.cause or '(none)'}",
            "confidence: "
            + (
                f"{synthesis.confidence}"
                if synthesis.confidence is not None
                else "(none)"
            ),
            "candidate causes still standing: "
            + (
                ", ".join(c.cause for c in synthesis.candidate_causes)
                or "(none listed)"
            ),
            f"resolving measurement: {synthesis.resolving_measurement or '(none)'}",
            f"recommended action: {synthesis.recommended_action}",
            "",
            synthesis.answer,
        ]
    )

    user = "\n".join(
        [
            brief,
            "",
            "CANDIDATE CAUSES",
            hypothesis_digest(state),
            "",
            "EVERY MEASUREMENT TAKEN",
            evidence_digest(results, errors),
            "",
            "MEASUREMENTS RUN, IN ORDER",
            ", ".join(state.tools_called) or "(none)",
            "",
            "THE DRAFT ANSWER",
            draft,
            "",
            "LOOK-ALIKES THAT MUST BE WEIGHED ON EVERY INVESTIGATION",
            ", ".join(LOOKALIKE_CHECKLIST),
        ]
    )

    response = client.complete(
        node="critic", system=CRITIC_SYSTEM, user=user, schema=CRITIC_SCHEMA
    )
    payload = response.parsed or {}

    # --- the parts the model does not get to decide ------------------------
    grounding = check_numeric_grounding(
        "\n".join([synthesis.answer, synthesis.summary]),
        ledger_of(results),
        # What the synthesiser was shown, carried on the draft. Rebuilding it
        # here would leave out the knowledge text — which the critic is not
        # given — and the physics the agent was handed would be scored as
        # invented. That is not hypothetical: it cost a correct answer.
        quotable=synthesis.citable,
    )
    # Two kinds of objection, deliberately not merged.
    #
    # `unsupported` is what the *arithmetic* found: a figure in the prose that
    # appears in no tool's provenance ledger. Objective, verifiable, and a hard
    # veto — "no fabricated numerics reach the interface" is the guarantee this
    # whole node exists to keep. A build error is the same kind of thing: the
    # answer could not be constructed at all.
    #
    # `observations` is what the *reviewer* said. Also valuable, and it drives
    # the revision request — but it is prose judgement, not a measurement, and
    # it cannot veto on its own.
    #
    # These used to be one list, and any entry in it blocked `accept`. A
    # competent reviewer always finds something to say, so the critic
    # disqualified every answer it reviewed by doing its job properly: ten
    # reviews across four cases, zero accepts, and one correct answer pushed
    # into a wrong abstention. The observations were good — one of them
    # independently identified a real defect in the fault injector — which is
    # exactly why they must inform the next cycle rather than end it.
    unsupported = list(dict.fromkeys(grounding.as_claims()))
    if synthesis.build_error:
        unsupported.append(synthesis.build_error)

    observations = [
        str(c)
        for c in payload.get("unsupported_claims", [])
        if str(c).strip() and str(c) not in unsupported
    ]

    # What the run measured, regardless of what the critic thought to mention.
    # See `lookalikes_measured` for why the critic's own list is not consulted.
    measured = lookalikes_measured(list(state.tools_called))

    still_standing = [str(x) for x in payload.get("hypotheses_still_standing", [])]
    wanted = str(payload.get("verdict", "send_back"))
    request = str(payload.get("revision_request") or "").strip() or None

    # --- the verdict, tightened but never loosened -------------------------
    verdict = wanted
    unmeasured = sorted(set(LOOKALIKE_CHECKLIST) - set(measured))
    if wanted == "accept" and (unsupported or unmeasured):
        verdict = "send_back"
        request = request or _repair_request(unsupported, unmeasured)
    elif wanted == "accept" and observations:
        # Accepted with reservations. The reviewer's points are recorded on the
        # verdict and shown, but a reviewer that finds something to say is not
        # by itself grounds to reject — that is the difference between a review
        # and a veto.
        request = None
    if wanted == "not_enough_evidence" and len(still_standing) < 2:
        # Declining to commit needs two survivors. With fewer, the answer is
        # either settled or the review itself is incoherent; another cycle is
        # the honest response to both.
        verdict = "send_back"
        request = request or (
            "the review declined to commit but named fewer than two surviving "
            "causes; either exclude the remaining one on evidence or name the "
            "second survivor and the measurement that would separate them"
        )
    if verdict == "send_back" and not request:
        # The reviewer's own words are the best available instruction; falling
        # back to boilerplate throws away the one thing it produced.
        request = (
            "address these: " + "; ".join(observations[:3])
            if observations
            else (
                "the review asked for changes without saying what; re-examine "
                "the strongest surviving cause and name the measurement that "
                "would exclude it"
            )
        )

    excluded = [
        {
            "hypothesis": str(item.get("hypothesis", "")),
            "excluded_by": [str(x) for x in item.get("excluded_by", [])]
            or ["(unnamed)"],
            "reasoning": str(item.get("reasoning", "")),
        }
        for item in payload.get("hypotheses_excluded", [])
        if str(item.get("hypothesis", "")).strip()
    ]

    try:
        built = CriticVerdict(
            hypotheses_considered=[
                str(x) for x in payload.get("hypotheses_considered", [])
            ],
            hypotheses_excluded=excluded,
            hypotheses_still_standing=still_standing,
            unsupported_claims=unsupported,
            observations=observations,
            lookalikes_checked=measured,
            verdict=verdict,
            revision_request=request if verdict == "send_back" else None,
        )
    except ValidationError as exc:
        # A critic that returns something the contract rejects has not reviewed
        # the answer. Falling through to accept would be the exact failure this
        # node exists to prevent, so it becomes another cycle instead.
        built = CriticVerdict(
            hypotheses_considered=[h.id for h in state.hypotheses],
            hypotheses_still_standing=[h.id for h in state.still_standing],
            unsupported_claims=unsupported,
            observations=observations,
            lookalikes_checked=measured,
            verdict="send_back",
            revision_request=(
                "the review could not be recorded in a usable form "
                f"({_first_error(exc)}); redo the answer, stating for each "
                "surviving cause which measurement bears on it"
            ),
        )

    return Review(verdict=built, response=response)


def _repair_request(unsupported: list[str], missing_lookalikes: list[str]) -> str:
    parts = []
    if unsupported:
        parts.append(
            "these claims have no measurement behind them: "
            + "; ".join(unsupported[:3])
        )
    if missing_lookalikes:
        parts.append(
            "these look-alikes were never measured against: "
            + ", ".join(missing_lookalikes)
        )
    return "; ".join(parts) + ". Take the measurements or drop the claims."


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    return str(errors[0].get("msg", exc)) if errors else str(exc)
