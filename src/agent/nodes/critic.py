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

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from src.agent.grounding import check_numeric_grounding
from src.agent.llm import LLMClient, LLMResponse
from src.agent.nodes.prompts import (
    CRITIC_SYSTEM,
    evidence_digest,
    hypothesis_digest,
    lookalike_coverage,
    lookalike_coverage_text,
)
from src.agent.nodes.synthesizer import Synthesis, ledger_of
from src.agent.state import LOOKALIKE_CHECKLIST, AgentState, CriticVerdict
from src.tools import ToolResult, tool_names

__all__ = [
    "CRITIC_SCHEMA",
    "MechanicalObjections",
    "Review",
    "inspect_draft",
    "lookalikes_measured",
    "review",
    "unrun_tools_named_in",
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
        "previous_request_addressed",
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
        "previous_request_addressed": {
            "type": "string",
            "enum": ["yes", "no", "no_previous_request"],
            "description": (
                "Did the investigation do what you asked for last time? "
                "'no_previous_request' on the first review. Answering 'yes' and "
                "then sending back for something else is how a review becomes "
                "an unbounded loop — say so only if a specific measurement "
                "would change which cause is named."
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

    Expressed against `lookalike_coverage()` rather than re-walking the registry
    here, because that same mapping is what the planner is shown in the brief.
    The enforcement and the instruction now read one dict: a tool whose
    `discriminates` changes moves both at once, and neither can quietly become
    stricter than the other.
    """
    called = set(tools_called)
    coverage = lookalike_coverage()
    return [item for item in LOOKALIKE_CHECKLIST if called & set(coverage[item])]


def unrun_tools_named_in(text: str | None, tools_called: list[str]) -> list[str]:
    """Measurement tools the text says are needed and the run never took.

    The discriminator between an honest abstention and an unfinished one.
    `not_enough_evidence` is a first-class outcome (CLAUDE.md) *when the evidence
    genuinely cannot be obtained* — G-039's resolving measurement is the grid
    operator's dispatch log, which is outside the telemetry and outside the tool
    set, so declining there is the right answer. G-017's was "run
    `string_onset_scan`", a tool in its own registry that it had simply not got
    to before the per-cycle measurement cap. Those two are not the same
    outcome, and until this function existed nothing told them apart.

    A substring match is enough because tool names are long, snake_cased and
    unlike ordinary prose: `string_onset_scan` does not occur in a sentence by
    accident.
    """
    called = set(tools_called)
    said = text or ""
    return [n for n in tool_names() if n in said and n not in called]


def _what_you_asked_last_time(
    previous: CriticVerdict | None,
    previous_answer: str | None,
    measurements_since: int,
) -> list[str]:
    """The critic's own history, which it could not previously see."""
    if previous is None:
        return []

    lines = [
        "",
        "YOUR OWN PREVIOUS REVIEW OF THIS INVESTIGATION",
        f"You returned '{previous.verdict}' and asked for: "
        + (previous.revision_request or "(nothing specific)"),
        f"Since then the investigation took {measurements_since} further "
        "measurement(s), listed above.",
    ]
    if previous_answer is not None:
        lines.append(
            f"The previous draft committed to: {previous_answer}. "
            "Compare it with the draft below."
        )
    lines.append(
        "Judge whether what you asked for was done. If it was, and the answer "
        "is unchanged, that is evidence the investigation has converged and "
        "should be accepted — not a reason to find something new. Asking for "
        "one more thing is always possible and is not free: every send_back "
        "costs another full cycle. Send back only if a *specific* measurement "
        "would change which cause is named."
    )
    return lines


@dataclass(frozen=True)
class MechanicalObjections:
    """Everything wrong with a draft that needs no model to establish.

    Factored out of `review` so the convergence stop in `loop_plain` can apply
    the identical guarantees without paying for a review. An answer that skips
    the critic must not thereby skip "no fabricated numerics" or "every
    look-alike was weighed" — those are arithmetic, and arithmetic is free.
    """

    unsupported: list[str]
    unmeasured: list[str]
    avoidable: list[str]

    @property
    def clean(self) -> bool:
        return not (self.unsupported or self.unmeasured or self.avoidable)

    def as_request(self) -> str | None:
        """The repair instruction, or None when there is nothing to repair."""
        if self.avoidable:
            return (
                "the answer declines to commit but names "
                + ", ".join(self.avoidable)
                + " as what would resolve it, and that measurement was never "
                "taken. Run it. An abstention is only honest when the evidence "
                "cannot be obtained, not when it has not been collected yet."
            )
        if self.unsupported or self.unmeasured:
            return _repair_request(self.unsupported, self.unmeasured)
        return None


def inspect_draft(
    state: AgentState, results: list[ToolResult], synthesis: Synthesis
) -> MechanicalObjections:
    """The deterministic half of a review. No LLM call, no judgement."""
    grounding = check_numeric_grounding(
        "\n".join([synthesis.answer, synthesis.summary]),
        ledger_of(results),
        quotable=synthesis.citable,
    )
    unsupported = list(dict.fromkeys(grounding.as_claims()))
    if synthesis.build_error:
        unsupported.append(synthesis.build_error)

    measured = lookalikes_measured(list(state.tools_called))
    unmeasured = sorted(set(LOOKALIKE_CHECKLIST) - set(measured))

    avoidable = (
        unrun_tools_named_in(synthesis.resolving_measurement, list(state.tools_called))
        if not synthesis.settled
        else []
    )
    return MechanicalObjections(
        unsupported=unsupported, unmeasured=unmeasured, avoidable=avoidable
    )


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
    previous: CriticVerdict | None = None,
    previous_answer: str | None = None,
    measurements_since: int = 0,
) -> Review:
    """Review one draft answer and return a structured verdict.

    Args:
        previous: The verdict this critic returned on the last cycle, if any.
        previous_answer: The cause the last cycle committed to, for comparison.
        measurements_since: How many measurements were taken in response.

    Those three arguments are the fix for the failure that dominated every run
    of this project. The critic's prompt held the brief, the hypotheses, the
    evidence, the draft and the checklist — and **nothing about its own previous
    review**. Meanwhile the planner and synthesiser both receive
    `revision_request`, so information flowed one way and nothing came back.

    Every review was therefore a fresh reviewer meeting the case for the first
    time, with unlimited standards and no memory. It could not say "you
    addressed my concern", because it did not know it had one; it re-raised the
    same point in different words; and it could not notice that the answer had
    not changed, which is the strongest available evidence of convergence. A
    competent reviewer always finds something and nothing priced another cycle,
    so `send_back` was the equilibrium rather than an accident.

    Observed four times: G-001, G-005, G-006 and G-017 each reached the right
    answer and were talked out of it. G-017's two send_backs were "reconcile the
    sharp onset detected by characterize_onset" and then "split the string-1
    evidence around the 2017-03-16" — the same objection, asked twice, after the
    first had been answered with three more measurements.
    """
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
            lookalike_coverage_text(),
            *_what_you_asked_last_time(previous, previous_answer, measurements_since),
        ]
    )

    response = client.complete(
        node="critic", system=CRITIC_SYSTEM, user=user, schema=CRITIC_SCHEMA
    )
    payload = response.parsed or {}

    # --- the parts the model does not get to decide ------------------------
    # The deterministic half, shared with `loop_plain`'s convergence stop so the
    # two paths cannot enforce different guarantees.
    mechanical = inspect_draft(state, results, synthesis)
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
    unsupported = mechanical.unsupported

    observations = [
        str(c)
        for c in payload.get("unsupported_claims", [])
        if str(c).strip() and str(c) not in unsupported
    ]

    # What the run measured, regardless of what the critic thought to mention.
    # See `lookalikes_measured` for why the critic's own list is not consulted.
    measured = lookalikes_measured(list(state.tools_called))
    unmeasured = mechanical.unmeasured

    still_standing = [str(x) for x in payload.get("hypotheses_still_standing", [])]
    wanted = str(payload.get("verdict", "send_back"))
    request = str(payload.get("revision_request") or "").strip() or None

    # --- the verdict, tightened but never loosened -------------------------
    verdict = wanted
    if wanted == "accept" and (unsupported or unmeasured):
        verdict = "send_back"
        request = request or _repair_request(unsupported, unmeasured)
    elif wanted == "accept" and observations:
        # Accepted with reservations. The reviewer's points are recorded on the
        # verdict and shown, but a reviewer that finds something to say is not
        # by itself grounds to reject — that is the difference between a review
        # and a veto.
        request = None
    # An abstention that names a measurement the agent could have taken is not
    # an abstention; it is an unfinished investigation wearing one. G-017 spent
    # its whole per-cycle measurement budget covering the look-alike checklist,
    # reached 7/7, then declined to answer — giving "run `string_onset_scan`" as
    # what would settle it, a tool in its own registry. Nothing stopped that,
    # and it scored as a missed fault on a case it had answered correctly in an
    # earlier run.
    #
    # Deliberately applied to `accept` as well: a reviewer that waves through
    # an abstention with an available next step has made the same mistake as one
    # that wrote it.
    avoidable = mechanical.avoidable
    if verdict != "send_back" and avoidable:
        verdict = "send_back"
        # This message replaces rather than defers to the model's own, because
        # it is the concrete blocking reason and the next cycle has to act on
        # exactly it.
        request = (
            "the answer declines to commit but names "
            + ", ".join(avoidable)
            + " as what would resolve it, and that measurement was never taken. "
            "Run it. An abstention is only honest when the evidence cannot be "
            "obtained, not when it has not been collected yet."
        )
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
            previous_request_addressed=str(
                payload.get("previous_request_addressed") or "no_previous_request"
            ),
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
