"""Agent state, the critic contract, and the saved-analysis spec.

The critic is the piece most likely to rot into decoration. A critic that
returns "looks good" is worse than no critic, because it launders an unchecked
answer as a reviewed one. So `CriticVerdict` has no free-text verdict field:
it must name what it considered, what it excluded and on what evidence, what is
still standing, and which look-alikes it checked.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "LOOKALIKE_CHECKLIST",
    "AgentState",
    "AnalysisSpec",
    "CriticVerdict",
    "ExclusionRecord",
    "Hypothesis",
    "HypothesisStatus",
]

# Checked on every run, without exception. An unchecked item alone is grounds
# for send_back. These are the things that look like faults and are not — they
# are the evaluation (handoff §5.1), not an afterthought.
LOOKALIKE_CHECKLIST: tuple[str, ...] = (
    "weather",
    "seasonal_temperature_derating",
    "clipping",
    "curtailment",
    "sensor_drift",
    "snow_or_dust_event",
    "telemetry_gap",
)

HypothesisStatus = Literal["open", "excluded", "supported", "still_standing"]


class Hypothesis(BaseModel):
    """A candidate cause under test."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="e.g. 'H1'")
    cause: str
    status: HypothesisStatus = "open"
    consequence_if_true: str = Field(
        ...,
        description="What it costs to act on this one if it is true.",
    )
    discriminating_measurements: list[str] = Field(
        default_factory=list,
        description=(
            "Tools that would separate this cause from its look-alikes. The "
            "planner orders hypotheses by discriminating power, so this is the "
            "field that drives what gets measured next."
        ),
    )


class ExclusionRecord(BaseModel):
    """Why a hypothesis was ruled out, and on what evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypothesis: str
    excluded_by: list[str] = Field(
        ..., min_length=1, description="Tool call ids that did the excluding."
    )
    reasoning: str


class CriticVerdict(BaseModel):
    """Structured verdict. There is no prose-only path out of the critic."""

    model_config = ConfigDict(extra="forbid")

    hypotheses_considered: list[str]
    hypotheses_excluded: list[ExclusionRecord] = Field(default_factory=list)
    hypotheses_still_standing: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Figures in the answer that appear in no tool's provenance ledger, "
            "found by arithmetic rather than judgement. A hard veto on accept: "
            "this is the 'no fabricated numerics' guarantee."
        ),
    )
    observations: list[str] = Field(
        default_factory=list,
        description=(
            "What the reviewer said, in its own words. Recorded and shown, and "
            "it drives the revision request — but it does not veto. These were "
            "once merged into `unsupported_claims`, and because a competent "
            "reviewer always finds something to say, the critic disqualified "
            "every answer it reviewed by doing its job properly."
        ),
    )
    lookalikes_checked: list[str] = Field(default_factory=list)
    verdict: Literal["accept", "send_back", "not_enough_evidence"]
    revision_request: str | None = Field(
        default=None,
        description=(
            "Required on send_back, and must be actionable: 'exclude "
            "curtailment independently by comparing the same period last "
            "year', not 'consider other causes'."
        ),
    )

    @property
    def unchecked_lookalikes(self) -> tuple[str, ...]:
        checked = set(self.lookalikes_checked)
        return tuple(item for item in LOOKALIKE_CHECKLIST if item not in checked)

    @model_validator(mode="after")
    def _enforce_verdict_contract(self) -> CriticVerdict:
        if self.verdict == "send_back" and not self.revision_request:
            raise ValueError(
                "send_back must carry a concrete revision request naming which "
                "hypothesis and which tool; otherwise the loop cannot act on it"
            )
        if self.verdict == "accept":
            if self.unsupported_claims:
                raise ValueError(
                    "cannot accept an answer that still has unsupported "
                    "claims: " + ", ".join(self.unsupported_claims)
                )
            if self.unchecked_lookalikes:
                raise ValueError(
                    "cannot accept with look-alikes unchecked: "
                    + ", ".join(self.unchecked_lookalikes)
                )
        if (
            self.verdict == "not_enough_evidence"
            and len(self.hypotheses_still_standing) < 2
        ):
            raise ValueError(
                "not_enough_evidence means two or more causes survived every "
                "available measurement; with fewer than two the answer is "
                "settled and should be accepted"
            )
        return self


class AgentState(BaseModel):
    """The object the planner/router/executor/synthesiser/critic loop mutates.

    Deliberately framework-independent: the plain-Python loop (step 4) and the
    LangGraph port (step 12) both operate on this, which is what makes the
    comparison in docs/LANGGRAPH_TRADEOFF.md a real one.
    """

    model_config = ConfigDict(extra="forbid")

    investigation_id: str
    question: str
    scope: str
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    planned_tools: list[str] = Field(
        default_factory=list,
        description=(
            "The initial plan. Any tool call outside this list is recorded "
            "with was_planned=False — that is the agency signal."
        ),
    )
    tools_called: list[str] = Field(default_factory=list)
    cycle: int = Field(default=0, ge=0, description="Planner<->critic cycles used.")
    max_cycles: int = Field(default=4, ge=1)
    verdicts: list[CriticVerdict] = Field(default_factory=list)
    answer: str | None = None
    settled: bool = False

    @property
    def cap_reached(self) -> bool:
        return self.cycle >= self.max_cycles

    @property
    def unplanned_tool_calls(self) -> list[str]:
        planned = set(self.planned_tools)
        return [tool for tool in self.tools_called if tool not in planned]

    @property
    def still_standing(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.status in ("open", "still_standing")]


class AnalysisSpec(BaseModel):
    """A saved analysis (handoff §3.6). A config row — never generated code.

    v1 supports level 1 only: a new question plus a selection over existing
    tools. Not a new tool, not new orchestration. The interesting constraint is
    the last field: saving requires at least one golden case, so the registry
    cannot fill up with unevaluated analyses.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    question: str = Field(..., description="May contain {slots}.")
    allowed_tools: list[str] = Field(..., min_length=1)
    scope: Literal["plant", "block", "inverter", "fleet"]
    trigger: Literal["manual", "on_new_data", "scheduled"]
    golden_case_ids: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Enforced at save time. An agent factory that produces unevaluated "
            "agents defeats the point of the project."
        ),
    )
