"""The invariants that keep the interface honest.

Each test here corresponds to a rule in CLAUDE.md. They are the mechanism that
makes those rules enforceable rather than aspirational.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.agent.state import (
    LOOKALIKE_CHECKLIST,
    AgentState,
    AnalysisSpec,
    CriticVerdict,
    ExclusionRecord,
    Hypothesis,
)
from src.findings.models import CandidateCause, Finding
from src.trace.models import TraceStep

NOW = datetime(2019, 3, 14, tzinfo=UTC)

SETTLED = {
    "id": "F-118",
    "detected_at": NOW,
    "scope": "INV-03 / MPPT-2",
    "title": "One string offline",
    "summary": "MPPT-2 current dropped by a fifth in a single interval.",
    "category": "fault",
    "lifecycle": "new",
    "energy_at_stake_kwh": 4210.0,
    "recommended_action": "Send a technician to inspect the string fuses.",
    "investigation_id": "INV-2019-03-14-001",
}


# --------------------------------------------------------------------------
# Finding — the settled contract
# --------------------------------------------------------------------------
def test_settled_finding_is_valid() -> None:
    finding = Finding(
        **SETTLED,
        settled=True,
        cause="String fault",
        confidence=0.91,
        energy_verified=True,
    )
    assert finding.cause == "String fault"


def test_unsettled_finding_cannot_show_a_cause() -> None:
    # The bug this prevents: "soiling, confidence 0.78" on an investigation
    # that could not separate array soiling from sensor soiling.
    with pytest.raises(ValidationError, match="wash crew"):
        Finding(
            **SETTLED,
            settled=False,
            cause="Array soiling",
            candidate_causes=[
                CandidateCause(cause="Array soiling", consequence_if_true="wash"),
                CandidateCause(cause="Soiled pyranometer", consequence_if_true="wipe"),
            ],
            resolving_measurement="Compare measured POA against clear-sky",
            energy_verified=False,
        )


def test_unsettled_finding_cannot_show_a_confidence() -> None:
    with pytest.raises(ValidationError, match="confidence"):
        Finding(
            **SETTLED,
            settled=False,
            confidence=0.78,
            candidate_causes=[
                CandidateCause(cause="A", consequence_if_true="x"),
                CandidateCause(cause="B", consequence_if_true="y"),
            ],
            resolving_measurement="Paired clean/dirty soiling station",
            energy_verified=False,
        )


def test_unsettled_finding_needs_two_causes_and_a_resolving_measurement() -> None:
    with pytest.raises(ValidationError, match="at least two"):
        Finding(
            **SETTLED,
            settled=False,
            candidate_causes=[CandidateCause(cause="A", consequence_if_true="x")],
            resolving_measurement="something",
            energy_verified=False,
        )

    with pytest.raises(ValidationError, match="resolve it"):
        Finding(
            **SETTLED,
            settled=False,
            candidate_causes=[
                CandidateCause(cause="A", consequence_if_true="x"),
                CandidateCause(cause="B", consequence_if_true="y"),
            ],
            energy_verified=False,
        )


def test_unsettled_energy_cannot_be_verified() -> None:
    with pytest.raises(ValidationError, match="unverified"):
        Finding(
            **SETTLED,
            settled=False,
            candidate_causes=[
                CandidateCause(cause="A", consequence_if_true="x"),
                CandidateCause(cause="B", consequence_if_true="y"),
            ],
            resolving_measurement="something",
            energy_verified=True,
        )


def test_settled_finding_must_not_carry_candidates() -> None:
    with pytest.raises(ValidationError, match="candidate"):
        Finding(
            **SETTLED,
            settled=True,
            cause="String fault",
            confidence=0.9,
            energy_verified=True,
            candidate_causes=[CandidateCause(cause="A", consequence_if_true="x")],
        )


def test_settled_finding_needs_cause_and_confidence() -> None:
    with pytest.raises(ValidationError, match="cause"):
        Finding(**SETTLED, settled=True, confidence=0.9, energy_verified=True)
    with pytest.raises(ValidationError, match="confidence"):
        Finding(**SETTLED, settled=True, cause="String fault", energy_verified=True)


# --------------------------------------------------------------------------
# TraceStep — the agency signal
# --------------------------------------------------------------------------
def test_unplanned_step_must_state_its_reason() -> None:
    with pytest.raises(ValidationError, match="evidence of agency"):
        TraceStep(
            kind="tool", node="compare_to_same_period_last_year", was_planned=False
        )


def test_unplanned_step_with_a_reason_is_valid() -> None:
    step = TraceStep(
        kind="adaptive",
        node="compare_to_same_period_last_year",
        was_planned=False,
        reason_for_choosing=(
            "Clipping and curtailment produce the same ceiling, so the absence "
            "of clipping does not exclude curtailment. Comparing the same "
            "period last year separates them."
        ),
        excludes=["H3"],
    )
    assert step.excludes == ["H3"]


def test_planned_step_needs_no_reason() -> None:
    assert TraceStep(kind="tool", node="compute_pr", was_planned=True).result is None


# --------------------------------------------------------------------------
# CriticVerdict — no prose-only path out of the critic
# --------------------------------------------------------------------------
def _full_checklist() -> list[str]:
    return list(LOOKALIKE_CHECKLIST)


def test_send_back_needs_an_actionable_revision_request() -> None:
    with pytest.raises(ValidationError, match="revision request"):
        CriticVerdict(hypotheses_considered=["H1"], verdict="send_back")


def test_accept_is_blocked_while_a_lookalike_is_unchecked() -> None:
    with pytest.raises(ValidationError, match="look-alikes unchecked"):
        CriticVerdict(
            hypotheses_considered=["H1"],
            lookalikes_checked=["weather", "clipping"],
            verdict="accept",
        )


def test_accept_is_blocked_by_unsupported_claims() -> None:
    with pytest.raises(ValidationError, match="unsupported"):
        CriticVerdict(
            hypotheses_considered=["H1"],
            unsupported_claims=["PR fell 12% in March"],
            lookalikes_checked=_full_checklist(),
            verdict="accept",
        )


def test_accept_with_a_complete_checklist_is_valid() -> None:
    verdict = CriticVerdict(
        hypotheses_considered=["H1", "H2"],
        hypotheses_excluded=[
            ExclusionRecord(
                hypothesis="H2",
                excluded_by=["t-004"],
                reasoning="Same period last year shows the identical ceiling.",
            )
        ],
        lookalikes_checked=_full_checklist(),
        verdict="accept",
    )
    assert verdict.unchecked_lookalikes == ()


def test_not_enough_evidence_needs_two_survivors() -> None:
    # With one survivor the answer is settled; abstaining would be wrong.
    with pytest.raises(ValidationError, match="two or more"):
        CriticVerdict(
            hypotheses_considered=["H1", "H2"],
            hypotheses_still_standing=["H1"],
            lookalikes_checked=_full_checklist(),
            verdict="not_enough_evidence",
        )


# --------------------------------------------------------------------------
# AgentState and AnalysisSpec
# --------------------------------------------------------------------------
def test_agent_state_reports_unplanned_calls_and_the_cycle_cap() -> None:
    state = AgentState(
        investigation_id="INV-1",
        question="Why is INV-03 down 8% this week?",
        scope="INV-03",
        planned_tools=["compute_pr", "compare_inverter_to_fleet"],
        tools_called=["compute_pr", "compare_to_same_period_last_year"],
        hypotheses=[
            Hypothesis(id="H1", cause="String fault", consequence_if_true="dispatch"),
            Hypothesis(
                id="H2",
                cause="Curtailment",
                consequence_if_true="no action",
                status="excluded",
            ),
        ],
        cycle=4,
        max_cycles=4,
    )
    assert state.unplanned_tool_calls == ["compare_to_same_period_last_year"]
    assert state.cap_reached is True
    assert [h.id for h in state.still_standing] == ["H1"]


def test_saved_analysis_requires_a_golden_case() -> None:
    # An agent factory that produces unevaluated agents defeats the project.
    with pytest.raises(ValidationError):
        AnalysisSpec(
            id="A-1",
            name="Weekly soiling check",
            question="Is block {block} soiling faster than the plant median?",
            allowed_tools=["compute_pr", "detect_rain_reset"],
            scope="block",
            trigger="scheduled",
            golden_case_ids=[],
        )

    spec = AnalysisSpec(
        id="A-1",
        name="Weekly soiling check",
        question="Is block {block} soiling faster than the plant median?",
        allowed_tools=["compute_pr", "detect_rain_reset"],
        scope="block",
        trigger="scheduled",
        golden_case_ids=["G-014"],
    )
    assert spec.golden_case_ids == ["G-014"]
