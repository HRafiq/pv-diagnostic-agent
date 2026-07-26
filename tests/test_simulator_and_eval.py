"""Injectors, golden-set composition, scoring, and the rules baseline.

The tests that matter most here are the ones asserting the *evaluation cannot
flatter itself*: that injection happens at the physics layer, that look-alikes
are a large share of the set, that abstaining everywhere scores badly, and that
the agency metrics correctly identify a pipeline as a pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval.golden import GoldenCase, build_golden_set, composition
from eval.metrics import (
    Prediction,
    aggregate,
    confusion,
    measure_agency,
    score_case,
)
from simulator.injectors import (
    INJECTORS,
    inject_curtailment,
    inject_inverter_clipping,
    inject_sensor_drift,
    inject_soiling,
    inject_string_outage,
    inject_telemetry_gap,
)
from src.physics.performance import compute_pr

GAMMA = -0.0045


def base_frame(days: int = 20) -> pd.DataFrame:
    """A synthetic but physically shaped plant: 7 balanced strings."""
    index = pd.date_range("2017-06-01", periods=days * 96, freq="15min", tz="UTC")
    hour = index.hour + index.minute / 60.0
    poa = np.clip(1000 * np.sin(np.pi * (hour - 6) / 12), 0, None)
    frame = pd.DataFrame(
        {
            "poa_wm2": poa,
            "ac_power_kw": poa / 10.0,
            "dc_power_kw": poa / 9.5,
            "temp_ambient_c": 22.0,
            "temp_module_c": 22.0 + poa / 32.0,
            "wind_speed_ms": 1.5,
        },
        index=index,
    )
    for i in range(1, 8):
        frame[f"string_current_a_{i}"] = poa / 70.0
    return frame


WINDOW = ("2017-06-05 00:00:00+00:00", "2017-06-15 00:00:00+00:00")


# --------------------------------------------------------------------------
# Injectors act on physics, not signatures
# --------------------------------------------------------------------------
def test_string_outage_touches_one_string_and_leaves_irradiance_alone() -> None:
    frame = base_frame()
    out, record = inject_string_outage(
        frame, *WINDOW, string_index=3, fraction_lost=1.0
    )

    # Irradiance is untouched — that is what makes it a plant fault rather
    # than weather, and the diagnostic has to discover it.
    pd.testing.assert_series_equal(out["poa_wm2"], frame["poa_wm2"])
    assert out["string_current_a_3"].sum() < frame["string_current_a_3"].sum()
    for i in (1, 2, 4, 5, 6, 7):
        pd.testing.assert_series_equal(
            out[f"string_current_a_{i}"], frame[f"string_current_a_{i}"]
        )
    assert record.category == "fault"
    assert record.energy_lost_kwh > 0


def test_string_outage_signature_is_emergent_not_written_in() -> None:
    """The share-of-current step must fall out of the physics, not be set."""
    frame = base_frame()
    out, _ = inject_string_outage(frame, *WINDOW, string_index=2, fraction_lost=1.0)
    columns = [f"string_current_a_{i}" for i in range(1, 8)]
    daylight = out["poa_wm2"] > 400
    shares = out.loc[daylight, columns].div(
        out.loc[daylight, columns].sum(axis=1), axis=0
    )
    inside = (out.index >= pd.Timestamp(WINDOW[0])) & (
        out.index <= pd.Timestamp(WINDOW[1])
    )
    during = shares[inside[daylight]]["string_current_a_2"].mean()
    before = shares[~inside[daylight]]["string_current_a_2"].mean()
    assert during < before - 0.1


def test_partial_outage_scales_with_fraction() -> None:
    frame = base_frame()
    small, rec_small = inject_string_outage(frame, *WINDOW, fraction_lost=0.25)
    large, rec_large = inject_string_outage(frame, *WINDOW, fraction_lost=0.75)
    assert rec_large.energy_lost_kwh > rec_small.energy_lost_kwh
    assert small["ac_power_kw"].sum() > large["ac_power_kw"].sum()


def test_ramped_outage_differs_from_a_step() -> None:
    frame = base_frame()
    step, _ = inject_string_outage(frame, *WINDOW, transition_minutes=0)
    ramp, _ = inject_string_outage(frame, *WINDOW, transition_minutes=2880)
    assert ramp["string_current_a_1"].sum() > step["string_current_a_1"].sum()


def test_string_index_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="string_index must be"):
        inject_string_outage(base_frame(), *WINDOW, string_index=99)


def test_injection_is_deterministic() -> None:
    frame = base_frame()
    a, _ = inject_string_outage(frame, *WINDOW, string_index=4, seed=7)
    b, _ = inject_string_outage(frame, *WINDOW, string_index=4, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_soiling_accumulates_then_resets() -> None:
    frame = base_frame(days=30)
    out, record = inject_soiling(frame, *WINDOW, rate_per_day=0.01, max_loss=0.3)
    assert record.category == "recoverable"
    assert out["ac_power_kw"].sum() < frame["ac_power_kw"].sum()
    # Loss deepens over the window rather than stepping.
    inside = (out.index >= pd.Timestamp(WINDOW[0])) & (
        out.index <= pd.Timestamp(WINDOW[1])
    )
    ratio = (out["ac_power_kw"] / frame["ac_power_kw"])[inside].dropna()
    assert ratio.iloc[-1] < ratio.iloc[0]


def test_sensor_drift_raises_pr_and_loses_no_energy() -> None:
    """The counter-intuitive one, and the reason the project exists.

    A soiled *array* drops power and PR falls. A soiled *sensor* drops the
    irradiance reading with power untouched, so PR RISES and no energy is lost.
    """
    frame = base_frame(days=30)
    out, record = inject_sensor_drift(
        frame, *WINDOW, drift_per_day=-0.01, max_drift=-0.2
    )

    pd.testing.assert_series_equal(out["ac_power_kw"], frame["ac_power_kw"])
    assert record.energy_lost_kwh == 0.0
    assert record.category == "not_the_plant"

    before = compute_pr(frame, 100.0, GAMMA)
    after = compute_pr(out, 100.0, GAMMA)
    assert after.pr > before.pr


def test_telemetry_gap_blanks_power_but_not_irradiance() -> None:
    frame = base_frame()
    out, record = inject_telemetry_gap(frame, *WINDOW)
    inside = (out.index >= pd.Timestamp(WINDOW[0])) & (
        out.index <= pd.Timestamp(WINDOW[1])
    )
    assert out.loc[inside, "ac_power_kw"].isna().all()
    assert out.loc[inside, "poa_wm2"].notna().all()
    assert record.category == "not_the_plant"


def test_gap_does_not_depress_pr() -> None:
    """A data problem must not read as a plant problem."""
    frame = base_frame()
    out, _ = inject_telemetry_gap(frame, *WINDOW)
    assert compute_pr(out, 100.0, GAMMA).pr == pytest.approx(
        compute_pr(frame, 100.0, GAMMA).pr, abs=0.02
    )


def test_clipping_and_curtailment_are_indistinguishable_on_power() -> None:
    """The canonical unresolvable pair — identical ceiling, different cause."""
    frame = base_frame()
    clipped, rec_clip = inject_inverter_clipping(frame, *WINDOW, ac_ceiling_kw=60.0)
    curtailed, rec_curt = inject_curtailment(frame, *WINDOW, ceiling_kw=60.0)

    pd.testing.assert_series_equal(clipped["ac_power_kw"], curtailed["ac_power_kw"])
    assert rec_clip.category == "by_design"
    assert rec_curt.category == "not_the_plant"


def test_every_injector_is_registered() -> None:
    assert set(INJECTORS) == {
        "string_outage",
        "shading",
        "soiling",
        "clipping",
        "curtailment",
        "sensor_drift",
        "telemetry_gap",
    }


def test_injector_record_serialises() -> None:
    _, record = inject_soiling(base_frame(days=30), *WINDOW)
    payload = record.to_dict()
    assert payload["category"] == "recoverable" and "note" in payload


def test_empty_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="selects no rows"):
        inject_string_outage(base_frame(), "2020-01-01", "2020-01-02")


# --------------------------------------------------------------------------
# Golden set composition
# --------------------------------------------------------------------------
def test_golden_set_is_adversarial_enough() -> None:
    cases = build_golden_set()
    frame = composition(cases)
    for _, row in frame.iterrows():
        # §5.1: at least 40% of cases must be things that look like faults and
        # are not. Below that, a detector that alarms on any deficit scores well.
        assert row["look_alike_share"] >= 0.40, row["split"]
        assert row["unresolvable"] >= 1
        assert row["real_data_no_injection"] >= 1


def test_splits_use_different_windows() -> None:
    cases = build_golden_set()
    tuning = {(c.start, c.end) for c in cases if c.split == "tuning"}
    held = {(c.start, c.end) for c in cases if c.split == "heldback"}
    assert not (tuning & held), "held-back windows must not overlap tuning"


def test_case_ids_are_unique() -> None:
    ids = [c.id for c in build_golden_set()]
    assert len(ids) == len(set(ids))


def test_golden_set_is_reproducible() -> None:
    assert [c.to_dict() for c in build_golden_set()] == [
        c.to_dict() for c in build_golden_set()
    ]


def test_unresolvable_cases_name_a_resolving_measurement() -> None:
    for case in build_golden_set():
        if not case.settled:
            assert case.resolving_measurement, case.id
            assert len(case.candidate_causes) >= 2, case.id


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _case(**kw: object) -> GoldenCase:
    defaults = dict(
        id="G-001",
        split="tuning",
        system_id=1,
        start="2017-01-01",
        end="2017-01-02",
        question="?",
        expected_category="fault",
        expected_cause="string_outage",
        settled=True,
    )
    defaults.update(kw)
    return GoldenCase(**defaults)  # type: ignore[arg-type]


def _pred(**kw: object) -> Prediction:
    defaults = dict(
        case_id="G-001", category="fault", cause="string_outage", settled=True
    )
    defaults.update(kw)
    return Prediction(**defaults)  # type: ignore[arg-type]


def test_correct_fault_scores_clean() -> None:
    s = score_case(_case(), _pred())
    assert s.category_correct and s.cause_correct
    assert not s.false_alarm and not s.missed_fault


def test_false_alarm_only_counts_on_lookalikes() -> None:
    lookalike = _case(expected_category="by_design", expected_cause="clipping")
    assert score_case(lookalike, _pred(category="fault")).false_alarm
    assert not score_case(_case(), _pred(category="fault")).false_alarm


def test_missed_fault_detected() -> None:
    assert score_case(_case(), _pred(category="by_design")).missed_fault
    assert score_case(_case(), _pred(settled=False)).missed_fault


def test_abstention_is_correct_only_where_truth_is_unresolvable() -> None:
    unresolvable = _case(expected_category=None, expected_cause=None, settled=False)
    good = score_case(unresolvable, _pred(settled=False, category=None, cause=None))
    assert good.correct_abstention and good.category_correct

    bad = score_case(_case(), _pred(settled=False))
    assert bad.wrong_abstention and not bad.category_correct


def test_committing_on_an_unresolvable_case_is_wrong() -> None:
    unresolvable = _case(expected_category=None, expected_cause=None, settled=False)
    s = score_case(unresolvable, _pred(category="by_design", cause="clipping"))
    assert not s.category_correct


def test_abstaining_everywhere_scores_badly() -> None:
    """Abstention must not be a way to dodge the evaluation."""
    cases = [_case(id=f"G-{i}") for i in range(8)]
    scores = [score_case(c, _pred(case_id=c.id, settled=False)) for c in cases]
    report = aggregate(scores)
    assert report.by_split["tuning"]["macro_f1"] < 0.4


def test_report_separates_splits_and_computes_the_gap() -> None:
    scores = [
        score_case(_case(id="G-1", split="tuning"), _pred()),
        score_case(_case(id="H-1", split="heldback"), _pred(category="by_design")),
    ]
    report = aggregate(scores)
    assert set(report.by_split) == {"tuning", "heldback"}
    assert report.overfitting_gap is not None


def test_v1_targets_reported() -> None:
    report = aggregate([score_case(_case(split="heldback"), _pred())])
    targets = report.meets_v1_targets()
    assert set(targets) >= {"macro_f1_at_least_0.75", "false_alarm_rate_at_most_0.15"}


def test_false_alarm_rate_is_none_without_lookalikes() -> None:
    report = aggregate([score_case(_case(), _pred())])
    assert report.by_split["tuning"]["false_alarm_rate"] is None


def test_confusion_includes_abstention_as_a_class() -> None:
    scores = [
        score_case(_case(expected_category=None, settled=False), _pred(settled=False)),
        score_case(_case(), _pred()),
    ]
    assert "not_enough_evidence" in confusion(scores).index


# --------------------------------------------------------------------------
# Agency metrics — the control that proves they discriminate
# --------------------------------------------------------------------------
def test_agency_metrics_identify_a_pipeline() -> None:
    """A fixed sequence must score near zero on every agency measure."""
    fixed = tuple(["compute_pr", "check_ac_ceiling"])
    predictions = [
        Prediction(
            case_id=f"c{i}",
            category="fault",
            cause="x",
            settled=True,
            tools_called=fixed,
            unplanned_tools=(),
            critic_cycles=0,
        )
        for i in range(10)
    ]
    agency = measure_agency(predictions)
    assert agency["distinct_tool_trajectories"] == 1
    assert agency["unplanned_measurement_rate"] == 0.0
    assert agency["self_initiated_abstentions"] == 0


def test_agency_metrics_reward_varied_trajectories() -> None:
    predictions = [
        Prediction(
            case_id="a",
            category="fault",
            cause="x",
            settled=True,
            tools_called=("compute_pr",),
            unplanned_tools=(),
        ),
        Prediction(
            case_id="b",
            category="fault",
            cause="x",
            settled=True,
            tools_called=("compute_pr", "detect_rain_reset"),
            unplanned_tools=("detect_rain_reset",),
            critic_cycles=2,
        ),
        Prediction(
            case_id="c",
            category=None,
            cause=None,
            settled=False,
            tools_called=("compute_pr", "check_clearsky_consistency"),
            unplanned_tools=("check_clearsky_consistency",),
            critic_cycles=1,
        ),
    ]
    agency = measure_agency(predictions)
    assert agency["distinct_tool_trajectories"] == 3
    assert agency["unplanned_measurement_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert agency["self_initiated_abstentions"] == 1
    assert agency["iteration_distribution"] == {0: 1, 1: 1, 2: 1}


def test_agency_on_empty_input() -> None:
    assert measure_agency([]) == {}


# --------------------------------------------------------------------------
# Rules baseline
# --------------------------------------------------------------------------
def _engine() -> object:
    """The baseline, configured for the synthetic fixture.

    Since step 7 the engine measures through the shared tool registry rather
    than its own arithmetic, so it needs a `ToolContext` rather than plant
    parameters — the same object the agent gets. That is the point of the
    change: both engines take identical measurements, so a difference between
    them is a difference in reasoning.

    The clear-sky check is relaxed here because the fixture is a plain sine
    rather than a real clear-sky curve and sits far from the modelled envelope
    by construction. These tests are about the rules, not the fixture.
    """
    from src.baseline.rules import RulesEngine

    return RulesEngine(clearsky_shortfall=0.99, clearsky_drift_per_day=-99.0)


def _diagnose(frame: pd.DataFrame) -> object:
    from tests.conftest import context_for

    return _engine().diagnose(context_for(frame))  # type: ignore[attr-defined]


def test_rules_engine_flags_a_telemetry_gap() -> None:
    frame, _ = inject_telemetry_gap(base_frame(), "2017-06-02", "2017-06-18")
    verdict = _diagnose(frame)
    assert verdict.cause == "telemetry_gap"


def test_rules_engine_finds_a_string_outage() -> None:
    frame, _ = inject_string_outage(
        base_frame(), "2017-06-01", "2017-06-21", string_index=3, fraction_lost=1.0
    )
    verdict = _diagnose(frame)
    assert verdict.category == "fault"


def test_rules_engine_always_commits() -> None:
    """Its structural limit: it cannot say 'not enough evidence'.

    This is why the unresolvable cases exist, and why the agent has somewhere
    to be better rather than merely different.
    """
    frame, _ = inject_curtailment(
        base_frame(), "2017-06-01", "2017-06-21", ceiling_kw=55.0
    )
    verdict = _diagnose(frame)
    assert verdict.settled is True
    assert verdict.candidate_causes == ()


def test_rules_engine_runs_a_fixed_check_sequence() -> None:
    a = _diagnose(base_frame())
    frame, _ = inject_soiling(base_frame(days=30), *WINDOW)
    b = _diagnose(frame)
    # Same checks regardless of what the data showed — the definition of a
    # pipeline, and the thing the agency metrics are built to detect.
    assert a.checks_run[: len(b.checks_run)] == b.checks_run[: len(a.checks_run)]
