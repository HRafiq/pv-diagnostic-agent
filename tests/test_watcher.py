"""The Watcher: sweep, findings store, lifecycle, and the ranking over it.

The store's real job is not storage — it is knowing that the thing it saw today
is the thing it saw yesterday. Without that you get the same three findings
shouted at you every morning until you stop reading them, and a monitoring
system nobody reads is worse than none because it provides cover.

So most of these tests are about *identity* and *ageing*, not about writing
files.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.clock import FrozenClock
from src.detect.sweep import DeficitSignal, sweep_window, trailing_window
from src.findings.build import energy_at_stake, finding_from_verdict
from src.findings.models import CandidateCause, Finding
from src.findings.store import FindingsStore, identity_of, rank_by_energy
from src.tools import ToolContext
from tests.conftest import context_for


def a_finding(
    scope: str = "INV-01",
    cause: str | None = "string_outage",
    energy: float = 100.0,
    **overrides: Any,
) -> Finding:
    settled = cause is not None
    fields: dict[str, Any] = {
        "id": "F-1",
        "detected_at": datetime(2017, 4, 1, tzinfo=UTC),
        "scope": scope,
        "title": "t",
        "summary": "s",
        "category": "fault" if settled else "not_the_plant",
        "lifecycle": "new",
        "settled": settled,
        "cause": cause,
        "confidence": 0.8 if settled else None,
        "candidate_causes": (
            []
            if settled
            else [
                CandidateCause(cause="clipping", consequence_if_true="nothing"),
                CandidateCause(cause="curtailment", consequence_if_true="claim it"),
            ]
        ),
        "resolving_measurement": None if settled else "check the dispatch log",
        "energy_at_stake_kwh": energy,
        "energy_verified": settled,
        "recommended_action": "go",
        "investigation_id": "INV-1",
    }
    fields.update(overrides)
    return Finding(**fields)


@pytest.fixture
def store(tmp_path: Path) -> FindingsStore:
    return FindingsStore(tmp_path / "store.jsonl", clock=FrozenClock(_at(1)))


def _at(day: int) -> datetime:
    return datetime(2017, 4, day, tzinfo=UTC)


# ===========================================================================
# Identity
# ===========================================================================
def test_identity_is_scope_and_answer_not_wording() -> None:
    """Two runs of the same investigation write different prose. Matching on
    wording would open a new finding every sweep."""
    first = a_finding(title="One string is down", summary="A")
    second = a_finding(title="Combiner 3 has failed", summary="B")
    assert identity_of(first) == identity_of(second)


def test_a_different_scope_is_a_different_finding() -> None:
    assert identity_of(a_finding(scope="INV-01")) != identity_of(
        a_finding(scope="INV-02")
    )


def test_an_unsettled_finding_is_identified_by_its_surviving_causes() -> None:
    """Narrowing four candidates to two is a genuinely different finding."""
    two = a_finding(cause=None)
    three = a_finding(
        cause=None,
        candidate_causes=[
            CandidateCause(cause="clipping", consequence_if_true="a"),
            CandidateCause(cause="curtailment", consequence_if_true="b"),
            CandidateCause(cause="soiling", consequence_if_true="c"),
        ],
    )
    assert identity_of(two) != identity_of(three)


def test_candidate_order_does_not_change_identity(tmp_path: Path) -> None:
    reversed_order = a_finding(
        cause=None,
        candidate_causes=[
            CandidateCause(cause="curtailment", consequence_if_true="claim it"),
            CandidateCause(cause="clipping", consequence_if_true="nothing"),
        ],
    )
    assert identity_of(a_finding(cause=None)) == identity_of(reversed_order)


# ===========================================================================
# Lifecycle
# ===========================================================================
def test_a_first_sighting_is_new(store: FindingsStore) -> None:
    written = store.observe([a_finding()])
    assert [r.lifecycle for r in written] == ["new"]


def test_a_second_sighting_is_ongoing(tmp_path: Path) -> None:
    clock = FrozenClock(_at(1))
    store = FindingsStore(tmp_path / "s.jsonl", clock=clock)
    store.observe([a_finding()])
    clock.set(_at(2))
    assert [r.lifecycle for r in store.observe([a_finding()])] == ["ongoing"]


def test_a_finding_resolves_after_the_configured_quiet_sweeps(
    tmp_path: Path,
) -> None:
    """One quiet sweep is too eager — a cloudy day can hide a real fault from a
    detector. Three is the knob that decides between crying wolf and going
    quiet on something real.
    """
    clock = FrozenClock(_at(1))
    store = FindingsStore(tmp_path / "s.jsonl", clock=clock, resolve_after_sweeps=3)
    store.observe([a_finding()])

    for day in (2, 3):
        clock.set(_at(day))
        assert not [r for r in store.observe([]) if r.lifecycle == "resolved"]
        assert store.open_findings()

    clock.set(_at(4))
    resolved = store.observe([])
    assert [r.lifecycle for r in resolved] == ["resolved"]
    assert store.open_findings() == []


def test_a_quiet_sweep_still_ages_a_finding(tmp_path: Path) -> None:
    """The store has to hear about sweeps that found nothing.

    A sweep that writes nothing leaves no trace in the log, so a fixed fault
    would sit open forever on a plant that had gone quiet.
    """
    clock = FrozenClock(_at(1))
    store = FindingsStore(tmp_path / "s.jsonl", clock=clock, resolve_after_sweeps=2)
    store.observe([a_finding()])
    clock.set(_at(2))
    store.observe([])
    clock.set(_at(3))
    assert [r.lifecycle for r in store.observe([])] == ["resolved"]


def test_a_reappearance_resets_the_miss_count(tmp_path: Path) -> None:
    clock = FrozenClock(_at(1))
    store = FindingsStore(tmp_path / "s.jsonl", clock=clock, resolve_after_sweeps=3)
    store.observe([a_finding()])
    clock.set(_at(2))
    store.observe([])
    clock.set(_at(3))
    store.observe([a_finding()])
    for day in (4, 5):
        clock.set(_at(day))
        store.observe([])
    assert store.open_findings()  # not yet resolved: the count restarted


def test_a_suppressed_finding_stays_suppressed(tmp_path: Path) -> None:
    """Re-raising it next sweep is exactly how an operator learns to ignore the
    whole interface."""
    clock = FrozenClock(_at(1))
    store = FindingsStore(tmp_path / "s.jsonl", clock=clock)
    store.observe([a_finding()])
    identity = identity_of(a_finding())
    store.set_lifecycle(identity, "suppressed", note="known, smaller combiner")

    clock.set(_at(2))
    assert [r.lifecycle for r in store.observe([a_finding()])] == ["suppressed"]
    assert store.open_findings() == []


def test_an_acknowledged_finding_stays_acknowledged_but_stays_open(
    tmp_path: Path,
) -> None:
    clock = FrozenClock(_at(1))
    store = FindingsStore(tmp_path / "s.jsonl", clock=clock)
    store.observe([a_finding()])
    store.set_lifecycle(identity_of(a_finding()), "acknowledged")
    clock.set(_at(2))
    assert [r.lifecycle for r in store.observe([a_finding()])] == ["acknowledged"]
    assert len(store.open_findings()) == 1


def test_acknowledging_something_unknown_returns_nothing(
    store: FindingsStore,
) -> None:
    assert store.set_lifecycle("nothing::here", "acknowledged") is None


# ===========================================================================
# The log is append-only
# ===========================================================================
def test_nothing_is_ever_mutated(tmp_path: Path) -> None:
    """A finding that was wrong stays visible rather than vanishing."""
    clock = FrozenClock(_at(1))
    store = FindingsStore(tmp_path / "s.jsonl", clock=clock)
    store.observe([a_finding()])
    clock.set(_at(2))
    store.observe([a_finding()])

    history = store.history(identity_of(a_finding()))
    assert [r.lifecycle for r in history] == ["new", "ongoing"]
    assert [r.observed_at for r in history] == [_at(1), _at(2)]
    assert len(store.current()) == 1


def test_records_survive_a_reload(tmp_path: Path) -> None:
    clock = FrozenClock(_at(1))
    FindingsStore(tmp_path / "s.jsonl", clock=clock).observe([a_finding()])
    reloaded = FindingsStore(tmp_path / "s.jsonl", clock=clock)
    assert len(reloaded.current()) == 1
    assert reloaded.current()[0].finding.cause == "string_outage"


def test_the_since_query_answers_what_is_new(tmp_path: Path) -> None:
    clock = FrozenClock(_at(1))
    store = FindingsStore(tmp_path / "s.jsonl", clock=clock)
    store.observe([a_finding()])
    clock.set(_at(5))
    store.observe([a_finding(scope="INV-02")])
    assert len(store.since(_at(3))) == 1


def test_a_corrupt_line_names_the_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text('{"identity": "x"}\n')
    with pytest.raises(ValueError, match=r"s\.jsonl:1"):
        list(FindingsStore(path, clock=FrozenClock(_at(1))).records())


def test_timestamps_come_from_the_injected_clock(tmp_path: Path) -> None:
    """A replayed 2017 sweep dates its findings in 2017, not today."""
    clock = FrozenClock(datetime(2016, 7, 15, tzinfo=UTC))
    store = FindingsStore(tmp_path / "s.jsonl", clock=clock)
    assert store.observe([a_finding()])[0].observed_at.year == 2016


# ===========================================================================
# Ranking
# ===========================================================================
def test_ranking_puts_the_most_energy_first(tmp_path: Path) -> None:
    clock = FrozenClock(_at(1))
    store = FindingsStore(tmp_path / "s.jsonl", clock=clock)
    store.observe(
        [
            a_finding(scope="A", energy=50.0),
            a_finding(scope="B", energy=500.0),
        ]
    )
    assert [r.finding.scope for r in store.open_findings()] == ["B", "A"]


def test_an_unverified_figure_never_outranks_a_verified_one(tmp_path: Path) -> None:
    """An unverified number is a smaller claim and the ordering should say so."""
    clock = FrozenClock(_at(1))
    store = FindingsStore(tmp_path / "s.jsonl", clock=clock)
    store.observe(
        [
            a_finding(scope="settled", energy=10.0),
            a_finding(scope="open", cause=None, energy=9000.0),
        ]
    )
    ranked = store.open_findings()
    assert ranked[0].finding.scope == "settled"
    assert ranked[0].finding.energy_verified is True


def test_ranking_an_empty_list_is_empty() -> None:
    assert rank_by_energy([]) == []


# ===========================================================================
# The sweep
# ===========================================================================
def test_a_signal_carries_no_cause() -> None:
    """A detector that named a cause would be a rules engine on a timer, and
    the agent would be writing up a conclusion already reached."""
    fields = set(DeficitSignal.__dataclass_fields__)
    assert not fields & {"cause", "category", "diagnosis", "fault"}


def test_the_question_is_a_question_not_an_assertion() -> None:
    signal = DeficitSignal(
        "plant", "2017-04-01", "2017-04-14", "d", "m", 0.1, 0.05, "Why is output down?"
    )
    assert signal.question.endswith("?")


def test_severity_orders_by_how_far_past_the_threshold() -> None:
    mild = DeficitSignal("p", "a", "b", "d", "m", 0.06, 0.05, "?")
    severe = DeficitSignal("p", "a", "b", "d", "m", 0.50, 0.05, "?")
    assert severe.severity > mild.severity


def test_the_trailing_window_stops_before_today() -> None:
    """The current day is still accumulating; scoring a half-finished day
    produces a deficit every single morning."""
    start, end = trailing_window(datetime(2017, 4, 20, tzinfo=UTC), days=14)
    assert end == "2017-04-19"
    assert start == "2017-04-06"


def test_a_healthy_window_fires_nothing(ctx: ToolContext) -> None:
    assert sweep_window(ctx, "2017-05-01", "2017-05-30") == []


def test_a_missing_power_channel_fires_the_completeness_detector(
    plant: pd.DataFrame,
) -> None:
    holed = plant.copy()
    window = (pd.DatetimeIndex(holed.index) >= pd.Timestamp("2017-05-16", tz="UTC")) & (
        pd.DatetimeIndex(holed.index) <= pd.Timestamp("2017-05-30 23:59", tz="UTC")
    )
    holed.loc[window, "ac_power_kw"] = float("nan")

    signals = sweep_window(context_for(holed), "2017-05-16", "2017-05-30")
    assert [s.detector for s in signals] == ["data_completeness"]
    assert signals[0].value < 0.8


def test_a_dead_string_fires_the_imbalance_detector(plant: pd.DataFrame) -> None:
    broken = plant.copy()
    broken["string_current_a_3"] *= 0.1
    detectors = {
        s.detector
        for s in sweep_window(context_for(broken), "2017-05-16", "2017-05-30")
    }
    assert "string_imbalance" in detectors


def test_a_ceiling_fires_the_ceiling_detector(plant: pd.DataFrame) -> None:
    capped = plant.copy()
    capped["ac_power_kw"] = capped["ac_power_kw"].clip(upper=55.0)
    detectors = {
        s.detector
        for s in sweep_window(context_for(capped), "2017-05-16", "2017-05-30")
    }
    assert "flat_ceiling" in detectors


def test_detectors_overlap_on_purpose(plant: pd.DataFrame) -> None:
    """An investigation starting from three independent signals is better
    grounded than one starting from whichever fired first."""
    bad = plant.copy()
    bad["string_current_a_3"] *= 0.1
    bad.loc[
        pd.DatetimeIndex(bad.index) >= pd.Timestamp("2017-05-16", tz="UTC"),
        "ac_power_kw",
    ] *= 0.7
    signals = sweep_window(context_for(bad), "2017-05-16", "2017-05-30")
    assert len(signals) >= 2
    # Ordered by severity, so the ranking is stable rather than alphabetical.
    assert [s.severity for s in signals] == sorted(
        (s.severity for s in signals), reverse=True
    )


# ===========================================================================
# Building findings from a verdict
# ===========================================================================
class _Verdict:
    def __init__(self, **kwargs: Any) -> None:
        self.category = kwargs.get("category", "fault")
        self.cause = kwargs.get("cause", "string_outage")
        self.settled = kwargs.get("settled", True)
        self.candidate_causes = kwargs.get("candidate_causes", ())
        self.resolving_measurement = kwargs.get("resolving_measurement")
        self.evidence = kwargs.get("evidence", {})


SIGNAL = DeficitSignal(
    "INV-01", "2017-04-01", "2017-04-14", "string_imbalance", "m", 0.3, 0.22, "Why?"
)


def test_a_settled_verdict_becomes_a_settled_finding() -> None:
    finding = finding_from_verdict(
        _Verdict(evidence={"compute_expected_output.shortfall_kwh": 1234.0}),
        SIGNAL,
        "F-1",
        _at(1),
        "INV-1",
    )
    assert finding.settled and finding.cause == "string_outage"
    assert finding.energy_at_stake_kwh == 1234.0
    assert finding.energy_verified is True


def test_an_unsettled_verdict_carries_no_cause_and_no_verified_energy() -> None:
    finding = finding_from_verdict(
        _Verdict(
            settled=False,
            cause=None,
            category=None,
            candidate_causes=("clipping", "curtailment"),
            resolving_measurement="check the dispatch log",
            evidence={"compute_expected_output.shortfall_kwh": 900.0},
        ),
        SIGNAL,
        "F-2",
        _at(1),
        "INV-2",
        consequences={"clipping": "nothing", "curtailment": "claim it"},
    )
    assert finding.cause is None and finding.confidence is None
    assert finding.energy_verified is False
    assert {c.cause for c in finding.candidate_causes} == {"clipping", "curtailment"}


def test_an_unsettled_verdict_with_one_survivor_is_refused_not_padded() -> None:
    with pytest.raises(Exception, match="at least two"):
        finding_from_verdict(
            _Verdict(
                settled=False,
                cause=None,
                category=None,
                candidate_causes=("clipping",),
                resolving_measurement="check the log",
            ),
            SIGNAL,
            "F-3",
            _at(1),
            "INV-3",
        )


def test_energy_is_measured_never_estimated() -> None:
    """An absent figure stays zero rather than being guessed from a percentage."""
    assert energy_at_stake({}) == 0.0
    assert energy_at_stake({"compute_expected_output.deficit_fraction": 0.2}) == 0.0
    assert energy_at_stake({"compute_expected_output.shortfall_kwh": 50.0}) == 50.0


def test_both_engines_build_findings_the_same_way() -> None:
    """If they built findings differently, a difference on the Watcher tab
    would be a difference in presentation."""
    import inspect

    from src.findings import build

    assert "engine" not in inspect.signature(build.finding_from_verdict).parameters
