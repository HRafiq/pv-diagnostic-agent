from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.clock import ReplayClock
from src.trace import TraceStep, TraceWriter, read_trace

START = datetime(2019, 3, 14, tzinfo=UTC)


def test_round_trip_preserves_steps(tmp_path: Path) -> None:
    clock = ReplayClock(start=START)
    with TraceWriter("INV-1", clock=clock, root=tmp_path) as trace:
        trace.write(TraceStep(kind="plan", node="planner", was_planned=True))
        clock.advance(timedelta(seconds=90))
        trace.write(
            TraceStep(
                kind="tool",
                node="compute_pr",
                args={"scope": "INV-03", "range": "2019-03-01/2019-03-14"},
                result="temp-corrected PR 0.71 vs 0.79 fleet median",
                was_planned=True,
                tokens=0,
                latency_ms=142,
            )
        )

    steps = list(read_trace(tmp_path / "INV-1.jsonl"))
    assert [s.step_index for s in steps] == [0, 1]
    assert steps[0].timestamp == START
    # Stamped from the injected clock, so trace time is simulated time.
    assert steps[1].timestamp == START + timedelta(seconds=90)
    assert steps[1].args == {"scope": "INV-03", "range": "2019-03-01/2019-03-14"}


def test_writer_appends_across_sessions(tmp_path: Path) -> None:
    clock = ReplayClock(start=START)
    for _ in range(2):
        with TraceWriter("INV-2", clock=clock, root=tmp_path) as trace:
            trace.write(TraceStep(kind="tool", node="compute_pr", was_planned=True))
    assert len(list(read_trace(tmp_path / "INV-2.jsonl"))) == 2


def test_writer_requires_context_manager(tmp_path: Path) -> None:
    writer = TraceWriter("INV-3", clock=ReplayClock(start=START), root=tmp_path)
    with pytest.raises(RuntimeError, match="context manager"):
        writer.write(TraceStep(kind="plan", node="planner", was_planned=True))


def test_path_traversal_in_the_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path-safe"):
        TraceWriter("../escape", clock=ReplayClock(start=START), root=tmp_path)


def test_corrupt_line_names_the_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"kind": "plan", "node": "planner", "was_planned": true}\nnope\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        list(read_trace(path))


def test_unplanned_step_survives_the_round_trip(tmp_path: Path) -> None:
    # was_planned is the agency signal; it has to survive serialisation intact
    # or the eval harness silently measures zero.
    clock = ReplayClock(start=START)
    with TraceWriter("INV-4", clock=clock, root=tmp_path) as trace:
        trace.write(
            TraceStep(
                kind="adaptive",
                node="compare_to_same_period_last_year",
                was_planned=False,
                reason_for_choosing="Absence of clipping does not exclude curtailment.",
                excludes=["H3"],
            )
        )
    step = next(iter(read_trace(tmp_path / "INV-4.jsonl")))
    assert step.was_planned is False
    assert step.excludes == ["H3"]
