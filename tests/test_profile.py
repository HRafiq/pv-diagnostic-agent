"""The latency and cost breakdown, read off traces already on disk.

Tested for one reason above the others: it exists to stop decisions being made
from estimates, so if it estimates anything itself it is worse than useless.
Every number here must come from a recorded field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.profile import profile_run, profile_traces, render, summary


def a_step(
    kind: str, node: str, cycle: int, ms: int, usd: float, index: int = 0
) -> dict[str, Any]:
    return {
        "kind": kind,
        "node": node,
        "args": {"cycle": cycle},
        "result": "x",
        "was_planned": True,
        "step_index": index,
        "tokens": 100,
        "cost_usd": usd,
        "latency_ms": ms,
    }


def numbered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stamp sequential indices, as `TraceWriter` does.

    The helper used to leave every step at index 0. That was invisible until
    `split_attempts` started reading a non-increasing index as a run boundary —
    at which point a fixture that never happens in reality made a real feature
    look broken.
    """
    for n, row in enumerate(rows):
        row["step_index"] = n
    return rows


def write_trace(path: Path, rows: list[dict[str, Any]], renumber: bool = True) -> Path:
    if renumber:
        rows = numbered(rows)
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def a_run(cycles: int = 3) -> list[dict[str, Any]]:
    """A shape like the real G-017: three cycles, six measurements each."""
    rows: list[dict[str, Any]] = []
    for cycle in range(cycles):
        rows.append(a_step("plan", "planner", cycle, 52_000, 0.09))
        rows += [a_step("tool", "compute_pr", cycle, 4_000, 0.01) for _ in range(6)]
        rows.append(a_step("answer", "synthesizer", cycle, 95_000, 0.15))
        rows.append(a_step("critic", "critic", cycle, 88_000, 0.13))
    return rows


def test_time_is_attributed_to_the_node_that_spent_it(tmp_path: Path) -> None:
    profile = profile_run(write_trace(tmp_path / "INV-A.jsonl", a_run()))

    assert profile.cycles == 3
    assert profile.measurements == 18
    assert profile.by_node["synthesiser"].seconds == 285.0
    assert profile.by_node["review"].seconds == 264.0
    assert profile.by_node["planner"].seconds == 156.0
    # One router turn per measurement — not the tool, which is a pure function.
    assert profile.by_node["router + measure"].calls == 18
    assert profile.by_node["router + measure"].seconds == 72.0
    assert round(profile.usd, 3) == 1.29


def test_the_re_review_overhead_is_reported_separately(tmp_path: Path) -> None:
    """The number the latency argument turns on.

    Four cases reached the right answer in cycle 1 and were sent back. If most
    of a run is spent after cycle 1, the review loop is the latency cost and the
    accuracy cost at once — which is the claim this measurement either supports
    or refutes.
    """
    profile = profile_run(write_trace(tmp_path / "INV-B.jsonl", a_run()))
    seconds, usd = profile.after_first_cycle()

    # Two of three cycles: (52 planner + 24 router + 95 synth + 88 review) x 2.
    assert seconds == 518.0
    assert round(seconds / profile.seconds, 2) == 0.67
    assert round(usd, 2) == 0.86


def test_a_single_cycle_run_has_no_overhead(tmp_path: Path) -> None:
    profile = profile_run(write_trace(tmp_path / "INV-C.jsonl", a_run(cycles=1)))
    assert profile.after_first_cycle() == (0.0, 0.0)


def test_a_trace_with_no_timing_is_skipped_rather_than_called_fast(
    tmp_path: Path,
) -> None:
    """A scripted trace records zero latency. Averaging it in would be a lie of
    the most convenient kind — it would halve every reported figure."""
    write_trace(
        tmp_path / "INV-SCRIPTED.jsonl",
        [a_step("plan", "planner", 0, 0, 0.0), a_step("answer", "s", 0, 0, 0.0)],
    )
    write_trace(tmp_path / "INV-REAL.jsonl", a_run(cycles=1))

    profiles = profile_traces(tmp_path)
    report = render(profiles)

    assert "no timing recorded, skipped" in report
    assert "across 1 timed run(s) of 2" in report
    # The real run's numbers are untouched by the scripted one.
    assert summary(profiles)["timed_runs"] == 1
    assert summary(profiles)["seconds"] == 259.0


def test_an_older_trace_says_that_review_is_under_reported(tmp_path: Path) -> None:
    """Review costs were accrued to the run total but never written onto the
    step. A naive reading of such a trace shows review as free — the exact node
    whose value is most in question."""
    rows = [
        a_step("plan", "planner", 0, 52_000, 0.09),
        a_step("answer", "synthesizer", 0, 95_000, 0.15),
        a_step("critic", "critic", 0, 0, 0.0),
    ]
    write_trace(tmp_path / "INV-OLD.jsonl", rows)

    profile = profile_run(tmp_path / "INV-OLD.jsonl")
    assert profile.review_cost_missing
    assert "under-reported" in render([profile])


def test_a_current_trace_is_not_flagged(tmp_path: Path) -> None:
    write_trace(tmp_path / "INV-NEW.jsonl", a_run(cycles=1))
    assert not profile_run(tmp_path / "INV-NEW.jsonl").review_cost_missing


def test_an_empty_directory_says_so_rather_than_reporting_zeros() -> None:
    assert "no traces found" in render([])


def test_the_report_names_what_the_measure_time_actually_is(tmp_path: Path) -> None:
    """A reader must not conclude the physics is slow. It is milliseconds."""
    write_trace(tmp_path / "INV-D.jsonl", a_run(cycles=1))
    report = render(profile_traces(tmp_path))
    assert "none of this time is the physics" in report


def test_a_rerun_appended_to_the_same_file_is_split_into_two(tmp_path: Path) -> None:
    """`TraceWriter` opens with mode "a" and the evaluation names traces after
    the case, so re-running a case appends to the file it wrote last time.

    Read naively, `INV-G-004` reported 59 measurements, 4 cycles and $3.37 —
    while the run that wrote its last entries took 11 measurements and cost
    $0.34. Every per-run figure was a session's worth of attempts summed
    together. `step_index` restarts at zero per writer, so the boundary is
    exact rather than inferred.
    """
    first = numbered(a_run(cycles=2))
    second = numbered(a_run(cycles=1))
    write_trace(tmp_path / "INV-G-004.jsonl", first + second, renumber=False)

    profiles = profile_traces(tmp_path)
    assert [p.investigation_id for p in profiles] == [
        "INV-G-004 #1",
        "INV-G-004 #2",
    ]
    assert profiles[0].cycles == 2
    assert profiles[1].cycles == 1
    assert profiles[1].measurements == 6
    # The second attempt is not carrying the first's cost.
    assert round(profiles[1].usd, 3) == 0.43


def test_a_single_attempt_keeps_its_plain_name(tmp_path: Path) -> None:
    write_trace(tmp_path / "INV-G-010.jsonl", a_run(cycles=1))
    assert profile_traces(tmp_path)[0].investigation_id == "INV-G-010"
