"""Watcher CLI.

The watcher is the seam that resolves §6.3 against §6.5: it *is* the arrival of
data. These tests pin the argument parsing and the invariant that the dashboard
never has to trigger anything.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import watcher
from src.config import build_clock

# One dry sweep of one day. The `--data-dir` these tests pass points at a
# synthetic ingested system (see `ingested_plant` in conftest), so they assert
# CLI plumbing without depending on whether the PVDAQ download has been run on
# this machine. Nothing here is scored.
SWEEP = ["step", "--step", "1D", "--dry-run"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1D", timedelta(days=1)),
        ("7D", timedelta(days=7)),
        ("6H", timedelta(hours=6)),
        ("15min", timedelta(minutes=15)),
        ("0.5D", timedelta(hours=12)),
    ],
)
def test_step_parsing(text: str, expected: timedelta) -> None:
    assert watcher._parse_step(text) == expected


@pytest.mark.parametrize("text", ["1", "D", "1W", "banana", "", "1Dx"])
def test_bad_steps_are_rejected(text: str) -> None:
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        watcher._parse_step(text)


def test_naive_dates_are_treated_as_utc() -> None:
    assert watcher._parse_date("2019-06-30") == datetime(2019, 6, 30, tzinfo=UTC)


def test_aware_dates_are_preserved() -> None:
    parsed = watcher._parse_date("2019-06-30T12:00:00+09:30")
    assert parsed.utcoffset() == timedelta(hours=9, minutes=30)


def test_status_reports_clock_and_models(capsys: pytest.CaptureFixture[str]) -> None:
    assert watcher.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "simulated now" in out
    assert "model profile" in out
    for node in ("planner", "router", "synthesizer", "critic"):
        assert node in out


def test_step_advances_and_reports_the_window(
    capsys: pytest.CaptureFixture[str],
    ingested_plant: Path,
) -> None:
    assert watcher.main([*SWEEP, "--data-dir", str(ingested_plant)]) == 0
    out = capsys.readouterr().out
    assert "advanced" in out
    assert "sweeping" in out


def test_run_rejects_a_past_target(capsys: pytest.CaptureFixture[str]) -> None:
    # The replay clock starts inside the ingested record; anything before that
    # is behind it, and a clock that refuses to rewind is the whole point.
    assert watcher.main(["run", "--until", "2015-01-01"]) == 1
    assert "not in the future" in capsys.readouterr().err


def test_run_walks_forward_in_steps(
    capsys: pytest.CaptureFixture[str],
    ingested_plant: Path,
) -> None:
    start = build_clock().now().date()
    until = start + timedelta(days=4)
    assert (
        watcher.main(
            [
                "run",
                "--until",
                str(until),
                "--step",
                "1D",
                "--dry-run",
                "--data-dir",
                str(ingested_plant),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert out.count("sweeping") == 4
    assert "reached" in out


def test_a_dry_run_writes_no_findings(tmp_path: Any, ingested_plant: Path) -> None:
    """A dry run must be visibly a dry run, not a silent no-op."""
    store = tmp_path / "s.jsonl"
    argv = [*SWEEP, "--data-dir", str(ingested_plant), "--store", str(store)]
    assert watcher.main(argv) == 0
    assert not store.exists()


def test_a_missing_ingest_is_reported_not_raised(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Any,
) -> None:
    """The remedy is a command, so the sweep prints it and exits non-zero.

    This is the path the three tests above used to take on any machine without
    the PVDAQ download — it exits 1, which is correct, and it is asserted here
    rather than being mistaken for a plumbing failure.
    """
    assert watcher.main([*SWEEP, "--data-dir", str(tmp_path / "empty")]) == 1
    assert "is not ingested" in capsys.readouterr().err


def test_the_replay_clock_starts_inside_the_ingested_record() -> None:
    """A start date the data does not cover leaves every trailing window empty
    and the Watcher finds nothing, silently."""
    assert 2016 <= build_clock().now().year <= 2017


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        watcher.main([])
