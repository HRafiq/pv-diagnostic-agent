"""Watcher CLI.

The watcher is the seam that resolves §6.3 against §6.5: it *is* the arrival of
data. These tests pin the argument parsing and the invariant that the dashboard
never has to trigger anything.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

import pytest

import watcher


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
) -> None:
    assert watcher.main(["step", "--step", "1D"]) == 0
    out = capsys.readouterr().out
    assert "advanced" in out
    assert "dry-run" in out


def test_run_rejects_a_past_target(capsys: pytest.CaptureFixture[str]) -> None:
    # The replay clock starts in 2019; 2018 is behind it.
    assert watcher.main(["run", "--until", "2018-01-01"]) == 1
    assert "not in the future" in capsys.readouterr().err


def test_run_walks_forward_in_steps(capsys: pytest.CaptureFixture[str]) -> None:
    assert watcher.main(["run", "--until", "2019-01-05", "--step", "1D"]) == 0
    out = capsys.readouterr().out
    # Four one-day steps to cross from the 1st to the 5th.
    assert out.count("would sweep") == 4
    assert "reached" in out


def test_sweep_without_dry_run_is_not_silently_a_no_op() -> None:
    # Until detectors land at step 8, a real sweep must fail loudly rather than
    # return success and write nothing.
    with pytest.raises(NotImplementedError, match="step 8"):
        watcher._sweep(
            datetime(2019, 1, 1, tzinfo=UTC),
            datetime(2019, 1, 2, tzinfo=UTC),
            dry_run=False,
        )


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        watcher.main([])
