from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from src.clock import Clock, FrozenClock, ReplayClock

START = datetime(2019, 1, 1, tzinfo=UTC)


def test_replay_clock_is_frozen_until_advanced() -> None:
    clock = ReplayClock(start=START)
    assert clock.now() == START
    assert clock.now() == START  # no drift at speed 0


def test_advance_moves_forward() -> None:
    clock = ReplayClock(start=START)
    assert clock.advance(timedelta(days=1)) == START + timedelta(days=1)
    assert clock.advance(timedelta(hours=6)) == START + timedelta(days=1, hours=6)


def test_advance_to_absolute_time() -> None:
    clock = ReplayClock(start=START)
    target = datetime(2019, 6, 30, 12, tzinfo=UTC)
    assert clock.advance_to(target) == target


def test_clock_cannot_rewind() -> None:
    clock = ReplayClock(start=START)
    clock.advance(timedelta(days=10))
    with pytest.raises(ValueError, match="rewind"):
        clock.advance_to(START)
    with pytest.raises(ValueError, match="negative"):
        clock.advance(timedelta(days=-1))


def test_naive_datetimes_are_rejected() -> None:
    # A naive timestamp is how a site-local dataset silently gets compared
    # against UTC, so it fails loudly at construction.
    with pytest.raises(ValueError, match="timezone-aware"):
        ReplayClock(start=datetime(2019, 1, 1))


def test_non_utc_input_is_normalised() -> None:
    acst = timezone(timedelta(hours=9, minutes=30))
    clock = ReplayClock(start=datetime(2019, 1, 1, 9, 30, tzinfo=acst))
    assert clock.now() == datetime(2019, 1, 1, 0, 0, tzinfo=UTC)


def test_end_bound_is_enforced() -> None:
    clock = ReplayClock(start=START, end=START + timedelta(days=2))
    clock.advance(timedelta(days=2))
    with pytest.raises(ReplayClock.ClockExhausted):
        clock.advance(timedelta(seconds=1))


def test_end_before_start_is_rejected() -> None:
    with pytest.raises(ValueError, match="precedes start"):
        ReplayClock(start=START, end=START - timedelta(days=1))


def test_negative_speed_is_rejected() -> None:
    with pytest.raises(ValueError, match="speed"):
        ReplayClock(start=START, speed=-1.0)


def test_frozen_clock_never_moves() -> None:
    clock = FrozenClock(at=START)
    assert clock.now() == clock.now() == START
    clock.set(START + timedelta(days=5))
    assert clock.now() == START + timedelta(days=5)


def test_both_clocks_satisfy_the_protocol() -> None:
    assert isinstance(ReplayClock(start=START), Clock)
    assert isinstance(FrozenClock(at=START), Clock)
