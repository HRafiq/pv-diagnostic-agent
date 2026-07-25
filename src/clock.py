"""The only time source in this project.

There is no free real-time public PV feed, so "now" is a *position in a
historical dataset*. Everything that depends on the present moment — the
Watcher's "what arrived since yesterday", a finding's age, a trailing baseline
window — reads it from an injected `Clock`.

`datetime.now()` must never appear anywhere in ``src/``. ``tests/test_no_wall_
clock.py`` scans the source tree and fails the build if it does. The reason is
not purity: a stray wall-clock read means a replayed 2019 dataset gets compared
against a 2026 "today", every trailing window comes back empty, and the failure
is silent.

Accelerated replay still needs *some* real elapsed-time source. It uses
``time.monotonic()``, which measures a duration rather than reading a calendar,
so it cannot leak a real date into a computation.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "FrozenClock", "ReplayClock"]


@runtime_checkable
class Clock(Protocol):
    """Anything that can tell the system what time it is."""

    def now(self) -> datetime:
        """Current simulated time, always timezone-aware UTC."""
        ...


def _require_utc(ts: datetime, field: str) -> datetime:
    if ts.tzinfo is None:
        raise ValueError(
            f"{field} must be timezone-aware; got naive {ts!r}. "
            "Naive timestamps are how a dataset in site-local time silently "
            "gets compared against UTC."
        )
    return ts.astimezone(UTC)


class ReplayClock:
    """A clock positioned somewhere in a historical dataset.

    Two modes:

    * ``speed == 0`` (default) — the clock is frozen until something advances
      it. This is the mode ``watcher.py`` uses: it steps the clock forward one
      data interval, runs the sweep against everything that "arrived", writes
      findings, and steps again. Fully deterministic and reproducible.
    * ``speed > 0`` — the clock also drifts forward with real elapsed time,
      multiplied by ``speed``. Useful for demoing the dashboard live. Not used
      in evaluation, because it makes runs unreproducible.

    Args:
        start: Where in history to begin. Must be timezone-aware.
        end: Optional hard stop. Advancing past it raises ``ClockExhausted``.
        speed: Simulated seconds per real second. 0 disables real-time drift.
    """

    class ClockExhausted(RuntimeError):
        """Raised when the replay would move past the configured end."""

    def __init__(
        self,
        start: datetime,
        end: datetime | None = None,
        speed: float = 0.0,
    ) -> None:
        if speed < 0:
            raise ValueError(f"speed must be >= 0; got {speed}")
        self._start = _require_utc(start, "start")
        self._end = _require_utc(end, "end") if end is not None else None
        if self._end is not None and self._end < self._start:
            raise ValueError(f"end {self._end} precedes start {self._start}")
        self._speed = speed
        # Simulated time accrued by explicit advance() calls.
        self._offset = timedelta(0)
        # Real-time anchor for accelerated mode. monotonic() is a duration
        # source, not a calendar — it cannot leak a real date into the model.
        self._anchor = time.monotonic()

    @property
    def start(self) -> datetime:
        return self._start

    @property
    def end(self) -> datetime | None:
        return self._end

    @property
    def speed(self) -> float:
        return self._speed

    def now(self) -> datetime:
        drift = timedelta(0)
        if self._speed > 0:
            drift = timedelta(seconds=(time.monotonic() - self._anchor) * self._speed)
        return self._start + self._offset + drift

    def advance(self, delta: timedelta) -> datetime:
        """Step the clock forward. Returns the new simulated time."""
        if delta < timedelta(0):
            raise ValueError(
                f"cannot advance by a negative interval ({delta}); "
                "a replay clock only moves forward"
            )
        candidate = self.now() + delta
        self._guard(candidate)
        self._offset += delta
        return self.now()

    def advance_to(self, ts: datetime) -> datetime:
        """Jump the clock to an absolute time. Must not move backwards."""
        target = _require_utc(ts, "ts")
        current = self.now()
        if target < current:
            raise ValueError(
                f"cannot rewind from {current.isoformat()} to {target.isoformat()}"
            )
        self._guard(target)
        self._offset += target - current
        return self.now()

    def _guard(self, candidate: datetime) -> None:
        if self._end is not None and candidate > self._end:
            raise self.ClockExhausted(
                f"replay would move to {candidate.isoformat()}, past the "
                f"configured end {self._end.isoformat()}"
            )

    def __repr__(self) -> str:
        return (
            f"ReplayClock(now={self.now().isoformat()}, "
            f"start={self._start.isoformat()}, speed={self._speed})"
        )


class FrozenClock:
    """A clock that never moves. For tests and for rendering a saved trace."""

    def __init__(self, at: datetime) -> None:
        self._at = _require_utc(at, "at")

    def now(self) -> datetime:
        return self._at

    def set(self, at: datetime) -> None:
        self._at = _require_utc(at, "at")

    def __repr__(self) -> str:
        return f"FrozenClock(at={self._at.isoformat()})"
