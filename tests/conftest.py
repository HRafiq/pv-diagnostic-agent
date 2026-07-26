"""Shared fixtures.

The synthetic plant here is a *test article*, not a simulator. It exists so tool
behaviour can be asserted against hand-computable answers — "a string at zero
current shows a share of zero" — and it is deliberately not used for any
evaluation case. Golden cases run on real measured NIST data with physics-level
injections, because a tool tested against a plant it also scores would only be
proving the two agree with each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clock import FrozenClock
from src.tools import ToolContext

STRINGS = 7
CAPACITY_KW = 100.0
GAMMA = -0.0045


def synthetic_frame(
    days: int = 60,
    start: str = "2017-04-01",
    peak_poa: float = 900.0,
    freq: str = "15min",
    strings: int = STRINGS,
    weather: float = 0.06,
    seed: int = 7,
) -> pd.DataFrame:
    """A healthy plant. Sinusoidal day, flat performance, no faults.

    Times are UTC and the notional site is at UTC+0, so "site-local hour" and
    "UTC hour" coincide — one fewer thing to reason about when an hour-of-day
    assertion fails.

    `weather` puts a small day-to-day spread on irradiance, and it is on by
    default for a reason: a plant with *identical* days has exactly zero
    day-to-day scatter, which is a degenerate input no real dataset produces. A
    tool tuned against it would divide by zero, or worse, look correct because
    the denominator happened never to be exercised. Set it to 0.0 only when a
    test is specifically about the degenerate case.
    """
    index = pd.date_range(start, periods=days * 24 * 4, freq=freq, tz="UTC")
    hour = index.hour + index.minute / 60.0
    # Daylight 06:00-18:00, half-sine.
    daylight = (hour >= 6.0) & (hour <= 18.0)
    poa = np.where(daylight, peak_poa * np.sin(np.pi * (hour - 6.0) / 12.0), 0.0)
    if weather:
        rng = np.random.default_rng(seed)
        by_day = 1.0 - rng.uniform(0.0, weather, size=days)
        day_number = (index - index[0]).days.to_numpy()
        poa = poa * by_day[day_number]
    poa = np.clip(poa, 0.0, None)

    ambient = 15.0 + 8.0 * np.sin(np.pi * np.clip((hour - 6.0) / 12.0, 0, 1))
    module = ambient + poa / 30.0
    dc = CAPACITY_KW * (poa / 1000.0) * (1.0 + GAMMA * (module - 25.0)) * 0.88
    dc = np.clip(dc, 0.0, None)
    ac = dc * 0.97

    data: dict[str, np.ndarray] = {
        "poa_wm2": poa,
        "ac_power_kw": ac,
        "dc_power_kw": dc,
        "temp_ambient_c": ambient,
        "temp_module_c": module,
        # Not a constant: a channel that never changes is legitimately
        # indistinguishable from a frozen sensor, and `detect_stuck_channels`
        # is right to say so. A fixture that trips it on every run would train
        # the reader to ignore the finding.
        "wind_speed_ms": 1.5
        + 0.6 * np.sin(np.arange(len(index)) * 0.37)
        + 0.2 * np.cos(np.arange(len(index)) * 0.11),
    }
    # Total string current is proportional to irradiance; each string carries an
    # equal share on a healthy array.
    total_current = poa / 1000.0 * 350.0
    for n in range(1, strings + 1):
        data[f"string_current_a_{n}"] = total_current / strings
    return pd.DataFrame(data, index=index)


def two_year_frame(**kwargs: object) -> pd.DataFrame:
    """The same span in two consecutive years.

    A seasonal comparison — "was this window's weather unusual?" — needs the
    same calendar weeks from somewhere else in the record. One year of data
    cannot answer it, and the tool correctly says so rather than comparing a
    window against itself.
    """
    first = synthetic_frame(start="2016-04-01", **kwargs)  # type: ignore[arg-type]
    second = synthetic_frame(start="2017-04-01", **kwargs)  # type: ignore[arg-type]
    return pd.concat([first, second])


def context_for(frame: pd.DataFrame, **overrides: object) -> ToolContext:
    kwargs: dict[str, object] = {
        "frame": frame,
        "dc_capacity_kw": CAPACITY_KW,
        "gamma_pdc": GAMMA,
        "latitude": 39.13,
        # Longitude 0 so that solar noon lands at 12:00 UTC, matching the
        # fixture's noon-peaking day. At the real site's -77.2 the modelled
        # clear-sky curve peaks five hours after the synthetic one, and every
        # measured-against-clear-sky ratio comes out around 0.18 — a fixture
        # artefact that would look exactly like a catastrophically soiled
        # pyranometer.
        "longitude": 0.0,
        "altitude_m": 138.0,
        "tilt_deg": 20.0,
        "azimuth_deg": 180.0,
        "ac_ceiling_kw": 95.0,
        "utc_offset_hours": 0.0,
        "scope": "test plant",
    }
    kwargs.update(overrides)
    return ToolContext(**kwargs)  # type: ignore[arg-type]


@pytest.fixture
def plant() -> pd.DataFrame:
    """Sixty days, so a tool that needs history before its window has some.

    Several measurements are only meaningful against a trailing baseline or
    against the same calendar span elsewhere in the record. A fixture no longer
    than one investigation window turns those into errors and quietly excludes
    them from every contract test.
    """
    return synthetic_frame()


# The window the tool-contract tests measure, leaving 45 days of history in
# front of it for anything that reaches backwards.
WINDOW = {"start": "2017-05-16", "end": "2017-05-30"}


@pytest.fixture
def ctx(plant: pd.DataFrame) -> ToolContext:
    return context_for(plant)


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(pd.Timestamp("2017-04-15T12:00:00Z").to_pydatetime())
