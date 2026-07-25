"""Clear-sky irradiance, and the timezone recovery that depends on it.

Clear-sky modelling answers a question the plant's own sensors cannot: *how much
light should there have been?* That is what separates a soiled or drifting
irradiance sensor from a genuinely dim day — and, before any of that, it is what
tells us which timezone the data logger was keeping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pvlib

__all__ = [
    "ClearSkyResult",
    "OffsetScan",
    "clearsky_poa",
    "scan_utc_offset",
    "solar_position",
]


def solar_position(
    times: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    altitude_m: float = 0.0,
) -> pd.DataFrame:
    """Apparent solar position. Requires a tz-aware index."""
    if times.tz is None:
        raise ValueError(
            "solar position needs tz-aware timestamps; a naive index silently "
            "computes the sun's position for the wrong moment"
        )
    result: pd.DataFrame = pvlib.solarposition.get_solarposition(
        times, latitude, longitude, altitude=altitude_m
    )
    return result


@dataclass(frozen=True)
class ClearSkyResult:
    """Modelled clear-sky irradiance in the plane of the array."""

    poa_global: pd.Series
    ghi: pd.Series
    dni: pd.Series
    dhi: pd.Series
    solar_zenith: pd.Series
    solar_azimuth: pd.Series


def clearsky_poa(
    times: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    altitude_m: float,
    tilt_deg: float,
    azimuth_deg: float,
    albedo: float = 0.2,
) -> ClearSkyResult:
    """Clear-sky plane-of-array irradiance for a fixed-tilt array.

    Uses the Ineichen model with the bundled Linke turbidity climatology — the
    pvlib default, and the one that needs no external data. Transposition is
    Perez, which handles the circumsolar and horizon-brightening components
    that matter most at high incidence angles (winter mornings, exactly where a
    naive isotropic model under-predicts and manufactures a phantom deficit).

    The result is an *upper bound* on what the array could have received. Real
    POA below it means cloud, soiling, shading, or a sensor problem — telling
    those apart is the agent's job, and this is the reference it works against.
    """
    location = pvlib.location.Location(
        latitude, longitude, altitude=altitude_m, tz=str(times.tz)
    )
    clearsky = location.get_clearsky(times, model="ineichen")
    solpos = solar_position(times, latitude, longitude, altitude_m)

    # Extraterrestrial irradiance and airmass feed the Perez coefficients.
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    airmass = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])

    total = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=azimuth_deg,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
        dni_extra=dni_extra,
        airmass=airmass,
        albedo=albedo,
        model="perez",
    )
    return ClearSkyResult(
        poa_global=total["poa_global"].fillna(0.0),
        ghi=clearsky["ghi"],
        dni=clearsky["dni"],
        dhi=clearsky["dhi"],
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
    )


def _clearest_days(series: pd.Series, keep_days: int) -> pd.Series:
    """Keep the N smoothest daylight profiles.

    A clear day traces a smooth single hump; cloud puts high-frequency notches
    into it. Scoring by the mean absolute *second* difference, normalised by the
    day's own peak, makes the measure scale-free and resolution-independent —
    so the same code works on 1-minute and 15-minute data without retuning.
    Selection is by rank, so it degrades gracefully at a site that simply has
    no perfectly clear days.
    """
    scores: dict[object, float] = {}
    index = pd.DatetimeIndex(series.index)
    for day, group in series.groupby(index.date):
        daylight = group[group > 50.0]
        if len(daylight) < 12:
            continue
        peak = float(daylight.max())
        if peak < 200.0:
            continue
        curvature = np.abs(np.diff(daylight.to_numpy(dtype=float), n=2))
        scores[day] = float(np.nanmean(curvature)) / peak

    if not scores:
        return series
    chosen = {d for d, _ in sorted(scores.items(), key=lambda kv: kv[1])[:keep_days]}
    mask = pd.Series(index.date, index=series.index).isin(chosen)
    selected = series[mask.to_numpy()]
    # Only daylight carries timing information; night is flat at every offset.
    return selected[selected > 20.0]


@dataclass(frozen=True)
class OffsetScan:
    """Result of recovering a logger's UTC offset from solar geometry."""

    offset_hours: float
    correlation: float
    runner_up_hours: float
    runner_up_correlation: float
    samples: int
    curve: dict[float, float]

    @property
    def margin(self) -> float:
        """How decisively the winner beat the next-best offset."""
        return self.correlation - self.runner_up_correlation

    @property
    def trustworthy(self) -> bool:
        # A real alignment correlates strongly and beats its neighbours
        # clearly. A weak or ambiguous peak means something else is wrong.
        return self.correlation >= 0.95 and self.margin >= 0.01


def scan_utc_offset(
    poa: pd.Series,
    latitude: float,
    longitude: float,
    altitude_m: float,
    tilt_deg: float,
    azimuth_deg: float,
    step_hours: float = 0.5,
    max_samples: int = 20000,
    clear_days: int = 30,
) -> OffsetScan:
    """Recover the logger's UTC offset by aligning measured POA to clear-sky.

    PVDAQ does not record a timezone, and both usual guesses are wrong
    somewhere: many loggers keep local *standard* time year round, not UTC and
    not local-with-DST. An hour of error puts the modelled sun an hour away from
    the measured sun, which makes a healthy plant look like it under-performs
    every morning and over-performs every afternoon.

    Rather than assume, try every half-hour offset and keep the one whose
    modelled clear-sky curve best correlates with the measured irradiance. Only
    the *timing* of the curve drives the answer, so cloud, soiling and
    calibration error do not bias it — they lower the correlation at every
    candidate offset equally.
    """
    series = pd.to_numeric(poa, errors="coerce").dropna()
    if series.empty:
        raise ValueError("no usable irradiance samples for timezone recovery")

    naive = series.copy()
    naive.index = pd.to_datetime(naive.index).tz_localize(None)

    # Score on the clearest days only. Correlating across all weather caps out
    # near r=0.85 — a cloudy day is dim whatever the offset, so it lowers every
    # candidate equally and flattens the very peak we are trying to locate.
    # Clear days are picked by *rank* on curve smoothness rather than an
    # absolute threshold, so the selection works at any sampling interval.
    naive = _clearest_days(naive, keep_days=clear_days)

    if len(naive) > max_samples:
        naive = naive.iloc[:: max(1, len(naive) // max_samples)]

    curve: dict[float, float] = {}
    for step in range(int(-12 / step_hours), int(14 / step_hours) + 1):
        offset = step * step_hours
        utc_index = pd.DatetimeIndex(
            naive.index - pd.Timedelta(hours=offset)
        ).tz_localize("UTC")
        try:
            modelled = clearsky_poa(
                utc_index, latitude, longitude, altitude_m, tilt_deg, azimuth_deg
            ).poa_global
        except Exception:  # pragma: no cover - defensive, pvlib edge cases
            continue
        measured = naive.to_numpy(dtype=float)
        expected = modelled.to_numpy(dtype=float)
        if np.std(measured) == 0 or np.std(expected) == 0:
            continue
        curve[offset] = float(np.corrcoef(measured, expected)[0, 1])

    if not curve:
        raise ValueError("timezone scan produced no candidates")

    ranked = sorted(curve.items(), key=lambda kv: kv[1], reverse=True)
    best_offset, best_corr = ranked[0]
    second_offset, second_corr = (
        ranked[1] if len(ranked) > 1 else (best_offset, best_corr)
    )
    return OffsetScan(
        offset_hours=best_offset,
        correlation=best_corr,
        runner_up_hours=second_offset,
        runner_up_correlation=second_corr,
        samples=len(series),
        curve=curve,
    )
