"""Performance ratio and temperature correction — the normalisation layer.

This is the module that makes a real deficit visible. Output tracks sunlight, so
a cloudy week and a failed string look identical in MWh; dividing by the light
that actually arrived removes the weather and leaves the plant.

    PR = E_measured / (P_STC * H_POA / G_STC)

with E in kWh, P_STC the DC nameplate in kW, H_POA the plane-of-array insolation
in kWh/m^2 and G_STC = 1 kW/m^2. PR is dimensionless: 1.0 would be a lossless
plant at STC, and a healthy real plant runs 0.75-0.85.

**Uncorrected PR falls every summer on a perfectly healthy plant.** Silicon
loses roughly 0.4% of its power per degree above 25 °C, and a module in full sun
sits 25-30 K above ambient. At a 45 °C cell temperature that is already an 8%
loss — indistinguishable, in the raw number, from a real fault. Temperature-
corrected PR removes it, and the difference between the two curves *is* the
seasonal false alarm made visible.

Every function here is deterministic. No LLM, no thresholds beyond the ones
passed in explicitly (CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "G_STC",
    "T_STC",
    "PerformanceResult",
    "cell_temperature_faiman",
    "compute_pr",
    "energy_kwh",
    "insolation_kwh_m2",
    "temperature_correction_factor",
]

G_STC = 1000.0  # W/m^2
T_STC = 25.0  # degC


def _interval_hours(index: pd.DatetimeIndex) -> float:
    """Sampling interval in hours, inferred from the index."""
    if len(index) < 2:
        raise ValueError("need at least two samples to infer the interval")
    deltas = pd.Series(index).diff().dropna()
    seconds = float(deltas.mode().iloc[0].total_seconds())
    if seconds <= 0:
        raise ValueError("non-monotonic or duplicated timestamps")
    return seconds / 3600.0


def energy_kwh(power_kw: pd.Series) -> float:
    """Integrate a power series to energy.

    Uses the modal interval rather than the mean so that a few gaps do not
    stretch the assumed interval and inflate the energy total.
    """
    clean = pd.to_numeric(power_kw, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    return float(clean.sum() * _interval_hours(pd.DatetimeIndex(clean.index)))


def insolation_kwh_m2(poa_wm2: pd.Series) -> float:
    """Integrate plane-of-array irradiance to insolation in kWh/m^2."""
    clean = pd.to_numeric(poa_wm2, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    hours = _interval_hours(pd.DatetimeIndex(clean.index))
    return float(clean.sum() * hours / 1000.0)


def cell_temperature_faiman(
    poa_wm2: pd.Series,
    temp_ambient_c: pd.Series,
    wind_speed_ms: pd.Series | None = None,
    u0: float = 25.0,
    u1: float = 6.84,
) -> pd.Series:
    """Estimate cell temperature with the Faiman model.

        T_cell = T_amb + G_poa / (u0 + u1 * v_wind)

    Chosen over the SAPM temperature model because it needs only ambient
    temperature and wind speed — no module-specific coefficient set — so it
    still works on a dataset with no measured module temperature. The default
    coefficients are the IEC 61853-2 values for an open-rack module.

    At 800 W/m^2 with 1 m/s of wind this puts the cell about 25 K above ambient,
    which is the number behind the whole summer-derating story.
    """
    poa = pd.to_numeric(poa_wm2, errors="coerce")
    ambient = pd.to_numeric(temp_ambient_c, errors="coerce")
    if wind_speed_ms is None:
        wind = pd.Series(1.0, index=poa.index)
    else:
        wind = pd.to_numeric(wind_speed_ms, errors="coerce").fillna(1.0).clip(lower=0.0)
    return ambient + poa / (u0 + u1 * wind)


def temperature_correction_factor(
    cell_temp_c: pd.Series,
    gamma_pdc: float,
) -> pd.Series:
    """The factor that converts measured output to its 25 °C equivalent.

    A module at ``T_cell`` produces ``1 + gamma * (T_cell - 25)`` times its STC
    power. Dividing by that factor answers "what would this plant have produced
    if it were at STC temperature?", which is the only way to compare a July
    week against a January one.

    Args:
        gamma_pdc: Power temperature coefficient per °C, negative for silicon
            (e.g. -0.00458 for the Sharp NU-U235F2, i.e. -0.458 %/°C).
    """
    if gamma_pdc > 0:
        raise ValueError(
            f"gamma_pdc must be negative for silicon; got {gamma_pdc:+g}. "
            "A positive coefficient would make hot modules look better, "
            "inverting the summer correction."
        )
    factor = 1.0 + gamma_pdc * (pd.to_numeric(cell_temp_c, errors="coerce") - T_STC)
    # Guard the divisor: a spurious 300 °C reading would otherwise drive the
    # factor negative and flip the sign of the corrected output.
    return factor.clip(lower=0.3)


@dataclass(frozen=True)
class PerformanceResult:
    """Performance ratio over one window, with the terms that produced it."""

    pr: float
    pr_temperature_corrected: float
    energy_kwh: float
    insolation_kwh_m2: float
    reference_yield_kwh: float
    mean_cell_temp_c: float
    samples_used: int
    samples_excluded_low_light: int
    samples_missing_power: int
    interval_hours: float

    @property
    def data_completeness(self) -> float:
        """Fraction of usable-light intervals that actually had a power reading.

        Load-bearing for honesty: a PR computed from 14% of the intervals is
        not the same claim as a PR computed from 98%, and a finding must never
        present them identically.
        """
        denominator = self.samples_used + self.samples_missing_power
        return self.samples_used / denominator if denominator else 0.0

    @property
    def temperature_loss_fraction(self) -> float:
        """How much of the deficit is explained by temperature alone.

        This is the number that separates a real fault from the seasonal false
        alarm: if uncorrected PR is down 8% and this says 8%, nothing is wrong.
        """
        if self.pr_temperature_corrected <= 0:
            return 0.0
        return 1.0 - (self.pr / self.pr_temperature_corrected)


def compute_pr(
    frame: pd.DataFrame,
    dc_capacity_kw: float,
    gamma_pdc: float,
    power_column: str = "ac_power_kw",
    poa_column: str = "poa_wm2",
    min_poa_wm2: float = 200.0,
    u0: float = 25.0,
    u1: float = 6.84,
) -> PerformanceResult:
    """Performance ratio, raw and temperature-corrected, over a window.

    Args:
        min_poa_wm2: Samples below this irradiance are excluded. At low light
            the PR denominator becomes small and the ratio explodes; including
            dawn and dusk makes PR a noise generator rather than a measurement.
            It also excludes night, where the numerator is zero.

    Measured module temperature is used when the column is present, since it is
    a direct observation. The Faiman estimate is the fallback, and which one was
    used is reported so a reviewer can tell.
    """
    if dc_capacity_kw <= 0:
        raise ValueError(f"dc_capacity_kw must be positive; got {dc_capacity_kw}")
    if power_column not in frame or poa_column not in frame:
        missing = {power_column, poa_column} - set(frame.columns)
        raise KeyError(f"frame is missing required columns: {sorted(missing)}")

    poa_all = pd.to_numeric(frame[poa_column], errors="coerce")
    power_all = pd.to_numeric(frame[power_column], errors="coerce")

    # Infer the sampling interval from the full index, before any masking:
    # a daylight mask can leave a single sample on a short or gappy day, and
    # one sample cannot tell you its own sampling rate.
    try:
        interval = _interval_hours(pd.DatetimeIndex(frame.index))
    except ValueError:
        interval = 0.0

    light_mask = poa_all >= min_poa_wm2
    excluded = int((poa_all.notna() & ~light_mask).sum())

    # The numerator and the denominator MUST cover the same intervals. If power
    # is missing while irradiance keeps logging — a telemetry gap, which is
    # common and lasts weeks — integrating all the insolation against only the
    # surviving energy manufactures a catastrophic phantom deficit. NIST
    # system 4902 loses AC power for 35 days in mid-2016 while the weather
    # station runs normally; scored naively that month reads PR = 0.10, a 90%
    # loss that never happened.
    mask = light_mask & power_all.notna()
    gaps = int((light_mask & power_all.isna()).sum())
    window = frame[mask]

    if len(window) < 1 or interval <= 0:
        return PerformanceResult(
            pr=float("nan"),
            pr_temperature_corrected=float("nan"),
            energy_kwh=0.0,
            insolation_kwh_m2=0.0,
            reference_yield_kwh=0.0,
            mean_cell_temp_c=float("nan"),
            samples_used=0,
            samples_excluded_low_light=excluded,
            samples_missing_power=gaps,
            interval_hours=0.0,
        )

    power_w = pd.to_numeric(window[power_column], errors="coerce")
    poa_w = pd.to_numeric(window[poa_column], errors="coerce")
    energy = float(power_w.sum() * interval)
    insolation = float(poa_w.sum() * interval / 1000.0)
    # Reference yield: what a lossless plant of this rating would make from
    # exactly this much light.
    reference = dc_capacity_kw * insolation / (G_STC / 1000.0)

    if "temp_module_c" in window and window["temp_module_c"].notna().any():
        cell_temp = pd.to_numeric(window["temp_module_c"], errors="coerce")
    else:
        cell_temp = cell_temperature_faiman(
            window[poa_column],
            window.get("temp_ambient_c", pd.Series(T_STC, index=window.index)),
            window.get("wind_speed_ms"),
            u0=u0,
            u1=u1,
        )

    factor = temperature_correction_factor(cell_temp, gamma_pdc)
    power = pd.to_numeric(window[power_column], errors="coerce")
    corrected_energy = float((power / factor).sum() * interval)

    return PerformanceResult(
        pr=energy / reference if reference > 0 else float("nan"),
        pr_temperature_corrected=(
            corrected_energy / reference if reference > 0 else float("nan")
        ),
        energy_kwh=energy,
        insolation_kwh_m2=insolation,
        reference_yield_kwh=reference,
        mean_cell_temp_c=float(np.nanmean(cell_temp.to_numpy(dtype=float))),
        samples_used=len(window),
        samples_excluded_low_light=excluded,
        samples_missing_power=gaps,
        interval_hours=interval,
    )


def pr_timeseries(
    frame: pd.DataFrame,
    dc_capacity_kw: float,
    gamma_pdc: float,
    freq: str = "1D",
    min_poa_wm2: float = 200.0,
    power_column: str = "ac_power_kw",
) -> pd.DataFrame:
    """Rolled-up PR per period — the series behind the Plant tab's PR chart.

    Returns columns ``pr``, ``pr_temperature_corrected``, ``energy_kwh``,
    ``insolation_kwh_m2``, ``mean_cell_temp_c`` and ``samples_used``.
    """
    rows: list[dict[str, float]] = []
    index: list[Any] = []
    for period, group in frame.groupby(pd.Grouper(freq=freq)):
        if len(group) < 2:
            continue
        result = compute_pr(
            group,
            dc_capacity_kw=dc_capacity_kw,
            gamma_pdc=gamma_pdc,
            min_poa_wm2=min_poa_wm2,
            power_column=power_column,
        )
        if result.samples_used == 0:
            continue
        index.append(period)
        rows.append(
            {
                "pr": result.pr,
                "pr_temperature_corrected": result.pr_temperature_corrected,
                "energy_kwh": result.energy_kwh,
                "insolation_kwh_m2": result.insolation_kwh_m2,
                "mean_cell_temp_c": result.mean_cell_temp_c,
                "samples_used": float(result.samples_used),
                "data_completeness": result.data_completeness,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "pr",
                "pr_temperature_corrected",
                "energy_kwh",
                "insolation_kwh_m2",
                "mean_cell_temp_c",
                "samples_used",
                "data_completeness",
            ]
        )
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="period"))
