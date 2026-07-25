"""The expectation model: what *should* this plant be producing right now?

Deliberately the **coarse** stack — PVWatts DC, Faiman cell temperature,
PVWatts inverter. Two reasons, and the second is the important one:

1. PVWatts needs one free parameter beyond the nameplate (``gamma_pdc``), so it
   works on a plant whose exact module coefficients are unknown, which is the
   normal case in the field.
2. **Anti-circularity** (CLAUDE.md, DECISION 0008). Faults are injected by
   perturbing *measured* series, never by simulating a plant. If expectations
   were instead formed with the same detailed model used to generate the truth,
   every injected deficit would be trivially detectable and the accuracy figure
   would measure nothing but self-consistency.

The model is driven by *measured* POA rather than modelled clear-sky, so it
answers "given the light that actually arrived, how much power should there have
been?". Comparing measured POA against clear-sky is a separate question — that
is how a soiled or drifting sensor is caught — and lives in `clearsky.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.physics.performance import G_STC, T_STC, cell_temperature_faiman

__all__ = ["ExpectationModel", "ExpectationResult", "module_gamma_pdc"]


def module_gamma_pdc(module_name: str) -> float | None:
    """Look up a module's power temperature coefficient in the CEC database.

    Returns the coefficient per °C (negative), or ``None`` if the module is not
    in the database. Using the manufacturer's published figure keeps the
    expectation model independent of the data it is meant to predict: fitting
    gamma to the plant's own output would absorb a real fault into the
    "expected" curve and hide it.
    """
    import pvlib

    try:
        cec = pvlib.pvsystem.retrieve_sam("CECMod")
    except Exception:  # pragma: no cover - offline
        return None
    key = module_name.replace("-", "_").replace(" ", "_").lower()
    if not key:
        return None
    columns = list(cec.columns)
    # Exact match first.
    for column in columns:
        if column.lower() == key:
            return float(cec[column]["gamma_r"]) / 100.0
    # PVDAQ records the model without the manufacturer ("NU-U235F2"), while the
    # CEC key includes it ("Sharp_NU_U235F2"). Fall back to a suffix match, but
    # only when it is unambiguous — silently picking one of several candidates
    # would put a wrong temperature coefficient at the centre of every
    # correction in the project.
    suffix = [c for c in columns if c.lower().endswith("_" + key)]
    if len(suffix) == 1:
        return float(cec[suffix[0]]["gamma_r"]) / 100.0
    return None


@dataclass(frozen=True)
class ExpectationResult:
    """Expected output alongside the measured series it is compared against."""

    expected_ac_kw: pd.Series
    expected_dc_kw: pd.Series
    cell_temperature_c: pd.Series
    measured_ac_kw: pd.Series
    clipped: pd.Series

    @property
    def residual_kw(self) -> pd.Series:
        """Measured minus expected. Negative means under-performance."""
        return self.measured_ac_kw - self.expected_ac_kw

    def deficit_fraction(self, min_expected_kw: float = 1.0) -> float:
        """Energy shortfall as a fraction of expected, over the whole window.

        Restricted to intervals where a meaningful amount was expected: near
        sunrise the ratio of two tiny numbers is noise, and averaging it in
        would swamp the real signal from the middle of the day.
        """
        mask = self.expected_ac_kw >= min_expected_kw
        expected = float(self.expected_ac_kw[mask].sum())
        measured = float(self.measured_ac_kw[mask].sum())
        if expected <= 0:
            return float("nan")
        return 1.0 - measured / expected


@dataclass(frozen=True)
class ExpectationModel:
    """Coarse PVWatts-style expectation model for a fixed-tilt plant.

    Args:
        dc_capacity_kw: DC nameplate at STC.
        gamma_pdc: Power temperature coefficient per °C (negative).
        ac_capacity_kw: Inverter AC rating. Sets the ceiling that produces
            clipping — flat-topped midday power that is by design, not a fault.
        inverter_efficiency: Nominal peak efficiency.
        system_losses: Everything PVWatts lumps together — DC wiring,
            mismatch, connections, light-induced degradation, nameplate
            tolerance, availability. 0.14 is the PVWatts default for a
            well-built system and is NOT fitted to this plant's data.
    """

    dc_capacity_kw: float
    gamma_pdc: float
    ac_capacity_kw: float | None = None
    inverter_efficiency: float = 0.96
    system_losses: float = 0.14
    u0: float = 25.0
    u1: float = 6.84

    def __post_init__(self) -> None:
        if self.dc_capacity_kw <= 0:
            raise ValueError("dc_capacity_kw must be positive")
        if self.gamma_pdc > 0:
            raise ValueError(
                f"gamma_pdc must be negative for silicon; got {self.gamma_pdc:+g}"
            )
        if not 0.0 <= self.system_losses < 1.0:
            raise ValueError("system_losses must be a fraction in [0, 1)")

    def run(
        self,
        frame: pd.DataFrame,
        poa_column: str = "poa_wm2",
        power_column: str = "ac_power_kw",
    ) -> ExpectationResult:
        """Predict AC power from measured irradiance and weather."""
        poa = pd.to_numeric(frame[poa_column], errors="coerce").fillna(0.0)

        if "temp_module_c" in frame and frame["temp_module_c"].notna().any():
            cell_temp = pd.to_numeric(frame["temp_module_c"], errors="coerce")
            cell_temp = cell_temp.fillna(
                cell_temperature_faiman(
                    poa,
                    frame.get("temp_ambient_c", pd.Series(T_STC, index=frame.index)),
                    frame.get("wind_speed_ms"),
                    self.u0,
                    self.u1,
                )
            )
        else:
            cell_temp = cell_temperature_faiman(
                poa,
                frame.get("temp_ambient_c", pd.Series(T_STC, index=frame.index)),
                frame.get("wind_speed_ms"),
                self.u0,
                self.u1,
            )

        # PVWatts DC: linear in irradiance, linearly derated by temperature.
        dc = (
            self.dc_capacity_kw
            * (poa / G_STC)
            * (1.0 + self.gamma_pdc * (cell_temp - T_STC))
        ).clip(lower=0.0)
        dc *= 1.0 - self.system_losses

        ac = dc * self.inverter_efficiency
        clipped = pd.Series(False, index=frame.index)
        if self.ac_capacity_kw is not None:
            ceiling = float(self.ac_capacity_kw)
            clipped = ac > ceiling
            ac = ac.clip(upper=ceiling)

        measured = pd.to_numeric(
            frame.get(power_column, pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )

        return ExpectationResult(
            expected_ac_kw=ac,
            expected_dc_kw=dc,
            cell_temperature_c=cell_temp,
            measured_ac_kw=measured,
            clipped=clipped,
        )
