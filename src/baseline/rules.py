"""Deterministic rules engine — the *baseline*, not a component.

This is what the agent must beat to justify itself. It is a genuine attempt, not
a straw man: it uses the same measurements the agent's tools expose and applies
the checks a competent engineer would script.

Its structural limits are the interesting part, and they are inherent rather
than accidental:

* it evaluates a fixed sequence of checks in a fixed order, so it cannot decide
  that one result makes a different measurement worth taking;
* it commits to the first rule that fires, so it cannot hold two causes open;
* it has no way to say "these two survive and here is the test that separates
  them", because that requires reasoning about what was *not* measured.

If it wins outright on the golden set, that is a more credible and more useful
finding than "I built an agent", and it gets published either way.

No LLM anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.physics.clearsky import clearsky_poa
from src.physics.modelchain import ExpectationModel
from src.physics.performance import compute_pr

__all__ = ["RulesEngine", "RulesVerdict"]


@dataclass(frozen=True)
class RulesVerdict:
    category: str | None
    cause: str | None
    settled: bool
    checks_run: tuple[str, ...]
    evidence: dict[str, float]
    candidate_causes: tuple[str, ...] = ()
    resolving_measurement: str | None = None


@dataclass
class RulesEngine:
    """Threshold-based diagnosis over a fixed check sequence."""

    dc_capacity_kw: float
    gamma_pdc: float
    latitude: float
    longitude: float
    altitude_m: float
    tilt_deg: float
    azimuth_deg: float
    ac_ceiling_kw: float | None = None

    # Thresholds, all explicit.
    completeness_floor: float = 0.80
    deficit_open: float = 0.05
    string_imbalance: float = 0.03
    clearsky_shortfall: float = 0.12
    pr_rise: float = 0.03
    flat_ceiling_fraction: float = 0.02

    def diagnose(self, frame: pd.DataFrame) -> RulesVerdict:
        checks: list[str] = []
        evidence: dict[str, float] = {}

        # --- 1. is the data even usable? ---------------------------------
        checks.append("check_data_completeness")
        result = compute_pr(frame, self.dc_capacity_kw, self.gamma_pdc)
        evidence["completeness"] = round(result.data_completeness, 4)
        if result.data_completeness < self.completeness_floor:
            return RulesVerdict(
                "not_the_plant", "telemetry_gap", True, tuple(checks), evidence
            )

        # --- 2. per-string balance ---------------------------------------
        checks.append("per_mppt_current_balance")
        imbalance, worst = self._string_imbalance(frame)
        evidence["worst_string_deviation"] = round(imbalance, 4)
        evidence["worst_string_index"] = float(worst)

        # --- 3. is the sensor lying? -------------------------------------
        checks.append("check_clearsky_consistency")
        shortfall = self._clearsky_shortfall(frame)
        evidence["clearsky_shortfall"] = round(shortfall, 4)

        # --- 4. how far below the model? ---------------------------------
        checks.append("compute_expected_output")
        model = ExpectationModel(
            self.dc_capacity_kw, self.gamma_pdc, self.ac_ceiling_kw
        )
        expectation = model.run(frame)
        deficit = expectation.deficit_fraction()
        evidence["deficit_vs_model"] = round(float(deficit), 4)

        checks.append("compute_temp_corrected_pr")
        evidence["pr"] = round(result.pr, 4)
        evidence["pr_temperature_corrected"] = round(result.pr_temperature_corrected, 4)

        # --- 5. flat ceiling? --------------------------------------------
        checks.append("check_ac_ceiling")
        ceiling_share = self._ceiling_share(frame)
        evidence["flat_ceiling_share"] = round(ceiling_share, 4)

        # ---------------- fixed decision order ----------------------------
        # A sensor reading low makes PR rise, which a real loss cannot do.
        if shortfall > self.clearsky_shortfall and deficit < self.deficit_open:
            return RulesVerdict(
                "not_the_plant", "sensor_drift", True, tuple(checks), evidence
            )

        if ceiling_share > self.flat_ceiling_fraction:
            # The engine cannot separate clipping from curtailment, and has no
            # way to express that, so it commits. This is exactly the failure
            # the unresolvable cases are designed to expose.
            return RulesVerdict("by_design", "clipping", True, tuple(checks), evidence)

        if imbalance > self.string_imbalance:
            return RulesVerdict("fault", "string_outage", True, tuple(checks), evidence)

        if deficit > self.deficit_open:
            return RulesVerdict("recoverable", "soiling", True, tuple(checks), evidence)

        return RulesVerdict("by_design", "healthy", True, tuple(checks), evidence)

    # ------------------------------------------------------------------
    def _string_imbalance(self, frame: pd.DataFrame) -> tuple[float, int]:
        columns = sorted(
            (c for c in frame.columns if c.startswith("string_current_a_")),
            key=lambda c: int(c.rsplit("_", 1)[-1]),
        )
        if len(columns) < 2 or "poa_wm2" not in frame:
            return 0.0, 0
        mask = pd.to_numeric(frame["poa_wm2"], errors="coerce") >= 400.0
        window = frame.loc[mask, columns].apply(pd.to_numeric, errors="coerce")
        totals = window.sum(axis=1)
        floor = 0.2 * float(totals.median()) if len(totals) else 0.0
        usable = totals > max(floor, 1e-6)
        if not usable.any():
            return 0.0, 0
        shares = window[usable].div(totals[usable], axis=0).mean()
        expected = 1.0 / len(columns)
        deviations = (shares - expected).abs()
        worst = int(deviations.idxmax().rsplit("_", 1)[-1])
        return float(deviations.max()), worst

    def _clearsky_shortfall(self, frame: pd.DataFrame) -> float:
        """How far measured irradiance sits below clear-sky on the clearest days.

        Compared on the top decile of days only: an overcast fortnight is
        legitimately far below clear-sky, and averaging it in would flag every
        cloudy period as a broken sensor.
        """
        if "poa_wm2" not in frame or frame.empty:
            return 0.0
        index = pd.DatetimeIndex(frame.index)
        modelled = clearsky_poa(
            index,
            self.latitude,
            self.longitude,
            self.altitude_m,
            self.tilt_deg,
            self.azimuth_deg,
        ).poa_global
        measured = pd.to_numeric(frame["poa_wm2"], errors="coerce")
        good = modelled > 300.0
        if not good.any():
            return 0.0
        ratio = (measured[good] / modelled[good]).replace([np.inf, -np.inf], np.nan)
        daily = ratio.groupby(pd.DatetimeIndex(ratio.index).normalize()).median()
        if daily.empty:
            return 0.0
        clearest = daily.quantile(0.9)
        return float(max(0.0, 1.0 - clearest))

    def _ceiling_share(self, frame: pd.DataFrame, min_run: int = 3) -> float:
        """Fraction of daylight intervals sitting on a genuinely *flat* ceiling.

        Proximity to the peak is not enough: every smooth midday curve has
        samples near its own maximum, so a "within 1% of max" test flags a
        perfectly healthy sine as clipped. What distinguishes a real ceiling is
        that consecutive samples stop changing — the inverter holds one value
        while the sun keeps climbing. So require a *run* of near-identical
        samples at high power, not merely a high one.
        """
        if "ac_power_kw" not in frame:
            return 0.0
        power = pd.to_numeric(frame["ac_power_kw"], errors="coerce").dropna()
        if power.empty:
            return 0.0
        peak = float(power.max())
        if peak <= 0:
            return 0.0
        daylight = power[power > 0.15 * peak]
        if len(daylight) < min_run:
            return 0.0

        values = daylight.astype(float)
        near_top = values >= 0.98 * peak
        # Group consecutive samples that are both near the top and flat
        # relative to their neighbour.
        flat = near_top & (values.diff().abs() <= 0.005 * peak)
        run_id = (~flat).cumsum()
        run_length = flat.groupby(run_id).transform("sum")
        pinned = int((flat & (run_length >= min_run)).sum())
        return float(pinned) / float(len(daylight))
