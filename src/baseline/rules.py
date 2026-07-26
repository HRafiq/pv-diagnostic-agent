"""Deterministic rules engine — the *baseline*, not a component.

This is what the agent must beat to justify itself, and it is a genuine attempt
rather than a straw man. It calls **the same eighteen tools the agent calls**,
through the same registry, with the same argument validation. That is deliberate
and it is the point of the comparison: if the baseline measured differently, a
gap between the two would be a gap in *measurement quality* and would say
nothing about reasoning. Measuring identically isolates the only variable worth
testing — what gets done with the numbers.

Its limits are structural rather than accidental, and no amount of threshold
tuning removes them:

* it runs a fixed sequence of measurements in a fixed order, so it cannot decide
  that one result makes a different measurement worth taking;
* it commits to the first rule that fires, so it cannot hold two causes open;
* it has no way to say "these two survive and here is the test that separates
  them", because that requires reasoning about what was *not* measured.

If it wins outright on the golden set, that is a more credible and more useful
finding than "I built an agent", and it gets published either way.

No LLM anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.tools import ToolContext, ToolError, ToolResult, run_tool

__all__ = ["CHECK_SEQUENCE", "RulesEngine", "RulesVerdict"]

# The fixed sequence. Every run takes exactly these measurements, in this order,
# whatever the data says — which is the definition of a pipeline and precisely
# what the agency metrics are built to detect.
CHECK_SEQUENCE: tuple[str, ...] = (
    "profile_data_quality",
    "compute_temp_corrected_pr",
    "per_mppt_current_balance",
    "check_clearsky_consistency",
    "check_ac_ceiling",
    "compute_expected_output",
    "time_of_day_profile",
    "weather_context",
    "daily_performance_trend",
)


@dataclass(frozen=True)
class RulesVerdict:
    category: str | None
    cause: str | None
    settled: bool
    checks_run: tuple[str, ...]
    evidence: dict[str, float]
    candidate_causes: tuple[str, ...] = ()
    resolving_measurement: str | None = None
    failed_checks: tuple[str, ...] = ()


@dataclass
class RulesEngine:
    """Threshold-based diagnosis over a fixed check sequence.

    Every threshold is an explicit field with a stated default. They are the
    thresholds a competent engineer would script, and they are tuned on the
    tuning split only — the held-back split is never looked at while setting
    them, which is the same discipline the agent is held to.
    """

    completeness_floor: float = 0.80
    # This plant has one combiner sitting persistently ~12% below an even
    # share in untouched data. A threshold under that fires on every window
    # ever measured, which is how the first version of this engine came to
    # call all 86 cases a string fault. Set from the tuning split only.
    # 0.22, not the 0.18 that maximises overall accuracy on the tuning split.
    # 0.18 buys 0.012 of macro-F1 and doubles the false-alarm rate, and the
    # false-alarm rate is the number the field cares about — crews dispatched
    # to healthy plant. It also sits uncomfortably close to the 12% standing
    # anomaly, so a slightly different window would push it over.
    string_share_deficit: float = 0.22
    clearsky_shortfall: float = 0.12
    clearsky_drift_per_day: float = -0.004
    ceiling_share: float = 0.02
    deficit_open: float = 0.05
    deficit_change: float = 0.04
    shape_spread: float = 0.25
    dim_weather_ratio: float = 0.85
    soiling_slope_per_day: float = -0.0015
    window: dict[str, str] = field(default_factory=dict)

    def diagnose(self, ctx: ToolContext) -> RulesVerdict:
        checks: list[str] = []
        failed: list[str] = []
        measured: dict[str, ToolResult] = {}
        evidence: dict[str, float] = {}

        for name in CHECK_SEQUENCE:
            checks.append(name)
            try:
                result = run_tool(name, ctx, dict(self.window))
            except ToolError:
                # A measurement that could not be taken is not a measurement of
                # zero. The engine records the failure and reasons without it,
                # exactly as the agent's loop does.
                failed.append(name)
                continue
            measured[name] = result
            evidence.update({f"{name}.{k}": v for k, v in result.values.items()})

        def value(tool: str, key: str, default: float = 0.0) -> float:
            result = measured.get(tool)
            return result.values.get(key, default) if result else default

        def verdict(category: str, cause: str, settled: bool = True) -> RulesVerdict:
            return RulesVerdict(
                category,
                cause,
                settled,
                tuple(checks),
                evidence,
                failed_checks=tuple(failed),
            )

        # ---------------- fixed decision order -----------------------------
        # 1. Is the data even usable? Any deficit computed across a gap is a
        #    measurement of the logger, so this has to come first.
        #
        # Note which number is used. Row coverage stays at 100% when a logger
        # blanks the power channel but keeps writing lines, which is what a
        # real gap looks like in this dataset. The share of *well-lit* intervals
        # that carried power is the one that moves.
        if (
            value("profile_data_quality", "lit_interval_completeness", 1.0)
            < self.completeness_floor
        ):
            return verdict("not_the_plant", "telemetry_gap")
        if value("profile_data_quality", "coverage", 1.0) < self.completeness_floor:
            return verdict("not_the_plant", "telemetry_gap")
        # A performance ratio that could not be computed at all is itself a
        # statement about the data. Reading its absence as "completeness 1.0"
        # would turn the worst data problem into a clean bill of health.
        if "compute_temp_corrected_pr" in failed:
            return verdict("not_the_plant", "telemetry_gap")
        if value("compute_temp_corrected_pr", "data_completeness", 1.0) < (
            self.completeness_floor
        ):
            return verdict("not_the_plant", "telemetry_gap")

        # 2. Is the sensor lying? A sensor reading low makes performance ratio
        #    *rise*, which a real loss cannot do — so this outranks any deficit.
        drift = value("check_clearsky_consistency", "ratio_change_per_day")
        shortfall = value("check_clearsky_consistency", "shortfall_vs_clearsky")
        clear_days = min(
            value("check_clearsky_consistency", "clear_days_first_half"),
            value("check_clearsky_consistency", "clear_days_second_half"),
        )
        if clear_days >= 3 and (
            drift < self.clearsky_drift_per_day or shortfall > self.clearsky_shortfall
        ):
            return verdict("not_the_plant", "sensor_drift")

        # 3. A flat ceiling. The engine cannot separate clipping from
        #    curtailment and has no way to express that, so it commits — which
        #    is exactly the failure the unresolvable cases are built to expose.
        if value("check_ac_ceiling", "flat_ceiling_share") > self.ceiling_share:
            return verdict("by_design", "clipping")

        # 4. Is the loss confined to part of the array?
        lowest = value("per_mppt_current_balance", "lowest_string_share")
        even = value("per_mppt_current_balance", "even_share", 1.0)
        if even > 0 and (1.0 - lowest / even) > self.string_share_deficit:
            # Confined to a band of hours means a shadow rather than a fuse.
            if value("time_of_day_profile", "shape_spread_across_day") > (
                self.shape_spread
            ):
                return verdict("fault", "shading")
            return verdict("fault", "string_outage")

        # 5. Was the sunlight itself unusual?
        if value("weather_context", "relative_to_norm", 1.0) < self.dim_weather_ratio:
            return verdict("not_the_plant", "weather")

        # 6. A real, uniform, worsening loss.
        deficit = value("compute_expected_output", "deficit_fraction")
        change = value("compute_expected_output", "deficit_change_across_window")
        slope = value("daily_performance_trend", "slope_per_day")
        if (
            slope < self.soiling_slope_per_day
            or change > self.deficit_change
            or deficit > self.deficit_open
        ):
            return verdict("recoverable", "soiling")

        # 7. Nothing fired. On this plant the residue is almost always heat.
        return verdict("by_design", "seasonal_temperature_derating")
