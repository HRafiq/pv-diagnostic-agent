"""The second nine measurements, completing the ~18-tool set of handoff §3.4.

The first nine (`measurements.py`) answer "is something wrong, and where". These
answer the questions that separate one cause from another once it is: is this
new, is the instrument lying, is the loss upstream or downstream of the
inverter, and does it recover on its own.

The same three rules hold: no classification, no LLM, thresholds passed in
explicitly. One rule is worth repeating because it shaped what is *not* here —
there is no `compare_to_same_period_last_year`. That is the named resolving
measurement for the clipping/curtailment pair, and a tool set that contains the
answer to its own unresolvable case is not testing abstention, it is testing
whether the agent found the right function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field

from src.data.ingest import string_columns
from src.physics.clearsky import clearsky_poa
from src.physics.performance import (
    cell_temperature_faiman,
    compute_pr,
    pr_timeseries,
)
from src.tools.base import (
    ToolContext,
    ToolError,
    ToolResult,
    ToolSpec,
    WindowArgs,
    slice_window,
)
from src.tools.measurements import _significance

__all__ = ["DIAGNOSTIC_SPECS"]


# ---------------------------------------------------------------------------
# 10. Is this new?
# ---------------------------------------------------------------------------
class BaselineArgs(WindowArgs):
    baseline_days: int = Field(
        default=30,
        ge=3,
        le=400,
        description="Days of record immediately before the window to compare against.",
    )


def compare_to_trailing_baseline(ctx: ToolContext, args: BaselineArgs) -> ToolResult:
    """This window's performance against the stretch immediately before it.

    The single most useful question about any deficit: has it always been like
    this? A plant that has run at 0.78 for two years is not developing a fault,
    whatever its absolute number looks like against a generic model.
    """
    window = slice_window(ctx.frame, args.start, args.end)
    index = pd.DatetimeIndex(ctx.frame.index)
    window_start = pd.DatetimeIndex(window.index).min()
    baseline_start = window_start - pd.Timedelta(days=args.baseline_days)
    prior = ctx.frame.loc[(index >= baseline_start) & (index < window_start)]

    if len(prior) < 96:
        raise ToolError(
            f"only {len(prior)} readings exist in the {args.baseline_days} days "
            "before this window, which is not enough to compare against"
        )

    now = compute_pr(window, ctx.dc_capacity_kw, ctx.gamma_pdc)
    before = compute_pr(prior, ctx.dc_capacity_kw, ctx.gamma_pdc)
    if now.samples_used == 0 or before.samples_used == 0:
        raise ToolError("one of the two periods has no scoreable daylight intervals")

    change = now.pr_temperature_corrected - before.pr_temperature_corrected

    # Judge the change against how much the baseline period itself wandered
    # day to day. A drop of 0.03 means one thing on a steady plant and nothing
    # at all on a plant that swings 0.05 between consecutive days.
    daily_before = pr_timeseries(prior, ctx.dc_capacity_kw, ctx.gamma_pdc, freq="1D")[
        "pr_temperature_corrected"
    ].dropna()
    scatter = float(daily_before.std()) if len(daily_before) > 2 else 0.0

    return ToolResult(
        tool="compare_to_trailing_baseline",
        summary=(
            f"Temperature-corrected performance ratio is "
            f"{now.pr_temperature_corrected:.3f} in this window against "
            f"{before.pr_temperature_corrected:.3f} over the preceding "
            f"{args.baseline_days} days — a change of {change:+.3f}, where the "
            f"baseline period's own day-to-day spread was {scatter:.3f}."
        ),
        values={
            "pr_corrected_window": now.pr_temperature_corrected,
            "pr_corrected_baseline": before.pr_temperature_corrected,
            "change": change,
            "pr_raw_window": now.pr,
            "pr_raw_baseline": before.pr,
            "baseline_day_to_day_spread": scatter,
            "change_over_spread": _significance(change, scatter),
            "baseline_days_requested": float(args.baseline_days),
            "baseline_intervals_used": float(before.samples_used),
            "baseline_completeness": before.data_completeness,
        },
        labels={"baseline_start": str(baseline_start.date())},
        samples_used=before.samples_used + now.samples_used,
        caveats=[
            "the baseline is the immediately preceding stretch, so a fault that "
            "began before the window is already in it and this will read as no "
            "change",
            "a seasonal shift between the two periods survives the temperature "
            "correction only partly; a 30-day gap in spring or autumn moves this "
            "number on a healthy plant",
        ],
    )


# ---------------------------------------------------------------------------
# 11. Upstream or downstream of the inverter?
# ---------------------------------------------------------------------------
def dc_to_ac_conversion(ctx: ToolContext, args: WindowArgs) -> ToolResult:
    """Measured AC against measured DC — where in the chain the loss sits.

    A loss on the array side (dust, a failed string, a shadow) takes DC and AC
    down together and leaves this ratio alone. A loss between them (a failing
    inverter, a tripped phase) shows up here and nowhere else.
    """
    window = slice_window(ctx.frame, args.start, args.end)
    if "dc_power_kw" not in window or "ac_power_kw" not in window:
        raise ToolError("this system does not expose both DC and AC power")

    dc = pd.to_numeric(window["dc_power_kw"], errors="coerce")
    ac = pd.to_numeric(window["ac_power_kw"], errors="coerce")
    # Only where the inverter is genuinely working: at 2% of nameplate the
    # ratio is dominated by the inverter's own standby draw and says nothing.
    lit = (dc >= 0.05 * ctx.dc_capacity_kw) & ac.notna()
    if not lit.any():
        raise ToolError("no interval in this window has meaningful DC power")

    ratio = (ac[lit] / dc[lit]).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        raise ToolError("DC power is present but never positive enough to divide by")

    daily = ratio.groupby(pd.DatetimeIndex(ratio.index).normalize()).median()
    half = len(daily) // 2

    return ToolResult(
        tool="dc_to_ac_conversion",
        summary=(
            f"The inverter converts DC to AC at a median {ratio.median():.3f} "
            f"over {len(ratio)} intervals, ranging {ratio.quantile(0.05):.3f} to "
            f"{ratio.quantile(0.95):.3f} across the window."
        ),
        values={
            "median_conversion": float(ratio.median()),
            "conversion_5th_percentile": float(ratio.quantile(0.05)),
            "conversion_95th_percentile": float(ratio.quantile(0.95)),
            "conversion_first_half": (
                float(daily.iloc[:half].median()) if half else float(daily.median())
            ),
            "conversion_second_half": (
                float(daily.iloc[half:].median())
                if len(daily) - half
                else float(daily.median())
            ),
            "intervals_scored": float(len(ratio)),
            "days_compared": float(len(daily)),
        },
        series={"daily_conversion": [float(v) for v in daily.to_numpy()]},
        series_index=[str(t) for t in daily.index],
        samples_used=len(ratio),
        caveats=[
            "a healthy inverter sits near 0.96-0.98 at load and lower at part "
            "load, so the low end of the range is normal rather than a fault",
            "a power limit applied at the inverter pulls DC and AC down "
            "together and leaves this ratio unchanged; it does not show up here",
        ],
    )


# ---------------------------------------------------------------------------
# 12. Which strings, and in what shape?
# ---------------------------------------------------------------------------
class ProfileArgs(WindowArgs):
    min_poa_wm2: float = Field(default=300.0, gt=0.0)


def _profiles_over(block: pd.DataFrame, currents: list[str]) -> dict[str, float]:
    """Each string's level against the array median, as a regression slope."""
    reference = block.median(axis=1)
    if float(reference.std()) <= 0:
        return {}
    out: dict[str, float] = {}
    for column in currents:
        series = block[column]
        if float(series.std()) <= 0:
            out[column] = 0.0
            continue
        out[column] = float(
            np.polyfit(reference.to_numpy(float), series.to_numpy(float), 1)[0]
        )
    return out


def _reference_levels(
    ctx: ToolContext, args: ProfileArgs, currents: list[str], window: pd.DataFrame
) -> dict[str, float]:
    """The same levels over the record *before* this window.

    Without it, a string that has always carried less than its neighbours reads
    as a deficit in every window ever scored. That is not hypothetical: G-004
    committed to `string_outage` on a soiling case, and the sentence it quoted
    was "String 7 is running at only 0.791 of the array's median level" — a
    permanent characteristic of this array, reported as though it had just
    happened.

    `per_mppt_current_balance` was given the same treatment first (DECISION
    0079); this tool has the identical hole and is the one the wrong answer
    actually rested on. A fix applied to one of two tools that make the same
    comparison is half a fix.
    """
    start = window.index.min()
    history = ctx.frame.loc[ctx.frame.index < start]
    if len(history) < len(window):
        return {}
    lit = pd.to_numeric(history.get("poa_wm2"), errors="coerce") >= args.min_poa_wm2
    if not lit.any():
        return {}
    block = history.loc[lit, currents].apply(pd.to_numeric, errors="coerce")
    if len(block) < 100:
        return {}
    return _profiles_over(block, currents)


def compare_string_profiles(ctx: ToolContext, args: ProfileArgs) -> ToolResult:
    """Each string's daily shape against the array's, not just its level.

    `per_mppt_current_balance` answers "is one string low". This answers
    "is one string a different *shape*", which is a different fault. A blown
    fuse scales a string's whole day by a constant and leaves its shape
    matching everyone else's; a shadow bends the shape only where the shadow
    falls, so the correlation drops while the daily total may barely move.
    """
    window = slice_window(ctx.frame, args.start, args.end)
    currents, _ = string_columns(window)
    if len(currents) < 2:
        raise ToolError(
            "fewer than two per-string channels, so shapes cannot be compared"
        )
    if "poa_wm2" not in window:
        raise ToolError("no irradiance channel to select well-lit intervals with")

    lit = pd.to_numeric(window["poa_wm2"], errors="coerce") >= args.min_poa_wm2
    block = window.loc[lit, currents].apply(pd.to_numeric, errors="coerce")
    if len(block) < 20:
        raise ToolError(
            f"only {len(block)} well-lit intervals; a shape comparison needs "
            "at least twenty"
        )

    reference = block.median(axis=1)
    if float(reference.std()) <= 0:
        raise ToolError(
            "the array's current never varies, so shapes cannot be compared"
        )

    correlations = {}
    slopes = {}
    for column in currents:
        series = block[column]
        if float(series.std()) <= 0:
            correlations[column] = 0.0
            slopes[column] = 0.0
            continue
        correlations[column] = float(series.corr(reference))
        slopes[column] = float(
            np.polyfit(reference.to_numpy(float), series.to_numpy(float), 1)[0]
        )

    worst = min(correlations, key=lambda c: correlations[c])
    worst_index = int(worst.rsplit("_", 1)[-1])

    values = {
        f"shape_match_string_{c.rsplit('_', 1)[-1]}": v for c, v in correlations.items()
    }
    values.update(
        {f"level_string_{c.rsplit('_', 1)[-1]}": v for c, v in slopes.items()}
    )
    values.update(
        {
            "lowest_shape_match": float(min(correlations.values())),
            "lowest_shape_match_string": float(worst_index),
            "highest_shape_match": float(max(correlations.values())),
            "shape_match_spread": float(
                max(correlations.values()) - min(correlations.values())
            ),
            "intervals_scored": float(len(block)),
        }
    )

    baseline = _reference_levels(ctx, args, currents, window)
    level_note = ""
    if baseline:
        moves = {c: slopes[c] - baseline[c] for c in currents if c in baseline}
        fallen = min(moves, key=lambda c: moves[c])
        values.update(
            {
                f"baseline_level_string_{str(c).rsplit('_', 1)[-1]}": v
                for c, v in baseline.items()
            }
        )
        values.update(
            {
                f"level_change_string_{str(c).rsplit('_', 1)[-1]}": v
                for c, v in moves.items()
            }
        )
        values.update(
            {
                "largest_level_fall": float(-min(moves.values())),
                "largest_level_fall_string": float(int(str(fallen).rsplit("_", 1)[-1])),
                "weakest_shape_baseline_level": float(baseline.get(worst, 0.0)),
                "weakest_shape_level_change": float(moves.get(worst, 0.0)),
            }
        )
        level_note = (
            f" Against its own history it carried {baseline[worst]:.3f}, so its "
            f"level has moved {moves[worst]:+.3f}."
        )

    return ToolResult(
        tool="compare_string_profiles",
        summary=(
            f"String {worst_index}'s daily shape matches the array's at "
            f"{min(correlations.values()):.3f}, the weakest of "
            f"{len(currents)}; the best is "
            f"{max(correlations.values()):.3f}. Its level relative to the array "
            f"median is {slopes[worst]:.3f}." + level_note
        ),
        values=values,
        labels={"weakest_shape_channel": worst},
        samples_used=len(block),
        caveats=[
            "a string scaled down by a constant keeps a shape match near 1.0; "
            "level and shape are different measurements and a fault may move "
            "only one of them",
            *(
                [
                    "level is measured against the array's other strings, not "
                    "against nameplate: a string can sit below its neighbours "
                    "for the whole record without anything having failed. The "
                    "change from its own history is what separates a standing "
                    "difference from a new one"
                ]
                if baseline
                else [
                    "no history before this window, so a string sitting below "
                    "its neighbours cannot be told apart from one that has "
                    "been low since the record began"
                ]
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 13. Is the module temperature sensor telling the truth?
# ---------------------------------------------------------------------------
def check_module_temperature_sensor(ctx: ToolContext, args: WindowArgs) -> ToolResult:
    """Measured module temperature against what the weather implies.

    This one sits underneath every other measurement in the set. Temperature-
    corrected performance ratio divides by a factor built from this channel, so
    a sensor reading 10 K high manufactures a 4% deficit that is not there — and
    it manufactures it in the *corrected* number, the one that is supposed to be
    trustworthy.
    """
    window = slice_window(ctx.frame, args.start, args.end)
    if "temp_module_c" not in window or not window["temp_module_c"].notna().any():
        raise ToolError(
            "this system has no measured module temperature, so the corrected "
            "performance ratio already rests on the modelled estimate"
        )
    if "temp_ambient_c" not in window:
        raise ToolError(
            "no ambient temperature to model an expected module temperature"
        )

    measured = pd.to_numeric(window["temp_module_c"], errors="coerce")
    modelled = cell_temperature_faiman(
        pd.to_numeric(window["poa_wm2"], errors="coerce"),
        pd.to_numeric(window["temp_ambient_c"], errors="coerce"),
        window.get("wind_speed_ms"),
    )
    lit = pd.to_numeric(window["poa_wm2"], errors="coerce") >= 300.0
    pair = pd.DataFrame({"measured": measured, "modelled": modelled})[lit].dropna()
    if len(pair) < 20:
        raise ToolError("too few well-lit intervals to check the temperature sensor")

    residual = pair["measured"] - pair["modelled"]
    ambient = pd.to_numeric(window["temp_ambient_c"], errors="coerce")[lit].dropna()
    rise = pair["measured"] - ambient.reindex(pair.index)

    return ToolResult(
        tool="check_module_temperature_sensor",
        summary=(
            f"Measured module temperature runs {residual.median():+.1f} °C "
            f"against the weather-based estimate over {len(pair)} well-lit "
            f"intervals, sitting a median {rise.median():.1f} °C above ambient "
            f"and peaking at {pair['measured'].max():.1f} °C."
        ),
        values={
            "median_offset_from_model_c": float(residual.median()),
            "offset_5th_percentile_c": float(residual.quantile(0.05)),
            "offset_95th_percentile_c": float(residual.quantile(0.95)),
            "median_rise_above_ambient_c": float(rise.median()),
            "max_module_temp_c": float(pair["measured"].max()),
            "min_module_temp_c": float(pair["measured"].min()),
            "distinct_values": float(pair["measured"].round(2).nunique()),
            "intervals_scored": float(len(pair)),
        },
        samples_used=len(pair),
        caveats=[
            "a module in full sun sits 25-30 K above ambient, so a small offset "
            "against the model is normal and reflects mounting rather than a "
            "faulty sensor",
            "a very low count of distinct values means the channel is stuck",
        ],
    )


# ---------------------------------------------------------------------------
# 14. Was the weather itself unusual?
# ---------------------------------------------------------------------------
def weather_context(ctx: ToolContext, args: WindowArgs) -> ToolResult:
    """This window's sunlight against the same calendar weeks in the record.

    The first look-alike on the list and the cheapest to dismiss. Energy is down
    because the sun was down; performance ratio already removes that, but a
    plant manager asking "where did my week go" needs the fact stated, and
    "irradiance was 31% below the same weeks in other years" is the statement.
    """
    window = slice_window(ctx.frame, args.start, args.end)
    if "poa_wm2" not in window:
        raise ToolError("no irradiance channel")

    poa_all = pd.to_numeric(ctx.frame["poa_wm2"], errors="coerce")
    index = pd.DatetimeIndex(ctx.frame.index)
    window_index = pd.DatetimeIndex(window.index)
    interval_h = (
        float(pd.Series(index).diff().dropna().mode().iloc[0].total_seconds()) / 3600.0
    )

    days = window_index.dayofyear
    lo, hi = int(days.min()), int(days.max())
    # Same calendar span, every year in the record, this one included. The
    # comparison has to be seasonal: a January fortnight is legitimately half a
    # July one and comparing them measures the Earth's tilt.
    same_season = (index.dayofyear >= lo) & (index.dayofyear <= hi)
    other_years = same_season & ~index.isin(window_index)

    daily = (
        poa_all[same_season]
        .groupby(pd.DatetimeIndex(poa_all[same_season].index).normalize())
        .sum()
        * interval_h
        / 1000.0
    )
    window_daily = (
        poa_all[index.isin(window_index)]
        .groupby(pd.DatetimeIndex(poa_all[index.isin(window_index)].index).normalize())
        .sum()
        * interval_h
        / 1000.0
    )
    if window_daily.empty:
        raise ToolError("no daily insolation could be computed for this window")

    this_mean = float(window_daily.mean())
    if not other_years.any():
        seasonal_mean = this_mean
        caveat = (
            "the record covers this calendar span only once, so there is nothing "
            "to compare the weather against — this figure is the window itself"
        )
    else:
        other = (
            poa_all[other_years]
            .groupby(pd.DatetimeIndex(poa_all[other_years].index).normalize())
            .sum()
            * interval_h
            / 1000.0
        )
        seasonal_mean = float(other.mean())
        caveat = (
            f"compared against {len(other)} days from the same calendar span in "
            "other years, which is a thin sample for a claim about weather"
        )

    # A norm of zero cannot be divided by, and no guard on the ledger helps:
    # the summary string does the same division. Physically this is a record
    # whose irradiance channel reads zero throughout — a dead pyranometer, not
    # weather — and there is no honest statement to make about the weather.
    if seasonal_mean <= 0.0:
        raise ToolError(
            "the seasonal insolation norm for this span is zero, so this "
            "window cannot be expressed relative to it; the irradiance channel "
            "is likely dead rather than the sky dark"
        )

    dim_days = int((window_daily < 0.6 * seasonal_mean).sum())

    # `Series.std()` uses ddof=1, so one day yields NaN rather than raising.
    # The day-to-day spread is genuinely undefined on a single day — there is
    # no day-over-day to measure — but every other figure here is still real,
    # so the key is omitted rather than the whole measurement discarded. An
    # absent key reads as "not measured"; a zero would read as "perfectly
    # steady", which is the opposite of what one day tells you.
    spread = float(window_daily.std()) if len(window_daily) > 1 else None
    if spread is None:
        caveats_extra = [
            f"the window holds {len(window_daily)} day(s), so there is no "
            "day-to-day spread to report"
        ]
    else:
        caveats_extra = []

    return ToolResult(
        tool="weather_context",
        summary=(
            f"This window received {this_mean:.2f} kWh/m² per day against "
            f"{seasonal_mean:.2f} for the same calendar span elsewhere in the "
            f"record — {(this_mean / seasonal_mean - 1) * 100:+.1f}% — with "
            f"{dim_days} of {len(window_daily)} days below 60% of that norm."
        ),
        values={
            "mean_daily_insolation_kwh_m2": this_mean,
            "seasonal_norm_kwh_m2": seasonal_mean,
            "relative_to_norm": this_mean / seasonal_mean if seasonal_mean else 1.0,
            "percent_vs_norm": (
                (this_mean / seasonal_mean - 1.0) * 100.0 if seasonal_mean else 0.0
            ),
            "dim_days": float(dim_days),
            "days_in_window": float(len(window_daily)),
            "brightest_day_kwh_m2": float(window_daily.max()),
            "dimmest_day_kwh_m2": float(window_daily.min()),
            **({"day_to_day_spread": spread} if spread is not None else {}),
            "dim_day_threshold_pct_of_norm": 60.0,
            "seasonal_days_compared": float(max(len(daily) - len(window_daily), 0)),
        },
        series={"daily_insolation_kwh_m2": [float(v) for v in window_daily.to_numpy()]},
        series_index=[str(t) for t in window_daily.index],
        samples_used=len(window),
        caveats=[
            caveat,
            *caveats_extra,
            "dim weather explains low energy, never a low performance ratio; if "
            "both are down, the weather is not the whole story",
        ],
    )


# ---------------------------------------------------------------------------
# 15. Stuck channels
# ---------------------------------------------------------------------------
class StuckArgs(WindowArgs):
    min_run: int = Field(
        default=8,
        ge=3,
        description="Consecutive identical readings counted as stuck.",
    )


def detect_stuck_channels(ctx: ToolContext, args: StuckArgs) -> ToolResult:
    """Channels repeating a value for longer than physics allows.

    A stuck sensor is the quietest failure in the set: it reports a plausible
    number forever, so nothing looks missing and every downstream figure is
    wrong in a way that no completeness check catches.
    """
    window = slice_window(ctx.frame, args.start, args.end)
    candidates = [
        c
        for c in ("poa_wm2", "temp_module_c", "temp_ambient_c", "wind_speed_ms")
        if c in window
    ]
    candidates += [c for c in window.columns if c.startswith("string_current_a_")]
    if not candidates:
        raise ToolError("no channels to check")

    # Night is legitimately flat on every irradiance and power channel, so
    # scoring it would report a stuck sensor on every clear night in the record.
    lit = pd.to_numeric(window.get("poa_wm2", pd.Series(dtype=float)), errors="coerce")
    daytime = (lit >= 50.0) if len(lit) else pd.Series(True, index=window.index)

    values: dict[str, float] = {}
    worst_channel, worst_share = "", 0.0
    for column in candidates:
        series = pd.to_numeric(window[column], errors="coerce")[daytime].dropna()
        if len(series) < args.min_run:
            continue
        same = series.diff().abs().astype(float) < 1e-9
        run_id = (~same).cumsum()
        length = same.groupby(run_id).transform("sum")
        stuck = same & (length >= args.min_run - 1)
        share = float(stuck.sum()) / float(len(series))
        values[f"stuck_share_{column}"] = share
        if share > worst_share:
            worst_channel, worst_share = column, share

    if not values:
        raise ToolError("no channel has enough daylight readings to check")

    values["worst_stuck_share"] = worst_share
    values["channels_checked"] = float(len(values) - 1)
    values["run_length_required"] = float(args.min_run)

    return ToolResult(
        tool="detect_stuck_channels",
        summary=(
            f"The most repetitive channel is {worst_channel or 'none'}, holding "
            f"an identical value for runs of {args.min_run} or more daylight "
            f"readings across {worst_share * 100:.1f}% of them."
            if worst_share > 0
            else "No channel repeats a value for long enough to look stuck."
        ),
        values=values,
        labels={"worst_channel": worst_channel} if worst_channel else {},
        samples_used=int(daytime.sum()),
        caveats=[
            "coarse logger resolution produces repeated values on a working "
            "sensor, especially on wind speed and at low irradiance"
        ],
    )


# ---------------------------------------------------------------------------
# 16. Night-time zero offset
# ---------------------------------------------------------------------------
def check_night_offset(ctx: ToolContext, args: WindowArgs) -> ToolResult:
    """What the sensors read when the answer is known to be zero.

    Every channel's zero is checkable once a night, for free, against physics
    rather than against a calibration certificate. A pyranometer reading
    +8 W/m² at midnight is reading +8 too high all day, which biases the
    performance-ratio denominator and drags the whole plant's number down.
    """
    window = slice_window(ctx.frame, args.start, args.end)
    if "poa_wm2" not in window:
        raise ToolError("no irradiance channel")

    poa = pd.to_numeric(window["poa_wm2"], errors="coerce")
    solar = clearsky_poa(
        pd.DatetimeIndex(window.index),
        ctx.latitude,
        ctx.longitude,
        ctx.altitude_m,
        ctx.tilt_deg,
        ctx.azimuth_deg,
        albedo=ctx.albedo,
    )
    # Deep night by solar geometry, not by the sensor's own reading — using the
    # channel to decide when the channel should read zero is circular.
    night = solar.solar_zenith > 100.0
    if not night.any():
        raise ToolError(
            "this window contains no interval with the sun well below the horizon"
        )

    night_poa = poa[night].dropna()
    values = {
        "night_intervals": float(len(night_poa)),
        "median_night_irradiance_wm2": float(night_poa.median())
        if len(night_poa)
        else 0.0,
        "max_night_irradiance_wm2": float(night_poa.max()) if len(night_poa) else 0.0,
    }
    summary_bits = [
        f"At night the irradiance sensor reads a median "
        f"{values['median_night_irradiance_wm2']:.2f} W/m² over "
        f"{int(values['night_intervals'])} intervals"
    ]

    if "ac_power_kw" in window:
        night_ac = pd.to_numeric(window["ac_power_kw"], errors="coerce")[night].dropna()
        if len(night_ac):
            values["median_night_ac_kw"] = float(night_ac.median())
            values["max_night_ac_kw"] = float(night_ac.max())
            values["night_generation_intervals"] = float(
                (night_ac > 0.01 * ctx.dc_capacity_kw).sum()
            )
            summary_bits.append(
                f"and the meter reads a median {values['median_night_ac_kw']:.3f} kW"
            )

    return ToolResult(
        tool="check_night_offset",
        summary=". ".join([" ".join(summary_bits)]) + ".",
        values=values,
        samples_used=int(night.sum()),
        caveats=[
            "a small negative night reading is normal on a thermopile "
            "pyranometer, which radiates to a cold sky; a positive one is not",
            "night power above zero is a meter or logger problem, never generation",
        ],
    )


# ---------------------------------------------------------------------------
# 17. Does it recover on its own?
# ---------------------------------------------------------------------------
class SoilingArgs(WindowArgs):
    wash_quantile: float = Field(
        default=0.35,
        gt=0.0,
        lt=1.0,
        description=(
            "A day this far below the window's median insolation is treated as "
            "a candidate rain day. There is no rainfall channel in this "
            "dataset, so this is a proxy and is reported as one."
        ),
    )


def soiling_recovery_pattern(ctx: ToolContext, args: SoilingArgs) -> ToolResult:
    """Whether performance falls between wet days and jumps after them.

    The measurement that separates dirt from damage, and it is a *pattern*, not
    a level. Dust accumulates while it is dry and washes off when it rains, so
    the series is a sawtooth. A failed component never recovers on its own, so
    its series is a step that stays down. Both can produce the same monthly
    average.
    """
    window = slice_window(ctx.frame, args.start, args.end)
    daily = pr_timeseries(window, ctx.dc_capacity_kw, ctx.gamma_pdc, freq="1D")
    series = daily["pr_temperature_corrected"].dropna().astype(float)
    insolation = daily["insolation_kwh_m2"].reindex(series.index)
    if len(series) < 10:
        raise ToolError(
            f"only {len(series)} scored days; a recovery pattern needs at least ten"
        )

    insolation = pd.to_numeric(insolation, errors="coerce").astype(float)
    threshold = float(insolation.median()) * args.wash_quantile
    wet = insolation < threshold
    dry_runs: list[float] = []
    jumps: list[float] = []
    run: list[float] = []
    wet_flags = wet.to_numpy(dtype=bool)
    for position, value in enumerate(series.to_numpy(dtype=float)):
        if bool(wet_flags[position]):
            if len(run) >= 3:
                # Slope across the dry run: dust accumulating between washes.
                x = np.arange(len(run), dtype=float)
                dry_runs.append(float(np.polyfit(x, np.array(run), 1)[0]))
            run = []
        else:
            run.append(value)
    if len(run) >= 3:
        x = np.arange(len(run), dtype=float)
        dry_runs.append(float(np.polyfit(x, np.array(run), 1)[0]))

    positions = np.flatnonzero(wet.to_numpy())
    for position in positions:
        before = series.iloc[max(0, position - 2) : position]
        after = series.iloc[position + 1 : position + 3]
        if len(before) and len(after):
            jumps.append(float(after.mean() - before.mean()))

    return ToolResult(
        tool="soiling_recovery_pattern",
        summary=(
            f"Across {len(dry_runs)} dry stretches, performance moves a mean "
            f"{np.mean(dry_runs) if dry_runs else 0.0:+.5f} per day, and after "
            f"{len(jumps)} low-insolation days it changes by a mean "
            f"{np.mean(jumps) if jumps else 0.0:+.4f}."
        ),
        values={
            "dry_stretches": float(len(dry_runs)),
            "mean_dry_slope_per_day": float(np.mean(dry_runs)) if dry_runs else 0.0,
            "steepest_dry_slope_per_day": float(min(dry_runs)) if dry_runs else 0.0,
            "candidate_wash_days": float(len(jumps)),
            "mean_change_after_a_wash": float(np.mean(jumps)) if jumps else 0.0,
            "largest_change_after_a_wash": float(max(jumps)) if jumps else 0.0,
            "days_scored": float(len(series)),
            "wash_day_insolation_threshold_kwh_m2": threshold,
        },
        samples_used=len(series),
        caveats=[
            "there is no rainfall channel in this dataset, so a wet day is "
            "inferred from a day of unusually low insolation — a proxy, and one "
            "that also catches a merely overcast day",
            "a decline that never recovers after a wet day is not soiling, "
            "whatever its slope",
        ],
    )


# ---------------------------------------------------------------------------
# 18. Which string changed, and when?
# ---------------------------------------------------------------------------
class StringOnsetArgs(WindowArgs):
    min_poa_wm2: float = Field(default=400.0, gt=0.0)


def string_onset_scan(ctx: ToolContext, args: StringOnsetArgs) -> ToolResult:
    """A changepoint scan run per string rather than on the plant total.

    A single string is one seventh of this array, so a fault that takes it out
    entirely moves the plant total by 14% — visible, but blended with weather.
    Run on the string's own *share*, where the weather has already divided out,
    the same event is unmistakable and dated.
    """
    window = slice_window(ctx.frame, args.start, args.end)
    currents, _ = string_columns(window)
    if len(currents) < 2:
        raise ToolError("fewer than two per-string channels")
    if "poa_wm2" not in window:
        raise ToolError("no irradiance channel")

    lit = pd.to_numeric(window["poa_wm2"], errors="coerce") >= args.min_poa_wm2
    block = window.loc[lit, currents].apply(pd.to_numeric, errors="coerce")
    totals = block.sum(axis=1)
    usable = totals > max(0.2 * float(totals.median()), 1e-6)
    if not usable.any():
        raise ToolError("no well-lit interval carries measurable string current")

    shares = block[usable].div(totals[usable], axis=0)
    daily = shares.groupby(pd.DatetimeIndex(shares.index).normalize()).mean().dropna()
    if len(daily) < 6:
        raise ToolError(
            f"only {len(daily)} scored days; a per-string onset needs at least six"
        )

    values: dict[str, float] = {}
    labels: dict[str, str] = {}
    best_column = currents[0]
    best_significance, best_change, best_day = -1.0, 0.0, daily.index[0]

    for column in currents:
        series = daily[column].astype(float)
        n = len(series)
        k, gap = 0, 0.0
        for split in range(2, n - 1):
            candidate = abs(
                float(series.iloc[:split].mean()) - float(series.iloc[split:].mean())
            )
            if candidate > gap:
                k, gap = split, candidate
        if k == 0:
            # Nothing beat a split at position zero, which means the series is
            # flat — including the flat-at-zero case of a string that was
            # already dead when the window opened. There is no "before", so the
            # change is zero rather than a mean of no days.
            change = 0.0
        else:
            change = float(series.iloc[k:].mean()) - float(series.iloc[:k].mean())
        scatter = float(series.diff().abs().median())
        significance = _significance(change, scatter)
        suffix = column.rsplit("_", 1)[-1]
        values[f"share_change_string_{suffix}"] = change
        values[f"significance_string_{suffix}"] = significance
        # Rank by significance, then by the size of the change. On a very
        # steady array every string saturates the significance ceiling and the
        # comparison becomes a tie; without the second key the answer would be
        # whichever string happens to be first in the list.
        if (significance, abs(change)) > (best_significance, abs(best_change)):
            best_column, best_significance = column, significance
            best_change, best_day = change, series.index[min(k, n - 1)]

    index = int(best_column.rsplit("_", 1)[-1]) if best_column else 0
    values.update(
        {
            "most_changed_string": float(index),
            "its_share_change": best_change,
            "its_change_over_scatter": best_significance,
            "days_scored": float(len(daily)),
            "strings_compared": float(len(currents)),
        }
    )
    labels["onset_date"] = str(pd.Timestamp(best_day).date())

    return ToolResult(
        tool="string_onset_scan",
        summary=(
            f"String {index}'s share of array current changes by "
            f"{best_change:+.4f} around {pd.Timestamp(best_day).date()}, "
            f"{best_significance:.1f} times its own day-to-day scatter — the "
            f"largest of {len(currents)} strings over {len(daily)} days."
        ),
        values=values,
        labels=labels,
        samples_used=int(usable.sum()),
        caveats=[
            "this always returns a largest change, even on an array where "
            "nothing happened; read the change against the scatter",
            "shares sum to one, so one string failing shifts every other "
            "string's share upward and they will all report a change",
        ],
    )


DIAGNOSTIC_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="compare_to_trailing_baseline",
        description=(
            "This window's temperature-corrected performance ratio against the "
            "stretch of record immediately before it."
        ),
        question_answered="Is this new, or has the plant always run like this?",
        args_model=BaselineArgs,
        fn=compare_to_trailing_baseline,
        discriminates=(
            "real_loss",
            "long_standing_condition",
            "seasonal_temperature_derating",
        ),
    ),
    ToolSpec(
        name="dc_to_ac_conversion",
        description=(
            "Measured AC against measured DC power, and how it moves across the window."
        ),
        question_answered="Is the loss on the array side or at the inverter?",
        args_model=WindowArgs,
        fn=dc_to_ac_conversion,
        discriminates=("inverter_fault", "string_outage", "soiling"),
    ),
    ToolSpec(
        name="compare_string_profiles",
        description=(
            "How closely each string's daily shape tracks the array's, alongside "
            "its level."
        ),
        question_answered=(
            "Is a string merely lower, or a different shape from the rest?"
        ),
        args_model=ProfileArgs,
        fn=compare_string_profiles,
        discriminates=("shading", "string_outage"),
    ),
    ToolSpec(
        name="check_module_temperature_sensor",
        description=(
            "Measured module temperature against the weather-based estimate, "
            "its rise above ambient, and how many distinct values it reports."
        ),
        question_answered=("Can the temperature correction be trusted on this plant?"),
        args_model=WindowArgs,
        fn=check_module_temperature_sensor,
        discriminates=("seasonal_temperature_derating", "sensor_fault"),
    ),
    ToolSpec(
        name="weather_context",
        description=(
            "This window's daily insolation against the same calendar span "
            "elsewhere in the record."
        ),
        question_answered="Was the sunlight itself unusual?",
        args_model=WindowArgs,
        fn=weather_context,
        discriminates=("weather", "real_loss", "snow_or_dust_event"),
    ),
    ToolSpec(
        name="detect_stuck_channels",
        description=(
            "Channels repeating an identical value across runs of daylight readings."
        ),
        question_answered="Is a sensor frozen rather than merely wrong?",
        args_model=StuckArgs,
        fn=detect_stuck_channels,
        discriminates=("sensor_fault", "telemetry_gap"),
    ),
    ToolSpec(
        name="check_night_offset",
        description=(
            "What irradiance and power read when the sun is well below the "
            "horizon and the answer is known to be zero."
        ),
        question_answered="Is a channel biased away from zero?",
        args_model=WindowArgs,
        fn=check_night_offset,
        discriminates=("sensor_drift", "sensor_fault"),
    ),
    ToolSpec(
        name="soiling_recovery_pattern",
        description=(
            "Slope across dry stretches and the change after low-insolation "
            "days, i.e. whether performance recovers on its own."
        ),
        question_answered="Does this loss come back by itself?",
        args_model=SoilingArgs,
        fn=soiling_recovery_pattern,
        discriminates=("soiling", "string_outage", "sensor_drift"),
    ),
    ToolSpec(
        name="string_onset_scan",
        description=(
            "A changepoint scan on each string's share of array current, with "
            "the date and the size of the largest change."
        ),
        question_answered="Which string changed, and when?",
        args_model=StringOnsetArgs,
        fn=string_onset_scan,
        discriminates=("string_outage", "shading", "soiling"),
    ),
)
