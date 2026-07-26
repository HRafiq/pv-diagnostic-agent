"""The measurement tools. Nine of them in the vertical slice; ~18 by step 5.

Each one answers a narrow physical question and returns numbers. None of them
answers "what is wrong", and that separation is the whole architecture: the
agent has to notice that a per-string imbalance *plus* unchanged irradiance
*plus* a step onset means a string fault, and that the same imbalance with a
time-of-day pattern means a shadow instead. If any single function returned
"string_outage", the run would be a lookup and every trajectory would be
identical.

Everything here is deterministic, calls no model, and takes its thresholds as
explicit arguments with stated defaults (CLAUDE.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field

from src.data.ingest import string_columns
from src.data.quality import profile_quality
from src.physics.clearsky import clearsky_poa
from src.physics.modelchain import ExpectationModel
from src.physics.performance import T_STC, compute_pr, pr_timeseries
from src.tools.base import (
    ToolArgs,
    ToolContext,
    ToolError,
    ToolResult,
    ToolSpec,
    WindowArgs,
    slice_window,
)

__all__ = ["SPECS"]

# Shared with `diagnostics.py`; both are one-line helpers with no home of
# their own worth making.


def _finite(value: float) -> float:
    """Ledger values must be real numbers. NaN in a ledger is a silent lie."""
    number = float(value)
    if not np.isfinite(number):
        raise ToolError("the measurement is undefined over this window")
    return number


# A change this many times the day-to-day scatter is already decisive; beyond
# it the ratio carries no further information and an unbounded value would just
# be noise divided by noise.
_SIGNIFICANCE_CEILING = 999.0


def _significance(change: float, scatter: float) -> float:
    if abs(change) <= 0.0:
        return 0.0
    if scatter <= 0.0:
        return _SIGNIFICANCE_CEILING
    return min(abs(change) / scatter, _SIGNIFICANCE_CEILING)


def _local_hour(index: pd.DatetimeIndex, offset_hours: float) -> pd.Series:
    shifted = index + pd.Timedelta(hours=offset_hours)
    return pd.Series(shifted.hour + shifted.minute / 60.0, index=index)


# ---------------------------------------------------------------------------
# 1. Performance ratio, raw and temperature-corrected
# ---------------------------------------------------------------------------
class PrArgs(WindowArgs):
    min_poa_wm2: float = Field(
        default=200.0,
        gt=0.0,
        description=(
            "Intervals dimmer than this are excluded. Below it the PR "
            "denominator is small and the ratio becomes noise."
        ),
    )


def compute_temp_corrected_pr(ctx: ToolContext, args: PrArgs) -> ToolResult:
    window = slice_window(ctx.frame, args.start, args.end)
    result = compute_pr(
        window,
        dc_capacity_kw=ctx.dc_capacity_kw,
        gamma_pdc=ctx.gamma_pdc,
        min_poa_wm2=args.min_poa_wm2,
    )
    if result.samples_used == 0:
        raise ToolError(
            f"no interval in this window reached {args.min_poa_wm2:g} W/m^2 with "
            "a power reading present"
        )

    caveats: list[str] = []
    if result.data_completeness < 0.95:
        caveats.append(
            f"only {result.data_completeness * 100:.1f}% of daylight intervals "
            "carried a power reading, so both figures rest on partial data"
        )
    if result.samples_used < 96:
        caveats.append(
            f"{result.samples_used} intervals is a short window for a stable "
            "performance ratio"
        )

    return ToolResult(
        tool="compute_temp_corrected_pr",
        summary=(
            f"Performance ratio is {result.pr:.3f} as measured and "
            f"{result.pr_temperature_corrected:.3f} once corrected to 25 °C, "
            f"over {result.samples_used} intervals at a mean cell temperature "
            f"of {result.mean_cell_temp_c:.1f} °C."
        ),
        values={
            "pr": _finite(result.pr),
            "pr_temperature_corrected": _finite(result.pr_temperature_corrected),
            "temperature_loss_fraction": result.temperature_loss_fraction,
            "energy_kwh": result.energy_kwh,
            "insolation_kwh_m2": result.insolation_kwh_m2,
            "reference_yield_kwh": result.reference_yield_kwh,
            "mean_cell_temp_c": _finite(result.mean_cell_temp_c),
            "data_completeness": result.data_completeness,
            "samples_used": float(result.samples_used),
            "samples_missing_power": float(result.samples_missing_power),
            # Constants the summary quotes belong in the ledger too. A figure
            # the tool states but does not record is, to the grounding check,
            # indistinguishable from one the model invented.
            "reference_cell_temp_c": T_STC,
        },
        samples_used=result.samples_used,
        caveats=caveats,
    )


# ---------------------------------------------------------------------------
# 2. Measured against the expectation model
# ---------------------------------------------------------------------------
def compute_expected_output(ctx: ToolContext, args: WindowArgs) -> ToolResult:
    window = slice_window(ctx.frame, args.start, args.end)
    model = ExpectationModel(ctx.dc_capacity_kw, ctx.gamma_pdc, ctx.ac_ceiling_kw)
    expectation = model.run(window)

    interval_h = _interval_hours(window)
    lit = expectation.expected_ac_kw >= 1.0
    expected_kwh = float(expectation.expected_ac_kw[lit].sum()) * interval_h
    measured_kwh = float(expectation.measured_ac_kw[lit].fillna(0.0).sum()) * interval_h
    deficit = expectation.deficit_fraction()

    residual = expectation.residual_kw[lit]
    daily = residual.groupby(pd.DatetimeIndex(residual.index).normalize()).sum()
    worst_day = daily.idxmin() if not daily.empty else None

    # The *level* of this number is largely a property of the model, not of the
    # plant: generic 14% system losses and a 0.96 inverter efficiency leave a
    # standing offset of several per cent in either direction on any real site.
    # What is diagnostic is a *change* in the offset across the window, so
    # measure that here rather than making the agent take a second window.
    daily_expected = (
        expectation.expected_ac_kw[lit]
        .groupby(pd.DatetimeIndex(expectation.expected_ac_kw[lit].index).normalize())
        .sum()
    )
    daily_measured = (
        expectation.measured_ac_kw[lit]
        .fillna(0.0)
        .groupby(pd.DatetimeIndex(expectation.measured_ac_kw[lit].index).normalize())
        .sum()
    )
    half = len(daily_expected) // 2
    first_deficit = _half_deficit(
        daily_expected.iloc[:half], daily_measured.iloc[:half]
    )
    second_deficit = _half_deficit(
        daily_expected.iloc[half:], daily_measured.iloc[half:]
    )

    caveats: list[str] = []
    if ctx.ac_ceiling_kw is None:
        caveats.append(
            "no inverter AC rating is configured, so the model applies no "
            "ceiling and would under-predict a genuinely clipped midday"
        )
    caveats.append(
        "the expectation model is the coarse PVWatts stack with generic 14% "
        "system losses, so it carries a standing offset of several per cent on "
        "any real plant; a change in the shortfall across the window is "
        "evidence, its absolute level mostly is not"
    )
    if deficit < 0:
        caveats.append(
            f"the shortfall is negative: this plant produced {-deficit * 100:.1f}% "
            "MORE than the coarse model predicted, which means the model "
            "under-rates it rather than that the plant is over-performing"
        )

    return ToolResult(
        tool="compute_expected_output",
        summary=(
            f"Measured {measured_kwh:,.0f} kWh against an expected "
            f"{expected_kwh:,.0f} kWh from the light that actually arrived — "
            f"a shortfall of {deficit * 100:.2f}%, moving from "
            f"{first_deficit * 100:.2f}% in the first half of the window to "
            f"{second_deficit * 100:.2f}% in the second."
        ),
        values={
            "expected_energy_kwh": expected_kwh,
            "measured_energy_kwh": measured_kwh,
            "shortfall_kwh": expected_kwh - measured_kwh,
            "deficit_fraction": _finite(deficit),
            "deficit_first_half": first_deficit,
            "deficit_second_half": second_deficit,
            "deficit_change_across_window": second_deficit - first_deficit,
            "worst_day_shortfall_kwh": (
                float(-daily.min()) * interval_h if not daily.empty else 0.0
            ),
            "modelled_clipped_intervals": float(expectation.clipped.sum()),
            "days_compared": float(len(daily)),
        },
        labels=(
            {"worst_day": str(pd.Timestamp(worst_day).date())} if worst_day else {}
        ),
        samples_used=int(lit.sum()),
        caveats=caveats,
    )


def _half_deficit(expected: pd.Series, measured: pd.Series) -> float:
    total = float(expected.sum())
    if total <= 0:
        return 0.0
    return 1.0 - float(measured.sum()) / total


def _interval_hours(frame: pd.DataFrame) -> float:
    index = pd.DatetimeIndex(frame.index)
    if len(index) < 2:
        raise ToolError("need at least two samples to infer the sampling interval")
    seconds = float(pd.Series(index).diff().dropna().mode().iloc[0].total_seconds())
    return seconds / 3600.0


# ---------------------------------------------------------------------------
# 3. Per-string / per-combiner current balance
# ---------------------------------------------------------------------------
class BalanceArgs(WindowArgs):
    min_poa_wm2: float = Field(
        default=400.0,
        gt=0.0,
        description=(
            "Shares are only meaningful in decent light; at dawn every string "
            "carries almost nothing and the ratios are noise."
        ),
    )
    hour_start: int | None = Field(
        default=None,
        ge=0,
        le=23,
        description=(
            "Restrict to this site-local hour onwards. A shadow depresses some "
            "strings for part of the day only, so averaging over the whole day "
            "dilutes it into nothing; narrowing to the suspect hours is what "
            "separates a shadow from a failed string."
        ),
    )
    hour_end: int | None = Field(
        default=None,
        ge=1,
        le=24,
        description="Restrict to before this site-local hour.",
    )


def per_mppt_current_balance(ctx: ToolContext, args: BalanceArgs) -> ToolResult:
    window = slice_window(ctx.frame, args.start, args.end)
    currents, _ = string_columns(window)
    if len(currents) < 2:
        raise ToolError(
            "this system exposes fewer than two per-string current channels, so "
            "strings cannot be compared against each other"
        )
    if "poa_wm2" not in window:
        raise ToolError(
            "no plane-of-array irradiance channel, so light cannot be held constant"
        )

    lit = pd.to_numeric(window["poa_wm2"], errors="coerce") >= args.min_poa_wm2
    if args.hour_start is not None or args.hour_end is not None:
        hours = _local_hour(pd.DatetimeIndex(window.index), ctx.utc_offset_hours)
        if args.hour_start is not None:
            lit &= hours >= float(args.hour_start)
        if args.hour_end is not None:
            lit &= hours < float(args.hour_end)
        if not lit.any():
            raise ToolError(
                f"no well-lit interval falls between {args.hour_start}:00 and "
                f"{args.hour_end}:00 site time in this window"
            )
    block = window.loc[lit, currents].apply(pd.to_numeric, errors="coerce")
    totals = block.sum(axis=1)
    # A total near zero turns every share into a vast meaningless number, so
    # score only intervals where the array is genuinely carrying current.
    floor = 0.2 * float(totals.median()) if len(totals) else 0.0
    usable = totals > max(floor, 1e-6)
    if not usable.any():
        raise ToolError(
            f"no interval above {args.min_poa_wm2:g} W/m^2 carried measurable "
            "string current"
        )

    shares = block[usable].div(totals[usable], axis=0).mean()
    even = 1.0 / len(currents)
    deviations = (shares - even).abs()
    worst_column = str(deviations.idxmax())
    worst_index = int(worst_column.rsplit("_", 1)[-1])
    # Report the lowest share explicitly as well as the furthest from even. A
    # loss on some strings pushes the *others* above the even share, so "worst
    # deviation" can point at a perfectly healthy string that merely gained
    # share when its neighbours lost it.
    lowest_column = str(shares.idxmin())
    lowest_index = int(lowest_column.rsplit("_", 1)[-1])
    # Two depths rather than one threshold. This plant has a string that sits
    # persistently ~12% below even in untouched data — either a smaller combiner
    # or a long-standing fault, unresolvable without a maintenance log — so a
    # single "below even" count would fire on every window ever measured. Giving
    # both depths lets the agent see the difference between a standing offset
    # and something that has actually failed, instead of a threshold deciding
    # for it.
    below_10 = int((shares < even * 0.90).sum())
    below_25 = int((shares < even * 0.75).sum())

    values = {
        f"share_string_{c.rsplit('_', 1)[-1]}": float(v) for c, v in shares.items()
    }
    values.update(
        {
            "string_count": float(len(currents)),
            "even_share": even,
            "worst_deviation_from_even": float(deviations.max()),
            "worst_string_index": float(worst_index),
            "worst_string_share": float(shares[worst_column]),
            "lowest_string_index": float(lowest_index),
            "lowest_string_share": float(shares.min()),
            "highest_string_share": float(shares.max()),
            "strings_more_than_10pct_below_even": float(below_10),
            "strings_more_than_25pct_below_even": float(below_25),
            "below_even_threshold_a_pct": 10.0,
            "below_even_threshold_b_pct": 25.0,
            "share_spread": float(shares.max() - shares.min()),
            "intervals_scored": float(int(usable.sum())),
        }
    )

    hours_note = ""
    if args.hour_start is not None or args.hour_end is not None:
        hours_note = (
            f" restricted to {args.hour_start or 0:02d}:00-"
            f"{args.hour_end or 24:02d}:00 site time"
        )

    return ToolResult(
        tool="per_mppt_current_balance",
        summary=(
            f"Across {len(currents)} string channels the lowest share is string "
            f"{lowest_index} at {shares.min():.3f} against an even "
            f"{even:.3f}; {below_10} strings sit more than 10% below even and "
            f"{below_25} more than 25% below. Scored over "
            f"{int(usable.sum())} well-lit intervals{hours_note}."
        ),
        values=values,
        labels={
            "lowest_string_channel": lowest_column,
            "furthest_from_even_channel": worst_column,
        },
        samples_used=int(usable.sum()),
        caveats=[
            "shares are a within-array comparison; a loss affecting every "
            "string equally leaves them flat and is invisible here",
            "shares sum to one, so strings losing current push the rest above "
            "the even share — read which strings are low, not which moved most",
        ],
    )


# ---------------------------------------------------------------------------
# 4. Measured irradiance against the clear-sky envelope
# ---------------------------------------------------------------------------
class ClearSkyArgs(WindowArgs):
    clearest_day_quantile: float = Field(
        default=0.9,
        ge=0.5,
        le=1.0,
        description=(
            "Only the clearest days are compared. An overcast fortnight sits "
            "legitimately far below clear-sky and would otherwise read as a "
            "broken sensor."
        ),
    )


def check_clearsky_consistency(ctx: ToolContext, args: ClearSkyArgs) -> ToolResult:
    window = slice_window(ctx.frame, args.start, args.end)
    if "poa_wm2" not in window:
        raise ToolError("no plane-of-array irradiance channel to check")

    index = pd.DatetimeIndex(window.index)
    modelled = clearsky_poa(
        index,
        ctx.latitude,
        ctx.longitude,
        ctx.altitude_m,
        ctx.tilt_deg,
        ctx.azimuth_deg,
        albedo=ctx.albedo,
    ).poa_global
    measured = pd.to_numeric(window["poa_wm2"], errors="coerce")

    bright = modelled > 300.0
    if not bright.any():
        raise ToolError("no interval in this window is modelled above 300 W/m^2")

    ratio = (measured[bright] / modelled[bright]).replace([np.inf, -np.inf], np.nan)
    daily = ratio.groupby(pd.DatetimeIndex(ratio.index).normalize()).median().dropna()
    if daily.empty:
        raise ToolError("no usable day-level comparison against clear-sky")

    clearest = float(daily.quantile(args.clearest_day_quantile))
    # A sensor that is drifting reads progressively lower against a fixed
    # physical reference; a sensor that is merely mis-calibrated reads low by a
    # constant. Splitting the window is what tells those apart.
    half = len(daily) // 2
    first = (
        float(daily.iloc[:half].quantile(args.clearest_day_quantile))
        if half
        else clearest
    )
    second = (
        float(daily.iloc[half:].quantile(args.clearest_day_quantile))
        if len(daily) - half
        else clearest
    )
    span_days = max(float(len(daily)), 1.0)

    # A half of the window containing no genuinely clear day cannot say anything
    # about the sensor: an overcast fortnight reads low against clear-sky
    # whatever the pyranometer is doing. Counting the clear days is what stops
    # "it was cloudy in the second half" from being reported as a drifting
    # sensor, which is otherwise exactly what this number looks like.
    clear_enough = daily >= 0.85
    clear_first = int(clear_enough.iloc[:half].sum())
    clear_second = int(clear_enough.iloc[half:].sum())

    caveats = [
        "the clear-sky model is an upper bound, so a small standing shortfall "
        "is normal; cloud, soiling on the sensor and a drifting sensor all push "
        "this number the same way"
    ]
    if min(clear_first, clear_second) < 3:
        caveats.append(
            f"only {clear_first} clear days in the first half of the window and "
            f"{clear_second} in the second: the half-to-half change is a "
            "measurement of the weather at least as much as of the sensor, and "
            "must not be read as drift on its own"
        )

    return ToolResult(
        tool="check_clearsky_consistency",
        summary=(
            f"On its clearest days the irradiance sensor reads {clearest:.3f} of "
            f"the modelled clear-sky value, moving from {first:.3f} in the first "
            f"half of the window to {second:.3f} in the second, over "
            f"{len(daily)} days of which {clear_first + clear_second} were clear."
        ),
        values={
            "clearest_day_ratio": clearest,
            "shortfall_vs_clearsky": max(0.0, 1.0 - clearest),
            "ratio_first_half": first,
            "ratio_second_half": second,
            "ratio_change_across_window": second - first,
            "ratio_change_per_day": (second - first) / span_days,
            "days_compared": float(len(daily)),
            "clear_days_first_half": float(clear_first),
            "clear_days_second_half": float(clear_second),
        },
        series={"daily_clearsky_ratio": [float(v) for v in daily.to_numpy()]},
        series_index=[str(t) for t in daily.index],
        samples_used=int(bright.sum()),
        caveats=caveats,
    )


# ---------------------------------------------------------------------------
# 5. Flat ceiling on the AC channel
# ---------------------------------------------------------------------------
class CeilingArgs(WindowArgs):
    flatness_tolerance: float = Field(
        default=0.005,
        gt=0.0,
        description="Sample-to-sample change, as a fraction of peak, counted as flat.",
    )
    min_run_intervals: int = Field(
        default=3,
        ge=2,
        description=(
            "Consecutive flat samples required. Every smooth midday curve has "
            "samples near its own maximum; only a run of them is a ceiling."
        ),
    )


def check_ac_ceiling(ctx: ToolContext, args: CeilingArgs) -> ToolResult:
    window = slice_window(ctx.frame, args.start, args.end)
    if "ac_power_kw" not in window:
        raise ToolError("no AC power channel")
    power = pd.to_numeric(window["ac_power_kw"], errors="coerce").dropna()
    if power.empty:
        raise ToolError("the AC power channel is empty over this window")

    peak = float(power.max())
    if peak <= 0:
        raise ToolError("the plant produced nothing over this window")

    daylight = power[power > 0.15 * peak]
    if len(daylight) < args.min_run_intervals:
        raise ToolError("too few generating intervals to look for a ceiling")

    values_series = daylight.astype(float)
    near_top = values_series >= 0.98 * peak
    flat = near_top & (values_series.diff().abs() <= args.flatness_tolerance * peak)
    run_id = (~flat).cumsum()
    run_length = flat.groupby(run_id).transform("sum")
    pinned = flat & (run_length >= args.min_run_intervals)

    plateau = float(values_series[pinned].mean()) if pinned.any() else float("nan")
    longest = float(run_length.max()) if len(run_length) else 0.0
    days_with_plateau = (
        int(
            pd.Series(pinned.to_numpy(), index=pd.DatetimeIndex(daylight.index))
            .groupby(pd.DatetimeIndex(daylight.index).normalize())
            .any()
            .sum()
        )
        if pinned.any()
        else 0
    )

    values = {
        "peak_ac_kw": peak,
        "flat_ceiling_share": float(pinned.sum()) / float(len(daylight)),
        "longest_flat_run_intervals": longest,
        "days_with_a_plateau": float(days_with_plateau),
        "generating_intervals": float(len(daylight)),
    }
    if pinned.any():
        values["plateau_level_kw"] = plateau
    if ctx.ac_ceiling_kw:
        values["inverter_ac_rating_kw"] = float(ctx.ac_ceiling_kw)
        values["peak_as_fraction_of_ac_rating"] = peak / float(ctx.ac_ceiling_kw)
        if pinned.any():
            values["plateau_as_fraction_of_ac_rating"] = plateau / float(
                ctx.ac_ceiling_kw
            )

    if pinned.any():
        summary = (
            f"Power sits on a flat plateau of {plateau:.1f} kW for "
            f"{values['flat_ceiling_share'] * 100:.1f}% of generating intervals "
            f"on {days_with_plateau} days; the window's peak is {peak:.1f} kW."
        )
    else:
        summary = (
            f"No flat plateau: power peaks at {peak:.1f} kW and keeps changing "
            "between samples throughout."
        )

    return ToolResult(
        tool="check_ac_ceiling",
        summary=summary,
        values=values,
        samples_used=len(daylight),
        caveats=[
            "a flat ceiling is a shape, not a cause: an inverter at its AC "
            "rating and a grid operator's export limit are identical on this "
            "channel"
        ],
    )


# ---------------------------------------------------------------------------
# 6. Data quality
# ---------------------------------------------------------------------------
def profile_data_quality(ctx: ToolContext, args: WindowArgs) -> ToolResult:
    window = slice_window(ctx.frame, args.start, args.end)
    report = profile_quality(window, dc_capacity_kw=ctx.dc_capacity_kw)

    # The number that actually matters for a phantom deficit: intervals where
    # the sun was up and the irradiance sensor logged, but power did not.
    lit = pd.to_numeric(window.get("poa_wm2", pd.Series(dtype=float)), errors="coerce")
    power = pd.to_numeric(
        window.get("ac_power_kw", pd.Series(dtype=float)), errors="coerce"
    )
    daylight = lit >= 200.0
    orphaned = int((daylight & power.isna()).sum()) if len(power) else 0

    values = {
        "coverage": report.coverage,
        "expected_samples": float(report.expected_samples),
        "actual_samples": float(report.actual_samples),
        "missing_days": float(len(report.missing_days)),
        "lit_intervals_without_power": float(orphaned),
        "issue_count": float(len(report.issues)),
    }
    labels: dict[str, str] = {}
    worst = report.worst
    if worst is not None:
        values["worst_issue_share"] = worst.fraction
        labels["worst_issue"] = worst.kind.replace("_", " ")
        labels["worst_issue_channel"] = worst.column
    if report.missing_days:
        labels["first_missing_day"] = report.missing_days[0]
        labels["last_missing_day"] = report.missing_days[-1]

    return ToolResult(
        tool="profile_data_quality",
        summary=(
            f"{report.actual_samples} of an expected {report.expected_samples} "
            f"readings are present ({report.coverage * 100:.1f}%), with "
            f"{orphaned} well-lit intervals logging irradiance but no power, "
            f"across {len(report.missing_days)} entirely missing days."
        ),
        values=values,
        labels=labels,
        samples_used=len(window),
        caveats=[
            "missing readings remove energy from the record without removing it "
            "from the plant; a deficit computed across a gap is a measurement "
            "of the logger"
        ],
    )


# ---------------------------------------------------------------------------
# 7. Onset shape
# ---------------------------------------------------------------------------
class OnsetArgs(WindowArgs):
    metric: str = Field(
        default="pr_temperature_corrected",
        description=(
            "Daily series to characterise: 'pr_temperature_corrected', 'pr', "
            "or 'energy_kwh'."
        ),
    )


def characterize_onset(ctx: ToolContext, args: OnsetArgs) -> ToolResult:
    window = slice_window(ctx.frame, args.start, args.end)
    daily_frame = pr_timeseries(window, ctx.dc_capacity_kw, ctx.gamma_pdc, freq="1D")
    if args.metric not in daily_frame.columns:
        raise ToolError(
            f"unknown metric {args.metric!r}; available: "
            + ", ".join(daily_frame.columns)
        )
    daily = daily_frame[args.metric].dropna()
    if len(daily) < 6:
        raise ToolError(
            f"only {len(daily)} scored days in this window; an onset needs at "
            "least six to separate a level change from day-to-day scatter"
        )

    series = daily.astype(float)
    n = len(series)
    # Exhaustive single-changepoint scan. At window sizes of a few weeks this
    # is trivially cheap and needs no tuning, unlike a threshold on the daily
    # difference — which fires on any cloudy day.
    best_k, best_gap = 0, 0.0
    for k in range(2, n - 1):
        gap = abs(float(series.iloc[:k].mean()) - float(series.iloc[k:].mean()))
        if gap > best_gap:
            best_k, best_gap = k, gap

    before = float(series.iloc[:best_k].mean()) if best_k else float(series.mean())
    after = float(series.iloc[best_k:].mean()) if best_k else float(series.mean())
    scatter = float(series.diff().abs().median())
    drop = before - after

    # How many days the series spends in transit between the two levels. Zero
    # or one is a step (a fuse, a breaker); several is a ramp (a degrading
    # connection, accumulating dirt). The distinction is a *duration*, not a
    # label — naming it is the agent's job.
    if abs(drop) > 1e-9:
        low, high = sorted((before, after))
        band_lo = low + 0.15 * abs(drop)
        band_hi = high - 0.15 * abs(drop)
        in_transit = int(((series > band_lo) & (series < band_hi)).sum())
    else:
        in_transit = 0

    onset_stamp = pd.Timestamp(series.index[min(best_k, n - 1)])

    return ToolResult(
        tool="characterize_onset",
        summary=(
            f"Daily {args.metric} averages {before:.3f} before "
            f"{onset_stamp.date()} and {after:.3f} after, a change of "
            f"{drop:+.3f} against a day-to-day scatter of {scatter:.3f}, with "
            f"{in_transit} days spent between the two levels."
        ),
        values={
            "level_before": before,
            "level_after": after,
            "absolute_change": drop,
            "relative_change": drop / before if before else 0.0,
            "days_in_transition": float(in_transit),
            "days_before": float(best_k),
            "days_after": float(n - best_k),
            "day_to_day_scatter": scatter,
            # Zero scatter means the change is *infinitely* significant, not
            # insignificant. Returning 0.0 there would invert the meaning of
            # the one number that tells an event from day-to-day noise, so it
            # saturates at a stated ceiling instead.
            "change_over_scatter": _significance(drop, scatter),
        },
        labels={"onset_date": str(onset_stamp.date())},
        series={"daily_metric": [float(v) for v in series.to_numpy()]},
        series_index=[str(t) for t in series.index],
        samples_used=n,
        caveats=[
            "this always returns the largest level change in the window, even "
            "when there is nothing to find; compare the change against the "
            "day-to-day scatter before treating it as an event"
        ],
    )


# ---------------------------------------------------------------------------
# 8. Where in the day the loss sits
# ---------------------------------------------------------------------------
def time_of_day_profile(ctx: ToolContext, args: WindowArgs) -> ToolResult:
    window = slice_window(ctx.frame, args.start, args.end)
    model = ExpectationModel(ctx.dc_capacity_kw, ctx.gamma_pdc, ctx.ac_ceiling_kw)
    expectation = model.run(window)

    lit = expectation.expected_ac_kw >= 0.05 * ctx.dc_capacity_kw
    if not lit.any():
        raise ToolError("no interval in this window has meaningful expected output")

    hours = _local_hour(pd.DatetimeIndex(window.index), ctx.utc_offset_hours)
    frame = pd.DataFrame(
        {
            "hour": hours[lit].astype(int),
            "expected": expectation.expected_ac_kw[lit],
            "measured": expectation.measured_ac_kw[lit].fillna(0.0),
        }
    )
    grouped = frame.groupby("hour")[["expected", "measured"]].sum()
    grouped = grouped[grouped["expected"] > 0]
    if grouped.empty:
        raise ToolError("no hour of the day has enough expected output to compare")

    ratio = (grouped["measured"] / grouped["expected"]).astype(float)
    # The coarse model carries a standing offset, so the *level* of these ratios
    # says more about the model than about the plant. Dividing each hour by the
    # day's median leaves the shape, which is the thing that separates a shadow
    # (confined to a band of hours) from a fuse (flat across the day).
    median_ratio = float(ratio.median())
    shape = ratio / median_ratio if median_ratio > 0 else ratio
    worst_hour = int(shape.idxmin())
    best_hour = int(shape.idxmax())
    overall = float(grouped["measured"].sum() / grouped["expected"].sum())

    values = {f"ratio_hour_{h:02d}": float(v) for h, v in ratio.items()}
    values.update({f"shape_hour_{h:02d}": float(v) for h, v in shape.items()})
    values.update(
        {
            "worst_hour_local": float(worst_hour),
            "worst_hour_shape": float(shape.min()),
            "worst_hour_ratio": float(ratio[worst_hour]),
            "best_hour_local": float(best_hour),
            "best_hour_shape": float(shape.max()),
            "shape_spread_across_day": float(shape.max() - shape.min()),
            "ratio_spread_across_day": float(ratio.max() - ratio.min()),
            "median_hour_ratio": median_ratio,
            "whole_day_ratio": overall,
            "hours_compared": float(len(ratio)),
        }
    )

    return ToolResult(
        tool="time_of_day_profile",
        summary=(
            f"Measured output runs at {overall:.3f} of expected across the day. "
            f"Relative to the day's median hour, the weakest is "
            f"{worst_hour:02d}:00 site time at {shape.min():.3f} and the "
            f"strongest {best_hour:02d}:00 at {shape.max():.3f}, a spread of "
            f"{shape.max() - shape.min():.3f}."
        ),
        values=values,
        series={
            "hourly_ratio": [float(v) for v in ratio.to_numpy()],
            "hourly_shape": [float(v) for v in shape.to_numpy()],
        },
        series_index=[f"{int(h):02d}:00" for h in ratio.index],
        samples_used=int(lit.sum()),
        caveats=[
            "hours are site-local, converted from UTC with the logger offset "
            "recovered at ingest",
            "the expectation model has no incidence-angle term, so the first "
            "and last generating hour of the day read low on a perfectly "
            "healthy plant; treat the edges of this profile with suspicion",
            "a loss that costs the same fraction at every hour flattens this "
            "profile and is invisible here",
        ],
    )


# ---------------------------------------------------------------------------
# 9. Trend and recovery across the window
# ---------------------------------------------------------------------------
class TrendArgs(WindowArgs):
    recovery_step: float = Field(
        default=0.02,
        gt=0.0,
        description=(
            "Single-day improvement counted as a recovery. Rain washing an "
            "array back to clean is a step up; a failed component never "
            "recovers on its own."
        ),
    )


def daily_performance_trend(ctx: ToolContext, args: TrendArgs) -> ToolResult:
    window = slice_window(ctx.frame, args.start, args.end)
    daily_frame = pr_timeseries(window, ctx.dc_capacity_kw, ctx.gamma_pdc, freq="1D")
    series = daily_frame["pr_temperature_corrected"].dropna().astype(float)
    if len(series) < 5:
        raise ToolError(f"only {len(series)} scored days; a trend needs at least five")

    x = np.arange(len(series), dtype=float)
    y = series.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    steps = series.diff().dropna()
    recoveries = steps[steps >= args.recovery_step]

    return ToolResult(
        tool="daily_performance_trend",
        summary=(
            f"Temperature-corrected performance ratio moves "
            f"{slope:+.5f} per day over {len(series)} days "
            f"({slope * len(series):+.4f} across the window, R² "
            f"{r_squared:.2f}), with {len(recoveries)} single-day recoveries of "
            f"at least {args.recovery_step:.2f}."
        ),
        values={
            "slope_per_day": float(slope),
            "change_across_window": float(slope * len(series)),
            "r_squared": float(r_squared),
            "days": float(len(series)),
            "first_day_value": float(y[0]),
            "last_day_value": float(y[-1]),
            "recovery_count": float(len(recoveries)),
            "largest_recovery": float(recoveries.max()) if len(recoveries) else 0.0,
            "recovery_threshold": float(args.recovery_step),
        },
        series={"daily_pr_temperature_corrected": [float(v) for v in y]},
        series_index=[str(t) for t in series.index],
        samples_used=len(series),
        caveats=[
            "a straight line through a fortnight of weather is a weak "
            "instrument; read R² before reading the slope"
        ],
    )


# ---------------------------------------------------------------------------
# Registry entries
# ---------------------------------------------------------------------------
SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="compute_temp_corrected_pr",
        description=(
            "Performance ratio over a window, both as measured and corrected to "
            "a 25 °C cell temperature, with the share of the gap that heat "
            "explains."
        ),
        question_answered=(
            "Is the plant under-performing once the weather and the heat are "
            "divided out?"
        ),
        args_model=PrArgs,
        fn=compute_temp_corrected_pr,
        discriminates=("seasonal_temperature_derating", "weather", "real_loss"),
    ),
    ToolSpec(
        name="compute_expected_output",
        description=(
            "Energy actually produced against the energy a coarse model "
            "predicts from the light that arrived."
        ),
        question_answered="How large is the shortfall, in kWh and as a fraction?",
        args_model=WindowArgs,
        fn=compute_expected_output,
        discriminates=("real_loss", "weather"),
    ),
    ToolSpec(
        name="per_mppt_current_balance",
        description=(
            "Each string's share of total array current in good light, against "
            "an even split."
        ),
        question_answered=(
            "Is the loss confined to part of the array, or does it affect all "
            "of it equally?"
        ),
        args_model=BalanceArgs,
        fn=per_mppt_current_balance,
        discriminates=("string_outage", "shading", "soiling", "weather"),
    ),
    ToolSpec(
        name="check_clearsky_consistency",
        description=(
            "Measured plane-of-array irradiance against a modelled clear-sky "
            "envelope on the clearest days, split across the window to expose "
            "drift."
        ),
        question_answered="Is the irradiance sensor telling the truth?",
        args_model=ClearSkyArgs,
        fn=check_clearsky_consistency,
        discriminates=("sensor_drift", "soiling", "weather"),
    ),
    ToolSpec(
        name="check_ac_ceiling",
        description=(
            "Whether power sits on a flat plateau, how long the runs are, and "
            "where the plateau sits relative to the inverter's AC rating."
        ),
        question_answered="Is output being held at a ceiling?",
        args_model=CeilingArgs,
        fn=check_ac_ceiling,
        discriminates=("clipping", "curtailment"),
    ),
    ToolSpec(
        name="profile_data_quality",
        description=(
            "Coverage, missing days, flatlines, night generation, and lit "
            "intervals that logged irradiance but no power."
        ),
        question_answered="Is the apparent loss a plant problem or a logger problem?",
        args_model=WindowArgs,
        fn=profile_data_quality,
        discriminates=("telemetry_gap", "real_loss"),
    ),
    ToolSpec(
        name="characterize_onset",
        description=(
            "The largest sustained level change in a daily metric: the levels "
            "either side, the date, the days spent in transit, and the "
            "day-to-day scatter to judge it against."
        ),
        question_answered="Did this arrive abruptly or accumulate?",
        args_model=OnsetArgs,
        fn=characterize_onset,
        discriminates=("string_outage", "soiling", "sensor_drift"),
    ),
    ToolSpec(
        name="time_of_day_profile",
        description=(
            "Measured over expected output by site-local hour, with the spread "
            "across the day."
        ),
        question_answered="Is the loss confined to a band of hours?",
        args_model=WindowArgs,
        fn=time_of_day_profile,
        discriminates=("shading", "string_outage", "clipping"),
    ),
    ToolSpec(
        name="daily_performance_trend",
        description=(
            "Slope, fit quality and single-day recoveries in daily "
            "temperature-corrected performance ratio."
        ),
        question_answered=(
            "Is performance drifting steadily, and does it recover on its own?"
        ),
        args_model=TrendArgs,
        fn=daily_performance_trend,
        discriminates=("soiling", "sensor_drift", "string_outage"),
    ),
)


# Every tool must accept a `ToolArgs` subclass and return a `ToolResult`; the
# registry is built once and checked at import so a typo is a startup error.
for _spec in SPECS:
    if not issubclass(_spec.args_model, ToolArgs):
        raise TypeError(f"{_spec.name}: args_model must subclass ToolArgs")
