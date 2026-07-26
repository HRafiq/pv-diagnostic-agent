"""The measurement tools, tested against answers that can be worked out by hand.

Two kinds of test here, and the second matters more:

* **Does the number come out right?** A string at zero current has a share of
  zero; a plant at a flat ceiling reports a plateau at that ceiling.
* **Does the tool stay inside its contract?** No tool may return a fault class,
  every number it states must be in its own ledger, and a measurement it cannot
  take must raise rather than return zero. Those are the properties that keep
  the architecture from collapsing into a rules engine, and they are the ones a
  future change is most likely to break without anyone noticing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agent.grounding import check_numeric_grounding
from src.tools import (
    REGISTRY,
    ToolContext,
    ToolError,
    catalogue,
    run_tool,
    slice_window,
    tool_names,
)
from src.tools.base import ToolResult
from tests.conftest import STRINGS, context_for, synthetic_frame


# ===========================================================================
# The contract
# ===========================================================================
def test_no_tool_returns_a_classification() -> None:
    """The load-bearing architectural rule (CLAUDE.md).

    A tool that could name a fault class would make every run identical and the
    agency metrics meaningless. `ToolResult` has no field for one, so this test
    guards the field list rather than the names.
    """
    forbidden = {"category", "cause", "fault", "diagnosis", "classification"}
    assert not (forbidden & set(ToolResult.model_fields))


def test_tool_names_describe_measurements_not_verdicts() -> None:
    for name in tool_names():
        assert not any(
            word in name for word in ("classify", "diagnose", "identify_fault")
        )


def test_every_spec_is_registered_under_its_own_name() -> None:
    for name, spec in REGISTRY.items():
        assert spec.name == name


def test_catalogue_is_stable_and_complete() -> None:
    entries = catalogue()
    assert [e["name"] for e in entries] == tool_names()
    assert catalogue() == entries  # stable ordering, so prompts cache
    for entry in entries:
        assert entry["description"] and entry["answers"]


def test_unknown_tool_raises_and_names_the_alternatives(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="no tool named"):
        run_tool("classify_fault", ctx, {})


def test_invalid_arguments_are_rejected_not_coerced(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="invalid arguments"):
        run_tool("compute_temp_corrected_pr", ctx, {"min_poa_wm2": -50.0})
    with pytest.raises(ToolError, match="invalid arguments"):
        run_tool("compute_temp_corrected_pr", ctx, {"nonsense": 1})


@pytest.mark.parametrize("name", tool_names())
def test_every_number_in_a_summary_is_in_its_own_ledger(
    name: str, ctx: ToolContext
) -> None:
    """The provenance rule applied to the tools themselves.

    If a tool's own prose quotes a figure it did not put in `values`, the
    grounding check on the agent's answer is scoring against an incomplete
    ledger and the "zero fabricated numerics" target is unenforceable.
    """
    result = run_tool(name, ctx, {})
    report = check_numeric_grounding(result.summary, result.values)
    assert report.ok, f"{name} states ungrounded figures: {report.ungrounded}"


@pytest.mark.parametrize("name", tool_names())
def test_ledgers_hold_only_finite_numbers(name: str, ctx: ToolContext) -> None:
    result = run_tool(name, ctx, {})
    for key, value in result.values.items():
        assert np.isfinite(value), f"{name}.{key} is not a real number"


@pytest.mark.parametrize("name", tool_names())
def test_tools_are_deterministic(name: str, ctx: ToolContext) -> None:
    assert run_tool(name, ctx, {}).values == run_tool(name, ctx, {}).values


# ===========================================================================
# Window slicing
# ===========================================================================
def test_a_bare_end_date_includes_the_whole_of_that_day(plant: pd.DataFrame) -> None:
    """The off-by-one that shifts an onset by 24 hours and sends the wrong shift."""
    window = slice_window(plant, "2017-04-02", "2017-04-02")
    assert window.index.max().hour == 23
    assert len(window) == 96


def test_an_empty_window_raises_rather_than_measuring_nothing(
    plant: pd.DataFrame,
) -> None:
    with pytest.raises(ToolError, match="no data between"):
        slice_window(plant, "2020-01-01", "2020-01-02")


# ===========================================================================
# compute_temp_corrected_pr
# ===========================================================================
def test_pr_is_higher_once_corrected_for_temperature(ctx: ToolContext) -> None:
    """The seasonal false alarm, made visible.

    The synthetic plant runs above 25 °C in the middle of the day, so the
    corrected figure must exceed the raw one. If it did not, the correction
    would be inverted and every summer month would read as a fault.
    """
    values = run_tool("compute_temp_corrected_pr", ctx, {}).values
    assert values["pr_temperature_corrected"] > values["pr"]
    assert 0.0 < values["temperature_loss_fraction"] < 0.2


def test_pr_ignores_irradiance_level(plant: pd.DataFrame) -> None:
    """Halving the sun halves the energy and leaves PR alone. That is the point.

    Only the *corrected* figure is invariant, and the reason is instructive: a
    dimmer day is also a cooler one, so raw PR rises when the sun weakens. That
    is exactly the confound temperature correction exists to remove, and it is
    why the raw number cannot be compared across seasons.
    """
    bright = run_tool("compute_temp_corrected_pr", context_for(plant), {}).values
    dim_frame = synthetic_frame(peak_poa=450.0)
    dim = run_tool("compute_temp_corrected_pr", context_for(dim_frame), {}).values

    assert dim["energy_kwh"] < bright["energy_kwh"] * 0.6
    assert dim["pr_temperature_corrected"] == pytest.approx(
        bright["pr_temperature_corrected"], abs=0.02
    )
    assert dim["pr"] > bright["pr"]
    assert dim["mean_cell_temp_c"] < bright["mean_cell_temp_c"]


def test_pr_reports_incomplete_coverage_rather_than_a_phantom_deficit(
    plant: pd.DataFrame,
) -> None:
    """A telemetry gap must not read as a 40% loss.

    The numerator and denominator have to cover the same intervals. This is the
    bug that made July 2016 read PR = 0.10 on real data.
    """
    holed = plant.copy()
    lit = holed["poa_wm2"] >= 200.0
    victims = holed.index[lit][:200]
    holed.loc[victims, "ac_power_kw"] = np.nan

    clean = run_tool("compute_temp_corrected_pr", context_for(plant), {}).values
    gappy = run_tool("compute_temp_corrected_pr", context_for(holed), {}).values

    assert gappy["samples_missing_power"] == 200
    assert gappy["data_completeness"] < 1.0
    # PR itself barely moves; only the completeness figure does.
    assert gappy["pr"] == pytest.approx(clean["pr"], abs=0.02)


def test_pr_raises_when_nothing_is_lit_enough_to_score(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="no interval"):
        run_tool("compute_temp_corrected_pr", ctx, {"min_poa_wm2": 5000.0})


def test_thin_coverage_is_carried_as_a_caveat(plant: pd.DataFrame) -> None:
    holed = plant.copy()
    lit = holed["poa_wm2"] >= 200.0
    holed.loc[holed.index[lit][:400], "ac_power_kw"] = np.nan
    result = run_tool("compute_temp_corrected_pr", context_for(holed), {})
    assert any("power reading" in c for c in result.caveats)


# ===========================================================================
# per_mppt_current_balance
# ===========================================================================
def test_a_healthy_array_shows_even_shares(ctx: ToolContext) -> None:
    values = run_tool("per_mppt_current_balance", ctx, {}).values
    assert values["even_share"] == pytest.approx(1.0 / STRINGS)
    assert values["worst_deviation_from_even"] < 1e-9
    assert values["strings_more_than_10pct_below_even"] == 0


def test_a_dead_string_shows_a_zero_share(plant: pd.DataFrame) -> None:
    broken = plant.copy()
    broken["string_current_a_3"] = 0.0
    values = run_tool("per_mppt_current_balance", context_for(broken), {}).values
    assert values["share_string_3"] == pytest.approx(0.0, abs=1e-9)
    assert values["lowest_string_index"] == 3
    assert values["strings_more_than_25pct_below_even"] == 1
    # The six survivors each pick up the lost share, which is why "furthest from
    # even" alone can point at a healthy string.
    assert values["share_string_1"] == pytest.approx(1.0 / (STRINGS - 1), abs=1e-6)


def test_the_hour_filter_finds_a_loss_the_daily_average_hides(
    plant: pd.DataFrame,
) -> None:
    """The shading case in miniature — and the reason the filter exists.

    A shadow on three strings for three hours barely moves the whole-day
    shares. Narrowed to those hours it is unmistakable. An agent that cannot
    narrow the window cannot tell a shadow from a failed string.
    """
    shaded = plant.copy()
    hours = pd.DatetimeIndex(shaded.index).hour
    band = (hours >= 7) & (hours < 10)
    for n in (1, 2, 3):
        shaded.loc[band, f"string_current_a_{n}"] *= 0.45

    ctx = context_for(shaded)
    whole_day = run_tool("per_mppt_current_balance", ctx, {}).values
    narrowed = run_tool(
        "per_mppt_current_balance", ctx, {"hour_start": 7, "hour_end": 10}
    ).values

    assert whole_day["strings_more_than_25pct_below_even"] == 0
    assert narrowed["strings_more_than_25pct_below_even"] == 3
    assert narrowed["lowest_string_share"] < whole_day["lowest_string_share"]


def test_balance_raises_without_string_channels(plant: pd.DataFrame) -> None:
    bare = plant.drop(columns=[c for c in plant.columns if c.startswith("string_")])
    with pytest.raises(ToolError, match="fewer than two"):
        run_tool("per_mppt_current_balance", context_for(bare), {})


def test_balance_raises_when_the_hour_filter_selects_nothing(
    ctx: ToolContext,
) -> None:
    with pytest.raises(ToolError, match="no well-lit interval"):
        run_tool("per_mppt_current_balance", ctx, {"hour_start": 1, "hour_end": 3})


# ===========================================================================
# check_ac_ceiling
# ===========================================================================
def test_a_smooth_day_is_not_a_ceiling(ctx: ToolContext) -> None:
    """The false positive that would call every healthy midday 'clipping'.

    Every smooth curve has samples near its own maximum. Only a *run* of
    near-identical samples is a ceiling.
    """
    values = run_tool("check_ac_ceiling", ctx, {}).values
    assert values["flat_ceiling_share"] == 0.0
    assert "plateau_level_kw" not in values


def test_a_capped_plant_reports_the_plateau(plant: pd.DataFrame) -> None:
    capped = plant.copy()
    capped["ac_power_kw"] = capped["ac_power_kw"].clip(upper=55.0)
    values = run_tool("check_ac_ceiling", context_for(capped), {}).values
    assert values["plateau_level_kw"] == pytest.approx(55.0, abs=0.01)
    assert values["flat_ceiling_share"] > 0.1
    assert values["days_with_a_plateau"] == 14
    assert values["longest_flat_run_intervals"] > 3


def test_the_ceiling_tool_refuses_to_name_a_cause(plant: pd.DataFrame) -> None:
    """Clipping and curtailment are the same shape. The tool must say so."""
    capped = plant.copy()
    capped["ac_power_kw"] = capped["ac_power_kw"].clip(upper=55.0)
    result = run_tool("check_ac_ceiling", context_for(capped), {})
    assert "clipping" not in result.summary.lower()
    assert "curtail" not in result.summary.lower()
    assert any("shape, not a cause" in c for c in result.caveats)


# ===========================================================================
# check_clearsky_consistency
# ===========================================================================
def test_a_drifting_sensor_shows_a_falling_ratio(plant: pd.DataFrame) -> None:
    drifting = plant.copy()
    days = (pd.DatetimeIndex(drifting.index) - drifting.index[0]).days.to_numpy()
    drifting["poa_wm2"] = drifting["poa_wm2"] * (1.0 - 0.012 * days)

    clean = run_tool("check_clearsky_consistency", context_for(plant), {}).values
    drift = run_tool("check_clearsky_consistency", context_for(drifting), {}).values

    # Compared against the *same window undrifted*, not against zero. A healthy
    # sensor already shows some half-to-half movement here from weather alone,
    # so only the difference between the two carries information about the
    # instrument.
    #
    # The gap is smaller than the drift itself (7.5 points against 15.6) and
    # that is the tool working as intended: it scores the *clearest* day in
    # each half, which is the least-drifted one. Buying robustness against a
    # cloudy fortnight costs sensitivity to slow drift, and the trade is worth
    # making — a tool that called every overcast week a broken pyranometer
    # would be dispatching technicians to healthy weather stations.
    assert (
        drift["ratio_change_across_window"]
        < clean["ratio_change_across_window"] - 0.05
    )
    assert drift["clearest_day_ratio"] < clean["clearest_day_ratio"] - 0.02
    assert drift["shortfall_vs_clearsky"] > clean["shortfall_vs_clearsky"]


def test_a_cloudy_half_is_flagged_rather_than_read_as_drift(
    plant: pd.DataFrame,
) -> None:
    """An overcast fortnight is not a broken sensor, and must not read as one."""
    cloudy = plant.copy()
    second_half = pd.DatetimeIndex(cloudy.index) >= cloudy.index[len(cloudy) // 2]
    cloudy.loc[second_half, "poa_wm2"] *= 0.3
    result = run_tool("check_clearsky_consistency", context_for(cloudy), {})
    assert result.values["clear_days_second_half"] == 0
    assert any("weather at least as much" in c for c in result.caveats)


# ===========================================================================
# time_of_day_profile
# ===========================================================================
def test_a_flat_loss_leaves_the_time_of_day_shape_flat(plant: pd.DataFrame) -> None:
    """A fuse costs the same fraction all day. That is what separates it."""
    even_loss = plant.copy()
    even_loss["ac_power_kw"] *= 0.85
    clean = run_tool("time_of_day_profile", context_for(plant), {}).values
    lossy = run_tool("time_of_day_profile", context_for(even_loss), {}).values
    assert lossy["whole_day_ratio"] < clean["whole_day_ratio"]
    assert lossy["shape_spread_across_day"] == pytest.approx(
        clean["shape_spread_across_day"], abs=0.02
    )


def test_a_morning_shadow_shows_up_in_the_shape(plant: pd.DataFrame) -> None:
    shaded = plant.copy()
    hours = pd.DatetimeIndex(shaded.index).hour
    band = (hours >= 7) & (hours < 10)
    shaded.loc[band, "ac_power_kw"] *= 0.6

    clean = run_tool("time_of_day_profile", context_for(plant), {}).values
    dark = run_tool("time_of_day_profile", context_for(shaded), {}).values

    assert dark["shape_spread_across_day"] > clean["shape_spread_across_day"] + 0.15
    assert 7 <= dark["worst_hour_local"] < 10


def test_time_of_day_hours_are_site_local(plant: pd.DataFrame) -> None:
    """UTC hours are meaningless to a plant manager and to a shadow."""
    shaded = plant.copy()
    hours = pd.DatetimeIndex(shaded.index).hour
    shaded.loc[(hours >= 12) & (hours < 15), "ac_power_kw"] *= 0.5
    values = run_tool(
        "time_of_day_profile", context_for(shaded, utc_offset_hours=-5.0), {}
    ).values
    assert 7 <= values["worst_hour_local"] < 10


# ===========================================================================
# characterize_onset
# ===========================================================================
def test_a_step_reports_two_levels_and_a_short_transition(
    plant: pd.DataFrame,
) -> None:
    stepped = plant.copy()
    after = pd.DatetimeIndex(stepped.index) >= pd.Timestamp("2017-04-08", tz="UTC")
    stepped.loc[after, "ac_power_kw"] *= 0.8

    values = run_tool("characterize_onset", context_for(stepped), {}).values
    assert values["absolute_change"] > 0.1
    assert values["days_in_transition"] <= 1
    assert values["change_over_scatter"] > 5


def test_a_ramp_spends_days_in_transition(plant: pd.DataFrame) -> None:
    """A corroding connection is a ramp; a blown fuse is a step. The tool
    reports the *duration*, and leaves the naming to the agent."""
    ramped = plant.copy()
    days = (pd.DatetimeIndex(ramped.index) - ramped.index[0]).days.to_numpy()
    ramped["ac_power_kw"] *= 1.0 - np.clip((days - 4) / 6.0, 0.0, 1.0) * 0.3

    step = run_tool("characterize_onset", context_for(plant.assign()), {})
    values = run_tool("characterize_onset", context_for(ramped), {}).values
    assert values["days_in_transition"] >= 3
    assert step.values["days_in_transition"] <= values["days_in_transition"]


def test_onset_admits_it_always_finds_something(ctx: ToolContext) -> None:
    """On clean data it still returns a changepoint — and says so.

    A tool that reports its own weakness is what stops the agent treating
    day-to-day scatter as an event.
    """
    result = run_tool("characterize_onset", ctx, {})
    assert result.values["change_over_scatter"] < 3.0
    assert any("even when there is nothing to find" in c for c in result.caveats)


def test_a_change_with_no_scatter_saturates_rather_than_reading_as_zero() -> None:
    """The degenerate case, and why it is not left at 0.0.

    Divide a real change by a zero denominator and the obvious guard returns
    0.0 — which reads as "indistinguishable from noise" for a change that is
    infinitely distinguishable from it. On a windless, cloudless test article
    that inversion would silently hide every injected step.
    """
    flat = synthetic_frame(weather=0.0)
    stepped = flat.copy()
    after = pd.DatetimeIndex(stepped.index) >= pd.Timestamp("2017-04-08", tz="UTC")
    stepped.loc[after, "ac_power_kw"] *= 0.8

    values = run_tool("characterize_onset", context_for(stepped), {}).values
    assert values["day_to_day_scatter"] == pytest.approx(0.0, abs=1e-12)
    assert values["change_over_scatter"] > 100


def test_onset_refuses_a_window_too_short_to_judge(plant: pd.DataFrame) -> None:
    with pytest.raises(ToolError, match="scored days"):
        run_tool(
            "characterize_onset",
            context_for(plant),
            {"start": "2017-04-01", "end": "2017-04-04"},
        )


def test_onset_rejects_an_unknown_metric(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="unknown metric"):
        run_tool("characterize_onset", ctx, {"metric": "vibes"})


# ===========================================================================
# daily_performance_trend
# ===========================================================================
def test_a_healthy_plant_has_no_trend(ctx: ToolContext) -> None:
    values = run_tool("daily_performance_trend", ctx, {}).values
    assert abs(values["slope_per_day"]) < 1e-6
    assert values["recovery_count"] == 0


def test_accumulating_soiling_shows_a_negative_slope_and_a_recovery(
    plant: pd.DataFrame,
) -> None:
    soiled = plant.copy()
    days = (pd.DatetimeIndex(soiled.index) - soiled.index[0]).days.to_numpy()
    # Dust accumulates at 1%/day and rain washes it off on day 8.
    loss = np.where(days < 8, 0.01 * days, 0.01 * (days - 8))
    soiled["ac_power_kw"] *= 1.0 - loss

    values = run_tool("daily_performance_trend", context_for(soiled), {}).values
    assert values["slope_per_day"] < -0.001
    assert values["recovery_count"] >= 1
    assert values["largest_recovery"] > 0.05


def test_trend_refuses_too_few_days(plant: pd.DataFrame) -> None:
    with pytest.raises(ToolError, match="at least five"):
        run_tool(
            "daily_performance_trend",
            context_for(plant),
            {"start": "2017-04-01", "end": "2017-04-03"},
        )


# ===========================================================================
# compute_expected_output and profile_data_quality
# ===========================================================================
def test_expected_output_measures_the_change_not_just_the_level(
    plant: pd.DataFrame,
) -> None:
    """The absolute deficit is mostly a property of the coarse model.

    A loss confined to the second half must show up as a change even though the
    standing offset is unchanged.
    """
    dropped = plant.copy()
    second_half = pd.DatetimeIndex(dropped.index) >= pd.Timestamp(
        "2017-04-08", tz="UTC"
    )
    dropped.loc[second_half, "ac_power_kw"] *= 0.75

    values = run_tool("compute_expected_output", context_for(dropped), {}).values
    assert values["deficit_change_across_window"] > 0.2
    assert values["deficit_second_half"] > values["deficit_first_half"]


def test_a_negative_deficit_says_the_model_under_rates_the_plant(
    plant: pd.DataFrame,
) -> None:
    strong = plant.copy()
    strong["ac_power_kw"] *= 1.3
    result = run_tool("compute_expected_output", context_for(strong), {})
    assert result.values["deficit_fraction"] < 0
    assert any("under-rates it" in c for c in result.caveats)


def test_data_quality_counts_lit_intervals_with_no_power(
    plant: pd.DataFrame,
) -> None:
    """The measurement that stops a logger outage reading as a plant outage."""
    holed = plant.copy()
    lit = holed["poa_wm2"] >= 200.0
    holed.loc[holed.index[lit][:120], "ac_power_kw"] = np.nan
    values = run_tool("profile_data_quality", context_for(holed), {}).values
    assert values["lit_intervals_without_power"] == 120


def test_data_quality_on_a_complete_record_reports_full_coverage(
    ctx: ToolContext,
) -> None:
    values = run_tool("profile_data_quality", ctx, {}).values
    assert values["coverage"] == pytest.approx(1.0)
    assert values["lit_intervals_without_power"] == 0
