"""Physics, tested against analytically known answers.

The point of testing against synthetic frames with hand-computable answers is
that a plausible-looking number is the failure mode here. A PR of 0.78 looks
right whether or not the arithmetic is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.physics.modelchain import ExpectationModel, module_gamma_pdc
from src.physics.performance import (
    G_STC,
    T_STC,
    cell_temperature_faiman,
    compute_pr,
    energy_kwh,
    insolation_kwh_m2,
    pr_timeseries,
    temperature_correction_factor,
)

GAMMA = -0.0045


def frame(
    hours: int = 8,
    poa: float = 1000.0,
    power: float = 100.0,
    capacity: float = 100.0,
    module_temp: float | None = T_STC,
    freq: str = "1h",
) -> pd.DataFrame:
    index = pd.date_range("2017-06-01 08:00", periods=hours, freq=freq, tz="UTC")
    data = {
        "poa_wm2": np.full(hours, poa),
        "ac_power_kw": np.full(hours, power),
        "temp_ambient_c": np.full(hours, 20.0),
        "wind_speed_ms": np.full(hours, 1.0),
    }
    if module_temp is not None:
        data["temp_module_c"] = np.full(hours, module_temp)
    _ = capacity
    return pd.DataFrame(data, index=index)


# --------------------------------------------------------------------------
# The anchor case: a lossless plant at STC has PR = 1.0 exactly.
# --------------------------------------------------------------------------
def test_lossless_plant_at_stc_has_pr_of_one() -> None:
    result = compute_pr(frame(), dc_capacity_kw=100.0, gamma_pdc=GAMMA)
    assert result.pr == pytest.approx(1.0)
    # At exactly 25 °C the correction is a no-op.
    assert result.pr_temperature_corrected == pytest.approx(1.0)
    assert result.temperature_loss_fraction == pytest.approx(0.0)


def test_half_output_halves_pr() -> None:
    result = compute_pr(frame(power=50.0), dc_capacity_kw=100.0, gamma_pdc=GAMMA)
    assert result.pr == pytest.approx(0.5)


def test_half_irradiance_leaves_pr_unchanged() -> None:
    # This is the whole reason PR exists: a dim day is not a fault.
    result = compute_pr(
        frame(poa=500.0, power=50.0), dc_capacity_kw=100.0, gamma_pdc=GAMMA
    )
    assert result.pr == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Temperature correction
# --------------------------------------------------------------------------
def test_hot_module_correction_raises_pr() -> None:
    # 45 °C is 20 K above STC: 20 * 0.0045 = 9% below rating.
    hot = frame(module_temp=45.0, power=91.0)
    result = compute_pr(hot, dc_capacity_kw=100.0, gamma_pdc=GAMMA)
    assert result.pr == pytest.approx(0.91)
    assert result.pr_temperature_corrected == pytest.approx(1.0, abs=0.002)
    # Nearly all of the apparent deficit is heat, not a fault.
    assert result.temperature_loss_fraction == pytest.approx(0.09, abs=0.002)


def test_cold_module_correction_lowers_pr() -> None:
    # Below STC the modules genuinely outperform their rating; correcting
    # removes that bonus, so corrected PR is LOWER than measured. Getting this
    # sign wrong would make every winter look like an improvement.
    cold = frame(module_temp=5.0, power=109.0)
    result = compute_pr(cold, dc_capacity_kw=100.0, gamma_pdc=GAMMA)
    assert result.pr > result.pr_temperature_corrected


def test_correction_factor_matches_the_formula() -> None:
    temps = pd.Series([25.0, 45.0, 5.0])
    factor = temperature_correction_factor(temps, -0.004)
    assert list(factor.round(4)) == [1.0, 0.92, 1.08]


def test_positive_gamma_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative for silicon"):
        temperature_correction_factor(pd.Series([25.0]), +0.004)


def test_correction_factor_is_floored() -> None:
    # A spurious 300 °C reading must not drive the divisor negative and flip
    # the sign of the corrected output.
    assert float(temperature_correction_factor(pd.Series([300.0]), -0.005).iloc[0]) > 0


# --------------------------------------------------------------------------
# Faiman cell temperature
# --------------------------------------------------------------------------
def test_faiman_rises_above_ambient_with_irradiance() -> None:
    poa = pd.Series([0.0, 500.0, 1000.0])
    ambient = pd.Series([20.0, 20.0, 20.0])
    wind = pd.Series([1.0, 1.0, 1.0])
    cell = cell_temperature_faiman(poa, ambient, wind)
    assert cell.iloc[0] == pytest.approx(20.0)
    # 1000 / (25 + 6.84) = 31.4 K rise.
    assert cell.iloc[2] == pytest.approx(20.0 + 1000.0 / 31.84, abs=0.1)
    assert cell.is_monotonic_increasing


def test_wind_cools_the_module() -> None:
    poa, ambient = pd.Series([1000.0]), pd.Series([20.0])
    calm = cell_temperature_faiman(poa, ambient, pd.Series([0.5]))
    windy = cell_temperature_faiman(poa, ambient, pd.Series([8.0]))
    assert float(windy.iloc[0]) < float(calm.iloc[0])


# --------------------------------------------------------------------------
# Integration and masking
# --------------------------------------------------------------------------
def test_energy_and_insolation_integrate_over_the_interval() -> None:
    f = frame(hours=4, poa=1000.0, power=100.0)
    assert energy_kwh(f["ac_power_kw"]) == pytest.approx(400.0)
    assert insolation_kwh_m2(f["poa_wm2"]) == pytest.approx(4.0)


def test_low_light_samples_are_excluded() -> None:
    f = frame(hours=6)
    f.iloc[:3, f.columns.get_loc("poa_wm2")] = 50.0
    result = compute_pr(f, 100.0, GAMMA, min_poa_wm2=200.0)
    assert result.samples_used == 3
    assert result.samples_excluded_low_light == 3


def test_missing_power_excludes_its_insolation_too() -> None:
    """The July-2016 bug: a telemetry gap must not manufacture a deficit.

    If power is missing while irradiance keeps logging, integrating all the
    insolation against only the surviving energy reads as a catastrophic loss
    that never happened.
    """
    f = frame(hours=8, poa=1000.0, power=100.0)
    f.iloc[4:, f.columns.get_loc("ac_power_kw")] = np.nan
    result = compute_pr(f, 100.0, GAMMA)
    assert result.pr == pytest.approx(1.0)  # not 0.5
    assert result.samples_used == 4
    assert result.samples_missing_power == 4
    assert result.data_completeness == pytest.approx(0.5)


def test_full_data_reports_complete() -> None:
    assert compute_pr(frame(), 100.0, GAMMA).data_completeness == pytest.approx(1.0)


def test_empty_window_is_not_an_error() -> None:
    f = frame()
    f["poa_wm2"] = 0.0
    result = compute_pr(f, 100.0, GAMMA)
    assert result.samples_used == 0
    assert np.isnan(result.pr)


def test_negative_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        compute_pr(frame(), -1.0, GAMMA)


def test_missing_columns_named_in_the_error() -> None:
    with pytest.raises(KeyError, match="poa_wm2"):
        compute_pr(pd.DataFrame({"ac_power_kw": [1.0]}), 100.0, GAMMA)


def test_pr_timeseries_rolls_up_per_period() -> None:
    index = pd.date_range("2017-06-01", periods=48, freq="1h", tz="UTC")
    f = pd.DataFrame(
        {
            "poa_wm2": 800.0,
            "ac_power_kw": 80.0,
            "temp_ambient_c": 20.0,
            "temp_module_c": 25.0,
            "wind_speed_ms": 1.0,
        },
        index=index,
    )
    out = pr_timeseries(f, 100.0, GAMMA, freq="1D")
    assert len(out) == 2
    assert out["pr"].iloc[0] == pytest.approx(1.0)
    assert "data_completeness" in out


# --------------------------------------------------------------------------
# Expectation model
# --------------------------------------------------------------------------
def test_expectation_scales_with_irradiance() -> None:
    model = ExpectationModel(100.0, GAMMA, system_losses=0.0, inverter_efficiency=1.0)
    f = frame(hours=2, poa=1000.0, module_temp=T_STC)
    f.iloc[1, f.columns.get_loc("poa_wm2")] = 500.0
    result = model.run(f)
    assert float(result.expected_ac_kw.iloc[0]) == pytest.approx(100.0)
    assert float(result.expected_ac_kw.iloc[1]) == pytest.approx(50.0)


def test_expectation_applies_losses_and_efficiency() -> None:
    model = ExpectationModel(100.0, GAMMA, system_losses=0.14, inverter_efficiency=0.96)
    result = model.run(frame(hours=1, module_temp=T_STC))
    assert float(result.expected_ac_kw.iloc[0]) == pytest.approx(100 * 0.86 * 0.96)


def test_ac_ceiling_clips_and_is_flagged() -> None:
    model = ExpectationModel(
        200.0, GAMMA, ac_capacity_kw=100.0, system_losses=0.0, inverter_efficiency=1.0
    )
    result = model.run(frame(hours=2, module_temp=T_STC))
    assert float(result.expected_ac_kw.max()) == pytest.approx(100.0)
    assert bool(result.clipped.any())


def test_deficit_fraction_measures_the_shortfall() -> None:
    model = ExpectationModel(100.0, GAMMA, system_losses=0.0, inverter_efficiency=1.0)
    f = frame(hours=4, poa=1000.0, power=90.0, module_temp=T_STC)
    assert model.run(f).deficit_fraction() == pytest.approx(0.10)


def test_expectation_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError, match="gamma_pdc"):
        ExpectationModel(100.0, +0.004)
    with pytest.raises(ValueError, match="dc_capacity_kw"):
        ExpectationModel(0.0, GAMMA)
    with pytest.raises(ValueError, match="system_losses"):
        ExpectationModel(100.0, GAMMA, system_losses=1.5)


def test_module_lookup_tolerates_a_missing_manufacturer_prefix() -> None:
    # PVDAQ stores "NU-U235F2"; the CEC key is "Sharp_NU_U235F2". Silently
    # falling back to a default here would put a wrong temperature coefficient
    # at the centre of every correction in the project.
    assert module_gamma_pdc("NU-U235F2") == pytest.approx(-0.00458)
    assert module_gamma_pdc("Sharp_NU_U235F2") == pytest.approx(-0.00458)


def test_module_lookup_returns_none_when_unknown() -> None:
    assert module_gamma_pdc("definitely-not-a-real-module") is None
    assert module_gamma_pdc("") is None


def test_stc_constants() -> None:
    assert (G_STC, T_STC) == (1000.0, 25.0)
