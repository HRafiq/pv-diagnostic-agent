"""Channel resolution, quality profiling, and clear-sky timezone recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.quality import profile_quality, summarise_string_balance
from src.data.schema import resolve_channels
from src.data.sources import SystemMetadata, day_key, iter_days
from src.physics.clearsky import clearsky_poa, scan_utc_offset, solar_position


# --------------------------------------------------------------------------
# Channel resolution — the unit trap
# --------------------------------------------------------------------------
def _raw(**columns: object) -> pd.DataFrame:
    n = 24
    base = {
        "irradiance_poa_o_2204": np.linspace(0, 1000, n),
        "temperature_ambient_o_2205": np.linspace(10, 30, n),
        "ac_power_inv_14538": np.linspace(0, 200, n),
    }
    base.update(columns)
    return pd.DataFrame(base)


def test_resolver_picks_the_channel_in_the_right_units() -> None:
    # NIST 4902 exposes three POA channels; only one reads in W/m². Choosing on
    # name alone picks a channel ~115x off and every PR in the project is wrong
    # by that factor while still looking plausible.
    frame = _raw(irradiance_poa_o_2203=np.linspace(0, 10, 24))
    resolution = resolve_channels(frame)
    assert resolution.chosen["poa_wm2"].source_column == "irradiance_poa_o_2204"
    rejected = dict(resolution.chosen["poa_wm2"].rejected)
    assert "irradiance_poa_o_2203" in rejected
    assert "wrong units" in rejected["irradiance_poa_o_2203"]


def test_resolver_rejects_an_implausibly_high_channel() -> None:
    frame = _raw(wind_speed_o_1=np.full(24, 292.0), wind_speed_o_2=np.full(24, 4.0))
    resolution = resolve_channels(frame)
    assert resolution.chosen["wind_speed_ms"].source_column == "wind_speed_o_2"


def test_missing_required_quantity_fails_loudly() -> None:
    # A missing POA channel silently zeroes every performance ratio, so the
    # ingest must refuse rather than continue.
    frame = pd.DataFrame(
        {"ac_power_inv_1": [1.0, 2.0], "temperature_ambient_o": [20, 21]}
    )
    with pytest.raises(ValueError, match="poa_wm2"):
        resolve_channels(frame)


def test_optional_quantity_is_recorded_as_unresolved() -> None:
    resolution = resolve_channels(_raw())
    assert "ghi_wm2" in resolution.unresolved
    assert "poa_wm2" not in resolution.unresolved


def test_string_channels_sort_naturally() -> None:
    frame = _raw(
        **{
            "shuntcurrent_a_avg_10__1": np.ones(24),
            "shuntcurrent_a_avg_2__2": np.ones(24),
            "shuntcurrent_a_avg_1__3": np.ones(24),
        }
    )
    resolution = resolve_channels(frame)
    order = [
        c.split("_avg_")[1].split("__")[0] for c in resolution.string_current_columns
    ]
    assert order == ["1", "2", "10"]  # not lexicographic


def test_rename_map_covers_strings_and_quantities() -> None:
    frame = _raw(**{"shuntcurrent_a_avg_1__1": np.ones(24)})
    mapping = resolve_channels(frame).rename_map
    assert mapping["irradiance_poa_o_2204"] == "poa_wm2"
    assert mapping["shuntcurrent_a_avg_1__1"] == "string_current_a_1"


def test_resolution_serialises_for_the_manifest() -> None:
    payload = resolve_channels(_raw()).to_dict()
    assert "chosen" in payload and "unresolved" in payload


# --------------------------------------------------------------------------
# Quality profiling
# --------------------------------------------------------------------------
def _series(hours: int = 96) -> pd.DataFrame:
    index = pd.date_range("2017-06-01", periods=hours, freq="1h", tz="UTC")
    hour = index.hour.to_numpy()
    poa = np.where((hour >= 8) & (hour <= 16), 800.0, 0.0)
    return pd.DataFrame(
        {
            "poa_wm2": poa,
            "ac_power_kw": poa / 10.0,
            "temp_ambient_c": 20.0 + np.sin(np.arange(hours)),
            "temp_module_c": 30.0 + np.sin(np.arange(hours)),
        },
        index=index,
    )


def test_clean_data_reports_no_night_generation() -> None:
    report = profile_quality(_series())
    assert not any(i.kind == "night_generation" for i in report.issues)


def test_night_generation_is_detected() -> None:
    frame = _series()
    frame.loc[frame["poa_wm2"] == 0, "ac_power_kw"] = 12.0
    report = profile_quality(frame)
    issue = next(i for i in report.issues if i.kind == "night_generation")
    assert issue.count > 0
    assert "timezone or metering" in issue.detail


def test_missing_values_are_counted_and_stamped() -> None:
    frame = _series()
    frame.iloc[10:20, frame.columns.get_loc("ac_power_kw")] = np.nan
    report = profile_quality(frame)
    issue = next(
        i for i in report.issues if i.kind == "missing" and i.column == "ac_power_kw"
    )
    assert issue.count == 10
    assert issue.first_seen is not None


def test_flatline_detected_in_daylight_only() -> None:
    frame = _series()
    daylight = frame["poa_wm2"] > 50
    frame.loc[daylight, "temp_module_c"] = 42.0
    report = profile_quality(frame, flatline_min_consecutive=4)
    assert any(
        i.kind == "flatline" and i.column == "temp_module_c" for i in report.issues
    )


def test_impossible_output_flagged_against_nameplate() -> None:
    frame = _series()
    frame.iloc[12, frame.columns.get_loc("ac_power_kw")] = 5000.0
    report = profile_quality(frame, dc_capacity_kw=100.0)
    assert any(i.kind == "out_of_range" for i in report.issues)


def test_fully_missing_days_listed() -> None:
    frame = _series(hours=72)
    day2 = frame.index.normalize() == pd.Timestamp("2017-06-02", tz="UTC")
    frame.loc[day2, "ac_power_kw"] = np.nan
    report = profile_quality(frame)
    assert "2017-06-02" in report.missing_days


def test_empty_frame_is_handled() -> None:
    report = profile_quality(pd.DataFrame())
    assert report.coverage == 0.0 and report.worst is None


def test_string_balance_shares_sum_to_one() -> None:
    index = pd.date_range("2017-06-01 10:00", periods=10, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "poa_wm2": 800.0,
            "string_current_a_1": 10.0,
            "string_current_a_2": 10.0,
            "string_current_a_3": 5.0,
        },
        index=index,
    )
    balance = summarise_string_balance(frame, list(frame.columns[1:]))
    assert balance["mean_share"].sum() == pytest.approx(1.0)
    # The weak string sits below its even share; the others above.
    assert balance.loc["string_current_a_3", "mean_share"] < 1 / 3


# --------------------------------------------------------------------------
# Clear-sky and timezone recovery
# --------------------------------------------------------------------------
def test_solar_position_requires_tz_aware_input() -> None:
    naive = pd.date_range("2017-06-01", periods=4, freq="1h")
    with pytest.raises(ValueError, match="tz-aware"):
        solar_position(naive, 39.13, -77.21)


def test_clearsky_peaks_near_solar_noon() -> None:
    index = pd.date_range("2017-06-15", periods=96, freq="15min", tz="UTC")
    result = clearsky_poa(index, 39.13, -77.21, 138.0, 20.0, 180.0)
    # Gaithersburg is UTC-5 standard, so solar noon lands near 17:00 UTC.
    assert 15 <= int(result.poa_global.idxmax().hour) <= 19
    assert 700 < float(result.poa_global.max()) < 1200
    assert float(result.poa_global.min()) >= 0.0


def test_timezone_scan_recovers_a_known_offset() -> None:
    """Synthesise a logger keeping UTC-5 and check the scan finds it."""
    utc = pd.date_range("2017-06-01", periods=96 * 20, freq="15min", tz="UTC")
    modelled = clearsky_poa(utc, 39.13, -77.21, 138.0, 20.0, 180.0).poa_global
    # Relabel the same values as if stamped in local standard time.
    local = modelled.copy()
    local.index = (utc + pd.Timedelta(hours=-5)).tz_localize(None).tz_localize("UTC")

    scan = scan_utc_offset(local, 39.13, -77.21, 138.0, 20.0, 180.0, clear_days=10)
    assert scan.offset_hours == pytest.approx(-5.0)
    assert scan.correlation > 0.95
    assert scan.trustworthy


def test_timezone_scan_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no usable irradiance"):
        scan_utc_offset(pd.Series(dtype=float), 39.0, -77.0, 0.0, 20.0, 180.0)


# --------------------------------------------------------------------------
# PVDAQ source helpers
# --------------------------------------------------------------------------
def test_day_key_pads_the_filename_but_not_the_path() -> None:
    from datetime import date

    key = day_key(4902, date(2016, 3, 7))
    assert "year=2016/month=3/day=7/" in key
    assert key.endswith("system_4902__date_2016_03_07.csv")


def test_iter_days_is_inclusive() -> None:
    from datetime import date

    days = list(iter_days(date(2017, 1, 1), date(2017, 1, 3)))
    assert len(days) == 3 and days[-1] == date(2017, 1, 3)


@pytest.mark.parametrize(
    ("model", "expected"),
    [("PVP 260kW", 260.0), ("PVS-500", None), ("", None)],
)
def test_ac_capacity_hint_parsing(model: str, expected: float | None) -> None:
    meta = SystemMetadata(
        system_id=1,
        name="x",
        latitude=0,
        longitude=0,
        altitude_m=0,
        location="",
        dc_capacity_kw=100,
        tilt_deg=20,
        azimuth_deg=180,
        tracking=False,
        module_model="m",
        module_quantity=1,
        modules_per_string=1,
        strings=1,
        inverter_model=model,
        raw={},
    )
    assert meta.ac_capacity_kw_hint == expected
