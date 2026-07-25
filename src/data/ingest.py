"""Download PVDAQ days, normalise them, and write a reproducible parquet set.

The output is gitignored. What gets committed is the manifest: source keys,
SHA-256 of every raw file, the resolved channel map, and the timezone finding.
That makes the dataset reproducible without shipping it.

**Timezone is determined from the data, not assumed.** PVDAQ does not state a
timezone per system, and the usual guesses are both wrong somewhere: UTC is
wrong for most systems, and site-local-with-DST is wrong for the many loggers
that record local *standard* time year round. Guessing costs an hour of shift,
which lands the modelled clear-sky curve an hour off the measured one and makes
a healthy plant look like it has a tracking fault every morning. So the ingest
measures the offset instead: it tries every half-hour offset and keeps the one
whose modelled clear-sky curve best correlates with the measured irradiance.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import (
    STRING_CURRENT_PREFIX,
    STRING_DC_POWER_PREFIX,
    ChannelResolution,
    resolve_channels,
)
from src.data.sources import SystemMetadata, day_key, fetch_bytes, iter_days
from src.physics.clearsky import scan_utc_offset

__all__ = [
    "IngestResult",
    "TimezoneFinding",
    "detect_utc_offset",
    "ingest_system",
    "load_dataset",
    "normalise_frame",
]


@dataclass(frozen=True)
class TimezoneFinding:
    """The empirically measured logger offset, and how confident we are."""

    offset_hours: float
    samples_used: int
    correlation: float
    margin_over_runner_up: float
    note: str

    @property
    def trustworthy(self) -> bool:
        # A real alignment correlates strongly with clear-sky *and* beats the
        # neighbouring half-hour clearly. A high correlation with a thin margin
        # means the scan cannot actually distinguish adjacent offsets.
        return self.correlation >= 0.95 and self.margin_over_runner_up >= 0.01


@dataclass(frozen=True)
class IngestResult:
    parquet_path: Path
    manifest_path: Path
    rows: int
    interval_minutes: int
    start: str
    end: str
    timezone: TimezoneFinding
    resolution: ChannelResolution


def _read_day(system_id: int, day: date) -> tuple[date, bytes | None]:
    try:
        return day, fetch_bytes(day_key(system_id, day))
    except urllib.error.HTTPError:
        # Genuine gaps are normal in a decade-long archive; a missing day is
        # data, not an error. It shows up in the quality panel's gap calendar.
        return day, None
    except (urllib.error.URLError, TimeoutError):
        return day, None


def detect_utc_offset(
    frame: pd.DataFrame,
    meta: SystemMetadata,
    max_days: int = 40,  # retained for signature stability; unused
) -> TimezoneFinding:
    """Measure the logger's UTC offset by aligning POA against clear-sky.

    Delegates to `physics.clearsky.scan_utc_offset`, which tries every
    half-hour offset and keeps the one whose modelled clear-sky curve best
    correlates with the measured irradiance. Only the *timing* of the curve
    drives the answer, so cloud, soiling and calibration error cannot bias it.
    """
    try:
        scan = scan_utc_offset(
            frame["poa_wm2"],
            latitude=meta.latitude,
            longitude=meta.longitude,
            altitude_m=meta.altitude_m,
            tilt_deg=meta.tilt_deg,
            azimuth_deg=meta.azimuth_deg,
        )
    except ValueError as exc:
        return TimezoneFinding(
            offset_hours=0.0,
            samples_used=0,
            correlation=0.0,
            margin_over_runner_up=0.0,
            note=f"timezone recovery failed ({exc}); assumed UTC — VERIFY",
        )

    if scan.trustworthy:
        note = (
            f"recovered by clear-sky alignment: r={scan.correlation:.3f}, "
            f"beats UTC{scan.runner_up_hours:+g} by {scan.margin:.3f}"
        )
    else:
        note = (
            f"weak alignment (r={scan.correlation:.3f}, margin {scan.margin:.3f}) "
            "— treat the offset as unverified"
        )
    return TimezoneFinding(
        offset_hours=scan.offset_hours,
        samples_used=scan.samples,
        correlation=round(scan.correlation, 4),
        margin_over_runner_up=round(scan.margin, 4),
        note=note,
    )


def normalise_frame(
    raw: pd.DataFrame,
    resolution: ChannelResolution,
    interval_minutes: int,
) -> pd.DataFrame:
    """Rename to canonical columns, coerce to numeric, and resample."""
    frame = raw.rename(columns=resolution.rename_map)
    keep = [
        column
        for column in frame.columns
        if column in resolution.rename_map.values() or column == "timestamp"
    ]
    frame = frame[keep].copy()

    for column in frame.columns:
        if column != "timestamp":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]

    # Mean over the interval is right for power and irradiance: the quantity is
    # an average rate, and energy is that mean times the interval. Taking the
    # instantaneous sample instead would alias a partly-cloudy day badly.
    return frame.resample(f"{interval_minutes}min").mean()


def _clip_implausible(frame: pd.DataFrame, meta: SystemMetadata) -> pd.DataFrame:
    """Blank physically impossible values rather than letting them propagate.

    Negative irradiance at night is a normal thermopile offset of a few W/m²
    and is floored to zero. Anything beyond that is flagged as missing so the
    quality panel counts it, instead of quietly biasing a monthly PR.
    """
    out = frame.copy()
    if "poa_wm2" in out:
        out.loc[out["poa_wm2"] < -20, "poa_wm2"] = np.nan
        out["poa_wm2"] = out["poa_wm2"].clip(lower=0.0)
    if "ghi_wm2" in out:
        out.loc[out["ghi_wm2"] < -20, "ghi_wm2"] = np.nan
        out["ghi_wm2"] = out["ghi_wm2"].clip(lower=0.0)
    for column in ("ac_power_kw", "dc_power_kw"):
        if column in out:
            # Small negative overnight draw is real (inverter tare); a large
            # negative is a sign error or a meter fault.
            out.loc[out[column] < -0.05 * max(meta.dc_capacity_kw, 1.0), column] = (
                np.nan
            )
    return out


def ingest_system(
    meta: SystemMetadata,
    years: list[int],
    out_dir: Path,
    interval_minutes: int = 15,
    max_workers: int = 8,
    probe_days: int = 20,
) -> IngestResult:
    """Download, resolve, normalise and persist one system's data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    days = [
        day
        for year in sorted(years)
        for day in iter_days(date(year, 1, 1), date(year, 12, 31))
    ]

    raw_frames: list[pd.DataFrame] = []
    checksums: dict[str, str] = {}
    missing: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for day, payload in pool.map(lambda d: _read_day(meta.system_id, d), days):
            if payload is None:
                missing.append(day.isoformat())
                continue
            checksums[day.isoformat()] = hashlib.sha256(payload).hexdigest()
            frame = pd.read_csv(io.BytesIO(payload), low_memory=False)
            frame = frame.rename(columns={"measured_on": "timestamp"})
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            raw_frames.append(frame.dropna(subset=["timestamp"]))

    if not raw_frames:
        raise RuntimeError(
            f"no PVDAQ data downloaded for system {meta.system_id} in {years}"
        )

    combined = pd.concat(raw_frames, ignore_index=True)

    # Resolve channels on a spread-out sample so the probe sees clear days in
    # more than one season.
    stride = max(1, len(raw_frames) // probe_days)
    probe = pd.concat(raw_frames[::stride], ignore_index=True)
    resolution = resolve_channels(probe)

    normalised = normalise_frame(combined, resolution, interval_minutes)
    normalised = _clip_implausible(normalised, meta)

    timezone = detect_utc_offset(normalised, meta)
    # Shift naive logger stamps onto real UTC, then label them.
    shifted = normalised.copy()
    shifted.index = (
        pd.to_datetime(shifted.index) - timedelta(hours=timezone.offset_hours)
    ).tz_localize("UTC")
    shifted.index.name = "timestamp"

    parquet_path = (
        out_dir / f"system_{meta.system_id}_{min(years)}_{max(years)}.parquet"
    )
    shifted.to_parquet(parquet_path)

    manifest = {
        "system_id": meta.system_id,
        "name": meta.name,
        "location": meta.location,
        "years": sorted(years),
        "interval_minutes": interval_minutes,
        "rows": len(shifted),
        "start": shifted.index.min().isoformat(),
        "end": shifted.index.max().isoformat(),
        "columns": sorted(shifted.columns),
        "string_channels": len(resolution.string_current_columns),
        "timezone": asdict(timezone),
        "channel_resolution": resolution.to_dict(),
        "system_metadata": {k: v for k, v in asdict(meta).items() if k != "raw"},
        "missing_days": missing,
        "source_bucket": "s3://oedi-data-lake/pvdaq/csv/pvdata/",
        "sha256_by_day": checksums,
    }
    manifest_path = out_dir / f"system_{meta.system_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return IngestResult(
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        rows=len(shifted),
        interval_minutes=interval_minutes,
        start=str(shifted.index.min()),
        end=str(shifted.index.max()),
        timezone=timezone,
        resolution=resolution,
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """Load an ingested parquet set with a tz-aware UTC index."""
    frame = pd.read_parquet(path)
    index = pd.DatetimeIndex(frame.index)
    if index.tzinfo is None:
        index = index.tz_localize("UTC")
    frame.index = index
    frame.index.name = "timestamp"
    return frame


def string_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """The per-string current and DC-power columns present in a frame."""
    currents = sorted(
        (c for c in frame.columns if c.startswith(STRING_CURRENT_PREFIX)),
        key=lambda c: int(c.rsplit("_", 1)[-1]),
    )
    powers = sorted(
        (c for c in frame.columns if c.startswith(STRING_DC_POWER_PREFIX)),
        key=lambda c: int(c.rsplit("_", 1)[-1]),
    )
    return currents, powers
