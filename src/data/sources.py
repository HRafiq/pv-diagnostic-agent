"""Access to the NREL PVDAQ archive on the OEDI data lake.

PVDAQ is public and needs no credentials: it is a plain S3 bucket, hive
partitioned as ``pvdata/system_id=N/year=Y/month=M/day=D/*.csv``.

Nothing here is committed. Downloads land in a gitignored directory and a
manifest with SHA-256 checksums is committed instead, so the dataset is
reproducible without shipping it (CLAUDE.md, constraint 6).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta

__all__ = [
    "BUCKET_URL",
    "PVDAQ_PREFIX",
    "SystemMetadata",
    "day_key",
    "fetch_bytes",
    "fetch_system_metadata",
    "iter_days",
    "list_prefixes",
]

BUCKET_URL = "https://oedi-data-lake.s3.amazonaws.com/"
PVDAQ_PREFIX = "pvdaq/csv/"
_S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def fetch_bytes(key: str, timeout: int = 120) -> bytes:
    """Fetch one object from the bucket."""
    with urllib.request.urlopen(BUCKET_URL + key, timeout=timeout) as response:
        data: bytes = response.read()
    return data


def list_prefixes(prefix: str, timeout: int = 60) -> list[str]:
    """List the immediate sub-prefixes of a bucket prefix."""
    url = f"{BUCKET_URL}?list-type=2&prefix={prefix}&delimiter=/&max-keys=1000"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        tree = ET.fromstring(response.read())
    return [
        node.text or "" for node in tree.findall("s:CommonPrefixes/s:Prefix", _S3_NS)
    ]


def day_key(system_id: int, day: date) -> str:
    """The bucket key for one system-day.

    PVDAQ does not zero-pad month or day in the partition path, but *does*
    zero-pad them in the filename. Getting this backwards yields a 404 on every
    request, which is a slow thing to discover one file at a time.
    """
    return (
        f"{PVDAQ_PREFIX}pvdata/system_id={system_id}/"
        f"year={day.year}/month={day.month}/day={day.day}/"
        f"system_{system_id}__date_{day.year}_{day.month:02d}_{day.day:02d}.csv"
    )


def iter_days(start: date, end: date) -> Iterator[date]:
    """Every calendar day in ``[start, end]``, inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


@dataclass(frozen=True)
class SystemMetadata:
    """The subset of a PVDAQ system record the physics layer needs.

    Everything here is measured or specified by the site operator. None of it
    is inferred from the power data, which matters: an expectation model tuned
    against the very series it is meant to predict cannot detect a deficit in
    that series.
    """

    system_id: int
    name: str
    latitude: float
    longitude: float
    altitude_m: float
    location: str
    dc_capacity_kw: float
    tilt_deg: float
    azimuth_deg: float
    tracking: bool
    module_model: str
    module_quantity: int
    modules_per_string: int
    strings: int
    inverter_model: str
    raw: dict[str, object]

    @property
    def ac_capacity_kw_hint(self) -> float | None:
        """AC rating parsed out of the inverter model name, when it states one.

        Used only to draw the AC ceiling reference line and to seed clipping
        detection. `None` means "unknown"; nothing downstream may invent a
        value, because a fabricated ceiling turns healthy clipping into a
        phantom fault.
        """
        text = self.inverter_model.upper().replace("-", " ")
        for token in text.split():
            digits = "".join(ch for ch in token if ch.isdigit())
            if digits and "KW" in token:
                return float(digits)
        # "PVP 260kW" splits as ["PVP", "260KW"]; also handle "260 KW".
        parts = text.split()
        for index, token in enumerate(parts[:-1]):
            if parts[index + 1].startswith("KW") and token.isdigit():
                return float(token)
        return None


def _first(mapping: dict[str, object]) -> dict[str, object]:
    """PVDAQ nests repeated records as {'Inverter 0': {...}}. Take the first."""
    for value in mapping.values():
        if isinstance(value, dict):
            return value
    return {}


def _num(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def fetch_system_metadata(system_id: int, timeout: int = 60) -> SystemMetadata:
    """Fetch and flatten one system's metadata record."""
    key = f"{PVDAQ_PREFIX}system_metadata/{system_id}_system_metadata.json"
    try:
        raw = json.loads(fetch_bytes(key, timeout=timeout))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        raise LookupError(f"no PVDAQ metadata for system {system_id}") from exc

    system = raw.get("System", {})
    site = raw.get("Site", {})
    mount = _first(raw.get("Mount", {}))
    inverter = _first(raw.get("Inverters", {}))
    module = _first(raw.get("Modules", {}))

    return SystemMetadata(
        system_id=system_id,
        name=str(system.get("public_name", f"system_{system_id}")),
        latitude=_num(site.get("latitude")),
        longitude=_num(site.get("longitude")),
        altitude_m=_num(site.get("elevation")),
        location=str(site.get("location", "")).strip(),
        dc_capacity_kw=_num(system.get("power")),
        tilt_deg=_num(mount.get("tilt")),
        azimuth_deg=_num(mount.get("azimuth"), 180.0),
        # PVDAQ encodes fixed mounts as 'f'.
        tracking=str(mount.get("tracking", "f")).lower() not in ("f", "false", ""),
        module_model=str(module.get("model", "")),
        module_quantity=int(_num(module.get("quantity"))),
        modules_per_string=int(_num(inverter.get("modules_per_string"))),
        strings=int(_num(inverter.get("num_strings"))),
        inverter_model=str(inverter.get("model", "")),
        raw=raw,
    )
