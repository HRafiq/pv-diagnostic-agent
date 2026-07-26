"""Assemble a `ToolContext` from an ingested system.

`ToolContext` is deliberately source-agnostic — it knows about tilt and
nameplate, not about PVDAQ — so the translation from a dataset manifest into one
lives here rather than in `src/tools/`. Adding a second data source means a
second function in this module and no change to any tool.

Everything comes from the committed manifest, never a live fetch, so the agent,
the dashboard and the evaluation harness all describe the data actually on disk
and all agree with each other offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT
from src.data.ingest import load_dataset
from src.data.sources import SystemMetadata
from src.physics.modelchain import module_gamma_pdc
from src.tools import ToolContext

__all__ = ["DEFAULT_DATA_DIR", "PlantBundle", "load_plant"]

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"

# Used when the module is not in the CEC database. Stated rather than silent:
# the temperature coefficient is the single most important number for telling a
# real fault from summer derating, and a wrong one shifts every corrected
# performance ratio in the project.
_FALLBACK_GAMMA = -0.0040


class PlantBundle:
    """An ingested system: its frame, its metadata, and a ready `ToolContext`."""

    def __init__(
        self,
        frame: pd.DataFrame,
        meta: SystemMetadata,
        gamma_pdc: float,
        utc_offset_hours: float,
        manifest: dict[str, object],
    ) -> None:
        self.frame = frame
        self.meta = meta
        self.gamma_pdc = gamma_pdc
        self.utc_offset_hours = utc_offset_hours
        self.manifest = manifest

    def context(
        self,
        frame: pd.DataFrame | None = None,
        scope: str | None = None,
    ) -> ToolContext:
        """Build a `ToolContext`, optionally over a narrowed frame."""
        return ToolContext(
            frame=self.frame if frame is None else frame,
            dc_capacity_kw=self.meta.dc_capacity_kw,
            gamma_pdc=self.gamma_pdc,
            latitude=self.meta.latitude,
            longitude=self.meta.longitude,
            altitude_m=self.meta.altitude_m,
            tilt_deg=self.meta.tilt_deg,
            azimuth_deg=self.meta.azimuth_deg,
            ac_ceiling_kw=self.meta.ac_capacity_kw_hint,
            utc_offset_hours=self.utc_offset_hours,
            scope=scope or f"{self.meta.name} (whole plant)",
        )


_CACHE: dict[tuple[int, str], PlantBundle] = {}


def load_plant(system_id: int, data_dir: Path | None = None) -> PlantBundle:
    """Load an ingested system and everything the physics needs to score it."""
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    key = (system_id, str(root))
    if key in _CACHE:
        return _CACHE[key]

    manifest_path = root / f"system_{system_id}_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"system {system_id} is not ingested. Run: python -m src.data.cli "
            f"ingest --system {system_id} --years 2016 2017"
        )
    manifest = json.loads(manifest_path.read_text())

    parquet = sorted(root.glob(f"system_{system_id}_*.parquet"))
    if not parquet:
        raise FileNotFoundError(f"no parquet file for system {system_id} in {root}")

    meta = SystemMetadata(**{**manifest["system_metadata"], "raw": {}})
    gamma = module_gamma_pdc(meta.module_model) or _FALLBACK_GAMMA
    offset = float(manifest.get("timezone", {}).get("offset_hours", 0.0))

    bundle = PlantBundle(
        frame=load_dataset(parquet[-1]),
        meta=meta,
        gamma_pdc=gamma,
        utc_offset_hours=offset,
        manifest=manifest,
    )
    _CACHE[key] = bundle
    return bundle
