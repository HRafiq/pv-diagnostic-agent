"""Turn a `GoldenCase` into the data a diagnostic actually sees.

Materialising a case means slicing the real measured window and, if the case
specifies one, applying its injector. The result carries both the perturbed
frame and the untouched baseline, so the harness can report what was actually
lost rather than trusting the injector's own accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from eval.golden import GoldenCase
from simulator.injectors import INJECTORS, InjectionRecord
from src.data.ingest import load_dataset

__all__ = ["MaterialisedCase", "load_system_frame", "materialise"]

_CACHE: dict[int, pd.DataFrame] = {}


def load_system_frame(system_id: int, data_dir: Path) -> pd.DataFrame:
    """Load and cache one system's ingested data."""
    if system_id not in _CACHE:
        matches = sorted(data_dir.glob(f"system_{system_id}_*.parquet"))
        if not matches:
            raise FileNotFoundError(
                f"system {system_id} is not ingested. Run: "
                f"python -m src.data.cli ingest --system {system_id} --years 2016 2017"
            )
        _CACHE[system_id] = load_dataset(matches[-1])
    return _CACHE[system_id]


@dataclass(frozen=True)
class MaterialisedCase:
    """A case with its data realised.

    `frame` is the case window, perturbed. `full_record` is the *entire*
    ingested series with that same perturbation spliced back in, and it is what
    the diagnostic actually gets. The distinction is load-bearing: a real
    investigation is asked about a fortnight but has the whole history to reach
    for, and several measurements are meaningless without it — "is this new?"
    needs the month before, and "was the weather unusual?" needs the same
    calendar weeks in other years. Handing over only the window silently turns
    those tools into errors and hides half the tool set from the evaluation.
    """

    case: GoldenCase
    frame: pd.DataFrame
    baseline: pd.DataFrame
    record: InjectionRecord | None
    full_record: pd.DataFrame

    @property
    def truly_lost_kwh(self) -> float:
        """Energy the injection actually removed, measured not asserted.

        Recomputed from the two frames rather than read off the injector, so a
        bug in an injector's own accounting cannot quietly become ground truth.
        """
        if "ac_power_kw" not in self.frame:
            return 0.0
        index = pd.DatetimeIndex(self.frame.index)
        if len(index) < 2:
            return 0.0
        hours = (
            float(pd.Series(index).diff().dropna().mode().iloc[0].total_seconds())
            / 3600.0
        )
        before = pd.to_numeric(self.baseline["ac_power_kw"], errors="coerce").fillna(0)
        after = pd.to_numeric(self.frame["ac_power_kw"], errors="coerce").fillna(0)
        return float((before - after).sum() * hours)

    def summary(self) -> dict[str, Any]:
        return {
            "case_id": self.case.id,
            "split": self.case.split,
            "rows": len(self.frame),
            "start": str(self.frame.index.min()),
            "end": str(self.frame.index.max()),
            "expected_category": self.case.expected_category,
            "expected_cause": self.case.expected_cause,
            "settled": self.case.settled,
            "injection": self.record.to_dict() if self.record else None,
            "energy_removed_kwh": round(self.truly_lost_kwh, 2),
        }


def materialise(case: GoldenCase, data_dir: Path) -> MaterialisedCase:
    """Slice the window and apply the case's injection, if any."""
    frame = load_system_frame(case.system_id, data_dir)
    index = pd.DatetimeIndex(frame.index)
    mask = (index >= pd.Timestamp(case.start, tz="UTC")) & (
        index <= pd.Timestamp(case.end, tz="UTC") + pd.Timedelta(days=1)
    )
    window = frame.loc[mask].copy()
    if window.empty:
        raise ValueError(f"case {case.id}: window {case.start}..{case.end} has no data")

    if case.injection is None:
        return MaterialisedCase(case, window, window.copy(), None, frame)

    kind = case.injection["kind"]
    if kind not in INJECTORS:
        raise KeyError(f"case {case.id}: unknown injector {kind!r}")

    params = dict(case.injection.get("params", {}))
    injected, record = INJECTORS[kind](
        window,
        start=str(window.index.min()),
        end=str(window.index.max()),
        **params,
    )
    # Splice the perturbed window back into the untouched record. History
    # before the window stays clean, which is what makes a trailing baseline a
    # real comparison rather than a comparison against the fault itself.
    full = frame.copy()
    full.loc[injected.index, injected.columns] = injected
    return MaterialisedCase(case, injected, window.copy(), record, full)
