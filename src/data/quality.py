"""Data-quality checks — the pathologies that masquerade as plant faults.

Every check here corresponds to something that produces a convincing but false
performance signal. They are deterministic and threshold-driven; no LLM reads or
writes any of it.

The July 2016 gap in NIST system 4902 is the motivating example: irradiance
logged normally for 35 days while AC power vanished entirely. Scored without a
completeness check that month reads as a 90% loss — a catastrophic fault that
never happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["QualityIssue", "QualityReport", "profile_quality"]


@dataclass(frozen=True)
class QualityIssue:
    """One detected data problem."""

    kind: str
    column: str
    count: int
    fraction: float
    detail: str
    first_seen: str | None = None
    last_seen: str | None = None

    @property
    def plain_summary(self) -> str:
        """Interface-safe wording (CLAUDE.md vocabulary rule)."""
        return f"{self.detail} ({self.fraction * 100:.1f}% of readings)"


@dataclass
class QualityReport:
    """Everything the Plant tab's data-quality panel shows."""

    interval_minutes: float
    expected_samples: int
    actual_samples: int
    issues: list[QualityIssue] = field(default_factory=list)
    daily_completeness: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )
    missing_days: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return (
            self.actual_samples / self.expected_samples
            if self.expected_samples
            else 0.0
        )

    @property
    def worst(self) -> QualityIssue | None:
        return max(self.issues, key=lambda i: i.fraction, default=None)

    def to_dict(self) -> dict[str, object]:
        return {
            "interval_minutes": self.interval_minutes,
            "expected_samples": self.expected_samples,
            "actual_samples": self.actual_samples,
            "coverage": round(self.coverage, 4),
            "missing_days": self.missing_days,
            "issues": [
                {
                    "kind": i.kind,
                    "column": i.column,
                    "count": i.count,
                    "fraction": round(i.fraction, 5),
                    "detail": i.detail,
                    "first_seen": i.first_seen,
                    "last_seen": i.last_seen,
                }
                for i in sorted(self.issues, key=lambda i: -i.fraction)
            ],
        }


def _stamp(series: pd.Series) -> tuple[str | None, str | None]:
    hits = series[series]
    if hits.empty:
        return None, None
    return str(hits.index.min()), str(hits.index.max())


def _flatline_mask(series: pd.Series, min_run: int) -> pd.Series:
    """True where a value repeats for at least ``min_run`` samples.

    A stuck sensor holds one value exactly. Real irradiance never does, even at
    night: the thermopile offset dithers by fractions of a W/m². A frozen
    channel therefore reads as a perfectly stable plant, which is the opposite
    of the truth.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    changed = numeric.ne(numeric.shift())
    group = changed.cumsum()
    run_length = numeric.groupby(group).transform("size")
    return (run_length >= min_run) & numeric.notna()


def profile_quality(
    frame: pd.DataFrame,
    night_power_tolerance_kw: float = 0.5,
    flatline_min_consecutive: int = 12,
    poa_column: str = "poa_wm2",
    power_column: str = "ac_power_kw",
    night_poa_wm2: float = 5.0,
    dc_capacity_kw: float | None = None,
) -> QualityReport:
    """Profile one system's data for the pathologies that mimic faults."""
    if frame.empty:
        return QualityReport(interval_minutes=0.0, expected_samples=0, actual_samples=0)

    index = pd.DatetimeIndex(frame.index)
    deltas = pd.Series(index).diff().dropna()
    interval_min = float(deltas.mode().iloc[0].total_seconds() / 60.0)
    span_minutes = (index.max() - index.min()).total_seconds() / 60.0
    expected = int(span_minutes / interval_min) + 1

    report = QualityReport(
        interval_minutes=interval_min,
        expected_samples=expected,
        actual_samples=len(frame),
    )
    total = float(len(frame))

    # --- missing values, per column ------------------------------------
    for column in frame.columns:
        missing = pd.to_numeric(frame[column], errors="coerce").isna()
        if missing.any():
            first, last = _stamp(missing)
            report.issues.append(
                QualityIssue(
                    kind="missing",
                    column=column,
                    count=int(missing.sum()),
                    fraction=float(missing.sum()) / total,
                    detail=f"no reading recorded for {column}",
                    first_seen=first,
                    last_seen=last,
                )
            )

    # --- power reported at night ---------------------------------------
    # Real output at night is impossible. A positive reading means a mislabelled
    # timezone, a meter reading the wrong circuit, or a sign error — and it
    # inflates every energy total it touches.
    if poa_column in frame and power_column in frame:
        poa = pd.to_numeric(frame[poa_column], errors="coerce")
        power = pd.to_numeric(frame[power_column], errors="coerce")
        night_generation = (poa < night_poa_wm2) & (power > night_power_tolerance_kw)
        if night_generation.any():
            first, last = _stamp(night_generation)
            report.issues.append(
                QualityIssue(
                    kind="night_generation",
                    column=power_column,
                    count=int(night_generation.sum()),
                    fraction=float(night_generation.sum()) / total,
                    detail=(
                        "power reported while the sun was down — usually a "
                        "timezone or metering error, never real generation"
                    ),
                    first_seen=first,
                    last_seen=last,
                )
            )

    # --- stuck channels -------------------------------------------------
    for column in (poa_column, power_column, "temp_ambient_c", "temp_module_c"):
        if column not in frame:
            continue
        # Night is legitimately flat for irradiance and power; only look where
        # the quantity should be moving.
        candidate = frame[column]
        if column in (poa_column, power_column) and poa_column in frame:
            daylight = pd.to_numeric(frame[poa_column], errors="coerce") > 50.0
            candidate = candidate.where(daylight)
        stuck = _flatline_mask(candidate, flatline_min_consecutive)
        if stuck.any():
            first, last = _stamp(stuck)
            minutes = flatline_min_consecutive * interval_min
            report.issues.append(
                QualityIssue(
                    kind="flatline",
                    column=column,
                    count=int(stuck.sum()),
                    fraction=float(stuck.sum()) / total,
                    detail=(
                        f"{column} held one value for {minutes:.0f} minutes or "
                        "more in daylight — a stuck sensor reads as a perfectly "
                        "steady plant"
                    ),
                    first_seen=first,
                    last_seen=last,
                )
            )

    # --- physically impossible magnitudes -------------------------------
    if dc_capacity_kw and power_column in frame:
        power = pd.to_numeric(frame[power_column], errors="coerce")
        # Above 1.2x nameplate is not achievable even with cloud-edge boost.
        impossible = power > 1.2 * dc_capacity_kw
        if impossible.any():
            first, last = _stamp(impossible)
            report.issues.append(
                QualityIssue(
                    kind="out_of_range",
                    column=power_column,
                    count=int(impossible.sum()),
                    fraction=float(impossible.sum()) / total,
                    detail=(
                        f"output above 120% of the {dc_capacity_kw:.0f} kW "
                        "nameplate, which the array cannot physically produce"
                    ),
                    first_seen=first,
                    last_seen=last,
                )
            )

    # --- daily completeness (drives the gap calendar) -------------------
    if power_column in frame:
        power = pd.to_numeric(frame[power_column], errors="coerce")
        per_day = power.groupby(pd.DatetimeIndex(power.index).date).agg(
            ["count", "size"]
        )
        completeness = (per_day["count"] / per_day["size"]).astype(float)
        completeness.index = pd.to_datetime(completeness.index)
        report.daily_completeness = completeness
        zero_days = pd.DatetimeIndex(completeness.index[completeness == 0.0])
        report.missing_days = [str(day.date()) for day in zero_days]

    return report


def summarise_string_balance(
    frame: pd.DataFrame,
    string_columns: list[str],
    min_poa_wm2: float = 400.0,
    poa_column: str = "poa_wm2",
) -> pd.DataFrame:
    """Per-string share of total current, in good light.

    The basis for `per_mppt_current_balance`. Strings on one array see the same
    irradiance, so in steady light their currents track each other closely. One
    channel drifting below its siblings is a combiner, fuse or string problem;
    *all* of them falling together is irradiance, not equipment — which is
    exactly the discrimination the ratio makes possible and an absolute
    threshold does not.
    """
    present = [c for c in string_columns if c in frame]
    if not present:
        return pd.DataFrame(columns=["mean_share", "std_share", "samples"])

    mask = pd.to_numeric(frame[poa_column], errors="coerce") >= min_poa_wm2
    window = frame.loc[mask, present].apply(pd.to_numeric, errors="coerce")
    total = window.sum(axis=1)
    usable = total > 0
    shares = window[usable].div(total[usable], axis=0)

    return pd.DataFrame(
        {
            "mean_share": shares.mean(),
            "std_share": shares.std(),
            "samples": shares.count().astype(float),
            "expected_share": np.repeat(1.0 / len(present), len(present)),
        }
    )
