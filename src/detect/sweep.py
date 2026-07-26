"""The sweep: deterministic detectors that decide *when to look*, never *why*.

This is the only part of the system that runs unprompted, so it has exactly one
job — notice that a window is worth investigating — and it must not do the
investigating. A detector that named a cause would be a rules engine on a timer,
and the agent would be reduced to writing up a conclusion already reached.

So a `DeficitSignal` carries a scope, a window, a measured size and the name of
the measurement that produced it. It carries no cause, and there is no field it
could be smuggled into.

Everything here is deterministic and threshold-driven, with the thresholds read
from `config/site_defaults.yaml`. No LLM (CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.tools import ToolContext, ToolError, run_tool

__all__ = ["DeficitSignal", "sweep_window", "trailing_window"]


@dataclass(frozen=True)
class DeficitSignal:
    """Something worth investigating. Not a diagnosis.

    `question` is the plain-language prompt handed to whichever diagnostic runs
    next. It is phrased as a question rather than an assertion on purpose: "why
    is output down" invites a differential, "the array is soiled" does not.
    """

    scope: str
    start: str
    end: str
    detector: str
    measurement: str
    value: float
    threshold: float
    question: str

    @property
    def severity(self) -> float:
        """How far past the threshold, as a multiple. Used only for ordering."""
        if self.threshold == 0:
            return abs(self.value)
        return abs(self.value / self.threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "start": self.start,
            "end": self.end,
            "detector": self.detector,
            "measurement": self.measurement,
            "value": round(self.value, 6),
            "threshold": self.threshold,
            "question": self.question,
        }


def trailing_window(now: datetime, days: int) -> tuple[str, str]:
    """The window a sweep at simulated `now` should look at.

    Ends the day *before* `now`: the current day is still accumulating, and
    scoring a half-finished day produces a deficit every single morning.
    """
    end = (now - timedelta(days=1)).date()
    start = end - timedelta(days=max(1, days) - 1)
    return str(start), str(end)


def sweep_window(
    ctx: ToolContext,
    start: str,
    end: str,
    *,
    scope: str | None = None,
    pr_deficit: float = 0.05,
    baseline_days: int = 30,
    string_share_deficit: float = 0.22,
    completeness_floor: float = 0.80,
    ceiling_share: float = 0.02,
) -> list[DeficitSignal]:
    """Run every detector over one window and return what fired.

    Detectors overlap on purpose. A window can trip three of them at once, and
    an investigation that starts from three independent signals is better
    grounded than one that starts from whichever fired first. Ordering is by
    severity so the ranking is stable, not by detector name.
    """
    where = scope or ctx.scope
    signals: list[DeficitSignal] = []

    def measure(tool: str, args: dict[str, Any] | None = None) -> dict[str, float]:
        try:
            return run_tool(
                tool, ctx, {"start": start, "end": end, **(args or {})}
            ).values
        except ToolError:
            return {}

    quality = measure("profile_data_quality")
    completeness = quality.get("lit_interval_completeness", 1.0)
    if completeness < completeness_floor:
        signals.append(
            DeficitSignal(
                where,
                start,
                end,
                "data_completeness",
                "profile_data_quality.lit_interval_completeness",
                completeness,
                completeness_floor,
                "A stretch of production is missing from the record. What happened?",
            )
        )

    # A trailing comparison rather than an absolute level. A plant that has run
    # at 0.78 for two years is not developing a fault, and an absolute threshold
    # would open the same investigation every day forever.
    baseline = measure("compare_to_trailing_baseline", {"baseline_days": baseline_days})
    change = baseline.get("change")
    if change is not None and change < -pr_deficit:
        signals.append(
            DeficitSignal(
                where,
                start,
                end,
                "performance_ratio_drop",
                "compare_to_trailing_baseline.change",
                change,
                -pr_deficit,
                "Performance has dropped against the preceding weeks. Why?",
            )
        )

    balance = measure("per_mppt_current_balance")
    lowest = balance.get("lowest_string_share")
    even = balance.get("even_share", 0.0)
    if lowest is not None and even > 0:
        deficit = 1.0 - lowest / even
        if deficit > string_share_deficit:
            signals.append(
                DeficitSignal(
                    where,
                    start,
                    end,
                    "string_imbalance",
                    "per_mppt_current_balance.lowest_string_share",
                    deficit,
                    string_share_deficit,
                    "Part of this array is producing less than the rest. "
                    "What is going on?",
                )
            )

    ceiling = measure("check_ac_ceiling")
    share = ceiling.get("flat_ceiling_share", 0.0)
    if share > ceiling_share:
        signals.append(
            DeficitSignal(
                where,
                start,
                end,
                "flat_ceiling",
                "check_ac_ceiling.flat_ceiling_share",
                share,
                ceiling_share,
                "Output goes flat in the middle of the day. Is the inverter faulty?",
            )
        )

    return sorted(signals, key=lambda s: -s.severity)


def sweep_since(
    ctx: ToolContext,
    previous: datetime,
    now: datetime,
    *,
    lookback_days: int = 14,
    **thresholds: Any,
) -> list[DeficitSignal]:
    """Sweep the data that landed between two positions of the replay clock.

    The window is the *lookback*, not the gap between sweeps. A daily sweep over
    one day of data would compare a Tuesday against a Wednesday and find weather;
    detectors need a fortnight to say anything, so each sweep re-examines a
    trailing window that overlaps the last one.
    """
    if now <= previous:
        return []
    start, end = trailing_window(now, lookback_days)
    return sweep_window(ctx, start, end, **thresholds)


def signals_frame(signals: list[DeficitSignal]) -> pd.DataFrame:
    if not signals:
        return pd.DataFrame(
            columns=["scope", "start", "end", "detector", "value", "threshold"]
        )
    return pd.DataFrame([s.to_dict() for s in signals])
