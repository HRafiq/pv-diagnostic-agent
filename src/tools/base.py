"""The tool contract: what a measurement is, and what it is allowed to return.

Three rules from CLAUDE.md are enforced here rather than described:

1. **A tool takes a measurement. It never returns a fault class.** There is no
   `category` or `cause` field anywhere in `ToolResult`, so a tool physically
   cannot smuggle a diagnosis out. The diagnosis is assembled by the agent from
   several results, which is what makes the run a differential diagnosis rather
   than a lookup.
2. **No LLM inside a tool.** Every function in `src/tools/` is deterministic and
   takes its thresholds as explicit arguments.
3. **Every number the agent may state has to come from here.** `ToolResult.values`
   is a provenance ledger: a flat `{name: number}` map of every quantity the
   tool measured. The synthesiser's output is checked against the union of these
   ledgers (`src/agent/grounding.py`), so a fabricated figure is detectable
   rather than merely discouraged.

`summary` is a plain-language sentence describing *what was measured*, never what
it means. "String 3 carries 4.1% of the array current against an even share of
14.3%" — yes. "String 3 has failed" — no; that is the agent's call to make, and
only after it has looked at whether irradiance moved too.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ToolArgs",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "ToolSpec",
    "WindowArgs",
    "slice_window",
]


class ToolError(RuntimeError):
    """A tool could not run — missing channel, empty window, bad argument.

    Raised rather than returned as a result so a failed measurement can never be
    mistaken for a measurement of zero. The loop catches it, records it on the
    tape, and lets the agent choose a different tool.
    """


@dataclass(frozen=True)
class ToolContext:
    """Everything a tool needs about the plant, and nothing about the agent.

    Deliberately not `SystemMetadata`: tools are UI- and source-agnostic, so a
    second dataset with a different metadata shape needs no change here. The
    frame is the *whole* ingested series; each tool slices its own window, so a
    tool that needs trailing context (a baseline before the onset) can reach
    outside the window under investigation.
    """

    frame: pd.DataFrame
    dc_capacity_kw: float
    gamma_pdc: float
    latitude: float
    longitude: float
    altitude_m: float
    tilt_deg: float
    azimuth_deg: float
    ac_ceiling_kw: float | None = None
    albedo: float = 0.2
    scope: str = "plant"
    # The logger's offset from UTC, recovered from physics at ingest time (see
    # `physics.scan_utc_offset`). Timestamps stay UTC everywhere internally;
    # this exists so a tool that reports a *time of day* can say "09:00 at the
    # site" instead of "14:00 UTC". A shading window means nothing in UTC.
    utc_offset_hours: float = 0.0

    def __post_init__(self) -> None:
        if self.dc_capacity_kw <= 0:
            raise ValueError("dc_capacity_kw must be positive")
        if self.gamma_pdc > 0:
            raise ValueError(
                f"gamma_pdc must be negative for silicon; got {self.gamma_pdc:+g}"
            )


class ToolArgs(BaseModel):
    """Base class for validated tool arguments."""

    model_config = ConfigDict(extra="forbid")


class WindowArgs(ToolArgs):
    """Arguments for a tool that operates over a time window.

    Both bounds are optional. Omitting them means "the whole investigation
    window", which is the common case; supplying them is how the agent narrows
    in on a suspected onset without needing a separate tool.
    """

    start: str | None = Field(
        default=None, description="ISO date or timestamp, inclusive. UTC."
    )
    end: str | None = Field(
        default=None, description="ISO date or timestamp, inclusive. UTC."
    )


class ToolResult(BaseModel):
    """The structured output of one measurement.

    Note what is absent: no `category`, no `cause`, no `confidence`. A tool that
    could return one of those would collapse the whole system into a rules
    engine with a chat wrapper.
    """

    model_config = ConfigDict(extra="forbid")

    tool: str
    summary: str = Field(
        ...,
        description=(
            "One plain sentence stating what was measured. Never an "
            "interpretation, never a cause."
        ),
    )
    values: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "The provenance ledger. Every quantity this tool measured, flat and "
            "named. The agent's answer is checked against the union of these."
        ),
    )
    labels: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Non-numeric measured facts — a timestamp, a channel name. Kept "
            "apart from `values` so the numeric grounding check stays exact."
        ),
    )
    series: dict[str, list[float | None]] = Field(
        default_factory=dict,
        description="Optional supporting series, for the tape's inline charts.",
    )
    series_index: list[str] = Field(
        default_factory=list, description="ISO timestamps for `series`."
    )
    samples_used: int = Field(default=0, ge=0)
    caveats: list[str] = Field(
        default_factory=list,
        description=(
            "Anything that weakens this measurement — thin coverage, a short "
            "window, a missing channel. Carried into the answer, not dropped."
        ),
    )

    @model_validator(mode="after")
    def _ledger_holds_only_real_numbers(self) -> ToolResult:
        """No NaN or infinity in the provenance ledger, ever.

        A NaN reaching the ledger is worse than a failed measurement: it
        serialises to `null`, renders as a blank, and the agent reads the
        absence as "nothing there" rather than "this could not be computed".
        A tool that cannot measure something must raise `ToolError`.
        """
        bad = [k for k, v in self.values.items() if not math.isfinite(v)]
        if bad:
            raise ValueError(
                f"{self.tool} put non-finite values in its ledger: "
                + ", ".join(sorted(bad))
                + ". Raise ToolError instead — a measurement that could not be "
                "taken must not look like one that came out empty."
            )
        return self

    def digest(self, max_values: int = 24) -> str:
        """Compact rendering for an LLM prompt.

        Values are rounded for legibility but the ledger keeps full precision,
        so grounding is checked against the exact figure rather than this.
        """
        pairs = list(self.values.items())[:max_values]
        numbers = ", ".join(f"{k}={v:.6g}" for k, v in pairs)
        parts = [f"{self.tool}: {self.summary}"]
        if numbers:
            parts.append(f"  measured: {numbers}")
        if self.labels:
            parts.append("  " + ", ".join(f"{k}={v}" for k, v in self.labels.items()))
        if self.caveats:
            parts.append("  caveats: " + "; ".join(self.caveats))
        return "\n".join(parts)


@dataclass(frozen=True)
class ToolSpec:
    """A registry entry: the callable, its argument model, and how to pick it."""

    name: str
    description: str
    question_answered: str = ""
    args_model: type[ToolArgs] = WindowArgs
    fn: Callable[[ToolContext, Any], ToolResult] = lambda ctx, args: ToolResult(
        tool="undefined", summary=""
    )
    discriminates: tuple[str, ...] = ()

    def schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()

    def catalogue_entry(self) -> dict[str, Any]:
        """What the planner and router are shown. No implementation detail."""
        return {
            "name": self.name,
            "description": self.description,
            "answers": self.question_answered,
            "helps_separate": list(self.discriminates),
            "args": self.schema().get("properties", {}),
        }


def slice_window(
    frame: pd.DataFrame, start: str | None, end: str | None
) -> pd.DataFrame:
    """Slice a tz-aware frame by inclusive ISO bounds.

    A bare date as `end` means the *whole* of that day. Treating it as midnight
    would silently drop the final day of every window an agent asks for, which
    is the kind of off-by-one that shifts an onset date by 24 hours and sends
    someone to look at the wrong shift.
    """
    if frame.empty:
        raise ToolError("the window contains no data")
    index = pd.DatetimeIndex(frame.index)
    keep = pd.Series(True, index=frame.index)
    if start:
        keep &= index >= pd.Timestamp(start, tz="UTC")
    if end:
        stamp = pd.Timestamp(end, tz="UTC")
        if stamp == stamp.normalize() and len(str(end)) <= 10:
            stamp = stamp + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        keep &= index <= stamp
    window = frame.loc[keep]
    if window.empty:
        raise ToolError(
            f"no data between {start or 'the start of the record'} and "
            f"{end or 'the end of the record'}"
        )
    return window
