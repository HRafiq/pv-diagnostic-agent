"""Every tool, over windows too small or too flat to measure anything.

A tool has exactly two honest outcomes. It returns a `ToolResult` whose ledger
holds real numbers, or it raises `ToolError` saying what could not be measured.
There is no third one — and `ToolResult`'s validator agrees, rejecting any
non-finite value in a ledger.

The failure this file exists for is subtler than a crash. `pandas.Series.std()`
uses ddof=1, so on a single element it returns NaN rather than raising. A window
containing one day therefore produced a `day_to_day_spread` of NaN in
`weather_context`, which surfaced as a pydantic ValidationError from inside the
rules engine, sixty cases into an evaluation.

It was invisible for a different reason worth recording: the development
machine's ingest ran four months longer than a later one, so no case window ever
landed near the end of the record. The bug was in the code the whole time and
only one particular download revealed it. Hence this file, which does not wait
for a dataset to be short in the right way — it constructs windows that are.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.tools import REGISTRY, ToolContext, run_tool
from src.tools.base import ToolError
from tests.conftest import context_for, synthetic_frame

TOOL_NAMES = sorted(REGISTRY)


def _degenerate_cases() -> dict[str, tuple[ToolContext, dict[str, str]]]:
    """Contexts and windows that a real record can genuinely produce."""
    healthy = synthetic_frame(days=60, start="2017-04-01")

    cases: dict[str, tuple[ToolContext, dict[str, str]]] = {}

    # One calendar day. Any day-over-day statistic has a single sample, which
    # is where ddof=1 returns NaN instead of raising.
    cases["one_day"] = (
        context_for(healthy),
        {"start": "2017-05-16", "end": "2017-05-16"},
    )

    # Two intervals. Shorter than any averaging period the tools assume.
    cases["two_intervals"] = (
        context_for(healthy),
        {"start": "2017-05-16T12:00:00", "end": "2017-05-16T12:30:00"},
    )

    # The last day in the record, which is the case that actually fired: a
    # window at the tail has nothing after it and may have only one day inside
    # it. A shorter download moves this boundary without changing any code.
    truncated = healthy.loc[: pd.Timestamp("2017-05-16T23:45:00Z")]
    cases["tail_of_record"] = (
        context_for(truncated),
        {"start": "2017-05-16", "end": "2017-05-30"},
    )

    # Night only: no irradiance, no power, nothing to normalise by.
    cases["night_only"] = (
        context_for(healthy),
        {"start": "2017-05-16T00:00:00", "end": "2017-05-16T04:00:00"},
    )

    # Every channel frozen. Correlations and slopes are undefined on a constant
    # series, and 0/0 is NaN rather than an exception.
    flat = healthy.copy()
    for column in flat.columns:
        flat[column] = 1.0
    cases["all_channels_constant"] = (
        context_for(flat),
        {"start": "2017-05-16", "end": "2017-05-30"},
    )

    # Irradiance identically zero while power is not — physically impossible,
    # and exactly what a failed pyranometer looks like.
    dead_sensor = healthy.copy()
    dead_sensor["poa_wm2"] = 0.0
    cases["dead_irradiance_sensor"] = (
        context_for(dead_sensor),
        {"start": "2017-05-16", "end": "2017-05-30"},
    )

    return cases


DEGENERATE = _degenerate_cases()


@pytest.mark.parametrize("tool", TOOL_NAMES)
@pytest.mark.parametrize("case", sorted(DEGENERATE))
def test_a_tool_measures_or_says_it_cannot(tool: str, case: str) -> None:
    ctx, window = DEGENERATE[case]

    try:
        result = run_tool(tool, ctx, dict(window))
    except ToolError:
        # The honest outcome. A tool that cannot measure must say so.
        return

    bad = {k: v for k, v in result.values.items() if not math.isfinite(v)}
    assert not bad, (
        f"{tool} on a {case} window returned a ledger with non-finite values "
        f"{bad}. Raise ToolError, or omit the key and add a caveat — a "
        f"measurement that could not be taken must not look like one that was."
    )

    for name, points in result.series.items():
        rotten = [v for v in points if not math.isfinite(v)]
        assert not rotten, f"{tool} on a {case} window put NaN in series {name!r}"


@pytest.mark.parametrize("tool", TOOL_NAMES)
def test_a_tool_raises_toolerror_on_an_empty_window(tool: str) -> None:
    """A window outside the record entirely. Nothing can be measured, and the
    tool must not invent a zero to fill the gap."""
    ctx = context_for(synthetic_frame(days=60, start="2017-04-01"))

    with pytest.raises(ToolError):
        run_tool(tool, ctx, {"start": "2020-01-01", "end": "2020-01-14"})
