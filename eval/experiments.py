"""The two §5.4 experiments, and the N-run reporting the headline needs.

Everything here exists because a single number from a non-deterministic system
is not a result. `CLAUDE.md` scopes determinism: physics, tools, injection and
retrieval are bitwise reproducible; the planner, router, synthesiser and critic
are not. So headline metrics are reported as **mean ± spread over N fresh
runs**, never as one figure, and this module is what produces them.

Two experiments:

* **ablations** — the agent against itself with one layer removed. Review off,
  knowledge off, both off. Each answers "does this layer earn its cost?", and
  an ablation that cannot be run is a claim rather than a measurement.
* **cost/quality** — the `default` and `quality` model profiles on the same
  cases. The interesting outcome is a frontier, not a winner.

Both need an API key. Without one they raise rather than returning a shape that
looks like a result.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from eval.metrics import CaseScore, Prediction, aggregate

__all__ = [
    "ABLATIONS",
    "ExperimentRun",
    "RepeatedResult",
    "repeat",
    "summarise_runs",
]

# The metrics reported with a spread. Chosen because these are the ones quoted
# in a summary, and a quoted number without a spread invites being read as
# precise.
HEADLINE_METRICS = (
    "macro_f1",
    "category_accuracy",
    "cause_accuracy",
    "false_alarm_rate",
    "correct_abstention_rate",
    "missed_fault_rate",
)

# name -> kwargs for `run_agent_engine`. Each removes exactly one thing.
ABLATIONS: dict[str, dict[str, bool]] = {
    "full": {"review": True, "use_knowledge": True},
    "no review": {"review": False, "use_knowledge": True},
    "no knowledge": {"review": True, "use_knowledge": False},
    "neither": {"review": False, "use_knowledge": False},
}


@dataclass(frozen=True)
class ExperimentRun:
    """One complete pass over the cases under one configuration."""

    label: str
    run_index: int
    scores: list[CaseScore]
    predictions: list[Prediction]

    def metrics(self, split: str) -> dict[str, float | None]:
        stats = aggregate(self.scores, self.predictions).by_split.get(split, {})
        return {m: stats.get(m) for m in HEADLINE_METRICS}

    def agency(self) -> dict[str, Any]:
        return aggregate(self.scores, self.predictions).agency


@dataclass
class RepeatedResult:
    """N runs of one configuration, reported as mean ± spread."""

    label: str
    split: str
    runs: int = 0
    mean: dict[str, float] = field(default_factory=dict)
    spread: dict[str, float] = field(default_factory=dict)
    agency: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"configuration": self.label, "runs": self.runs}
        for metric in HEADLINE_METRICS:
            if metric in self.mean:
                spread = self.spread.get(metric, 0.0)
                row[metric] = f"{self.mean[metric]:.3f} ± {spread:.3f}"
        row["distinct_trajectories"] = self.agency.get("distinct_tool_trajectories")
        row["unplanned_rate"] = self.agency.get("unplanned_measurement_rate")
        row["median_cost_usd"] = self.agency.get("median_cost_usd")
        return row


def summarise_runs(runs: list[ExperimentRun], split: str) -> RepeatedResult:
    """Fold N runs into a mean and a spread.

    Spread is the *sample* standard deviation, and it is zero for N=1 rather
    than undefined — but a single run is reported as `runs: 1` so a reader can
    see that a ± 0.000 means "measured once", not "perfectly stable".
    """
    if not runs:
        return RepeatedResult(label="(none)", split=split)

    result = RepeatedResult(label=runs[0].label, split=split, runs=len(runs))
    by_metric: dict[str, list[float]] = {m: [] for m in HEADLINE_METRICS}
    for run in runs:
        for metric, value in run.metrics(split).items():
            if value is not None:
                by_metric[metric].append(float(value))

    for metric, values in by_metric.items():
        if not values:
            continue
        result.mean[metric] = round(statistics.fmean(values), 4)
        result.spread[metric] = round(
            statistics.stdev(values) if len(values) > 1 else 0.0, 4
        )

    # Agency is reported from the runs pooled, because trajectory *diversity*
    # across runs is itself part of what the metric is measuring.
    pooled = [p for run in runs for p in run.predictions]
    result.agency = aggregate([s for run in runs for s in run.scores], pooled).agency
    return result


def repeat(
    label: str,
    runner: Any,
    cases: list[Any],
    n: int = 3,
    **kwargs: Any,
) -> list[ExperimentRun]:
    """Run one configuration `n` times.

    `n=3` is the brief's number and it is small. Three runs give a spread that
    is indicative, not a confidence interval, and this docstring is the only
    place that will say so — the tables just print it.
    """
    runs: list[ExperimentRun] = []
    for index in range(n):
        scores, predictions = runner(cases, **kwargs)
        runs.append(ExperimentRun(label, index, scores, predictions))
    return runs


def ablation_table(results: list[RepeatedResult]) -> pd.DataFrame:
    """The ablation comparison. Read `correct_abstention_rate` first."""
    return pd.DataFrame([r.as_row() for r in results])


def frontier_table(results: list[RepeatedResult]) -> pd.DataFrame:
    """Cost against accuracy for each model profile.

    Deliberately not sorted by accuracy. The useful output is a frontier — what
    the extra money bought — and sorting by the winner hides the shape of it.
    """
    rows = [r.as_row() for r in results]
    return pd.DataFrame(rows)
