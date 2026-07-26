"""Rules engine against agent, on the same cases, scored by the same code.

The comparison is the deliverable, not the agent's number. Three properties
make it worth reading:

**Both engines take the same measurements.** The rules baseline calls the same
eighteen tools through the same registry. A gap between them is therefore a gap
in *reasoning*, not in measurement quality, which is the only variable worth
isolating.

**It is designed to be losable.** If the rules engine wins outright, that is a
more credible finding than "I built an agent", and it gets published either
way. Nothing here privileges one engine: the same `aggregate()` scores both, and
the summary prints the sign of the difference without editorialising.

**The headline is not overall accuracy.** A detector that alarms on any deficit
scores well on clean faults and is useless in the field, so the false-alarm rate
on look-alikes and the correct-abstention rate carry as much weight as macro-F1.
The rules engine cannot abstain at all — structurally, not by tuning — and that
column is where the two should differ most if the agent is worth its cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from eval.metrics import CaseScore, EvaluationReport, Prediction

__all__ = ["Comparison", "compare", "comparison_table"]

# The numbers that decide whether the agent earns its cost, in the order they
# should be read. Overall accuracy is deliberately not first.
HEADLINE = (
    ("false_alarm_rate", "false alarms on look-alikes", "lower"),
    ("correct_abstention_rate", "correct 'not enough evidence'", "higher"),
    ("missed_fault_rate", "missed real faults", "lower"),
    ("macro_f1", "overall accuracy", "higher"),
    ("category_accuracy", "category accuracy", "higher"),
    ("cause_accuracy", "cause accuracy", "higher"),
)


@dataclass
class Comparison:
    """Two engines, per split, plus the per-case disagreements."""

    reports: dict[str, EvaluationReport] = field(default_factory=dict)
    scores: dict[str, list[CaseScore]] = field(default_factory=dict)
    predictions: dict[str, list[Prediction]] = field(default_factory=dict)

    @property
    def engines(self) -> list[str]:
        return list(self.reports)

    def disagreements(self, split: str | None = None) -> pd.DataFrame:
        """Cases where the two engines answered differently.

        The most useful output in the file. An aggregate difference of a few
        points says almost nothing about *why*; a list of the cases that moved,
        with what each engine said, says all of it.
        """
        if len(self.engines) < 2:
            return pd.DataFrame()
        first, second = self.engines[0], self.engines[1]
        by_case = {
            engine: {s.case_id: s for s in scores}
            for engine, scores in self.scores.items()
        }
        rows: list[dict[str, Any]] = []
        for case_id, left in by_case[first].items():
            right = by_case[second].get(case_id)
            if right is None or (split and left.split != split):
                continue
            same = (
                left.predicted_settled == right.predicted_settled
                and left.predicted_cause == right.predicted_cause
            )
            if same:
                continue
            rows.append(
                {
                    "case": case_id,
                    "split": left.split,
                    "truth": (
                        left.truth_cause
                        if left.truth_settled
                        else "not enough evidence"
                    ),
                    first: _answer(left),
                    second: _answer(right),
                    "winner": _winner(left, right, first, second),
                }
            )
        return pd.DataFrame(rows)

    def summary(self, split: str = "heldback") -> pd.DataFrame:
        """The headline table for one split."""
        rows: list[dict[str, Any]] = []
        for key, label, direction in HEADLINE:
            row: dict[str, Any] = {"metric": label, "better": direction}
            values: list[float] = []
            for engine in self.engines:
                raw = self.reports[engine].by_split.get(split, {}).get(key)
                row[engine] = None if raw is None else round(float(raw), 4)
                if raw is not None:
                    values.append(float(raw))
            if len(values) == len(self.engines) and len(values) == 2:
                delta = values[1] - values[0]
                row["difference"] = round(delta, 4)
            rows.append(row)
        return pd.DataFrame(rows)

    def agency(self) -> pd.DataFrame:
        rows = []
        for engine in self.engines:
            rows.append({"engine": engine, **self.reports[engine].agency})
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engines": self.engines,
            "reports": {e: r.to_dict() for e, r in self.reports.items()},
            "disagreements": self.disagreements().to_dict(orient="records"),
        }


def _answer(score: CaseScore) -> str:
    if not score.predicted_settled:
        return "not enough evidence"
    return score.predicted_cause or "(no cause)"


def _winner(left: CaseScore, right: CaseScore, first: str, second: str) -> str:
    """Which engine got this case right. Neither, one, or both.

    Cause accuracy, not category, because two causes in the same category can
    lead to opposite actions — "wash the array" and "wipe the sensor" are both
    interventions and only one of them is on the plant.
    """
    if left.cause_correct and right.cause_correct:
        return "both"
    if left.cause_correct:
        return first
    if right.cause_correct:
        return second
    return "neither"


def compare(
    results: dict[str, tuple[list[CaseScore], list[Prediction], EvaluationReport]],
) -> Comparison:
    """Assemble a comparison from each engine's already-scored run."""
    out = Comparison()
    for engine, (scores, predictions, report) in results.items():
        out.scores[engine] = scores
        out.predictions[engine] = predictions
        out.reports[engine] = report
    return out


def comparison_table(comparison: Comparison, split: str = "heldback") -> str:
    """Render the comparison for a terminal, in plain language."""
    lines = [f"\n=== {' vs '.join(comparison.engines)} — {split} split ===\n"]
    lines.append(comparison.summary(split).to_string(index=False))

    disagreements = comparison.disagreements(split)
    if not disagreements.empty:
        counts = disagreements["winner"].value_counts().to_dict()
        lines.append(f"\n  {len(disagreements)} cases answered differently:")
        for engine in [*comparison.engines, "both", "neither"]:
            if engine in counts:
                lines.append(f"    {engine} right on {counts[engine]}")
        lines.append("")
        lines.append(disagreements.head(25).to_string(index=False))
    else:
        lines.append("\n  the two engines agreed on every case")

    lines.append("\n  agency")
    lines.append(comparison.agency().to_string(index=False))
    return "\n".join(lines)
