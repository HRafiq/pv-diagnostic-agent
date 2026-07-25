"""Scoring — built so that it can fail.

Two design choices matter more than the arithmetic:

**The false-alarm rate on look-alikes is the headline number**, not overall
accuracy. A detector that alarms on any deficit scores well on clean faults and
is useless in the field; only the look-alike column exposes that.

**`not_enough_evidence` is scored as a correct answer** on the unresolvable
cases, and as a *wrong* one elsewhere. An agent that abstains on everything must
score badly, or abstention becomes a way to dodge the evaluation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

__all__ = ["CaseScore", "EvaluationReport", "Prediction", "aggregate", "score_case"]

ABSTAIN = "not_enough_evidence"


@dataclass(frozen=True)
class Prediction:
    """What a diagnostic (rules engine or agent) concluded about one case."""

    case_id: str
    category: str | None
    cause: str | None
    settled: bool
    candidate_causes: tuple[str, ...] = ()
    resolving_measurement: str | None = None
    tools_called: tuple[str, ...] = ()
    unplanned_tools: tuple[str, ...] = ()
    critic_cycles: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


@dataclass(frozen=True)
class CaseScore:
    """How one prediction did against ground truth."""

    case_id: str
    split: str
    is_lookalike: bool
    truth_settled: bool
    truth_category: str | None
    truth_cause: str | None
    predicted_category: str | None
    predicted_cause: str | None
    predicted_settled: bool

    @property
    def category_correct(self) -> bool:
        # Settledness must match first. A prediction that abstains has not
        # named a category, so whatever it left in that field cannot earn
        # credit — otherwise abstaining on a fault would score as correctly
        # identifying it.
        if self.truth_settled != self.predicted_settled:
            return False
        if not self.truth_settled:
            return True  # correctly abstained
        return self.predicted_category == self.truth_category

    @property
    def cause_correct(self) -> bool:
        if self.truth_settled != self.predicted_settled:
            return False
        if not self.truth_settled:
            return True
        return self.predicted_cause == self.truth_cause

    @property
    def false_alarm(self) -> bool:
        """Called a fault on something that is not one.

        Only look-alikes can produce one. This is the number the field cares
        about: crews dispatched to healthy plant.
        """
        return (
            self.is_lookalike
            and self.predicted_settled
            and (self.predicted_category in ("fault", "recoverable"))
        )

    @property
    def missed_fault(self) -> bool:
        """A real fault called benign, or abstained on."""
        return self.truth_category == "fault" and (
            not self.predicted_settled
            or self.predicted_category not in ("fault", "recoverable")
        )

    @property
    def correct_abstention(self) -> bool:
        return not self.truth_settled and not self.predicted_settled

    @property
    def wrong_abstention(self) -> bool:
        """Abstained where the answer was actually available."""
        return self.truth_settled and not self.predicted_settled


def score_case(case: Any, prediction: Prediction) -> CaseScore:
    return CaseScore(
        case_id=case.id,
        split=case.split,
        is_lookalike=case.is_lookalike,
        truth_settled=case.settled,
        truth_category=case.expected_category,
        truth_cause=case.expected_cause,
        predicted_category=prediction.category,
        predicted_cause=prediction.cause,
        predicted_settled=prediction.settled,
    )


def _macro_f1(scores: list[CaseScore]) -> float:
    """Macro-averaged F1 over categories, abstention treated as its own class.

    Macro rather than micro so a rare class cannot be ignored for free: with
    only a handful of unresolvable cases, micro-averaging would let a model
    score well while never once abstaining.
    """
    settled = [s for s in scores]
    labels = {(s.truth_category if s.truth_settled else ABSTAIN) for s in settled}
    f1s: list[float] = []
    for label in sorted(x for x in labels if x is not None):
        tp = fp = fn = 0
        for s in settled:
            truth = s.truth_category if s.truth_settled else ABSTAIN
            pred = s.predicted_category if s.predicted_settled else ABSTAIN
            if truth == label and pred == label:
                tp += 1
            elif pred == label:
                fp += 1
            elif truth == label:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(f1s) / len(f1s) if f1s else 0.0


@dataclass
class EvaluationReport:
    """Per-split results. Tuning and held-back are never merged."""

    by_split: dict[str, dict[str, Any]] = field(default_factory=dict)
    agency: dict[str, Any] = field(default_factory=dict)

    @property
    def overfitting_gap(self) -> float | None:
        """Tuning accuracy minus held-back accuracy. Over 0.10 means overfitting."""
        tuning = self.by_split.get("tuning", {}).get("macro_f1")
        held = self.by_split.get("heldback", {}).get("macro_f1")
        if tuning is None or held is None:
            return None
        return round(float(tuning) - float(held), 4)

    def meets_v1_targets(self) -> dict[str, bool]:
        """The §5.6 success criteria, evaluated on the held-back split."""
        held = self.by_split.get("heldback", {})
        gap = self.overfitting_gap
        far = held.get("false_alarm_rate")
        abstention = held.get("correct_abstention_rate")
        return {
            "macro_f1_at_least_0.75": float(held.get("macro_f1") or 0.0) >= 0.75,
            # A split with no look-alikes cannot demonstrate a low false-alarm
            # rate, so it fails rather than passing by absence of evidence.
            "false_alarm_rate_at_most_0.15": far is not None and float(far) <= 0.15,
            "correct_abstention_at_least_0.70": abstention is not None
            and float(abstention) >= 0.70,
            "overfitting_gap_at_most_0.10": gap is not None and abs(gap) <= 0.10,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_split": self.by_split,
            "agency": self.agency,
            "overfitting_gap": self.overfitting_gap,
            "v1_targets": self.meets_v1_targets(),
        }


def aggregate(
    scores: list[CaseScore],
    predictions: list[Prediction] | None = None,
) -> EvaluationReport:
    """Roll per-case scores up into a report."""
    report = EvaluationReport()

    for split in sorted({s.split for s in scores}):
        subset = [s for s in scores if s.split == split]
        lookalikes = [s for s in subset if s.is_lookalike]
        unresolvable = [s for s in subset if not s.truth_settled]
        real_faults = [s for s in subset if s.truth_category == "fault"]

        report.by_split[split] = {
            "cases": len(subset),
            "macro_f1": round(_macro_f1(subset), 4),
            "category_accuracy": round(
                sum(s.category_correct for s in subset) / len(subset), 4
            ),
            "cause_accuracy": round(
                sum(s.cause_correct for s in subset) / len(subset), 4
            ),
            # The headline. None when the split has no look-alikes, rather than
            # a flattering zero.
            "false_alarm_rate": (
                round(sum(s.false_alarm for s in lookalikes) / len(lookalikes), 4)
                if lookalikes
                else None
            ),
            "look_alike_cases": len(lookalikes),
            "missed_fault_rate": (
                round(sum(s.missed_fault for s in real_faults) / len(real_faults), 4)
                if real_faults
                else None
            ),
            "correct_abstention_rate": (
                round(
                    sum(s.correct_abstention for s in unresolvable) / len(unresolvable),
                    4,
                )
                if unresolvable
                else None
            ),
            "unresolvable_cases": len(unresolvable),
            "wrong_abstentions": sum(s.wrong_abstention for s in subset),
        }

    if predictions:
        report.agency = measure_agency(predictions)
    return report


def measure_agency(predictions: list[Prediction]) -> dict[str, Any]:
    """The §5.5 agency metrics — measured, never asserted.

    The unplanned-measurement rate is the sharpest single indicator. Near zero
    means the planner is a sequencer and the "agent" is a pipeline that narrates.
    """
    if not predictions:
        return {}

    trajectories = Counter(tuple(p.tools_called) for p in predictions)
    with_unplanned = sum(1 for p in predictions if p.unplanned_tools)
    cycles = Counter(p.critic_cycles for p in predictions)
    costs = sorted(p.cost_usd for p in predictions)
    latencies = sorted(p.latency_ms for p in predictions)

    def median(values: list[Any]) -> Any:
        return values[len(values) // 2] if values else 0

    return {
        "runs": len(predictions),
        # > 15 of 64 in the brief's target; below that it is a pipeline.
        "distinct_tool_trajectories": len(trajectories),
        "unplanned_measurement_rate": round(with_unplanned / len(predictions), 4),
        "iteration_distribution": dict(sorted(cycles.items())),
        "self_initiated_abstentions": sum(1 for p in predictions if not p.settled),
        "median_cost_usd": round(median(costs), 4),
        "median_latency_ms": median(latencies),
        "mean_tools_per_run": round(
            sum(len(p.tools_called) for p in predictions) / len(predictions), 2
        ),
    }


def confusion(scores: list[CaseScore]) -> pd.DataFrame:
    """Truth against prediction, with abstention as its own row and column."""
    rows: list[dict[str, Any]] = []
    for s in scores:
        rows.append(
            {
                "truth": s.truth_category if s.truth_settled else ABSTAIN,
                "predicted": s.predicted_category if s.predicted_settled else ABSTAIN,
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return pd.crosstab(frame["truth"], frame["predicted"])
