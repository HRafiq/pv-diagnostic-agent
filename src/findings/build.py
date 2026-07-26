"""Turn a diagnostic's verdict into a `Finding`, for either engine.

One function for both, so the rules baseline and the agent produce findings the
store cannot tell apart. That matters more than it looks: if the two engines
built findings differently, a difference on the Watcher tab would be a
difference in *presentation* and the comparison would be measuring the wrong
thing.

The energy figure is the other reason this is shared. `energy_verified` is false
whenever the cause is unsettled — the `Finding` validator insists on it — and
the number itself comes from a measurement, never from an estimate.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from src.detect.sweep import DeficitSignal
from src.findings.models import CandidateCause, Finding

__all__ = ["VerdictLike", "finding_from_verdict"]


class VerdictLike(Protocol):
    """What both engines' verdicts have in common."""

    category: str | None
    cause: str | None
    settled: bool
    candidate_causes: tuple[str, ...]
    resolving_measurement: str | None
    evidence: dict[str, float]


_ACTIONS = {
    "fault": "Send someone.",
    "recoverable": "Schedule maintenance.",
    "by_design": "Do nothing — this is the plant working as designed.",
    "not_the_plant": "Do nothing to the plant.",
}

# Energy keys a tool might have measured, best first. The first one present is
# used; nothing is estimated, and an absent figure stays zero rather than being
# guessed at from a percentage.
_ENERGY_KEYS = (
    "compute_expected_output.shortfall_kwh",
    "compute_expected_output.worst_day_shortfall_kwh",
)


def energy_at_stake(evidence: dict[str, float]) -> float:
    for key in _ENERGY_KEYS:
        value = evidence.get(key)
        if value is not None and value > 0:
            return float(value)
    return 0.0


def finding_from_verdict(
    verdict: VerdictLike,
    signal: DeficitSignal,
    finding_id: str,
    detected_at: datetime,
    investigation_id: str,
    *,
    confidence: float | None = None,
    summary: str | None = None,
    title: str | None = None,
    recommended_action: str | None = None,
    consequences: dict[str, str] | None = None,
) -> Finding:
    """Build the `Finding`. Raises if the verdict cannot legally become one.

    An unsettled verdict with fewer than two survivors, or with no resolving
    measurement, is refused rather than padded — the validators exist to catch
    exactly that and working around them here would make them decoration.
    """
    known = consequences or {}
    if verdict.settled:
        cause = verdict.cause
        candidates: list[CandidateCause] = []
        resolving = None
        # A deterministic engine has no calibrated confidence to offer, and
        # inventing one would be the most quietly dishonest number in the
        # interface. 0.5 is stated as "the engine committed", nothing more.
        score = confidence if confidence is not None else 0.5
    else:
        cause = None
        score = None
        resolving = verdict.resolving_measurement
        candidates = [
            CandidateCause(
                cause=name,
                consequence_if_true=known.get(
                    name, "unknown — the investigation did not establish it"
                ),
            )
            for name in verdict.candidate_causes
        ]

    category = verdict.category or "not_the_plant"
    return Finding(
        id=finding_id,
        detected_at=detected_at,
        scope=signal.scope,
        title=title or _default_title(verdict, signal),
        summary=summary or _default_summary(verdict, signal),
        category=category,
        lifecycle="new",
        settled=verdict.settled,
        cause=cause,
        confidence=score,
        candidate_causes=candidates,
        resolving_measurement=resolving,
        energy_at_stake_kwh=energy_at_stake(verdict.evidence),
        energy_verified=verdict.settled,
        recommended_action=(
            recommended_action or _ACTIONS.get(category, "Review this manually.")
        ),
        investigation_id=investigation_id,
    )


def _default_title(verdict: VerdictLike, signal: DeficitSignal) -> str:
    if verdict.settled and verdict.cause:
        return verdict.cause.replace("_", " ").capitalize()
    return "Two causes remain open"


def _default_summary(verdict: VerdictLike, signal: DeficitSignal) -> str:
    opening = (
        f"The {signal.detector.replace('_', ' ')} check fired on "
        f"{signal.start} to {signal.end}."
    )
    if verdict.settled and verdict.cause:
        return (
            f"{opening} The investigation settled on {verdict.cause.replace('_', ' ')}."
        )
    remaining = " or ".join(c.replace("_", " ") for c in verdict.candidate_causes)
    return f"{opening} The measurements could not separate {remaining}."


def as_dict(finding: Finding) -> dict[str, Any]:
    return finding.model_dump(mode="json")
