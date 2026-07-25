"""The `Finding` contract (handoff §4), with its invariants enforced.

The `settled` flag is load-bearing. An unsettled finding that displays a single
cause and a confidence number is the exact bug that dispatches a wash crew to a
clean array: the investigation could not separate array soiling from sensor
soiling, but the card said "soiling, confidence 0.78".

So the rule is not documentation, it is a validator. A `Finding` that violates
it cannot be constructed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "CandidateCause",
    "Category",
    "Finding",
    "Lifecycle",
]

# fault          — broken equipment. Send someone.
# recoverable    — real loss that maintenance recovers (soiling). Schedule it.
# by_design      — clipping, temperature derating. Do nothing; it is not a fault.
# not_the_plant  — weather, curtailment, a bad sensor, a telemetry gap.
Category = Literal["fault", "recoverable", "by_design", "not_the_plant"]

# Without lifecycle you get the same three findings shouted at you every
# morning until you stop reading them.
Lifecycle = Literal["new", "ongoing", "resolved", "acknowledged", "suppressed"]


class CandidateCause(BaseModel):
    """A cause still standing after every available measurement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cause: str
    consequence_if_true: str = Field(
        ...,
        description=(
            "What it means operationally if this one is true — "
            "'real loss, wash needed' vs 'no loss, wipe the sensor'. This is "
            "the field that makes an unsettled finding actionable instead of "
            "just uncertain."
        ),
    )


class Finding(BaseModel):
    """One thing the Watcher noticed, ranked by energy at stake."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="e.g. 'F-118'")
    detected_at: datetime
    scope: str = Field(..., description="e.g. 'INV-03 / MPPT-2'")
    title: str
    summary: str = Field(..., description="1-3 sentences, plain language, no jargon")
    category: Category
    lifecycle: Lifecycle

    settled: bool = Field(
        ...,
        description="Did the investigation reach a single cause?",
    )
    cause: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate_causes: list[CandidateCause] = Field(default_factory=list)
    resolving_measurement: str | None = Field(
        default=None,
        description=(
            "The cheap test that would separate the surviving causes. Required "
            "when unsettled — naming it is what makes an abstention useful."
        ),
    )

    energy_at_stake_kwh: float = Field(..., ge=0.0)
    energy_verified: bool = Field(
        ...,
        description="False whenever the cause is not settled.",
    )
    recommended_action: str
    investigation_id: str = Field(..., description="Links to the JSONL trace.")

    @model_validator(mode="after")
    def _enforce_settled_contract(self) -> Finding:
        if self.settled:
            if not self.cause:
                raise ValueError("a settled finding must name its cause")
            if self.confidence is None:
                raise ValueError("a settled finding must carry a confidence")
            if self.candidate_causes:
                raise ValueError(
                    "a settled finding must not carry candidate causes; it "
                    "reached one answer"
                )
            if self.resolving_measurement is not None:
                raise ValueError(
                    "a settled finding has nothing left to resolve, so it must "
                    "not name a resolving measurement"
                )
        else:
            if self.cause is not None:
                raise ValueError(
                    "an unsettled finding must not display a single cause — "
                    "this is the bug that sends a wash crew to a clean array"
                )
            if self.confidence is not None:
                raise ValueError(
                    "an unsettled finding must not display a confidence number"
                )
            if len(self.candidate_causes) < 2:
                raise ValueError(
                    "an unsettled finding must list at least two surviving "
                    "causes; if only one survived, it is settled"
                )
            if not self.resolving_measurement:
                raise ValueError(
                    "an unsettled finding must name the measurement that would "
                    "resolve it — an abstention without a next step is useless"
                )
            if self.energy_verified:
                raise ValueError(
                    "energy cannot be verified while the cause is unsettled; "
                    "the figure must be labelled unverified in the interface"
                )
        return self
