"""`TraceStep` (handoff §4) — the record of what the agent actually did.

Traces are the dashboard's only data source; there is no separate logging path.
The Investigate tape, the possible-causes ledger, the cost panel, and every
agency metric in eval §5.5 are all views over this one stream.

`was_planned` is the field that matters most. It is what makes agency visible in
the interface (an "unplanned" entry on the tape) and measurable in the harness
(the unplanned-measurement rate — the sharpest single indicator of whether the
planner is reasoning or just sequencing). A pipeline can never produce one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["StepKind", "TraceStep"]

StepKind = Literal["plan", "tool", "adaptive", "retrieval", "critic", "answer"]


class TraceStep(BaseModel):
    """One entry on the investigation tape."""

    model_config = ConfigDict(extra="forbid")

    kind: StepKind
    node: str = Field(..., description="Which node emitted this, e.g. 'planner'")
    args: dict[str, object] | None = None
    result: str | None = None

    was_planned: bool = Field(
        ...,
        description="False renders as 'unplanned' on the tape.",
    )
    reason_for_choosing: str | None = Field(
        default=None,
        description="Required when was_planned is False.",
    )
    excludes: list[str] = Field(
        default_factory=list,
        description="Hypothesis ids ruled out by this step.",
    )

    tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: int = Field(default=0, ge=0)

    # Not in the §4 contract, but a JSONL stream is unreadable without them.
    step_index: int = Field(default=0, ge=0)
    timestamp: datetime | None = Field(
        default=None,
        description="Simulated time from the injected clock — never wall clock.",
    )

    @model_validator(mode="after")
    def _unplanned_steps_must_justify_themselves(self) -> TraceStep:
        if not self.was_planned and not self.reason_for_choosing:
            raise ValueError(
                "an unplanned step must state why it was chosen; that reason is "
                "the evidence of agency, and without it the step is just noise"
            )
        if self.kind == "tool" and self.node == "":
            raise ValueError("a tool step must name the tool it ran")
        return self
