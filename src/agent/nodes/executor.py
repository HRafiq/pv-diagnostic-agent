"""The executor: run one tool and write the tape entry for it.

This is where `was_planned` is decided, and it is decided here rather than asked
of the model. Two calls count as planned:

* the tool is named in `planned_tools`, **and**
* it has not already been run in this investigation.

The second clause is the interesting one. Re-running a tool the plan already
spent — usually over a narrowed window because something in an earlier result
pointed at a date — is not the plan being followed, it is the plan being
extended by evidence. Counting it as planned would hide exactly the behaviour
the agency metrics exist to detect.

A failed tool is written to the tape too. A measurement that could not be taken
is information the agent needs; silently dropping it would let the synthesiser
believe a channel was checked when it was not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agent.llm import LLMResponse
from src.agent.nodes.router import RouterDecision
from src.agent.state import AgentState
from src.tools import ToolContext, ToolError, ToolResult, run_tool
from src.trace.models import TraceStep

__all__ = ["Execution", "execute", "narrow_args"]


def narrow_args(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Keep only the arguments this tool accepts.

    The router chooses from one shared argument vocabulary — the union of every
    tool's fields — because structured outputs cannot express an open object
    whose shape depends on another field (DECISION 0047). Tool argument models
    are `extra="forbid"`, so a field belonging to a different tool is a hard
    validation error, and the router filling one in turned a good measurement
    into "could not take this measurement" on case after case.

    Narrowing here rather than loosening `run_tool` is deliberate: `run_tool`
    stays strict, so a genuinely wrong argument passed from code still fails
    loudly. What is dropped here is only the cost of sharing one vocabulary
    across eighteen tools.
    """
    from src.tools import REGISTRY

    spec = REGISTRY.get(tool)
    if spec is None:
        return dict(args)
    accepted = set(spec.args_model.model_fields)
    return {key: value for key, value in args.items() if key in accepted}


@dataclass(frozen=True)
class Execution:
    """The outcome of one tool call."""

    step: TraceStep
    result: ToolResult | None
    error: str | None
    was_planned: bool

    @property
    def ok(self) -> bool:
        return self.result is not None


def execute(
    ctx: ToolContext,
    state: AgentState,
    decision: RouterDecision,
    router_response: LLMResponse | None = None,
) -> Execution:
    """Run the tool the router chose and produce its trace entry.

    `router_response` is folded into this step's cost. The router's call is what
    produced this measurement, so attributing it here keeps the tape's cost
    column adding up to the investigation total with nothing unaccounted for.
    """
    was_planned = (
        decision.tool in state.planned_tools and decision.tool not in state.tools_called
    )

    result: ToolResult | None = None
    error: str | None = None
    try:
        result = run_tool(decision.tool, ctx, narrow_args(decision.tool, decision.args))
        summary = result.summary
    except ToolError as exc:
        error = f"{decision.tool}: {exc}"
        summary = f"could not take this measurement — {exc}"
    except Exception as exc:  # a tool bug, not a data problem
        error = f"{decision.tool} raised {type(exc).__name__}: {exc}"
        summary = f"the measurement failed unexpectedly — {exc}"

    reason = decision.reason.strip()
    if not was_planned and not reason:
        # The trace model refuses an unplanned step with no reason, and it is
        # right to: an unexplained departure is indistinguishable from noise.
        # Recording that the router gave none is more honest than inventing one.
        reason = "the router departed from the plan without stating a reason"

    step = TraceStep(
        kind="tool" if was_planned else "adaptive",
        node=decision.tool,
        args=dict(decision.args),
        result=summary,
        was_planned=was_planned,
        reason_for_choosing=reason or None,
        excludes=list(decision.excludes),
        tokens=router_response.tokens if router_response else 0,
        cost_usd=router_response.cost_usd if router_response else 0.0,
        latency_ms=router_response.latency_ms if router_response else 0,
    )
    return Execution(step=step, result=result, error=error, was_planned=was_planned)
