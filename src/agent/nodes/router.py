"""The router: choose the next measurement, or stop.

The one design decision worth defending here is what the router is **not** asked
for. It does not report `was_planned`. That flag is derived by the executor from
the plan the planner already committed to, because a model asked "was this in
your plan?" is being invited to flatter itself, and the unplanned-measurement
rate is the single sharpest agency metric in the evaluation (§5.5). A metric the
subject can self-report is not a measurement.

The router *is* asked for `reason_for_choosing` on every call, not only on
departures. Making it unconditional keeps the reason honest: a field that only
appears when the model has already decided it is going off-plan gets written to
justify the departure rather than to explain the choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.agent.llm import LLMClient, LLMResponse
from src.agent.nodes.prompts import (
    ROUTER_SYSTEM,
    evidence_digest,
    hypothesis_digest,
    tool_catalogue_text,
)
from src.agent.state import AgentState
from src.tools import ToolResult, tool_names

__all__ = ["ROUTE_SCHEMA", "RouterDecision", "route"]


ROUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "tool", "args", "reason_for_choosing", "excludes"],
    "properties": {
        "action": {
            "type": "string",
            "enum": ["call_tool", "stop"],
            "description": (
                "'stop' when one cause stands, or when no remaining tool would "
                "separate the survivors."
            ),
        },
        "tool": {
            "type": "string",
            "enum": [*tool_names(), ""],
            "description": "Empty string when stopping.",
        },
        "args": {
            "type": "object",
            "additionalProperties": True,
            "description": (
                "Arguments for the tool. {} for its defaults over the window."
            ),
        },
        "reason_for_choosing": {
            "type": "string",
            "description": (
                "One specific sentence: what this would separate, and why now."
            ),
        },
        "excludes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Hypothesis ids the previous measurement ruled out, if any.",
        },
    },
}


@dataclass(frozen=True)
class RouterDecision:
    action: Literal["call_tool", "stop"]
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    excludes: list[str] = field(default_factory=list)
    response: LLMResponse | None = None

    @property
    def stops(self) -> bool:
        return self.action == "stop" or not self.tool


def route(
    client: LLMClient,
    state: AgentState,
    brief: str,
    results: list[ToolResult],
    errors: list[str],
    calls_remaining: int,
    knowledge: str = "",
) -> RouterDecision:
    """Pick the next tool call."""
    already = ", ".join(state.tools_called) or "(none)"
    user = "\n".join(
        [
            brief,
            *(["", "WHAT IS KNOWN ABOUT THESE CAUSES", knowledge] if knowledge else []),
            "",
            "TOOLS AVAILABLE",
            tool_catalogue_text(),
            "",
            "THE PLAN",
            ", ".join(state.planned_tools) or "(no plan)",
            "",
            "CANDIDATE CAUSES",
            hypothesis_digest(state),
            "",
            "MEASUREMENTS TAKEN SO FAR",
            evidence_digest(results, errors),
            "",
            "ALREADY CALLED",
            already,
            "",
            f"You may take at most {calls_remaining} more measurements in this "
            "cycle. Stop earlier if nothing further would separate the causes "
            "still standing.",
        ]
    )

    response = client.complete(
        node="router",
        system=ROUTER_SYSTEM,
        user=user,
        schema=ROUTE_SCHEMA,
    )
    payload = response.parsed or {}
    action = payload.get("action", "stop")
    tool = str(payload.get("tool") or "")
    if action != "call_tool" or tool not in tool_names():
        return RouterDecision(
            action="stop",
            reason=str(payload.get("reason_for_choosing", "")),
            excludes=[str(x) for x in payload.get("excludes", [])],
            response=response,
        )

    args = payload.get("args") or {}
    return RouterDecision(
        action="call_tool",
        tool=tool,
        args=dict(args) if isinstance(args, dict) else {},
        reason=str(payload.get("reason_for_choosing", "")),
        excludes=[str(x) for x in payload.get("excludes", [])],
        response=response,
    )
