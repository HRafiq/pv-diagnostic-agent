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


# Structured outputs reject `additionalProperties: true`, so the tool-argument
# object cannot be open. It is built from the tool registry instead — which is
# strictly better than the open object it replaces: the router now sees the real
# argument vocabulary rather than being free to invent a name that `run_tool`
# would reject downstream.
#
# Every field is nullable and every field is listed in `required`. That is the
# shape that satisfies the strictest reading of the API's object rules, and
# nulls are dropped in `_clean_args` before the tool ever sees them, so "null"
# and "not supplied" mean the same thing.
_JSON_TYPES: dict[type, str] = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
}


def _args_schema() -> dict[str, Any]:
    from src.tools import REGISTRY

    properties: dict[str, Any] = {}
    for spec in REGISTRY.values():
        for name, info in spec.args_model.model_fields.items():
            if name in properties:
                continue
            annotation = info.annotation
            base = next(
                (
                    python
                    for python in _JSON_TYPES
                    if python is annotation or python.__name__ in str(annotation)
                ),
                str,
            )
            properties[name] = {
                "anyOf": [{"type": _JSON_TYPES[base]}, {"type": "null"}],
                "description": info.description or f"{name} for tools that take it.",
            }

    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(properties),
        "properties": dict(sorted(properties.items())),
        "description": (
            "Arguments for the tool. Set every field you do not need to null; "
            "null and omitted mean the same thing, and a tool with all-null "
            "arguments runs on its defaults over the window."
        ),
    }


ROUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "action",
        "tool",
        "args",
        "look_up_causes",
        "reason_for_choosing",
        "excludes",
    ],
    "properties": {
        "action": {
            "type": "string",
            "enum": ["call_tool", "look_up", "stop"],
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
        "args": _args_schema(),
        "reason_for_choosing": {
            "type": "string",
            "description": (
                "One specific sentence: what this would separate, and why now."
            ),
        },
        "look_up_causes": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "For 'look_up': the two causes to fetch the distinguishing "
                "test for. Empty otherwise."
            ),
        },
        "excludes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Hypothesis ids the previous measurement ruled out, if any.",
        },
    },
}


def _clean_args(raw: Any) -> dict[str, Any]:
    """Drop the nulls the schema requires the model to send.

    Every argument is `required` and nullable so the object satisfies the
    structured-output rules; a null means "not supplied". Passing them through
    would override a tool's own defaults with None.
    """
    if not isinstance(raw, dict):
        return {}
    return {key: value for key, value in raw.items() if value is not None}


@dataclass(frozen=True)
class RouterDecision:
    action: Literal["call_tool", "look_up", "stop"]
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    excludes: list[str] = field(default_factory=list)
    look_up_causes: list[str] = field(default_factory=list)
    response: LLMResponse | None = None

    @property
    def stops(self) -> bool:
        if self.action == "look_up":
            return not self.look_up_causes
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
    reason = str(payload.get("reason_for_choosing", ""))
    excludes = [str(x) for x in payload.get("excludes", [])]

    if action == "look_up":
        causes = [str(c) for c in payload.get("look_up_causes", []) if str(c).strip()]
        return RouterDecision(
            action="look_up",
            reason=reason,
            excludes=excludes,
            look_up_causes=causes,
            response=response,
        )

    if action != "call_tool" or tool not in tool_names():
        return RouterDecision(
            action="stop", reason=reason, excludes=excludes, response=response
        )

    args = _clean_args(payload.get("args"))
    return RouterDecision(
        action="call_tool",
        tool=tool,
        args=args,
        reason=reason,
        excludes=excludes,
        response=response,
    )
