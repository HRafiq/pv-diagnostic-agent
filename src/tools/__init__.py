"""The tool registry — the agent's entire surface onto the plant.

There is deliberately no `classify_fault`, `diagnose`, or `explain` entry here,
and adding one would end the project: the diagnosis is assembled by the agent
from several measurements, and a tool that returned a fault class would make
every run identical and the agency metrics meaningless (CLAUDE.md).

    from src.tools import REGISTRY, run_tool, catalogue

`catalogue()` is what the planner and router are shown. `run_tool()` is the only
way a tool is invoked, so argument validation and the provenance ledger cannot
be bypassed by calling a function directly.
"""

from __future__ import annotations

from typing import Any

from src.tools.base import (
    ToolArgs,
    ToolContext,
    ToolError,
    ToolResult,
    ToolSpec,
    WindowArgs,
    slice_window,
)
from src.tools.diagnostics import DIAGNOSTIC_SPECS
from src.tools.measurements import SPECS

__all__ = [
    "REGISTRY",
    "ToolArgs",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "ToolSpec",
    "WindowArgs",
    "catalogue",
    "run_tool",
    "slice_window",
    "tool_names",
]

REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec for spec in (*SPECS, *DIAGNOSTIC_SPECS)
}

_FORBIDDEN = ("classify", "diagnose", "identify_fault", "explain")
for _name in REGISTRY:
    if any(word in _name for word in _FORBIDDEN):
        raise RuntimeError(
            f"tool {_name!r} names a classification, not a measurement. Tools "
            "take measurements; the diagnosis is the agent's job (CLAUDE.md)."
        )


def tool_names() -> list[str]:
    return sorted(REGISTRY)


def catalogue() -> list[dict[str, Any]]:
    """What the planner and the router see. Stable order, so prompts cache."""
    return [REGISTRY[name].catalogue_entry() for name in tool_names()]


def run_tool(
    name: str, ctx: ToolContext, args: dict[str, Any] | None = None
) -> ToolResult:
    """Validate arguments and run one tool.

    Raises:
        ToolError: The tool does not exist, its arguments are invalid, or the
            measurement could not be taken. Never returns a zero-valued result
            in place of a failure — a failed measurement that reads as "0.0"
            would be indistinguishable from a real one, and the agent would
            reason on it.
    """
    spec = REGISTRY.get(name)
    if spec is None:
        raise ToolError(
            f"no tool named {name!r}; available: " + ", ".join(tool_names())
        )
    try:
        validated = spec.args_model.model_validate(args or {})
    except Exception as exc:
        raise ToolError(f"{name}: invalid arguments — {exc}") from exc
    result = spec.fn(ctx, validated)
    if result.tool != name:
        raise ToolError(
            f"{name} returned a result labelled {result.tool!r}; the provenance "
            "ledger would attribute its numbers to the wrong measurement"
        )
    return result
