"""The planner: enumerate candidate causes and open a line of enquiry.

The output is constrained to at least two hypotheses. That is not a stylistic
preference — a planner that emits one candidate has skipped the differential and
turned the rest of the run into confirmation, and every downstream agency metric
would then be measuring a pipeline.

The plan is recorded because the *departures from it* are the agency signal. A
tool called outside `planned_tools` is marked unplanned on the tape and counted
in the eval harness (§5.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agent.llm import LLMClient, LLMResponse
from src.agent.nodes.prompts import (
    PLANNER_SYSTEM,
    evidence_digest,
    hypothesis_digest,
    tool_catalogue_text,
)
from src.agent.state import AgentState, Hypothesis
from src.tools import ToolResult, tool_names

__all__ = ["PLAN_SCHEMA", "PlanOutcome", "plan"]


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scope", "opening_reasoning", "hypotheses", "planned_tools"],
    "properties": {
        "scope": {
            "type": "string",
            "description": "What part of the plant this is about, in plain words.",
        },
        "opening_reasoning": {
            "type": "string",
            "description": (
                "Two or three sentences on why these candidates, in this order."
            ),
        },
        "hypotheses": {
            "type": "array",
            "minItems": 2,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "cause",
                    "consequence_if_true",
                    "discriminating_measurements",
                ],
                "properties": {
                    "id": {"type": "string", "description": "H1, H2, ..."},
                    "cause": {"type": "string"},
                    "consequence_if_true": {
                        "type": "string",
                        "description": "What acting on this costs if it is true.",
                    },
                    "discriminating_measurements": {
                        "type": "array",
                        "items": {"type": "string", "enum": tool_names()},
                    },
                },
            },
        },
        "planned_tools": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "enum": tool_names()},
            "description": "Opening sequence, most discriminating first.",
        },
    },
}


@dataclass(frozen=True)
class PlanOutcome:
    hypotheses: list[Hypothesis]
    planned_tools: list[str]
    reasoning: str
    scope: str
    response: LLMResponse


def plan(
    client: LLMClient,
    state: AgentState,
    brief: str,
    results: list[ToolResult] | None = None,
    errors: list[str] | None = None,
    revision_request: str | None = None,
) -> PlanOutcome:
    """Produce (or revise) the plan.

    On the first cycle `results` is empty and this is an opening plan. On a
    later cycle the critic's `revision_request` is included and the planner sees
    everything measured so far, so replanning is genuinely informed rather than
    a retry of the same opening.
    """
    parts = [brief, "", "TOOLS AVAILABLE", tool_catalogue_text()]
    if results or errors:
        parts += [
            "",
            "ALREADY MEASURED",
            evidence_digest(results or [], errors or []),
            "",
            "CANDIDATES SO FAR",
            hypothesis_digest(state),
        ]
    if revision_request:
        parts += [
            "",
            "THE REVIEWER SENT THE PREVIOUS ANSWER BACK",
            revision_request,
            "",
            "Revise the candidates and the plan to address this specifically.",
        ]

    response = client.complete(
        node="planner",
        system=PLANNER_SYSTEM,
        user="\n".join(parts),
        schema=PLAN_SCHEMA,
    )
    payload = response.parsed or {}

    hypotheses = [
        Hypothesis(
            id=str(item.get("id") or f"H{n}"),
            cause=str(item.get("cause", "")),
            consequence_if_true=str(item.get("consequence_if_true", "")),
            discriminating_measurements=[
                t
                for t in item.get("discriminating_measurements", [])
                if t in tool_names()
            ],
        )
        for n, item in enumerate(payload.get("hypotheses", []), start=1)
    ]
    # Unknown tool names are dropped rather than passed through: a plan naming a
    # tool that does not exist would inflate the unplanned rate later, because
    # every real call would score as a departure from it.
    planned = [t for t in payload.get("planned_tools", []) if t in tool_names()]

    return PlanOutcome(
        hypotheses=hypotheses,
        planned_tools=planned,
        reasoning=str(payload.get("opening_reasoning", "")),
        scope=str(payload.get("scope", state.scope)),
        response=response,
    )
