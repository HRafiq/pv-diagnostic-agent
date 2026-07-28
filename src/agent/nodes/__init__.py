"""The four agent nodes.

Each is a plain function over `(client, state, ...)` returning a typed result.
None of them holds state, mutates `AgentState`, or writes to the trace — the
loop does both, in one place, which is what makes the LangGraph port at step 12
a change of orchestration only and the comparison between the two a real one.
"""

from src.agent.nodes.executor import Execution, execute
from src.agent.nodes.planner import PLAN_SCHEMA, PlanOutcome, plan
from src.agent.nodes.router import ROUTE_SCHEMA, RouterDecision, route
from src.agent.nodes.synthesizer import (
    SYNTHESIS_SCHEMA,
    Synthesis,
    ledger_of,
    synthesize,
    withdraw_commitment,
)

__all__ = [
    "PLAN_SCHEMA",
    "ROUTE_SCHEMA",
    "SYNTHESIS_SCHEMA",
    "Execution",
    "PlanOutcome",
    "RouterDecision",
    "Synthesis",
    "execute",
    "ledger_of",
    "plan",
    "route",
    "synthesize",
    "withdraw_commitment",
]
