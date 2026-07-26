"""The gate on the LangGraph port: both loops must produce the same run.

This is the only test file whose failure means "delete the new code and start
again". The plain loop is the specification; the graph is a port of it. Given
identical scripted replies the two must agree on the tools called, the tape
written, the costs charged and the finding produced — not approximately, exactly.

Comparing them is only meaningful because `ScriptedClient` makes a run
deterministic. With a live model the two would differ for reasons that have
nothing to do with orchestration, and this file could not exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.agent.llm import ScriptedClient
from src.agent.loop_graph import investigate_with_graph
from src.agent.loop_plain import investigate
from src.agent.state import AgentState, CriticVerdict
from src.clock import FrozenClock
from src.tools import ToolContext
from tests.test_agent_loop import (
    _ALL_LOOKALIKES,
    a_call,
    a_lookup,
    a_plan,
    a_settled_answer,
    a_stop,
    an_unsettled_answer,
)


def scripted(**queues: list[Any]) -> ScriptedClient:
    return ScriptedClient(replies={k: list(v) for k, v in queues.items()})


def both(
    ctx: ToolContext,
    clock: FrozenClock,
    *,
    plans: list[Any],
    routes: list[Any],
    answers: list[Any],
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Run the identical script through each loop and return both results."""
    kwargs.setdefault("critic", False)
    plain = investigate(
        "Output is down on this array. What happened?",
        ctx,
        scripted(planner=plans, router=routes, synthesizer=answers),
        clock,
        investigation_id="INV-PLAIN",
        **kwargs,
    )
    graph = investigate_with_graph(
        "Output is down on this array. What happened?",
        ctx,
        scripted(planner=plans, router=routes, synthesizer=answers),
        clock,
        investigation_id="INV-GRAPH",
        **kwargs,
    )
    return plain, graph


def tape(result: Any) -> list[tuple[str, str, bool]]:
    """The comparable shape of a tape: kind, node, and whether it was planned."""
    return [(s.kind, s.node, s.was_planned) for s in result.steps]


# ===========================================================================
# Equivalence
# ===========================================================================
def test_a_simple_run_is_identical(ctx: ToolContext, clock: FrozenClock) -> None:
    plain, graph = both(
        ctx,
        clock,
        plans=[a_plan(["compute_temp_corrected_pr", "per_mppt_current_balance"])],
        routes=[
            a_call("compute_temp_corrected_pr"),
            a_call("per_mppt_current_balance"),
            a_stop(),
        ],
        answers=[a_settled_answer()],
    )
    assert plain.tools_called == graph.tools_called
    assert tape(plain) == tape(graph)
    assert plain.llm_calls == graph.llm_calls


def test_the_finding_is_identical(ctx: ToolContext, clock: FrozenClock) -> None:
    plain, graph = both(
        ctx,
        clock,
        plans=[a_plan(["compute_temp_corrected_pr"])],
        routes=[a_call("compute_temp_corrected_pr"), a_stop()],
        answers=[a_settled_answer()],
    )
    assert plain.finding is not None and graph.finding is not None
    left = plain.finding.model_dump(exclude={"id", "investigation_id"})
    right = graph.finding.model_dump(exclude={"id", "investigation_id"})
    assert left == right


def test_measured_values_are_identical(ctx: ToolContext, clock: FrozenClock) -> None:
    """The tools are deterministic, so any difference here is orchestration."""
    plain, graph = both(
        ctx,
        clock,
        plans=[a_plan(["compute_temp_corrected_pr", "check_ac_ceiling"])],
        routes=[
            a_call("compute_temp_corrected_pr"),
            a_call("check_ac_ceiling"),
            a_stop(),
        ],
        answers=[a_settled_answer()],
    )
    assert [r.values for r in plain.results] == [r.values for r in graph.results]


def test_an_unplanned_measurement_is_recorded_the_same_way(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """The agency signal has to survive the port, or the metric changes meaning
    depending on which loop produced it."""
    plain, graph = both(
        ctx,
        clock,
        plans=[a_plan(["compute_temp_corrected_pr"])],
        routes=[
            a_call("compute_temp_corrected_pr"),
            a_call("check_clearsky_consistency", reason="the ratio rose"),
            a_stop(),
        ],
        answers=[a_settled_answer()],
    )
    assert (
        plain.unplanned_tools == graph.unplanned_tools == ["check_clearsky_consistency"]
    )
    assert tape(plain) == tape(graph)


def test_a_lookup_is_recorded_the_same_way(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    plain, graph = both(
        ctx,
        clock,
        plans=[
            a_plan(["check_ac_ceiling"], [("H1", "clipping"), ("H2", "curtailment")])
        ],
        routes=[
            a_lookup(["clipping", "curtailment"]),
            a_call("check_ac_ceiling"),
            a_stop(),
        ],
        answers=[an_unsettled_answer()],
    )
    assert tape(plain) == tape(graph)
    lookups = [s for s in graph.steps if s.kind == "retrieval"]
    assert len(lookups) == 2 and lookups[1].was_planned is False


def test_a_failed_measurement_is_handled_the_same_way(
    plant: Any, clock: FrozenClock
) -> None:
    from tests.conftest import context_for

    bare = plant.drop(columns=[c for c in plant.columns if c.startswith("string_")])
    plain, graph = both(
        context_for(bare),
        clock,
        plans=[a_plan(["per_mppt_current_balance", "compute_temp_corrected_pr"])],
        routes=[
            a_call("per_mppt_current_balance"),
            a_call("compute_temp_corrected_pr"),
            a_stop(),
        ],
        answers=[a_settled_answer()],
    )
    assert plain.errors == graph.errors
    assert tape(plain) == tape(graph)


def test_an_unsettled_answer_is_identical(ctx: ToolContext, clock: FrozenClock) -> None:
    """Abstention is the outcome most likely to be lost in a port, because it is
    the one that looks like a failure path."""
    plain, graph = both(
        ctx,
        clock,
        plans=[a_plan(["check_ac_ceiling"])],
        routes=[a_call("check_ac_ceiling"), a_stop()],
        answers=[an_unsettled_answer()],
    )
    assert plain.finding is not None and graph.finding is not None
    assert graph.finding.settled is False
    assert graph.finding.cause is None and graph.finding.confidence is None
    assert [c.cause for c in plain.finding.candidate_causes] == [
        c.cause for c in graph.finding.candidate_causes
    ]


def test_the_per_cycle_cap_stops_both_loops(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    def client() -> ScriptedClient:
        return ScriptedClient(
            replies={
                "planner": [a_plan(["compute_temp_corrected_pr"])],
                "synthesizer": [a_settled_answer()],
            },
            default=a_call("compute_temp_corrected_pr"),
        )

    common = {"max_tools_per_cycle": 3, "critic": False}
    plain = investigate("q", ctx, client(), clock, investigation_id="P", **common)
    graph = investigate_with_graph(
        "q", ctx, client(), clock, investigation_id="G", **common
    )
    assert len(plain.tools_called) == len(graph.tools_called) == 3
    assert plain.stopped_because == graph.stopped_because


# ===========================================================================
# Review cycles
# ===========================================================================
def _verdict(kind: str, **kw: Any) -> CriticVerdict:
    return CriticVerdict(
        hypotheses_considered=["H1", "H2"],
        hypotheses_still_standing=kw.pop("standing", ["H1", "H2"]),
        lookalikes_checked=list(_ALL_LOOKALIKES),
        verdict=kind,  # type: ignore[arg-type]
        **kw,
    )


def test_send_back_replans_identically(ctx: ToolContext, clock: FrozenClock) -> None:
    def critic(state: AgentState, results: Any, synthesis: Any) -> CriticVerdict:
        if not state.verdicts:
            return _verdict("send_back", revision_request="check the ceiling")
        return _verdict("accept", standing=[])

    plans = [a_plan(["compute_temp_corrected_pr"]), a_plan(["check_ac_ceiling"])]
    routes = [
        a_call("compute_temp_corrected_pr"),
        a_stop(),
        a_call("check_ac_ceiling"),
        a_stop(),
    ]
    answers = [a_settled_answer(), a_settled_answer()]

    plain, graph = both(
        ctx, clock, plans=plans, routes=routes, answers=answers, critic=critic
    )
    assert plain.state.cycle == graph.state.cycle == 1
    assert len(plain.state.verdicts) == len(graph.state.verdicts) == 2
    assert plain.tools_called == graph.tools_called
    assert tape(plain) == tape(graph)


def test_the_cycle_cap_stops_both_loops(ctx: ToolContext, clock: FrozenClock) -> None:
    def never_happy(state: AgentState, results: Any, synth: Any) -> CriticVerdict:
        return _verdict("send_back", revision_request="try again with check_ac_ceiling")

    plain, graph = both(
        ctx,
        clock,
        plans=[a_plan(["compute_temp_corrected_pr"])] * 6,
        routes=[a_call("compute_temp_corrected_pr"), a_stop()] * 6,
        answers=[a_settled_answer()] * 6,
        critic=never_happy,
        max_cycles=3,
    )
    assert plain.state.cycle == graph.state.cycle == 3
    assert plain.stopped_because == graph.stopped_because
    assert "3-cycle review cap" in graph.stopped_because


def test_not_enough_evidence_ends_both_loops(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    def undecided(state: AgentState, results: Any, synth: Any) -> CriticVerdict:
        return _verdict("not_enough_evidence")

    plain, graph = both(
        ctx,
        clock,
        plans=[a_plan(["check_ac_ceiling"])],
        routes=[a_call("check_ac_ceiling"), a_stop()],
        answers=[an_unsettled_answer()],
        critic=undecided,
    )
    assert plain.state.cycle == graph.state.cycle == 0
    assert tape(plain) == tape(graph)


# ===========================================================================
# The graph itself
# ===========================================================================
def test_the_graph_is_declared_data_not_traced_control_flow(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """The one thing the framework unambiguously buys: the structure exists as
    an object that can be drawn and checkpointed, not only as code to read."""
    from src.agent.loop_graph import _Run, build_graph

    run = _Run(
        ctx=ctx,
        client=ScriptedClient(replies={}),
        clock=clock,
        brief="",
        default_args={},
        knowledge=__import__(
            "src.knowledge", fromlist=["load_knowledge"]
        ).load_knowledge(),
        critic=False,
        max_tools_per_cycle=3,
        writer=None,
        out=None,  # type: ignore[arg-type]
    )
    graph = build_graph(run)
    nodes = set(graph.get_graph().nodes)
    assert {"plan", "retrieve", "route", "execute", "look_up", "synthesise"} <= nodes


def test_both_loops_write_the_same_trace_file(
    ctx: ToolContext, clock: FrozenClock, tmp_path: Path
) -> None:
    from src.trace.writer import read_trace

    plain, graph = both(
        ctx,
        clock,
        plans=[a_plan(["compute_temp_corrected_pr"])],
        routes=[a_call("compute_temp_corrected_pr"), a_stop()],
        answers=[a_settled_answer()],
        trace_root=tmp_path,
    )
    left = [(s.kind, s.node) for s in read_trace(plain.trace_path)]
    right = [(s.kind, s.node) for s in read_trace(graph.trace_path)]
    assert left == right


def test_the_two_entry_points_take_the_same_arguments() -> None:
    """A difference in signature would mean the two are not the same thing."""
    import inspect

    plain = inspect.signature(investigate).parameters
    graph = inspect.signature(investigate_with_graph).parameters
    assert set(plain) == set(graph)


@pytest.mark.parametrize("loop", [investigate, investigate_with_graph])
def test_both_loops_refuse_a_window_with_no_data(
    loop: Any, ctx: ToolContext, clock: FrozenClock
) -> None:
    from src.tools import ToolError

    with pytest.raises(ToolError):
        loop(
            "q",
            ctx,
            ScriptedClient(replies={}),
            clock,
            investigation_id="X",
            start="2030-01-01",
            end="2030-01-05",
            critic=False,
        )
