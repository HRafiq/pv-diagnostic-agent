"""The investigation loop, driven end to end without a network call.

Everything asserted here is a property of the *orchestration*, not of any
model: which node runs when, what gets written to the tape, how `was_planned`
is decided, what happens when a tool fails or a synthesis is unusable, and
whether the honesty rules survive a model that tries to break them.

That distinction is the reason `ScriptedClient` exists. If these behaviours
could only be checked by calling an API, they would be checked rarely,
non-deterministically, and never in CI — and the loop is exactly the part where
a regression is invisible until an evaluation run produces quietly wrong
numbers.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from src.agent.llm import LLMResponse, ScriptedClient
from src.agent.loop_plain import investigate, recheck_grounding
from src.agent.nodes import execute, ledger_of, plan, route, synthesize
from src.agent.nodes.planner import PLAN_SCHEMA
from src.agent.nodes.prompts import plant_brief
from src.agent.nodes.router import ROUTE_SCHEMA, RouterDecision
from src.agent.nodes.synthesizer import SYNTHESIS_SCHEMA
from src.agent.state import AgentState, CriticVerdict, Hypothesis
from src.clock import FrozenClock
from src.tools import ToolContext, run_tool, tool_names
from src.trace.writer import read_trace
from tests.conftest import WINDOW, context_for, synthetic_frame


# ---------------------------------------------------------------------------
# Scripted payload builders
# ---------------------------------------------------------------------------
def a_plan(
    tools: list[str],
    hypotheses: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    pairs = hypotheses or [("H1", "string_outage"), ("H2", "weather")]
    return {
        "scope": "INV-01 / whole plant",
        "opening_reasoning": "Separate a real loss from the weather first.",
        "hypotheses": [
            {
                "id": hid,
                "cause": cause,
                "consequence_if_true": f"acting on {cause} costs a site visit",
                "discriminating_measurements": tools[:1],
            }
            for hid, cause in pairs
        ],
        "planned_tools": tools,
    }


def a_call(
    tool: str, args: dict[str, Any] | None = None, reason: str = "", **kw: Any
) -> dict[str, Any]:
    return {
        "action": "call_tool",
        "tool": tool,
        "args": args or {},
        "reason_for_choosing": reason or f"{tool} separates the open candidates",
        "excludes": kw.get("excludes", []),
    }


def a_stop(reason: str = "one cause stands") -> dict[str, Any]:
    return {
        "action": "stop",
        "tool": "",
        "args": {},
        "reason_for_choosing": reason,
        "excludes": [],
    }


def a_settled_answer(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "settled": True,
        "category": "fault",
        "cause": "string_outage",
        "confidence": 0.85,
        "candidate_causes": [],
        "resolving_measurement": "",
        "title": "One string is carrying no current",
        "summary": "One of the seven strings is producing nothing.",
        "answer": "One string's share of array current is zero while the others hold.",
        "recommended_action": "Send someone to check the combiner fuse.",
        "energy_at_stake_kwh": 0.0,
        "evidence": [],
    }
    payload.update(overrides)
    return payload


def an_unsettled_answer(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "settled": False,
        "category": "",
        "cause": "",
        "confidence": 0.0,
        "candidate_causes": [
            {"cause": "clipping", "consequence_if_true": "nothing to do"},
            {"cause": "curtailment", "consequence_if_true": "bill the grid operator"},
        ],
        "resolving_measurement": "check the grid operator's dispatch log",
        "title": "Output is being held at a ceiling",
        "summary": "Power stops at a flat level every clear midday.",
        "answer": "A flat ceiling is present. Two causes produce it and nothing "
        "measured here separates them.",
        "recommended_action": "Check the dispatch log before sending anyone.",
        "energy_at_stake_kwh": 0.0,
        "evidence": [],
    }
    payload.update(overrides)
    return payload


def run(
    plans: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    ctx: ToolContext,
    clock: FrozenClock,
    reviews: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Any:
    """Drive one investigation from scripted replies.

    Review is off unless the caller scripts one, so a test about routing or
    grounding exercises exactly the path it is about. The critic has its own
    tests below.
    """
    client = ScriptedClient(
        replies={
            "planner": list(plans),
            "router": list(routes),
            "synthesizer": list(answers),
            "critic": list(reviews or []),
        }
    )
    kwargs.setdefault("critic", None if reviews else False)
    return investigate(
        "Output is down on this array. What happened?",
        ctx,
        client,
        clock,
        investigation_id=kwargs.pop("investigation_id", "INV-TEST-001"),
        **kwargs,
    )


# ===========================================================================
# The happy path
# ===========================================================================
def test_a_full_pass_plans_measures_and_answers(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    out = run(
        [a_plan(["compute_temp_corrected_pr", "per_mppt_current_balance"])],
        [
            a_call("compute_temp_corrected_pr"),
            a_call("per_mppt_current_balance"),
            a_stop(),
        ],
        [a_settled_answer()],
        ctx,
        clock,
    )

    assert [s.kind for s in out.steps] == [
        "plan",
        "retrieval",
        "tool",
        "tool",
        "answer",
    ]
    assert out.tools_called == [
        "compute_temp_corrected_pr",
        "per_mppt_current_balance",
    ]
    assert len(out.results) == 2
    assert not out.errors
    assert out.finding is not None
    assert out.finding.settled and out.finding.cause == "string_outage"


def test_the_window_is_pinned_so_the_agent_cannot_silently_measure_everything(
    clock: FrozenClock,
) -> None:
    """A tool defaults to the whole frame. Asked about a fortnight, an
    unpinned run would quietly answer about two years."""
    frame = synthetic_frame(days=40)
    ctx = context_for(frame)
    out = run(
        [a_plan(["compute_temp_corrected_pr"])],
        [a_call("compute_temp_corrected_pr"), a_stop()],
        [a_settled_answer()],
        ctx,
        clock,
        start="2017-04-01",
        end="2017-04-07",
    )
    step = next(s for s in out.steps if s.kind == "tool")
    assert step.args is not None
    assert str(step.args["start"]).startswith("2017-04-01")
    # Seven days at 15 minutes, daylight only.
    assert out.results[0].samples_used < 7 * 96


def test_a_router_narrowing_the_window_overrides_the_default(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    out = run(
        [a_plan(["per_mppt_current_balance"])],
        [
            a_call(
                "per_mppt_current_balance",
                {"hour_start": 7, "hour_end": 10},
                reason="the loss looked confined to the morning",
            ),
            a_stop(),
        ],
        [a_settled_answer()],
        ctx,
        clock,
    )
    # Three hours of a twelve-hour day, so roughly a quarter of what the
    # unnarrowed call would have scored.
    whole_day = run_tool("per_mppt_current_balance", ctx, {}).values
    assert (
        out.results[0].values["intervals_scored"] < 0.4 * whole_day["intervals_scored"]
    )


# ===========================================================================
# was_planned — the agency signal
# ===========================================================================
def test_a_tool_in_the_plan_is_planned(ctx: ToolContext, clock: FrozenClock) -> None:
    out = run(
        [a_plan(["compute_temp_corrected_pr"])],
        [a_call("compute_temp_corrected_pr"), a_stop()],
        [a_settled_answer()],
        ctx,
        clock,
    )
    step = next(s for s in out.steps if s.node == "compute_temp_corrected_pr")
    assert step.kind == "tool" and step.was_planned
    assert out.unplanned_tools == []


def test_a_tool_outside_the_plan_is_recorded_as_unplanned(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    out = run(
        [a_plan(["compute_temp_corrected_pr"])],
        [
            a_call("compute_temp_corrected_pr"),
            a_call(
                "check_clearsky_consistency",
                reason="performance ratio rose, which a real loss cannot do",
            ),
            a_stop(),
        ],
        [a_settled_answer()],
        ctx,
        clock,
    )
    unplanned = [s for s in out.steps if s.kind == "adaptive"]
    assert [s.node for s in unplanned] == ["check_clearsky_consistency"]
    assert "a real loss cannot do" in (unplanned[0].reason_for_choosing or "")
    assert out.unplanned_tools == ["check_clearsky_consistency"]


def test_rerunning_a_planned_tool_counts_as_adaptive(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """The plan spent that tool once. Going back to it over a narrowed window
    is the plan being extended by evidence, not followed."""
    out = run(
        [a_plan(["per_mppt_current_balance"])],
        [
            a_call("per_mppt_current_balance"),
            a_call(
                "per_mppt_current_balance",
                {"hour_start": 7, "hour_end": 10},
                reason="the whole-day average may be hiding a morning-only loss",
            ),
            a_stop(),
        ],
        [a_settled_answer()],
        ctx,
        clock,
    )
    kinds = [s.kind for s in out.steps if s.node == "per_mppt_current_balance"]
    assert kinds == ["tool", "adaptive"]


def test_the_router_cannot_self_report_was_planned(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """A metric the subject reports about itself is not a measurement.

    Even if a model volunteers `was_planned: true` for a tool that was never in
    the plan, the executor derives the flag from the plan and ignores the claim.
    """
    assert "was_planned" not in ROUTE_SCHEMA["properties"]
    lying = a_call("check_ac_ceiling", reason="following the evidence")
    lying["was_planned"] = True

    out = run(
        [a_plan(["compute_temp_corrected_pr"])],
        [lying, a_stop()],
        [a_settled_answer()],
        ctx,
        clock,
    )
    adaptive = next(s for s in out.steps if s.kind == "adaptive")
    assert adaptive.node == "check_ac_ceiling"
    assert adaptive.was_planned is False


def test_an_unplanned_step_with_no_reason_says_so_rather_than_inventing_one(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    out = run(
        [a_plan(["compute_temp_corrected_pr"])],
        [a_call("check_ac_ceiling", reason=" "), a_stop()],
        [a_settled_answer()],
        ctx,
        clock,
    )
    adaptive = next(s for s in out.steps if s.kind == "adaptive")
    assert "without stating a reason" in (adaptive.reason_for_choosing or "")


def test_the_trace_model_refuses_an_unjustified_departure() -> None:
    from src.trace.models import TraceStep

    with pytest.raises(ValidationError, match="evidence of agency"):
        TraceStep(kind="adaptive", node="check_ac_ceiling", was_planned=False)


# ===========================================================================
# Failure handling
# ===========================================================================
def test_a_failed_tool_is_recorded_not_swallowed(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """A measurement that could not be taken must reach the synthesiser.

    Dropping it would let the answer imply a channel was checked when it was
    not — an absence of evidence presented as evidence of absence.
    """
    bare = ctx.frame.drop(
        columns=[c for c in ctx.frame.columns if c.startswith("string_")]
    )
    out = run(
        [a_plan(["per_mppt_current_balance", "compute_temp_corrected_pr"])],
        [
            a_call("per_mppt_current_balance"),
            a_call("compute_temp_corrected_pr"),
            a_stop(),
        ],
        [a_settled_answer()],
        context_for(bare),
        clock,
    )
    assert len(out.errors) == 1
    assert "fewer than two" in out.errors[0]
    assert len(out.results) == 1
    failed = next(s for s in out.steps if s.node == "per_mppt_current_balance")
    assert "could not take this measurement" in (failed.result or "")


def test_a_router_naming_a_tool_that_does_not_exist_stops_the_cycle(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    out = run(
        [a_plan(["compute_temp_corrected_pr"])],
        [a_call("read_the_maintenance_log")],
        [a_settled_answer()],
        ctx,
        clock,
    )
    assert out.tools_called == []
    assert [s.kind for s in out.steps] == ["plan", "retrieval", "answer"]


def test_the_per_cycle_cap_stops_a_router_that_never_stops(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    client = ScriptedClient(
        replies={
            "planner": [a_plan(["compute_temp_corrected_pr"])],
            "synthesizer": [a_settled_answer()],
        },
        default=a_call("compute_temp_corrected_pr"),
    )
    out = investigate(
        "why is output down?",
        ctx,
        client,
        clock,
        investigation_id="INV-CAP",
        max_tools_per_cycle=3,
        critic=False,
    )
    assert len(out.tools_called) == 3
    assert "per-cycle measurement cap" in out.stopped_because


# ===========================================================================
# Honesty of the output
# ===========================================================================
def test_an_unsettled_answer_never_shows_a_cause_or_a_confidence(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """The bug that sends a wash crew to a clean array.

    Even when the model returns a cause and a confidence alongside
    `settled: false`, neither may reach the finding.
    """
    sneaky = an_unsettled_answer(
        cause="soiling", confidence=0.78, category="recoverable"
    )
    out = run(
        [a_plan(["check_ac_ceiling"])],
        [a_call("check_ac_ceiling"), a_stop()],
        [sneaky],
        ctx,
        clock,
    )
    assert out.synthesis is not None
    assert out.synthesis.cause is None
    assert out.synthesis.confidence is None
    assert out.finding is not None
    assert out.finding.cause is None
    assert out.finding.confidence is None
    assert out.finding.energy_verified is False
    assert len(out.finding.candidate_causes) == 2
    assert out.finding.resolving_measurement


def test_an_unsettled_answer_with_one_survivor_is_refused_not_padded(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """Stripping a cause that must not be shown is safe. Inventing a second
    candidate to satisfy a validator would be the same dishonesty inverted."""
    thin = an_unsettled_answer(
        candidate_causes=[{"cause": "clipping", "consequence_if_true": "nothing"}]
    )
    out = run(
        [a_plan(["check_ac_ceiling"])],
        [a_call("check_ac_ceiling"), a_stop()],
        [thin],
        ctx,
        clock,
    )
    assert out.finding is None
    assert out.synthesis is not None
    assert "fewer than two" in (out.synthesis.build_error or "")
    assert any("could not be filed" in e for e in out.errors)


def test_an_unsettled_answer_with_no_next_test_is_refused(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    out = run(
        [a_plan(["check_ac_ceiling"])],
        [a_call("check_ac_ceiling"), a_stop()],
        [an_unsettled_answer(resolving_measurement="")],
        ctx,
        clock,
    )
    assert out.finding is None
    assert "would resolve it" in (out.synthesis.build_error or "")  # type: ignore[union-attr]


def test_a_settled_answer_drops_leftover_candidates(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    hedging = a_settled_answer(
        candidate_causes=[
            {"cause": "soiling", "consequence_if_true": "wash it"},
            {"cause": "weather", "consequence_if_true": "nothing"},
        ],
        resolving_measurement="look again next week",
    )
    out = run(
        [a_plan(["per_mppt_current_balance"])],
        [a_call("per_mppt_current_balance"), a_stop()],
        [hedging],
        ctx,
        clock,
    )
    assert out.finding is not None
    assert out.finding.candidate_causes == []
    assert out.finding.resolving_measurement is None


def test_a_settled_answer_with_no_confidence_is_refused(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    out = run(
        [a_plan(["per_mppt_current_balance"])],
        [a_call("per_mppt_current_balance"), a_stop()],
        [a_settled_answer(confidence=None)],
        ctx,
        clock,
    )
    assert out.finding is None
    assert "no confidence" in (out.synthesis.build_error or "")  # type: ignore[union-attr]


# ===========================================================================
# Numeric grounding
# ===========================================================================
def test_a_fabricated_figure_is_caught(ctx: ToolContext, clock: FrozenClock) -> None:
    out = run(
        [a_plan(["compute_temp_corrected_pr"])],
        [a_call("compute_temp_corrected_pr"), a_stop()],
        [
            a_settled_answer(
                answer="The performance ratio is 0.4213, well below normal."
            )
        ],
        ctx,
        clock,
    )
    assert out.ungrounded_numbers == ["0.4213"]
    assert recheck_grounding(out) == ["0.4213"]


def test_a_quoted_figure_passes(ctx: ToolContext, clock: FrozenClock) -> None:
    client = ScriptedClient(
        replies={
            "planner": [a_plan(["compute_temp_corrected_pr"])],
            "router": [a_call("compute_temp_corrected_pr"), a_stop()],
            "synthesizer": [a_settled_answer()],
        }
    )
    first = investigate("q", ctx, client, clock, investigation_id="INV-A", critic=False)
    measured = first.results[0].values["pr_temperature_corrected"]

    out = run(
        [a_plan(["compute_temp_corrected_pr"])],
        [a_call("compute_temp_corrected_pr"), a_stop()],
        [
            a_settled_answer(
                answer=f"Temperature-corrected performance ratio is {measured:.3f}."
            )
        ],
        ctx,
        clock,
    )
    assert out.ungrounded_numbers == []


def test_the_ledger_namespaces_by_tool(ctx: ToolContext, clock: FrozenClock) -> None:
    out = run(
        [a_plan(["compute_temp_corrected_pr", "check_ac_ceiling"])],
        [a_call("compute_temp_corrected_pr"), a_call("check_ac_ceiling"), a_stop()],
        [a_settled_answer()],
        ctx,
        clock,
    )
    ledger = ledger_of(out.results)
    assert "compute_temp_corrected_pr.pr" in ledger
    assert "check_ac_ceiling.peak_ac_kw" in ledger


# ===========================================================================
# The critic seam and the cycle cap
# ===========================================================================
def _verdict(kind: str, **kw: Any) -> CriticVerdict:
    return CriticVerdict(
        hypotheses_considered=["H1", "H2"],
        hypotheses_still_standing=kw.pop("standing", ["H1", "H2"]),
        lookalikes_checked=list(kw.pop("checked", [])),
        verdict=kind,  # type: ignore[arg-type]
        **kw,
    )


def test_with_review_switched_off_the_loop_is_a_single_pass(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """The ablation that says whether the critic earns its cost."""
    out = run(
        [a_plan(["compute_temp_corrected_pr"])],
        [a_call("compute_temp_corrected_pr"), a_stop()],
        [a_settled_answer()],
        ctx,
        clock,
    )
    assert out.state.cycle == 0
    assert not any(s.kind == "critic" for s in out.steps)


def test_send_back_replans_with_the_instruction_in_hand(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    seen: list[str] = []

    def critic(state: AgentState, results: Any, synthesis: Any) -> CriticVerdict:
        if not state.verdicts:
            return _verdict(
                "send_back",
                revision_request="exclude curtailment with check_ac_ceiling",
            )
        return _verdict("accept", standing=[], checked=list(_ALL_LOOKALIKES))

    client = ScriptedClient(
        replies={
            "planner": [
                a_plan(["compute_temp_corrected_pr"]),
                a_plan(["check_ac_ceiling"]),
            ],
            "router": [
                a_call("compute_temp_corrected_pr"),
                a_stop(),
                a_call("check_ac_ceiling"),
                a_stop(),
            ],
            "synthesizer": [a_settled_answer(), a_settled_answer()],
        }
    )
    out = investigate("q", ctx, client, clock, investigation_id="INV-SB", critic=critic)
    seen = [user for node, _, user in client.calls if node == "planner"]

    assert out.state.cycle == 1
    assert len(out.state.verdicts) == 2
    assert "exclude curtailment" in seen[1]
    assert [s.kind for s in out.steps].count("critic") == 2
    # The revised plan extends the original rather than replacing it, so the
    # unplanned rate keeps measuring router adaptation and not replanning.
    assert out.state.planned_tools == ["compute_temp_corrected_pr", "check_ac_ceiling"]
    assert out.unplanned_tools == []


def test_the_cycle_cap_stops_a_critic_that_never_accepts(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    def never_happy(state: AgentState, results: Any, synthesis: Any) -> CriticVerdict:
        return _verdict(
            "send_back", revision_request="try harder with check_ac_ceiling"
        )

    client = ScriptedClient(replies={})
    client.replies = {
        "planner": [a_plan(["compute_temp_corrected_pr"])] * 6,
        "router": [a_call("compute_temp_corrected_pr"), a_stop()] * 6,
        "synthesizer": [a_settled_answer()] * 6,
    }
    out = investigate(
        "q",
        ctx,
        client,
        clock,
        investigation_id="INV-CYC",
        max_cycles=3,
        critic=never_happy,
    )
    assert out.state.cycle == 3
    assert len(out.state.verdicts) == 3
    assert "3-cycle review cap" in out.stopped_because


def test_not_enough_evidence_ends_the_loop_as_a_success(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """Abstention is an outcome, not a failure path. It must not burn cycles."""

    def undecided(state: AgentState, results: Any, synthesis: Any) -> CriticVerdict:
        return _verdict("not_enough_evidence")

    client = ScriptedClient(
        replies={
            "planner": [a_plan(["check_ac_ceiling"])],
            "router": [a_call("check_ac_ceiling"), a_stop()],
            "synthesizer": [an_unsettled_answer()],
        }
    )
    out = investigate(
        "q", ctx, client, clock, investigation_id="INV-NEE", critic=undecided
    )
    assert out.state.cycle == 0
    assert out.finding is not None and out.finding.settled is False


_ALL_LOOKALIKES = (
    "weather",
    "seasonal_temperature_derating",
    "clipping",
    "curtailment",
    "sensor_drift",
    "snow_or_dust_event",
    "telemetry_gap",
)


# ===========================================================================
# The trace file
# ===========================================================================
def test_the_tape_is_written_and_reloadable(
    ctx: ToolContext, clock: FrozenClock, tmp_path: Any
) -> None:
    out = run(
        [a_plan(["compute_temp_corrected_pr"])],
        [a_call("compute_temp_corrected_pr"), a_stop()],
        [a_settled_answer()],
        ctx,
        clock,
        trace_root=tmp_path,
    )
    assert out.trace_path is not None and out.trace_path.exists()
    replayed = list(read_trace(out.trace_path))
    assert [s.kind for s in replayed] == ["plan", "retrieval", "tool", "answer"]
    assert [s.step_index for s in replayed] == [0, 1, 2, 3]
    # Simulated time from the injected clock, never wall clock.
    assert all(s.timestamp == clock.now() for s in replayed)


def test_the_cost_column_sums_to_the_investigation_total(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """Every LLM call has to land on exactly one tape entry.

    The router call that produced a measurement is billed to that measurement;
    the router call that stopped the cycle is billed to the answer. If either
    were dropped, the cost panel would understate what a run cost.
    """
    priced = ScriptedClient(
        replies={
            "planner": [a_plan(["compute_temp_corrected_pr"])],
            "router": [a_call("compute_temp_corrected_pr"), a_stop()],
            "synthesizer": [a_settled_answer()],
        }
    )
    original = priced.complete

    def priced_complete(
        node: str, system: str, user: str, schema: Any = None
    ) -> LLMResponse:
        response = original(node, system, user, schema)
        response.cost_usd = 0.01
        response.input_tokens = 100
        response.output_tokens = 50
        return response

    priced.complete = priced_complete  # type: ignore[method-assign]
    out = investigate(
        "q", ctx, priced, clock, investigation_id="INV-COST", critic=False
    )

    assert out.llm_calls == 4
    assert out.cost_usd == pytest.approx(0.04)
    assert sum(s.cost_usd for s in out.steps) == pytest.approx(out.cost_usd)
    assert sum(s.tokens for s in out.steps) == 4 * 150


# ===========================================================================
# Node-level behaviour
# ===========================================================================
def test_the_planner_drops_tool_names_that_do_not_exist(
    ctx: ToolContext,
) -> None:
    """A plan naming a tool that does not exist would inflate the unplanned
    rate: every real call would score as a departure from it."""
    payload = a_plan(["compute_temp_corrected_pr"])
    payload["planned_tools"] = ["compute_temp_corrected_pr", "read_the_work_orders"]
    client = ScriptedClient(replies={"planner": [payload]})
    state = AgentState(investigation_id="X", question="q", scope="plant")
    outcome = plan(client, state, "brief")
    assert outcome.planned_tools == ["compute_temp_corrected_pr"]


def test_the_plan_schema_demands_more_than_one_candidate() -> None:
    """A planner that emits one hypothesis has skipped the differential."""
    assert PLAN_SCHEMA["properties"]["hypotheses"]["minItems"] == 2


def test_every_schema_that_drives_control_flow_is_closed() -> None:
    for schema in (PLAN_SCHEMA, ROUTE_SCHEMA, SYNTHESIS_SCHEMA):
        assert schema["additionalProperties"] is False
        assert schema["required"]


def test_the_router_stops_when_it_returns_prose_instead_of_a_decision(
    ctx: ToolContext,
) -> None:
    client = ScriptedClient(replies={"router": ["I think we should keep looking."]})
    state = AgentState(investigation_id="X", question="q", scope="plant")
    decision = route(client, state, "brief", [], [], calls_remaining=3)
    assert decision.stops


def test_the_executor_marks_exclusions_on_the_state(ctx: ToolContext) -> None:
    state = AgentState(
        investigation_id="X",
        question="q",
        scope="plant",
        hypotheses=[
            Hypothesis(id="H1", cause="weather", consequence_if_true="nothing"),
            Hypothesis(id="H2", cause="string_outage", consequence_if_true="visit"),
        ],
        planned_tools=["compute_temp_corrected_pr"],
    )
    decision = RouterDecision(
        action="call_tool",
        tool="compute_temp_corrected_pr",
        reason="baseline",
        excludes=["H1"],
    )
    execution = execute(ctx, state, decision)
    assert execution.ok
    assert execution.step.excludes == ["H1"]


def test_the_brief_states_no_performance_figure(ctx: ToolContext) -> None:
    """If the brief handed over a performance ratio, the planner would anchor
    on it and the first measurement would be decoration."""
    brief = plant_brief(ctx, "why is output down?", ctx.frame)
    assert "performance ratio" not in brief.lower()
    assert "270" not in brief  # no headline numbers beyond plant specification
    assert "LOOK-ALIKES" in brief


def test_the_brief_names_every_lookalike(ctx: ToolContext) -> None:
    brief = plant_brief(ctx, "q", ctx.frame)
    for item in _ALL_LOOKALIKES:
        assert item in brief


def test_prompts_carry_no_thresholds() -> None:
    """A threshold in a prompt is a rule engine written in English."""
    from src.agent.nodes.prompts import (
        PLANNER_SYSTEM,
        ROUTER_SYSTEM,
        SYNTHESIZER_SYSTEM,
    )

    for prompt in (PLANNER_SYSTEM, ROUTER_SYSTEM, SYNTHESIZER_SYSTEM):
        lowered = prompt.lower()
        assert "greater than" not in lowered
        assert "exceeds" not in lowered
        assert "%" not in lowered.replace("0.4% of its power", "")


def test_the_synthesizer_is_told_not_to_do_arithmetic() -> None:
    from src.agent.nodes.prompts import SYNTHESIZER_SYSTEM

    assert "must not estimate, extrapolate, or compute" in SYNTHESIZER_SYSTEM


def test_synthesize_reports_grounding_without_being_asked(
    ctx: ToolContext,
) -> None:
    client = ScriptedClient(
        replies={"synthesizer": [a_settled_answer(answer="PR was 0.9999.")]}
    )
    state = AgentState(investigation_id="X", question="q", scope="plant")
    synthesis = synthesize(client, state, "brief", [], [])
    assert synthesis.grounding.ungrounded == ["0.9999"]
    assert synthesis.grounding.as_claims()[0].startswith("the figure 0.9999")


def test_scripted_client_fails_loudly_when_a_reply_is_missing(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """A silent stub would produce a run that looks complete and means nothing."""
    client = ScriptedClient(replies={"planner": [a_plan(["check_ac_ceiling"])]})
    with pytest.raises(AssertionError, match="no reply queued for node 'router'"):
        investigate("q", ctx, client, clock, investigation_id="INV-X", critic=False)


def test_the_tool_catalogue_reaches_the_planner(ctx: ToolContext) -> None:
    client = ScriptedClient(replies={"planner": [a_plan(["check_ac_ceiling"])]})
    state = AgentState(investigation_id="X", question="q", scope="plant")
    plan(client, state, "brief")
    _, _, user = client.calls[0]
    for name in tool_names():
        assert name in user


def test_a_plan_reply_that_is_prose_yields_no_hypotheses(ctx: ToolContext) -> None:
    client = ScriptedClient(replies={"planner": ["let's have a look at the data"]})
    state = AgentState(investigation_id="X", question="q", scope="plant")
    outcome = plan(client, state, "brief")
    assert outcome.hypotheses == [] and outcome.planned_tools == []


def test_investigation_result_json_round_trips_for_the_dashboard(
    ctx: ToolContext, clock: FrozenClock, tmp_path: Any
) -> None:
    out = run(
        [a_plan(["compute_temp_corrected_pr"])],
        [a_call("compute_temp_corrected_pr"), a_stop()],
        [a_settled_answer()],
        ctx,
        clock,
        trace_root=tmp_path,
    )
    payload = json.loads(pd.Series([s.model_dump_json() for s in out.steps]).iloc[0])
    assert payload["kind"] == "plan"
    assert payload["was_planned"] is True


# ===========================================================================
# The knowledge layer
# ===========================================================================
def test_knowledge_is_retrieved_after_planning_never_before(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """The planner must not see the signature catalogue.

    If it did, it would enumerate whatever the catalogue contains and the
    evaluation would be measuring whether the knowledge file and the fault
    injector agree — two files written by the same person in the same week.
    """
    client = ScriptedClient(
        replies={
            "planner": [
                a_plan(
                    ["compute_temp_corrected_pr"],
                    [("H1", "dust on the modules"), ("H2", "a string has failed")],
                )
            ],
            "router": [a_call("compute_temp_corrected_pr"), a_stop()],
            "synthesizer": [a_settled_answer()],
        }
    )
    investigate("q", ctx, client, clock, investigation_id="INV-K", critic=False)

    planner_prompt = next(u for node, _, u in client.calls if node == "planner")
    router_prompt = next(u for node, _, u in client.calls if node == "router")

    assert "Told apart from" not in planner_prompt
    assert "WHAT IS KNOWN ABOUT THESE CAUSES" not in planner_prompt
    assert "Told apart from" in router_prompt


def test_the_lookup_appears_on_the_tape(ctx: ToolContext, clock: FrozenClock) -> None:
    out = run(
        [a_plan(["check_ac_ceiling"], [("H1", "clipping"), ("H2", "curtailment")])],
        [a_call("check_ac_ceiling"), a_stop()],
        [an_unsettled_answer()],
        ctx,
        clock,
    )
    lookup = next(s for s in out.steps if s.kind == "retrieval")
    assert lookup.node == "knowledge"
    assert "clipping" in (lookup.args or {}).get("matched_causes", [])
    assert "curtailment" in (lookup.args or {}).get("matched_causes", [])


def test_the_knowledge_layer_can_be_switched_off_for_the_ablation(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """Step 9 has to be able to measure whether retrieval is worth anything,
    which means running the identical loop without it."""
    from src.knowledge import KnowledgeBase

    empty = KnowledgeBase(signatures={}, tests=())
    client = ScriptedClient(
        replies={
            "planner": [
                a_plan(
                    ["check_ac_ceiling"], [("H1", "clipping"), ("H2", "curtailment")]
                )
            ],
            "router": [a_call("check_ac_ceiling"), a_stop()],
            "synthesizer": [an_unsettled_answer()],
        }
    )
    out = investigate(
        "q",
        ctx,
        client,
        clock,
        investigation_id="INV-ABL",
        knowledge=empty,
        critic=False,
    )
    assert not any(s.kind == "retrieval" for s in out.steps)
    router_prompt = next(u for node, _, u in client.calls if node == "router")
    assert "WHAT IS KNOWN ABOUT THESE CAUSES" not in router_prompt


# ===========================================================================
# The router's third action (step 6)
# ===========================================================================
def a_lookup(causes: list[str], reason: str = "") -> dict[str, Any]:
    return {
        "action": "look_up",
        "tool": "",
        "args": {},
        "look_up_causes": causes,
        "reason_for_choosing": reason or "not sure which observation separates these",
        "excludes": [],
    }


def test_the_router_can_ask_what_separates_two_causes(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """A lookup costs nothing to run and can save a measurement that would not
    have decided anything. That is a routing decision, so the router makes it."""
    out = run(
        [a_plan(["check_ac_ceiling"], [("H1", "clipping"), ("H2", "curtailment")])],
        [
            a_lookup(
                ["clipping", "curtailment"],
                reason="both fit the flat top; find out what tells them apart",
            ),
            a_call("check_ac_ceiling"),
            a_stop(),
        ],
        [an_unsettled_answer()],
        ctx,
        clock,
    )
    lookups = [s for s in out.steps if s.kind == "retrieval"]
    # One automatic after planning, one the router asked for.
    assert len(lookups) == 2
    asked = lookups[1]
    assert asked.args is not None
    assert asked.args["asked_about"] == ["clipping", "curtailment"]
    assert asked.was_planned is False
    assert "tells them apart" in (asked.reason_for_choosing or "")


def test_a_lookup_does_not_spend_the_measurement_budget(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    out = run(
        [a_plan(["check_ac_ceiling"])],
        [
            a_lookup(["clipping", "curtailment"]),
            a_lookup(["soiling", "sensor_drift"]),
            a_call("check_ac_ceiling"),
            a_stop(),
        ],
        [a_settled_answer()],
        ctx,
        clock,
        max_tools_per_cycle=2,
    )
    assert out.tools_called == ["check_ac_ceiling"]
    assert len([s for s in out.steps if s.kind == "retrieval"]) == 3


def test_a_lookup_with_no_causes_stops_the_cycle(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    out = run(
        [a_plan(["check_ac_ceiling"])],
        [a_lookup([])],
        [a_settled_answer()],
        ctx,
        clock,
    )
    assert out.tools_called == []
    assert out.synthesis is not None


def test_what_the_lookup_found_reaches_the_synthesiser(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    client = ScriptedClient(
        replies={
            "planner": [a_plan(["check_ac_ceiling"], [("H1", "a flat top")])],
            "router": [a_lookup(["clipping", "curtailment"]), a_stop()],
            "synthesizer": [an_unsettled_answer()],
            "critic": [],
        }
    )
    investigate("q", ctx, client, clock, investigation_id="INV-LU", critic=False)
    synth_prompt = next(u for node, _, u in client.calls if node == "synthesizer")
    assert "NOT SEPARABLE" in synth_prompt


# ===========================================================================
# What the first scored run exposed
# ===========================================================================
def test_a_repeated_lookup_does_not_spin_forever(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """G-007 spent seventeen consecutive calls re-reading one knowledge entry.

    A lookup whose answer is already in the brief changes nothing the router
    can see, so it asks again — and the look_up branch costs an LLM call but
    does not count against the per-cycle measurement cap. Only the
    per-investigation budget stopped it, minutes and dollars later.
    """
    lookup = a_lookup(["sensor_drift", "soiling"], "which of these is it?")
    result = run(
        plans=[a_plan(["compute_temp_corrected_pr"])],
        # Twenty identical lookups. Without the guard the loop takes all of
        # them; with it, the cycle ends and the synthesiser answers.
        routes=[lookup] * 20,
        answers=[an_unsettled_answer()],
        ctx=ctx,
        clock=clock,
        **WINDOW,
    )

    lookups = [s for s in result.steps if s.kind == "retrieval"]
    assert len(lookups) <= 5, (
        f"the router made {len(lookups)} lookups without learning anything new"
    )
    assert result.finding is not None


def test_arguments_meant_for_another_tool_are_dropped(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """The router picks from one shared argument vocabulary — the union of
    every tool's fields — because structured outputs cannot express an object
    whose shape depends on another field. Tool models are `extra="forbid"`, so
    a stray field turned a good measurement into "could not take this
    measurement" on case after case."""
    result = run(
        plans=[a_plan(["compute_temp_corrected_pr"])],
        routes=[
            a_call(
                "compute_temp_corrected_pr",
                args={"min_run": 8, "baseline_days": 30, "wash_quantile": 0.9},
            ),
            a_stop(),
        ],
        answers=[a_settled_answer()],
        ctx=ctx,
        clock=clock,
        **WINDOW,
    )

    assert not result.errors, f"the measurement failed: {result.errors}"
    measured = [s for s in result.steps if s.kind in ("tool", "adaptive")]
    assert measured and "could not take this measurement" not in (
        measured[0].result or ""
    )


def test_a_genuinely_wrong_argument_still_fails_loudly() -> None:
    """Narrowing happens at the router boundary, not inside `run_tool`. A wrong
    argument passed from code must still be an error, or the guard is gone."""
    from src.tools import ToolError, run_tool

    with pytest.raises(ToolError, match="invalid arguments"):
        run_tool(
            "compute_temp_corrected_pr",
            context_for(synthetic_frame()),
            {"start": "2017-05-16", "not_a_real_argument": 1},
        )


def test_a_review_cycle_that_measures_nothing_ends_the_investigation(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """A replan with no new measurement cannot produce different evidence.

    One real case spent two such cycles — `plan -> look up -> answer` with
    nothing measured in between — roughly five minutes and half its budget,
    re-reading the same ledger and being rejected for the same reasons each
    time. The critic was right to reject it; the loop was wrong to ask again.
    """

    def always_send_back(state: AgentState, results: Any, synthesis: Any) -> Any:
        return _verdict("send_back", revision_request="look again")

    result = run(
        plans=[a_plan(["compute_temp_corrected_pr"]), a_plan([])],
        # Cycle 1 measures, then stops. Cycle 2 stops without measuring.
        routes=[a_call("compute_temp_corrected_pr"), a_stop(), a_stop()],
        answers=[a_settled_answer(), a_settled_answer()],
        ctx=ctx,
        clock=clock,
        critic=always_send_back,
        **WINDOW,
    )

    assert "no new measurement" in (result.stopped_because or "")
    assert len([s for s in result.steps if s.kind == "plan"]) == 2, (
        "it should have replanned once and then stopped, not run to the cap"
    )


def test_a_cycle_that_does_measure_still_replans(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """The guard must not turn every send_back into a stop — a cycle that
    gathered new evidence has earned another look."""
    calls: list[int] = []

    def send_back_once(state: AgentState, results: Any, synthesis: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            return _verdict("send_back", revision_request="measure the ceiling")
        return _verdict("accept", standing=[], checked=list(_ALL_LOOKALIKES))

    result = run(
        plans=[a_plan(["compute_temp_corrected_pr"]), a_plan(["check_ac_ceiling"])],
        routes=[
            a_call("compute_temp_corrected_pr"),
            a_stop(),
            a_call("check_ac_ceiling"),
            a_stop(),
        ],
        answers=[a_settled_answer(), a_settled_answer()],
        ctx=ctx,
        clock=clock,
        critic=send_back_once,
        **WINDOW,
    )

    assert len([s for s in result.steps if s.kind == "plan"]) == 2
    assert "no new measurement" not in (result.stopped_because or "")


# ===========================================================================
# A review that refuses to commit must un-commit the answer
# ===========================================================================
def test_a_settled_answer_is_withdrawn_when_the_review_will_not_commit(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """The verdict the critic exists to be able to give must actually take.

    The loop already stopped on `not_enough_evidence` — and left the settled
    draft in place, so the reviewer's refusal published the very commitment it
    rejected. G-039's ground truth is "not decidable from this plant's
    telemetry"; the critic said `not_enough_evidence`; the run reported
    `settled: curtailment`.

    The pre-existing test for this verdict fed the loop an *unsettled* draft,
    so it passed throughout — which is why the bug survived to run.
    """

    def undecided(state: AgentState, results: Any, synthesis: Any) -> CriticVerdict:
        return _verdict("not_enough_evidence", standing=["clipping", "curtailment"])

    client = ScriptedClient(
        replies={
            "planner": [a_plan(["check_ac_ceiling"])],
            "router": [a_call("check_ac_ceiling"), a_stop()],
            "synthesizer": [a_settled_answer()],
        }
    )
    out = investigate(
        "q", ctx, client, clock, investigation_id="INV-WITHDRAW", critic=undecided
    )

    assert out.synthesis is not None
    assert out.synthesis.settled is False
    # The load-bearing part: no cause, no confidence on an unsettled answer.
    assert out.synthesis.cause is None
    assert out.synthesis.confidence is None
    assert [c.cause for c in out.synthesis.candidate_causes] == [
        "clipping",
        "curtailment",
    ]
    assert out.synthesis.resolving_measurement
    assert out.finding is not None and out.finding.settled is False
    assert out.finding.cause is None
    assert "insufficient to commit" in out.stopped_because


def test_withdrawal_leaves_an_already_unsettled_answer_alone(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """Nothing to withdraw. The draft's own surviving causes are kept."""

    def undecided(state: AgentState, results: Any, synthesis: Any) -> CriticVerdict:
        return _verdict("not_enough_evidence")

    client = ScriptedClient(
        replies={
            "planner": [a_plan(["check_ac_ceiling"])],
            "router": [a_call("check_ac_ceiling"), a_stop()],
            "synthesizer": [an_unsettled_answer()],
        }
    )
    out = investigate(
        "q", ctx, client, clock, investigation_id="INV-NOOP", critic=undecided
    )
    assert out.synthesis is not None and out.synthesis.settled is False
    assert len(out.synthesis.candidate_causes) >= 2


def test_an_accepted_answer_is_not_withdrawn(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """The withdrawal must key on the verdict, not on the loop ending."""

    def happy(state: AgentState, results: Any, synthesis: Any) -> CriticVerdict:
        return _verdict("accept", checked=_ALL_LOOKALIKES)

    client = ScriptedClient(
        replies={
            "planner": [a_plan(["check_ac_ceiling"])],
            "router": [a_call("check_ac_ceiling"), a_stop()],
            "synthesizer": [a_settled_answer()],
        }
    )
    out = investigate(
        "q", ctx, client, clock, investigation_id="INV-ACCEPT", critic=happy
    )
    assert out.synthesis is not None and out.synthesis.settled is True
    assert out.synthesis.cause == "string_outage"


def test_the_ledger_keeps_every_call_of_a_repeated_tool(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """A tool run twice used to erase its own earlier measurement.

    The namespace was the tool name alone, so last write won. One case ran
    `per_mppt_current_balance` four times over different windows and the six
    figures its answer quoted from the first two were reported as fabricated —
    the more thoroughly a case was measured, the less grounded it looked.
    """
    out = run(
        [a_plan(["per_mppt_current_balance"])],
        [
            a_call("per_mppt_current_balance"),
            a_call("per_mppt_current_balance", start="2017-04-10T00:00:00Z"),
            a_stop(),
        ],
        [a_settled_answer()],
        ctx,
        clock,
    )
    ledger = ledger_of(out.results)
    first = [k for k in ledger if k.startswith("per_mppt_current_balance.")]
    second = [k for k in ledger if k.startswith("per_mppt_current_balance#2.")]
    assert first and second, ledger
    assert len(out.results) == 2


def test_the_router_may_only_look_up_causes_the_knowledge_base_knows() -> None:
    """Free text meant the router asked about hypothesis ids.

    `Looked up what separates H4 and H1 — nothing known about those causes`,
    nine times across eight cases: a paid round trip that could not have
    returned anything, because the knowledge base is keyed by cause name. The
    synthesiser's `cause` was closed for the same reason.
    """
    from src.agent.nodes.router import ROUTE_SCHEMA
    from src.knowledge import cause_vocabulary

    items = ROUTE_SCHEMA["properties"]["look_up_causes"]["items"]
    assert items["enum"] == cause_vocabulary()
    assert "H1" not in items["enum"]


def test_every_node_that_reads_an_answer_may_write_as_long_as_it() -> None:
    """A reviewer that cannot finish its reply fails the whole case.

    The critic re-reads every measurement, the draft and the look-alike
    checklist, then writes an exclusion with reasoning for each — at
    effort='high' that can outrun what the draft cost to produce. Its ceiling
    was below the synthesiser's, it truncated mid-reply, and the case it took
    down was one the agent had answered correctly twice.
    """
    from src.config import load_models_config

    config = load_models_config()
    for name, profile in config.profiles.items():
        assert profile["critic"].max_tokens >= profile["synthesizer"].max_tokens, (
            f"profile {name}: the critic reads the synthesiser's whole output "
            "and must be able to write at least as much"
        )
