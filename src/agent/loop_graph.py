"""The same investigation loop, expressed as a LangGraph state machine.

A *port*, not a rewrite. Every node calls the identical function the plain loop
calls, the state object is the same `AgentState`, and the gate on this file is
`tests/test_langgraph_port.py`: given the same scripted replies, both loops must
produce the same tools, the same tape and the same finding. If they diverge,
this file is wrong — the plain loop is the specification.

Writing it second was the point (DECISION 0018). Having the plain version first
makes "did the framework buy anything?" answerable, and the answer is recorded
in `docs/LANGGRAPH_TRADEOFF.md` rather than assumed.

What LangGraph actually contributes here, honestly:

* **The graph is declared, not traced.** `plan -> retrieve -> route -> execute`
  is data, so it can be drawn, checkpointed and resumed. In the plain loop that
  structure exists only as control flow you have to read.
* **Conditional edges name the branches.** `_after_route` returning `"execute"`,
  `"look_up"` or `"synthesise"` is the same `if` the plain loop has, but as a
  labelled transition an operator can see in a diagram.
* **Checkpointing comes free**, which is the one capability the plain loop does
  not have and the one an 86-case evaluation actually wants: a run that dies at
  case 60 currently starts again.

What it costs: the cycle bound has to be restated as a recursion limit, the
inner measurement loop becomes a self-edge that is harder to read than a `for`,
and a stack trace now passes through the framework. On a loop this small that is
close to an even trade, which is itself the finding.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from src.agent.llm import BudgetExceeded, LLMClient
from src.agent.loop_plain import (
    _MAX_UNPRODUCTIVE_LOOKUPS,
    Critic,
    InvestigationResult,
    _accrue,
    _apply_exclusions,
    _build_finding,
    _verdict_in_plain_words,
)
from src.agent.nodes import (
    RouterDecision,
    execute,
    plan,
    route,
    synthesize,
    withdraw_commitment,
)
from src.agent.nodes.critic import review as run_review
from src.agent.nodes.prompts import plant_brief
from src.agent.state import AgentState
from src.clock import Clock
from src.knowledge import KnowledgeBase, load_knowledge
from src.rag.retriever import Retriever
from src.tools import ToolContext, slice_window
from src.trace.models import TraceStep
from src.trace.writer import TraceWriter

__all__ = ["build_graph", "investigate_with_graph"]


class _Run:
    """Everything the nodes share.

    Held on one object rather than threaded through the graph state because the
    graph state is `AgentState`, and `AgentState` is the contract both loops
    agree on. Putting a trace writer and an LLM client in it to satisfy a
    framework would make the two implementations structurally different, which
    is exactly what the equivalence test exists to prevent.
    """

    def __init__(
        self,
        ctx: ToolContext,
        client: LLMClient,
        clock: Clock,
        brief: str,
        default_args: dict[str, str],
        knowledge: Retriever | KnowledgeBase,
        critic: Critic | None | Literal[False],
        max_tools_per_cycle: int,
        writer: TraceWriter | None,
        out: InvestigationResult,
        on_step: Callable[[TraceStep], None] | None = None,
    ) -> None:
        self.on_step = on_step
        self.ctx = ctx
        self.client = client
        self.clock = clock
        self.brief = brief
        self.default_args = default_args
        self.knowledge = knowledge
        self.critic = critic
        self.max_tools_per_cycle = max_tools_per_cycle
        self.writer = writer
        self.out = out
        self.knowledge_brief = ""
        self.unproductive_lookups = 0
        self.revision_request: str | None = None
        self.measurements_this_cycle = 0
        self.router_turns = 0
        self.stop_decision: RouterDecision | None = None

    def emit(self, step: TraceStep) -> None:
        # With a writer, progress is reported from inside `write`; without one
        # it happens here, so a run with no trace directory still shows it.
        if self.writer:
            self.out.steps.append(self.writer.write(step))
            return
        self.out.steps.append(step)
        if self.on_step is not None:
            # Never let a display break a run.
            with contextlib.suppress(Exception):
                self.on_step(step)


def build_graph(run: _Run) -> Any:
    """Compile the state machine. Returns a LangGraph `CompiledStateGraph`."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)

    def plan_node(state: AgentState) -> AgentState:
        outcome = plan(
            run.client,
            state,
            run.brief,
            results=run.out.results,
            errors=run.out.errors,
            revision_request=run.revision_request,
        )
        _accrue(run.out, outcome.response)
        state.hypotheses = outcome.hypotheses
        for tool in outcome.planned_tools:
            if tool not in state.planned_tools:
                state.planned_tools.append(tool)
        if outcome.scope:
            state.scope = outcome.scope
        run.measurements_this_cycle = 0
        run.router_turns = 0
        run.stop_decision = None
        run.emit(
            TraceStep(
                kind="plan",
                node="planner",
                args={
                    "cycle": state.cycle,
                    "question": state.question,
                    "planned_tools": list(outcome.planned_tools),
                    "hypotheses": [
                        {
                            "id": h.id,
                            "cause": h.cause,
                            "status": h.status,
                            "consequence_if_true": h.consequence_if_true,
                            "discriminating_measurements": list(
                                h.discriminating_measurements
                            ),
                        }
                        for h in outcome.hypotheses
                    ],
                },
                result=(
                    outcome.reasoning
                    + "\nCandidates: "
                    + "; ".join(f"{h.id} {h.cause}" for h in outcome.hypotheses)
                    + "\nPlan: "
                    + ", ".join(outcome.planned_tools)
                ),
                was_planned=True,
                tokens=outcome.response.tokens,
                cost_usd=outcome.response.cost_usd,
                latency_ms=outcome.response.latency_ms,
            )
        )
        return state

    def retrieve_node(state: AgentState) -> AgentState:
        matched = run.knowledge.match([h.cause for h in state.hypotheses])
        if matched:
            run.knowledge_brief = run.knowledge.brief_for(matched)
            run.emit(
                TraceStep(
                    kind="retrieval",
                    node="knowledge",
                    args={"matched_causes": matched, "cycle": state.cycle},
                    result="Looked up what is known about: " + ", ".join(matched),
                    was_planned=True,
                )
            )
        return state

    def route_node(state: AgentState) -> AgentState:
        run.router_turns += 1
        decision = route(
            run.client,
            state,
            run.brief,
            run.out.results,
            run.out.errors,
            calls_remaining=run.max_tools_per_cycle - run.measurements_this_cycle,
            knowledge=run.knowledge_brief,
        )
        run._decision = decision  # type: ignore[attr-defined]
        return state

    def execute_node(state: AgentState) -> AgentState:
        decision: RouterDecision = run._decision  # type: ignore[attr-defined]
        decision = RouterDecision(
            action=decision.action,
            tool=decision.tool,
            args={**run.default_args, **decision.args},
            reason=decision.reason,
            excludes=decision.excludes,
            response=decision.response,
        )
        execution = execute(run.ctx, state, decision, decision.response)
        _accrue(run.out, decision.response)
        run.measurements_this_cycle += 1
        state.tools_called.append(decision.tool)
        _apply_exclusions(state, decision.excludes)

        step = execution.step
        if execution.result is not None:
            step = step.model_copy(
                update={
                    "args": {
                        **(step.args or {}),
                        "measured": execution.result.values,
                        "caveats": execution.result.caveats,
                    }
                }
            )
        run.emit(step)
        if execution.result is not None:
            run.out.results.append(execution.result)
        elif execution.error:
            run.out.errors.append(execution.error)
        return state

    def look_up_node(state: AgentState) -> AgentState:
        decision: RouterDecision = run._decision  # type: ignore[attr-defined]
        _accrue(run.out, decision.response)
        _apply_exclusions(state, decision.excludes)
        matched = run.knowledge.match(decision.look_up_causes)
        fetched = run.knowledge.brief_for(matched)
        # Same guard as the plain loop: a lookup whose answer is already in the
        # brief changes nothing the router can see, so it asks again. Kept in
        # step here because the equivalence test compares the two run for run.
        if fetched and fetched not in run.knowledge_brief:
            run.knowledge_brief = "\n\n".join(
                filter(None, [run.knowledge_brief, fetched])
            )
            run.unproductive_lookups = 0
        else:
            run.unproductive_lookups += 1
            note = (
                "ALREADY LOOKED UP: "
                + " and ".join(decision.look_up_causes)
                + ". Looking these up again returns nothing new. "
                "Take a measurement, or stop."
            )
            if note not in run.knowledge_brief:
                run.knowledge_brief = "\n\n".join(
                    filter(None, [run.knowledge_brief, note])
                )
        run.emit(
            TraceStep(
                kind="retrieval",
                node="knowledge",
                args={
                    "asked_about": decision.look_up_causes,
                    "matched_causes": matched,
                    "cycle": state.cycle,
                },
                result=(
                    "Looked up what separates "
                    + " and ".join(decision.look_up_causes)
                    + ("" if matched else " — nothing known about those causes")
                ),
                was_planned=False,
                reason_for_choosing=(
                    decision.reason
                    or "the router asked what separates these two before "
                    "spending a measurement"
                ),
                tokens=decision.response.tokens if decision.response else 0,
                cost_usd=decision.response.cost_usd if decision.response else 0.0,
                latency_ms=decision.response.latency_ms if decision.response else 0,
            )
        )
        return state

    def synthesise_node(state: AgentState) -> AgentState:
        synthesis = synthesize(
            run.client,
            state,
            run.brief,
            run.out.results,
            run.out.errors,
            revision_request=run.revision_request,
            knowledge=run.knowledge_brief,
        )
        _accrue(run.out, synthesis.response)
        if run.stop_decision is not None:
            _accrue(run.out, run.stop_decision.response)
        run.out.synthesis = synthesis
        state.answer = synthesis.answer
        state.settled = synthesis.settled

        cost = synthesis.response.cost_usd if synthesis.response else 0.0
        tokens = synthesis.response.tokens if synthesis.response else 0
        latency = synthesis.response.latency_ms if synthesis.response else 0
        if run.stop_decision is not None and run.stop_decision.response is not None:
            cost += run.stop_decision.response.cost_usd
            tokens += run.stop_decision.response.tokens
            latency += run.stop_decision.response.latency_ms

        run.emit(
            TraceStep(
                kind="answer",
                node="synthesizer",
                args={
                    "settled": synthesis.settled,
                    "cycle": state.cycle,
                    "title": synthesis.title,
                    "category": synthesis.category,
                    "cause": synthesis.cause,
                    "confidence": synthesis.confidence,
                    "candidate_causes": [
                        {
                            "cause": c.cause,
                            "consequence_if_true": c.consequence_if_true,
                        }
                        for c in synthesis.candidate_causes
                    ],
                    "resolving_measurement": synthesis.resolving_measurement,
                    "recommended_action": synthesis.recommended_action,
                    "answer": synthesis.answer,
                    "ungrounded_figures": list(synthesis.grounding.ungrounded),
                    "build_error": synthesis.build_error,
                },
                result=synthesis.summary or synthesis.answer,
                was_planned=True,
                tokens=tokens,
                cost_usd=cost,
                latency_ms=latency,
            )
        )
        return state

    def review_node(state: AgentState) -> AgentState:
        assert run.critic is not False
        if run.critic is None:
            reviewed = run_review(
                run.client,
                state,
                run.brief,
                run.out.results,
                run.out.errors,
                run.out.synthesis,  # type: ignore[arg-type]
            )
            _accrue(run.out, reviewed.response)
            verdict = reviewed.verdict
        else:
            verdict = run.critic(state, run.out.results, run.out.synthesis)  # type: ignore[arg-type]
        state.verdicts.append(verdict)
        run.emit(
            TraceStep(
                kind="critic",
                node="critic",
                args={
                    "cycle": state.cycle,
                    "verdict": verdict.verdict,
                    "still_standing": list(verdict.hypotheses_still_standing),
                    "unsupported_claims": list(verdict.unsupported_claims),
                    "unchecked_lookalikes": list(verdict.unchecked_lookalikes),
                    "revision_request": verdict.revision_request,
                },
                result=_verdict_in_plain_words(verdict, state),
                was_planned=True,
            )
        )
        if verdict.verdict == "send_back":
            state.cycle += 1
            run.revision_request = verdict.revision_request
        elif verdict.verdict == "not_enough_evidence" and run.out.synthesis is not None:
            # Mirrors the plain loop: a review that refuses to commit must not
            # leave the committed draft standing as the published answer.
            run.out.synthesis = withdraw_commitment(
                run.out.synthesis, list(verdict.hypotheses_still_standing), state
            )
            state.settled = False
            run.out.stopped_because = (
                "the review found the evidence insufficient to commit"
            )
        return state

    # ---------------- conditional edges --------------------------------
    def after_look_up(state: AgentState) -> str:
        """Back to the router, unless it keeps asking for what it already has.

        The plain loop expresses this as a `break` out of the inner loop; here
        it is a labelled transition. Both stop after the same number of
        unproductive lookups, which is what keeps the equivalence test honest.
        """
        if run.unproductive_lookups >= _MAX_UNPRODUCTIVE_LOOKUPS:
            run.out.stopped_because = (
                "the router kept asking for knowledge it already had"
            )
            return "synthesise"
        return "route"

    def after_route(state: AgentState) -> str:
        """The branch the plain loop expresses as an `if`, named.

        Same three outcomes, same order of checks. Being a labelled transition
        rather than control flow is most of what the framework buys.
        """
        decision: RouterDecision = run._decision  # type: ignore[attr-defined]
        if decision.stops:
            run.stop_decision = decision
            _apply_exclusions(state, decision.excludes)
            return "synthesise"
        if run.measurements_this_cycle >= run.max_tools_per_cycle:
            run.out.stopped_because = "hit the per-cycle measurement cap"
            return "synthesise"
        if run.router_turns > 2 * run.max_tools_per_cycle + 4:
            run.out.stopped_because = (
                "the router took its whole turn allowance without stopping"
            )
            return "synthesise"
        return "look_up" if decision.action == "look_up" else "execute"

    def after_review(state: AgentState) -> str:
        verdict = state.verdicts[-1]
        if verdict.verdict in ("accept", "not_enough_evidence"):
            return "done"
        # Same guard as the plain loop: a cycle that measured nothing cannot
        # produce different evidence next time round, so replanning is a
        # guaranteed-identical answer at full price.
        if run.measurements_this_cycle == 0 and run.out.results:
            run.out.stopped_because = (
                "a review cycle produced no new measurement, so another one "
                "could not change the evidence"
            )
            return "done"
        if state.cap_reached:
            run.out.stopped_because = (
                f"reached the {state.max_cycles}-cycle review cap without an "
                "accepted answer"
            )
            return "done"
        return "replan"

    def after_synthesise(state: AgentState) -> str:
        return "done" if run.critic is False else "review"

    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("route", route_node)
    graph.add_node("execute", execute_node)
    graph.add_node("look_up", look_up_node)
    graph.add_node("synthesise", synthesise_node)
    graph.add_node("review", review_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "route")
    graph.add_conditional_edges(
        "route",
        after_route,
        {"execute": "execute", "look_up": "look_up", "synthesise": "synthesise"},
    )
    graph.add_edge("execute", "route")
    graph.add_conditional_edges(
        "look_up", after_look_up, {"route": "route", "synthesise": "synthesise"}
    )
    graph.add_conditional_edges(
        "synthesise", after_synthesise, {"review": "review", "done": END}
    )
    graph.add_conditional_edges("review", after_review, {"replan": "plan", "done": END})
    return graph.compile()


def investigate_with_graph(
    question: str,
    ctx: ToolContext,
    client: LLMClient,
    clock: Clock,
    *,
    investigation_id: str,
    scope: str | None = None,
    start: str | None = None,
    end: str | None = None,
    max_cycles: int = 4,
    max_tools_per_cycle: int | None = None,
    trace_root: Path | str | None = None,
    critic: Critic | None | Literal[False] = None,
    knowledge: Retriever | KnowledgeBase | None = None,
    on_step: Callable[[TraceStep], None] | None = None,
) -> InvestigationResult:
    """Run one investigation through the graph.

    Signature-identical to `loop_plain.investigate` on purpose: the equivalence
    test calls both with the same arguments, and a difference in the signature
    would mean the two are not the same thing being compared.
    """
    if max_tools_per_cycle is None:
        from src.config import load_models_config

        max_tools_per_cycle = load_models_config().limits.max_tools_per_cycle

    window = slice_window(ctx.frame, start, end)
    brief = plant_brief(ctx, question, window)
    state = AgentState(
        investigation_id=investigation_id,
        question=question,
        scope=scope or ctx.scope,
        max_cycles=max_cycles,
    )
    out = InvestigationResult(investigation_id=investigation_id, state=state)

    writer: TraceWriter | None = None
    if trace_root is not None:
        writer = TraceWriter(
            investigation_id, clock=clock, root=trace_root, on_step=on_step
        )
        writer.__enter__()
        out.trace_path = writer.path

    run = _Run(
        ctx=ctx,
        client=client,
        clock=clock,
        brief=brief,
        default_args={
            "start": start or str(window.index.min()),
            "end": end or str(window.index.max()),
        },
        knowledge=load_knowledge() if knowledge is None else knowledge,
        critic=critic,
        max_tools_per_cycle=max_tools_per_cycle,
        writer=writer,
        out=out,
        on_step=on_step,
    )

    try:
        # The cycle cap is enforced by `after_review` as in the plain loop; the
        # recursion limit is a backstop against a graph that cannot terminate,
        # not the primary bound. Setting it as the only bound would make the
        # two implementations stop for different reasons.
        # LangGraph copies the state between nodes rather than threading one
        # object through, so the object handed in is *not* the one the nodes
        # mutated. Reading the local `state` back would report an untouched
        # AgentState — no tools called, no cycles used — and the equivalence
        # test would be comparing a real run against an empty one.
        final = build_graph(run).invoke(
            state,
            {"recursion_limit": _recursion_budget(max_cycles, max_tools_per_cycle)},
        )
        state = (
            final if isinstance(final, AgentState) else AgentState.model_validate(final)
        )
    except BudgetExceeded as exc:
        out.stopped_because = str(exc)
    finally:
        if writer is not None:
            writer.close()

    out.state = state
    out.finding = _build_finding(out, clock)
    return out


def _recursion_budget(max_cycles: int, max_tools_per_cycle: int) -> int:
    """Enough steps for every cycle to run its full allowance, plus slack."""
    per_cycle = 3 + 2 * (2 * max_tools_per_cycle + 4) + 2
    return max_cycles * per_cycle + 10


GraphFactory = Callable[[_Run], Any]
