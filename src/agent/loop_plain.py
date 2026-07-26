"""The investigation loop, in plain Python.

Written before the LangGraph port on purpose (DECISION 0018). Orchestration is
the thing being learned, and a framework that hides the loop hides exactly the
part worth understanding: where state is mutated, where the cycle terminates,
and what happens when a node returns something unusable. Step 12 ports this to
LangGraph and checks that the golden-set outputs are identical — which is only a
meaningful check because this version exists first and is small enough to read.

    plan  ->  route -> execute  (repeat)  ->  synthesise  ->  critic
      ^                                                          |
      +------------------- send_back ----------------------------+

The critic is a hook rather than a hard-wired node. At step 4 it is absent and
the loop is a single pass; step 6 supplies one and the cycle closes. That seam
also lets the evaluation run the identical loop with and without review, which
is the only way to know whether the critic earns its cost.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.agent.grounding import check_numeric_grounding
from src.agent.llm import BudgetExceeded, LLMClient, LLMResponse
from src.agent.nodes import (
    RouterDecision,
    Synthesis,
    execute,
    ledger_of,
    plan,
    route,
    synthesize,
)
from src.agent.nodes.prompts import plant_brief
from src.agent.state import AgentState, CriticVerdict
from src.clock import Clock
from src.findings.models import Finding
from src.knowledge import KnowledgeBase, load_knowledge
from src.tools import ToolContext, ToolResult, slice_window
from src.trace.models import TraceStep
from src.trace.writer import TraceWriter

__all__ = ["Critic", "InvestigationResult", "investigate"]

# A critic sees the state, the measurements and the draft, and returns a
# structured verdict. Never prose (CLAUDE.md).
Critic = Callable[[AgentState, list[ToolResult], Synthesis], CriticVerdict]


@dataclass
class InvestigationResult:
    """Everything one investigation produced."""

    investigation_id: str
    state: AgentState
    synthesis: Synthesis | None = None
    finding: Finding | None = None
    steps: list[TraceStep] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    llm_calls: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    trace_path: Path | None = None
    stopped_because: str = "completed"

    @property
    def tools_called(self) -> list[str]:
        return list(self.state.tools_called)

    @property
    def unplanned_tools(self) -> list[str]:
        return [s.node for s in self.steps if s.kind == "adaptive"]

    @property
    def ungrounded_numbers(self) -> list[str]:
        return list(self.synthesis.grounding.ungrounded) if self.synthesis else []


def _accrue(target: InvestigationResult, response: LLMResponse | None) -> None:
    if response is None:
        return
    target.llm_calls += 1
    target.cost_usd += response.cost_usd
    target.latency_ms += response.latency_ms


def investigate(
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
    max_tools_per_cycle: int = 8,
    trace_root: Path | str | None = None,
    critic: Critic | None = None,
    knowledge: KnowledgeBase | None = None,
) -> InvestigationResult:
    """Run one investigation end to end.

    Args:
        max_tools_per_cycle: A hard stop on the inner loop. The router is
            supposed to stop itself when nothing further would separate the
            surviving causes; this catches the case where it does not, so a run
            fails loudly at a known bound rather than draining the budget.
        critic: Absent at step 4. When supplied, a `send_back` verdict replans
            with the critic's instruction in hand.
        knowledge: Domain knowledge retrieved for the planner's candidate
            causes and handed to the router and synthesiser as evidence. Pass
            `False`-y to run without it — that is the step 9 ablation, and the
            whole reason it is an argument rather than an import.
    """
    window = slice_window(ctx.frame, start, end)
    brief = plant_brief(ctx, question, window)
    # Tools default to the whole frame, so pin every call to the investigation
    # window unless the router deliberately narrows it. Without this the agent
    # would silently measure two years when asked about a fortnight.
    default_args = {
        "start": start or str(window.index.min()),
        "end": end or str(window.index.max()),
    }

    state = AgentState(
        investigation_id=investigation_id,
        question=question,
        scope=scope or ctx.scope,
        max_cycles=max_cycles,
    )
    out = InvestigationResult(investigation_id=investigation_id, state=state)

    writer: TraceWriter | None = None
    if trace_root is not None:
        writer = TraceWriter(investigation_id, clock=clock, root=trace_root)
        writer.__enter__()
        out.trace_path = writer.path

    def emit(step: TraceStep) -> None:
        out.steps.append(writer.write(step) if writer else step)

    revision_request: str | None = None
    knowledge_base = load_knowledge() if knowledge is None else knowledge
    knowledge_brief = ""

    try:
        while True:
            # ---------------- plan ------------------------------------------
            outcome = plan(
                client,
                state,
                brief,
                results=out.results,
                errors=out.errors,
                revision_request=revision_request,
            )
            _accrue(out, outcome.response)
            state.hypotheses = outcome.hypotheses
            # Accumulate rather than replace: a revised plan is still a plan,
            # so its tools are planned. Only the router's own departures should
            # register as unplanned, or the metric measures replanning instead
            # of adaptation.
            for tool in outcome.planned_tools:
                if tool not in state.planned_tools:
                    state.planned_tools.append(tool)
            if outcome.scope:
                state.scope = outcome.scope

            emit(
                TraceStep(
                    kind="plan",
                    node="planner",
                    # The tape is the dashboard's only data source, so the
                    # possible-causes ledger has to be *in* it. Re-deriving it
                    # in the dashboard would be the dashboard reimplementing a
                    # computation that lives in src/ (CLAUDE.md).
                    args={
                        "cycle": state.cycle,
                        "question": question,
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

            # ---------------- retrieve --------------------------------------
            # Knowledge is fetched *after* the planner has named its candidates
            # and is never fed back into planning. If the planner saw the
            # signature catalogue first it would enumerate whatever the
            # catalogue contains, and the evaluation would be measuring whether
            # the knowledge file and the fault injector agree.
            matched = knowledge_base.match([h.cause for h in state.hypotheses])
            if matched:
                knowledge_brief = knowledge_base.brief_for(matched)
                emit(
                    TraceStep(
                        kind="retrieval",
                        node="knowledge",
                        args={"matched_causes": matched, "cycle": state.cycle},
                        result=("Looked up what is known about: " + ", ".join(matched)),
                        was_planned=True,
                    )
                )

            # ---------------- route / execute -------------------------------
            stop_decision: RouterDecision | None = None
            for _ in range(max_tools_per_cycle):
                decision = route(
                    client,
                    state,
                    brief,
                    out.results,
                    out.errors,
                    calls_remaining=max_tools_per_cycle - len(state.tools_called),
                    knowledge=knowledge_brief,
                )
                if decision.stops:
                    stop_decision = decision
                    _apply_exclusions(state, decision.excludes)
                    break

                # The router may narrow the window; it may not widen it past
                # the investigation.
                args = {**default_args, **decision.args}
                decision = RouterDecision(
                    action=decision.action,
                    tool=decision.tool,
                    args=args,
                    reason=decision.reason,
                    excludes=decision.excludes,
                    response=decision.response,
                )

                execution = execute(ctx, state, decision, decision.response)
                _accrue(out, decision.response)
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
                emit(step)

                if execution.result is not None:
                    out.results.append(execution.result)
                elif execution.error:
                    out.errors.append(execution.error)
            else:
                out.stopped_because = "hit the per-cycle measurement cap"

            # ---------------- synthesise ------------------------------------
            synthesis = synthesize(
                client,
                state,
                brief,
                out.results,
                out.errors,
                revision_request=revision_request,
                knowledge=knowledge_brief,
            )
            _accrue(out, synthesis.response)
            if stop_decision is not None:
                # Fold the stopping router call into the answer, so the tape's
                # cost column still sums to the investigation total.
                _accrue(out, stop_decision.response)
            out.synthesis = synthesis
            state.answer = synthesis.answer
            state.settled = synthesis.settled

            answer_cost = synthesis.response.cost_usd if synthesis.response else 0.0
            answer_tokens = synthesis.response.tokens if synthesis.response else 0
            answer_latency = synthesis.response.latency_ms if synthesis.response else 0
            if stop_decision is not None and stop_decision.response is not None:
                answer_cost += stop_decision.response.cost_usd
                answer_tokens += stop_decision.response.tokens
                answer_latency += stop_decision.response.latency_ms

            emit(
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
                    tokens=answer_tokens,
                    cost_usd=answer_cost,
                    latency_ms=answer_latency,
                )
            )

            # ---------------- review ----------------------------------------
            if critic is None:
                break

            verdict = critic(state, out.results, synthesis)
            state.verdicts.append(verdict)
            emit(
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
            if verdict.verdict in ("accept", "not_enough_evidence"):
                break

            state.cycle += 1
            revision_request = verdict.revision_request
            if state.cap_reached:
                out.stopped_because = (
                    f"reached the {max_cycles}-cycle review cap without an "
                    "accepted answer"
                )
                break

    except BudgetExceeded as exc:
        out.stopped_because = str(exc)
    finally:
        if writer is not None:
            writer.close()

    out.finding = _build_finding(out, clock)
    return out


def _verdict_in_plain_words(verdict: CriticVerdict, state: AgentState) -> str:
    """Render the verdict for the tape.

    The tape is shown to a plant manager, so this resolves hypothesis ids back
    to the causes they stand for and spells out the verdict. `H1, H2` and
    `not_enough_evidence` are internal vocabulary and neither belongs on screen
    (CLAUDE.md).
    """
    by_id = {h.id: h.cause for h in state.hypotheses}
    standing = [by_id.get(h, h) for h in verdict.hypotheses_still_standing]
    if verdict.verdict == "send_back":
        return f"Sent the answer back: {verdict.revision_request}"
    if verdict.verdict == "not_enough_evidence":
        return (
            "Agreed that the measurements do not separate "
            + " or ".join(standing)
            + ", so no single cause is named."
        )
    return "Accepted the answer." + (
        f" Still open: {', '.join(standing)}." if standing else ""
    )


def _apply_exclusions(state: AgentState, excludes: list[str]) -> None:
    """Mark hypotheses the router says a measurement ruled out."""
    if not excludes:
        return
    wanted = set(excludes)
    for hypothesis in state.hypotheses:
        if hypothesis.id in wanted or hypothesis.cause in wanted:
            hypothesis.status = "excluded"


def _build_finding(out: InvestigationResult, clock: Clock) -> Finding | None:
    """Construct the `Finding`, or record why it could not be built.

    A synthesis that cannot become a `Finding` is a real failure and is left as
    one. The alternative — padding the answer until the validators pass — would
    turn the honesty rules into decoration.
    """
    synthesis = out.synthesis
    if synthesis is None:
        return None
    try:
        return synthesis.to_finding(
            finding_id=f"F-{out.investigation_id}",
            detected_at=clock.now(),
            scope=out.state.scope,
            investigation_id=out.investigation_id,
        )
    except Exception as exc:
        out.errors.append(f"the answer could not be filed as a finding: {exc}")
        return None


def recheck_grounding(out: InvestigationResult) -> list[str]:
    """Re-run the numeric grounding check over the final answer.

    Exposed separately so the evaluation harness can assert the §5.6 "zero
    fabricated numerics" target without depending on the synthesiser having run
    it, and so the critic at step 6 can call it on a revised draft.
    """
    if out.synthesis is None:
        return []
    report = check_numeric_grounding(
        "\n".join([out.synthesis.answer, out.synthesis.summary]),
        ledger_of(out.results),
    )
    return list(report.ungrounded)
