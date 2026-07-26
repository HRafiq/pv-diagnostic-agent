"""The critic, and the parts of its verdict it does not get to decide.

A critic that returns "looks good" is worse than no critic — it launders an
unchecked answer as a reviewed one. The tests here are almost all about the same
property: the critic can be *stricter* than the model wanted, never looser.

Three things are computed rather than accepted:

* unsupported claims come from the deterministic grounding check, so a model
  cannot clear its own arithmetic;
* a look-alike counts as checked only if a tool that discriminates it actually
  ran, so the checklist cannot be ticked by assertion;
* a verdict that fails the contract becomes `send_back`, because a critic that
  could not produce a valid verdict has not reviewed anything.
"""

from __future__ import annotations

from typing import Any

from src.agent.llm import ScriptedClient
from src.agent.loop_plain import investigate
from src.agent.nodes.critic import (
    CRITIC_SCHEMA,
    lookalikes_actually_checked,
    review,
)
from src.agent.nodes.synthesizer import Synthesis
from src.agent.state import LOOKALIKE_CHECKLIST, AgentState, Hypothesis
from src.clock import FrozenClock
from src.findings.models import CandidateCause
from src.tools import ToolContext, ToolResult
from tests.test_agent_loop import (
    a_call,
    a_plan,
    a_settled_answer,
    a_stop,
    an_unsettled_answer,
    run,
)

ALL_TOOLS = [
    "compute_temp_corrected_pr",
    "check_ac_ceiling",
    "check_clearsky_consistency",
    "profile_data_quality",
    "weather_context",
]


def a_verdict(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "verdict": "accept",
        "hypotheses_considered": ["H1", "H2"],
        "hypotheses_excluded": [
            {
                "hypothesis": "H2",
                "excluded_by": ["compute_temp_corrected_pr"],
                "reasoning": "the corrected ratio did not move",
            }
        ],
        "hypotheses_still_standing": [],
        "unsupported_claims": [],
        "lookalikes_considered": list(LOOKALIKE_CHECKLIST),
        "revision_request": "",
    }
    payload.update(overrides)
    return payload


def a_state(tools: list[str] | None = None) -> AgentState:
    return AgentState(
        investigation_id="X",
        question="q",
        scope="plant",
        hypotheses=[
            Hypothesis(id="H1", cause="string_outage", consequence_if_true="a visit"),
            Hypothesis(id="H2", cause="weather", consequence_if_true="nothing"),
        ],
        planned_tools=list(tools or ALL_TOOLS),
        tools_called=list(tools if tools is not None else ALL_TOOLS),
    )


def a_synthesis(**overrides: Any) -> Synthesis:
    fields: dict[str, Any] = {
        "settled": True,
        "category": "fault",
        "cause": "string_outage",
        "confidence": 0.8,
        "candidate_causes": [],
        "resolving_measurement": None,
        "title": "One string is down",
        "summary": "One string is producing nothing.",
        "answer": "One string is producing nothing.",
        "recommended_action": "send someone",
        "energy_at_stake_kwh": 0.0,
    }
    fields.update(overrides)
    return Synthesis(**fields)


def a_result(tool: str, **values: float) -> ToolResult:
    return ToolResult(tool=tool, summary=f"{tool} ran", values=dict(values))


# ===========================================================================
# The schema
# ===========================================================================
def test_the_critic_has_no_prose_escape_hatch() -> None:
    """`CriticVerdict` has no free-text verdict field, and neither does this."""
    assert CRITIC_SCHEMA["properties"]["verdict"]["enum"] == [
        "accept",
        "send_back",
        "not_enough_evidence",
    ]
    assert CRITIC_SCHEMA["additionalProperties"] is False


def test_the_lookalike_field_is_closed_to_the_checklist() -> None:
    considered = CRITIC_SCHEMA["properties"]["lookalikes_considered"]
    assert considered["items"]["enum"] == list(LOOKALIKE_CHECKLIST)


# ===========================================================================
# Look-alike coverage is measured, not asserted
# ===========================================================================
def test_a_lookalike_counts_only_if_a_tool_that_discriminates_it_ran() -> None:
    claimed = list(LOOKALIKE_CHECKLIST)
    checked = lookalikes_actually_checked(["check_ac_ceiling"], claimed)
    assert set(checked) == {"clipping", "curtailment"}


def test_claiming_a_lookalike_without_measuring_it_does_not_count() -> None:
    """Box-ticking is what the checklist exists to stop."""
    assert lookalikes_actually_checked([], list(LOOKALIKE_CHECKLIST)) == []


def test_measuring_without_claiming_does_not_count_either() -> None:
    """Running a tool is not the same as weighing what it bears on."""
    assert lookalikes_actually_checked(ALL_TOOLS, []) == []


def test_every_lookalike_is_reachable_by_some_tool() -> None:
    """A checklist item no tool can address makes `accept` unreachable."""
    every = lookalikes_actually_checked(
        [*ALL_TOOLS, "detect_stuck_channels"], list(LOOKALIKE_CHECKLIST)
    )
    assert set(every) == set(LOOKALIKE_CHECKLIST)


# ===========================================================================
# The critic tightens, never loosens
# ===========================================================================
def test_accept_survives_a_clean_answer() -> None:
    client = ScriptedClient(replies={"critic": [a_verdict()]})
    verdict = review(
        client,
        a_state([*ALL_TOOLS, "detect_stuck_channels"]),
        "brief",
        [a_result("compute_temp_corrected_pr", pr=0.8)],
        [],
        a_synthesis(),
    ).verdict
    assert verdict.verdict == "accept"
    assert verdict.unchecked_lookalikes == ()


def test_a_fabricated_figure_blocks_an_accept() -> None:
    """The model does not get to clear its own arithmetic."""
    client = ScriptedClient(replies={"critic": [a_verdict()]})
    verdict = review(
        client,
        a_state([*ALL_TOOLS, "detect_stuck_channels"]),
        "brief",
        [a_result("compute_temp_corrected_pr", pr=0.857)],
        [],
        a_synthesis(answer="The performance ratio is 0.4213."),
    ).verdict
    assert verdict.verdict == "send_back"
    assert any("0.4213" in claim for claim in verdict.unsupported_claims)
    assert verdict.revision_request


def test_an_unchecked_lookalike_blocks_an_accept() -> None:
    """Grounds for sending the answer back on its own."""
    client = ScriptedClient(replies={"critic": [a_verdict()]})
    verdict = review(
        client,
        a_state(["compute_temp_corrected_pr"]),
        "brief",
        [a_result("compute_temp_corrected_pr", pr=0.8)],
        [],
        a_synthesis(),
    ).verdict
    assert verdict.verdict == "send_back"
    assert "clipping" in verdict.unchecked_lookalikes
    assert "never measured against" in (verdict.revision_request or "")


def test_declining_to_commit_needs_two_survivors() -> None:
    client = ScriptedClient(
        replies={
            "critic": [
                a_verdict(
                    verdict="not_enough_evidence", hypotheses_still_standing=["H1"]
                )
            ]
        }
    )
    verdict = review(
        client, a_state(), "brief", [], [], a_synthesis(settled=False)
    ).verdict
    assert verdict.verdict == "send_back"
    assert "fewer than two" in (verdict.revision_request or "")


def test_two_survivors_makes_not_enough_evidence_stand() -> None:
    """Abstention is a successful outcome, not a failure path."""
    client = ScriptedClient(
        replies={
            "critic": [
                a_verdict(
                    verdict="not_enough_evidence",
                    hypotheses_still_standing=["clipping", "curtailment"],
                )
            ]
        }
    )
    verdict = review(
        client,
        a_state(),
        "brief",
        [],
        [],
        a_synthesis(
            settled=False,
            cause=None,
            confidence=None,
            candidate_causes=[
                CandidateCause(cause="clipping", consequence_if_true="nothing"),
                CandidateCause(cause="curtailment", consequence_if_true="claim it"),
            ],
            resolving_measurement="check the dispatch log",
        ),
    ).verdict
    assert verdict.verdict == "not_enough_evidence"


def test_send_back_always_carries_an_actionable_request() -> None:
    """'Consider other causes' is not something a loop can act on."""
    client = ScriptedClient(
        replies={"critic": [a_verdict(verdict="send_back", revision_request="")]}
    )
    verdict = review(client, a_state(), "brief", [], [], a_synthesis()).verdict
    assert verdict.verdict == "send_back"
    assert verdict.revision_request
    assert len(verdict.revision_request) > 30


def test_a_synthesis_that_could_not_be_filed_is_an_unsupported_claim() -> None:
    client = ScriptedClient(replies={"critic": [a_verdict()]})
    verdict = review(
        client,
        a_state([*ALL_TOOLS, "detect_stuck_channels"]),
        "brief",
        [],
        [],
        a_synthesis(build_error="the answer names no cause"),
    ).verdict
    assert verdict.verdict == "send_back"
    assert "the answer names no cause" in verdict.unsupported_claims


def test_an_unusable_verdict_becomes_a_send_back_not_an_accept() -> None:
    """A critic that cannot produce a valid verdict has not reviewed anything.

    Falling through to accept here would be the exact failure this node exists
    to prevent.
    """
    client = ScriptedClient(replies={"critic": ["I think it reads well."]})
    verdict = review(client, a_state(), "brief", [], [], a_synthesis()).verdict
    assert verdict.verdict == "send_back"
    assert verdict.revision_request


# ===========================================================================
# In the loop
# ===========================================================================
# A route sequence that covers every look-alike on the checklist, so an
# `accept` is actually reachable. Anything shorter is sent back — correctly.
FULL_SWEEP = [a_call(name) for name in ALL_TOOLS] + [a_stop()]


def test_the_loop_uses_the_llm_critic_by_default(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    out = run(
        [a_plan(ALL_TOOLS)],
        list(FULL_SWEEP),
        [a_settled_answer()],
        ctx,
        clock,
        reviews=[a_verdict()],
    )
    assert [s.kind for s in out.steps].count("critic") == 1
    assert out.state.verdicts[0].verdict == "accept"
    assert out.state.cycle == 0


def test_an_answer_that_skipped_a_lookalike_is_sent_back_by_the_loop(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """One measurement cannot clear seven look-alikes, whatever the model says."""
    out = run(
        [a_plan(["compute_temp_corrected_pr"])] * 2,
        [a_call("compute_temp_corrected_pr"), a_stop()] * 2,
        [a_settled_answer()] * 2,
        ctx,
        clock,
        reviews=[a_verdict(), a_verdict()],
        max_cycles=2,
    )
    assert out.state.verdicts[0].verdict == "send_back"
    assert out.state.cycle == 2
    assert "2-cycle review cap" in out.stopped_because


def test_review_is_charged_to_the_investigation(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    out = run(
        [a_plan(ALL_TOOLS)],
        list(FULL_SWEEP),
        [a_settled_answer()],
        ctx,
        clock,
        reviews=[a_verdict()],
    )
    # planner, one router call per tool plus the stop, synthesiser, critic
    assert out.llm_calls == 1 + len(FULL_SWEEP) + 1 + 1


def test_the_review_reaches_the_tape_in_plain_words(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    """`not_enough_evidence: H1, H2` is internal vocabulary."""
    out = run(
        [
            a_plan(
                ["check_ac_ceiling"],
                [("H1", "clipping"), ("H2", "curtailment")],
            )
        ],
        [a_call("check_ac_ceiling"), a_stop()],
        [an_unsettled_answer()],
        ctx,
        clock,
        reviews=[
            a_verdict(
                verdict="not_enough_evidence",
                hypotheses_still_standing=["H1", "H2"],
            )
        ],
    )
    step = next(s for s in out.steps if s.kind == "critic")
    assert "not_enough_evidence" not in (step.result or "")
    assert "H1" not in (step.result or "")
    assert "clipping" in (step.result or "")


def test_the_cycle_cap_holds_with_the_real_critic(
    ctx: ToolContext, clock: FrozenClock
) -> None:
    sent_back = a_verdict(verdict="send_back", revision_request="try check_ac_ceiling")
    client = ScriptedClient(
        replies={
            "planner": [a_plan(["compute_temp_corrected_pr"])] * 6,
            "router": [a_call("compute_temp_corrected_pr"), a_stop()] * 6,
            "synthesizer": [a_settled_answer()] * 6,
            "critic": [sent_back] * 6,
        }
    )
    out = investigate(
        "q", ctx, client, clock, investigation_id="INV-CAP6", max_cycles=4
    )
    assert out.state.cycle == 4
    assert len(out.state.verdicts) == 4
    assert "4-cycle review cap" in out.stopped_because


def test_the_configured_cap_is_four() -> None:
    """§3.5 hard cap, and the loop's default must match the config."""
    from src.config import load_models_config

    assert load_models_config().limits.max_planner_critic_cycles == 4
