"""The live progress display.

Not scored, not measured — a view. It is tested for two reasons: it reads
`TraceStep`, so it can drift from the model (mypy already caught a version of
it that invented two step kinds that cannot occur), and it runs inside a paid
evaluation, where a display that raises would take down the run it is
describing.
"""

from __future__ import annotations

from typing import get_args

import pytest

from eval.progress import _LABELS, StepPrinter
from src.trace.models import StepKind, TraceStep


def a_step(kind: str = "tool", **kw: object) -> TraceStep:
    payload: dict[str, object] = {
        "kind": kind,
        "node": kw.pop("node", "compute_temp_corrected_pr"),
        "was_planned": kw.pop("was_planned", True),
    }
    # `TraceStep` requires an unplanned step to say why it was chosen — that
    # reason is the evidence of agency the project exists to make visible.
    if payload["was_planned"] is False:
        kw.setdefault("reason_for_choosing", "the balance result raised a new question")
    payload.update(kw)
    return TraceStep(**payload)  # type: ignore[arg-type]


def test_every_step_kind_has_a_label() -> None:
    """A kind with no label falls through to its internal name, which is the
    jargon the interface is not supposed to show."""
    assert set(_LABELS) == set(get_args(StepKind))


def test_labels_avoid_the_banned_vocabulary() -> None:
    """CLAUDE.md: the interface carries no ML jargon."""
    banned = {"hypothesis", "abstention", "confounder", "groundedness", "holdout"}
    for label in _LABELS.values():
        assert not (banned & set(label.lower().split()))


def test_a_plan_step_names_the_possible_causes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    StepPrinter(case_id="G-001")(
        a_step(
            "plan",
            node="planner",
            args={
                "hypotheses": [
                    {"id": "H1", "cause": "string_outage"},
                    {"id": "H2", "cause": "soiling"},
                ]
            },
        )
    )
    out = capsys.readouterr().out
    assert "2 possible causes" in out
    assert "string_outage" in out and "soiling" in out
    assert "hypothes" not in out.lower(), "internal vocabulary reached the screen"


def test_a_measurement_shows_the_tool_and_its_own_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    StepPrinter()(a_step("tool", result="Performance ratio is 0.778 as measured"))
    out = capsys.readouterr().out
    assert "measure" in out
    assert "compute_temp_corrected_pr" in out
    assert "0.778" in out


def test_an_unplanned_measurement_is_marked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Agency is the thing this project is trying to make visible."""
    StepPrinter()(a_step("adaptive", was_planned=False, result="checked the ceiling"))
    assert "unplanned" in capsys.readouterr().out


def test_an_unsettled_answer_says_not_enough_evidence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    StepPrinter()(
        a_step(
            "answer",
            node="synthesizer",
            args={"settled": False, "resolving_measurement": "wash one string"},
        )
    )
    out = capsys.readouterr().out
    assert "not enough evidence" in out
    assert "wash one string" in out


def test_long_detail_is_clipped_not_wrapped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    StepPrinter(width=40)(a_step("tool", result="x" * 500))
    line = capsys.readouterr().out.strip()
    assert len(line) < 120
    assert line.endswith("…")


def test_the_counter_tracks_cost_and_measurements() -> None:
    printer = StepPrinter(verbose=False)
    printer(a_step("tool", cost_usd=0.01))
    printer(a_step("adaptive", cost_usd=0.02, was_planned=False))
    printer(a_step("answer", node="synthesizer", cost_usd=0.03))

    assert printer.steps == 3
    assert printer.tools == 2
    assert printer.cost_usd == pytest.approx(0.06)


def test_a_display_failure_never_breaks_the_run() -> None:
    """It runs inside a paid evaluation. A broken view must not cost a run."""
    import pandas as pd

    from src.clock import FrozenClock
    from src.trace.writer import TraceWriter

    def explode(step: TraceStep) -> None:
        raise RuntimeError("the display is broken")

    clock = FrozenClock(pd.Timestamp("2017-04-15T12:00:00Z").to_pydatetime())
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        with TraceWriter("INV-X", clock=clock, root=tmp, on_step=explode) as writer:
            written = writer.write(a_step("tool", result="fine"))
        assert written.step_index == 0, "the step was still recorded"


def test_a_send_back_says_why(capsys: pytest.CaptureFixture[str]) -> None:
    """A send_back costs a whole extra cycle. Reading the reason out of the
    trace afterwards is too late to stop a run spending it forty-three times."""
    printer = StepPrinter(case_id="G-005")
    printer(
        a_step(
            "critic",
            node="critic",
            args={
                "verdict": "send_back",
                "unsupported_claims": [
                    "the figure 2014 does not appear in any tool result",
                    "the figure 6.9 does not appear in any tool result",
                ],
            },
        )
    )
    out = capsys.readouterr().out
    assert "send_back" in out
    assert "the figure 2014" in out
    assert "+1 more" in out


def test_a_send_back_falls_back_to_the_look_alikes_then_the_request(
    capsys: pytest.CaptureFixture[str],
) -> None:
    printer = StepPrinter()
    printer(
        a_step(
            "critic",
            node="critic",
            args={"verdict": "send_back", "unchecked_lookalikes": ["clipping"]},
        )
    )
    printer(
        a_step(
            "critic",
            node="critic",
            args={"verdict": "send_back", "revision_request": "measure string 7"},
        )
    )
    out = capsys.readouterr().out
    assert "look-alikes not weighed: clipping" in out
    assert "measure string 7" in out


def test_an_accept_is_not_padded_with_a_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    printer = StepPrinter()
    printer(a_step("critic", node="critic", args={"verdict": "accept"}))
    assert capsys.readouterr().out.strip().endswith("accept")
