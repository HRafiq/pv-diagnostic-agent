"""The experiment harness: N runs, mean ± spread, and the ablations.

Exercised with synthetic scores, because the property under test is the
*reporting* — a single number from a non-deterministic system is not a result,
and this file is what stops one being printed as though it were.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from eval.experiments import (
    ABLATIONS,
    ExperimentRun,
    ablation_table,
    repeat,
    summarise_runs,
)
from eval.metrics import CaseScore, Prediction


def a_score(case_id: str, correct: bool, **kw: Any) -> CaseScore:
    return CaseScore(
        case_id=case_id,
        split="heldback",
        is_lookalike=kw.get("is_lookalike", False),
        truth_settled=True,
        truth_category="fault",
        truth_cause="string_outage",
        predicted_category="fault" if correct else "recoverable",
        predicted_cause="string_outage" if correct else "soiling",
        predicted_settled=True,
    )


def a_run(label: str, index: int, correct: int, total: int = 4) -> ExperimentRun:
    scores = [a_score(f"H-{n:03d}", n < correct) for n in range(total)]
    predictions = [
        Prediction(
            case_id=s.case_id,
            category=s.predicted_category,
            cause=s.predicted_cause,
            settled=True,
            tools_called=("a", "b") if index == 0 else ("a", "c"),
            unplanned_tools=("c",) if index else (),
            cost_usd=0.10 + 0.01 * index,
        )
        for s in scores
    ]
    return ExperimentRun(label, index, scores, predictions)


# ===========================================================================
# Reporting
# ===========================================================================
def test_a_single_run_is_labelled_as_one() -> None:
    """A spread of 0.000 from one run must not read as 'perfectly stable'."""
    result = summarise_runs([a_run("full", 0, 4)], "heldback")
    assert result.runs == 1
    assert result.spread["cause_accuracy"] == 0.0
    assert "runs" in result.as_row()


def test_three_runs_report_a_real_spread() -> None:
    runs = [a_run("full", n, correct) for n, correct in enumerate((4, 3, 2))]
    result = summarise_runs(runs, "heldback")
    assert result.runs == 3
    assert result.mean["cause_accuracy"] == pytest.approx(0.75, abs=0.01)
    assert result.spread["cause_accuracy"] > 0


def test_the_row_prints_mean_and_spread_together() -> None:
    """A quoted number without a spread invites being read as precise."""
    runs = [a_run("full", n, correct) for n, correct in enumerate((4, 2))]
    row = summarise_runs(runs, "heldback").as_row()
    assert "±" in row["cause_accuracy"]


def test_every_headline_metric_is_reported_when_cases_support_it() -> None:
    runs = [a_run("full", 0, 4)]
    reported = summarise_runs(runs, "heldback").mean
    # These four are computable from any set of settled cases.
    assert {"macro_f1", "category_accuracy", "cause_accuracy", "missed_fault_rate"} <= (
        set(reported)
    )
    # These two are not: a split with no look-alikes cannot demonstrate a low
    # false-alarm rate, and one with no unresolvable cases cannot demonstrate
    # correct abstention. Absent is the honest answer; zero would be a claim.
    assert "false_alarm_rate" not in reported
    assert "correct_abstention_rate" not in reported


def test_a_metric_with_no_cases_behind_it_is_absent_not_zero() -> None:
    result = summarise_runs([a_run("full", 0, 4)], "heldback")
    assert "false_alarm_rate" not in result.mean


def test_agency_is_pooled_across_runs() -> None:
    """Trajectory diversity *across* runs is part of what the metric measures."""
    runs = [a_run("full", n, 4) for n in range(3)]
    agency = summarise_runs(runs, "heldback").agency
    assert agency["distinct_tool_trajectories"] == 2
    assert agency["unplanned_measurement_rate"] > 0


def test_summarising_nothing_does_not_explode() -> None:
    assert summarise_runs([], "heldback").runs == 0


# ===========================================================================
# The ablations
# ===========================================================================
def test_each_ablation_removes_exactly_one_thing() -> None:
    """An ablation that changes two things at once attributes nothing."""
    full = ABLATIONS["full"]
    for name, options in ABLATIONS.items():
        if name in ("full", "neither"):
            continue
        differences = [k for k in options if options[k] != full[k]]
        assert len(differences) == 1, f"{name} changes {differences}"


def test_the_ablations_cover_both_layers_and_their_combination() -> None:
    assert set(ABLATIONS) == {"full", "no review", "no knowledge", "neither"}
    assert ABLATIONS["neither"] == {"review": False, "use_knowledge": False}


def test_repeat_runs_the_configuration_n_times() -> None:
    calls: list[dict[str, Any]] = []

    def runner(cases: list[Any], **kwargs: Any) -> tuple[list[Any], list[Any]]:
        calls.append(kwargs)
        return [a_score("H-001", True)], [
            Prediction(case_id="H-001", category="fault", cause="x", settled=True)
        ]

    runs = repeat("full", runner, cases=[], n=3, review=False)
    assert len(runs) == 3
    assert calls == [{"review": False}] * 3


def test_the_ablation_table_names_every_configuration() -> None:
    results = [summarise_runs([a_run(label, 0, 4)], "heldback") for label in ABLATIONS]
    table = ablation_table(results)
    assert list(table["configuration"]) == list(ABLATIONS)
    assert "median_cost_usd" in table.columns


def _without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `Settings.from_env()` keyless, whatever the machine has.

    Deleting the environment variable is not enough: `from_env` calls
    `load_dotenv` and puts it straight back from `.env`. On a developer machine
    with a real key that would turn the tests below into a live 43-case
    evaluation against the paid API — the exact opposite of what they assert.
    """
    from src.config import Settings

    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(lambda cls: cls(anthropic_api_key=None)),
    )


def test_the_experiments_command_needs_a_key_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without one it must fail loudly, not return a shape that looks like a
    result.

    The exit code alone is not the assertion. This command has several ways to
    return 1 — no golden set, no ingested data — and a test that checked only
    the code would keep passing while the key check itself was gone.
    """
    from eval.runner import main

    _without_a_key(monkeypatch)

    assert main(["experiments", "--split", "heldback", "--runs", "1"]) == 1
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().out


def test_the_key_is_checked_before_any_case_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing fast is the point: four ablations over 43 cases each load a
    plant and materialise an injection before they would reach a client."""
    import eval.runner as runner

    def explode(*_: Any, **__: Any) -> None:
        raise AssertionError("cases were loaded before the key was checked")

    _without_a_key(monkeypatch)
    monkeypatch.setattr(runner, "_load_cases", explode)

    assert runner.main(["experiments", "--split", "heldback", "--runs", "1"]) == 1


# ---------------------------------------------------------------------------
# One bad case must not destroy the run
# ---------------------------------------------------------------------------
# A 43-case agent evaluation costs real money. When an investigation died —
# a malformed reply, a transient API error — the exception propagated out of
# run_agent_engine and every case that had already succeeded was lost along
# with what it cost. Twice, in practice, before this existed.
def test_a_failed_case_is_scored_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import eval.runner as runner

    cases = [
        SimpleNamespace(
            id=f"G-{n:03d}",
            system_id=4902,
            question="why is output low?",
            start="2017-05-16",
            end="2017-05-30",
            split="tuning",
            is_lookalike=False,
            truth_settled=True,
            truth_category="fault",
            truth_cause="string_fault",
        )
        for n in (1, 2, 3)
    ]

    monkeypatch.setattr(
        runner,
        "load_plant",
        lambda *a, **k: SimpleNamespace(
            context=lambda *a, **k: object(), meta=SimpleNamespace(name="plant")
        ),
    )
    monkeypatch.setattr(
        runner, "materialise", lambda *a, **k: SimpleNamespace(full_record=object())
    )
    monkeypatch.setattr(runner, "build_client", lambda *a, **k: object())
    monkeypatch.setattr(
        runner, "score_case", lambda case, pred: a_score(case.id, False)
    )

    calls: list[str] = []

    def flaky(question: str, ctx: Any, *a: Any, **kw: Any) -> Any:
        case_id = kw["investigation_id"].removeprefix("INV-")
        calls.append(case_id)
        if case_id == "G-002":
            raise ValueError("planner returned a reply that breaks its schema")
        return SimpleNamespace(
            finding=None,
            tools_called=["compute_temp_corrected_pr"],
            unplanned_tools=[],
            state=SimpleNamespace(cycle=1),
            cost_usd=0.01,
            llm_calls=4,
            cached_calls=0,
            cached_cost_usd=0.0,
            ungrounded_numbers=[],
        )

    monkeypatch.setattr(runner, "investigate", flaky)

    _scores, predictions = runner.run_agent_engine(cases)

    assert calls == ["G-001", "G-002", "G-003"], "the run stopped at the failure"
    assert len(predictions) == 3, "the failed case must stay in the denominator"

    failed = next(p for p in predictions if p.case_id == "G-002")
    assert failed.settled is False
    assert failed.category is None
    assert failed.tools_called == ()

    out = capsys.readouterr().out
    assert "G-002  FAILED" in out
    assert "1 of 3 cases failed outright" in out


def test_a_budget_breach_still_stops_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """BudgetExceeded means the loop is not terminating — a defect in the
    agent, not a bad case. Swallowing it would repeat it 43 times."""
    import eval.runner as runner
    from src.agent.llm import BudgetExceeded

    case = SimpleNamespace(
        id="G-001",
        system_id=4902,
        question="?",
        start="2017-05-16",
        end="2017-05-30",
        split="tuning",
    )
    monkeypatch.setattr(
        runner,
        "load_plant",
        lambda *a, **k: SimpleNamespace(
            context=lambda *a, **k: object(), meta=SimpleNamespace(name="plant")
        ),
    )
    monkeypatch.setattr(
        runner, "materialise", lambda *a, **k: SimpleNamespace(full_record=object())
    )
    monkeypatch.setattr(runner, "build_client", lambda *a, **k: object())
    monkeypatch.setattr(
        runner,
        "investigate",
        lambda *a, **k: (_ for _ in ()).throw(
            BudgetExceeded("investigation hit its 40-call cap")
        ),
    )

    with pytest.raises(BudgetExceeded):
        runner.run_agent_engine([case])


def test_a_dead_network_stops_the_run_rather_than_burning_the_case_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four real runs lost twenty-plus cases at zero work each to a dropped
    connection. Three failures in a row — after the client's own retries — is
    an environment problem, not three unlucky cases."""
    import eval.runner as runner

    cases = [
        SimpleNamespace(
            id=f"G-{n:03d}",
            system_id=4902,
            question="?",
            start="2017-05-16",
            end="2017-05-30",
            split="tuning",
        )
        for n in range(1, 11)
    ]
    monkeypatch.setattr(
        runner,
        "load_plant",
        lambda *a, **k: SimpleNamespace(
            context=lambda *a, **k: object(), meta=SimpleNamespace(name="p")
        ),
    )
    monkeypatch.setattr(
        runner, "materialise", lambda *a, **k: SimpleNamespace(full_record=object())
    )
    monkeypatch.setattr(runner, "build_client", lambda *a, **k: object())
    monkeypatch.setattr(
        runner, "score_case", lambda case, pred: a_score(case.id, False)
    )
    monkeypatch.setattr(
        runner,
        "investigate",
        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("Connection error.")),
    )

    with pytest.raises(RuntimeError, match="cases in a row failed"):
        runner.run_agent_engine(cases)


def test_isolated_failures_do_not_trip_the_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A case that dies on its own content is still just one case."""
    import eval.runner as runner

    cases = [
        SimpleNamespace(
            id=f"G-{n:03d}",
            system_id=4902,
            question="?",
            start="2017-05-16",
            end="2017-05-30",
            split="tuning",
        )
        for n in range(1, 8)
    ]
    monkeypatch.setattr(
        runner,
        "load_plant",
        lambda *a, **k: SimpleNamespace(
            context=lambda *a, **k: object(), meta=SimpleNamespace(name="p")
        ),
    )
    monkeypatch.setattr(
        runner, "materialise", lambda *a, **k: SimpleNamespace(full_record=object())
    )
    monkeypatch.setattr(runner, "build_client", lambda *a, **k: object())
    monkeypatch.setattr(
        runner, "score_case", lambda case, pred: a_score(case.id, False)
    )

    def every_other(question: str, ctx: Any, *a: Any, **kw: Any) -> Any:
        if kw["investigation_id"].endswith(("2", "4", "6")):
            raise ValueError("this one case is malformed")
        return SimpleNamespace(
            finding=None,
            tools_called=["compute_temp_corrected_pr"],
            unplanned_tools=[],
            state=SimpleNamespace(cycle=1),
            cost_usd=0.01,
            llm_calls=4,
            cached_calls=0,
            cached_cost_usd=0.0,
            ungrounded_numbers=[],
        )

    monkeypatch.setattr(runner, "investigate", every_other)

    _scores, predictions = runner.run_agent_engine(cases)
    assert len(predictions) == 7, "the run should have finished every case"


# ===========================================================================
# Resuming a run that died
# ===========================================================================
def test_the_journal_round_trips_a_prediction(tmp_path: Any) -> None:
    """What resuming needs, and nothing more.

    Not the trace files: those record *how* a case ran and the dashboard reads
    them. This is the much smaller record of what it concluded.
    """
    from eval.metrics import Prediction
    from eval.runner import _Journal

    path = tmp_path / "run.jsonl"
    journal = _Journal(path)
    assert journal.load() == {}

    journal.record(
        Prediction(
            case_id="G-017",
            category="fault",
            cause="shading",
            settled=True,
            tools_called=("time_of_day_profile", "check_ac_ceiling"),
            cost_usd=0.65,
        )
    )
    journal.record(
        Prediction(
            case_id="G-039",
            category=None,
            cause=None,
            settled=False,
            failed_with="APIStatusError: overloaded_error",
        )
    )

    done = journal.load()
    assert set(done) == {"G-017", "G-039"}
    assert done["G-017"].cause == "shading"
    assert done["G-017"].tools_called == ("time_of_day_profile", "check_ac_ceiling")
    # The failure marker survives, so a resumed run does not re-count a crash as
    # a chosen abstention.
    assert done["G-039"].failed_with is not None
    assert not done["G-039"].abstained


def test_a_half_written_last_line_does_not_block_a_resume(tmp_path: Any) -> None:
    """Exactly what a killed process leaves behind.

    Refusing to resume because the final line is truncated would throw away
    every case before it — which is the whole thing the journal exists to save.
    """
    from eval.runner import _Journal

    path = tmp_path / "run.jsonl"
    path.write_text(
        '{"case_id": "G-001", "category": "fault", "cause": "string_outage", '
        '"settled": true}\n{"case_id": "G-002", "cat'
    )
    done = _Journal(path).load()
    assert set(done) == {"G-001"}


def test_no_journal_path_means_no_journal(tmp_path: Any) -> None:
    """The flag is opt-in; without it nothing is written and nothing is skipped."""
    from eval.metrics import Prediction
    from eval.runner import _Journal

    journal = _Journal(None)
    journal.record(Prediction(case_id="X", category=None, cause=None, settled=False))
    assert journal.load() == {}
    assert not list(tmp_path.iterdir())
