"""The experiment harness: N runs, mean ± spread, and the ablations.

Exercised with synthetic scores, because the property under test is the
*reporting* — a single number from a non-deterministic system is not a result,
and this file is what stops one being printed as though it were.
"""

from __future__ import annotations

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
