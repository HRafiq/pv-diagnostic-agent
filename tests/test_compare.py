"""The rules-vs-agent comparison.

Exercised with synthetic scores rather than live runs, because the property
being tested is the *comparison*, not either engine: does it read the right
metrics, does it find the cases that actually differ, and — the one that
matters — does it stay neutral about which engine wins?

A comparison that quietly favours the thing being evaluated is worse than no
comparison. The tests below check the arithmetic in both directions.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from eval.compare import HEADLINE, Comparison, compare, comparison_table
from eval.metrics import CaseScore, Prediction, aggregate


def a_score(
    case_id: str,
    truth_cause: str | None,
    predicted_cause: str | None,
    *,
    split: str = "heldback",
    is_lookalike: bool = False,
    truth_category: str | None = "fault",
    predicted_category: str | None = "fault",
) -> CaseScore:
    return CaseScore(
        case_id=case_id,
        split=split,
        is_lookalike=is_lookalike,
        truth_settled=truth_cause is not None,
        truth_category=truth_category if truth_cause else None,
        truth_cause=truth_cause,
        predicted_category=predicted_category if predicted_cause else None,
        predicted_cause=predicted_cause,
        predicted_settled=predicted_cause is not None,
    )


def a_run(scores: list[CaseScore], **agency: Any) -> tuple[Any, Any, Any]:
    predictions = [
        Prediction(
            case_id=s.case_id,
            category=s.predicted_category,
            cause=s.predicted_cause,
            settled=s.predicted_settled,
            **agency,
        )
        for s in scores
    ]
    return scores, predictions, aggregate(scores, predictions)


# ===========================================================================
# The summary table
# ===========================================================================
def test_the_headline_is_not_overall_accuracy() -> None:
    """A detector that alarms on any deficit scores well on clean faults and is
    useless in the field. Only the look-alike column exposes that."""
    assert HEADLINE[0][0] == "false_alarm_rate"
    assert HEADLINE[1][0] == "correct_abstention_rate"
    assert [key for key, _, _ in HEADLINE].index("macro_f1") > 1


def test_each_metric_says_which_direction_is_better() -> None:
    for _, _, direction in HEADLINE:
        assert direction in ("higher", "lower")


def test_the_summary_carries_both_engines_and_their_difference() -> None:
    rules = a_run([a_score("H-001", "string_outage", "string_outage")])
    agent = a_run([a_score("H-001", "string_outage", "soiling")])
    table = compare({"rules": rules, "agent": agent}).summary("heldback")

    assert set(table.columns) >= {"metric", "rules", "agent", "difference"}
    row = table[table["metric"] == "cause accuracy"].iloc[0]
    assert row["rules"] == 1.0 and row["agent"] == 0.0
    assert row["difference"] == -1.0


def test_the_difference_is_signed_not_absolute() -> None:
    """The comparison must be readable when the agent loses."""
    rules = a_run([a_score("H-001", "string_outage", "soiling")])
    agent = a_run([a_score("H-001", "string_outage", "string_outage")])
    table = compare({"rules": rules, "agent": agent}).summary("heldback")
    row = table[table["metric"] == "cause accuracy"].iloc[0]
    assert row["difference"] == 1.0


def test_a_metric_with_no_cases_behind_it_is_none_not_zero() -> None:
    """A split with no look-alikes cannot demonstrate a low false-alarm rate,
    and must not appear to."""
    run = a_run([a_score("H-001", "string_outage", "string_outage")])
    table = compare({"rules": run, "agent": run}).summary("heldback")
    row = table[table["metric"] == "false alarms on look-alikes"].iloc[0]
    # pandas widens a column of Nones to NaN; either way it is not zero, which
    # is the property that matters — an absent measurement must not read as a
    # perfect score.
    assert pd.isna(row["rules"])


# ===========================================================================
# Disagreements — the useful output
# ===========================================================================
def test_only_cases_that_differ_are_listed() -> None:
    rules = a_run(
        [
            a_score("H-001", "string_outage", "string_outage"),
            a_score("H-002", "soiling", "string_outage"),
        ]
    )
    agent = a_run(
        [
            a_score("H-001", "string_outage", "string_outage"),
            a_score("H-002", "soiling", "soiling"),
        ]
    )
    rows = compare({"rules": rules, "agent": agent}).disagreements()
    assert list(rows["case"]) == ["H-002"]
    assert rows.iloc[0]["winner"] == "agent"


def test_an_abstention_reads_in_plain_language() -> None:
    """No ML vocabulary, even in a developer-facing table."""
    rules = a_run([a_score("H-009", None, "clipping", truth_category=None)])
    agent = a_run([a_score("H-009", None, None, truth_category=None)])
    rows = compare({"rules": rules, "agent": agent}).disagreements()
    assert rows.iloc[0]["truth"] == "not enough evidence"
    assert rows.iloc[0]["agent"] == "not enough evidence"
    assert rows.iloc[0]["winner"] == "agent"


def test_both_wrong_is_reported_as_neither() -> None:
    rules = a_run([a_score("H-002", "soiling", "string_outage")])
    agent = a_run([a_score("H-002", "soiling", "shading")])
    rows = compare({"rules": rules, "agent": agent}).disagreements()
    assert rows.iloc[0]["winner"] == "neither"


def test_the_winner_is_decided_on_cause_not_category() -> None:
    """Two causes in one category can lead to opposite actions.

    "Wash the array" and "wipe the sensor" are both interventions and only one
    of them is on the plant.
    """
    rules = a_run(
        [
            a_score(
                "H-004",
                "soiling",
                "sensor_drift",
                truth_category="recoverable",
                predicted_category="recoverable",
            )
        ]
    )
    agent = a_run(
        [
            a_score(
                "H-004",
                "soiling",
                "soiling",
                truth_category="recoverable",
                predicted_category="recoverable",
            )
        ]
    )
    rows = compare({"rules": rules, "agent": agent}).disagreements()
    assert rows.iloc[0]["winner"] == "agent"


def test_disagreements_can_be_filtered_by_split() -> None:
    rules = a_run(
        [
            a_score("G-001", "soiling", "string_outage", split="tuning"),
            a_score("H-001", "soiling", "string_outage", split="heldback"),
        ]
    )
    agent = a_run(
        [
            a_score("G-001", "soiling", "soiling", split="tuning"),
            a_score("H-001", "soiling", "soiling", split="heldback"),
        ]
    )
    comparison = compare({"rules": rules, "agent": agent})
    assert list(comparison.disagreements("tuning")["case"]) == ["G-001"]
    assert len(comparison.disagreements()) == 2


def test_a_single_engine_produces_no_disagreements() -> None:
    run = a_run([a_score("H-001", "string_outage", "soiling")])
    assert compare({"rules": run}).disagreements().empty


# ===========================================================================
# Agency and rendering
# ===========================================================================
def test_agency_is_reported_per_engine() -> None:
    rules = a_run(
        [a_score("H-001", "string_outage", "string_outage")],
        tools_called=("a", "b"),
        unplanned_tools=(),
    )
    agent = a_run(
        [a_score("H-001", "string_outage", "string_outage")],
        tools_called=("a", "c"),
        unplanned_tools=("c",),
    )
    table = compare({"rules": rules, "agent": agent}).agency()
    assert set(table["engine"]) == {"rules", "agent"}
    by_engine = table.set_index("engine")["unplanned_measurement_rate"].to_dict()
    assert by_engine["rules"] == 0.0
    assert by_engine["agent"] == 1.0


def test_the_rendered_table_names_both_engines_and_the_split() -> None:
    run = a_run([a_score("H-001", "string_outage", "string_outage")])
    text = comparison_table(compare({"rules": run, "agent": run}), "heldback")
    assert "rules vs agent" in text
    assert "heldback" in text
    assert "agreed on every case" in text


def test_the_rendered_table_lists_the_cases_that_moved() -> None:
    rules = a_run([a_score("H-002", "soiling", "string_outage")])
    agent = a_run([a_score("H-002", "soiling", "soiling")])
    text = comparison_table(compare({"rules": rules, "agent": agent}), "heldback")
    assert "1 cases answered differently" in text
    assert "agent right on 1" in text
    assert "H-002" in text


def test_the_comparison_serialises_for_the_findings_document() -> None:
    rules = a_run([a_score("H-002", "soiling", "string_outage")])
    agent = a_run([a_score("H-002", "soiling", "soiling")])
    payload = compare({"rules": rules, "agent": agent}).to_dict()
    assert payload["engines"] == ["rules", "agent"]
    assert payload["reports"]["agent"]["by_split"]["heldback"]["cases"] == 1
    assert payload["disagreements"][0]["case"] == "H-002"


def test_an_empty_comparison_does_not_explode() -> None:
    empty = Comparison()
    assert empty.engines == []
    assert empty.disagreements().empty
    assert empty.summary("heldback").shape[0] == len(HEADLINE)


# ===========================================================================
# The runner's ablation flags
# ===========================================================================
@pytest.mark.parametrize("flag", ["--no-review", "--no-knowledge"])
def test_the_ablation_flags_exist_on_both_commands(flag: str) -> None:
    """An ablation that cannot be run is a claim, not a measurement."""
    from eval.runner import main

    with pytest.raises(SystemExit) as exit_info:
        main(["run", "--engine", "agent", flag, "--split", "nonsense"])
    assert exit_info.value.code == 2  # rejected the split, accepted the flag


def test_compare_is_a_command() -> None:
    from eval.runner import main

    with pytest.raises(SystemExit):
        main(["compare", "--split", "nonsense"])
