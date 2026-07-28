"""Running part of the golden set, and being honest about it.

A subset is the right answer to "did the critic fix work?" and the wrong answer
to "what does the agent score". These tests pin the second half: that a subset
which cannot support a headline metric says so, in words, before the numbers.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from eval.golden import load_cases
from eval.subset import describe, select, warn_about

GOLDEN = Path("eval/golden/cases_tuning.jsonl")


@pytest.fixture(scope="module")
def cases() -> list:
    if not GOLDEN.exists():
        pytest.skip("golden set not built")
    return load_cases(GOLDEN)


def test_only_runs_exactly_the_named_cases(cases: list) -> None:
    chosen = select(cases, only="G-003,G-001")
    assert [c.id for c in chosen] == ["G-003", "G-001"], "order should be as asked"


def test_an_unknown_case_id_fails_loudly(cases: list) -> None:
    """Silently running 42 cases when you asked for one is worse than an error."""
    with pytest.raises(ValueError, match="no such case"):
        select(cases, only="G-001,G-999")


def test_a_sample_keeps_the_shape_of_the_split(cases: list) -> None:
    """The point of stratifying: a small run can still answer the same
    questions as the whole split."""
    chosen = select(cases, sample=10, seed=7)

    assert len(chosen) == 10
    assert {c.expected_category for c in chosen} == {
        c.expected_category for c in cases
    }, "a category vanished, taking its share of macro-F1 with it"
    assert any(not c.settled for c in chosen), "no unresolvable cases in the sample"
    assert any(c.is_lookalike for c in chosen), "no look-alikes in the sample"


def test_a_rare_class_survives_a_small_sample(cases: list) -> None:
    """One `by_design` case in forty-three must not round away to zero."""
    rare = [c for c in cases if c.expected_category == "by_design"]
    assert len(rare) == 1, "fixture assumption changed"

    chosen = select(cases, sample=6, seed=3)
    assert any(c.expected_category == "by_design" for c in chosen)


def test_a_sample_is_reproducible(cases: list) -> None:
    """Two runs must be comparable, which means the same seed draws the same
    cases."""
    first = [c.id for c in select(cases, sample=8, seed=11)]
    second = [c.id for c in select(cases, sample=8, seed=11)]
    third = [c.id for c in select(cases, sample=8, seed=12)]

    assert first == second
    assert first != third, "different seeds should draw differently"


def test_asking_for_more_than_exists_returns_everything(cases: list) -> None:
    assert len(select(cases, sample=999)) == len(cases)


def test_the_whole_split_carries_no_warning(cases: list) -> None:
    assert warn_about(cases, cases) is None


def test_a_subset_is_labelled_as_not_a_score(cases: list) -> None:
    warning = warn_about(cases[:5], cases)
    assert warning is not None
    assert "not a score" in warning
    assert "do not quote" in warning


def test_the_warning_names_the_metric_that_cannot_be_measured(cases: list) -> None:
    """The first five cases are all settled faults and near-faults."""
    first_five = cases[:5]
    assert not any(not c.settled for c in first_five), "fixture assumption changed"

    warning = warn_about(first_five, cases)
    assert warning and "not enough evidence" in warning
    assert "cannot be measured" in warning


def test_a_stratified_sample_warns_about_less(cases: list) -> None:
    """It is still a subset, so it is still labelled — but it should not be
    accused of dropping a metric it kept."""
    warning = warn_about(select(cases, sample=10, seed=7), cases)

    assert warning and "not a score" in warning
    assert "correct 'not enough evidence' cannot be measured" not in warning
    assert "false-alarm rate cannot be measured" not in warning


def test_describe_reports_what_matters(cases: list) -> None:
    line = describe(cases[:5])
    assert "5 cases" in line
    assert "unresolvable" in line and "look-alike" in line


def test_no_selection_returns_the_whole_split(cases: list) -> None:
    assert len(select(cases)) == len(cases)


def test_a_sample_is_a_subset_not_a_reshuffle(cases: list) -> None:
    """Every drawn case must be a real one, drawn once."""
    chosen = select(cases, sample=12, seed=5)
    ids = [c.id for c in chosen]

    assert len(set(ids)) == len(ids), "a case was drawn twice"
    assert set(ids) <= {c.id for c in cases}
    assert Counter(ids).most_common(1)[0][1] == 1
