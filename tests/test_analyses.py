"""The saved-analyses registry, and the rule that makes it worth having.

`AnalysisSpec.golden_case_ids` must be non-empty — the model enforces that. This
module enforces the rest: the cases have to exist, and so do the tools. An agent
factory that produces unevaluated agents defeats the point of the project, and
"evaluated against case G-999" is unevaluated with extra steps.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agent.state import AnalysisSpec
from src.analyses.registry import AnalysisRegistry
from src.tools import tool_names


def a_spec(**overrides: object) -> AnalysisSpec:
    fields: dict[str, object] = {
        "id": "morning-shade",
        "name": "Morning shading check",
        "question": "Is anything shading the array before {hour}?",
        "allowed_tools": ["time_of_day_profile", "per_mppt_current_balance"],
        "scope": "plant",
        "trigger": "scheduled",
        "golden_case_ids": ["G-003"],
    }
    fields.update(overrides)
    return AnalysisSpec(**fields)  # type: ignore[arg-type]


@pytest.fixture
def registry(tmp_path: Path) -> AnalysisRegistry:
    return AnalysisRegistry.with_golden_cases(tmp_path / "analyses.json")


# ===========================================================================
# The rule
# ===========================================================================
def test_an_analysis_cannot_be_saved_with_no_golden_case() -> None:
    """Enforced by the model, so it cannot be bypassed by a different caller."""
    with pytest.raises(ValidationError):
        a_spec(golden_case_ids=[])


def test_an_analysis_citing_a_case_that_does_not_exist_is_refused(
    registry: AnalysisRegistry,
) -> None:
    """ "Evaluated against G-999" is unevaluated with extra steps."""
    with pytest.raises(ValueError, match="golden cases that do not exist"):
        registry.save(a_spec(golden_case_ids=["G-999"]))


def test_an_analysis_allowing_a_tool_that_does_not_exist_is_refused(
    registry: AnalysisRegistry,
) -> None:
    with pytest.raises(ValueError, match="tools that do not exist"):
        registry.save(a_spec(allowed_tools=["classify_fault"]))


def test_a_valid_analysis_saves_and_reloads(registry: AnalysisRegistry) -> None:
    registry.save(a_spec())
    reloaded = registry.get("morning-shade")
    assert reloaded is not None
    assert reloaded.allowed_tools == ["time_of_day_profile", "per_mppt_current_balance"]


def test_saving_the_same_id_replaces_rather_than_duplicates(
    registry: AnalysisRegistry,
) -> None:
    registry.save(a_spec())
    registry.save(a_spec(name="Renamed"))
    assert len(registry.load()) == 1
    assert registry.get("morning-shade").name == "Renamed"  # type: ignore[union-attr]


def test_deleting_reports_whether_anything_went(registry: AnalysisRegistry) -> None:
    registry.save(a_spec())
    assert registry.delete("morning-shade") is True
    assert registry.delete("morning-shade") is False


def test_an_empty_registry_loads_as_empty(registry: AnalysisRegistry) -> None:
    assert registry.load() == []
    assert registry.summary() == []


def test_the_summary_reports_how_many_cases_back_each_analysis(
    registry: AnalysisRegistry,
) -> None:
    registry.save(a_spec(golden_case_ids=["G-003", "G-004"]))
    assert registry.summary()[0]["golden_cases"] == 2


# ===========================================================================
# v1 is level 1 only
# ===========================================================================
def test_an_analysis_is_a_config_row_not_code() -> None:
    """A registry that could add capability would need its own evaluation for
    every entry, and there is no honest way to build that at this scale."""
    fields = set(AnalysisSpec.model_fields)
    assert not fields & {"code", "script", "python", "prompt_template", "new_tools"}


def test_allowed_tools_are_a_selection_over_the_existing_set() -> None:
    spec = a_spec(allowed_tools=list(tool_names()[:3]))
    assert set(spec.allowed_tools) <= set(tool_names())


def test_a_question_may_carry_slots(registry: AnalysisRegistry) -> None:
    saved = registry.save(a_spec())
    assert "{hour}" in saved.question
