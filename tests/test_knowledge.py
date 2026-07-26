"""The knowledge layer — loaded, validated, and never executed.

The tests that matter here are not about content. They are about the boundary:
nothing in `src/knowledge/` may compare a signature to a measurement, and no
entry may contain a threshold. Break either and the accuracy figure stops
measuring diagnosis and starts measuring whether two files written in the same
week agree with each other.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.golden import build_golden_set
from src.knowledge import KnowledgeBase, Signature, load_knowledge
from src.tools import tool_names


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return load_knowledge()


# ===========================================================================
# The boundary
# ===========================================================================
def test_no_signature_states_a_threshold(kb: KnowledgeBase) -> None:
    """Enforced by the loader, so this asserts the loader is actually running.

    A signature saying "deviation over 3% means a string fault" plus any code
    comparing it to a tool result would make the evaluation measure whether
    this file agrees with `simulator/injectors.py`.
    """
    assert kb.signatures  # the loader ran and validated
    for signature in kb.signatures.values():
        blob = " ".join(signature.looks_like) + signature.what_happens
        assert ">" not in blob and "<" not in blob
        assert "threshold" not in blob.lower()


def test_a_signature_with_a_rule_in_it_fails_to_load(tmp_path: Path) -> None:
    (tmp_path / "fault_signatures.yaml").write_text(
        """
signatures:
  string_outage:
    category: fault
    plain_name: a string has failed
    what_happens: a fuse blows
    looks_like:
      - the worst string deviation is greater than 0.03
    distinguishing: {}
    action: send someone
"""
    )
    (tmp_path / "distinguishing_tests.yaml").write_text("pairs: []\n")
    with pytest.raises(ValueError, match="decision rule"):
        load_knowledge(tmp_path)


def test_a_percentage_cutoff_also_fails_to_load(tmp_path: Path) -> None:
    (tmp_path / "fault_signatures.yaml").write_text(
        """
signatures:
  soiling:
    category: recoverable
    plain_name: dirty modules
    what_happens: dust accumulates
    looks_like:
      - performance falls below 90% of its baseline
    distinguishing: {}
    action: schedule a wash
"""
    )
    (tmp_path / "distinguishing_tests.yaml").write_text("pairs: []\n")
    with pytest.raises(ValueError, match="decision rule"):
        load_knowledge(tmp_path)


def test_physics_constants_are_allowed(tmp_path: Path) -> None:
    """ "Roughly 0.4% per degree above 25 °C" is physics, not a decision rule."""
    (tmp_path / "fault_signatures.yaml").write_text(
        """
signatures:
  seasonal_temperature_derating:
    category: by_design
    plain_name: the modules are hot
    what_happens: >
      silicon loses roughly 0.4% of its power per degree of cell temperature
      above 25 degrees
    looks_like:
      - uncorrected performance ratio falls every summer on a healthy plant
    distinguishing: {}
    action: do nothing
"""
    )
    (tmp_path / "distinguishing_tests.yaml").write_text("pairs: []\n")
    assert "seasonal_temperature_derating" in load_knowledge(tmp_path).signatures


def test_the_module_exposes_no_way_to_classify() -> None:
    """There is no `diagnose(measurements) -> cause` anywhere in here."""
    import src.knowledge as knowledge

    for name in dir(knowledge):
        assert not name.startswith(("classify", "diagnose", "match_signature"))
    assert not hasattr(KnowledgeBase, "classify")


# ===========================================================================
# Contract of the entries themselves
# ===========================================================================
def test_every_signature_names_at_least_one_look_alike(kb: KnowledgeBase) -> None:
    """A description of a fault on its own is nearly worthless.

    A fault is only ever diagnosed against its nearest look-alike, so an entry
    with nothing to be told apart from carries no diagnostic information.
    """
    for key, signature in kb.signatures.items():
        assert signature.distinguishing, f"{key} names nothing it is confused with"


def test_every_signature_carries_an_action(kb: KnowledgeBase) -> None:
    """Send someone, schedule something, or do nothing."""
    for signature in kb.signatures.values():
        assert signature.action.strip()


def test_categories_are_the_four_the_finding_model_allows(kb: KnowledgeBase) -> None:
    allowed = {"fault", "recoverable", "by_design", "not_the_plant"}
    assert {s.category for s in kb.signatures.values()} <= allowed


def test_a_separable_pair_must_name_the_measurement(tmp_path: Path) -> None:
    (tmp_path / "fault_signatures.yaml").write_text(
        """
signatures:
  a: {category: fault, plain_name: a, what_happens: x, looks_like: [y],
      distinguishing: {}, action: go}
  b: {category: fault, plain_name: b, what_happens: x, looks_like: [y],
      distinguishing: {}, action: go}
"""
    )
    (tmp_path / "distinguishing_tests.yaml").write_text(
        """
pairs:
  - between: [a, b]
    why_confused: they look the same
    how: look harder
    separable: true
"""
    )
    with pytest.raises(ValueError, match="names no measurement"):
        load_knowledge(tmp_path)


def test_an_unresolvable_pair_must_name_the_external_check(tmp_path: Path) -> None:
    """An abstention without a next step is useless."""
    (tmp_path / "fault_signatures.yaml").write_text(
        """
signatures:
  a: {category: fault, plain_name: a, what_happens: x, looks_like: [y],
      distinguishing: {}, action: go}
  b: {category: fault, plain_name: b, what_happens: x, looks_like: [y],
      distinguishing: {}, action: go}
"""
    )
    (tmp_path / "distinguishing_tests.yaml").write_text(
        """
pairs:
  - between: [a, b]
    why_confused: they look the same
    how: nothing separates them
    separable: false
"""
    )
    with pytest.raises(ValueError, match="external evidence"):
        load_knowledge(tmp_path)


def test_tests_only_reference_causes_that_have_signatures(tmp_path: Path) -> None:
    (tmp_path / "fault_signatures.yaml").write_text(
        """
signatures:
  a: {category: fault, plain_name: a, what_happens: x, looks_like: [y],
      distinguishing: {}, action: go}
"""
    )
    (tmp_path / "distinguishing_tests.yaml").write_text(
        """
pairs:
  - between: [a, ghost]
    why_confused: x
    how: y
    separable: true
    measurement: compute_temp_corrected_pr
"""
    )
    with pytest.raises(ValueError, match="no signature"):
        load_knowledge(tmp_path)


def test_every_named_measurement_is_a_real_tool(kb: KnowledgeBase) -> None:
    """A pointer to a tool that does not exist is worse than no pointer."""
    available = set(tool_names())
    for test in kb.tests:
        if test.measurement:
            assert test.measurement in available, test.measurement
        for extra in test.also:
            assert extra in available, extra


# ===========================================================================
# Agreement with the golden set
# ===========================================================================
def test_the_knowledge_and_the_golden_set_agree_on_what_is_unresolvable(
    kb: KnowledgeBase,
) -> None:
    """If these two disagree, one of them is wrong and the abstention metric is
    scoring against the wrong list."""
    from_knowledge = {frozenset(pair) for pair in kb.unresolvable_pairs()}
    from_cases = {
        frozenset(case.candidate_causes)
        for case in build_golden_set()
        if not case.settled
    }
    assert from_cases <= from_knowledge


def test_the_unresolvable_pair_is_clipping_and_curtailment(kb: KnowledgeBase) -> None:
    assert kb.unresolvable_pairs() == (("clipping", "curtailment"),)


def test_every_injected_cause_has_a_signature(kb: KnowledgeBase) -> None:
    """One vocabulary for causes across the injector, the golden set and the
    knowledge base. Two spellings of the same fault mean the metrics compare
    labels that can never match, and the failure is silent."""
    causes = {c.expected_cause for c in build_golden_set() if c.expected_cause}
    for cause in causes:
        assert cause in kb.signatures, f"{cause} has no signature"


# ===========================================================================
# Retrieval
# ===========================================================================
def test_matching_finds_the_obvious_causes(kb: KnowledgeBase) -> None:
    matched = kb.match(["a string has failed", "dust on the modules"])
    assert "string_outage" in matched
    assert "soiling" in matched


def test_matching_returns_nothing_for_unrelated_text(kb: KnowledgeBase) -> None:
    assert kb.match(["the price of copper"]) == []


def test_a_brief_includes_the_pair_test_not_only_the_signatures(
    kb: KnowledgeBase,
) -> None:
    brief = kb.brief_for(["soiling", "sensor_drift"])
    assert "soiling vs sensor_drift" in brief or "sensor_drift vs soiling" in brief
    assert "wash crew" in brief


def test_the_unresolvable_brief_says_so_in_capitals(kb: KnowledgeBase) -> None:
    brief = kb.brief_for(["clipping", "curtailment"])
    assert "NOT SEPARABLE" in brief
    assert "dispatch log" in brief


def test_a_brief_for_unknown_causes_says_it_has_nothing(kb: KnowledgeBase) -> None:
    assert kb.brief_for(["gremlins"]) == "(no knowledge for these causes)"


def test_signature_text_is_readable_without_a_key(kb: KnowledgeBase) -> None:
    text = kb.signatures["soiling"].as_text()
    assert "dust or dirt on the modules" in text
    assert "Action:" in text


def test_an_empty_knowledge_base_is_constructible_for_the_ablation() -> None:
    """Step 9 has to be able to run the identical loop with retrieval removed."""
    empty = KnowledgeBase(signatures={}, tests=())
    assert empty.match(["anything"]) == []
    assert empty.brief_for(["soiling"]) == "(no knowledge for these causes)"
    assert empty.unresolvable_pairs() == ()


def test_signature_is_frozen() -> None:
    signature = Signature(
        key="x",
        category="fault",
        plain_name="x",
        what_happens="x",
        looks_like=(),
        distinguishing={},
        action="go",
    )
    with pytest.raises(AttributeError):
        signature.category = "recoverable"  # type: ignore[misc]
