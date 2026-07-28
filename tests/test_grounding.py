"""The numeric grounding check — what makes "zero fabricated numerics" checkable.

The target in §5.6 is only worth stating if a violation can be detected. This
module is that detector, so its own failure modes matter in both directions:

* a **false negative** lets an invented figure reach a plant manager wearing the
  authority of a measurement;
* a **false positive** buries the real finding under complaints about the "three"
  in "three days", and a check nobody reads is a check that is not running.
"""

from __future__ import annotations

from src.agent.grounding import (
    check_numeric_grounding,
    numeric_literals,
    quotable_values,
)


def test_a_measured_figure_is_grounded() -> None:
    report = check_numeric_grounding("Performance ratio is 0.857.", {"pr": 0.857})
    assert report.ok and report.grounded == ["0.857"]


def test_an_invented_figure_is_caught() -> None:
    report = check_numeric_grounding("Performance ratio is 0.421.", {"pr": 0.857})
    assert not report.ok
    assert report.ungrounded == ["0.421"]
    assert "does not appear in any tool result" in report.as_claims()[0]


def test_a_figure_the_agent_was_shown_may_be_repeated() -> None:
    """The guarantee is "no invented numbers", not "no numbers outside the ledger".

    The knowledge base tells the agent that output falls about 0.4% per °C above
    25°C. Repeating the physics it was handed is quotation, not fabrication —
    and scoring it as fabrication vetoed a correct answer on a real case.
    """
    knowledge = "Output falls about 0.4% per °C of cell temperature above 25°C."
    assert check_numeric_grounding("about 0.4% per °C", {}, quoted_from=knowledge).ok
    # But only what was actually shown.
    assert not check_numeric_grounding(
        "about 0.9% per °C", {}, quoted_from=knowledge
    ).ok


def test_a_year_in_the_record_grounds_a_year_in_the_prose() -> None:
    """The masks strip ISO dates, so a *bare* year was checked and flagged.

    The window is given to the agent as `2014-05-01`; the natural sentence about
    a seasonal comparison says "2014". Three of those, on one case, were enough
    to force two extra review cycles and a wrong abstention.
    """
    brief = "Window 2014-05-01 to 2014-05-15. Record spans 2013-01-01 to 2017-12-31."
    report = check_numeric_grounding(
        "Compared with the same weeks in 2014, output is unchanged.",
        {},
        quoted_from=brief,
    )
    assert report.ok, report.ungrounded
    # A year the record does not contain is still a fabrication.
    assert not check_numeric_grounding("in 1998", {}, quoted_from=brief).ok


def test_tool_prose_is_quotable_but_the_draft_is_not() -> None:
    """A caveat's figures are shown to the model; the draft's are not evidence."""
    caveat = "a healthy inverter sits near 0.96-0.98 at load"
    assert check_numeric_grounding(
        "the healthy band starts at 0.96", {}, quoted_from=caveat
    ).ok
    # The endpoints were shown; a value interpolated between them was not. That
    # is arithmetic, and it stays flagged.
    assert not check_numeric_grounding("the ratio of 0.97", {}, quoted_from=caveat).ok
    # Self-grounding is the failure mode that makes this whole check pointless.
    draft = "the ratio was 0.412"
    assert not check_numeric_grounding(draft, {}).ok


def test_a_hyphenated_range_is_two_positive_numbers() -> None:
    """`0.96-0.98` is a range. Read as a sign it invents a figure that nothing
    can ground — the check manufacturing its own violation."""
    assert [item[0] for item in numeric_literals("near 0.96-0.98 at load")] == [
        "0.96",
        "0.98",
    ]
    # A genuine negative is still read as one.
    assert numeric_literals("a drift of -3.5 %/yr")[0][1] == -3.5


def test_quotable_values_keeps_the_digits_the_masks_remove() -> None:
    """The two extractions face opposite directions, on purpose."""
    assert quotable_values("window 2014-05-01") == [2014.0, 5.0, 1.0]
    assert numeric_literals("window 2014-05-01") == []


def test_arithmetic_is_still_not_quotable() -> None:
    """Both operands shown, their difference is not. CLAUDE.md: no LLM in
    arithmetic — widening the sources must not widen this."""
    shown = "compute_expected_output: expected=16090 measured=17812"
    report = check_numeric_grounding("a gap of 1722 kWh", {}, quoted_from=shown)
    assert report.ungrounded == ["1722"]


def test_rounding_to_fewer_digits_is_accepted() -> None:
    """A figure printed as 0.86 legitimately stands for anything in
    [0.855, 0.865). Flagging correct rounding as fabrication would make the
    check unusable."""
    assert check_numeric_grounding("about 0.86", {"pr": 0.8571234}).ok
    assert check_numeric_grounding("about 0.9", {"pr": 0.8571234}).ok
    # But not rounding to something else entirely.
    assert not check_numeric_grounding("about 0.84", {"pr": 0.8571234}).ok


def test_a_fraction_may_be_written_as_a_percentage() -> None:
    """Unit presentation, not arithmetic."""
    ledger = {"deficit_fraction": 0.0823}
    assert check_numeric_grounding("a shortfall of 8.2%", ledger).ok
    assert check_numeric_grounding("a shortfall of 8.23%", ledger).ok
    assert not check_numeric_grounding("a shortfall of 12.5%", ledger).ok


def test_a_percentage_may_be_written_as_a_fraction() -> None:
    assert check_numeric_grounding("0.067 of the gap", {"temperature_loss_pct": 6.7}).ok


def test_a_negative_measurement_may_be_stated_either_way() -> None:
    ledger = {"deficit_fraction": -0.107}
    assert check_numeric_grounding("produced 10.7% more than expected", ledger).ok


def test_arithmetic_on_measured_values_is_not_allowed() -> None:
    """CLAUDE.md puts no LLM in arithmetic.

    Both operands are measured; their difference is not. The subtraction belongs
    in a tool, where it is deterministic and testable, not in a sentence.
    """
    ledger = {"expected_kwh": 16090.0, "measured_kwh": 17812.0}
    report = check_numeric_grounding("a gap of 1722 kWh", ledger)
    assert report.ungrounded == ["1722"]


def test_small_integers_pass_unchecked() -> None:
    """How sentences count things, not claims about the plant."""
    report = check_numeric_grounding(
        "Two causes remain after 14 days and 7 strings were compared.", {}
    )
    assert report.checked == 0
    assert report.ok


def test_a_large_integer_is_still_checked() -> None:
    report = check_numeric_grounding("The plant made 17812 kWh.", {})
    assert report.ungrounded == ["17812"]


def test_thousands_separators_are_understood() -> None:
    assert check_numeric_grounding("17,812 kWh", {"energy_kwh": 17812.1}).ok


def test_dates_and_times_are_not_treated_as_claims() -> None:
    report = check_numeric_grounding(
        "The step happened on 2017-04-23 at 09:15, well before 2017-05-01T00:00.",
        {},
    )
    assert report.checked == 0


def test_channel_and_hypothesis_names_are_not_treated_as_claims() -> None:
    report = check_numeric_grounding(
        "H3 was excluded once string_current_a_7 was compared to shape_hour_09.",
        {},
    )
    assert report.checked == 0


def test_the_report_counts_only_what_it_checked() -> None:
    report = check_numeric_grounding(
        "PR 0.857 over 14 days, with a made-up 0.333.", {"pr": 0.857}
    )
    assert report.checked == 2
    assert report.grounded == ["0.857"]
    assert report.ungrounded == ["0.333"]


def test_an_empty_ledger_grounds_nothing() -> None:
    report = check_numeric_grounding("PR is 0.857.", {})
    assert report.ungrounded == ["0.857"]


def test_a_plain_iterable_of_numbers_works_as_a_ledger() -> None:
    assert check_numeric_grounding("PR is 0.857.", [0.857, 1.2]).ok


def test_numeric_literals_reports_precision_and_percent() -> None:
    literals = numeric_literals("0.857 and 8.2% and 42")
    assert [item[0] for item in literals] == ["0.857", "8.2%", "42"]
    assert [item[2] for item in literals] == [3, 1, 0]
    assert [item[3] for item in literals] == [False, True, False]


def test_prose_with_no_numbers_is_trivially_grounded() -> None:
    report = check_numeric_grounding("Nothing is wrong with this plant.", {"pr": 0.8})
    assert report.ok and report.checked == 0
