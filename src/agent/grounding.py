"""Is every number in the answer a number some tool actually measured?

The §5.6 target is "zero fabricated numerics". That is only meaningful if it can
be checked, so this module checks it: pull every numeric literal out of the
model's prose and look for it in the union of the tool results' provenance
ledgers. A figure that is not there did not come from a measurement.

Two allowances, and both are deliberate:

* **Percentage presentation.** A ledger holding ``deficit_fraction = 0.0823`` and
  prose saying "8.2%" is the same measurement in different clothes, so a factor
  of 100 either way is accepted.
* **Small integers.** Values up to 24 pass unchecked. They are how sentences
  count things — "two causes remain", "three days", "09:00" — and treating them
  as claims would bury the real finding in noise. Anything that carries
  information about the plant is larger or has a decimal point.

What is *not* allowed is arithmetic. If the ledger holds an expected and a
measured energy and the prose states their difference, that difference is
flagged, because CLAUDE.md puts no LLM in arithmetic: the subtraction belongs in
a tool, where it is deterministic and testable, not in a sentence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

__all__ = ["GroundingReport", "check_numeric_grounding", "numeric_literals"]

# Removed before scanning: their digits are structure, not measurements.
_MASKS = (
    re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?"),  # ISO dates
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),  # clock times
    re.compile(r"\bH\d+\b"),  # hypothesis ids
    re.compile(r"\b[A-Za-z_]+_\d+\b"),  # channel and tool names
    re.compile(r"\bstep \d+\b", re.IGNORECASE),
)

_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?\s*%?")

# Below this, an integer is sentence furniture rather than a claim about the
# plant. 24 covers hours of the day and any plausible count of strings,
# hypotheses or days in a short window.
_FREE_INTEGER_CEILING = 24


@dataclass(frozen=True)
class GroundingReport:
    """Which numbers in a piece of prose trace back to a measurement."""

    checked: int = 0
    grounded: list[str] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.ungrounded

    def as_claims(self) -> list[str]:
        """Phrasing for `CriticVerdict.unsupported_claims`."""
        return [
            f"the figure {literal} does not appear in any tool result"
            for literal in self.ungrounded
        ]


def numeric_literals(text: str) -> list[tuple[str, float, int, bool]]:
    """Every numeric literal in `text`, as (literal, value, decimals, is_pct)."""
    masked = text
    for pattern in _MASKS:
        masked = pattern.sub(lambda m: " " * len(m.group(0)), masked)

    out: list[tuple[str, float, int, bool]] = []
    for match in _NUMBER.finditer(masked):
        literal = match.group(0).strip()
        is_pct = literal.endswith("%")
        body = literal.rstrip("%").strip().replace(",", "")
        if not body or body in {"-", "+"}:
            continue
        try:
            value = float(body)
        except ValueError:  # pragma: no cover - regex should prevent this
            continue
        decimals = len(body.split(".")[1]) if "." in body else 0
        out.append((literal, value, decimals, is_pct))
    return out


def _tolerance(decimals: int, value: float) -> float:
    """Half a unit in the last displayed place, plus a whisker for float noise.

    A figure printed as "0.86" legitimately stands for anything in
    [0.855, 0.865), so the check has to accept the whole interval or it would
    flag correct rounding as fabrication.
    """
    return 0.5 * 10.0 ** (-decimals) + 1e-9 + abs(value) * 1e-9


def check_numeric_grounding(
    text: str,
    ledger: Mapping[str, float] | Iterable[float],
) -> GroundingReport:
    """Check every number in `text` against measured values.

    Args:
        ledger: The union of `ToolResult.values` across the run, or any iterable
            of measured numbers.
    """
    measured = (
        [float(v) for v in ledger.values()]
        if isinstance(ledger, Mapping)
        else [float(v) for v in ledger]
    )
    # Percentage presentation is a unit change, not arithmetic, so admit both
    # framings of every measured value up front.
    candidates: list[float] = []
    for value in measured:
        candidates.extend((value, value * 100.0, value / 100.0))
        candidates.extend((-value, -value * 100.0, -value / 100.0))

    grounded: list[str] = []
    ungrounded: list[str] = []
    checked = 0

    for literal, value, decimals, _is_pct in numeric_literals(text):
        if decimals == 0 and abs(value) <= _FREE_INTEGER_CEILING:
            continue
        checked += 1
        tol = _tolerance(decimals, value)
        if any(abs(value - candidate) <= tol for candidate in candidates):
            grounded.append(literal)
        else:
            ungrounded.append(literal)

    return GroundingReport(checked=checked, grounded=grounded, ungrounded=ungrounded)
