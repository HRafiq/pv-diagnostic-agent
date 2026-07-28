"""Is every number in the answer a number some tool actually measured?

The §5.6 target is "zero fabricated numerics". That is only meaningful if it can
be checked, so this module checks it: pull every numeric literal out of the
model's prose and look for it in the union of the tool results' provenance
ledgers. A figure that is not there did not come from a measurement.

Three allowances, and all three are deliberate:

* **Percentage presentation.** A ledger holding ``deficit_fraction = 0.0823`` and
  prose saying "8.2%" is the same measurement in different clothes, so a factor
  of 100 either way is accepted.
* **Small integers.** Values up to 24 pass unchecked. They are how sentences
  count things — "two causes remain", "three days", "09:00" — and treating them
  as claims would bury the real finding in noise. Anything that carries
  information about the plant is larger or has a decimal point.
* **Quotation.** A figure the agent was *shown* may be repeated. See below.

What is *not* allowed is arithmetic. If the ledger holds an expected and a
measured energy and the prose states their difference, that difference is
flagged, because CLAUDE.md puts no LLM in arithmetic: the subtraction belongs in
a tool, where it is deterministic and testable, not in a sentence.

Why quotation had to be added
-----------------------------

The check began as "every figure must be in some tool's provenance ledger". That
is narrower than the guarantee it stands for. The guarantee is *no invented
numbers*; the ledger is only one of the places a legitimate number comes from.
The agent is also shown, and expected to use:

* **tool prose** — a ``ToolResult`` renders its ``summary``, ``caveats`` and
  ``labels`` into the digest, and those carry figures the ledger does not. A
  caveat reading "a healthy inverter sits near 0.96-0.98 at load" puts two real,
  measured-domain numbers in front of the model that it may not repeat.
* **the brief** — the window under investigation, the record's span, the plant's
  capacity.
* **the knowledge base** — the signature for ``seasonal_temperature_derating``
  states that output falls about 0.4% per °C above 25°C. Quoting the physics it
  was handed was scored as fabrication.

That last one is not a hypothetical. It ran: an evaluation case whose true cause
*is* seasonal derating was answered correctly on the first cycle, then vetoed for
"ungrounded" figures — among them ``2014``, a year, three times — and after two
further cycles and ten minutes ended in "not enough evidence". A false positive
here is not cosmetic. `unsupported_claims` is a hard veto, so every one of these
costs a correct answer.

So a second reference source is accepted: the prose the agent was shown, passed
as ``quoted_from``. Numbers are pulled from it *without* the masks below, so a
window written ``2014-05-01`` grounds a sentence that says "since 2014" — while a
year the record does not contain is still flagged. Callers must pass the source
material and never the draft, or the answer would ground itself.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

__all__ = [
    "GroundingReport",
    "check_numeric_grounding",
    "numeric_literals",
    "quotable_values",
]

# Removed before scanning: their digits are structure, not measurements.
_MASKS = (
    re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?"),  # ISO dates
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),  # clock times
    re.compile(r"\bH\d+\b"),  # hypothesis ids
    re.compile(r"\b[A-Za-z_]+_\d+\b"),  # channel and tool names
    re.compile(r"\bstep \d+\b", re.IGNORECASE),
)

# The leading lookbehind stops a hyphen between two numbers from being read as a
# sign. "a healthy inverter sits near 0.96-0.98" is a range, and parsing it as
# 0.96 and *minus* 0.98 invented a negative figure that nothing could ground —
# the check manufacturing its own violation.
_NUMBER = re.compile(r"(?<![\d.])[-+]?\d[\d,]*(?:\.\d+)?\s*%?")

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
            f"the figure {literal} does not appear in any tool result, and was "
            "not quoted from anything the investigation was shown"
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


def quotable_values(*texts: str) -> list[float]:
    """Every number the agent was shown, as plain floats.

    Deliberately *not* masked. `numeric_literals` strips dates, clock times and
    channel names because on the answer side their digits are structure rather
    than claims; here the opposite is wanted. A brief that says the window runs
    from ``2014-05-01`` is exactly what entitles the answer to say "since 2014",
    so the year, the month and the day are all admitted as quotable.

    Small integers are admitted too, though the answer side never checks them —
    keeping the two extractions independent means a later change to the free
    integer ceiling cannot silently narrow what may be quoted.
    """
    out: list[float] = []
    for text in texts:
        for match in _NUMBER.finditer(text or ""):
            body = match.group(0).strip().rstrip("%").strip().replace(",", "")
            if not body or body in {"-", "+"}:
                continue
            try:
                out.append(float(body))
            except ValueError:  # pragma: no cover - regex should prevent this
                continue
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
    *,
    quoted_from: str | Iterable[str] = (),
    quotable: Iterable[float] = (),
) -> GroundingReport:
    """Check every number in `text` against measured values.

    Args:
        ledger: The union of `ToolResult.values` across the run, or any iterable
            of measured numbers.
        quoted_from: Prose the agent was shown — the brief, the knowledge it was
            given, the tool digests. Numbers appearing here may be repeated.
            Never pass the draft answer: it would ground itself.
        quotable: The same allowance, pre-extracted. `Synthesis` carries this so
            the critic checks a draft against exactly what the synthesiser
            showed the model, rather than a reconstruction that can drift.
    """
    measured = (
        [float(v) for v in ledger.values()]
        if isinstance(ledger, Mapping)
        else [float(v) for v in ledger]
    )
    if isinstance(quoted_from, str):
        quoted_from = (quoted_from,)
    measured.extend(quotable_values(*quoted_from))
    measured.extend(float(v) for v in quotable)
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
