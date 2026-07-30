"""Domain knowledge, loaded and validated but never *executed*.

The two YAML files describe what each cause does to a plant's telemetry and
what separates each pair of look-alikes. Nothing in this package compares a
signature against a measurement, scores a match, or returns a cause. It loads,
validates, and formats text for retrieval — and that restraint is the point.

If a signature carried a threshold and any code compared it to a tool result,
the evaluation would be measuring whether this file agrees with
`simulator/injectors.py`. Two files written by the same person in the same week
agree with each other almost perfectly, and the resulting accuracy figure would
be near 100% and worth nothing.

The validator below enforces the rule mechanically: a signature or test that
reads as a decision rule fails to load. Step 9 replaces `brief_for` with real
retrieval over a document corpus; the contract stays the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "DistinguishingTest",
    "KnowledgeBase",
    "Signature",
    "category_for",
    "cause_vocabulary",
    "load_knowledge",
]

HERE = Path(__file__).resolve().parent

# A signature may name a physical constant ("roughly 0.4% per degree above
# 25 °C") because that is physics. It may not state a decision rule. These are
# the shapes a rule takes when written in prose.
_RULE_PATTERNS = (
    re.compile(r"[<>]=?\s*-?\d"),
    re.compile(r"\b(?:greater|less|more|fewer)\s+than\s+-?\d"),
    re.compile(
        r"\b(?:above|below|over|under|exceeds?|at\s+least)\s+-?\d+(?:\.\d+)?\s*%"
    ),
    re.compile(r"\bthreshold\b", re.IGNORECASE),
)


def _scan_for_rules(where: str, text: str) -> None:
    for pattern in _RULE_PATTERNS:
        match = pattern.search(text)
        if match:
            raise ValueError(
                f"{where} contains what reads as a decision rule "
                f"({match.group(0)!r}). Knowledge describes shapes and "
                "relationships; thresholds live in tools and config where they "
                "are deterministic and testable. A threshold here would make the "
                "evaluation measure whether this file agrees with the injector."
            )


@dataclass(frozen=True)
class Signature:
    """What one cause does to a plant's telemetry."""

    key: str
    category: str
    plain_name: str
    what_happens: str
    looks_like: tuple[str, ...]
    distinguishing: dict[str, str]
    action: str
    typical_cost: str = ""

    def as_text(self) -> str:
        lines = [
            f"### {self.key} ({self.plain_name}) — {self.category}",
            self.what_happens.strip(),
            "Looks like:",
            *(f"  - {item.strip()}" for item in self.looks_like),
        ]
        lines.extend(
            f"Told apart from {other}: {how.strip()}"
            for other, how in self.distinguishing.items()
        )
        lines.append(f"Action: {self.action}")
        return "\n".join(lines)


@dataclass(frozen=True)
class DistinguishingTest:
    """The observation that separates one pair of look-alikes."""

    between: tuple[str, str]
    why_confused: str
    how: str
    separable: bool
    measurement: str | None = None
    also: tuple[str, ...] = ()
    external_evidence: str | None = None
    the_mistake: str = ""

    @property
    def key(self) -> frozenset[str]:
        return frozenset(self.between)

    def as_text(self) -> str:
        lines = [
            f"### {' vs '.join(self.between)}",
            f"Confused because: {self.why_confused.strip()}",
        ]
        if self.separable:
            lines.append(f"Separated by: {self.measurement} — {self.how.strip()}")
            if self.also:
                lines.append(f"Supporting: {', '.join(self.also)}")
        else:
            lines.append("NOT SEPARABLE from plant telemetry.")
            lines.append(self.how.strip())
            if self.external_evidence:
                lines.append(
                    f"External evidence needed: {self.external_evidence.strip()}"
                )
        if self.the_mistake:
            lines.append(f"The usual mistake: {self.the_mistake.strip()}")
        return "\n".join(lines)


@dataclass(frozen=True)
class KnowledgeBase:
    signatures: dict[str, Signature]
    tests: tuple[DistinguishingTest, ...]

    def signature(self, cause: str) -> Signature | None:
        return self.signatures.get(cause)

    def test_for(self, a: str, b: str) -> DistinguishingTest | None:
        wanted = frozenset((a, b))
        return next((t for t in self.tests if t.key == wanted), None)

    def unresolvable_pairs(self) -> tuple[tuple[str, str], ...]:
        """Pairs no measurement in this tool set can separate.

        Exposed so a test can check that the golden set's unresolvable cases and
        the knowledge base agree about which pairs those are. A disagreement
        means one of the two is wrong, and the abstention metric would then be
        scoring against the wrong list.
        """
        return tuple(t.between for t in self.tests if not t.separable)

    def match(self, phrases: list[str], limit: int = 6) -> list[str]:
        """Map free-text candidate causes onto signature keys.

        Deliberately crude token overlap against each signature's key and plain
        name. This is *retrieval* — matching text to documents — not matching a
        measurement to a verdict, and the distinction is what keeps it on the
        right side of the architecture rule. Step 9 replaces it with BM25 plus a
        dense retriever plus a reranker and measures whether that is worth
        anything.
        """
        scored: dict[str, int] = {}
        for phrase in phrases:
            words = {w for w in re.split(r"[^a-z]+", phrase.lower()) if len(w) > 3}
            if not words:
                continue
            for key, signature in self.signatures.items():
                vocabulary = {
                    w
                    for w in re.split(
                        r"[^a-z]+", f"{key} {signature.plain_name}".lower()
                    )
                    if len(w) > 3
                }
                overlap = len(words & vocabulary)
                if overlap:
                    scored[key] = max(scored.get(key, 0), overlap)
        return [k for k, _ in sorted(scored.items(), key=lambda kv: -kv[1])][:limit]

    def brief_for(self, causes: list[str]) -> str:
        """The knowledge relevant to a set of candidate causes.

        Step 9 replaces this with retrieval over a real corpus and measures
        whether it moves accuracy. Until then this is a deliberate stand-in with
        the same shape: text handed to the agent as evidence, never matched
        against a measurement by code.
        """
        wanted = [c for c in causes if c in self.signatures]
        blocks = [self.signatures[c].as_text() for c in wanted]
        seen: set[frozenset[str]] = set()
        for first in wanted:
            for second in wanted:
                if first >= second:
                    continue
                test = self.test_for(first, second)
                if test is not None and test.key not in seen:
                    seen.add(test.key)
                    blocks.append(test.as_text())
        return "\n\n".join(blocks) if blocks else "(no knowledge for these causes)"


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


@lru_cache(maxsize=2)
def load_knowledge(root: Path | None = None) -> KnowledgeBase:
    """Load and validate both knowledge files."""
    base = root or HERE

    raw_signatures = _load_yaml(base / "fault_signatures.yaml")["signatures"]
    signatures: dict[str, Signature] = {}
    for key, entry in raw_signatures.items():
        looks_like = tuple(entry["looks_like"])
        distinguishing = dict(entry.get("distinguishing", {}))
        for text in (*looks_like, *distinguishing.values(), entry["what_happens"]):
            _scan_for_rules(f"signature {key!r}", text)
        signatures[key] = Signature(
            key=key,
            category=entry["category"],
            plain_name=entry["plain_name"],
            what_happens=entry["what_happens"],
            looks_like=looks_like,
            distinguishing=distinguishing,
            action=entry["action"],
            typical_cost=entry.get("typical_cost", ""),
        )

    raw_tests = _load_yaml(base / "distinguishing_tests.yaml")["pairs"]
    tests: list[DistinguishingTest] = []
    for entry in raw_tests:
        between = tuple(entry["between"])
        if len(between) != 2:
            raise ValueError(
                f"a distinguishing test needs exactly two causes; got {between}"
            )
        _scan_for_rules(f"test {between}", entry["how"])
        separable = bool(entry.get("separable", True))
        if separable and not entry.get("measurement"):
            raise ValueError(
                f"test {between} claims the pair is separable but names no "
                "measurement that would do it"
            )
        if not separable and not entry.get("external_evidence"):
            raise ValueError(
                f"test {between} says the pair cannot be separated but names no "
                "external evidence that would settle it. An abstention without a "
                "next step is useless."
            )
        tests.append(
            DistinguishingTest(
                between=(str(between[0]), str(between[1])),
                why_confused=entry["why_confused"],
                how=entry["how"],
                separable=separable,
                measurement=entry.get("measurement"),
                also=tuple(entry.get("also", [])),
                external_evidence=entry.get("external_evidence"),
                the_mistake=entry.get("the_mistake", ""),
            )
        )

    unknown = {
        cause for test in tests for cause in test.between if cause not in signatures
    }
    if unknown:
        raise ValueError(
            "distinguishing tests reference causes with no signature: "
            + ", ".join(sorted(unknown))
        )

    return KnowledgeBase(signatures=signatures, tests=tuple(tests))


def cause_vocabulary() -> list[str]:
    """The canonical cause names every node must use for the same thing.

    A *label*, not an explanation. Free text meant the same cause arrived as
    `shading` from one node and a whole sentence from another, and anything
    comparing them by name — scoring, retrieval, the look-alike checklist —
    silently missed. The explanation belongs in prose fields; this says *which*
    cause, in one spelling.

    Read from the signatures rather than restated anywhere, so a cause added to
    the knowledge base is immediately sayable by every node and no node can
    drift from the vocabulary the agent is shown.
    """
    return sorted(load_knowledge().signatures)


def category_for(cause: str) -> str | None:
    """Which bucket a cause belongs to. `None` for a cause not in the base.

    The category is a *property of the cause*, written down once in
    `fault_signatures.yaml`: soiling is `recoverable`, sensor_drift is
    `not_the_plant`, shading is `fault`. It is a lookup, not a judgement.

    The synthesiser was choosing both independently and could therefore
    contradict the knowledge base — and did, once in eight cases: the
    eight-case run scored 0.875 on cause and 0.750 on category, and the whole
    gap is one answer that named the right cause and filed it in the wrong
    bucket. That is not a reasoning failure; it is a dictionary lookup done by
    hand. CLAUDE.md already keeps the LLM out of arithmetic, thresholds,
    scoring and data scope, and this belongs on that list.
    """
    signature = load_knowledge().signatures.get(cause)
    return str(signature.category) if signature else None
