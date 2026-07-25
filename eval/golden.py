"""Golden cases: scenario definitions, ground truth, and the blinded split.

Composition follows the brief's §5.2. The single most important property is that
**at least 40% of cases are things that look like faults and are not** — if the
set is mostly clean single faults, a detector that alarms on any deficit scores
well and the evaluation has taught nothing.

`not_enough_evidence` is a scored *correct* answer on the unresolvable cases.
The clipping/curtailment pair is the canonical one: identical ceilings on the
power channel, different cause, different action, and no single measurement
separates them.

**The held-back split is blinded.** Its seeds and windows derive from a
different base seed and are not inspected while iterating. Tuning and held-back
results are always reported separately.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.determinism import DEFAULT_SEED, derive_seed
from src.findings.models import Category

Split = Literal["tuning", "heldback"]

__all__ = ["GoldenCase", "build_golden_set", "load_cases", "write_cases"]


@dataclass
class GoldenCase:
    """One evaluation case: a window, an optional injection, and the truth."""

    id: str
    split: Split
    system_id: int
    start: str
    end: str
    question: str

    # Ground truth.
    expected_category: Category | None
    expected_cause: str | None
    settled: bool
    resolving_measurement: str | None = None
    candidate_causes: list[str] = field(default_factory=list)

    # How the case was made. `None` means untouched real data.
    injection: dict[str, Any] | None = None
    expected_reasoning_elements: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def is_lookalike(self) -> bool:
        """Does this case look like a fault without being one?"""
        return self.expected_category in ("by_design", "not_the_plant")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Scenario templates
# ---------------------------------------------------------------------------
# Windows are chosen to sit inside the dataset's well-covered stretches. The
# mid-2016 telemetry gap is deliberately avoided for injected cases (it is its
# own case) — an injection on top of a real gap confounds two things at once.
_TUNING_WINDOWS = [
    ("2017-02-06", "2017-02-20"),
    ("2017-03-13", "2017-03-27"),
    ("2017-04-10", "2017-04-24"),
    ("2017-05-08", "2017-05-22"),
    ("2017-08-07", "2017-08-21"),
    ("2017-09-04", "2017-09-18"),
    ("2017-10-02", "2017-10-16"),
    ("2017-11-06", "2017-11-20"),
]
_HELDBACK_WINDOWS = [
    ("2016-02-08", "2016-02-22"),
    ("2016-03-14", "2016-03-28"),
    ("2016-09-05", "2016-09-19"),
    ("2016-10-03", "2016-10-17"),
    ("2016-11-07", "2016-11-21"),
    ("2017-01-09", "2017-01-23"),
]


def _case_id(split: Split, index: int) -> str:
    return f"{'G' if split == 'tuning' else 'H'}-{index:03d}"


def _build_for_split(
    split: Split,
    windows: list[tuple[str, str]],
    system_id: int,
    base_seed: int,
) -> list[GoldenCase]:
    cases: list[GoldenCase] = []

    def add(**kwargs: Any) -> None:
        cases.append(
            GoldenCase(id=_case_id(split, len(cases) + 1), split=split, **kwargs)
        )

    def seed_for(label: str) -> int:
        return derive_seed(base_seed, split, label)

    def w_at(i: int) -> tuple[str, str]:
        # Splits carry different numbers of windows, so wrap rather than
        # index out of range. Reusing a window across cases is fine — the
        # injections differ, and the base data is identical either way.
        return windows[i % len(windows)]

    # --- real faults ------------------------------------------------------
    add(
        system_id=system_id,
        start=w_at(0)[0],
        end=w_at(0)[1],
        question="Output is down on this array. What happened?",
        expected_category="fault",
        expected_cause="string_outage",
        settled=True,
        injection={
            "kind": "string_outage",
            "params": {
                "string_index": 3,
                "fraction_lost": 1.0,
                "seed": seed_for("string-full"),
            },
        },
        expected_reasoning_elements=[
            "one string's share of DC current steps down while the others hold",
            "irradiance is unchanged across the step",
            "the loss is the same fraction all day, so it is not shading",
        ],
        notes="Clean single fault. Both the agent and a rules engine should get this.",
    )
    add(
        system_id=system_id,
        start=w_at(1)[0],
        end=w_at(1)[1],
        question="Something changed on this plant last week. What is it?",
        expected_category="fault",
        expected_cause="string_outage",
        settled=True,
        injection={
            "kind": "string_outage",
            "params": {
                "string_index": 5,
                "fraction_lost": 0.45,
                "transition_minutes": 2880,
                "seed": seed_for("string-partial"),
            },
        },
        expected_reasoning_elements=[
            "a ramp rather than a step, consistent with a degrading connection",
            "confined to one string",
        ],
        notes="Partial, ramped loss — harder than the clean step.",
    )
    add(
        system_id=system_id,
        start=w_at(2)[0],
        end=w_at(2)[1],
        question="Morning production looks weak. Is there a fault?",
        expected_category="fault",
        expected_cause="shading",
        settled=True,
        injection={
            "kind": "shading",
            "params": {
                "hour_start": 12,
                "hour_end": 15,
                "depth": 0.4,
                "seed": seed_for("shading"),
            },
        },
        expected_reasoning_elements=[
            "the loss appears only in a fixed band of the day",
            "correlates with time of day, not with irradiance level",
        ],
        notes="Time-of-day dependence is the discriminator against a string fault.",
    )

    # --- recoverable ------------------------------------------------------
    add(
        system_id=system_id,
        start=w_at(3)[0],
        end=w_at(3)[1],
        question="Performance has been slipping for a month. Why?",
        expected_category="recoverable",
        expected_cause="soiling",
        settled=True,
        injection={
            "kind": "soiling",
            "params": {
                "rate_per_day": 0.004,
                "max_loss": 0.2,
                "seed": seed_for("soil"),
            },
        },
        expected_reasoning_elements=[
            "a slow decline that partially recovers after low-insolation days",
            "affects all strings equally, so it is not a string fault",
            "irradiance agrees with clear-sky, so the sensor is not at fault",
        ],
        notes="Recoverable loss. Schedule a wash, do not dispatch a technician.",
    )

    # --- look-alikes (the evaluation) -------------------------------------
    add(
        system_id=system_id,
        start=w_at(4)[0],
        end=w_at(4)[1],
        question="Performance ratio dropped this month. What is wrong?",
        expected_category="by_design",
        expected_cause="seasonal_temperature_derating",
        settled=True,
        injection=None,
        expected_reasoning_elements=[
            "uncorrected performance ratio falls but the temperature-corrected "
            "one does not",
            "module temperature is high, and the gap matches the temperature "
            "coefficient",
        ],
        notes=(
            "REAL DATA, NO INJECTION. Summer derating on a healthy plant — the "
            "most common false alarm there is."
        ),
    )
    add(
        system_id=system_id,
        start=w_at(5)[0],
        end=w_at(5)[1],
        question="Output was well below normal this week. What failed?",
        expected_category="not_the_plant",
        expected_cause="weather",
        settled=True,
        injection=None,
        expected_reasoning_elements=[
            "energy is down but the performance ratio is not",
            "irradiance is down by the same proportion",
        ],
        notes="REAL DATA, NO INJECTION. Cloud, not a fault.",
    )
    add(
        system_id=system_id,
        start=w_at(6)[0],
        end=w_at(6)[1],
        question="The performance ratio has been climbing. Has the plant improved?",
        expected_category="not_the_plant",
        expected_cause="sensor_drift",
        settled=True,
        injection={
            "kind": "sensor_drift",
            "params": {
                "drift_per_day": -0.004,
                "max_drift": -0.18,
                "seed": seed_for("drift"),
            },
        },
        expected_reasoning_elements=[
            "the performance ratio rises, which a real loss cannot do",
            "measured irradiance falls below the clear-sky envelope while output holds",
        ],
        notes=(
            "The signature is the OPPOSITE of array soiling: a soiled sensor "
            "makes the plant look better, not worse."
        ),
    )
    add(
        system_id=system_id,
        start=w_at(7)[0],
        end=w_at(7)[1],
        question="A month of production is missing. What happened to the plant?",
        expected_category="not_the_plant",
        expected_cause="telemetry_gap",
        settled=True,
        injection={
            "kind": "telemetry_gap",
            "params": {"seed": seed_for("gap")},
        },
        expected_reasoning_elements=[
            "irradiance kept recording while the power channel did not",
            "the apparent loss is a data problem, not a plant problem",
        ],
        notes="Modelled on the real 35-day gap in this dataset in mid-2016.",
    )

    # --- unresolvable by design ------------------------------------------
    add(
        system_id=system_id,
        start=w_at(0)[0],
        end=w_at(0)[1],
        question="Midday output is flat-topped. Is the inverter faulty?",
        expected_category=None,
        expected_cause=None,
        settled=False,
        candidate_causes=["clipping", "curtailment"],
        resolving_measurement=(
            "compare the same period last year, or check the grid operator's "
            "dispatch log for a curtailment instruction"
        ),
        injection={
            "kind": "curtailment",
            "params": {"ceiling_kw": 150.0, "seed": seed_for("curtail")},
        },
        expected_reasoning_elements=[
            "a flat ceiling is present",
            "clipping and curtailment produce the same ceiling, so the absence "
            "of one does not exclude the other",
            "no available measurement separates them",
        ],
        notes=(
            "THE canonical unresolvable pair. Committing to either answer here "
            "is wrong even if it happens to be right."
        ),
    )

    # --- fault hidden inside a look-alike ---------------------------------
    add(
        system_id=system_id,
        start=w_at(1)[0],
        end=w_at(1)[1],
        question="A cloudy stretch, but something still looks off. Anything real?",
        expected_category="fault",
        expected_cause="string_outage",
        settled=True,
        injection={
            "kind": "string_outage",
            "params": {
                "string_index": 2,
                "fraction_lost": 0.8,
                "seed": seed_for("hidden"),
            },
        },
        expected_reasoning_elements=[
            "low output is mostly explained by low irradiance",
            "but one string's share is still depressed, which weather cannot do",
        ],
        notes=(
            "A real fault inside a period that already looks bad for innocent "
            "reasons. This is where a threshold detector fails."
        ),
    )
    return cases


def build_golden_set(system_id: int = 4902) -> list[GoldenCase]:
    """Build both splits.

    The held-back split derives from a *different* base seed and different
    windows, and is not looked at while iterating — otherwise the gap between
    the two is not evidence of generalisation.
    """
    tuning = _build_for_split("tuning", _TUNING_WINDOWS, system_id, DEFAULT_SEED)
    heldback = _build_for_split(
        "heldback", _HELDBACK_WINDOWS, system_id, derive_seed(DEFAULT_SEED, "heldback")
    )
    return tuning + heldback


def write_cases(cases: Iterable[GoldenCase], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(cases)
    with path.open("w", encoding="utf-8") as handle:
        for case in rows:
            handle.write(json.dumps(case.to_dict(), sort_keys=True) + "\n")
    return len(rows)


def load_cases(path: Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(GoldenCase(**json.loads(line)))
    return cases


def composition(cases: list[GoldenCase]) -> pd.DataFrame:
    """Summarise the set — the check that it is adversarial enough to be useful."""
    rows = []
    for split in ("tuning", "heldback"):
        subset = [c for c in cases if c.split == split]
        if not subset:
            continue
        lookalikes = sum(1 for c in subset if c.is_lookalike)
        rows.append(
            {
                "split": split,
                "cases": len(subset),
                "faults": sum(1 for c in subset if c.expected_category == "fault"),
                "recoverable": sum(
                    1 for c in subset if c.expected_category == "recoverable"
                ),
                "look_alikes": lookalikes,
                "look_alike_share": round(lookalikes / len(subset), 3),
                "unresolvable": sum(1 for c in subset if not c.settled),
                "real_data_no_injection": sum(1 for c in subset if c.injection is None),
            }
        )
    return pd.DataFrame(rows)
