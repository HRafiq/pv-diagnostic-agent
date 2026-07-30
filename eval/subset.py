"""Run part of the golden set, and say what that costs you.

An agent evaluation is 43 cases at a few minutes and roughly a dollar each. That
is the right price for a *result* and the wrong price for a question like "did
the critic fix work?", which two cases answer. Iterating on the full set also
means every network blip costs the whole run.

So subsets are supported — and reported. The distinction this module exists to
keep is between a **debugging run** and a **score**:

* A debugging run is a subset. It tells you whether the plumbing works and what
  the agent does. It is not a number anybody may quote.
* A score is the whole split, because several headline metrics are defined over
  case classes that a subset can silently drop.

The second half is what makes the first half safe. `warn_about` names, in
words, which metrics the chosen subset cannot support — `correct_abstention_rate`
needs unresolvable cases, `false_alarm_rate` needs look-alikes, and macro-F1
over three categories is not macro-F1 over four. `eval/metrics.py` already omits
a metric with no cases behind it rather than printing zero; this says the same
thing before the run instead of after.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.determinism import DEFAULT_SEED, run_rng

__all__ = ["describe", "select", "warn_about"]


def _stratum(case: Any) -> tuple[str, bool]:
    """The two properties the headline metrics are defined over.

    Category drives macro-F1 and the false-alarm rate; `settled` drives correct
    abstention. Sampling within these keeps a subset able to answer the same
    questions as the whole split, in miniature.
    """
    return (str(case.expected_category), bool(case.settled))


def select(
    cases: list[Any],
    *,
    only: str | None = None,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = DEFAULT_SEED,
) -> list[Any]:
    """Choose the cases to run.

    Args:
        only: Comma-separated case ids. Exact, ordered, for debugging one thing.
        limit: The first N in file order. Cheap and deterministic, but the file
            is ordered by fault type, so the first N are not a cross-section —
            `warn_about` will say so.
        sample: N drawn *stratified* by category and settledness, so a small run
            still contains look-alikes and unresolvable cases in roughly the
            proportions the whole split has. This is the one to use when a
            subset needs to be informative rather than merely quick.
        seed: Fixes the sample. The same seed and split give the same cases, so
            two runs are comparable.
    """
    chosen = [c for c in cases]

    if only:
        wanted = [name.strip() for name in only.split(",") if name.strip()]
        by_id = {str(c.id): c for c in cases}
        missing = [name for name in wanted if name not in by_id]
        if missing:
            raise ValueError(
                f"no such case(s): {', '.join(missing)}. "
                f"Ids look like {sorted(by_id)[0]}."
            )
        return [by_id[name] for name in wanted]

    if sample is not None:
        chosen = _stratified(chosen, sample, seed)
    elif limit is not None:
        chosen = chosen[:limit]

    return chosen


def _stratified(cases: list[Any], size: int, seed: int) -> list[Any]:
    """Draw `size` cases keeping the split's shape.

    Largest-remainder allocation: each stratum gets its proportional share, and
    the leftover places go to the strata with the largest fractional parts. Ties
    break on the stratum key, so the result depends only on the seed.
    """
    if size >= len(cases):
        return list(cases)

    strata: dict[tuple[str, bool], list[Any]] = defaultdict(list)
    for case in cases:
        strata[_stratum(case)].append(case)

    total = len(cases)
    exact = {key: len(group) * size / total for key, group in strata.items()}
    counts = {key: int(value) for key, value in exact.items()}

    # Every stratum that exists at all gets at least one place, so a rare class
    # — one `by_design` case in forty-three — cannot vanish from the sample and
    # take its metric with it.
    for key in strata:
        counts[key] = max(counts[key], 1)

    while sum(counts.values()) > size:
        # Trim the most over-represented stratum that can spare a place.
        spare = [k for k, n in counts.items() if n > 1]
        if not spare:
            break
        counts[max(spare, key=lambda k: (counts[k] - exact[k], k))] -= 1
    while sum(counts.values()) < size:
        counts[max(strata, key=lambda k: (exact[k] - counts[k], k))] += 1

    rng = run_rng(seed)
    drawn: list[Any] = []
    for key in sorted(strata):
        group = strata[key]
        take = min(counts[key], len(group))
        picks = rng.choice(len(group), size=take, replace=False)
        drawn.extend(group[int(i)] for i in sorted(picks))

    # Back into the split's own order, so traces and logs read predictably.
    order = {id(c): n for n, c in enumerate(cases)}
    return sorted(drawn, key=lambda c: order[id(c)])


def describe(cases: list[Any]) -> str:
    """One line: how many cases, and of what."""
    categories = Counter(str(c.expected_category) for c in cases)
    unresolvable = sum(1 for c in cases if not c.settled)
    lookalikes = sum(1 for c in cases if c.is_lookalike)
    shape = ", ".join(f"{n} {name}" for name, n in sorted(categories.items()))
    return (
        f"{len(cases)} cases — {shape}; "
        f"{lookalikes} look-alike(s), {unresolvable} unresolvable"
    )


def warn_about(subset: list[Any], full: list[Any]) -> str | None:
    """What this subset cannot tell you. `None` when it is the whole split.

    Returned rather than printed so the caller decides where it goes; every
    command prints it above the results, because a number is quoted far more
    often than the paragraph under it.
    """
    if len(subset) >= len(full):
        return None

    lines = [
        f"SUBSET: {len(subset)} of {len(full)} cases. This is a debugging run, "
        "not a score — do not quote these numbers.",
    ]

    if not any(not c.settled for c in subset):
        lines.append(
            "  - no unresolvable cases, so correct 'not enough evidence' cannot "
            "be measured at all"
        )
    if not any(c.is_lookalike for c in subset):
        lines.append(
            "  - no look-alikes, so the false-alarm rate cannot be measured — "
            "and that is the headline metric"
        )

    missing = {str(c.expected_category) for c in full} - {
        str(c.expected_category) for c in subset
    }
    if missing:
        lines.append(
            f"  - no {', '.join(sorted(missing))} case(s), so overall accuracy "
            "is averaged over fewer classes than the baseline's and the two are "
            "not comparable"
        )

    return "\n".join(lines)
