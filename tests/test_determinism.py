"""Determinism guarantees for the non-LLM half of the system.

Scope reminder: these cover physics, tools, injection and retrieval. Nothing
here claims the LLM path is reproducible — it is not, and CLAUDE.md says so.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from src.determinism import (
    DEFAULT_SEED,
    derive_seed,
    run_rng,
    seed_everything,
    stable_hash,
)


def test_same_seed_gives_identical_draws() -> None:
    a = run_rng(1234).normal(size=1000)
    b = run_rng(1234).normal(size=1000)
    assert np.array_equal(a, b)


def test_different_seeds_diverge() -> None:
    assert not np.array_equal(run_rng(1).normal(size=100), run_rng(2).normal(size=100))


def test_generators_are_independent_of_global_state() -> None:
    # The whole point of an explicit Generator: an unrelated global seed call
    # between two runs must not change the fault injector's output.
    first = run_rng(99).normal(size=50)
    np.random.seed(7)
    _ = np.random.normal(size=1000)
    second = run_rng(99).normal(size=50)
    assert np.array_equal(first, second)


def test_generator_advances_within_a_run() -> None:
    rng = run_rng(5)
    assert not np.array_equal(rng.normal(size=10), rng.normal(size=10))


def test_stable_hash_is_deterministic_and_order_sensitive() -> None:
    assert stable_hash("a", 1) == stable_hash("a", 1)
    assert stable_hash("a", 1) != stable_hash(1, "a")


def test_stable_hash_separator_prevents_collisions() -> None:
    # Without a separator, ("ab", "c") and ("a", "bc") would hash alike, and
    # two different prompts would share one LLM cache entry.
    assert stable_hash("ab", "c") != stable_hash("a", "bc")


def test_stable_hash_survives_a_fresh_interpreter() -> None:
    # Python's built-in hash() is salted per process, so it cannot key a cache
    # that must survive a restart. This is the property that matters.
    code = (
        "from src.determinism import stable_hash; "
        "print(stable_hash('planner', 'prompt-v1'))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1


def test_derive_seed_is_stable_distinct_and_in_range() -> None:
    assert derive_seed(DEFAULT_SEED, "case", "G-014") == derive_seed(
        DEFAULT_SEED, "case", "G-014"
    )
    assert derive_seed(DEFAULT_SEED, "case", "G-014") != derive_seed(
        DEFAULT_SEED, "case", "G-015"
    )
    for label in ("G-001", "G-042", "G-080"):
        assert 0 <= derive_seed(DEFAULT_SEED, label) < 2**31 - 1


def test_derived_seeds_produce_independent_streams() -> None:
    a = run_rng(derive_seed(DEFAULT_SEED, "G-014")).normal(size=200)
    b = run_rng(derive_seed(DEFAULT_SEED, "G-015")).normal(size=200)
    assert not np.array_equal(a, b)


def test_seed_everything_is_reproducible() -> None:
    import random

    seed_everything(42)
    first = (random.random(), float(np.random.random()))
    seed_everything(42)
    assert first == (random.random(), float(np.random.random()))
