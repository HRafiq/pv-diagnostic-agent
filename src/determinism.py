"""Seeding, so that "same seed, same input, same output" is actually true.

Scope note (CLAUDE.md): this makes the *deterministic* half of the system
reproducible — physics, detectors, tools, fault injection, retrieval. It cannot
make the LLM half reproducible. `temperature=0` no longer exists as a parameter
on current Claude models, and even where it did it never guaranteed identical
sampling. The LLM path is handled by response caching plus reporting metrics as
mean ± spread over repeated runs.

Prefer `run_rng()` over the module-level `numpy.random` functions. A global
`np.random.seed()` is a shared mutable that any import can disturb; an explicit
`Generator` threaded through the call is the reason two runs of the fault
injector produce byte-identical output.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

__all__ = ["DEFAULT_SEED", "derive_seed", "run_rng", "seed_everything", "stable_hash"]

DEFAULT_SEED = 20260101


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed every global RNG we can reach.

    A backstop for third-party code that reaches for a global RNG. First-party
    code should take an explicit `Generator` from `run_rng()` instead.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """An independent, explicitly-seeded generator.

    Threaded through the fault injector and any sampling code so that the same
    seed yields byte-identical output regardless of what else has run.
    """
    return np.random.default_rng(seed)


def stable_hash(*parts: object) -> str:
    """A hash that is stable across processes and Python versions.

    Python's built-in `hash()` is salted per process, so it cannot key a cache
    that must survive a restart. Used for the LLM response cache key and for
    the corpus snapshot hash recorded on every eval run.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\x00")  # unambiguous separator
    return digest.hexdigest()


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a child seed from a base seed and some labels.

    Lets each golden case get its own reproducible seed without hand-maintaining
    a table of magic numbers: ``derive_seed(DEFAULT_SEED, "case", "G-014")``.
    """
    digest = stable_hash(base_seed, *parts)
    return int(digest[:16], 16) % (2**31 - 1)
