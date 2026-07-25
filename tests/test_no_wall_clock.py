"""Mechanically enforce the no-wall-clock rule.

A stray `datetime.now()` in `src/` does not raise. It silently compares a
replayed 2019 dataset against a 2026 "today", every trailing window comes back
empty, and the detector quietly finds nothing. That failure mode is invisible in
review, so it is checked here instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories where the replay clock is the only legitimate time source.
GUARDED = ("src", "simulator", "eval")

# (module, attribute) pairs that read the real calendar.
BANNED_CALLS = {
    ("datetime", "now"),
    ("datetime", "utcnow"),
    ("datetime", "today"),
    ("date", "today"),
    ("time", "time"),
    ("pd", "Timestamp.now"),
    ("pandas", "Timestamp.now"),
    ("Timestamp", "now"),
}

# time.monotonic measures a duration, not a calendar instant, so it cannot leak
# a real date into a computation. ReplayClock uses it for accelerated replay.
ALLOWED = {("time", "monotonic"), ("time", "perf_counter")}


def _guarded_files() -> list[Path]:
    files: list[Path] = []
    for directory in GUARDED:
        root = REPO_ROOT / directory
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))
    return files


def _dotted(node: ast.AST) -> str | None:
    """Render an attribute chain like `pd.Timestamp.now` back to a string."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def test_no_wall_clock_reads_in_guarded_packages() -> None:
    offences: list[str] = []

    for path in _guarded_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted(node.func)
            if dotted is None:
                continue
            head, _, tail = dotted.partition(".")
            if (head, tail) in ALLOWED:
                continue
            if (head, tail) in BANNED_CALLS:
                rel = path.relative_to(REPO_ROOT)
                offences.append(f"{rel}:{node.lineno}  {dotted}()")

    assert not offences, (
        "wall-clock reads found in guarded packages — use the injected Clock "
        "(src/clock.py) instead:\n  " + "\n  ".join(offences)
    )


@pytest.mark.parametrize("directory", GUARDED)
def test_guarded_directories_exist(directory: str) -> None:
    # If a directory is renamed, the guard above silently stops checking it.
    assert (REPO_ROOT / directory).is_dir()
