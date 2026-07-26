"""Mechanically enforce that no source file is hidden from git.

A bare directory name in `.gitignore` — `findings/`, `build/`, `lib/` — matches
at *every* depth, not just the repo root. `findings/` was there for the runtime
findings output directory and it also swallowed `src/findings/`, so two modules
were never committed. Everything kept working locally and the clone was broken.

That failure is invisible: `git status` is clean, the tests pass, the push
succeeds. It only surfaces when somebody else clones. So it is checked here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that hold hand-written source. Everything in them belongs in git.
SOURCE_DIRS = ("src", "simulator", "eval", "dashboard", "tests", "config")

# Extensions that carry source or configuration rather than build output.
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".html", ".js", ".css", ".jinja"}

# Paths inside the source dirs that are legitimately generated, not authored.
GENERATED = ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")


def _is_git_repo() -> bool:
    return (REPO_ROOT / ".git").exists()


def _source_files() -> list[Path]:
    files: list[Path] = []
    for directory in SOURCE_DIRS:
        root = REPO_ROOT / directory
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            if any(part in GENERATED for part in path.parts):
                continue
            files.append(path)
    return files


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_no_source_file_is_gitignored() -> None:
    files = _source_files()
    assert files, "found no source files to check — SOURCE_DIRS is probably stale"

    relative = [str(path.relative_to(REPO_ROOT)) for path in files]

    # check-ignore exits 0 when at least one path matched, 1 when none did.
    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input="\n".join(relative),
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    ignored = [line for line in result.stdout.splitlines() if line.strip()]

    assert not ignored, (
        "these source files are excluded by .gitignore and will be missing from "
        "a fresh clone — anchor the offending pattern to the repo root "
        "(`/findings/`, not `findings/`):\n  " + "\n  ".join(ignored)
    )


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_committed_tree_can_import_the_source_packages() -> None:
    """Every source package's files are tracked, not merely un-ignored.

    A file can escape `.gitignore` and still be missing from the index because
    nobody ran `git add`. Untracked-but-visible shows up in `git status`, so
    this only checks packages that already have at least one tracked file —
    the case where a package looks committed but is partially absent.
    """
    tracked = set(
        subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        ).stdout.split()
    )

    missing: list[str] = []
    for path in _source_files():
        if path.suffix != ".py":
            continue
        rel = str(path.relative_to(REPO_ROOT))
        package = str(path.parent.relative_to(REPO_ROOT))
        package_is_tracked = any(name.startswith(package + "/") for name in tracked)
        if package_is_tracked and rel not in tracked:
            missing.append(rel)

    assert not missing, (
        "these modules sit in an otherwise-committed package but are not in the "
        "git index — a fresh clone will fail to import them:\n  " + "\n  ".join(missing)
    )


@pytest.mark.parametrize("directory", SOURCE_DIRS)
def test_source_directories_exist(directory: str) -> None:
    # If a directory is renamed, the guards above silently stop checking it.
    assert (REPO_ROOT / directory).is_dir()
