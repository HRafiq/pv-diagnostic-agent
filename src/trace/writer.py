"""Append-only JSONL trace writer.

One file per investigation, flushed on every write. Flushing matters: the
dashboard tails these files to render a run in progress, so a buffered writer
would make a live investigation look frozen.

Traces are gitignored. They are output, not source.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import TracebackType

from src.clock import Clock
from src.trace.models import TraceStep

__all__ = ["TraceWriter", "read_trace"]


class TraceWriter:
    """Writes one investigation's steps to ``<root>/<investigation_id>.jsonl``.

    Usage::

        with TraceWriter("INV-2019-03-14-001", clock=clock) as trace:
            trace.write(step)

    Args:
        investigation_id: Names the file and links a Finding back to its trace.
        clock: Injected time source. Stamps each step with *simulated* time.
        root: Output directory. Created if missing.
        on_step: Called with each written step. Injected rather than printed
            from here, because `src/` is UI-agnostic (CLAUDE.md) — this is the
            seam every step already passes through, so a caller that wants live
            progress needs no hooks scattered across the nodes. Exceptions from
            the callback are swallowed: a broken progress display must never
            take down an investigation that is otherwise fine.
    """

    def __init__(
        self,
        investigation_id: str,
        clock: Clock,
        root: Path | str = "traces",
        on_step: Callable[[TraceStep], None] | None = None,
    ) -> None:
        if not investigation_id or "/" in investigation_id:
            raise ValueError(
                f"investigation_id must be a non-empty path-safe token; "
                f"got {investigation_id!r}"
            )
        self.investigation_id = investigation_id
        self._clock = clock
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / f"{investigation_id}.jsonl"
        self._handle: object | None = None
        self._count = 0
        self._on_step = on_step

    @property
    def path(self) -> Path:
        return self._path

    @property
    def step_count(self) -> int:
        return self._count

    def __enter__(self) -> TraceWriter:
        self._handle = self._path.open("a", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def write(self, step: TraceStep) -> TraceStep:
        """Append a step. Fills in ``step_index`` and simulated ``timestamp``."""
        if self._handle is None:
            raise RuntimeError(
                "TraceWriter is not open; use it as a context manager "
                "(`with TraceWriter(...) as trace:`)"
            )
        stamped = step.model_copy(
            update={
                "step_index": self._count,
                "timestamp": step.timestamp or self._clock.now(),
            }
        )
        line = stamped.model_dump_json(exclude_none=False)
        self._handle.write(line + "\n")  # type: ignore[attr-defined]
        # Flush every step: the dashboard tails this file to render a live run.
        self._handle.flush()  # type: ignore[attr-defined]
        self._count += 1

        if self._on_step is not None:
            # A progress display is never worth a failed run.
            with contextlib.suppress(Exception):
                self._on_step(stamped)
        return stamped

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()  # type: ignore[attr-defined]
            self._handle = None


def read_trace(path: Path | str) -> Iterator[TraceStep]:
    """Replay a trace file. Used by the dashboard and the eval harness."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield TraceStep.model_validate(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{lineno} is not a valid TraceStep") from exc
