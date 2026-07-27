"""Live progress for a running investigation.

An agent evaluation is 43 cases, each several minutes of planning, measuring
and reviewing. Printing only on completion means minutes of silence and no way
to tell a working run from a hung one — or to notice, three minutes in, that
every case is going to fail the same way.

This lives in `eval/` rather than `src/` on purpose. `src/` is UI-agnostic
(CLAUDE.md); it emits trace steps and this decides how to show them. The
dashboard makes the same choice differently from the same data.

Nothing here is measured or scored. It is a view.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from src.trace.models import TraceStep

__all__ = ["StepPrinter"]

# One glyph per kind of step, so a long run reads as a shape rather than a wall
# of text. Deliberately not the internal vocabulary: `adaptive` is a tool call
# the router chose mid-investigation, and that is what it says.
# The six kinds `TraceStep` actually defines. mypy rejected an earlier version
# of this map that invented "route" and "review" — a reminder that a display is
# still code, and a label for a step kind that cannot occur is dead.
_LABELS: dict[str, str] = {
    "plan": "plan",
    "retrieval": "look up",
    "tool": "measure",
    "adaptive": "measure*",  # a tool the router chose mid-investigation
    "critic": "review",
    "answer": "answer",
}


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


@dataclass
class StepPrinter:
    """Prints each trace step on one line as it happens.

    Args:
        case_id: Prefixed to every line so interleaved output stays readable.
        verbose: When False, prints a running one-line counter instead of a
            line per step — enough to see the run is alive without burying the
            per-case summaries.
        width: Characters of detail to show per line.
    """

    case_id: str = ""
    verbose: bool = True
    width: int = 96
    steps: int = field(default=0, init=False)
    tools: int = field(default=0, init=False)
    cost_usd: float = field(default=0.0, init=False)

    def __call__(self, step: TraceStep) -> None:
        self.steps += 1
        self.cost_usd += step.cost_usd
        if step.kind in ("tool", "adaptive"):
            self.tools += 1

        if not self.verbose:
            self._counter()
            return

        label = _LABELS.get(step.kind, step.kind)
        detail = self._detail(step)
        marker = "" if step.was_planned else "  <- unplanned"
        prefix = f"    [{self.case_id}] " if self.case_id else "    "
        print(f"{prefix}{label:<9} {_clip(detail, self.width)}{marker}", flush=True)

    def _counter(self) -> None:
        # `\r` only overwrites on a terminal. Piped to a file or a CI log it
        # would produce one unreadable concatenated line, so there the counter
        # goes quiet and only the per-case summaries remain.
        if not sys.stdout.isatty():
            return
        prefix = f"  {self.case_id}" if self.case_id else "  working"
        measurements = "measurement" if self.tools == 1 else "measurements"
        step_word = "step" if self.steps == 1 else "steps"
        sys.stdout.write(
            f"\r{prefix}: {self.steps} {step_word}, "
            f"{self.tools} {measurements}, ${self.cost_usd:.3f}"
        )
        sys.stdout.flush()

    def _detail(self, step: TraceStep) -> str:
        """The most informative field this kind of step carries."""
        args = step.args or {}

        if step.kind == "plan":
            causes = args.get("hypotheses")
            if isinstance(causes, list):
                named = [
                    str(c.get("cause", "?")) for c in causes if isinstance(c, dict)
                ]
                return f"{len(named)} possible causes: " + ", ".join(named)
            return str(args.get("question", ""))

        if step.kind in ("tool", "adaptive"):
            # The tool's own summary is the interesting part — it is the
            # measurement, in the words the tool chose.
            return f"{step.node}: {step.result or ''}"

        if step.kind == "answer":
            if args.get("settled"):
                return f"settled: {args.get('cause', '')}"
            return "not enough evidence: " + str(args.get("resolving_measurement", ""))

        if step.kind == "critic":
            excluded = (
                f" — rules out {', '.join(step.excludes)}" if step.excludes else ""
            )
            return f"{args.get('verdict', '')}{excluded}"

        if step.kind == "retrieval":
            return step.result or str(args.get("query", ""))

        return step.result or str(args)[: self.width]

    def finish(self) -> None:
        """Clear the in-place counter so the next line starts clean."""
        if not self.verbose and self.steps and sys.stdout.isatty():
            sys.stdout.write("\r" + " " * 78 + "\r")
            sys.stdout.flush()
