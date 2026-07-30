"""Where the time and the money actually went, read off runs already paid for.

Every decision about this agent's speed for the last several rounds was made on
an estimate: count the calls in a log, multiply by a guessed per-call latency,
argue from the product. That is how "80% of the wait is the Sonnet nodes" got
said out loud without ever being checked.

It did not need checking against a new run. Every `TraceStep` records
`latency_ms` and `cost_usd`, and there are already several runs on disk. This
module reads them.

Two things it is careful about, because both would flatter the answer:

**A "tool" step's latency is not the tool's.** Tools are pure functions and take
milliseconds. What the step records is the *router turn that chose it* — one LLM
call per measurement. So `measure` time in this report is router time, and the
physics is free. The report says so rather than letting a reader assume the
plant analysis is slow.

**A step with no recorded latency is not a fast step.** Until the change that
landed with this module, the critic's own cost was accrued to the run total and
never written onto its step, so a naive reading of an older trace would show
review as costing nothing — the exact node whose value is most in question.
Older traces are reported with that caveat attached rather than silently
averaged in.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.trace.models import TraceStep
from src.trace.writer import read_trace

__all__ = [
    "NodeCost",
    "RunProfile",
    "explain",
    "profile_run",
    "profile_traces",
    "render",
    "split_attempts",
]

# What a reader recognises, mapped from what the trace calls it. `tool` and
# `adaptive` both mean "a router turn chose a measurement".
_LABELS = {
    "plan": "planner",
    "retrieval": "look up",
    "tool": "router + measure",
    "adaptive": "router + measure*",
    "critic": "review",
    "answer": "synthesiser",
}


@dataclass
class NodeCost:
    """What one kind of step cost across a run."""

    calls: int = 0
    seconds: float = 0.0
    usd: float = 0.0
    tokens: int = 0

    def add(self, step: TraceStep) -> None:
        self.calls += 1
        self.seconds += step.latency_ms / 1000.0
        self.usd += step.cost_usd
        self.tokens += step.tokens


@dataclass
class RunProfile:
    """One investigation, broken down."""

    investigation_id: str
    by_node: dict[str, NodeCost] = field(default_factory=dict)
    by_cycle: dict[int, NodeCost] = field(default_factory=dict)
    cycles: int = 0
    measurements: int = 0
    # True when the trace predates review costs being written onto the step.
    review_cost_missing: bool = False
    attempt: int = 1
    of_attempts: int = 1

    @property
    def seconds(self) -> float:
        return sum(n.seconds for n in self.by_node.values())

    @property
    def usd(self) -> float:
        return sum(n.usd for n in self.by_node.values())

    def after_first_cycle(self) -> tuple[float, float]:
        """Seconds and dollars spent after cycle 0 — the re-review overhead.

        The number the latency argument turns on. Four cases reached the right
        answer in cycle 1 and were sent back; if this is most of the run, the
        review loop is the cost *and* the accuracy problem.
        """
        later = [c for cycle, c in self.by_cycle.items() if cycle > 0]
        return sum(c.seconds for c in later), sum(c.usd for c in later)


def split_attempts(steps: list[TraceStep]) -> list[list[TraceStep]]:
    """Cut a trace file into the separate runs concatenated inside it.

    `TraceWriter` opens with mode `"a"`, and the evaluation names every trace
    after the case (`INV-G-004`), so re-running a case **appends** to the file
    it wrote last time. A trace on disk is therefore every attempt at that case
    ever made, end to end.

    That is not a cosmetic problem. Read naively, `INV-G-004` reports 59
    measurements, 4 cycles and $3.37 — while the run that produced its last
    entries took 11 measurements and cost $0.34. Every per-run figure was the
    sum of a session's worth of attempts, and "after cycle 1" was mixing runs
    that never saw each other.

    `step_index` is stamped from a counter that starts at zero per writer, so a
    new attempt is exactly where the index stops increasing. No timestamps, no
    heuristics.
    """
    attempts: list[list[TraceStep]] = []
    current: list[TraceStep] = []
    last = -1
    for step in steps:
        if step.step_index <= last and current:
            attempts.append(current)
            current = []
        current.append(step)
        last = step.step_index
    if current:
        attempts.append(current)
    return attempts


def profile_run(path: Path | str, steps: list[TraceStep] | None = None) -> RunProfile:
    """Read one attempt. Pass `steps` to profile one segment of a trace file."""
    steps = list(read_trace(path)) if steps is None else steps
    out = RunProfile(investigation_id=Path(path).stem)
    saw_review = False
    review_with_cost = False

    for step in steps:
        label = _LABELS.get(step.kind, step.kind)
        out.by_node.setdefault(label, NodeCost()).add(step)

        cycle = step.args.get("cycle") if isinstance(step.args, dict) else None
        key = int(cycle) if isinstance(cycle, int) else 0
        out.by_cycle.setdefault(key, NodeCost()).add(step)
        out.cycles = max(out.cycles, key + 1)

        if step.kind in ("tool", "adaptive"):
            out.measurements += 1
        if step.kind == "critic":
            saw_review = True
            review_with_cost = review_with_cost or step.latency_ms > 0

    out.review_cost_missing = saw_review and not review_with_cost
    return out


def profile_traces(root: Path | str, pattern: str = "**/*.jsonl") -> list[RunProfile]:
    """Every *attempt* under `root`, oldest first.

    One entry per run, not per file — see `split_attempts` for why those differ.
    """
    base = Path(root)
    files = [base] if base.is_file() else sorted(base.glob(pattern))

    out: list[RunProfile] = []
    for path in files:
        attempts = split_attempts(list(read_trace(path)))
        for n, steps in enumerate(attempts, start=1):
            profile = profile_run(path, steps=steps)
            if len(attempts) > 1:
                # Named so a reader can see this case was run more than once and
                # which attempt they are looking at.
                profile.investigation_id = f"{profile.investigation_id} #{n}"
                profile.attempt = n
                profile.of_attempts = len(attempts)
            out.append(profile)
    return out


def _table(rows: list[tuple[str, NodeCost]], total_seconds: float) -> list[str]:
    lines = [
        f"    {'node':<20}{'calls':>7}{'seconds':>10}{'share':>8}{'usd':>9}",
        "    " + "-" * 54,
    ]
    for name, cost in rows:
        share = (cost.seconds / total_seconds * 100.0) if total_seconds else 0.0
        lines.append(
            f"    {name:<20}{cost.calls:>7}{cost.seconds:>10.1f}"
            f"{share:>7.1f}%{cost.usd:>9.3f}"
        )
    return lines


def render(profiles: list[RunProfile]) -> str:
    """The report. Per run, then the totals that answer the latency question."""
    if not profiles:
        return (
            "no traces found. Runs write them under traces/eval/<INV-ID>.jsonl; "
            "point this at that directory."
        )

    lines: list[str] = []
    totals: dict[str, NodeCost] = defaultdict(NodeCost)
    grand_seconds = 0.0
    grand_usd = 0.0
    later_seconds = 0.0
    later_usd = 0.0
    stale = 0
    timed = 0

    for run in profiles:
        if run.seconds <= 0.0:
            # A scripted or replayed trace. Reporting it as a fast run would be
            # a lie of the most convenient kind.
            lines.append(f"\n  {run.investigation_id}: no timing recorded, skipped")
            continue

        after_s, after_usd = run.after_first_cycle()
        grand_seconds += run.seconds
        grand_usd += run.usd
        later_seconds += after_s
        later_usd += after_usd
        timed += 1
        stale += 1 if run.review_cost_missing else 0

        lines.append(
            f"\n  {run.investigation_id}  {run.seconds:.0f}s, ${run.usd:.3f}, "
            f"{run.cycles} cycle(s), {run.measurements} measurement(s)"
        )
        rows = sorted(run.by_node.items(), key=lambda kv: -kv[1].seconds)
        lines.extend(_table(rows, run.seconds))
        if run.cycles > 1:
            share = after_s / run.seconds * 100.0 if run.seconds else 0.0
            lines.append(
                f"    after cycle 1: {after_s:.0f}s ({share:.0f}% of the run), "
                f"${after_usd:.3f}"
            )
        if run.review_cost_missing:
            lines.append(
                "    (review cost not recorded in this trace — predates the fix, "
                "so review is under-reported here)"
            )
        for name, cost in run.by_node.items():
            totals[name].calls += cost.calls
            totals[name].seconds += cost.seconds
            totals[name].usd += cost.usd
            totals[name].tokens += cost.tokens

    if not grand_seconds:
        return "\n".join(lines) + "\n\n  nothing with recorded timing to summarise."

    lines.append(f"\n\n  across {timed} timed run(s) of {len(profiles)}")
    lines.extend(
        _table(sorted(totals.items(), key=lambda kv: -kv[1].seconds), grand_seconds)
    )
    lines.append(
        f"    {'TOTAL':<20}{'':>7}{grand_seconds:>10.1f}{'':>8}{grand_usd:>9.3f}"
    )

    if later_seconds:
        lines.append(
            f"\n  spent after cycle 1: {later_seconds:.0f}s of {grand_seconds:.0f}s "
            f"({later_seconds / grand_seconds * 100:.0f}%), ${later_usd:.3f} of "
            f"${grand_usd:.3f}"
        )
        lines.append(
            "  That is the re-review overhead. Four cases reached the right "
            "answer in cycle 1\n  and were sent back, so this is the latency "
            "cost and the accuracy cost together."
        )

    lines.append(
        "\n  Reading this: 'router + measure' is the LLM turn that *chose* each\n"
        "  measurement. The measurements themselves are pure functions and take\n"
        "  milliseconds — none of this time is the physics."
    )
    if stale:
        lines.append(
            "  Some traces predate review costs being recorded on the step; where\n"
            "  that is noted above, review is under-reported."
        )
    return "\n".join(lines)


def summary(profiles: list[RunProfile]) -> dict[str, Any]:
    """The same numbers as data, for a test or a report file."""
    timed = [p for p in profiles if p.seconds > 0]
    by_node: dict[str, NodeCost] = defaultdict(NodeCost)
    for run in timed:
        for name, cost in run.by_node.items():
            by_node[name].calls += cost.calls
            by_node[name].seconds += cost.seconds
            by_node[name].usd += cost.usd
    later = [p.after_first_cycle() for p in timed]
    return {
        "runs": len(profiles),
        "timed_runs": len(timed),
        "seconds": round(sum(p.seconds for p in timed), 1),
        "usd": round(sum(p.usd for p in timed), 4),
        "by_node": {
            name: {
                "calls": c.calls,
                "seconds": round(c.seconds, 1),
                "usd": round(c.usd, 4),
            }
            for name, c in sorted(by_node.items(), key=lambda kv: -kv[1].seconds)
        },
        "seconds_after_first_cycle": round(sum(s for s, _ in later), 1),
        "usd_after_first_cycle": round(sum(u for _, u in later), 4),
    }


def explain(path: Path | str, attempt: int | None = None) -> str:
    """One attempt, in full, in the order it happened.

    The live display clips every line to the terminal width, which is right for
    watching a run and wrong for diagnosing one: G-004 committed to
    `string_outage` on a soiling case, and the sentence that would say why was
    cut at 96 characters in every log of it. The trace has the whole thing.

    Prints the plan's candidate causes, each measurement's full summary, and the
    final answer with the evidence it cited — the four things needed to say what
    the agent believed and what it read.
    """
    attempts = split_attempts(list(read_trace(path)))
    if not attempts:
        return f"{path} has no steps."
    index = (attempt or len(attempts)) - 1
    if not 0 <= index < len(attempts):
        return (
            f"{path} holds {len(attempts)} attempt(s); asked for {attempt}. "
            "Attempts are numbered from 1, and the last is the default."
        )

    steps = attempts[index]
    lines = [
        f"{Path(path).stem}  attempt {index + 1} of {len(attempts)}  "
        f"({len(steps)} steps)"
    ]
    for step in steps:
        args = step.args if isinstance(step.args, dict) else {}
        cycle = args.get("cycle", 0)

        if step.kind == "plan":
            lines.append(f"\n--- cycle {cycle}: PLAN " + "-" * 46)
            hypotheses = args.get("hypotheses") or []
            for h in hypotheses if isinstance(hypotheses, list) else []:
                if isinstance(h, dict):
                    lines.append(f"  {h.get('id', '?')}  {h.get('cause', '?')}")
                    if h.get("consequence_if_true"):
                        lines.append(f"      if true: {h['consequence_if_true']}")
            planned = args.get("planned_tools") or []
            if isinstance(planned, list) and planned:
                lines.append("  plan: " + ", ".join(str(t) for t in planned))

        elif step.kind in ("tool", "adaptive"):
            mark = "  *" if step.kind == "adaptive" else "   "
            lines.append(f"{mark} {step.result or ''}")
            if not step.was_planned and step.reason_for_choosing:
                lines.append(f"      chosen because: {step.reason_for_choosing}")

        elif step.kind == "retrieval":
            lines.append(f"    [{step.result or ''}]")

        elif step.kind == "answer":
            lines.append(f"\n--- cycle {cycle}: ANSWER " + "-" * 44)
            lines.append(f"  settled: {args.get('settled')}")
            lines.append(f"  cause: {args.get('cause')}")
            lines.append(f"  category: {args.get('category')}")
            if args.get("category_as_written"):
                lines.append(
                    f"  (the model wrote category "
                    f"{args['category_as_written']!r}; the knowledge base "
                    "assigns the one above to this cause)"
                )
            lines.append(f"  confidence: {args.get('confidence')}")
            if args.get("resolving_measurement"):
                lines.append(f"  would be resolved by: {args['resolving_measurement']}")
            if args.get("ungrounded_figures"):
                lines.append(f"  UNGROUNDED: {args['ungrounded_figures']}")
            lines.append(f"\n  {step.result or args.get('answer', '')}")

        elif step.kind == "critic":
            lines.append(f"\n--- cycle {cycle}: REVIEW " + "-" * 44)
            lines.append(f"  {step.result or args.get('verdict', '')}")
            if args.get("revision_request"):
                lines.append(f"  asked for: {args['revision_request']}")

    return "\n".join(lines)
