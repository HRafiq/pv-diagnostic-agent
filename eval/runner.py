"""Run a diagnostic over the golden set and report per split.

    python -m eval.runner build          # write the golden set
    python -m eval.runner run --engine rules
    python -m eval.runner run --engine agent --split tuning
    python -m eval.runner run --engine rules --split heldback

Both engines answer the same questions over the same data and are scored by the
same code, which is the only way the comparison in `docs/FINDINGS.md` means
anything. The rules engine needs no API key; the agent does, and says so rather
than degrading to something that looks like a result.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from eval.compare import compare, comparison_table
from eval.golden import build_golden_set, composition, load_cases, write_cases
from eval.metrics import CaseScore, Prediction, aggregate, confusion, score_case
from eval.progress import StepPrinter
from eval.scenarios import materialise
from eval.subset import describe, select, warn_about
from src.agent.llm import BudgetExceeded, build_client, is_systemic_request_error

# The graph, not the plain loop. Both produce identical results
# (`tests/test_langgraph_port.py`), the plain loop remains the specification,
# and the port had sat unused since it was written — a second implementation
# nobody ran is a liability, not a safety net.
from src.agent.loop_graph import investigate_with_graph as investigate
from src.baseline.rules import RulesEngine
from src.config import REPO_ROOT, Settings, build_clock
from src.data.plant import load_plant
from src.data.sources import SystemMetadata
from src.determinism import DEFAULT_SEED
from src.knowledge import KnowledgeBase
from src.physics.modelchain import module_gamma_pdc

# Consecutive per-case failures that end a run. Three rather than one: a
# single case can legitimately die on its own content, but three in a row
# after the client's own retries means the environment is the problem.
_CONSECUTIVE_FAILURE_LIMIT = 3

GOLDEN_DIR = REPO_ROOT / "eval" / "golden"
DATA_DIR = REPO_ROOT / "data" / "raw"
TRACE_DIR = REPO_ROOT / "traces"


def _system_metadata(system_id: int) -> tuple[SystemMetadata, float]:
    manifest_path = DATA_DIR / f"system_{system_id}_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"system {system_id} is not ingested; run "
            f"`python -m src.data.cli ingest --system {system_id} --years 2016 2017`"
        )
    manifest = json.loads(manifest_path.read_text())
    meta = SystemMetadata(**{**manifest["system_metadata"], "raw": {}})
    gamma = module_gamma_pdc(meta.module_model) or -0.0040
    return meta, gamma


class _Journal:
    """Per-case results, written as they happen so a dead run can be resumed.

    One JSON object per line, appended the moment a case finishes. Append-only
    and flushed each time: a process killed mid-write loses at most the case it
    was working on, and every case before it is still there.

    Deliberately not the trace files. Traces are the record of *how* a case ran
    and the dashboard reads them; this is the much smaller record of *what* it
    concluded, which is all that resuming needs.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def load(self) -> dict[str, Prediction]:
        if self.path is None or not self.path.exists():
            return {}
        out: dict[str, Prediction] = {}
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                out[str(payload["case_id"])] = Prediction(
                    **{
                        **payload,
                        "candidate_causes": tuple(payload.get("candidate_causes", [])),
                        "tools_called": tuple(payload.get("tools_called", [])),
                        "unplanned_tools": tuple(payload.get("unplanned_tools", [])),
                    }
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                # A half-written final line is exactly what a killed process
                # leaves. Skip it and re-run that one case rather than refusing
                # to resume at all.
                continue
        return out

    def record(self, prediction: Prediction) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(asdict(prediction), default=str) + "\n")
            handle.flush()


def run_rules_engine(cases: list[Any]) -> tuple[list[CaseScore], list[Prediction]]:
    scores: list[CaseScore] = []
    predictions: list[Prediction] = []

    for case in cases:
        bundle = load_plant(case.system_id, DATA_DIR)
        materialised = materialise(case, DATA_DIR)
        # The same context the agent gets: the whole record, perturbed inside
        # the case window. Both engines measure identically, so a gap between
        # them is a gap in reasoning rather than in measurement.
        ctx = bundle.context(materialised.full_record)
        engine = RulesEngine(window={"start": case.start, "end": case.end})
        started = time.monotonic()
        verdict = engine.diagnose(ctx)
        elapsed_ms = int((time.monotonic() - started) * 1000)

        prediction = Prediction(
            case_id=case.id,
            category=verdict.category,
            cause=verdict.cause,
            settled=verdict.settled,
            candidate_causes=verdict.candidate_causes,
            resolving_measurement=verdict.resolving_measurement,
            tools_called=verdict.checks_run,
            unplanned_tools=(),  # a fixed sequence has none, by construction
            critic_cycles=0,
            cost_usd=0.0,
            latency_ms=elapsed_ms,
        )
        predictions.append(prediction)
        scores.append(score_case(case, prediction))
    return scores, predictions


def run_agent_engine(
    cases: list[Any],
    trace_root: Path | None = None,
    max_tools_per_cycle: int | None = None,
    review: bool = True,
    use_knowledge: bool = True,
    verbose: bool = True,
    resume_from: Path | None = None,
) -> tuple[list[CaseScore], list[Prediction]]:
    """Run the investigation loop over the golden set.

    Every case gets a fresh client, so one investigation's budget cannot be
    spent by another and the per-question cost figure means what it says.
    Traces are written per case and are what the dashboard's Investigate tab
    replays.

    Args:
        resume_from: A journal written case-by-case as the run proceeds. Cases
            already in it are skipped and their results reused.

            This is the resumability that matters, and it is deliberately *not*
            LangGraph's. A checkpointer resumes one investigation across node
            boundaries; the failures this project actually hit — an API overload
            at case 6, an exhausted credit balance mid-answer — kill the
            interpreter, and an in-memory saver dies with it. What is expensive
            to lose is the twenty cases already paid for, and only a file
            survives that.
    """
    settings = Settings.from_env()
    clock = build_clock()
    root = trace_root or TRACE_DIR / "eval"
    # The two ablations. Both run the *identical* loop with one layer removed,
    # which is the only way to say whether that layer earns its cost.
    knowledge = None if use_knowledge else KnowledgeBase(signatures={}, tests=())

    scores: list[CaseScore] = []
    predictions: list[Prediction] = []
    failures: list[tuple[str, str]] = []
    journal = _Journal(resume_from)
    done = journal.load()
    if done:
        print(f"  resuming: {len(done)} case(s) already recorded, skipping them")
    # A run that has failed this many cases in a row is not meeting bad luck;
    # something outside the cases is broken — usually the network. Continuing
    # burns the rest of the case list at zero work each, which is what four
    # killed runs looked like. The client already retries transient errors in
    # place, so reaching this count means the retries did not help either.
    consecutive_failures = 0

    for case in cases:
        recorded = done.get(str(case.id))
        if recorded is not None:
            predictions.append(recorded)
            scores.append(score_case(case, recorded))
            print(f"\n  {case.id}  (already recorded, skipping)")
            continue

        bundle = load_plant(case.system_id, DATA_DIR)
        materialised = materialise(case, DATA_DIR)
        # The whole record, perturbed inside the case window. The agent is
        # asked about the window and may reach outside it, exactly as an
        # engineer would — several tools are meaningless without the history.
        ctx = bundle.context(
            materialised.full_record, scope=f"{bundle.meta.name} / array"
        )
        client = build_client(load_models_config_cached(), settings.anthropic_api_key)

        # A case takes minutes. Without this the run is silent until it
        # finishes, which makes a working run indistinguishable from a hung one
        # — and hides the fact that every case is failing the same way until
        # the first one gives up.
        print(f"\n  {case.id}  {case.question[:80]}", flush=True)
        printer = StepPrinter(case_id=case.id, verbose=verbose)

        started = time.monotonic()
        # One case must not be able to destroy the other forty-two. An
        # investigation can die on a malformed reply, a transient API error, or
        # a budget cap, and aborting the run then throws away every case that
        # already succeeded — and the money they cost. A failed case is scored
        # as what it is: an answer the agent could not produce.
        #
        # BudgetExceeded is deliberately *not* caught here. It means the loop
        # is not terminating, which is a defect in the agent rather than a bad
        # case, and it would otherwise repeat on every remaining case.
        try:
            out = investigate(
                case.question,
                ctx,
                client,
                clock,
                investigation_id=f"INV-{case.id}",
                start=case.start,
                end=case.end,
                trace_root=root,
                max_tools_per_cycle=max_tools_per_cycle,
                critic=None if review else False,
                knowledge=knowledge,
                on_step=printer,
            )
        except BudgetExceeded:
            raise
        except Exception as exc:  # reported, then scored as a failure
            printer.finish()
            consecutive_failures += 1
            # A malformed request is a defect here, not a bad case. Isolating it
            # per case just reproduces it once per case — which is exactly what
            # happened: forty-three identical 400s, one round trip each.
            if is_systemic_request_error(exc):
                raise RuntimeError(
                    f"{case.id} failed with a malformed request, which every "
                    f"remaining case would reproduce identically. Stopping so "
                    f"the schema or parameter can be fixed once rather than "
                    f"{len(cases)} times.\n\n  {type(exc).__name__}: {exc}"
                ) from exc

            elapsed_ms = int((time.monotonic() - started) * 1000)
            print(f"  {case.id}  FAILED: {type(exc).__name__}: {exc}")
            failures.append((case.id, f"{type(exc).__name__}: {exc}"))
            predictions.append(
                Prediction(
                    case_id=case.id,
                    category=None,
                    cause=None,
                    settled=False,
                    candidate_causes=(),
                    resolving_measurement=None,
                    tools_called=(),
                    unplanned_tools=(),
                    critic_cycles=0,
                    cost_usd=0.0,
                    latency_ms=elapsed_ms,
                    # Scored as unsettled, which is what it was — but marked,
                    # so the agency metrics do not report a 529 as the agent
                    # choosing to abstain.
                    failed_with=f"{type(exc).__name__}: {exc}",
                )
            )
            scores.append(score_case(case, predictions[-1]))
            journal.record(predictions[-1])
            if consecutive_failures >= _CONSECUTIVE_FAILURE_LIMIT:
                raise RuntimeError(
                    f"{consecutive_failures} cases in a row failed, the last "
                    f"with {type(exc).__name__}: {exc}\n"
                    f"  Stopping: this is an environment problem rather than a "
                    f"case problem, and the remaining "
                    f"{len(cases) - len(predictions)} cases would fail the same "
                    f"way. {len(predictions) - consecutive_failures} case(s) "
                    f"completed before it started."
                ) from exc
            continue
        printer.finish()
        consecutive_failures = 0
        elapsed_ms = int((time.monotonic() - started) * 1000)

        finding = out.finding
        # A synthesis that could not be filed as a finding is scored as an
        # unsettled answer with nothing behind it, which is what it is. Quietly
        # dropping the case would flatter the engine by removing its failures
        # from the denominator.
        predictions.append(
            Prediction(
                case_id=case.id,
                category=finding.category if finding and finding.settled else None,
                cause=finding.cause if finding else None,
                settled=bool(finding and finding.settled),
                candidate_causes=tuple(
                    c.cause for c in (finding.candidate_causes if finding else [])
                ),
                resolving_measurement=(
                    finding.resolving_measurement if finding else None
                ),
                tools_called=tuple(out.tools_called),
                unplanned_tools=tuple(out.unplanned_tools),
                critic_cycles=out.state.cycle,
                cost_usd=out.cost_usd,
                latency_ms=elapsed_ms,
            )
        )
        scores.append(score_case(case, predictions[-1]))
        journal.record(predictions[-1])
        print(
            f"  -> {case.id}  {len(out.tools_called)} measurements, "
            f"{len(out.unplanned_tools)} unplanned, "
            f"${out.cost_usd:.3f}, {elapsed_ms / 1000:.0f}s, "
            f"{'settled' if predictions[-1].settled else 'not enough evidence'}"
        )
        if out.ungrounded_numbers:
            print(f"         UNGROUNDED FIGURES: {out.ungrounded_numbers}")

    if failures:
        # Said out loud, not buried. These cases are in the denominator as
        # failures, so a reader must know how much of the score is "the agent
        # answered badly" versus "the agent did not answer".
        print(f"\n  {len(failures)} of {len(cases)} cases failed outright:")
        for case_id, reason in failures:
            print(f"    {case_id}  {reason}")
        print("  They are scored as unsettled, which is what they were.")

    return scores, predictions


def load_models_config_cached() -> Any:
    from src.config import load_models_config

    return load_models_config()


def _run_engine(
    engine: str, cases: list[Any], args: argparse.Namespace
) -> tuple[list[CaseScore], list[Prediction]]:
    if engine == "rules":
        return run_rules_engine(cases)
    # Which model configuration produced these numbers. Without it, two runs
    # with different effort settings are indistinguishable in the output and a
    # comparison between them is not a comparison.
    profile_name = load_models_config_cached().active_profile
    print(
        f"\nrunning the agent over {len(cases)} cases [model profile: {profile_name}]\n"
    )
    return run_agent_engine(
        cases,
        review=not getattr(args, "no_review", False),
        use_knowledge=not getattr(args, "no_knowledge", False),
        verbose=not getattr(args, "quiet", False),
        resume_from=(Path(args.resume) if getattr(args, "resume", None) else None),
    )


def _load_cases(split: str) -> list[Any]:
    paths = {
        "tuning": GOLDEN_DIR / "cases_tuning.jsonl",
        "heldback": GOLDEN_DIR / "cases_heldback.jsonl",
    }
    splits = ["tuning", "heldback"] if split == "both" else [split]
    cases: list[Any] = []
    for name in splits:
        if not paths[name].exists():
            raise FileNotFoundError(
                f"missing {paths[name]}; run `python -m eval.runner build` first"
            )
        cases.extend(load_cases(paths[name]))
    return cases


def _cmd_compare(args: argparse.Namespace) -> int:
    """Run both engines over the same cases and print the comparison.

    Published whichever way it falls. If the rules engine wins outright that is
    a more credible finding than "I built an agent" (CLAUDE.md).
    """
    try:
        cases = _load_cases(args.split)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    cases = _apply_subset(cases, args)

    try:
        _check_the_record_covers_every_case(cases)
    except (RecordTooShort, FileNotFoundError) as exc:
        print(f"\n{exc}")
        return 1

    runs: dict[str, tuple[list[CaseScore], list[Prediction], Any]] = {}
    for engine in ("rules", "agent"):
        try:
            scores, predictions = _run_engine(engine, cases, args)
        except RuntimeError as exc:
            print(f"\ncannot run the {engine} engine: {exc}")
            return 1
        runs[engine] = (scores, predictions, aggregate(scores, predictions))

    comparison = compare(runs)
    for split in sorted({s.split for s in runs["rules"][0]}):
        print(comparison_table(comparison, split))

    if args.out:
        Path(args.out).write_text(json.dumps(comparison.to_dict(), indent=2))
        print(f"\n  wrote {args.out}")
    return 0


def _cmd_experiments(args: argparse.Namespace) -> int:
    """Run the ablations over N fresh runs and report mean ± spread.

    A single number from a non-deterministic system is not a result
    (CLAUDE.md), so nothing here reports one.
    """
    from eval.experiments import ABLATIONS, ablation_table, repeat, summarise_runs

    # Checked before any work, not on the first LLM call. Four ablations over
    # 43 cases each load a plant and materialise an injection before they reach
    # a client, so a missing key would otherwise surface minutes in, after the
    # command has already printed a header that looks like a run in progress.
    try:
        Settings.from_env().require_api_key()
    except RuntimeError as exc:
        print(f"cannot run the experiments: {exc}")
        return 1

    try:
        cases = _load_cases(args.split)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    cases = _apply_subset(cases, args)

    try:
        _check_the_record_covers_every_case(cases)
    except (RecordTooShort, FileNotFoundError) as exc:
        print(f"\n{exc}")
        return 1

    wanted = (
        {k: v for k, v in ABLATIONS.items() if k in args.only}
        if args.only
        else ABLATIONS
    )
    results = []
    for label, options in wanted.items():
        print(f"\n=== {label} — {args.runs} run(s) over {len(cases)} cases ===")
        try:
            runs = repeat(label, run_agent_engine, cases, n=args.runs, **options)
        except RuntimeError as exc:
            print(f"\ncannot run the experiments: {exc}")
            return 1
        except FileNotFoundError as exc:
            # A missing ingest is a setup problem with a known remedy, and the
            # exception already carries the command that fixes it. Printing it
            # beats a traceback out of the middle of an ablation loop.
            print(f"\ncannot run the experiments: {exc}")
            return 1
        results.append(summarise_runs(runs, args.split))

    print("\n" + ablation_table(results).to_string(index=False))
    print(
        "\n  read correct_abstention_rate first: it is the column the rules "
        "baseline scores 0.000 on structurally, so it is where an agent has "
        "somewhere to be better rather than merely different."
    )
    if args.runs < 3:
        print(
            f"\n  NOTE: {args.runs} run(s). A spread of 0.000 here means "
            "'measured once', not 'perfectly stable'."
        )
    if args.out:
        Path(args.out).write_text(json.dumps([r.as_row() for r in results], indent=2))
        print(f"\n  wrote {args.out}")
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    """Read runs already paid for, rather than estimating from call counts."""
    from eval.profile import profile_traces, render, summary

    root = Path(args.traces) if args.traces else TRACE_DIR / "eval"
    if not root.exists():
        print(
            f"\nno traces at {root}. Runs write them there; pass --traces to "
            "point somewhere else."
        )
        return 1

    profiles = profile_traces(root)
    print(f"\nprofiling {len(profiles)} trace(s) under {root}")
    print(render(profiles))

    if args.out:
        Path(args.out).write_text(json.dumps(summary(profiles), indent=2))
        print(f"\n  wrote {args.out}")
    return 0


def _cmd_retrieval(args: argparse.Namespace) -> int:
    """Score retrieval over the golden queries and print the ablation."""
    from eval.retrieval import ablation, score_retrieval
    from src.rag.corpus import build_corpus
    from src.rag.index import HybridIndex, corpus_stats

    chunks, files = build_corpus()
    stats = corpus_stats(chunks)
    print(
        f"\ncorpus: {stats.chunks} chunks from {stats.sources} source(s), "
        f"{stats.tokens} tokens"
    )
    for name, count in sorted(stats.by_source.items()):
        print(f"  {count:>4}  {name}")
    if not files:
        print(
            "  (no ingested documents — drop .txt or .md files in corpus/ to "
            "grow it; only checksums are ever committed)"
        )

    table = ablation(chunks, k=args.k)
    print("\n" + table.to_string(index=False))
    note = table.attrs.get("note")
    if note:
        print(f"\n  {note}")

    report = score_retrieval(HybridIndex(chunks), k=args.k)
    if report.misses:
        print("\n  queries where nothing relevant came back:")
        for miss in report.misses:
            print(f"    {miss}")
    if args.out:
        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\n  wrote {args.out}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    cases = build_golden_set(system_id=args.system)
    tuning = [c for c in cases if c.split == "tuning"]
    heldback = [c for c in cases if c.split == "heldback"]
    write_cases(tuning, GOLDEN_DIR / "cases_tuning.jsonl")
    write_cases(heldback, GOLDEN_DIR / "cases_heldback.jsonl")
    print(f"wrote {len(tuning)} tuning and {len(heldback)} held-back cases\n")
    print(composition(cases).to_string(index=False))
    return 0


def _apply_subset(cases: list[Any], args: argparse.Namespace) -> list[Any]:
    """Narrow the case list, and say what the narrowing costs."""
    chosen = select(
        cases,
        only=getattr(args, "only_cases", None),
        limit=getattr(args, "limit", None),
        sample=getattr(args, "sample", None),
        seed=getattr(args, "seed", DEFAULT_SEED),
    )
    print(f"\n  {describe(chosen)}")
    warning = warn_about(chosen, cases)
    if warning:
        print("\n  " + warning.replace("\n", "\n  ") + "\n")
    return chosen


class RecordTooShort(RuntimeError):
    """The ingested record does not cover every golden-case window."""


def _check_the_record_covers_every_case(cases: list[Any]) -> None:
    """Fail before the first case rather than partway through.

    The golden windows are fixed dates. A record that stops early — almost
    always a download cut short by a network failure rather than an archive
    that ends there — makes some of them unmeasurable, and `materialise` says
    so one case at a time, deep in a run, in terms that sound like a bug in the
    case rather than a hole in the data.
    """
    by_system: dict[int, list[Any]] = {}
    for case in cases:
        by_system.setdefault(case.system_id, []).append(case)

    for system_id, group in sorted(by_system.items()):
        frame = load_plant(system_id, DATA_DIR).frame
        first, last = frame.index.min(), frame.index.max()
        outside = [
            c
            for c in group
            if pd.Timestamp(c.start, tz="UTC") < first
            or pd.Timestamp(c.end, tz="UTC") > last
        ]
        if not outside:
            continue
        names = ", ".join(sorted(c.id for c in outside)[:6])
        more = f" (+{len(outside) - 6} more)" if len(outside) > 6 else ""
        raise RecordTooShort(
            f"system {system_id}: the ingested record covers "
            f"{first.date()} .. {last.date()}, but {len(outside)} of "
            f"{len(group)} cases need data outside it — {names}{more}.\n"
            "  This usually means the download was cut short by a network "
            "failure. Re-run:\n"
            f"    python -m src.data.cli ingest --system {system_id} "
            "--years 2016 2017\n"
            "  It is idempotent, and it now refuses to write a record short "
            "by fetch failure rather than reporting success."
        )


def _cmd_run(args: argparse.Namespace) -> int:
    paths = {
        "tuning": GOLDEN_DIR / "cases_tuning.jsonl",
        "heldback": GOLDEN_DIR / "cases_heldback.jsonl",
    }
    splits = ["tuning", "heldback"] if args.split == "both" else [args.split]
    cases: list[Any] = []
    for split in splits:
        if not paths[split].exists():
            print(f"missing {paths[split]}; run `python -m eval.runner build` first")
            return 1
        cases.extend(load_cases(paths[split]))

    cases = _apply_subset(cases, args)

    try:
        _check_the_record_covers_every_case(cases)
        scores, predictions = _run_engine(args.engine, cases, args)
    except (RecordTooShort, FileNotFoundError) as exc:
        print(f"\n{exc}")
        return 1
    except RuntimeError as exc:
        print(exc)
        return 1
    report = aggregate(scores, predictions)

    print(f"\n=== {args.engine} engine, {len(cases)} cases ===\n")
    for split, stats in report.by_split.items():
        print(f"  {split}  ({stats['cases']} cases)")
        print(f"    overall accuracy (macro-F1)   {stats['macro_f1']:.3f}")
        print(f"    category accuracy             {stats['category_accuracy']:.3f}")
        print(f"    cause accuracy                {stats['cause_accuracy']:.3f}")
        far = stats["false_alarm_rate"]
        print(
            f"    false alarms on look-alikes   "
            f"{'n/a' if far is None else f'{far:.3f}'}"
            f"   ({stats['look_alike_cases']} cases)"
        )
        ca = stats["correct_abstention_rate"]
        print(
            f"    correct 'not enough evidence' "
            f"{'n/a' if ca is None else f'{ca:.3f}'}"
            f"   ({stats['unresolvable_cases']} cases)"
        )
        mf = stats["missed_fault_rate"]
        print(
            f"    missed real faults            {'n/a' if mf is None else f'{mf:.3f}'}"
        )
        print()

    if report.overfitting_gap is not None:
        print(f"  tuning minus held-back gap: {report.overfitting_gap:+.3f}")

    print("\n  agency")
    for key, value in report.agency.items():
        print(f"    {key:<32} {value}")

    targets = report.meets_v1_targets()
    print("\n  v1 targets (held-back split)")
    for name, passed in targets.items():
        # Three-valued. "not measured" is not a failure, and printing it as one
        # made a tuning-only run look like four missed targets it was never in
        # a position to assess.
        label = "n/a " if passed is None else ("PASS" if passed else "FAIL")
        print(f"    {label}  {name}")
    if all(passed is None for passed in targets.values()):
        print("    (the held-back split was not run, so none of these were assessed)")

    if args.confusion:
        print("\n  truth (rows) vs predicted (columns)")
        print(confusion(scores).to_string())

    if args.out:
        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\n  wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pv-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Generate the golden set.")
    p_build.add_argument("--system", type=int, default=4902)
    p_build.set_defaults(func=_cmd_build)

    def add_subset_args(
        target: argparse.ArgumentParser, include_only: bool = True
    ) -> None:
        """Run part of the split. Every command that runs cases accepts these.

        A subset is for debugging, not for scoring — the harness prints which
        metrics the chosen subset cannot support, above the results rather than
        below them.
        """
        target.add_argument(
            "--sample",
            type=int,
            default=None,
            metavar="N",
            help=(
                "Run N cases drawn stratified by category and settledness, so "
                "a small run still contains look-alikes and unresolvable cases "
                "in roughly the split's proportions. Prefer this to --limit."
            ),
        )
        target.add_argument(
            "--limit",
            type=int,
            default=None,
            metavar="N",
            help=(
                "Run the first N cases in file order. The file is grouped by "
                "fault type, so these are not a cross-section."
            ),
        )
        if include_only:
            # `experiments` already spends --only on ablation configurations.
            target.add_argument(
                "--only",
                dest="only_cases",
                type=str,
                default=None,
                metavar="IDS",
                help="Run exactly these cases, e.g. --only G-001,G-007.",
            )
        target.add_argument(
            "--seed",
            type=int,
            default=DEFAULT_SEED,
            help="Fixes --sample, so two runs draw the same cases.",
        )

    p_run = sub.add_parser("run", help="Score a diagnostic over the golden set.")
    p_run.add_argument("--engine", default="rules", choices=["rules", "agent"])
    p_run.add_argument(
        "--split", default="both", choices=["tuning", "heldback", "both"]
    )
    add_subset_args(p_run)
    p_run.add_argument("--confusion", action="store_true")
    p_run.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Collapse the per-step trace to a one-line counter per case. "
            "The run still reports what it is doing; it just says less."
        ),
    )
    p_run.add_argument("--out", type=str, default=None)
    p_run.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Record each case to FILE as it finishes, and skip cases already "
            "in it. Re-run the same command with the same FILE after a run "
            "dies and it picks up where it stopped."
        ),
    )
    p_run.add_argument(
        "--no-review",
        action="store_true",
        help="Agent only: skip the critic. The ablation that prices review.",
    )
    p_run.add_argument(
        "--no-knowledge",
        action="store_true",
        help="Agent only: run with no domain knowledge retrieved.",
    )
    p_run.set_defaults(func=_cmd_run)

    p_cmp = sub.add_parser(
        "compare", help="Run both engines over the same cases and compare."
    )
    p_cmp.add_argument(
        "--split", default="heldback", choices=["tuning", "heldback", "both"]
    )
    p_cmp.add_argument("--out", type=str, default=None)
    add_subset_args(p_cmp)
    p_cmp.add_argument("--no-review", action="store_true")
    p_cmp.add_argument("--quiet", action="store_true")
    p_cmp.add_argument("--no-knowledge", action="store_true")
    p_cmp.set_defaults(func=_cmd_compare)

    p_prof = sub.add_parser(
        "profile",
        help="Where time and money went, from traces already on disk.",
    )
    p_prof.add_argument(
        "--traces",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory of trace files (default: traces/eval).",
    )
    p_prof.add_argument("--out", type=str, default=None)
    p_prof.set_defaults(func=_cmd_profile)

    p_ret = sub.add_parser(
        "retrieval", help="Score retrieval over the golden queries and ablate."
    )
    p_ret.add_argument("--k", type=int, default=10)
    p_ret.add_argument("--out", type=str, default=None)
    p_ret.set_defaults(func=_cmd_retrieval)

    p_exp = sub.add_parser(
        "experiments",
        help="Ablate the agent against itself over N runs. Needs an API key.",
    )
    p_exp.add_argument(
        "--split", default="heldback", choices=["tuning", "heldback", "both"]
    )
    p_exp.add_argument(
        "--runs", type=int, default=3, help="Fresh runs per configuration."
    )
    p_exp.add_argument(
        "--only", nargs="*", default=None, help="Subset of configurations to run."
    )
    add_subset_args(p_exp, include_only=False)
    p_exp.add_argument("--out", type=str, default=None)
    p_exp.set_defaults(func=_cmd_experiments)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
