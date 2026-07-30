"""The Watcher CLI — advances the replay clock, sweeps, writes findings.

Resolves the tension in the handoff between §6.3 (time is a replay clock) and
§6.5 ("the sweep runs when new data lands, the dashboard never triggers a run").
With a replay clock there is no real data arrival, and a request-scoped web app
has nowhere to put a daemon. So *this process is the arrival of data*: it steps
the clock forward one interval, treats everything that fell inside that interval
as newly landed, runs the deterministic detectors over it, and writes findings
to the store. The dashboard only ever reads.

    python watcher.py status
    python watcher.py step --engine rules
    python watcher.py run --until 2017-04-01 --step 7D --engine rules
    python watcher.py findings

The sweep decides *when to look*; the diagnostic decides *why*. Keeping those
apart is what stops the Watcher becoming a rules engine on a timer, with the
agent reduced to writing up a conclusion already reached.

`--engine rules` needs no API key and is the default, so the whole Watcher path
is demonstrable offline. `--engine agent` runs the investigation loop instead
and needs one.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.agent.llm import build_client
from src.agent.loop_graph import investigate_with_graph as investigate
from src.baseline.rules import RulesEngine
from src.clock import Clock
from src.config import (
    REPO_ROOT,
    Settings,
    build_clock,
    load_models_config,
    load_site_defaults,
)
from src.data.plant import load_plant
from src.detect.sweep import DeficitSignal, sweep_window, trailing_window
from src.findings.build import finding_from_verdict
from src.findings.models import Finding
from src.findings.store import FindingsStore

DEFAULT_SYSTEM = 4902
# What each cause means operationally, for the unsettled path. The store cannot
# file an unsettled finding without it: an abstention that does not say what
# each surviving cause would cost is uncertainty without information.
CONSEQUENCES = {
    "clipping": "nothing is wrong and no energy is recoverable",
    "curtailment": (
        "the energy is real and may be compensable under the connection agreement"
    ),
}


def _parse_step(text: str) -> timedelta:
    """Parse a compact interval such as '1D', '6H', '15min'."""
    raw = text.strip()
    for suffix, unit in (("min", "minutes"), ("H", "hours"), ("D", "days")):
        if raw.endswith(suffix):
            value = raw[: -len(suffix)]
            try:
                return timedelta(**{unit: float(value)})
            except ValueError:
                break
    raise argparse.ArgumentTypeError(
        f"cannot parse step {text!r}; expected forms like '1D', '6H', '15min'"
    )


def _parse_date(text: str) -> datetime:
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="watcher",
        description="Advance the replay clock and sweep for new findings.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show clock and configuration state.")
    p_status.set_defaults(func=_cmd_status)

    def add_sweep_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--step", type=_parse_step, default=timedelta(days=7))
        target.add_argument("--system", type=int, default=DEFAULT_SYSTEM)
        target.add_argument("--engine", default="rules", choices=["rules", "agent"])
        target.add_argument("--lookback", type=int, default=14)
        target.add_argument("--store", type=str, default=None)
        target.add_argument("--dry-run", action="store_true")
        # Defaults to data/raw/. Overridable so a sweep can run against an
        # alternate ingest, and so the CLI tests do not silently depend on
        # whether the PVDAQ download happens to be present on this machine.
        target.add_argument(
            "--data-dir",
            type=str,
            default=None,
            help="Directory of ingested systems (default: data/raw).",
        )

    p_step = sub.add_parser("step", help="Advance one interval and sweep once.")
    add_sweep_args(p_step)
    p_step.set_defaults(func=_cmd_step)

    p_run = sub.add_parser("run", help="Advance repeatedly until a date.")
    p_run.add_argument("--until", type=_parse_date, required=True)
    add_sweep_args(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_find = sub.add_parser("findings", help="Show what is currently open.")
    p_find.add_argument("--store", type=str, default=None)
    p_find.add_argument("--all", action="store_true", help="Include closed findings.")
    p_find.set_defaults(func=_cmd_findings)

    p_ack = sub.add_parser(
        "ack", help="Acknowledge or suppress a finding. Appends; never edits."
    )
    p_ack.add_argument("identity")
    p_ack.add_argument(
        "--state", default="acknowledged", choices=["acknowledged", "suppressed"]
    )
    p_ack.add_argument("--note", default="")
    p_ack.add_argument("--store", type=str, default=None)
    p_ack.set_defaults(func=_cmd_ack)

    args = parser.parse_args(argv)
    return int(args.func(args))


def _cmd_status(_: argparse.Namespace) -> int:
    settings = Settings.from_env()
    site_defaults = load_site_defaults()
    models = load_models_config()
    clock = build_clock(site_defaults)

    print(f"simulated now   {clock.now().isoformat()}")
    print(f"replay start    {clock.start.isoformat()}  (speed {clock.speed}x)")
    print(f"sites           {', '.join(sorted(site_defaults.sites))}")
    print(f"model profile   {models.active_profile}")
    for node in ("planner", "router", "synthesizer", "critic"):
        cfg = models.node(node, models.active_profile)
        effort = cfg.effort or "-"
        print(
            f"  {node:<12} {cfg.model:<20} "
            f"effort={effort:<7} max_tokens={cfg.max_tokens}"
        )
    print(f"api key         {'set' if settings.anthropic_api_key else 'NOT SET'}")
    return 0


def _store_path(args: argparse.Namespace) -> Path:
    return Path(args.store) if args.store else REPO_ROOT / "findings" / "store.jsonl"


def _cmd_step(args: argparse.Namespace) -> int:
    clock = build_clock()
    before = clock.now()
    after = clock.advance(args.step)
    print(f"advanced {before.isoformat()} -> {after.isoformat()}")
    return sweep_once(args, clock)


def _cmd_run(args: argparse.Namespace) -> int:
    clock = build_clock()
    if args.until <= clock.now():
        print(f"--until {args.until.isoformat()} is not in the future", file=sys.stderr)
        return 1
    while clock.now() < args.until:
        clock.advance(args.step)
        code = sweep_once(args, clock)
        if code != 0:
            return code
    print(f"reached {clock.now().isoformat()}")
    return 0


def sweep_once(args: argparse.Namespace, clock: Clock) -> int:
    """One sweep: detect, investigate what fired, write findings.

    The two halves are deliberately separate. Detectors say a window is worth
    looking at and carry no cause; the diagnostic says why. Collapsing them
    would make the Watcher a rules engine on a timer.
    """
    now = clock.now()
    start, end = trailing_window(now, args.lookback)
    print(f"  sweeping {start} .. {end}")

    data_dir = Path(args.data_dir) if getattr(args, "data_dir", None) else None
    try:
        bundle = load_plant(args.system, data_dir)
    except FileNotFoundError as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1

    ctx = bundle.context(bundle.frame, scope=f"{bundle.meta.name} (whole plant)")
    try:
        signals = sweep_window(ctx, start, end)
    except Exception as exc:  # a window outside the record, most often
        print(f"  no usable data in this window ({exc})")
        return 0

    for signal in signals:
        print(
            f"  ! {signal.detector}: {signal.measurement} = {signal.value:.4f} "
            f"(threshold {signal.threshold})"
        )
    if args.dry_run:
        print("  [dry-run] not investigating and not writing findings")
        return 0

    findings = _investigate(signals, ctx, args, clock) if signals else []
    # The store is told about *every* sweep, including the quiet ones. A sweep
    # that finds nothing is what ages an open finding towards resolved, so
    # returning early here would leave a fixed fault open forever on a plant
    # that had gone quiet — which is the failure the lifecycle exists to avoid.
    store = FindingsStore(_store_path(args), clock=clock)
    written = store.observe(findings, note=f"sweep at {now.date()}")
    if not signals and not written:
        print("  nothing worth investigating")
        return 0
    for record in written:
        print(f"  -> {record.lifecycle:<12} {record.identity}")
    return 0


def _investigate(
    signals: list[DeficitSignal],
    ctx: object,
    args: argparse.Namespace,
    clock: Clock,
) -> list[Finding]:
    """Diagnose each signal with the requested engine."""
    findings: list[Finding] = []
    for index, signal in enumerate(signals, start=1):
        investigation_id = f"INV-{clock.now().date()}-{index:03d}"
        try:
            if args.engine == "rules":
                verdict = RulesEngine(
                    window={"start": signal.start, "end": signal.end}
                ).diagnose(ctx)  # type: ignore[arg-type]
                findings.append(
                    finding_from_verdict(
                        verdict,
                        signal,
                        finding_id=f"F-{investigation_id}",
                        detected_at=clock.now(),
                        investigation_id=investigation_id,
                        consequences=CONSEQUENCES,
                    )
                )
            else:
                findings.append(_run_agent(signal, ctx, investigation_id, clock))
        except Exception as exc:
            # A finding that cannot legally be built is not silently dropped —
            # it is reported, because a Watcher that quietly loses findings is
            # worse than one that reports nothing.
            print(f"  x {signal.detector}: could not file a finding — {exc}")
    return findings


def _run_agent(
    signal: DeficitSignal, ctx: object, investigation_id: str, clock: Clock
) -> Finding:
    settings = Settings.from_env()
    client = build_client(load_models_config(), settings.anthropic_api_key)
    out = investigate(
        signal.question,
        ctx,  # type: ignore[arg-type]
        client,
        clock,
        investigation_id=investigation_id,
        start=signal.start,
        end=signal.end,
        trace_root=REPO_ROOT / "traces",
    )
    if out.finding is None:
        raise RuntimeError(
            "; ".join(out.errors) or "the investigation produced no usable answer"
        )
    return out.finding


def _cmd_findings(args: argparse.Namespace) -> int:
    clock = build_clock()
    store = FindingsStore(_store_path(args), clock=clock)
    records = store.current() if args.all else store.open_findings()
    if not records:
        print("nothing open")
        return 0
    print(f"{'lifecycle':<13} {'energy kWh':>11}  finding")
    for record in records:
        finding = record.finding
        energy = f"{finding.energy_at_stake_kwh:,.0f}"
        mark = "" if finding.energy_verified else " (unverified)"
        print(f"{record.lifecycle:<13} {energy:>11}{mark}  {finding.title}")
        print(f"{'':<13} {'':>11}  {record.identity}")
    return 0


def _cmd_ack(args: argparse.Namespace) -> int:
    clock = build_clock()
    store = FindingsStore(_store_path(args), clock=clock)
    record = store.set_lifecycle(args.identity, args.state, note=args.note)
    if record is None:
        print(f"no finding with identity {args.identity!r}", file=sys.stderr)
        return 1
    print(f"{args.identity} -> {args.state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
