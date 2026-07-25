"""The Watcher CLI — advances the replay clock, sweeps, writes findings.

Resolves the tension in the handoff between §6.3 (time is a replay clock) and
§6.5 ("the sweep runs when new data lands, the dashboard never triggers a run").
With a replay clock there is no real data arrival, and a request-scoped web app
has nowhere to put a daemon. So *this process is the arrival of data*: it steps
the clock forward one interval, treats everything that fell inside that interval
as newly landed, runs the deterministic detectors over it, and writes findings
to the store. The dashboard only ever reads.

    python watcher.py step --days 1
    python watcher.py run --until 2019-06-30 --step 1D

Step 0 wires the clock, config and trace plumbing and stops there. Detectors
arrive at step 8; until then `--dry-run` is the only supported mode.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta

from src.config import Settings, build_clock, load_models_config, load_site_defaults


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

    p_step = sub.add_parser("step", help="Advance one interval and sweep once.")
    p_step.add_argument("--step", type=_parse_step, default=timedelta(days=1))
    p_step.add_argument("--dry-run", action="store_true", default=True)
    p_step.set_defaults(func=_cmd_step)

    p_run = sub.add_parser("run", help="Advance repeatedly until a date.")
    p_run.add_argument("--until", type=_parse_date, required=True)
    p_run.add_argument("--step", type=_parse_step, default=timedelta(days=1))
    p_run.add_argument("--dry-run", action="store_true", default=True)
    p_run.set_defaults(func=_cmd_run)

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


def _cmd_step(args: argparse.Namespace) -> int:
    clock = build_clock()
    before = clock.now()
    after = clock.advance(args.step)
    print(f"advanced {before.isoformat()} -> {after.isoformat()}")
    return _sweep(before, after, dry_run=args.dry_run)


def _cmd_run(args: argparse.Namespace) -> int:
    clock = build_clock()
    if args.until <= clock.now():
        print(f"--until {args.until.isoformat()} is not in the future", file=sys.stderr)
        return 1
    while clock.now() < args.until:
        before = clock.now()
        after = clock.advance(args.step)
        rc = _sweep(before, after, dry_run=args.dry_run)
        if rc != 0:
            return rc
    print(f"reached {clock.now().isoformat()}")
    return 0


def _sweep(window_start: datetime, window_end: datetime, dry_run: bool) -> int:
    """Run the deterministic detectors over data that landed in the window."""
    if dry_run:
        print(
            f"  [dry-run] would sweep {window_start.date()} .. {window_end.date()} "
            "(detectors land at step 8)"
        )
        return 0
    raise NotImplementedError(
        "the sweep is implemented at step 8; run with --dry-run until then"
    )


if __name__ == "__main__":
    raise SystemExit(main())
