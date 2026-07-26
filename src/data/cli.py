"""``python -m src.data.cli`` — fetch a PVDAQ system into ``data/raw/``.

python -m src.data.cli ingest --system 4902 --years 2016 2017
python -m src.data.cli describe --system 4902
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import REPO_ROOT
from src.data.ingest import PartialDownload, ingest_system, load_dataset
from src.data.sources import fetch_system_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pv-data")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Download and normalise a system.")
    p_ingest.add_argument("--system", type=int, required=True)
    p_ingest.add_argument("--years", type=int, nargs="+", required=True)
    p_ingest.add_argument("--interval", type=int, default=15, help="minutes")
    p_ingest.add_argument("--workers", type=int, default=8)
    p_ingest.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "raw")
    p_ingest.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Write the record even if some days could not be fetched. Off by "
            "default: a silently short record changes what every trailing "
            "baseline measures."
        ),
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    p_desc = sub.add_parser("describe", help="Summarise an ingested system.")
    p_desc.add_argument("--system", type=int, required=True)
    p_desc.add_argument("--dir", type=Path, default=REPO_ROOT / "data" / "raw")
    p_desc.set_defaults(func=_cmd_describe)

    args = parser.parse_args(argv)
    return int(args.func(args))


def _cmd_ingest(args: argparse.Namespace) -> int:
    meta = fetch_system_metadata(args.system)
    print(f"{meta.name}  ({meta.location})")
    print(
        f"  {meta.dc_capacity_kw:.1f} kW DC | tilt {meta.tilt_deg:g}deg "
        f"azimuth {meta.azimuth_deg:g}deg | {meta.strings} strings x "
        f"{meta.modules_per_string} modules"
    )
    print(f"  downloading {len(args.years)} year(s): {args.years}")

    try:
        result = ingest_system(
            meta,
            years=args.years,
            out_dir=args.out,
            interval_minutes=args.interval,
            max_workers=args.workers,
            allow_partial=args.allow_partial,
        )
    except PartialDownload as exc:
        # Nothing was written. Saying so matters: the natural assumption on
        # seeing an error is that a half-finished file is sitting on disk.
        print(f"\n  download incomplete — nothing written\n\n  {exc}", file=sys.stderr)
        return 1

    tz = result.timezone
    print(f"\n  rows       {result.rows:,} at {result.interval_minutes} min")
    print(f"  range      {result.start} .. {result.end}")
    if result.missing_days:
        print(f"  gaps       {result.missing_days} day(s) absent from the archive")
    if result.fetch_failures:
        # Only reachable under --allow-partial; otherwise the ingest raised.
        print(
            f"  INCOMPLETE {result.fetch_failures} day(s) could not be fetched. "
            "This record is short by network failure, not by archive gap. "
            "Every trailing-baseline and same-span-last-year measurement over "
            "the affected span is unreliable; re-run without --allow-partial "
            "once the connection is stable."
        )
    print(
        f"  timezone   UTC{tz.offset_hours:+g} from "
        f"{tz.samples_used:,} clear-day samples, r={tz.correlation:.4f} "
        f"margin={tz.margin_over_runner_up:.4f} "
        f"({'trustworthy' if tz.trustworthy else 'NEEDS REVIEW'})"
    )
    print(f"             {tz.note}")
    print("\n  channels resolved:")
    for name, choice in sorted(result.resolution.chosen.items()):
        print(
            f"    {name:<16} <- {choice.source_column:<32} "
            f"max {choice.observed_max:.1f}"
        )
        for column, why in choice.rejected:
            print(f"        rejected {column}: {why}")
    if result.resolution.unresolved:
        print(f"    unresolved: {', '.join(result.resolution.unresolved)}")
    print(
        f"    per-string channels: "
        f"{len(result.resolution.string_current_columns)} current, "
        f"{len(result.resolution.string_power_columns)} power"
    )
    print(f"\n  parquet    {result.parquet_path}")
    print(f"  manifest   {result.manifest_path}")
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    matches = sorted(args.dir.glob(f"system_{args.system}_*.parquet"))
    if not matches:
        print(
            f"no ingested data for system {args.system} in {args.dir}", file=sys.stderr
        )
        return 1
    frame = load_dataset(matches[-1])
    print(
        f"{matches[-1].name}: {len(frame):,} rows, "
        f"{frame.index.min()} .. {frame.index.max()}"
    )
    described = frame.describe().T[["count", "mean", "min", "max"]]
    print(described.round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
