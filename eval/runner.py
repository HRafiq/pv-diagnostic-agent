"""Run a diagnostic over the golden set and report per split.

    python -m eval.runner build          # write the golden set
    python -m eval.runner run --engine rules
    python -m eval.runner run --engine rules --split heldback

The agent engine lands at step 4. Until then `rules` is the only one, which is
the intended order: the harness exists before the thing it measures, so the
first number the agent ever produces is comparable to a baseline that already
ran.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from eval.golden import build_golden_set, composition, load_cases, write_cases
from eval.metrics import CaseScore, Prediction, aggregate, confusion, score_case
from eval.scenarios import materialise
from src.baseline.rules import RulesEngine
from src.config import REPO_ROOT
from src.data.sources import SystemMetadata
from src.physics.modelchain import module_gamma_pdc

GOLDEN_DIR = REPO_ROOT / "eval" / "golden"
DATA_DIR = REPO_ROOT / "data" / "raw"


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


def run_rules_engine(cases: list[Any]) -> tuple[list[CaseScore], list[Prediction]]:
    scores: list[CaseScore] = []
    predictions: list[Prediction] = []

    for case in cases:
        meta, gamma = _system_metadata(case.system_id)
        engine = RulesEngine(
            dc_capacity_kw=meta.dc_capacity_kw,
            gamma_pdc=gamma,
            latitude=meta.latitude,
            longitude=meta.longitude,
            altitude_m=meta.altitude_m,
            tilt_deg=meta.tilt_deg,
            azimuth_deg=meta.azimuth_deg,
            ac_ceiling_kw=meta.ac_capacity_kw_hint,
        )
        materialised = materialise(case, DATA_DIR)
        started = time.monotonic()
        verdict = engine.diagnose(materialised.frame)
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


def _cmd_build(args: argparse.Namespace) -> int:
    cases = build_golden_set(system_id=args.system)
    tuning = [c for c in cases if c.split == "tuning"]
    heldback = [c for c in cases if c.split == "heldback"]
    write_cases(tuning, GOLDEN_DIR / "cases_tuning.jsonl")
    write_cases(heldback, GOLDEN_DIR / "cases_heldback.jsonl")
    print(f"wrote {len(tuning)} tuning and {len(heldback)} held-back cases\n")
    print(composition(cases).to_string(index=False))
    return 0


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

    if args.engine != "rules":
        print(f"engine {args.engine!r} arrives at a later build step")
        return 1

    scores, predictions = run_rules_engine(cases)
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

    print("\n  v1 targets (held-back split)")
    for name, passed in report.meets_v1_targets().items():
        print(f"    {'PASS' if passed else 'FAIL'}  {name}")

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

    p_run = sub.add_parser("run", help="Score a diagnostic over the golden set.")
    p_run.add_argument("--engine", default="rules", choices=["rules", "agent"])
    p_run.add_argument(
        "--split", default="both", choices=["tuning", "heldback", "both"]
    )
    p_run.add_argument("--confusion", action="store_true")
    p_run.add_argument("--out", type=str, default=None)
    p_run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
