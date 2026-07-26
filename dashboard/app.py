"""Dashboard server — FastAPI + Jinja2, with Alpine.js on the client.

Architecture rule (CLAUDE.md): this package is *thin*. It imports from ``src/``
and renders; it never recomputes anything. A chart of PR calls
``physics.compute_pr()`` — the same function the tool node calls — which is what
guarantees the dashboard and the agent cannot diverge.

The server is read-only with respect to the investigation loop. It never
triggers a sweep; ``watcher.py`` does that and writes findings, and the
dashboard reads what is already there.

    uvicorn dashboard.app:app --reload
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src import __version__
from src.config import REPO_ROOT, Settings, build_clock, load_models_config
from src.data.ingest import load_dataset, string_columns
from src.data.quality import profile_quality
from src.data.sources import SystemMetadata
from src.physics.modelchain import ExpectationModel, module_gamma_pdc
from src.physics.performance import compute_pr, pr_timeseries
from src.trace.writer import read_trace
from src.viz import specs

HERE = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data" / "raw"

app = FastAPI(title="PV Diagnostic Agent", version=__version__)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))

TABS: list[dict[str, Any]] = [
    {"key": "plant", "label": "Plant", "enabled": True, "step": 1},
    {"key": "watcher", "label": "Watcher", "enabled": False, "step": 8},
    {"key": "investigate", "label": "Investigate", "enabled": True, "step": 4},
    {"key": "scenarios", "label": "Scenario builder", "enabled": False, "step": 2},
    {"key": "evaluation", "label": "Evaluation", "enabled": False, "step": 3},
]


# ---------------------------------------------------------------------------
# Dataset discovery and caching
# ---------------------------------------------------------------------------
def _manifests() -> list[dict[str, Any]]:
    import json

    out: list[dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("system_*_manifest.json")):
        try:
            out.append(json.loads(path.read_text()))
        except Exception:
            continue
    return out


@lru_cache(maxsize=4)
def _dataset(system_id: int) -> tuple[pd.DataFrame, SystemMetadata, float]:
    """Load an ingested system plus the metadata the physics needs.

    Metadata comes from the committed manifest rather than a live fetch, so the
    dashboard works offline and always describes the data actually on disk.
    """
    import json

    manifest_path = DATA_DIR / f"system_{system_id}_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, f"system {system_id} has not been ingested")
    manifest = json.loads(manifest_path.read_text())

    parquet = sorted(DATA_DIR.glob(f"system_{system_id}_*.parquet"))
    if not parquet:
        raise HTTPException(404, f"no parquet for system {system_id}")

    fields = manifest["system_metadata"]
    meta = SystemMetadata(**{**fields, "raw": {}})
    gamma = module_gamma_pdc(meta.module_model) or -0.0040
    return load_dataset(parquet[-1]), meta, gamma


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/state")
def api_state() -> dict[str, Any]:
    """Shell state. Read-only; computed entirely from ``src/``."""
    settings = Settings.from_env()
    models = load_models_config()
    clock = build_clock()
    manifests = _manifests()

    return {
        "version": __version__,
        "now": clock.now().isoformat(),
        "replay_start": clock.start.isoformat(),
        "replay_speed": clock.speed,
        "model_profile": models.active_profile,
        "nodes": [
            {
                "node": node,
                "model": models.node(node).model,
                "effort": models.node(node).effort,
                "max_tokens": models.node(node).max_tokens,
            }
            for node in ("planner", "router", "synthesizer", "critic")
        ],
        "api_key_present": bool(settings.anthropic_api_key),
        "tabs": TABS,
        "systems": [
            {
                "system_id": m["system_id"],
                "name": m["name"],
                "location": m["location"],
                "years": m["years"],
                "rows": m["rows"],
                "interval_minutes": m["interval_minutes"],
                "start": m["start"],
                "end": m["end"],
                "string_channels": m.get("string_channels", 0),
                "timezone": m.get("timezone", {}),
                "capacity_kw": m["system_metadata"]["dc_capacity_kw"],
            }
            for m in manifests
        ],
    }


@app.get("/api/plant/{system_id}")
def api_plant(
    system_id: int,
    start: str | None = None,
    end: str | None = None,
    mode: str = Query("dark", pattern="^(dark|light)$"),
    resolution: str = Query("1D", pattern="^(15min|1H|1D|1W|1ME)$"),
) -> dict[str, Any]:
    """Everything the Plant tab draws, for one system and window.

    Every number here comes from ``src/physics`` — the same functions the
    agent's tools call. The dashboard does no arithmetic of its own.
    """
    frame, meta, gamma = _dataset(system_id)
    window: pd.DataFrame = frame
    if start or end:
        index = pd.DatetimeIndex(frame.index)
        keep = pd.Series(True, index=frame.index)
        if start:
            keep &= index >= pd.Timestamp(start, tz="UTC")
        if end:
            keep &= index <= pd.Timestamp(end, tz="UTC")
        window = frame.loc[keep]
    if window.empty:
        raise HTTPException(400, "no data in that window")

    currents, _ = string_columns(window)
    ac_ceiling = meta.ac_capacity_kw_hint

    overall = compute_pr(window, meta.dc_capacity_kw, gamma)
    pr_frame = pr_timeseries(window, meta.dc_capacity_kw, gamma, freq=resolution)
    model = ExpectationModel(meta.dc_capacity_kw, gamma, ac_ceiling)
    expectation = model.run(window)
    quality = profile_quality(window, dc_capacity_kw=meta.dc_capacity_kw)

    # The power/irradiance panels are per-interval, so show a bounded slice —
    # two years at 15 minutes is 70k points and communicates nothing.
    detail = window
    if len(detail) > 400:
        detail = window.iloc[-400:]

    power_specs = specs.power_and_irradiance_spec(detail, ac_ceiling, mode=mode)

    return {
        "system": {
            "id": meta.system_id,
            "name": meta.name,
            "location": meta.location,
            "capacity_kw": meta.dc_capacity_kw,
            "ac_ceiling_kw": ac_ceiling,
            "tilt_deg": meta.tilt_deg,
            "azimuth_deg": meta.azimuth_deg,
            "module_model": meta.module_model,
            "gamma_pdc": gamma,
            "strings": meta.strings,
            "string_channels": len(currents),
        },
        "window": {
            "start": str(window.index.min()),
            "end": str(window.index.max()),
            "rows": len(window),
            "detail_start": str(detail.index.min()),
            "detail_end": str(detail.index.max()),
        },
        "summary": {
            "pr": None if pd.isna(overall.pr) else round(overall.pr, 4),
            "pr_temperature_corrected": (
                None
                if pd.isna(overall.pr_temperature_corrected)
                else round(overall.pr_temperature_corrected, 4)
            ),
            "temperature_loss_pct": round(overall.temperature_loss_fraction * 100, 2),
            "energy_mwh": round(overall.energy_kwh / 1000.0, 1),
            "insolation_kwh_m2": round(overall.insolation_kwh_m2, 1),
            "mean_cell_temp_c": round(overall.mean_cell_temp_c, 1),
            "data_completeness_pct": round(overall.data_completeness * 100, 1),
            "deficit_vs_model_pct": round(expectation.deficit_fraction() * 100, 2),
            "clipped_intervals": int(expectation.clipped.sum()),
        },
        "quality": quality.to_dict(),
        "charts": {
            "power": power_specs[0],
            "irradiance": power_specs[1],
            "pr": specs.pr_timeseries_spec(pr_frame, mode=mode),
            "strings": specs.string_share_heatmap_spec(window, currents, mode=mode),
            "scatter": specs.actual_vs_expected_spec(
                expectation.measured_ac_kw,
                expectation.expected_ac_kw,
                window.get("temp_module_c"),
                mode=mode,
            ),
            "gaps": specs.gap_calendar_spec(quality.daily_completeness, mode=mode),
        },
        # The table view. Identity is never carried by colour alone, so every
        # chart has a readable equivalent (accessibility pass, and the relief
        # for the sub-3:1 aqua slot on the light surface).
        "table": [
            {
                "period": str(period),
                "pr": None if pd.isna(row.pr) else round(float(row.pr), 4),
                "pr_temperature_corrected": (
                    None
                    if pd.isna(row.pr_temperature_corrected)
                    else round(float(row.pr_temperature_corrected), 4)
                ),
                "energy_kwh": round(float(row.energy_kwh), 1),
                "mean_cell_temp_c": round(float(row.mean_cell_temp_c), 1),
                "completeness": round(float(row.data_completeness), 3),
            }
            for period, row in pr_frame.tail(400).iterrows()
        ],
    }


# ---------------------------------------------------------------------------
# Investigate tab — the tape
# ---------------------------------------------------------------------------
def _trace_files() -> list[Path]:
    root = Settings.from_env().traces_dir
    if not root.exists():
        return []
    return sorted(
        root.rglob("INV-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )


@app.get("/api/investigations")
def api_investigations() -> dict[str, Any]:
    """Every trace on disk, newest first.

    The dashboard is read-only with respect to the loop: it never starts an
    investigation, it replays ones `watcher.py` or the evaluation harness
    already wrote.
    """
    out: list[dict[str, Any]] = []
    for path in _trace_files()[:200]:
        try:
            steps = list(read_trace(path))
        except Exception:
            continue
        if not steps:
            continue
        plan_step = next((s for s in steps if s.kind == "plan"), None)
        answer = next((s for s in reversed(steps) if s.kind == "answer"), None)
        args = answer.args or {} if answer else {}
        out.append(
            {
                "id": path.stem,
                "question": (plan_step.args or {}).get("question")
                if plan_step
                else None,
                "title": args.get("title"),
                "settled": bool(args.get("settled")),
                "cause": args.get("cause"),
                "category": args.get("category"),
                "measurements": sum(1 for s in steps if s.kind in ("tool", "adaptive")),
                "unplanned": sum(1 for s in steps if s.kind == "adaptive"),
                "cost_usd": round(sum(s.cost_usd for s in steps), 4),
                "steps": len(steps),
                "at": steps[-1].timestamp.isoformat() if steps[-1].timestamp else None,
            }
        )
    return {
        "investigations": out,
        # Traces are output, not source, and they are gitignored. An empty tab
        # on a fresh clone is expected, so say why rather than showing nothing.
        "traces_dir": str(Settings.from_env().traces_dir),
        "api_key_present": bool(Settings.from_env().anthropic_api_key),
    }


# Keys whose *names* are internal vocabulary. The trace keeps the precise term
# — `src/` may (CLAUDE.md) — and the translation happens here, at the boundary,
# because that is the only place that knows something is about to be rendered.
_INTERNAL_ARG_KEYS = frozenset({"hypotheses", "still_standing", "unchecked_lookalikes"})


def _ui_args(args: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip internal-vocabulary keys before a step's args reach the browser.

    The ledger those keys hold is already served as `possible_causes`, so
    nothing is lost — only the word.
    """
    if not args:
        return args
    return {k: v for k, v in args.items() if k not in _INTERNAL_ARG_KEYS}


@app.get("/api/investigation/{investigation_id}")
def api_investigation(investigation_id: str) -> dict[str, Any]:
    """One investigation's tape, plus the counters computed from it."""
    if "/" in investigation_id or ".." in investigation_id:
        raise HTTPException(400, "bad investigation id")
    match = next((p for p in _trace_files() if p.stem == investigation_id), None)
    if match is None:
        raise HTTPException(404, f"no trace for {investigation_id}")

    steps = list(read_trace(match))
    plan_step = next((s for s in steps if s.kind == "plan"), None)
    answer = next((s for s in reversed(steps) if s.kind == "answer"), None)
    excluded = {h for s in steps for h in s.excludes}

    # The possible-causes ledger comes off the tape, not from a recomputation.
    # The last plan wins: a replan after review supersedes the opening one.
    #
    # The field is named `possible_causes`, not `hypotheses`, because it is
    # rendered directly and the interface carries no jargon (CLAUDE.md). The
    # rename happens here rather than in the template so the banned word never
    # reaches a file the browser loads.
    latest_plan = next((s for s in reversed(steps) if s.kind == "plan"), plan_step)
    possible_causes = [
        {**h, "status": "excluded" if h.get("id") in excluded else h.get("status")}
        for h in ((latest_plan.args or {}).get("hypotheses", []) if latest_plan else [])
    ]

    return {
        "id": investigation_id,
        "question": (plan_step.args or {}).get("question") if plan_step else None,
        "possible_causes": possible_causes,
        "answer": answer.args if answer else None,
        "counters": {
            "measurements": sum(1 for s in steps if s.kind in ("tool", "adaptive")),
            "unplanned": sum(1 for s in steps if s.kind == "adaptive"),
            "review_cycles": sum(1 for s in steps if s.kind == "critic"),
            "cost_usd": round(sum(s.cost_usd for s in steps), 4),
            "tokens": sum(s.tokens for s in steps),
            "seconds": round(sum(s.latency_ms for s in steps) / 1000.0, 1),
        },
        "steps": [
            {
                "index": s.step_index,
                "kind": s.kind,
                "node": s.node,
                "result": s.result,
                "was_planned": s.was_planned,
                "reason_for_choosing": s.reason_for_choosing,
                "excludes": s.excludes,
                "args": _ui_args(s.args),
                "cost_usd": round(s.cost_usd, 4),
                "tokens": s.tokens,
                "at": s.timestamp.isoformat() if s.timestamp else None,
            }
            for s in steps
        ],
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"tabs": TABS})


_FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#f0a500"/>'
    '<path d="M8 22 L16 8 L24 22 Z" fill="#191919"/></svg>'
)


@app.get("/favicon.svg")
def favicon() -> Response:
    return Response(content=_FAVICON, media_type="image/svg+xml")
