"""Dashboard server — FastAPI + Jinja2, with Alpine.js on the client.

Architecture rule (CLAUDE.md): this package is *thin*. It imports from ``src/``
and renders; it never recomputes anything. A chart of PR calls
``src.physics.compute_pr()`` — the same function the tool node calls — which is
what guarantees the dashboard and the agent cannot diverge.

The server is read-only with respect to the investigation loop. It never
triggers a sweep; ``watcher.py`` does that and writes findings, and the
dashboard reads what is already there.

    uvicorn dashboard.app:app --reload

Step 0 ships the shell and one endpoint proving the seam: the page reads the
clock and the model config through ``src/``, so the wiring is verified before
anything is built on it. Tab 1 (Plant) goes live at step 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src import __version__
from src.config import Settings, build_clock, load_models_config, load_site_defaults

HERE = Path(__file__).resolve().parent

app = FastAPI(title="PV Diagnostic Agent", version=__version__)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))

# Tab order matches the handoff §6.4-6.8. Only the ones that exist are enabled.
TABS: list[dict[str, Any]] = [
    {"key": "plant", "label": "Plant", "enabled": False, "step": 1},
    {"key": "watcher", "label": "Watcher", "enabled": False, "step": 8},
    {"key": "investigate", "label": "Investigate", "enabled": False, "step": 4},
    {"key": "scenarios", "label": "Scenario builder", "enabled": False, "step": 2},
    {"key": "evaluation", "label": "Evaluation", "enabled": False, "step": 3},
]


@app.get("/api/state")
def api_state() -> dict[str, Any]:
    """Everything the shell needs. Read-only; computed entirely from ``src/``."""
    settings = Settings.from_env()
    site_defaults = load_site_defaults()
    models = load_models_config()
    clock = build_clock(site_defaults)

    return {
        "version": __version__,
        # Simulated time, not wall clock. The replay clock is the only time
        # source, so this is what "now" means everywhere in the interface.
        "now": clock.now().isoformat(),
        "replay_start": clock.start.isoformat(),
        "replay_speed": clock.speed,
        "sites": [
            {"key": key, "name": site.name, "timezone": site.timezone}
            for key, site in sorted(site_defaults.sites.items())
        ],
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
