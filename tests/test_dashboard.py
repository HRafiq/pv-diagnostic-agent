"""Dashboard seam.

The architecture rule is that the dashboard never reimplements anything in
``src/``. These tests check the two consequences that matter: the state
endpoint's values agree with ``src/`` rather than being computed locally, and
the vocabulary rule holds on the rendered page.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.app import app
from src.config import build_clock, load_models_config

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_state_endpoint_responds() -> None:
    assert client.get("/api/state").status_code == 200


def test_state_agrees_with_src_rather_than_recomputing() -> None:
    payload = client.get("/api/state").json()
    assert payload["now"] == build_clock().now().isoformat()
    assert payload["model_profile"] == load_models_config().active_profile
    # Sites now come from ingested manifests rather than static config, so the
    # page always describes the data actually on disk.
    assert isinstance(payload["systems"], list)


def test_state_exposes_every_agent_node() -> None:
    nodes = {n["node"] for n in client.get("/api/state").json()["nodes"]}
    assert nodes == {"planner", "router", "synthesizer", "critic"}


def test_no_secret_leaks_through_the_api() -> None:
    body = client.get("/api/state").text
    assert "sk-ant" not in body
    # Presence is reported as a boolean, never the key itself.
    assert isinstance(client.get("/api/state").json()["api_key_present"], bool)


def test_index_renders() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "PV Diagnostic Agent" in response.text


def test_tabs_are_disabled_until_their_step_lands() -> None:
    tabs = client.get("/api/state").json()["tabs"]
    assert len(tabs) == 5
    enabled = {t["key"] for t in tabs if t["enabled"]}
    # Plant went live at step 1; the rest arrive with their own steps.
    assert enabled == {"plant"}
    assert {t["key"] for t in tabs} == {
        "plant",
        "watcher",
        "investigate",
        "scenarios",
        "evaluation",
    }


# --------------------------------------------------------------------------
# Vocabulary rule (CLAUDE.md): no ML jargon reaches the interface.
# --------------------------------------------------------------------------
BANNED_UI_TERMS = [
    "abstention",
    "confounder",
    "groundedness",
    "recall@",
    "macro-f1",
    "holdout",
    "hypothesis",
    "hypotheses",
]


@pytest.mark.parametrize("term", BANNED_UI_TERMS)
def test_banned_jargon_is_absent_from_rendered_html(term: str) -> None:
    assert term not in client.get("/").text.lower()


@pytest.mark.parametrize("term", BANNED_UI_TERMS)
def test_banned_jargon_is_absent_from_the_template_source(term: str) -> None:
    # Catches jargon sitting in a branch that step 0 does not render.
    for path in (REPO_ROOT / "dashboard" / "templates").rglob("*.html"):
        body = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)
        assert term not in body.lower(), f"{path.name} shows UI jargon: {term!r}"


# --------------------------------------------------------------------------
# Vendored assets — the dashboard must work offline, at pinned versions.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "asset",
    ["vendor/alpine-3.14.9.min.js", "vendor/plotly-2.35.2.min.js", "app.css"],
)
def test_static_assets_are_served(asset: str) -> None:
    assert client.get(f"/static/{asset}").status_code == 200


def test_no_cdn_references_in_templates() -> None:
    # A CDN reference makes the dashboard fail offline and lets a chart library
    # change version silently underneath an evaluation run.
    for path in (REPO_ROOT / "dashboard" / "templates").rglob("*.html"):
        html = path.read_text(encoding="utf-8")
        for host in ("cdn.jsdelivr.net", "cdn.plot.ly", "unpkg.com", "cdnjs."):
            assert host not in html, f"{path.name} loads from a CDN: {host}"


def test_favicon_is_served_so_the_page_logs_no_404() -> None:
    response = client.get("/favicon.svg")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
