"""Dashboard seam.

The architecture rule is that the dashboard never reimplements anything in
``src/``. These tests check the two consequences that matter: the state
endpoint's values agree with ``src/`` rather than being computed locally, and
the vocabulary rule holds on the rendered page.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
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
    # Plant went live at step 1, Investigate at step 4; the rest arrive with
    # their own steps.
    assert enabled == {"plant", "investigate"}
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


# --------------------------------------------------------------------------
# Investigate tab (step 4) — the tape
# --------------------------------------------------------------------------
def _write_trace(root: Path, steps: list[dict[str, object]]) -> None:
    from src.clock import FrozenClock
    from src.trace.models import TraceStep
    from src.trace.writer import TraceWriter

    clock = FrozenClock(datetime(2017, 4, 25, 9, 0, tzinfo=UTC))
    with TraceWriter("INV-UNIT-001", clock=clock, root=root) as trace:
        for step in steps:
            trace.write(TraceStep(**step))  # type: ignore[arg-type]


@pytest.fixture
def traces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PV_TRACES_DIR", str(tmp_path))
    return tmp_path


def test_no_investigations_is_reported_not_hidden(traces: Path) -> None:
    """A fresh clone has no traces — they are output, not source. The tab has
    to explain that rather than render an empty page."""
    payload = client.get("/api/investigations").json()
    assert payload["investigations"] == []
    assert payload["traces_dir"] == str(traces)
    assert isinstance(payload["api_key_present"], bool)


def test_a_trace_is_listed_with_its_counters(traces: Path) -> None:
    _write_trace(
        traces,
        [
            {
                "kind": "plan",
                "node": "planner",
                "was_planned": True,
                "result": "opening plan",
                "args": {
                    "question": "why is output down?",
                    "hypotheses": [
                        {
                            "id": "H1",
                            "cause": "a string has failed",
                            "status": "open",
                            "consequence_if_true": "a site visit",
                        },
                        {
                            "id": "H2",
                            "cause": "cloudy weather",
                            "status": "open",
                            "consequence_if_true": "nothing",
                        },
                    ],
                },
                "cost_usd": 0.01,
                "tokens": 500,
            },
            {
                "kind": "tool",
                "node": "compute_temp_corrected_pr",
                "was_planned": True,
                "result": "PR is 0.784",
                "cost_usd": 0.005,
                "tokens": 200,
            },
            {
                "kind": "adaptive",
                "node": "time_of_day_profile",
                "was_planned": False,
                "reason_for_choosing": "the loss looked confined to the morning",
                "result": "worst hour 07:00",
                "excludes": ["H2"],
                "cost_usd": 0.005,
                "tokens": 200,
            },
            {
                "kind": "answer",
                "node": "synthesizer",
                "was_planned": True,
                "result": "one string is down",
                "args": {
                    "settled": True,
                    "title": "One string is down",
                    "cause": "string outage",
                    "category": "fault",
                    "confidence": 0.9,
                },
                "cost_usd": 0.02,
                "tokens": 900,
            },
        ],
    )
    listed = client.get("/api/investigations").json()["investigations"]
    assert len(listed) == 1
    row = listed[0]
    assert row["id"] == "INV-UNIT-001"
    assert row["measurements"] == 2 and row["unplanned"] == 1
    assert row["cost_usd"] == pytest.approx(0.04)
    assert row["settled"] is True


def test_one_investigation_returns_the_tape_and_the_causes(traces: Path) -> None:
    test_a_trace_is_listed_with_its_counters(traces)
    payload = client.get("/api/investigation/INV-UNIT-001").json()

    assert [s["kind"] for s in payload["steps"]] == [
        "plan",
        "tool",
        "adaptive",
        "answer",
    ]
    assert payload["counters"]["unplanned"] == 1
    assert payload["counters"]["tokens"] == 1800

    # A cause a measurement ruled out is shown as ruled out, off the tape
    # itself rather than recomputed.
    causes = {c["cause"]: c["status"] for c in payload["possible_causes"]}
    assert causes == {"a string has failed": "open", "cloudy weather": "excluded"}


def test_the_unplanned_step_carries_its_reason(traces: Path) -> None:
    """The reason is the evidence of agency; without it the tape is just a log."""
    test_a_trace_is_listed_with_its_counters(traces)
    steps = client.get("/api/investigation/INV-UNIT-001").json()["steps"]
    unplanned = next(s for s in steps if not s["was_planned"])
    assert unplanned["reason_for_choosing"] == "the loss looked confined to the morning"


def test_a_missing_investigation_is_a_404(traces: Path) -> None:
    assert client.get("/api/investigation/INV-NOPE").status_code == 404


def test_a_traversing_id_is_rejected(traces: Path) -> None:
    assert client.get("/api/investigation/..%2F..%2Fetc%2Fpasswd").status_code in (
        400,
        404,
    )


def test_the_api_never_speaks_of_hypotheses(traces: Path) -> None:
    """Internal code may use the precise term. The interface may not."""
    test_a_trace_is_listed_with_its_counters(traces)
    body = client.get("/api/investigation/INV-UNIT-001").text.lower()
    assert "hypothes" not in body
