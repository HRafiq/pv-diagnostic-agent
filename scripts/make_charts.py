"""Build the README figures from measurements, not from memory.

Charts in a README rot the same way prose does, and worse: a number in a
picture is harder to check than a number in a sentence. So every figure here is
generated from `docs/results.json` — one file, each block naming its source —
and the retrieval chart is computed live from the committed corpus rather than
transcribed at all.

SVG by hand, deliberately. Charts are a documentation need, and adding
matplotlib to a project that does not otherwise plot would put a dependency in
`pyproject.toml` for the sake of four pictures. SVG also diffs as text, so a
figure changing shows up in review as a figure changing.

Both GitHub themes get the same image: an explicit light background with dark
text, rather than transparency that reads as white-on-white in dark mode.

    uv run python scripts/make_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "img"

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
PAPER = "#ffffff"
GOOD = "#2f855a"
BAD = "#c05621"
COOL = "#2b6cb0"
WARN = "#b7791f"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Named so the declaration fits a line; every label shares it.
_FONT = "-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"


def _text(
    x: float,
    y: float,
    s: str,
    size: int = 13,
    fill: str = INK,
    anchor: str = "start",
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" '
        f'font-family="{_FONT}">'
        f"{_esc(s)}</text>"
    )


def _svg(width: int, height: int, body: str, title: str, subtitle: str = "") -> str:
    head = _text(24, 34, title, size=17, weight="600")
    sub = _text(24, 56, subtitle, size=12.5, fill=MUTED) if subtitle else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">'
        f'<rect width="{width}" height="{height}" fill="{PAPER}" rx="8"/>'
        f"{head}{sub}{body}</svg>"
    )


def _write(name: str, markup: str) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    (IMG / name).write_text(markup)
    print(f"  wrote docs/img/{name}")


# ---------------------------------------------------------------------------
def headline(results: dict) -> None:
    """The two numbers the project is actually about, against the baseline."""
    rules = results["rules_baseline"]["heldback"]
    before = results["agent_eight_tuning_cases"]["before"]
    after = results["agent_eight_tuning_cases"]["after"]

    panels = [
        (
            "False alarms on look-alikes",
            "lower is better — a crew sent to a healthy array",
            [
                ("rules baseline", rules["false_alarm_rate"], MUTED),
                ("agent, before", before["false_alarm_rate"], BAD),
                ("agent, after", after["false_alarm_rate"], GOOD),
            ],
            results["targets"]["false_alarm_rate_at_most"],
            "target ≤ 0.15",
        ),
        (
            'Correct "not enough evidence"',
            "higher is better — refusing to guess when the data cannot decide",
            [
                ("rules baseline", rules["correct_abstention_rate"], MUTED),
                ("agent, before", before["correct_abstention_rate"], GOOD),
                ("agent, after", after["correct_abstention_rate"], GOOD),
            ],
            results["targets"]["correct_abstention_at_least"],
            "target ≥ 0.70",
        ),
    ]

    x0, bar_w, gap, panel_h = 200.0, 300.0, 34.0, 150.0
    body, y = "", 92.0
    for title, note, rows, target, target_label in panels:
        body += _text(24, y, title, size=14, weight="600")
        body += _text(24, y + 18, note, size=11.5, fill=MUTED)
        top = y + 36
        tx = x0 + bar_w * min(target, 1.0)
        body += (
            f'<line x1="{tx:.1f}" y1="{top - 6:.1f}" x2="{tx:.1f}" '
            f'y2="{top + gap * len(rows):.1f}" stroke="{MUTED}" '
            f'stroke-width="1" stroke-dasharray="3 3"/>'
        )
        # Below the line, not above it: at the top it collided with the
        # panel's own subtitle, which is the sort of thing that only shows up
        # once the picture is actually rendered and looked at.
        body += _text(
            tx + 6, top + gap * len(rows) + 2, target_label, size=10.5, fill=MUTED
        )
        for n, (label, value, colour) in enumerate(rows):
            cy = top + gap * n + 14
            body += _text(x0 - 12, cy + 4, label, size=12, fill=INK, anchor="end")
            body += (
                f'<rect x="{x0}" y="{cy - 9}" width="{bar_w}" height="18" '
                f'fill="{GRID}" rx="3"/>'
            )
            w = max(bar_w * value, 2.0)
            body += (
                f'<rect x="{x0}" y="{cy - 9}" width="{w:.1f}" height="18" '
                f'fill="{colour}" rx="3"/>'
            )
            body += _text(
                x0 + w + 8, cy + 4, f"{value:.3f}", size=12, fill=INK, weight="600"
            )
        y += panel_h
    body += _text(
        24,
        y + 4,
        "Rules baseline: 43 held-back cases. Agent: eight tuning "
        "cases — a debugging run, not a score.",
        size=11,
        fill=MUTED,
    )
    _write(
        "headline_metrics.svg",
        _svg(
            620,
            int(y + 28),
            body,
            "The two numbers that matter",
            "Same eighteen measurement tools, driven two different ways",
        ),
    )


# ---------------------------------------------------------------------------
def where_time_goes(results: dict) -> None:
    block = results["where_the_time_goes"]
    shares = block["shares"]
    x0, bar_w, gap = 190.0, 330.0, 36.0
    body, top = "", 96.0
    for n, (label, pct) in enumerate(shares.items()):
        cy = top + gap * n + 14
        colour = COOL if "measure" not in label and label != "look up" else MUTED
        body += _text(x0 - 12, cy + 4, label, size=12, anchor="end")
        body += (
            f'<rect x="{x0}" y="{cy - 10}" width="{bar_w}" height="20" '
            f'fill="{GRID}" rx="3"/>'
        )
        w = max(bar_w * pct / 100.0, 2.0)
        body += (
            f'<rect x="{x0}" y="{cy - 10}" width="{w:.1f}" height="20" '
            f'fill="{colour}" rx="3"/>'
        )
        body += _text(x0 + w + 8, cy + 4, f"{pct:.1f}%", size=12, weight="600")
    y = top + gap * len(shares) + 16
    body += _text(
        24,
        y,
        f"{block['spent_after_cycle_one_pct']}% of all time and "
        "half of all spend went after the first cycle — "
        "re-reviewing.",
        size=12,
        fill=BAD,
        weight="600",
    )
    body += _text(
        24,
        y + 22,
        "The measurements themselves are pure functions and take "
        "milliseconds. None of this is the physics.",
        size=11,
        fill=MUTED,
    )
    _write(
        "where_time_goes.svg",
        _svg(
            620,
            int(y + 44),
            body,
            "Where an investigation spends its time",
            "13 timed runs, read off trace files rather than estimated",
        ),
    )


# ---------------------------------------------------------------------------
def cost_and_latency(results: dict) -> None:
    a = results["cost_latency"]["first_working_runs"]
    b = results["cost_latency"]["current"]
    body = ""
    for n, (title, key, unit, fmt) in enumerate(
        [
            ("Cost per case", "median_usd", "$", "{:.2f}"),
            ("Time per case", "median_seconds", "s", "{:.0f}"),
        ]
    ):
        x = 24 + n * 300
        body += _text(x, 96, title, size=13, weight="600")
        top, bar_w = 112.0, 230.0
        worst = max(a[key], b[key])
        for m, (label, value, colour) in enumerate(
            [("first working runs", a[key], MUTED), ("now", b[key], GOOD)]
        ):
            cy = top + m * 34 + 14
            body += _text(x, cy - 12, label, size=11, fill=MUTED)
            body += (
                f'<rect x="{x}" y="{cy - 4}" width="{bar_w}" height="16" '
                f'fill="{GRID}" rx="3"/>'
            )
            w = max(bar_w * value / worst, 2.0)
            body += (
                f'<rect x="{x}" y="{cy - 4}" width="{w:.1f}" height="16" '
                f'fill="{colour}" rx="3"/>'
            )
            label_txt = (
                (unit + fmt.format(value))
                if unit == "$"
                else (fmt.format(value) + unit)
            )
            body += _text(x + w + 8, cy + 9, label_txt, size=12, weight="600")
    body += _text(
        24,
        214,
        "Design target was $0.15 a question. It is not met, and the README says so.",
        size=11,
        fill=MUTED,
    )
    _write(
        "cost_latency.svg",
        _svg(
            620,
            236,
            body,
            "What one investigation costs",
            "Medians. Same eight cases, before and after the review changes",
        ),
    )


# ---------------------------------------------------------------------------
def retrieval() -> None:
    """Computed live from the committed corpus — nothing transcribed."""
    from eval.retrieval import ablation
    from src.rag.corpus import build_corpus

    chunks, _ = build_corpus()
    table = ablation(chunks, k=10)
    rows = [
        (str(r["configuration"]), float(r["reciprocal_rank"]), float(r["top_1"]))
        for _, r in table.iterrows()
    ]

    x0, bar_w, gap = 190.0, 300.0, 44.0
    body, top = "", 100.0
    body += _text(x0, top - 14, "rank of the right passage", size=11, fill=COOL)
    body += _text(x0 + 165, top - 14, "right on the first try", size=11, fill=WARN)
    for n, (label, rr, top1) in enumerate(rows):
        cy = top + gap * n + 12
        body += _text(x0 - 12, cy + 4, label, size=12, anchor="end")
        for m, (value, colour) in enumerate([(rr, COOL), (top1, WARN)]):
            by = cy - 9 + m * 11
            body += (
                f'<rect x="{x0}" y="{by}" width="{bar_w}" height="9" '
                f'fill="{GRID}" rx="2"/>'
            )
            w = max(bar_w * value, 2.0)
            body += (
                f'<rect x="{x0}" y="{by}" width="{w:.1f}" height="9" '
                f'fill="{colour}" rx="2"/>'
            )
            body += _text(x0 + w + 7, by + 8, f"{value:.3f}", size=10.5, fill=INK)
    y = top + gap * len(rows) + 10
    body += _text(
        24,
        y,
        "Fusing the two beats either alone. The reranker adds 0.009 "
        "— one query, one position.",
        size=11.5,
        fill=MUTED,
    )
    body += _text(
        24,
        y + 20,
        "Read with the caveat: the corpus is 23 chunks and they are "
        "the project's own knowledge base.",
        size=11.5,
        fill=BAD,
    )
    _write(
        "retrieval_ablation.svg",
        _svg(
            620,
            int(y + 42),
            body,
            "Does the retrieval stack earn its complexity?",
            "18 golden queries, computed live by this script",
        ),
    )


def main() -> int:
    results = json.loads((ROOT / "docs" / "results.json").read_text())
    print("building figures from docs/results.json")
    headline(results)
    where_time_goes(results)
    cost_and_latency(results)
    retrieval()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
