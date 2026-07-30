"""Chart *specs* — data plus config, never images.

Every builder returns a plain dict that Plotly.js renders directly. That keeps
`src/` UI-agnostic: the dashboard is a thin caller, and a later swap to a
different front end is a UI change rather than a rewrite.

Each chart here calls the same physics functions the agent's tools call, so the
two cannot diverge (CLAUDE.md).

**Deviation from handoff §6.4, recorded in DECISION 0011.** The brief asks for
power and irradiance on a *dual axis*. Two y-scales let the author place the
crossover anywhere, so the reader cannot tell a real divergence from a chosen
one — it is the single most misleading chart form in common use. The teaching
moment the brief wants ("output halving while irradiance halves with it") is
better served by two stacked panels on a shared time axis: the covariance is
read off aligned peaks and troughs, with no scale trickery in the way.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.viz.palette import plotly_layout, theme

__all__ = [
    "actual_vs_expected_spec",
    "gap_calendar_spec",
    "power_and_irradiance_spec",
    "pr_timeseries_spec",
    "string_share_heatmap_spec",
]

Spec = dict[str, Any]


def _iso(index: pd.Index) -> list[str]:
    return [str(t) for t in pd.DatetimeIndex(index)]


def power_and_irradiance_spec(
    frame: pd.DataFrame,
    ac_capacity_kw: float | None = None,
    power_column: str = "ac_power_kw",
    poa_column: str = "poa_wm2",
    mode: str = "dark",
) -> list[Spec]:
    """Two stacked panels sharing a time axis — power above, irradiance below.

    The AC ceiling is drawn as a reference line on the power panel. Flat-topped
    midday power sitting exactly on that line is clipping: by design, not a
    fault, and the most common false alarm in the business.
    """
    t = theme(mode)
    times = _iso(frame.index)
    power = pd.to_numeric(frame[power_column], errors="coerce")
    poa = pd.to_numeric(frame[poa_column], errors="coerce")

    power_layout = plotly_layout(
        "AC power",
        "kW",
        height=230,
        show_legend=ac_capacity_kw is not None,
        mode=mode,
    )
    if ac_capacity_kw:
        power_layout["shapes"] = [
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "yref": "y",
                "y0": ac_capacity_kw,
                "y1": ac_capacity_kw,
                "line": {"color": t.status["warning"], "width": 1, "dash": "dot"},
            }
        ]
        power_layout["annotations"] = [
            {
                "xref": "paper",
                "x": 1,
                "xanchor": "right",
                "y": ac_capacity_kw,
                "yanchor": "bottom",
                "text": f"inverter AC ceiling {ac_capacity_kw:.0f} kW",
                "showarrow": False,
                "font": {"color": t.status["warning"], "size": 10},
            }
        ]

    return [
        {
            "data": [
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "AC power",
                    "x": times,
                    "y": [None if pd.isna(v) else round(float(v), 2) for v in power],
                    "line": {"color": t.series[0], "width": 2},
                    "hovertemplate": "%{y:.1f} kW<extra></extra>",
                }
            ],
            "layout": power_layout,
        },
        {
            "data": [
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "plane-of-array irradiance",
                    "x": times,
                    "y": [None if pd.isna(v) else round(float(v), 1) for v in poa],
                    "line": {"color": t.series[1], "width": 2},
                    "hovertemplate": "%{y:.0f} W/m²<extra></extra>",
                }
            ],
            "layout": plotly_layout(
                "Sunlight reaching the array",
                "W/m²",
                height=200,
                show_legend=False,
                mode=mode,
            ),
        },
    ]


def pr_timeseries_spec(pr_frame: pd.DataFrame, mode: str = "dark") -> Spec:
    """Performance ratio, measured and temperature-corrected, on one axis.

    Both series are the same dimensionless quantity, so one scale is honest
    here. The gap between them is the whole point: it widens every summer on a
    perfectly healthy plant, and that gap is the seasonal false alarm.
    """
    t = theme(mode)
    times = _iso(pr_frame.index)

    def series(column: str) -> list[float | None]:
        return [None if pd.isna(v) else round(float(v), 4) for v in pr_frame[column]]

    layout = plotly_layout(
        "Performance ratio", "ratio (1.0 = lossless at STC)", mode=mode
    )
    layout["yaxis"]["range"] = [0.4, 1.05]

    # Where completeness is poor the ratio is computed from a fraction of the
    # intervals; shade it so a reader cannot take it at face value.
    if "data_completeness" in pr_frame:
        low = pr_frame.index[pr_frame["data_completeness"] < 0.5]
        width = (
            pr_frame.index[1] - pr_frame.index[0] if len(pr_frame.index) > 1 else None
        )
        layout["shapes"] = [
            {
                "type": "rect",
                "xref": "x",
                "x0": str(stamp),
                "x1": str(stamp + width) if width is not None else str(stamp),
                "yref": "paper",
                "y0": 0,
                "y1": 1,
                "fillcolor": t.status["warning"],
                "opacity": 0.13,
                "line": {"width": 0},
                "layer": "below",
            }
            for stamp in low
        ]

    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": "as measured",
                "x": times,
                "y": series("pr"),
                "line": {"color": t.series[0], "width": 2},
                "hovertemplate": "%{y:.3f}<extra>as measured</extra>",
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "corrected for module temperature",
                "x": times,
                "y": series("pr_temperature_corrected"),
                "line": {"color": t.series[1], "width": 2},
                "hovertemplate": "%{y:.3f}<extra>temperature corrected</extra>",
            },
        ],
        "layout": layout,
    }


def string_share_heatmap_spec(
    frame: pd.DataFrame,
    string_columns: list[str],
    poa_column: str = "poa_wm2",
    min_poa_wm2: float = 400.0,
    freq: str = "1D",
    mode: str = "dark",
) -> Spec:
    """Each string's share of total DC current, per day.

    The single most useful chart in the product. Strings on one array see the
    same sunlight, so in good light their shares sit flat near 1/N. **One dark
    row is equipment; every row dark on the same day is weather** — and that
    distinction is exactly what a plant-level number cannot make.

    Shares rather than raw amps, because raw current falls on every string when
    a cloud passes, which is not a fault.
    """
    t = theme(mode)
    present = [c for c in string_columns if c in frame]
    if not present:
        empty = plotly_layout("Per-string balance", "", mode=mode)
        return {"data": [], "layout": empty}

    mask = pd.to_numeric(frame[poa_column], errors="coerce") >= min_poa_wm2
    window = frame.loc[mask, present].apply(pd.to_numeric, errors="coerce")
    total = window.sum(axis=1)

    # A share is only meaningful when the array is actually carrying current.
    # `total > 0` is not enough: an interval totalling 0.01 A produces shares in
    # the hundreds of thousands, and one such interval stretches the colour
    # scale until every real value collapses into a single indistinguishable
    # band. Require the total to be a real fraction of its own typical level.
    floor = 0.2 * float(total.median()) if len(total) else 0.0
    usable = total > max(floor, 1e-6)
    shares = window[usable].div(total[usable], axis=0)
    daily = shares.resample(freq).mean()

    expected = 1.0 / len(present)
    z = [
        [None if pd.isna(v) else round(float(v), 5) for v in daily[column]]
        for column in present
    ]
    labels = [f"string {i}" for i in range(1, len(present) + 1)]

    layout = plotly_layout(
        f"Per-string share of DC current (even split would be {expected:.3f})",
        "",
        height=240,
        show_legend=False,
        mode=mode,
    )
    layout["yaxis"]["gridcolor"] = "rgba(0,0,0,0)"
    return {
        "data": [
            {
                "type": "heatmap",
                "x": _iso(daily.index),
                "y": labels,
                "z": z,
                # Sequential single hue: this is magnitude, not identity.
                "colorscale": t.colorscale(),
                # Fix the band at 0..2/N so the even split sits mid-scale and
                # the colour of a given share never depends on the window
                # selected. An auto-scaled heatmap re-normalises to whatever is
                # on screen, so a healthy array and a failing one can render
                # identically — which defeats the entire point of the chart.
                "zmin": 0.0,
                "zmax": 2.0 * expected,
                "hovertemplate": "%{y} · %{x}<br>share %{z:.3f}<extra></extra>",
                "colorbar": {
                    "title": {
                        "text": "share",
                        "font": {"size": 10, "color": t.ink_dim},
                    },
                    "tickfont": {"size": 9, "color": t.ink_dim},
                    "outlinecolor": t.grid,
                    "thickness": 10,
                },
            }
        ],
        "layout": layout,
    }


def actual_vs_expected_spec(
    measured_kw: pd.Series,
    expected_kw: pd.Series,
    colour_by: pd.Series | None = None,
    colour_label: str = "module temperature (°C)",
    mode: str = "dark",
) -> Spec:
    """Measured against modelled output, with the 1:1 line.

    Points on the line mean the model describes the plant. A cloud of points
    bending *below* the line only at high module temperature is not a fault —
    it is a mis-specified temperature coefficient, and colouring by temperature
    is what makes that visible instead of mysterious.
    """
    t = theme(mode)
    measured = pd.to_numeric(measured_kw, errors="coerce")
    expected = pd.to_numeric(expected_kw, errors="coerce")
    valid = measured.notna() & expected.notna() & (expected > 1.0)
    measured, expected = measured[valid], expected[valid]

    # Thin to keep the payload small; the shape of a cloud does not need
    # every one of 70,000 points.
    if len(measured) > 4000:
        stride = len(measured) // 4000
        measured, expected = measured.iloc[::stride], expected.iloc[::stride]

    marker: dict[str, Any] = {
        "size": 4,
        "opacity": 0.5,
        "color": t.series[0],
        "line": {"width": 0},
    }
    if colour_by is not None:
        aligned = pd.to_numeric(colour_by, errors="coerce").reindex(measured.index)
        marker = {
            "size": 4,
            "opacity": 0.6,
            "color": [None if pd.isna(v) else round(float(v), 2) for v in aligned],
            "colorscale": t.colorscale(),
            "colorbar": {
                "title": {
                    "text": colour_label,
                    "font": {"size": 10, "color": t.ink_dim},
                },
                "tickfont": {"size": 9, "color": t.ink_dim},
                "outlinecolor": t.grid,
                "thickness": 10,
            },
            "line": {"width": 0},
        }

    top = float(max(measured.max(), expected.max())) if len(measured) else 1.0
    layout = plotly_layout(
        "Measured against modelled output",
        "measured (kW)",
        x_title="modelled (kW)",
        height=320,
        show_legend=False,
        mode=mode,
    )
    layout["hovermode"] = "closest"
    return {
        "data": [
            {
                "type": "scattergl",
                "mode": "markers",
                "name": "intervals",
                "x": [round(float(v), 2) for v in expected],
                "y": [round(float(v), 2) for v in measured],
                "marker": marker,
                "hovertemplate": (
                    "modelled %{x:.1f} kW<br>measured %{y:.1f} kW<extra></extra>"
                ),
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "1:1",
                "x": [0, top],
                "y": [0, top],
                "line": {"color": t.ink_dim, "width": 1, "dash": "dash"},
                "hoverinfo": "skip",
            },
        ],
        "layout": layout,
    }


def gap_calendar_spec(daily_completeness: pd.Series, mode: str = "dark") -> Spec:
    """Fraction of each day's intervals that actually recorded power.

    A missing day is not a zero-output day. Conflating them is what turned the
    July 2016 telemetry gap in this dataset into an apparent 90% loss, and this
    calendar is how a reader sees the difference at a glance.
    """
    t = theme(mode)
    if daily_completeness.empty:
        empty = plotly_layout("Data completeness", "", mode=mode)
        return {"data": [], "layout": empty}

    values = daily_completeness.astype(float)
    colours = [
        t.status["critical"]
        if v == 0
        else t.status["serious"]
        if v < 0.5
        else t.status["warning"]
        if v < 0.95
        else t.series[0]
        for v in values
    ]
    layout = plotly_layout(
        "Readings recorded per day",
        "fraction of the day",
        height=190,
        show_legend=False,
        mode=mode,
    )
    layout["yaxis"]["range"] = [0, 1.02]
    layout["bargap"] = 0.0
    return {
        "data": [
            {
                "type": "bar",
                "x": _iso(values.index),
                "y": [round(float(v), 4) for v in values],
                "marker": {"color": colours, "line": {"width": 0}},
                "hovertemplate": "%{x}<br>%{y:.0%} of readings<extra></extra>",
            }
        ],
        "layout": layout,
    }
