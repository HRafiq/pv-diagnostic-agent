"""Validated colour roles, for both themes.

Dark mode is **selected, not flipped**. It is the same three hues re-stepped for
a dark surface and validated against it independently — an automatic inversion
of light-mode colours fails the contrast and colour-vision checks.

Both palettes pass all-pairs on their own surface: lightness band, chroma floor,
colour-vision separation (worst pair ΔE 9.2 light / 9.4 dark, deutan),
normal-vision separation (24.0 / 20.9) and contrast.

**Three slots is the cap, not a coincidence.** Past three the palette cannot
clear the all-pairs floors, so a fourth measure folds into "other", becomes its
own small multiple, or takes a non-colour channel. Never a generated hue.

One documented relief: on the light surface the aqua slot sits at 2.72:1, below
the 3:1 bar. The mitigation is that identity is never carried by colour alone —
a legend is always present and the table view is always reachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["THEMES", "Mode", "Theme", "plotly_layout", "theme"]

Mode = Literal["dark", "light"]


@dataclass(frozen=True)
class Theme:
    """Every colour role a chart needs, for one mode."""

    mode: Mode
    surface: str
    grid: str
    ink: str
    ink_dim: str
    series: tuple[str, ...]
    sequential: tuple[str, ...]
    diverging: tuple[str, ...]
    status: dict[str, str]

    def colorscale(self) -> list[list[Any]]:
        """The sequential ramp in Plotly's normalised-stop form."""
        last = len(self.sequential) - 1
        return [[i / last, c] for i, c in enumerate(self.sequential)]


_STATUS = {
    # Reserved roles. Never reused as a series colour, and always shipped
    # alongside a label so state is never carried by colour alone.
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

THEMES: dict[str, Theme] = {
    "dark": Theme(
        mode="dark",
        surface="#171b21",
        grid="#262c35",
        ink="#e6e9ee",
        ink_dim="#9aa4b2",
        series=("#3987e5", "#d95926", "#199e70"),
        sequential=(
            "#0d366b",
            "#184f95",
            "#256abf",
            "#3987e5",
            "#6da7ec",
            "#9ec5f4",
            "#cde2fb",
        ),
        diverging=(
            "#184f95",
            "#3987e5",
            "#9ec5f4",
            "#383835",
            "#f0a3a3",
            "#e34948",
            "#a52020",
        ),
        status=_STATUS,
    ),
    "light": Theme(
        mode="light",
        surface="#fbfbfa",
        grid="#e3e2df",
        ink="#12151a",
        ink_dim="#5c6470",
        series=("#2a78d6", "#eb6834", "#1baf7a"),
        sequential=(
            "#cde2fb",
            "#9ec5f4",
            "#6da7ec",
            "#3987e5",
            "#256abf",
            "#184f95",
            "#0d366b",
        ),
        diverging=(
            "#184f95",
            "#3987e5",
            "#9ec5f4",
            "#f0efec",
            "#f0a3a3",
            "#e34948",
            "#a52020",
        ),
        status=_STATUS,
    ),
}


def theme(mode: str = "dark") -> Theme:
    """Look up a theme, defaulting to dark rather than raising."""
    return THEMES.get(mode, THEMES["dark"])


def plotly_layout(
    title: str,
    y_title: str,
    x_title: str = "",
    height: int = 280,
    show_legend: bool = True,
    mode: str = "dark",
) -> dict[str, Any]:
    """Shared layout: recessive grid and axes, text in neutral ink.

    Series colour lives on the marks only. Axis labels, tick labels and legend
    text stay in ink tokens — a coloured mark beside the label carries identity,
    so the text never has to.
    """
    t = theme(mode)
    return {
        "title": {
            "text": title,
            "font": {"size": 13, "color": t.ink},
            "x": 0,
            "xanchor": "left",
        },
        "height": height,
        "margin": {"l": 58, "r": 18, "t": 34, "b": 38},
        "paper_bgcolor": t.surface,
        "plot_bgcolor": t.surface,
        "font": {"color": t.ink_dim, "size": 11},
        "xaxis": {
            "title": {"text": x_title, "font": {"size": 11, "color": t.ink_dim}},
            "gridcolor": t.grid,
            "zerolinecolor": t.grid,
            "linecolor": t.grid,
        },
        "yaxis": {
            "title": {"text": y_title, "font": {"size": 11, "color": t.ink_dim}},
            "gridcolor": t.grid,
            "zerolinecolor": t.grid,
            "linecolor": t.grid,
        },
        "showlegend": show_legend,
        "legend": {
            "orientation": "h",
            "y": -0.22,
            "x": 0,
            "font": {"color": t.ink_dim, "size": 10},
        },
        # Crosshair plus one tooltip covering every series at that instant: the
        # question is always "what was happening at this moment", never "what
        # was this single point".
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": "#0f1216" if t.mode == "dark" else "#ffffff",
            "bordercolor": t.grid,
            "font": {"color": t.ink, "size": 11},
        },
    }
