"""Canonical column names, and the resolver that maps a PVDAQ file onto them.

PVDAQ column names carry per-system instrument ids (``irradiance_poa_o_2204``,
``shuntcurrent_a_avg_3__82646``), so no fixed rename table works across systems.
Worse, a system can expose several channels of the same *kind* in different
units — NIST Ground 1 has three POA channels where only one reads in W/m² and
the others are ~115x smaller.

So resolution is by name **and** physical plausibility: a candidate is only
accepted if its daytime distribution lands in the range the quantity actually
occupies. Choosing on name alone would silently pick a channel two orders of
magnitude off, and every PR in the project would be wrong by that factor while
still looking like a plausible number.

Every choice and every rejection is recorded so the ingest manifest can be
audited later.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

__all__ = [
    "CANONICAL_COLUMNS",
    "QUANTITY_SPECS",
    "STRING_CURRENT_PREFIX",
    "STRING_DC_POWER_PREFIX",
    "ChannelChoice",
    "ChannelResolution",
    "QuantitySpec",
    "resolve_channels",
]

TIMESTAMP = "timestamp"
STRING_CURRENT_PREFIX = "string_current_a_"
STRING_DC_POWER_PREFIX = "string_dc_power_kw_"

CANONICAL_COLUMNS: tuple[str, ...] = (
    "poa_wm2",
    "ghi_wm2",
    "temp_ambient_c",
    "temp_module_c",
    "wind_speed_ms",
    "ac_power_kw",
    "dc_power_kw",
)


@dataclass(frozen=True)
class QuantitySpec:
    """How to find one physical quantity, and what values it may take.

    Args:
        canonical: Canonical column name.
        include: Regexes a candidate column name must match (any of).
        exclude: Regexes that disqualify a candidate (any of).
        plausible_max: (low, high) range the *daytime maximum* must fall in.
        unit: Human-readable unit, for the manifest and the interface.
        required: Ingest fails if this quantity cannot be resolved.
    """

    canonical: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    plausible_max: tuple[float, float]
    unit: str
    required: bool = False


# Plausibility bands are deliberately wide — they exist to catch unit errors and
# dead channels, not to filter out unusual-but-real weather.
QUANTITY_SPECS: tuple[QuantitySpec, ...] = (
    QuantitySpec(
        canonical="poa_wm2",
        include=(r"irradiance_poa", r"^poa_irradiance", r"poa.*irrad"),
        exclude=(r"_ghi_", r"diffuse", r"albedo"),
        # Clear-sky POA peaks near 1000; cloud-edge enhancement reaches ~1400.
        # A channel topping out at 10 is in different units, not a dark day.
        plausible_max=(300.0, 1500.0),
        unit="W/m^2",
        required=True,
    ),
    QuantitySpec(
        canonical="ghi_wm2",
        include=(r"irradiance_ghi", r"^ghi", r"global_horizontal"),
        exclude=(r"_poa_", r"diffuse"),
        plausible_max=(300.0, 1400.0),
        unit="W/m^2",
    ),
    QuantitySpec(
        canonical="temp_ambient_c",
        include=(r"temperature_ambient", r"^ambient_temp", r"air_temp"),
        exclude=(r"module", r"inverter", r"cell"),
        plausible_max=(5.0, 60.0),
        unit="degC",
        required=True,
    ),
    QuantitySpec(
        canonical="temp_module_c",
        include=(r"temperature_module", r"^module_temp", r"back_of_module"),
        exclude=(r"ambient", r"inverter"),
        # Modules run 20-35 K above ambient in full sun.
        plausible_max=(20.0, 95.0),
        unit="degC",
    ),
    QuantitySpec(
        canonical="wind_speed_ms",
        include=(r"wind_speed", r"^wind_spd"),
        exclude=(r"direction", r"_other_"),
        plausible_max=(1.0, 45.0),
        unit="m/s",
    ),
    QuantitySpec(
        canonical="ac_power_kw",
        include=(r"^ac_power_inv", r"^ac_power__", r"^metered_ac_power", r"^ac_power$"),
        exclude=(r"_other_", r"meter", r"voltage", r"current"),
        # Bounded by capacity at ingest time, not here — this only rejects a
        # dead or wildly mis-scaled channel.
        plausible_max=(1.0, 100_000.0),
        unit="kW",
        required=True,
    ),
    QuantitySpec(
        canonical="dc_power_kw",
        include=(r"^dc_power_inv", r"^dc_power__", r"^dc_power$"),
        exclude=(r"_other_", r"voltage", r"current", r"_1__", r"_2__"),
        plausible_max=(1.0, 100_000.0),
        unit="kW",
    ),
)

# Per-string / per-combiner DC channels. These are what make
# `per_mppt_current_balance` a measurement on real data rather than simulation.
_STRING_CURRENT_RE = re.compile(
    r"(shuntcurrent_a_avg|string\d*current|inv\d+_dc_current)"
)
_STRING_POWER_RE = re.compile(r"(shuntpdc_kw_avg|inv\d+_dc_power)")


@dataclass(frozen=True)
class ChannelChoice:
    """One resolved quantity and why that channel won."""

    canonical: str
    source_column: str
    observed_max: float
    unit: str
    rejected: tuple[tuple[str, str], ...] = ()


@dataclass
class ChannelResolution:
    """The full mapping for one system, plus an audit trail."""

    chosen: dict[str, ChannelChoice] = field(default_factory=dict)
    string_current_columns: tuple[str, ...] = ()
    string_power_columns: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()

    @property
    def rename_map(self) -> dict[str, str]:
        mapping = {c.source_column: name for name, c in self.chosen.items()}
        for index, column in enumerate(self.string_current_columns, start=1):
            mapping[column] = f"{STRING_CURRENT_PREFIX}{index}"
        for index, column in enumerate(self.string_power_columns, start=1):
            mapping[column] = f"{STRING_DC_POWER_PREFIX}{index}"
        return mapping

    def to_dict(self) -> dict[str, object]:
        return {
            "chosen": {
                name: {
                    "source_column": c.source_column,
                    "observed_max": round(c.observed_max, 3),
                    "unit": c.unit,
                    "rejected": [list(r) for r in c.rejected],
                }
                for name, c in self.chosen.items()
            },
            "string_current_columns": list(self.string_current_columns),
            "string_power_columns": list(self.string_power_columns),
            "unresolved": list(self.unresolved),
        }


def _natural_key(name: str) -> tuple[int, str]:
    """Sort ``..._2`` before ``..._10`` so string indices stay in order."""
    match = re.search(r"_(\d+)__?\d*$", name)
    return (int(match.group(1)) if match else 0, name)


def resolve_channels(
    frame: pd.DataFrame,
    specs: Sequence[QuantitySpec] = QUANTITY_SPECS,
) -> ChannelResolution:
    """Map a raw PVDAQ frame onto canonical names.

    Args:
        frame: A representative sample — ideally several clear days, so that a
            healthy channel actually reaches its daytime maximum. A sample of
            overcast days can push a good irradiance channel below its
            plausibility floor and get it rejected.

    Raises:
        ValueError: If a `required` quantity has no plausible channel. Failing
            here is deliberate; a missing POA channel silently zeroes every
            performance ratio downstream.
    """
    resolution = ChannelResolution()
    unresolved: list[str] = []

    for spec in specs:
        candidates = [
            column
            for column in frame.columns
            if any(re.search(pattern, column, re.I) for pattern in spec.include)
            and not any(re.search(pattern, column, re.I) for pattern in spec.exclude)
        ]
        rejected: list[tuple[str, str]] = []
        winner: tuple[str, float] | None = None

        for column in sorted(candidates):
            series = pd.to_numeric(frame[column], errors="coerce")
            if series.notna().sum() == 0:
                rejected.append((column, "all values missing"))
                continue
            observed = float(series.max())
            low, high = spec.plausible_max
            if observed < low:
                rejected.append(
                    (
                        column,
                        f"max {observed:.4g} below {low:g} {spec.unit} — "
                        "wrong units or a dead channel",
                    )
                )
                continue
            if observed > high:
                rejected.append(
                    (
                        column,
                        f"max {observed:.4g} above {high:g} {spec.unit} — "
                        "wrong units or a stuck sensor",
                    )
                )
                continue
            # Among plausible channels prefer the most complete one.
            if winner is None or series.notna().sum() > frame[winner[0]].notna().sum():
                if winner is not None:
                    rejected.append(
                        (winner[0], "a more complete channel was available")
                    )
                winner = (column, observed)

        if winner is None:
            if spec.required:
                detail = (
                    "; ".join(f"{c}: {why}" for c, why in rejected) or "no candidates"
                )
                raise ValueError(
                    f"could not resolve required quantity {spec.canonical!r} "
                    f"({spec.unit}). Considered — {detail}"
                )
            unresolved.append(spec.canonical)
            continue

        resolution.chosen[spec.canonical] = ChannelChoice(
            canonical=spec.canonical,
            source_column=winner[0],
            observed_max=winner[1],
            unit=spec.unit,
            rejected=tuple(rejected),
        )

    resolution.string_current_columns = tuple(
        sorted(
            (c for c in frame.columns if _STRING_CURRENT_RE.search(c.lower())),
            key=_natural_key,
        )
    )
    resolution.string_power_columns = tuple(
        sorted(
            (c for c in frame.columns if _STRING_POWER_RE.search(c.lower())),
            key=_natural_key,
        )
    )
    resolution.unresolved = tuple(unresolved)
    return resolution
