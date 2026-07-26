"""Physics-level fault injection onto *measured* series.

Two rules govern everything here, and both exist to keep the evaluation
meaningful rather than self-congratulatory.

**1. Inject at the physics layer, never the signature layer.** A string fault
zeroes a fraction of DC current on one combiner with a realistic transition and
noise; it never "reduces PR by 0.12". The diagnostic logic has to *rediscover*
the signature. Injecting the signature directly measures nothing but whether a
rule matches its own generator.

**2. Perturb measured data; never simulate the plant.** The base series are real
NIST measurements. Nothing here builds a synthetic plant with pvlib, so the
expectation model in `physics/modelchain.py` shares no configuration with the
truth — there is no model to be circular with. This is stronger than merely
using a different model, and it only became possible because the chosen dataset
carries real per-string DC currents.

Every injector is deterministic given a seed, and returns both the modified
frame and a record of exactly what it did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.determinism import run_rng

__all__ = [
    "INJECTORS",
    "InjectionRecord",
    "inject_inverter_clipping",
    "inject_sensor_drift",
    "inject_shading",
    "inject_soiling",
    "inject_string_outage",
    "inject_telemetry_gap",
]


@dataclass(frozen=True)
class InjectionRecord:
    """What was done to the data — the ground truth for one golden case."""

    kind: str
    category: str
    params: dict[str, Any]
    start: str
    end: str
    affected_columns: tuple[str, ...]
    energy_lost_kwh: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "category": self.category,
            "params": self.params,
            "start": self.start,
            "end": self.end,
            "affected_columns": list(self.affected_columns),
            "energy_lost_kwh": round(self.energy_lost_kwh, 2),
            "note": self.note,
        }


def _interval_hours(frame: pd.DataFrame) -> float:
    deltas = pd.Series(pd.DatetimeIndex(frame.index)).diff().dropna()
    return float(deltas.mode().iloc[0].total_seconds()) / 3600.0


def _energy_delta(before: pd.Series, after: pd.Series, hours: float) -> float:
    return float((before.fillna(0) - after.fillna(0)).sum() * hours)


def _window_mask(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    index = pd.DatetimeIndex(frame.index)
    return pd.Series(
        (index >= pd.Timestamp(start, tz="UTC"))
        & (index <= pd.Timestamp(end, tz="UTC")),
        index=frame.index,
    )


def _numeric_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    """A numeric view of a column, or an all-NaN series if it is absent."""
    if column not in frame:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _string_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        (c for c in frame.columns if c.startswith("string_current_a_")),
        key=lambda c: int(c.rsplit("_", 1)[-1]),
    )


def _string_power_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        (c for c in frame.columns if c.startswith("string_dc_power_kw_")),
        key=lambda c: int(c.rsplit("_", 1)[-1]),
    )


def _scale_power(
    frame: pd.DataFrame,
    mask: pd.Series,
    factor: pd.Series,
    scale_strings: bool = False,
) -> tuple[pd.DataFrame, float]:
    """Apply a multiplicative loss to DC and AC power over a window.

    Args:
        scale_strings: Whether every per-string channel moves with the loss.
            This is a physical question, not a convenience: a loss upstream of
            the array (dust on the glass) or at the inverter (a power limit
            backing the array off its maximum power point) reduces the current
            on *every* string, while a blown fuse or a shadow reduces it on
            some. Leaving the string channels alone for a whole-plant loss
            implies an inverter efficiency that cannot exist, and an agent that
            noticed would be reading a bug in the injector rather than physics.
    """
    out = frame.copy()
    hours = _interval_hours(frame)
    lost = 0.0
    columns = ["dc_power_kw", "ac_power_kw"]
    if scale_strings:
        columns += _string_columns(out) + _string_power_columns(out)
    for column in columns:
        if column not in out:
            continue
        before = pd.to_numeric(out[column], errors="coerce")
        after = before.copy()
        after[mask] = before[mask] * factor[mask]
        if column == "ac_power_kw":
            lost = _energy_delta(before, after, hours)
        out[column] = after
    return out, lost


# ---------------------------------------------------------------------------
# Faults — real equipment failures. "Send someone."
# ---------------------------------------------------------------------------
def inject_string_outage(
    frame: pd.DataFrame,
    start: str,
    end: str,
    string_index: int = 1,
    fraction_lost: float = 1.0,
    transition_minutes: float = 0.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, InjectionRecord]:
    """One combiner loses a fraction of its current.

    The physical event is a blown fuse or a disconnected string: current on that
    combiner drops, the others are untouched, and total DC power falls by that
    combiner's share. The *signature* — a step in one channel's share of the
    total while irradiance is unchanged — is a consequence, not something
    written in here.

    A `transition_minutes` above zero ramps the loss instead of stepping it,
    which is what a corroding connection looks like as against a blown fuse.
    """
    out = frame.copy()
    columns = _string_columns(out)
    if not columns:
        raise ValueError("no per-string current channels in this dataset")
    if not 1 <= string_index <= len(columns):
        raise ValueError(f"string_index must be 1..{len(columns)}; got {string_index}")

    target = columns[string_index - 1]
    mask = _window_mask(out, start, end)
    if not mask.any():
        raise ValueError(f"window {start}..{end} selects no rows")

    # Loss profile: step, or a linear ramp over the transition.
    profile = pd.Series(0.0, index=out.index)
    profile[mask] = fraction_lost
    if transition_minutes > 0:
        interval_min = _interval_hours(out) * 60.0
        steps = max(1, int(transition_minutes / interval_min))
        ramp = np.linspace(0.0, fraction_lost, steps)
        positions = np.flatnonzero(mask.to_numpy())[:steps].tolist()
        for offset, position in enumerate(positions):
            profile.iloc[position] = float(ramp[offset])

    before_current = pd.to_numeric(out[target], errors="coerce")
    out[target] = before_current * (1.0 - profile)

    # The array's share of total DC power carried by this combiner.
    totals = out[columns].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    share = float((before_current[mask] / totals[mask].replace(0, np.nan)).median())
    share = share if np.isfinite(share) else 1.0 / len(columns)

    out, lost = _scale_power(out, mask, 1.0 - profile * share)
    if f"string_dc_power_kw_{string_index}" in out:
        col = f"string_dc_power_kw_{string_index}"
        out[col] = pd.to_numeric(out[col], errors="coerce") * (1.0 - profile)

    _ = run_rng(seed)  # reserved: measurement noise on the affected channel
    return out, InjectionRecord(
        kind="string_outage",
        category="fault",
        params={
            "string_index": string_index,
            "fraction_lost": fraction_lost,
            "transition_minutes": transition_minutes,
            "share_of_array": round(share, 4),
        },
        start=start,
        end=end,
        affected_columns=(target, "dc_power_kw", "ac_power_kw"),
        energy_lost_kwh=lost,
        note=(
            f"string {string_index} lost {fraction_lost:.0%} of its current; "
            "other strings untouched, irradiance unchanged"
        ),
    )


def inject_shading(
    frame: pd.DataFrame,
    start: str,
    end: str,
    hour_start: int = 7,
    hour_end: int = 10,
    depth: float = 0.35,
    string_index: int = 1,
    strings_affected: int = 1,
    seed: int = 0,
) -> tuple[pd.DataFrame, InjectionRecord]:
    """A new obstruction shades part of the array at a fixed time of day.

    The discriminator against a string fault is *time-of-day dependence*: a
    blown fuse costs the same fraction all day, while a shadow tracks the sun
    and disappears by mid-morning.

    `strings_affected` matters more than it looks. A shadow is a physical object
    with an extent — a new building, a tree line, a row of modules in front —
    and it falls across a contiguous group of strings, not a single one. Shading
    one string of seven for three hours costs about 1.5% of the window's energy,
    which is under the noise of every measurement in the tool set; the case
    would be unlearnable and the evaluation would be scoring coin flips.
    """
    out = frame.copy()
    columns = _string_columns(out)
    mask = _window_mask(out, start, end)
    index = pd.DatetimeIndex(out.index)
    hours = pd.Series(index.hour, index=out.index)
    shaded = mask & (hours >= hour_start) & (hours < hour_end)
    if not shaded.any():
        raise ValueError("shading window selects no rows")

    profile = pd.Series(0.0, index=out.index)
    profile[shaded] = depth
    if columns:
        count = max(1, min(strings_affected, len(columns)))
        first = (string_index - 1) % len(columns)
        targets = [columns[(first + n) % len(columns)] for n in range(count)]
        for target in targets:
            out[target] = pd.to_numeric(out[target], errors="coerce") * (1.0 - profile)
            power_column = f"string_dc_power_kw_{target.rsplit('_', 1)[-1]}"
            if power_column in out:
                out[power_column] = pd.to_numeric(
                    out[power_column], errors="coerce"
                ) * (1.0 - profile)
        share = count / len(columns)
    else:
        targets = []
        share = 1.0

    out, lost = _scale_power(out, shaded, 1.0 - profile * share)
    _ = run_rng(seed)
    return out, InjectionRecord(
        kind="shading",
        category="fault",
        params={
            "hour_start": hour_start,
            "hour_end": hour_end,
            "depth": depth,
            "string_index": string_index,
            "strings_affected": len(targets),
            "share_of_array": round(share, 4),
        },
        start=start,
        end=end,
        affected_columns=(*targets, "dc_power_kw", "ac_power_kw"),
        energy_lost_kwh=lost,
        note=(
            f"obstruction shades {len(targets)} of {len(columns)} strings "
            f"between {hour_start:02d}:00 and {hour_end:02d}:00 — the loss "
            "tracks time of day, unlike a fuse"
        ),
    )


# ---------------------------------------------------------------------------
# Recoverable losses — real, but maintenance gets them back. "Schedule it."
# ---------------------------------------------------------------------------
def inject_soiling(
    frame: pd.DataFrame,
    start: str,
    end: str,
    rate_per_day: float = 0.003,
    max_loss: float = 0.25,
    rain_resets: bool = True,
    rain_threshold_wm2: float = 0.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, InjectionRecord]:
    """Dust accumulates on the modules, cutting optical transmission.

    Modelled as what it physically is — a transmission loss that grows with
    time and resets when it rains — not as "a gradual PR decline". The sawtooth
    the detector must find is emergent.

    With no rainfall channel in the dataset, a rain day is inferred from a day
    whose insolation is far below its neighbours. That is a proxy, and it is
    recorded in the params so nobody mistakes it for a measurement.
    """
    out = frame.copy()
    mask = _window_mask(out, start, end)
    if not mask.any():
        raise ValueError(f"window {start}..{end} selects no rows")

    index = pd.DatetimeIndex(out.index)
    days = pd.Series(index.normalize(), index=out.index)
    daily_poa = pd.to_numeric(out["poa_wm2"], errors="coerce").groupby(days).mean()
    wet = daily_poa < (rain_threshold_wm2 or 0.35 * float(daily_poa.median()))

    loss_by_day: dict[pd.Timestamp, float] = {}
    accumulated = 0.0
    window_days = sorted(days[mask].unique())
    for day in window_days:
        if rain_resets and bool(wet.get(day, False)):
            accumulated = 0.0
        else:
            accumulated = min(accumulated + rate_per_day, max_loss)
        loss_by_day[day] = accumulated

    profile = days.map(loss_by_day).fillna(0.0).astype(float)
    # Dust sits on the glass, upstream of everything electrical, so it costs
    # every string the same fraction of its current. That uniformity is the
    # signature that separates soiling from a string fault — and it only exists
    # in the data if the injection actually applies it.
    out, lost = _scale_power(out, mask, 1.0 - profile, scale_strings=True)
    _ = run_rng(seed)
    return out, InjectionRecord(
        kind="soiling",
        category="recoverable",
        params={
            "rate_per_day": rate_per_day,
            "max_loss": max_loss,
            "rain_resets": rain_resets,
            "rain_days_detected": int(wet.loc[window_days].sum()),
            "rain_proxy": "low-insolation day (no rainfall channel in dataset)",
        },
        start=start,
        end=end,
        affected_columns=("dc_power_kw", "ac_power_kw", "every string channel"),
        energy_lost_kwh=lost,
        note=(
            f"transmission loss accumulating at {rate_per_day:.2%}/day, "
            f"capped at {max_loss:.0%}"
            + (", reset by rain" if rain_resets else ", never reset")
        ),
    )


# ---------------------------------------------------------------------------
# By design — not a fault at all. "Do nothing."
# ---------------------------------------------------------------------------
def inject_inverter_clipping(
    frame: pd.DataFrame,
    start: str,
    end: str,
    ac_ceiling_kw: float,
    seed: int = 0,
) -> tuple[pd.DataFrame, InjectionRecord]:
    """Cap AC output at the inverter's rating.

    This is the plant working as designed, and it is the most expensive false
    alarm in the business: midday output goes flat, energy is "lost" against a
    naive expectation, and a crew gets dispatched to a healthy inverter.

    Note it is deliberately indistinguishable, on the power channel alone, from
    `inject_curtailment` below. Separating them needs evidence from somewhere
    else entirely — which is the whole point of the pair. Both apply the *same*
    cap through the same code path for exactly that reason: if one of them left
    a channel untouched that the other moved, the pair would be separable by a
    quirk of the simulator rather than by physics, and the case would be
    measuring the injector.
    """
    return _apply_power_cap(
        frame,
        start,
        end,
        ac_ceiling_kw,
        kind="clipping",
        category="by_design",
        params={"ac_ceiling_kw": ac_ceiling_kw},
        note=(
            f"AC output capped at the {ac_ceiling_kw:.0f} kW inverter rating — "
            "by design, not a fault"
        ),
        seed=seed,
    )


def _apply_power_cap(
    frame: pd.DataFrame,
    start: str,
    end: str,
    ceiling_kw: float,
    kind: str,
    category: str,
    params: dict[str, Any],
    note: str,
    seed: int,
) -> tuple[pd.DataFrame, InjectionRecord]:
    """Hold AC output at a ceiling, and back the array off to match.

    An inverter that cannot export more power does not keep drawing it: it
    walks the array up its I-V curve away from the maximum power point, so DC
    power and every string current fall with the AC output. Capping AC alone
    would leave the frame implying an inverter running at 60% efficiency all
    midday — a fingerprint of the simulator, not of a power limit, and one an
    agent could learn instead of the physics.
    """
    out = frame.copy()
    mask = _window_mask(out, start, end)
    before = pd.to_numeric(out["ac_power_kw"], errors="coerce")

    factor = pd.Series(1.0, index=out.index)
    capped = mask & (before > ceiling_kw)
    factor[capped] = (ceiling_kw / before[capped]).astype(float)

    out, lost = _scale_power(out, mask, factor, scale_strings=True)
    _ = run_rng(seed)
    return out, InjectionRecord(
        kind=kind,
        category=category,
        params={**params, "intervals_capped": int(capped.sum())},
        start=start,
        end=end,
        affected_columns=("ac_power_kw", "dc_power_kw", "every string channel"),
        energy_lost_kwh=lost,
        note=note,
    )


def inject_curtailment(
    frame: pd.DataFrame,
    start: str,
    end: str,
    ceiling_kw: float,
    seed: int = 0,
) -> tuple[pd.DataFrame, InjectionRecord]:
    """The grid operator caps export below the inverter rating.

    Produces the *same flat ceiling* as clipping. The only difference is the
    level and the fact that it starts and stops on command rather than with the
    sun — so the agent cannot tell them apart from the power channel alone, and
    must reach for the comparison against the same period last year.
    """
    return _apply_power_cap(
        frame,
        start,
        end,
        ceiling_kw,
        kind="curtailment",
        category="not_the_plant",
        params={"ceiling_kw": ceiling_kw},
        note=(
            f"export capped at {ceiling_kw:.0f} kW by the grid operator — "
            "identical ceiling to clipping, different cause and different action"
        ),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Not the plant — the instrument or the data is wrong. "Do nothing (to the plant)."
# ---------------------------------------------------------------------------
def inject_sensor_drift(
    frame: pd.DataFrame,
    start: str,
    end: str,
    drift_per_day: float = -0.002,
    max_drift: float = -0.15,
    seed: int = 0,
) -> tuple[pd.DataFrame, InjectionRecord]:
    """The irradiance sensor reads progressively low. The plant is fine.

    This is the case the whole project is organised around, and its signature is
    the opposite of what intuition suggests. A soiled *array* drops power with
    irradiance unchanged, so PR falls. A soiled *sensor* drops the irradiance
    reading with power unchanged, so PR **rises** — the plant appears to improve.

    They are separable, and `check_clearsky_consistency` is what separates them:
    a drifting sensor diverges from the clear-sky envelope while the array
    agrees with it.
    """
    out = frame.copy()
    mask = _window_mask(out, start, end)
    if not mask.any():
        raise ValueError(f"window {start}..{end} selects no rows")

    index = pd.DatetimeIndex(out.index)
    days = pd.Series(index.normalize(), index=out.index)
    window_days = sorted(days[mask].unique())
    drift_by_day = {
        day: max(min(drift_per_day * i, 0.0), max_drift)
        for i, day in enumerate(window_days)
    }
    profile = days.map(drift_by_day).fillna(0.0).astype(float)

    before = pd.to_numeric(out["poa_wm2"], errors="coerce")
    out["poa_wm2"] = before * (1.0 + profile)
    _ = run_rng(seed)
    return out, InjectionRecord(
        kind="sensor_drift",
        category="not_the_plant",
        params={"drift_per_day": drift_per_day, "max_drift": max_drift},
        start=start,
        end=end,
        affected_columns=("poa_wm2",),
        # No energy is lost: the plant produced exactly what it always did.
        energy_lost_kwh=0.0,
        note=(
            "irradiance sensor reads low while output is untouched, so the "
            "performance ratio RISES — no energy was actually lost"
        ),
    )


def inject_telemetry_gap(
    frame: pd.DataFrame,
    start: str,
    end: str,
    columns: tuple[str, ...] = ("ac_power_kw", "dc_power_kw"),
    seed: int = 0,
) -> tuple[pd.DataFrame, InjectionRecord]:
    """The logger stops recording power while the weather station runs on.

    Modelled on the real 35-day gap in NIST system 4902 in mid-2016. Scored
    without a completeness check, that month reads as a 90% loss that never
    happened.
    """
    out = frame.copy()
    mask = _window_mask(out, start, end)
    hours = _interval_hours(out)
    before = _numeric_or_nan(out, "ac_power_kw")
    for column in columns:
        if column in out:
            out.loc[mask, column] = np.nan
    after = _numeric_or_nan(out, "ac_power_kw")
    _ = run_rng(seed)
    return out, InjectionRecord(
        kind="telemetry_gap",
        category="not_the_plant",
        params={"columns": list(columns)},
        start=start,
        end=end,
        affected_columns=columns,
        # Apparent, not real: the energy is missing from the record, not the grid.
        energy_lost_kwh=_energy_delta(before, after, hours),
        note=(
            "power channel stopped recording while irradiance kept logging — "
            "an apparent loss that is a data problem, not a plant problem"
        ),
    )


INJECTORS: dict[str, Any] = {
    "string_outage": inject_string_outage,
    "shading": inject_shading,
    "soiling": inject_soiling,
    "clipping": inject_inverter_clipping,
    "curtailment": inject_curtailment,
    "sensor_drift": inject_sensor_drift,
    "telemetry_gap": inject_telemetry_gap,
}
