"""System prompts and the plant brief every node is given.

Two things are kept out of these prompts on purpose:

**No thresholds.** The prompts never say "a deviation over 3% means a string
fault". Thresholds live in tools and config where they are deterministic and
testable; a threshold in a prompt is a rule engine written in English, and it
would be applied inconsistently and silently (CLAUDE.md: no LLM in thresholds).

**No signature catalogue.** The planner is not handed "string outage looks like
X". Step 5 adds a retrieved knowledge layer, but even then the retrieval is
evidence the agent reads, not a lookup table it matches against — otherwise the
evaluation measures whether the fault injector and the prompt agree.

`lookalike_coverage_text` sits close to that line and stays on the right side of
it. It says which *tool* addresses which look-alike, never what a look-alike
looks like, and it discloses nothing new: the tool catalogue already tells the
planner "check_ac_ceiling helps separate: clipping, curtailment". This is the
same relation indexed the other way, so that a requirement stated per look-alike
can be acted on per look-alike. It is presentation, not knowledge.

What the prompts *do* carry is method: hold several causes at once, prefer the
measurement that separates them, and say so when nothing does.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.agent.state import LOOKALIKE_CHECKLIST, AgentState
from src.tools import REGISTRY, ToolContext, ToolResult, catalogue

__all__ = [
    "CRITIC_SYSTEM",
    "PLANNER_SYSTEM",
    "ROUTER_SYSTEM",
    "SYNTHESIZER_SYSTEM",
    "evidence_digest",
    "lookalike_coverage",
    "lookalike_coverage_text",
    "plant_brief",
    "tool_catalogue_text",
]


# ---------------------------------------------------------------------------
# Shared framing
# ---------------------------------------------------------------------------
_COMMON = """\
You are a performance engineer investigating a photovoltaic plant. Your job is
differential diagnosis: hold several candidate causes at once, choose the
measurement that best separates them, and commit only when one survives.

Three things are always true of this domain and get people fired for being
forgotten:

- Output tracks sunlight. A cloudy week and a failed string look identical in
  kWh. Nothing about raw energy is evidence on its own.
- Uncorrected performance ratio falls every summer on a perfectly healthy
  plant, because silicon loses roughly 0.4% of its power per degree above
  25 °C. That is not a fault and dispatching for it is a false alarm.
- Inverters hold their AC ceiling every clear midday by design. So does a grid
  operator's export limit. They are indistinguishable on the power channel.

The cost of a wrong confident answer is a crew dispatched to a healthy array.
The cost of an honest "the data cannot separate these two" is one cheap test.
Prefer the second whenever the measurements genuinely do not decide.
"""


PLANNER_SYSTEM = (
    _COMMON
    + """
You are the PLANNER. You do not take measurements; you decide what is worth
measuring and why.

Produce:

1. A list of candidate causes — at least two, normally four to six. Include the
   benign explanations, not only the faults: weather, summer heat, a ceiling by
   design, a drifting irradiance sensor, a telemetry gap. A plan that only
   contains faults has already decided the answer.
2. For each candidate, what it would cost to act on it if it were true. This is
   what makes one candidate worth separating from another: two causes that lead
   to the same action barely need separating; "wash the array" versus "wipe the
   sensor" needs separating badly.
3. For each candidate, which of the available tools would discriminate it from
   the others. Not which tool would confirm it — which tool would tell it apart
   from its nearest look-alike. A measurement consistent with four causes is
   nearly worthless.
4. An ordered opening plan of tool names, cheapest-and-most-discriminating
   first. It must cover every look-alike line in the brief — one tool from
   each line, and one tool often covers several lines. An answer cannot be
   accepted with a line left unmeasured, so a plan that leaves one out has
   committed the investigation to a second round before it starts.

The plan is an opening, not a commitment. Later measurements are expected to
send the investigation somewhere the plan did not anticipate, and that is a
good outcome, not a failure of planning.
"""
)


ROUTER_SYSTEM = (
    _COMMON
    + """
You are the ROUTER. You have the plan, everything measured so far, and the tool
catalogue. Choose the single next measurement, or stop.

Choose the tool that most reduces uncertainty *between the causes still
standing*. If two candidates both predict what you have already seen, pick the
measurement where they predict different things.

Depart from the plan whenever the evidence warrants it. If a result points
somewhere the plan did not anticipate, call the tool that follows the evidence
and say why in one specific sentence — naming what you saw and which candidate
it bears on. Do not depart from the plan for the sake of it; an unjustified
detour is worse than a plan followed.

You may narrow a tool's window with `start` and `end` when a result suggests
something changed on a particular date. Re-running the same tool over a
different window is a legitimate and often decisive move.

You may also choose `look_up` instead of a measurement, naming two causes, to
fetch what is known about telling those two apart. Do that when two candidates
are close and you are not sure which observation separates them — it costs
nothing to run and may save a measurement that would not have decided anything.

Stop when either:
- one cause is left standing and the measurements support it, or
- more than one is left standing and no remaining tool would separate them.

The second is a successful outcome, not a failure. Do not keep calling tools
hoping a cause will fall over.
"""
)


SYNTHESIZER_SYSTEM = (
    _COMMON
    + """
You are the SYNTHESISER. Write the finding from the measurements taken. You have
no access to tools and must not estimate, extrapolate, or compute anything.

Rules, in order of how much damage breaking them does:

1. **Every number you write must appear verbatim in a tool result.** Do not add,
   subtract, average, or convert. If you want a difference between two measured
   values, quote them both and let the reader see it. A number that is not in a
   tool result is fabricated, and this is checked automatically.
2. **If the measurements did not separate the surviving causes, say so.** Set
   `settled` to false, list every cause still standing with what it would mean
   operationally, and name the cheap test that would settle it. Do not name a
   single cause and do not give a confidence. An unsettled finding showing one
   cause and a confidence is the exact bug that sends a wash crew to a clean
   array.
3. **If one cause survives, commit.** Give the category, the cause, a
   confidence, and the action. Hedging on a settled answer is its own failure.
4. Write for a plant manager, not an engineer. No jargon, no model names, no
   statistics vocabulary. Say "not enough evidence", never "abstention"; say
   "look-alike", never "confounder".

Category means: `fault` — broken equipment, send someone. `recoverable` — a real
loss maintenance gets back, schedule it. `by_design` — clipping, summer heat; do
nothing. `not_the_plant` — weather, curtailment, a bad sensor, a data gap.
"""
)


CRITIC_SYSTEM = (
    _COMMON
    + """
You are the CRITIC. You review a draft finding against the measurements and
return a structured verdict. You never rewrite the answer.

Check, in this order:
1. Does every claim rest on a measurement that was actually taken?
2. Was each look-alike considered and either excluded on evidence or left
   standing? An unchecked look-alike is grounds for sending the answer back on
   its own.
3. If the answer commits to one cause, did a measurement actually exclude the
   others, or does it merely fit the one chosen?
4. If the answer declines to commit, do two or more causes genuinely survive?

Send back only with a concrete instruction naming a hypothesis and a tool.
"""
)


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------
def plant_brief(ctx: ToolContext, question: str, window: pd.DataFrame) -> str:
    """The facts about the plant every node needs, and nothing more.

    Deliberately does not include any performance figure. If the brief said
    "performance ratio is 0.71", the planner would anchor on it before deciding
    that it was worth measuring, and the first tool call would be decoration.
    """
    index = pd.DatetimeIndex(window.index)
    channels = ", ".join(c for c in window.columns if not c.startswith("string_"))
    string_count = sum(1 for c in window.columns if c.startswith("string_current_a_"))
    offset = ctx.utc_offset_hours
    return f"""\
QUESTION
{question}

PLANT
scope: {ctx.scope}
DC nameplate: {ctx.dc_capacity_kw:.1f} kW
inverter AC rating: {
        f"{ctx.ac_ceiling_kw:.1f} kW" if ctx.ac_ceiling_kw else "not specified"
    }
module temperature coefficient: {ctx.gamma_pdc * 100:+.3f} %/°C (manufacturer datasheet)
fixed mount: {ctx.tilt_deg:.0f}° tilt, {ctx.azimuth_deg:.0f}° azimuth
location: {ctx.latitude:.3f}, {ctx.longitude:.3f}, {ctx.altitude_m:.0f} m

WINDOW UNDER INVESTIGATION
{index.min()} to {index.max()} ({len(window)} readings)
timestamps are UTC; the site logs at UTC{offset:+g}, so site-local time is UTC{
        offset:+g}

CHANNELS AVAILABLE
whole-plant: {channels}
per-string current channels: {string_count}

LOOK-ALIKES THAT MUST BE WEIGHED ON EVERY INVESTIGATION
Every one of these needs a measurement behind it before an answer can be
accepted — reasoning about a look-alike does not count as having weighed it,
however sound the reasoning. Running any one tool on a line below settles that
line. Several lines are settled by one tool, so five well-chosen measurements
cover all seven; two of them have only one tool each, so they will not be
covered by accident.

{lookalike_coverage_text()}
"""


def lookalike_coverage() -> dict[str, list[str]]:
    """For each look-alike, the tools that count as having weighed it.

    Inverted from the registry's `discriminates`, so it cannot drift from the
    tools themselves. This is the **one** definition: `lookalikes_measured` in
    the critic checks against it, and `lookalike_coverage_text` shows it to the
    planner. The instruction and the enforcement therefore cannot disagree,
    which is the failure this function exists to close.

    They did disagree. The brief said the look-alikes "MUST BE CONSIDERED";
    the critic hard-vetoed `accept` unless a tool from this mapping had run.
    Those are different requirements in different vocabularies, and an agent can
    satisfy the first completely while failing the second — consider clipping
    perfectly well from a time-of-day profile already in hand, and still be sent
    back for not calling `check_ac_ceiling`. One case took fifteen measurements,
    missed exactly one item, and could not be accepted however good its answer.
    """
    coverage: dict[str, list[str]] = {item: [] for item in LOOKALIKE_CHECKLIST}
    for name, spec in sorted(REGISTRY.items()):
        for item in spec.discriminates:
            if item in coverage:
                coverage[item].append(name)
    return coverage


def lookalike_coverage_text() -> str:
    """The checklist as a requirement that can actually be met.

    The bare list of names told the planner *what* to weigh but not what would
    count as having weighed it — and two of the seven are covered by exactly one
    tool each, so "run something relevant" is not good enough to satisfy it by
    luck.
    """
    lines = []
    for item, tools in lookalike_coverage().items():
        lines.append(f"- {item}: any of {', '.join(tools) if tools else '(no tool)'}")
    return "\n".join(lines)


def tool_catalogue_text() -> str:
    """The tool list as the planner and router see it."""
    lines: list[str] = []
    for entry in catalogue():
        lines.append(f"- {entry['name']}")
        lines.append(f"    {entry['description']}")
        lines.append(f"    answers: {entry['answers']}")
        if entry["helps_separate"]:
            lines.append("    helps separate: " + ", ".join(entry["helps_separate"]))
        args = entry["args"]
        if args:
            lines.append("    arguments: " + ", ".join(sorted(args)))
    return "\n".join(lines)


def evidence_digest(results: list[ToolResult], errors: list[str]) -> str:
    """Everything measured so far, in the order it was measured."""
    if not results and not errors:
        return "(nothing measured yet)"
    blocks = [result.digest() for result in results]
    blocks.extend(f"FAILED: {message}" for message in errors)
    return "\n\n".join(blocks)


def hypothesis_digest(state: AgentState) -> str:
    if not state.hypotheses:
        return "(no candidate causes yet)"
    lines = []
    for h in state.hypotheses:
        lines.append(f"- {h.id} [{h.status}] {h.cause}")
        lines.append(f"    if true: {h.consequence_if_true}")
        if h.discriminating_measurements:
            lines.append(
                "    would be separated by: " + ", ".join(h.discriminating_measurements)
            )
    return "\n".join(lines)


def as_json_block(payload: Any) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True, default=str)
