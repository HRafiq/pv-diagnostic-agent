# PV Diagnostic Agent

Root-cause analysis for solar PV plant performance deficits.

A large plant loses money in two directions at once, and conventional monitoring
shows neither. Real losses hide in normal-looking data — output tracks sunlight,
so a cloudy week and a failed string look identical in MWh, and a string that
fails in March can go unnoticed until an inspection in September. And false
alarms cost more than the faults — performance ratio falls plant-wide every
summer because hot modules are less efficient, and inverters hit their AC
ceiling every clear midday by design. Both look like problems. Crews get
dispatched, arrays get cleaned, and nothing was wrong.

This system normalises production against irradiance and temperature so a real
deficit becomes visible; investigates each deficit by taking measurements that
discriminate between competing causes; sorts every finding into **fault** /
**recoverable loss** / **by design** / **not the plant**; ranks by energy at
stake; shows its full reasoning; and when the data genuinely cannot separate two
causes, says so and names the cheap test that would.

The output is one of three instructions: **send someone, schedule something, or
do nothing.** The third is the one nobody sells and often the most valuable.

> **Status: steps 0–5 of 13 complete.** Real data ingested, physics validated,
> fault injector and evaluation harness running with a rules baseline scored,
> and the agent loop taking measurements end to end with its reasoning on
> screen. See *Build progress* below.

---

## Why it is built this way

**Atomic tools, not a classifier.** There are 18 measurement tools —
`compute_temp_corrected_pr`, `characterize_onset`, `per_mppt_current_balance`,
`check_clearsky_consistency` — and no tool that returns a fault class. The
agent's job is differential diagnosis: holding competing causes and choosing
which discriminating measurement to take next, the way a clinician orders the
next test. If the diagnosis happened inside one function, this would be a rules
engine with a chat wrapper, every run would follow the same path, and the agent
would only be narrating.

**Agency is measured, not asserted.** Every tool call records `was_planned`. The
fraction of runs that use a tool outside the initial plan is the sharpest single
indicator of whether the planner is reasoning or just sequencing — near zero
means it is a pipeline in disguise. Alongside it: distinct tool trajectories,
critic-iteration spread, and self-initiated abstentions.

**A rules engine is the baseline, not a component.** A deterministic rules engine
runs on the same golden set, and the comparison is published whichever way it
falls. If the rules engine wins outright, that is a more credible finding than
"I built an agent".

**The evaluation is designed to be able to fail.** Faults are injected at the
physics layer, never the signature layer, so the diagnostic logic has to
*rediscover* the signature. At least 40% of cases are things that look like
faults and are not. `not_enough_evidence` is a scored correct answer on
deliberately unresolvable cases. The held-back split is generated with
parameters not inspected while iterating.

**Nothing dishonest can reach the interface.** The rules that keep the output
truthful are validators, not conventions — a `Finding` that shows a single cause
and a confidence number when the investigation could not separate two causes
cannot be constructed. That is the exact bug that dispatches a wash crew to a
clean array.

**Every number is traceable to a measurement.** Each tool returns a provenance
ledger of everything it measured, and the agent's prose is checked against the
union of them. A figure that is not in a ledger is reported as ungrounded —
including one the agent derived by arithmetic, because arithmetic belongs in a
tool where it is deterministic and testable, not in a sentence.

---

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # ANTHROPIC_API_KEY needed only for agent nodes
```

```bash
pytest                                    # 416 tests
ruff check . && mypy src eval simulator   # lint + types

python watcher.py status                  # clock, models, config
python watcher.py run --until 2019-06-30 --step 1D

python -m src.data.cli ingest --system 4902 --years 2016 2017
python -m eval.runner build
python -m eval.runner run --engine rules            # no API key needed
python -m eval.runner run --engine agent --split tuning

uvicorn dashboard.app:app --reload        # http://127.0.0.1:8000
```

Physics, detectors, the simulator, the dashboard and the whole test suite run
without an API key. Only the agent nodes need one.

---

## Layout

```
src/          UI-agnostic core — everything the agent and the dashboard share
  clock.py      the ONLY time source (see below)
  determinism.py  seeding and run-scoped RNG
  config.py     typed loaders for config/*.yaml
  trace/        TraceStep + JSONL writer — the dashboard's only data source
  findings/     Finding model, lifecycle, energy ranking
  agent/        state, nodes, plain loop (step 4), LangGraph port (step 12)
  physics/ data/ detect/ tools/ rag/ knowledge/ baseline/ analyses/ viz/
simulator/    physics-level fault injector
eval/         runner, metrics, agency metrics, golden sets
dashboard/    FastAPI + Jinja2 + Alpine.js + Plotly.js — thin, calls src/ only
watcher.py    CLI: advances the clock, sweeps, writes findings
```

**Time is a replay clock.** There is no free real-time public PV feed, so "now"
is a position in a historical dataset. `datetime.now()` is banned throughout
`src/` and the ban is enforced by an AST scan in the test suite — a stray
wall-clock read would compare a replayed 2019 dataset against today, empty every
trailing window, and fail silently.

**The dashboard cannot diverge from the agent.** A chart showing PR calls
`physics.compute_pr()`, the same function the tool node calls. The dashboard
never reimplements a computation and never triggers a run; `watcher.py` writes,
the dashboard reads.

---

## Build progress

| Step | Deliverable | State |
| ---- | ----------- | ----- |
| 0 | Skeleton, config, schemas, injected clock, JSONL trace writer, `CLAUDE.md` | **done** |
| 1 | PVDAQ ingestion + expectation model. Dashboard Tab 1 live | **done** |
| 2 | Physics-level fault injector, 7 injectors, 20 golden cases | **done** |
| 3 | Evaluation harness — runner, metrics, agency metrics, rules baseline | **done** |
| 4 | Vertical slice: plain-Python loop, 9 measurement tools, Tab 3 | **done** |
| 5 | Full atomic tool set (18), domain knowledge, golden set to 86 | **done** |
| 6 | Critic with structured verdict, iteration cap, not-enough-evidence path | next |
| 7 | Rules-engine baseline + first rules-vs-agent comparison | |
| 8 | Watcher: sweep, findings store with lifecycle, energy ranking. Tab 2 | |
| 9 | RAG layer + retrieval golden set + ablation | |
| 10 | Saved analyses registry with golden-case enforcement | |
| 11 | Full evaluation, agency metrics, both experiments, `docs/FINDINGS.md` | |
| 12 | LangGraph port; verify identical golden-set outputs | |
| 13 | README with honest results, including whatever the ablations showed | |

Each step is gated: it stops for review before the next one starts.

---

## How an investigation runs

    plan  ->  route -> measure  (repeat)  ->  write the answer  ->  review
      ^                                                               |
      +---------------------- send it back ---------------------------+

The planner enumerates candidate causes — benign ones included, since a plan
containing only faults has already decided the answer — and opens a line of
enquiry. The router picks one measurement at a time and may depart from the plan
whenever a result points elsewhere. The synthesiser writes the finding and is
allowed no arithmetic. The reviewer arrives at step 6; until then the loop is a
single pass.

The whole loop runs against a scripted client with no network and no API key, so
its control flow, its caps, its abstention path and the tape it writes are all
regression-tested in CI. That matters because the loop is exactly the part where
a fault stays invisible until an evaluation run quietly produces wrong numbers.

The **Investigate** tab replays a run: the plan, each measurement with its
result, unplanned measurements flagged with the reason the agent gave, the
possible-causes ledger with what has been ruled out, and the answer — which
shows a single cause and a confidence only when the investigation reached one.
The dashboard never starts a run; `watcher.py` and the evaluation harness write
traces and the dashboard reads them.

---

## Data

Public data only. The primary dataset is **NREL PVDAQ system 4902,
NIST_Ground_1** (Gaithersburg MD) — 270.7 kW, 1152 Sharp NU-U235F2 modules on a
fixed 20° / due-south mount, 2016–2017 at 15-minute resolution.

    python -m src.data.cli ingest --system 4902 --years 2016 2017

It carries **7 per-combiner DC current channels**, which is unusual and load
bearing: `per_mppt_current_balance` and the string-vs-array comparisons run on
real measurements rather than simulation. Plus POA irradiance, module and
ambient temperature, wind speed, and both AC and DC power.

Datasets are never committed. The ingest downloads into a gitignored `data/raw/`
and writes a manifest with a SHA-256 per source file, the resolved channel map,
and the timezone finding — so the dataset is reproducible without shipping it.

**No fault logs exist.** PVDAQ ships telemetry and equipment metadata only —
there is no maintenance, outage or event table anywhere in the archive. Ground
truth therefore comes from physics-level injection onto the measured series,
which is why injecting at the physics layer rather than the signature layer
matters so much: it is the only truth the evaluation has. Cases that need no
label survive as real untouched data — summer temperature derating and a cloudy
week are verifiable from the temperature coefficient and the irradiance record
alone.

Three things the ingest does that are easy to get wrong:

- **Channel resolution is by physical plausibility, not name.** This system
  exposes three POA channels and only one reads in W/m²; the others are ~115×
  smaller. Picking on name alone makes every performance ratio wrong by that
  factor while still looking plausible. It also rejected a wind channel reading
  292 m/s.
- **The logger's timezone is recovered, not assumed.** Every half-hour offset is
  scored against a modelled clear-sky curve. NIST 4902 came back UTC−5 at
  r = 0.997 — Eastern *Standard* Time year round, no daylight saving.
- **Faults are injected onto measured series, never onto a simulated plant.**
  There is no generating model for the expectation model to be circular with.

---

## Results

Nothing to report yet. When there is, it goes in `docs/FINDINGS.md` and here —
including whichever way the rules-vs-agent comparison and the retrieval ablation
fall. If retrieval turns out not to move accuracy, that is a finding and it gets
published.
