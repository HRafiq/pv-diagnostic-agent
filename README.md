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

> **Status: 12 of 13 steps complete. The agent has not been scored.**
>
> No `ANTHROPIC_API_KEY` was available in the environment this was built in, so
> every number published here comes from the physics, the rules baseline, the
> fault injector or the retrieval index — all of which run without one. The
> evaluation harness, the golden set, the metrics, both ablations and the
> rules-vs-agent comparison are complete and tested; **the agent column is empty
> because it has not been run, not because it is pending analysis.** One command
> fills it in. See *Results*.

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
runs on the same golden set *and calls the same eighteen tools*, so a gap between
the two is a gap in reasoning rather than in measurement. The comparison is
published whichever way it falls, its headline is the false-alarm rate rather
than overall accuracy, and its differences are signed so the table stays readable
when the agent loses. If the rules engine wins outright, that is a more credible
finding than "I built an agent".

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

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and resolved in
a committed `uv.lock`, so every machine and CI install the same versions.

```bash
uv sync --extra dev           # creates .venv and installs from the lockfile
cp .env.example .env          # ANTHROPIC_API_KEY needed only for agent nodes
```

```bash
uv run pytest                                    # 577 tests
uv run ruff check . && uv run mypy src eval simulator dashboard
```

`uv run` uses the project environment without needing it activated. If you
prefer to activate it, `source .venv/bin/activate` and drop the `uv run`
prefix from every command below.

> Adding or changing a dependency means editing `pyproject.toml` and running
> `uv lock`; commit the updated `uv.lock` with it. CI runs `uv lock --check`
> and fails if the two have drifted.

**Everything below runs without an API key**, including the whole test suite:

```bash
python -m src.data.cli ingest --system 4902 --years 2016 2017
python -m eval.runner build                        # 86 golden cases
python -m eval.runner run --engine rules --split both
python -m eval.runner retrieval                    # retrieval ablation

python watcher.py run --until 2016-09-01 --step 14D --engine rules
python watcher.py findings                         # finds the real 2016 gap

uvicorn dashboard.app:app --reload                 # http://127.0.0.1:8000
```

Two of them read runs already on disk, which is where most of the debugging
happens — `profile` says where the time and money went, `explain` prints one
case's whole reasoning rather than the 96-character clip the live display shows:

```bash
python -m eval.runner profile                      # seconds and $ per node
python -m eval.runner explain G-004                # one case, in full
python -m eval.runner explain G-004 --attempt 2    # an earlier run of it
```

**These need a key:**

```bash
cp .env.example .env                               # add ANTHROPIC_API_KEY
python -m eval.runner run --engine agent --split tuning
python -m eval.runner compare --split heldback     # rules vs agent
python -m eval.runner experiments --runs 3         # both ablations, mean ± spread
```

**Review has three modes**, because the critic node does two separable things:
deterministic checks that need no model — no fabricated numerics, every
look-alike weighed, no abstention naming a tool the agent owns and did not run —
and an LLM judgement. Only the second is slow, and only the second has a record
of talking correct answers out of themselves.

```bash
python -m eval.runner run --engine agent --split tuning                  # both
python -m eval.runner run --engine agent --split tuning --checks-only    # checks
python -m eval.runner run --engine agent --split tuning --no-review      # neither
```

`--no-review` is the control the ablation is measured against, so it stays a
clean single pass. `--checks-only` is the configuration the evidence so far
points at shipping.

**A long run should survive dying.** `--resume FILE` records each case as it
finishes and skips those already there, so an API overload at case 20 costs one
case rather than the run. Use a fresh file when the code has changed under it —
the journal has no idea it did.

```bash
python -m eval.runner run --engine agent --split tuning \
    --checks-only --resume runs/tuning.jsonl --out runs/tuning.json
```

**Iterating? Run a subset.** `--sample 8` draws eight cases stratified by
category and settledness, so a small run still contains look-alikes and
unresolvable cases and every headline metric stays computable. `--only
G-001,G-007` runs exactly two. `--limit 5` takes the first five, which the file
orders by fault type — so it is not a cross-section, and the harness says so.

Any subset prints, above the results, which metrics it cannot support:

```
SUBSET: 5 of 43 cases. This is a debugging run, not a score — do not quote these numbers.
  - no unresolvable cases, so correct 'not enough evidence' cannot be measured at all
```

### What a run actually costs, measured

The design target was $0.15 a question. **It is not met.** Measured over the
first real runs, a case costs **$0.65–$1.10** and takes **7–12 minutes**, so a
full `compare --split both` — 86 investigations, three runs for a headline
figure — is on the order of **$200 and many hours**, not the ~$40 this section
used to claim. `--split tuning` is ~$30, not ~$7.

Where it goes is measurable rather than guessed:

```bash
python -m eval.runner profile          # reads traces already on disk
```

Almost all of it is the three Sonnet nodes — planner, synthesiser, critic — at
`effort: high`. The measurements themselves are pure functions and take
milliseconds; none of the wait is the physics. Most of a multi-cycle run is
spent *after* cycle 1, re-reviewing an answer that four times over was already
right, which is why the convergence stop (DECISION 0070) is a latency fix and an
accuracy fix at once.

`config/models.yaml` carries a `fast` profile — lower effort on the planner and
critic — as something to measure against `default` rather than a change made on
a hunch:

```bash
PV_MODEL_PROFILE=fast python -m eval.runner run --engine agent --only G-017
```

Circuit breakers of 80 calls and $3.50 per investigation are enforced before
each call. They are ceilings, not budgets: `BudgetExceeded` aborts the whole
evaluation, so a breaker set below what four cycles legitimately cost turns an
expensive case into a dead run. Use `--resume FILE` so a run killed by an API
overload picks up where it stopped instead of re-paying for finished cases.

**This is a batch tool, not a chat box.** Nobody types a question and waits ten
minutes. The shape that makes sense is `watcher.py`: a sweep that runs
unattended and leaves a ranked findings queue with the evidence behind each one,
which an engineer triages in seconds. The alternative is an engineer spending an
afternoon on the same question.

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

## Results

### What has been measured

**The rules baseline, on 43 held-back cases:**

| | tuning | held back |
| --- | --- | --- |
| Overall accuracy | 0.345 | 0.469 |
| False alarms on look-alikes | 0.056 | 0.056 |
| **Correct "not enough evidence"** | **0.000** | **0.000** |
| Missed real faults | 0.375 | 0.438 |

That zero is structural, not a tuning failure. The engine commits to the first
rule that fires and cannot represent "two causes survive and here is the test
that separates them" — so on all five unresolvable cases per split it confidently
answers "clipping", and on the curtailed ones it is confidently wrong in the
direction that costs money. **It is the single number to watch when the agent
column is filled in.**

Its agency metrics read exactly as they should: one distinct tool trajectory
across all 86 runs, an unplanned-measurement rate of 0.000, zero self-initiated
abstentions. That is the metric correctly identifying a pipeline. If the agent
comes back looking like this, the agent is a pipeline too.

**Retrieval, on 18 golden queries:** fusing BM25 with a vector index beats either
alone by about 0.09 of reciprocal rank. The reranker adds 0.009 — one query
moving one position — so on this evidence it is close to cost without benefit.

Read that with a large caveat, stated here rather than buried: **the corpus is
23 chunks and they are the project's own knowledge base.** No external documents
have been ingested. So this measures finding the right entry among 23 items that
could equally be looked up by key, which is not the problem retrieval exists to
solve. Worse, `CorpusRetriever` is not wired into any run — `investigate`
defaults to the hand-written `KnowledgeBase`, a dict lookup on cause names. Every
`Looked up what is known about…` line in a run log is that lookup, not
retrieval. The stack is built, tested and unused; making the claim real needs a
real corpus behind it.
"Right document in top 10" is **not reported**, because on a 23-chunk corpus it
scores 1.000 for every configuration including ones that have learned nothing;
the harness detects that and drops the column rather than printing a number a
reader might quote.

**The physics, over two untouched years:** performance ratio 0.857 as measured,
0.893 corrected to 25 °C. Raw PR swings 0.80–0.95 across the year while the
corrected figure stays flat at 0.86–0.92 — every one of those raw summer troughs
is a false alarm waiting to be dispatched on.

**The agent, on eight tuning cases with `--checks-only`:**

| | before the string-baseline fix | after |
| --- | --- | --- |
| Overall accuracy | 0.493 | 0.700 |
| Cause accuracy | 0.625 | 0.875 |
| **False alarms on look-alikes** | **0.500** | **0.000** |
| Correct "not enough evidence" | 1.000 | 1.000 |
| Missed real faults | 0.000 | 0.000 |

Median $0.36 and 196 seconds a case, no failures, eight distinct tool
trajectories from eight runs.

**Read this as a debugging run, not a score, for three reasons.** Eight cases is
a subset and the harness prints so above the results. The tools were *changed in
response to these eight failing*, so the number is optimistic by construction —
the other 35 tuning cases are the fairer test and the held-back split is the
real one. And two cases improved while only one of them is attributable: G-005
read the corrected string figure and moved off `string_outage`, which is exactly
the predicted mechanism, while G-029 took a different measurement path and may
simply have varied.

That zero on false alarms is the number that moved, and it moved because a tool
was measuring against the wrong reference — this array has a string that has
carried ~12% less than its neighbours for the whole record, and two tools
reported that standing characteristic as a deficit. See DECISION 0079 and 0083.

### What has not

- **The agent's accuracy at a scale worth quoting.** Eight cases is a debugging
  run. The full tuning split, then the held-back split, reported separately with
  the gap stated, is what turns the v1 targets from `n/a` into a verdict.
- **Whether retrieval changes it.** `--no-knowledge` runs the identical loop
  without it. The retrieval numbers say the right passage comes back; they say
  nothing about whether the agent diagnoses better for having read it — and on a
  23-chunk corpus that is its own knowledge base, they say very little at all.
- **Whether RAG is in the loop.** It is not. See the caveat above.
- **Whether the LLM reviewer earns its cost, at any scale.** One case has been
  run both ways. G-017 with the reviewer took 644-900 seconds and $1.09-1.31 and
  ended on "not enough evidence"; with `--no-review` it answered `shading`
  correctly in 165 seconds for $0.29. That is n=1, on the one case the reviewer
  had already failed three times, so it is the least surprising case for it to
  happen on — but it is why `--checks-only` exists, and why the reviewer is off
  in the eight-case figures above. Across every case seen so far the reviewer
  has improved one answer and talked four correct ones out of themselves.

`docs/WALKTHROUGH.md` explains the whole project end to end — problem,
architecture, data, tools, guarantees, evaluation, watcher, dashboard — and is
the place to start if you are reading this repository for the first time.
`docs/FINDINGS.md` has the full detail, including a section on the bugs the
evaluation found in *itself* — each of which would have made a published accuracy
figure meaningless.

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
| 6 | Critic with structured verdict, iteration cap, not-enough-evidence path | **done** |
| 7 | Rules baseline through the shared tools + comparison harness | **done** |
| 8 | Watcher: sweep, findings store with lifecycle, energy ranking. Tab 2 | **done** |
| 9 | RAG layer + retrieval golden set + ablation | **done** |
| 10 | Saved analyses registry with golden-case enforcement | **done** |
| 11 | Full evaluation, agency metrics, both experiments | harness done, **eight cases run; full splits outstanding** |
| 12 | LangGraph port; verify identical golden-set outputs | **done** |
| 13 | README with honest results, including whatever the ablations showed | **done** |

Each step is gated: it stops for review before the next one starts.

Both loops ship. `src/agent/loop_plain.py` is the reference implementation;
`src/agent/loop_graph.py` is the LangGraph port, held to sixteen exact-equality
tests against it. `docs/LANGGRAPH_TRADEOFF.md` records what the framework
actually bought — checkpointing, which an 86-case evaluation that dies at case 60
genuinely wants — and what it cost.

---

## How an investigation runs

    plan  ->  route -> measure  (repeat)  ->  write the answer  ->  review
      ^                                                               |
      +---------------------- send it back ---------------------------+

The planner enumerates candidate causes — benign ones included, since a plan
containing only faults has already decided the answer — and opens a line of
enquiry. The router picks one measurement at a time, may depart from the plan
whenever a result points elsewhere, and may instead ask what separates two
candidates before spending a measurement on either. The synthesiser writes the
finding and is allowed no arithmetic.

The reviewer returns a structured verdict and cannot return prose. Three parts
of that verdict are not its to decide: a figure with no measurement behind it is
unsupported however the review reads it, a look-alike counts as considered only
if a measurement bearing on it was actually taken, and a review that cannot
produce a valid verdict becomes another round rather than a silent approval. The
reviewer can be stricter than the model wanted; it can never be looser.

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

**The agent has not been scored yet** — no API key was available in the
environment this was built in, so `docs/FINDINGS.md` carries the rules baseline,
the physics validation and the bugs the evaluation found in itself, with the
agent column explicitly empty rather than pending.

The rules baseline, on 43 held-back cases: overall accuracy 0.469, false alarms
on look-alikes 0.056, and **correct "not enough evidence" 0.000** — which no
amount of tuning changes, because the engine commits to the first rule that fires
and cannot hold two causes open. That number is the one to watch when the agent
column is filled in.
