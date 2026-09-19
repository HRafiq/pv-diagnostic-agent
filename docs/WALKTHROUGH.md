# The project, end to end

A guide to what this system is, why it is shaped the way it is, and what it has
actually been shown to do. Written to be read start to finish by someone who has
not seen the code.

The other documents have narrower jobs. `README.md` is the front door and the
commands. `docs/DECISION.md` is an append-only log of every non-obvious choice
with the alternatives considered. `docs/FINDINGS.md` is what the evaluation
measured. `CLAUDE.md` is the rules that stop the project quietly cheating. This
one is the map.

---

## 1. The problem

A utility-scale solar plant produces less than it should. Someone has to say
why, and the answer decides whether a truck rolls.

The difficulty is not that faults are hard to see. It is that **the benign
explanations look identical to the expensive ones** on the only signal most
people look at, which is energy:

| what you see | could be | costs |
| --- | --- | --- |
| output down 12% this month | a failed string | an electrician, same day |
| output down 12% this month | dust on the glass | a wash, scheduled |
| output down 12% this month | it was cloudy | nothing |
| output down 12% this month | it was hot | nothing — this is physics |
| output flat at midday | inverter clipping by design | nothing |
| output flat at midday | the grid operator curtailing you | a phone call |
| performance ratio falling | the array degrading | investigation |
| performance ratio falling | the irradiance sensor drifting | clean one sensor |

Three of those are faults and five are not. A monitoring system that alarms on
deficits is right about the deficit and useless about the cause, and the cost of
being wrong is asymmetric: **a crew dispatched to a healthy array is a wasted
day, and it happens far more often than a missed fault.**

Three facts about photovoltaics make this harder than it looks, and all three
are in every prompt the agent sees:

- **Output tracks sunlight.** A cloudy week and a failed string are the same
  number of kWh. Raw energy is not evidence of anything on its own.
- **Uncorrected performance ratio falls every summer on a perfectly healthy
  plant**, because silicon loses roughly 0.4% of its power per degree above
  25 °C. Every summer trough is a false alarm waiting to be dispatched on.
- **Inverters hold their AC ceiling every clear midday by design.** So does a
  grid export limit. On the power channel they are indistinguishable.

So the real task is not detection. It is **differential diagnosis**: hold
several candidate causes at once, choose the measurement that best separates
them, and commit only when one survives — or say plainly that the data cannot
decide, and name the cheap test that would.

That last part is the one most systems cannot do at all, and it is the one this
project treats as a first-class outcome.

---

## 2. What was built, in one paragraph

An agent that investigates a solar plant. It is given a question in plain
language and a window of telemetry. It proposes candidate causes, chooses
measurements from a fixed set of eighteen analysis tools, reads the results,
chooses more, and writes a finding — either a committed diagnosis with the
evidence behind it, or an honest "not enough evidence" naming what would settle
it. Every step is written to a trace file. The same tools, run by a hand-written
rules engine, form the baseline it is scored against. An evaluation harness
scores both on 86 cases with known answers, built by injecting faults into real
measured data at the physics level.

---

## 3. Architecture

### 3.1 The shape

```
                    ┌─────────────────────────────────────────┐
                    │  question + window of telemetry         │
                    └────────────────┬────────────────────────┘
                                     │
        ┌────────────────────────────▼────────────────────────────┐
        │  PLANNER   candidate causes, what each costs to act on, │
        │  (LLM)     which tool would separate which pair         │
        └────────────────────────────┬────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │  RETRIEVE   what is known about │
                    │  (no LLM)   these causes        │
                    └────────────────┬────────────────┘
                                     │
     ┌───────────────────────────────▼───────────────────────────────┐
     │  ROUTER    one call: which measurement next, or stop           │  ◄──┐
     │  (LLM)     may depart from the plan, must say why              │     │
     └───────────────────────────────┬───────────────────────────────┘     │
                                     │                                     │
                    ┌────────────────▼────────────────┐                    │
                    │  EXECUTOR   run the tool        │  ──────────────────┘
                    │  (no LLM)   pure function       │
                    └────────────────┬────────────────┘
                                     │
        ┌────────────────────────────▼────────────────────────────┐
        │  SYNTHESISER  the finding, or an honest refusal         │
        │  (LLM)        every figure must trace to a measurement  │
        └────────────────────────────┬────────────────────────────┘
                                     │
        ┌────────────────────────────▼────────────────────────────┐
        │  REVIEW     accept / send back / not enough evidence     │
        │             deterministic checks + optional LLM judgement│
        └────────────────────────────┬────────────────────────────┘
                                     │
                        send_back ───┘ (back to PLANNER)
```

Four LLM nodes, one non-LLM executor, and a review stage that is partly
arithmetic. The loop is written twice — once in plain Python
(`src/agent/loop_plain.py`) and once as a LangGraph state machine
(`src/agent/loop_graph.py`) — and a test asserts they produce identical output
given identical scripted replies. The plain loop is the specification; the graph
is what actually runs.

### 3.2 What each node is for

**Planner** — proposes at least two candidate causes, normally four to six,
including the benign ones. For each: what it would cost to act on it if true,
and which tool would tell it apart from its nearest look-alike. Then an ordered
opening plan. The "what would it cost" field is what makes one candidate worth
separating from another — two causes that lead to the same action barely need
separating; *wash the array* versus *wipe the sensor* needs separating badly.

**Router** — one call per measurement: which tool next, or stop. It may depart
from the plan and must give a specific reason when it does. It is deliberately
**not** asked whether a call was planned — that flag is derived by the executor
from the plan the planner already committed to, because a model asked "was this
in your plan?" is being invited to flatter itself, and the unplanned-measurement
rate is the sharpest agency metric in the evaluation.

**Executor** — not an LLM. Validates arguments against a Pydantic model, runs
the tool, records the result and the provenance.

**Synthesiser** — writes the finding. Every number it writes must appear in a
tool result; this is checked automatically, not trusted. If the measurements did
not separate the surviving causes it must say so, list every cause still
standing with what each would mean operationally, and name the test that would
settle it.

**Review** — three modes, because the node does two separable things:

| mode | what runs | when to use |
| --- | --- | --- |
| default | deterministic checks + LLM reviewer | the original design |
| `--checks-only` | deterministic checks alone | what the evidence points at |
| `--no-review` | nothing | the ablation's control |

The deterministic half — no fabricated numerics, every look-alike weighed, no
abstention naming a tool the agent owns and did not run — is free and
verifiable. The LLM half is slow and, so far, has improved one answer and talked
four correct ones out of themselves. See §7.3.

### 3.3 The rules that shape it

These are in `CLAUDE.md` and are not style preferences. Each exists because
breaking it silently destroys something the project is trying to measure.

**No tool returns a fault classification.** Tools take measurements.
`compute_pr`, `characterize_onset`, `per_mppt_current_balance` — yes.
`classify_fault` — no. If the diagnosis happens inside one function, this is a
rules engine with a chat wrapper.

**No LLM inside a tool.** Tools are pure, deterministic functions with validated
arguments and structured output. Strategy and interpretation belong to the
agent; arithmetic and thresholds do not.

**No LLM in arithmetic, thresholds, scoring, or data lookup.** The last one was
added late: the synthesiser was choosing a finding's *category* independently of
its cause, when the knowledge base already maps one to the other. That is a
dictionary lookup, and it got it wrong once in eight.

**The dashboard never reimplements a computation.** A chart showing performance
ratio calls the same `physics.compute_pr()` the tool node calls. This is what
guarantees the interface and the agent cannot diverge.

**No `datetime.now()` anywhere in `src/`.** There is no live public PV feed, so
"now" is a replay clock over historical data. A test scans the source tree to
enforce it.

**No ML vocabulary in the interface.** Internally: abstention, confounder,
groundedness, holdout. On screen: *not enough evidence*, *look-alike*, *claims
backed by a source*, *held back*.

---

## 4. The data, and why the fault injection is done the hard way

### 4.1 Real measured data

**NREL PVDAQ system 4902, NIST_Ground_1** — Gaithersburg, Maryland. 270.7 kW,
1152 modules, fixed 20° south-facing mount, 2016–2017 at 15-minute resolution,
with seven per-string current channels. Public data. Ingest scripts download
into gitignored directories and commit a manifest of SHA-256 checksums, so the
dataset is reproducible without shipping it.

Two years of untouched data give: performance ratio 0.857 as measured, 0.893
corrected to 25 °C. Raw PR swings 0.80–0.95 across the year while the corrected
figure stays flat at 0.86–0.92 — every one of those raw summer troughs is a
false alarm waiting to happen, visible in the real data before any fault was
injected.

### 4.2 Faults injected at the physics layer, never at the signature layer

This is the single most important decision in the project.

The easy way to build an evaluation is to write a case that says "performance
ratio drops by 0.12" and a rule that says "if performance ratio drops by 0.12,
report a string fault". That evaluation measures whether your rule matches your
generator. It always scores well and it means nothing.

So faults are injected as **physics on the real measured series**:

| injector | what it does to the data |
| --- | --- |
| `string_outage` | zeroes a fraction of DC current on one combiner, with a realistic transition and noise |
| `shading` | attenuates part of the array at a fixed time of day, bending the daily shape |
| `soiling` | multiplies optical transmission down gradually, recovering after rain |
| `sensor_drift` | biases the irradiance sensor, so a healthy plant looks sick |
| `telemetry_gap` | removes or freezes readings |
| `inverter_clipping` | caps output at a ceiling — DC *and* AC, see below |
| `curtailment` | the same cap, applied for a different reason |

The diagnostic must **rediscover** the signature from the data. Nothing tells it
what a string fault looks like.

### 4.3 The simulator and the expectation model share no configuration

If the synthetic plant were generated by pvlib and `compute_expected_output()`
were the same pvlib ModelChain with the same parameters, every injected deficit
would be trivially detectable and accuracy would be meaningless. So the two
stacks are deliberately different: simulate with the detailed one (SAPM/CEC,
full loss tree, sensor noise, parameter offsets), form expectations with the
coarser one (PVWatts + Faiman).

### 4.4 The golden set

86 cases, split 43 tuning / 43 held back, with identical composition:

| | count | | count |
| --- | --- | --- | --- |
| string_outage | 11 | sensor_drift | 9 |
| telemetry_gap | 7 | shading | 5 |
| soiling | 4 | curtailment | 3 |
| clipping | 2 | *no injection* | 2 |

By category: 16 `fault`, 4 `recoverable`, 1 `by_design`, 17 `not_the_plant`, and
**5 with no expected category at all** — the unresolvable cases, where the
correct answer is "not enough evidence" and committing either way is wrong.

**Never tune against the held-back set.** Tuning and held-back results are
reported separately, always, and a gap over 0.10 means overfitting and gets said
out loud.

---

## 5. The tools

Eighteen atomic measurements. Each answers one question and none returns a
verdict.

| tool | question it answers |
| --- | --- |
| `compute_temp_corrected_pr` | Is the plant under-performing once weather and heat are divided out? |
| `compute_expected_output` | How large is the shortfall, in kWh and as a fraction? |
| `weather_context` | Was the sunlight itself unusual? |
| `compare_to_trailing_baseline` | Is this new, or has the plant always run like this? |
| `daily_performance_trend` | Is performance drifting steadily, and does it recover on its own? |
| `characterize_onset` | Did this arrive abruptly or accumulate? |
| `time_of_day_profile` | Is the loss confined to a band of hours? |
| `per_mppt_current_balance` | Is the loss confined to part of the array? |
| `string_onset_scan` | Which string changed, and when? |
| `compare_string_profiles` | Is a string merely lower, or a different *shape*? |
| `check_ac_ceiling` | Is output being held at a ceiling? |
| `dc_to_ac_conversion` | Is the loss on the array side or at the inverter? |
| `check_clearsky_consistency` | Is the irradiance sensor telling the truth? |
| `check_night_offset` | Is a channel biased away from zero? |
| `check_module_temperature_sensor` | Can the temperature correction be trusted here? |
| `detect_stuck_channels` | Is a sensor frozen rather than merely wrong? |
| `profile_data_quality` | Is the apparent loss a plant problem or a logger problem? |
| `soiling_recovery_pattern` | Does this loss come back by itself? |

Each declares which look-alikes it discriminates, and that declaration is load
bearing: it drives the coverage requirement the agent is held to (§6.2) and is
inverted to tell the planner which tool settles which line.

---

## 6. Guarantees the system enforces on itself

Everything in this section is arithmetic. None of it asks a model to check its
own work.

### 6.1 Every figure must trace to a measurement

`src/agent/grounding.py` pulls every numeric literal out of the answer and looks
for it in the union of the tool results' provenance ledgers. A figure that is
not there did not come from a measurement, and the answer is sent back.

Three allowances, all deliberate: percentage presentation (0.0823 and "8.2%" are
the same measurement in different clothes), small integers up to 24 (how
sentences count things), and **quotation** — a figure the agent was *shown*, in
a tool's prose or the knowledge base, may be repeated. What is never allowed is
arithmetic: if the ledger holds an expected and a measured energy and the prose
states their difference, that difference is flagged, because the subtraction
belongs in a tool where it is testable.

### 6.2 Every look-alike must be weighed, by measurement

Seven look-alikes are checked on every investigation: weather, seasonal
temperature derating, clipping, curtailment, sensor drift, snow or dust, and
telemetry gap. "Weighed" means a tool that discriminates it actually ran —
asserting "I considered clipping" without ever measuring whether output sits on
a ceiling is box-ticking.

The mapping from look-alike to qualifying tools is derived from the registry and
is the **single** definition: the critic checks against it and the planner is
shown it. Neither can drift from the other.

### 6.3 An unsettled answer may never show a cause or a confidence

If the investigation could not separate array soiling from sensor soiling, the
finding must not say "soiling, confidence 0.78" — that is the exact bug that
dispatches a wash crew to a clean array. Model validators enforce it, and the
synthesiser strips the fields *before* construction rather than relying on the
validator to catch it.

### 6.4 An abstention may not name a measurement the agent could have taken

`not_enough_evidence` is a first-class successful outcome — **when the evidence
genuinely cannot be obtained.** One case declined to answer and gave its reason
as "run `string_onset_scan`", a tool in its own registry it had simply not got
to. That is an unfinished investigation wearing an abstention.

The discriminator is a registry lookup on the resolving measurement:

```
"Run string_onset_scan to see whether…"          -> ['string_onset_scan']  send back
"Pull the grid operator's dispatch log…"         -> []                     honest
```

### 6.5 Agency is measured, never asserted

Every tool call records `was_planned`, with a reason when it is `False`. The
evaluation reports distinct tool trajectories, unplanned-measurement rate,
iteration distribution and self-initiated abstentions. The rules baseline scores
one trajectory across all 86 runs and an unplanned rate of 0.000 — that is the
metric correctly identifying a pipeline. If the agent ever comes back looking
like that, the agent is a pipeline too.

---

## 7. How it is evaluated

### 7.1 The metrics, and which one matters

| metric | what it means |
| --- | --- |
| **false alarms on look-alikes** | called a fault on something that is not one. **The headline.** |
| correct "not enough evidence" | abstained on the genuinely undecidable cases |
| missed real faults | called a real fault benign, or abstained on it |
| cause / category accuracy | got the right answer |
| overall accuracy (macro-F1) | averaged over categories, abstention as its own class |

Overall accuracy is deliberately not the headline. A detector that alarms on any
deficit scores well on clean faults and is useless in the field; only the
look-alike column exposes that.

Abstention is scored as **correct** on the unresolvable cases and **wrong**
everywhere else, so an agent that abstains on everything must score badly.
Macro rather than micro averaging, so a rare class cannot be ignored for free.

### 7.2 The baseline it is scored against

A hand-written rules engine running **the same eighteen tools**, so a gap
between them is a gap in reasoning rather than in measurement.

| | tuning | held back |
| --- | --- | --- |
| Overall accuracy | 0.345 | 0.469 |
| False alarms on look-alikes | 0.056 | 0.056 |
| **Correct "not enough evidence"** | **0.000** | **0.000** |
| Missed real faults | 0.375 | 0.438 |

That zero is structural. The engine commits to the first rule that fires and
cannot represent "two causes survive and here is the test that separates them",
so on all five unresolvable cases per split it confidently answers "clipping".
**It is the single number to watch when the agent is scored** — the one column
where an agent has somewhere to be better rather than merely different.

### 7.3 What the agent has actually shown

Eight tuning cases, `--checks-only`, before and after a tool fix:

| | before | after |
| --- | --- | --- |
| Overall accuracy | 0.493 | 0.700 |
| Cause accuracy | 0.625 | 0.875 |
| **False alarms on look-alikes** | **0.500** | **0.000** |
| Correct "not enough evidence" | 1.000 | 1.000 |
| Missed real faults | 0.000 | 0.000 |
| Median cost / latency | — | $0.36 / 196s |

**This is not a score, for three independent reasons.** Eight cases is a subset.
The tools were changed *in response to these eight failing*, which is tuning, so
the number is optimistic by construction. And only one of the two improvements
is attributable — the other case took a different measurement path and may
simply have varied, because LLM nodes are not deterministic and one run per case
cannot separate a fix from a coin.

The reviewer has been priced on exactly one case: with it, 644–900s and
$1.09–1.31, ending on "not enough evidence"; with `--no-review`, correct in 165s
for $0.29. That is n=1 on the case it had already failed three times.

### 7.4 The ablations

| configuration | question it answers |
| --- | --- |
| full | the number |
| `--no-review` | does the reviewer earn its cost? |
| `--no-knowledge` | does the knowledge layer change accuracy? |
| neither | are the two substituting for each other? |

Headline metrics are reported as mean ± spread over three runs, never as a
single number, because `CLAUDE.md` scopes determinism honestly: physics, tools,
injection and retrieval are bitwise reproducible; the four LLM nodes are not and
never will be — `temperature` no longer exists on current Claude models.

---

## 8. The Watcher

The agent answers a question. The Watcher decides which questions are worth
asking.

`watcher.py` advances the replay clock in steps, sweeps each window for
deficits, investigates what fires, and writes findings to a store with a
lifecycle: `new` → `ongoing` → `resolved`, plus `acknowledged` and `suppressed`
for human decisions. Findings are ranked by energy at stake, so the queue is
ordered by what it costs to ignore.

**This is the shape the system is actually for.** Nobody types a question and
waits three minutes at a prompt. An engineer opens the morning list, reads
*"string 1's share dropped 0.029 around 2017-03-16, shading, here are the six
measurements behind it"*, and triages in seconds. The alternative being compared
against is an engineer spending an afternoon — not a chatbot answering
instantly.

Run against untouched 2016 data, the Watcher finds a real telemetry gap in the
NIST record that nobody planted.

---

## 9. The dashboard

FastAPI + Jinja2 + Alpine.js + Plotly. Deliberately thin: it calls `src/` and
computes nothing itself.

**Plant** — the dataset as ingested. Power, irradiance, temperature, per-string
currents; measured against modelled expectation; performance ratio raw and
temperature-corrected. This is where the "every summer trough is a false alarm"
claim becomes visible rather than asserted.

**Watcher** — the findings queue. Each finding shows its title, what it costs to
ignore in kWh, whether that figure is verified, its lifecycle state, and — only
when settled — a cause and a confidence. Unsettled findings show the surviving
causes and the test that would separate them. Closed findings are hidden by
default.

**Investigate** — the replay. Pick an investigation and watch it happen: the
candidate causes the planner proposed, each measurement in order with its result
and its cost, which calls were unplanned and why, the review verdicts, and the
final answer with the evidence behind each claim. This is the trace file
rendered, which means **the interface cannot show something the agent did not
do.**

Two further tabs (Scenario builder, Evaluation) are declared and disabled.

### Running it

```bash
uv sync
uv run uvicorn dashboard.app:app --reload      # http://127.0.0.1:8000
```

No API key needed — the dashboard only reads what is already on disk. What each
tab can show depends on what you have run:

| tab | needs | if missing |
| --- | --- | --- |
| **Plant** | ingested data in `data/raw/` | says "No data ingested yet" |
| **Watcher** | findings in `findings/store.jsonl` | says "Nothing open" |
| **Investigate** | trace files in `traces/` | says "No investigations yet" |

To fill all three from nothing:

```bash
# ~10 min, downloads two years of public PVDAQ data into gitignored dirs
uv run python -m src.data.cli ingest --system 4902 --years 2016 2017

# the Watcher on the rules engine — no API key, finds the real 2016 gap
uv run python watcher.py run --until 2017-08-01 --step 14D --engine rules

# any agent run writes traces the Investigate tab replays
uv run python -m eval.runner run --engine agent --split tuning --only G-017 \
    --checks-only
```

The Investigate tab is the one worth opening first if you have run the agent at
all — it is the trace file rendered, so it shows exactly what happened and
cannot show anything that did not.

The interface language rule is enforced here: the trace calls a mid-investigation
tool call `adaptive`, and the dashboard renders it `unplanned`.

---

## 10. Layout

```
src/            UI-agnostic core — everything the agent and dashboard share
  clock.py        the ONLY time source
  determinism.py  seeding and run-scoped RNG
  config.py       typed loaders for config/*.yaml
  trace/          TraceStep model + JSONL writer (the dashboard's data source)
  findings/       Finding model, lifecycle, ranking
  agent/          state, nodes, plain loop, LangGraph port, grounding
  physics/        PR, expectation model, cell temperature
  tools/          the 18 measurements + registry
  detect/         the Watcher's deficit sweep
  knowledge/      hand-written fault signatures (YAML) + cause vocabulary
  rag/            corpus, hybrid index, retriever
  baseline/       the rules engine
  analyses/       saved-analysis registry
  data/           PVDAQ ingestion
simulator/      physics-level fault injectors
eval/           runner, metrics, agency metrics, golden sets, profile, subset
dashboard/      FastAPI + Jinja2 + Alpine.js; thin, calls src/ only
watcher.py      CLI: advance the clock, sweep, write findings
```

Python 3.11, uv with a committed lockfile, Pydantic v2 at every boundary, ruff +
mypy + pytest, ~900 tests. All times timezone-aware UTC internally, site-local
only at the edges. Units in field names wherever there is doubt
(`energy_at_stake_kwh`).

---

## 11. Where it stands, honestly

**Works and is measured:** the physics, the injectors, the rules baseline, the
retrieval index, the Watcher, the dashboard, both loop implementations proven
equivalent.

**Works and is not yet measured at a quotable scale:** the agent. Eight tuning
cases have been run, and they are the cases the tools were tuned against. The
full tuning split and then the held-back split are what turn the v1 targets from
"not measured" into a verdict.

**Built and not in the loop:** RAG. `investigate` defaults to the hand-written
knowledge base — a dictionary lookup on cause names — so every "looked up what
is known about…" line in every log is that lookup, not retrieval.
`CorpusRetriever` is built, tested, measurably better than BM25 alone, and
constructed by nothing. The corpus is also 23 chunks and *is* the knowledge
base. Making the claim real needs real documents behind it.

**The open question the tool fixes do not answer.** One case reads
uncomfortably: the agent stated the correct discriminator unprompted — *"a
string outage shows exactly one step"* — measured it three times, got three
answers pointing at soiling, and cited none of them, quoting instead a figure
from the same sentence as the disconfirming one. Correcting tools raises the
floor. It does not establish that the reasoning above the floor is reliable.

**Cost and latency.** ~$0.36 and ~3 minutes per case, against an original design
target of $0.15. Measured from 13 timed runs, the synthesiser is 50.2% of all
latency and the planner 22.4%; the measurements themselves are pure functions
and take milliseconds, so none of the wait is the physics.

---

## 12. If you read one more thing

`docs/FINDINGS.md`, section **"Bugs the evaluation found in itself"**. It lists
the defects the harness found in the harness — an injector that left string
currents untouched while claiming a string fault, a case designed to be
unresolvable that was quietly resolvable, a fall printed as a rise with a test
that agreed with the bug, a tool measuring against a theoretical reference and
reporting a permanent plant characteristic as a deficit.

Each would have made a published accuracy figure meaningless, and each was
invisible until something was built to look at data that had already been
recorded. That section is the most honest thing in the repository and the best
argument that the numbers elsewhere in it can be trusted.
