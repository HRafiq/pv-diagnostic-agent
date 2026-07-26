# Decision log

Append-only. Every non-obvious choice gets a dated entry with the alternatives
considered. This is what makes the project explainable in an interview six
months later.

---

## 0001 — Replay clock as the only time source (2026-07-25)

**Decision.** All of `src/` reads "now" from an injected `Clock`. `datetime.now()`
and its relatives are banned and the ban is enforced by an AST scan in
`tests/test_no_wall_clock.py`.

**Why.** There is no free real-time public PV feed, so the system runs over
historical data with a configurable "now". Without the discipline, a single
stray wall-clock read compares a replayed 2019 dataset against a 2026 today:
every trailing baseline window comes back empty and the detector quietly finds
nothing. The failure is silent and survives code review, which is why it is
mechanically checked rather than documented.

**Alternatives.** (a) Freeze a module-level `NOW` — breaks as soon as two runs
sit at different points in the replay. (b) Monkeypatch `datetime` in tests —
catches it in tests only, not in the Watcher. (c) Pass a timestamp through every
signature — a lot of plumbing for a value that is genuinely ambient.

**Note.** `ReplayClock` uses `time.monotonic()` for accelerated replay. That is a
duration source, not a calendar, so it cannot leak a real date into a model.

---

## 0002 — `watcher.py` is a separate process; the dashboard is read-only (2026-07-25)

**Decision.** A CLI advances the clock, runs the deterministic detectors over
the interval that just "arrived", and writes findings. The dashboard only reads.

**Why.** The handoff says the sweep runs when new data lands (§6.5) and also that
time is a replay clock (§6.3). Those are in tension: with a replay clock nothing
ever *lands*, and a request-scoped web app has nowhere to host a daemon. Making
the watcher process itself the arrival of data resolves both, and keeps the
"dashboard never triggers a run" rule literally true rather than nearly true.

**Alternatives.** (a) An "advance clock" button in the UI — the dashboard would
be triggering runs, exactly the rule being avoided. (b) A background thread in
the web app — non-deterministic, and it dies with the reload.

---

## 0003 — Alpine.js + FastAPI instead of Streamlit (2026-07-25)

**Decision.** Dashboard is FastAPI + Jinja2 + Alpine.js + Plotly.js, with both
JS libraries vendored into `dashboard/static/vendor/` at pinned versions.

**Why.** User's call, overriding handoff §6.1 (Streamlit) and §10, which lists a
FastAPI front end as a non-goal. Recording it so the contradiction is deliberate.
Alpine is ~45 KB with no build step, so the cost is well short of the "triples
the effort" concern that ruled out React. The §6.1 architecture rule — `src/viz/`
returns chart *specs*, data and config rather than images — actually fits this
stack better than Streamlit, because a Plotly spec is JSON that Plotly.js
consumes directly.

**Cost.** A backend layer that Streamlit would have provided for free: API
endpoints, templates, and hand-written chart configs. Roughly 2-3x the dashboard
code.

**Vendored, not CDN.** The environment's network policy blocks jsDelivr and
cdn.plot.ly outright, but the deciding reason is different: an evaluation run
must not have a chart library change version underneath it, and the dashboard
has to work offline. A test asserts no CDN hostname appears in any template.

---

## 0004 — No `temperature` parameter; determinism is scoped (2026-07-25)

**Decision.** `config/models.yaml` has no `temperature`, `top_p` or `top_k`, and
`ModelConfig` has no such fields. Reasoning depth is set with `effort`. The
determinism guarantee is scoped to the deterministic half of the system.

**Why.** Handoff §9 and §5.6 specify "LLM temperature 0". That parameter was
**removed** on Claude Opus 5, Sonnet 5, Opus 4.7 and Opus 4.8 — sending it
returns HTTP 400. Even where it still exists it never guaranteed identical
sampling. Writing it into config would have produced a runtime error on the
first API call at step 4, and the guarantee it implied was never real.

**What replaces it.**

| Layer                                 | Guarantee                                |
| ------------------------------------- | ---------------------------------------- |
| Physics, detectors, tools, injection  | Bitwise identical for the same seed      |
| Retrieval                             | Identical for a fixed corpus snapshot    |
| LLM nodes                             | Not deterministic — never claimed        |

For regression runs, LLM responses are cached by `(model, prompt_hash)` so a
re-run compares like with like. Headline eval metrics are reported as mean ±
spread over N=3 fresh runs. Without this, Tab 5's "which cases changed verdict
since last run" view would be reading sampling noise and calling it a
regression.

---

## 0005 — `src` is an importable package (2026-07-25)

**Decision.** Imports read `from src.physics import compute_pr`, matching the
handoff §8 layout literally.

**Why.** §8 specifies `src/clock.py`, `src/agent/state.py` and so on, not
`src/pv_agent/clock.py`. Deviating would have meant every path in the handoff
needing mental translation for the life of the project. A package named `src` is
common in research repositories and carries a small collision risk that is
acceptable in a single-purpose private repo.

**Alternative.** `src/pv_agent/` with a proper distribution name — better
practice for a library, unnecessary ceremony for an application.

---

## 0006 — Invariants are validators, not documentation (2026-07-25)

**Decision.** The rules that keep the interface honest are Pydantic validators
that raise, not conventions in a style guide:

- an unsettled `Finding` cannot carry a cause, a confidence, fewer than two
  candidate causes, or verified energy;
- a `TraceStep` with `was_planned=False` must state why the tool was chosen;
- `CriticVerdict` cannot return `accept` with unsupported claims or an
  incomplete look-alike checklist, and cannot `send_back` without a concrete
  revision request;
- `AnalysisSpec` cannot be saved without at least one golden case.

**Why.** Each of these is a rule the handoff states, and each fails silently if
broken. A finding that reads "soiling, confidence 0.78" when the investigation
could not separate array soiling from sensor soiling *looks correct* — it is the
exact bug that dispatches a wash crew to a clean array. Made unconstructible
instead.

---

## 0007 — Corrections to the handoff's §1 framing (2026-07-25)

**Decision.** The claim "a dirty pyranometer produces the same signature as a
dirty array" is not carried into the knowledge layer.

**Why.** They have opposite signs in PR. A dirty array drops power with measured
irradiance unchanged, so PR **falls**. A dirty pyranometer under-reads
irradiance with power unchanged, so PR = E / (P_STC · G/G_STC) **rises**. They
are also separable by `check_clearsky_consistency` and `compare_poa_to_pvgis`: a
soiled sensor diverges from clear-sky and satellite POA, a soiled array agrees
with both while power drops. Encoding a false equivalence in
`fault_signatures.yaml` would teach the agent a wrong discriminator.

**What replaces it.** Two genuinely hard pairs:

1. **Co-soiling reference cell.** Many plants use an in-plane PV reference cell
   rather than a thermopile pyranometer. It soils at the same rate as the
   modules, so measured irradiance and power fall together, PR stays flat, and
   the soiling is invisible. Not a false alarm — a *missed* real loss, which is
   worse. Resolving measurement: a paired clean/dirty soiling station, or the
   step change after a cleaning event. A natural `not_enough_evidence` case.
2. **Clipping vs curtailment.** Identical ceiling, different cause, different
   action, and no single-signal test separates them. Already the handoff's own
   worked example for the critic (§3.5), and the better showcase.

---

## 0008 — Anti-circularity extended to the model layer (2026-07-25)

**Decision.** The fault simulator and `compute_expected_output()` must never
share a configuration. `config/site_defaults.yaml` holds the *coarse* stack
(PVWatts DC + Faiman temperature + PVWatts inverter); the simulator is
configured separately with a detailed stack (SAPM/CEC + full loss tree + sensor
noise + parameter offsets).

**Why.** Handoff §5.1 closes circularity at the signature layer but leaves it
open at the model layer. If a pvlib-simulated plant is scored against the same
pvlib ModelChain with the same parameters, the expectation matches the truth
exactly, every injected deficit is trivially detectable, and accuracy goes to
~0.95 for a different but equally hollow reason.

**Consequence.** Prefer injecting onto *real measured* series wherever the tool
set allows it. Simulation is reserved for the DC-level telemetry (per-MPPT
currents) that no public dataset provides — an honest limitation for the README.

---

## 0009 — Retrieval metric changed from recall@10 (2026-07-25)

**Decision.** Report top-1 and MRR split by query class, not just "right chunk
in top 10 ≥ 0.90".

**Why.** With the deliberately small corpus chosen in §3.7(b), a dozen documents
chunked leaves top-10 recall near-guaranteed regardless of retriever quality.
A metric that cannot fail cannot teach anything, and the whole point of the
option-(b) design is to show BM25 winning exact-term queries and dense winning
paraphrased symptom descriptions. That separation shows up at top-1, not top-10.
The top-10 figure is still reported; it is just no longer the headline.

---

## 0010 — PVDAQ (NIST) replaces DKASC as the dataset (2026-07-25)

**Decision.** Primary dataset is **NREL PVDAQ system 4902, NIST_Ground_1**,
Gaithersburg MD — 270.7 kW, 2016–2017 at 15-minute resolution.

**Why the change.** DKASC was the agreed primary, but `dkasolarcentre.com.au`,
`developer.nrel.gov` and PVGIS are all blocked by this environment's network
policy. The OEDI data lake on S3 is reachable, and PVDAQ lives there.

**Why it is better anyway.** NIST Ground 1 carries **7 per-combiner DC current
channels**. That was the one thing I said no public dataset had, and its absence
was going to force `per_mppt_current_balance` and the inverter-fleet comparisons
onto simulation. They now run on real measurements. It also has POA + GHI +
module temperature + ambient + wind, a documented module (Sharp NU-U235F2, in
the CEC database) and inverter (PV Powered PVP 260 kW), fixed 20° tilt at 180°
azimuth, and full public metadata.

**What is lost.** DKASC's ~30 co-located arrays would have given a genuine
fleet-comparison baseline; NIST has one array with seven combiners, so
`compare_inverter_to_fleet` becomes `compare_string_to_array`. Desert soiling is
also gone — Gaithersburg is humid continental (Köppen Cfa), so soiling is a much
weaker real signal and mostly arrives by injection.

---

## 0011 — Two stacked panels instead of a dual-axis chart (2026-07-25)

**Decision.** Handoff §6.4 asks for power and irradiance on a dual axis. Built
as two panels sharing a time axis instead.

**Why.** With two y-scales the author chooses where the series cross, so the
reader cannot distinguish a real divergence from a chosen one. The teaching
moment the brief wants — "output halving while irradiance halves with it" — is
read off aligned peaks and troughs, which stacked panels give without the scale
trickery.

Chart colours are drawn from a palette validated for both themes on their own
surfaces (colour-vision separation, contrast, lightness band). Three categorical
slots is the hard cap: past three, no ordering clears the all-pairs floors.

---

## 0012 — Logger timezone recovered from physics, not assumed (2026-07-25)

**Decision.** The ingest recovers each logger's UTC offset by scanning every
half-hour candidate and keeping the one whose modelled clear-sky curve best
correlates with measured irradiance. NIST 4902 came back **UTC−5, r = 0.997,
margin 0.026** over the runner-up — Eastern *Standard* Time year round, no DST.

**Why.** PVDAQ does not record a timezone, and both usual guesses are wrong
somewhere: UTC is wrong for most systems and local-with-DST is wrong for the many
loggers that keep local standard time. An hour of error puts the modelled sun an
hour from the measured sun, and a healthy plant then appears to under-perform
every morning and over-perform every afternoon.

**One refinement that mattered.** Correlating across all weather caps out near
r = 0.85 and leaves a 0.007 margin between adjacent offsets — a cloudy day is dim
at every candidate, so it lowers them all equally and flattens the peak. Scoring
only the clearest days (ranked by curve smoothness, so the selection is
resolution-independent) lifts it to r = 0.997 with a decisive margin.

---

## 0013 — PR must mask numerator and denominator identically (2026-07-25)

**Decision.** `compute_pr` excludes any interval where power is missing, from
*both* the energy sum and the insolation sum, and reports
`data_completeness` alongside every result.

**Why.** Found by the physics, not by review. NIST 4902 loses AC power for 35
days in mid-2016 while the weather station keeps logging normally. Integrating
all of July's insolation against only the five surviving days' energy reads as
**PR = 0.098** — a 90% loss that never happened. With matched masks the same
month reads PR = 0.798 at 12.4% completeness, which is the honest statement.

This is why the interface reports completeness next to every ratio: a PR from
14% of the intervals is not the same claim as a PR from 98%, and a finding must
never present them identically.

---

## 0014 — Faults are injected onto measured data, never onto a simulated plant (2026-07-25)

**Decision.** Every injector perturbs the real NIST series. Nothing builds a
synthetic plant with pvlib.

**Why this is stronger than DECISION 0008.** 0008 required the simulator and the
expectation model to use *different* configurations. Perturbing measured data
removes the problem rather than managing it: there is no generating model for
the expectation model to be circular with. This only became possible because the
chosen dataset carries real per-string currents — a simulated plant would have
been unavoidable on a dataset without them.

---

## 0015 — Rules-engine baseline reported before the agent exists (2026-07-25)

**First numbers, on 20 cases (10 per split), 40% look-alikes:**

| | tuning | held-back |
|---|---|---|
| Overall accuracy (macro-F1) | 0.329 | 0.385 |
| False alarms on look-alikes | 0.000 | 0.000 |
| Correct "not enough evidence" | 0.000 | 0.000 |
| Missed real faults | 0.250 | 0.250 |

Distinct tool trajectories: 2. Unplanned-measurement rate: 0.000.

**Reading it.** The baseline fails two of the four v1 targets, which is the
point — a harness that cannot report failure cannot report success either. Its
zero false-alarm rate comes from being conservative, not accurate: it calls most
things healthy, hence the 25% missed-fault rate.

The **0.000 correct-abstention rate is structural, not a tuning problem.** The
engine has no way to represent "two causes survive and here is the test that
separates them", so it commits to `clipping` on the curtailment case. That is
the gap the agent has to fill, and it is now measured rather than asserted.

The agency metrics double as their own control: a known pipeline scores 2
trajectories and 0.000 unplanned measurements. If the agent scores the same,
it is a pipeline too — and the metric will say so.

---

## 0016 — PVDAQ has no fault logs; ground truth is injection-only (2026-07-25)

**Finding.** Searched the entire `pvdaq/` tree and the `2023-solar-data-prize/`
datasets for any table named fault / event / maintenance / outage / alarm / log
/ label / annotation. There are none. The bucket holds telemetry plus equipment
metadata and nothing else. A system's metadata JSON has exactly seven sections
(System, Site, Mount, Inverters, Modules, Meters, Other Instruments); the
`comments` field on 4902 is an empty string, and a full-text search of a
metadata blob for `fault|outage|maintenance|event|alarm|repair|downtime|label`
returns nothing.

**This corrects an earlier expectation.** PVDAQ was chosen partly as "the best
chance at real labelled events". It does not have them. The §5.2 row "real data,
real known events" is **zero**, and the README says so.

**Consequences.**

1. Ground truth is injection-only, which raises the stakes on injecting at the
   physics layer rather than the signature layer — it is the only truth the
   evaluation has.
2. Real-data-no-injection cases survive for anything verifiable from physics
   alone: summer temperature derating and a genuinely cloudy week need no
   maintenance log, because the temperature coefficient and the irradiance
   record *are* the label. Two such cases sit in each split.
3. The 35-day mid-2016 telemetry gap is a real, found, unlabelled event —
   discovered by the physics, not by a log. Worth promoting into the golden set
   as its own case rather than only being the motivation for a bug fix.
4. **Open:** string 7's share of DC current runs persistently below its even
   share in the untouched data. That is either a smaller combiner or a real
   long-standing fault, and without a maintenance log it cannot be settled —
   which is itself a fair illustration of the problem this project is about.

**Alternatives considered.** (a) A dataset with logs — none reachable from this
environment, and few exist publicly in any case. (b) Treating operator forum
posts or publications as labels — unverifiable and not reproducible. (c)
Accepting injection-only ground truth and saying so plainly in the README —
chosen.

---

## 0017 — pvlib primitives, not `ModelChain` (2026-07-25)

**Decision.** The expectation model uses pvlib for solar position, Ineichen
clear-sky, Perez transposition, airmass, extraterrestrial irradiance and the CEC
module database. The PVWatts DC and inverter equations are written out directly
rather than assembled through `pvlib.modelchain.ModelChain`.

**Why.** Handoff §7 step 1 says "pvlib ModelChain". Two reasons for the
deviation. `ModelChain` expects a complete `PVSystem`/`Array` specification and
runs the full weather-to-AC chain, whereas this model is driven by *measured*
POA and needs only the DC-temperature-inverter portion. And keeping those five
lines of arithmetic explicit makes the anti-circularity boundary auditable: a
reader can see there is no plant simulation in the file. Inside a `ModelChain`
call that is a much harder claim to check.

**Cost.** Losing `ModelChain`'s loss tree and its ability to swap DC models by
name. Neither is wanted here — the coarseness is deliberate.

---

## 0018 — A plain Python loop before LangGraph (2026-07-26)

**Decision.** The investigation loop is ~200 lines of explicit Python in
`src/agent/loop_plain.py`. LangGraph arrives at step 12 as a *port*, and the
gate on that port is that both produce identical golden-set outputs.

**Why.** Orchestration is the thing being learned. A framework hides exactly the
parts worth understanding: where state is mutated, what terminates the cycle,
what happens when a node returns something unusable, and which call is billed to
which tape entry. Writing it out first also makes the step-12 comparison a real
one — "did the framework buy anything?" is only answerable against a version
that already works.

**Cost.** No checkpointing, no streaming, no built-in retries. All three are
wanted eventually, and they are exactly the things the step-12 comparison should
weigh.

---

## 0019 — `was_planned` is derived, never self-reported (2026-07-26)

**Decision.** The router's output schema has no `was_planned` field. The
executor computes it: a call counts as planned when the tool is in
`planned_tools` **and** has not already run in this investigation.

**Why.** The unplanned-measurement rate is the sharpest single agency metric in
§5.5 — near zero means the planner is a sequencer and the "agent" is a pipeline
that narrates. A metric the subject reports about itself is not a measurement. A
model asked "was this in your plan?" is being invited to flatter itself, and a
test confirms that a router volunteering `was_planned: true` for an off-plan
tool is ignored.

The second clause matters as much as the first. Re-running a tool the plan
already spent — usually over a window narrowed because an earlier result pointed
at a date — is the plan being *extended by evidence*, not followed. Counting it
as planned would hide the exact behaviour the metric exists to detect.

`reason_for_choosing` is required on every call, not only on departures. A field
that appears only once the model has decided to go off-plan gets written to
justify the departure rather than to explain the choice.

**Alternatives.** (a) Ask the model and trust it — unmeasurable. (b) Count any
tool outside the *opening* plan as unplanned, so a critic-driven replan inflates
the rate — rejected: that measures replanning, not adaptation. Revised plans
accumulate into `planned_tools`, so only router-level departures register.

---

## 0020 — A provenance ledger on every tool result (2026-07-26)

**Decision.** `ToolResult.values` is a flat `{name: number}` map of everything
the tool measured. The synthesiser's prose is checked against the union of those
ledgers by `src/agent/grounding.py`, and a figure that is not in one is reported
as ungrounded.

**Why.** §5.6 targets "zero fabricated numerics", which is worth stating only if
a violation can be detected. Two allowances are deliberate: a factor of 100
either way (a fraction written as a percentage is unit presentation), and
integers up to 24 (how sentences count things — "two causes", "three days",
"09:00"). Arithmetic is *not* allowed: if the ledger holds an expected and a
measured energy and the prose states their difference, that difference is
flagged, because CLAUDE.md puts no LLM in arithmetic — the subtraction belongs
in a tool where it is deterministic and testable.

A parametrised test applies the same check to each tool's own summary, which
caught three tools quoting constants (`25 °C`, a recovery threshold, a
percentage bound) that were not in their own ledgers. The rule that fell out:
**any constant a tool prints must be in its ledger**, or the check is scoring
against an incomplete reference and the target is unenforceable.

---

## 0021 — Injector physics corrected: string channels move with the loss (2026-07-26)

**Decision.** `_scale_power` gained a `scale_strings` flag. Soiling, clipping and
curtailment now scale every per-string current and per-string DC power channel
along with DC and AC power. Shading gained `strings_affected` and shades a
contiguous group.

**Why.** Found by pointing step 4's measurement tools at step 2's golden cases,
which is what that exercise is for. Three defects, in increasing order of
seriousness:

1. **Soiling left every string current untouched** while cutting DC and AC
   power. Dust sits on the glass, upstream of everything electrical, so it costs
   every string the same fraction — and that uniformity is precisely the
   signature that separates soiling from a string fault. The case's own stated
   reasoning element ("affects all strings equally, so it is not a string
   fault") was vacuous: the strings were flat because the injector forgot them.
2. **Clipping and curtailment capped AC only**, leaving DC at full. That implies
   an inverter running at ~58% efficiency all midday, which no inverter does. It
   is a fingerprint of the simulator rather than of a power limit, and an agent
   could have learned it instead of the physics — a mild form of exactly the
   circularity CLAUDE.md exists to prevent. Both now route through one
   `_apply_power_cap`, so neither can drift from the other and become separable
   by a quirk of the code.
3. **Shading one string of seven at 40% depth for three hours** removed 1.2% of
   the window's energy — under the noise floor of every tool in the set. The
   case was unlearnable in principle and would have scored coin flips. A shadow
   is a physical object with an extent; it falls across a contiguous group of
   strings. Three of seven at 55% now removes ~5%, with a clear time-of-day
   signature and a per-string one.

**Consequence.** The golden set changed, so the step-3 rules baseline was
re-scored: **macro-F1 0.329 tuning / 0.611 held back** (was 0.329 / 0.385).
Correct abstention stays 0.000 on both, which is structural and expected.

**Note on the held-back split.** Held-back cases were inspected here to confirm
the *physics* of the fix, not to tune anything. The remedy was applied to the
generator — where it affects both splits symmetrically — and no split-specific
parameter was touched. The distinction is the discipline: change the generator,
never the split.

---

## 0022 — The curtailment case's ceiling raised from 150 kW to 200 kW (2026-07-26)

**Decision.** The canonical unresolvable case caps export at 200 kW against a
260 kW inverter, on windows whose clear days reach 240–255 kW.

**Why.** At 150 kW the plateau sat at 58% of nameplate. No inverter clips itself
that far below its own rating, so "clipping" was not a live explanation and the
pair the case exists to test was quietly *resolvable* — the case was not
measuring what it claimed. At 200 kW a limit is clearly being applied and both
owners of that limit remain plausible: an inverter holding a configured power
limit (common for grid-code compliance) and a grid operator capping export are
indistinguishable on every channel this plant exposes.

Surfaced by `check_ac_ceiling` reporting `plateau_as_fraction_of_ac_rating`,
which is the kind of thing a tool should report precisely so that a case built
on the assumption can be checked against it.

---

## 0023 — The soiling case runs over 45 days, not 14 (2026-07-26)

**Decision.** The soiling case's window is extended by 30 days.

**Why.** At a realistic temperate accumulation rate, a fortnight of soiling moves
the temperature-corrected performance ratio by less than the weather does — the
fitted trend on the tuning case came back at R² 0.02. Soiling is diagnosed over
months in the field, and a case that is unanswerable in principle teaches the
evaluation nothing. This is a fact about the domain, not a threshold to tune:
the fix is a window long enough for the physics to be visible, not a more
sensitive detector.

---

## 0024 — Interface vocabulary is translated at the API boundary (2026-07-26)

**Decision.** The trace keeps the precise internal term (`hypotheses`,
`still_standing`, `not_enough_evidence`, `adaptive`). The dashboard API renames
`hypotheses` to `possible_causes`, strips internal-vocabulary keys from the
step args it echoes, and the loop writes the critic's verdict to the tape in
plain words with hypothesis ids resolved back to the causes they stand for.

**Why.** CLAUDE.md allows `src/` and `docs/` the precise term and forbids it in
the UI. Putting the translation at the boundary keeps both true at once, and the
boundary is the only place that knows something is about to be rendered. The
jargon test scans template *source*, not just rendered output, which caught
`run.hypotheses` sitting in an Alpine expression — a file the browser loads even
though no user ever sees the word.

---

## 0025 — The knowledge layer is retrieved, never executed (2026-07-26)

**Decision.** `src/knowledge/` holds two YAML files — what each cause does to a
plant's telemetry, and what separates each pair of look-alikes. Nothing in the
package compares a signature to a measurement, scores a match, or returns a
cause. It loads, validates, and formats text. A loader validator rejects any
entry that reads as a decision rule.

**Why.** This is the sharpest circularity risk left in the project. If a
signature said "worst string deviation over 3% means a string fault" and any
code compared that number to a tool result, the accuracy figure would be
measuring whether `fault_signatures.yaml` agrees with `simulator/injectors.py`
— two files written by the same person in the same week. It would come out near
100% and mean nothing.

So the file describes *shapes and relationships*: "falls on some strings and not
others", "recovers after rain", "the ratio rises, which a real loss cannot do".
Physical constants are allowed ("roughly 0.4% per degree above 25 °C" is
physics); cut-offs are not. Four regex patterns enforce it at load time, and a
test confirms both that a rule fails to load and that a constant does not.

**Retrieved after planning, never before.** The planner does not see the
catalogue. If it did, it would enumerate whatever the catalogue contains and the
evaluation would again be scoring agreement between two files. The loop matches
the planner's own candidate causes onto signature keys, emits a `retrieval` step
on the tape, and hands the result to the router and synthesiser as evidence.

**Passed as an argument, not imported.** `investigate(..., knowledge=...)`
accepts an empty `KnowledgeBase`, which is the step 9 ablation: the identical
loop with retrieval removed. A layer that cannot be switched off cannot be shown
to be worth anything.

**Alternatives.** (a) Signature matching in code with a confidence score — this
is a rules engine, and the project already has one as a baseline. (b) Putting
the signatures in the planner's system prompt — same circularity, plus it makes
the ablation impossible. (c) No knowledge layer at all — defensible, and the
step 9 ablation may yet say so.

---

## 0026 — One vocabulary for causes across injector, golden set and knowledge (2026-07-26)

**Decision.** `sensor_drift` and `telemetry_gap`, singular, everywhere —
injector, golden case labels, knowledge keys, and the critic's look-alike
checklist.

**Why.** The knowledge base originally said `irradiance_sensor_drift` and
`telemetry_gaps` because both are more precise. A test comparing the golden
set's causes against the knowledge base's keys caught the mismatch. Left alone,
`cause_accuracy` would compare labels that can never match and would report a
lower number for a reason that has nothing to do with diagnosis. The failure is
silent, which is why it is now a test rather than a convention.

---

## 0027 — The diagnostic gets the whole record, not just the window (2026-07-26)

**Decision.** `MaterialisedCase` gained `full_record`: the entire ingested
series with the case's perturbation spliced into its window. That is what the
agent's `ToolContext` holds; the window is pinned by `start`/`end` on every tool
call.

**Why.** Found the moment `compare_to_trailing_baseline` and `weather_context`
were pointed at the golden set: both raised on every case. The context had been
built from the case *window*, so there was no month before it to compare against
and no other year of the same calendar weeks. Two of the eighteen tools were
unreachable, silently, and would have been scored as failures of the agent.

It is also the more faithful setup. A real investigation is asked about a
fortnight and has years of history to reach for; an engineer's first question is
almost always "was it like this last month?".

---

## 0028 — Golden set to 86 cases, composition chosen not emergent (2026-07-26)

**Decision.** 43 cases per split. Ten hand-written archetypes plus sweeps across
windows and severities: string faults from a full outage to a 25% partial loss,
shadows of varying depth and extent, soiling over 45-day windows, eight sensor
drifts from steep to nearly invisible, six telemetry gaps, and five
clipping/curtailment ceilings. Look-alikes are 41.9% of each split.

**Why.** Ten clean archetypes measure whether a detector recognises archetypes.
The interesting failures are marginal, and marginal cases exist only if severity
is varied on purpose — hence the 25% string loss and the −0.0015/day sensor
drift, both of which sit close to the noise floor by design.

The look-alike share is a floor that composition is chosen to satisfy, not a
number that fell out. Below 40%, a detector that alarms on any deficit scores
well and the evaluation teaches nothing.

A duplicate check runs over the built set: the first ceiling variant originally
reproduced the hand-written ceiling case exactly, which would have inflated the
split without adding anything to measure.

**Cost.** An agent run over both splits is 86 investigations. At a target median
of $0.15 per question that is roughly $13 per full evaluation, and the headline
metrics are reported as mean ± spread over N=3 runs, so a complete figure is
three times that.

**Rules baseline on the expanded set:** macro-F1 **0.285 tuning / 0.414 held
back**, false alarms 0.000 / 0.056, correct abstention **0.000** on both, missed
faults 0.375 / 0.500. The abstention column is structural: the engine commits to
the first rule that fires and has no way to hold two causes open. That is the
gap the agent exists to fill, and it is now measured over five unresolvable
cases per split rather than one.

---

## 0029 — What the critic is not allowed to decide (2026-07-26)

**Decision.** Three parts of `CriticVerdict` are computed and then overridden
onto whatever the model returned. The critic can be stricter than the model
wanted; it can never be looser.

1. **Unsupported claims** come from the deterministic grounding check, unioned
   with anything the model adds. A model asked to inspect its own arithmetic and
   pronounce it sound is not a check. If a figure is not in a tool's ledger it is
   unsupported, and no verdict can clear it.
2. **A look-alike counts as checked** only when a tool whose `discriminates` tag
   names it was actually run *and* the critic says it weighed it. The
   intersection, not the union: measuring without weighing is not consideration,
   and claiming without measuring is box-ticking — which is exactly what the
   checklist exists to stop. All seven items are reachable, and a test asserts
   it, because an unreachable item would make `accept` impossible.
3. **A verdict that fails the contract becomes `send_back`.** A critic that
   cannot produce a valid verdict has not reviewed the answer. Falling through
   to accept would be the precise failure this node exists to prevent.

`accept` is additionally blocked by any unsupported claim or any unchecked
look-alike — those are `CriticVerdict`'s own validators, and the node now feeds
them figures the model did not choose.

**Consequence, seen immediately in a test.** An answer built on a single
measurement is sent back however confident the review is, because one tool
cannot address seven look-alikes. That is the intended behaviour and it is the
main reason the cycle cap exists.

---

## 0030 — The router gets a third action: look it up (2026-07-26)

**Decision.** `action` is now `call_tool | look_up | stop`. On `look_up` the
router names two causes and gets back what is known about telling those two
apart, which joins the evidence the synthesiser sees.

**Why.** This is the promotion the router needed to be a full node rather than a
tool selector. Choosing to find out *what would separate* two candidates before
spending a measurement on one of them is a routing decision, and often the right
one: the lookup is free and can avoid a measurement that would not have decided
anything.

It is recorded as `was_planned=False` on the tape, because it never appears in
the plan and it is a genuine departure — the reason the router gives for it is
as much an agency signal as the reason for an off-plan measurement.

**The two bounds.** A lookup does not consume a measurement slot, so the inner
loop now carries two limits: one on measurements, which is what costs money, and
one on router turns, so a router that keeps asking for free lookups still
terminates. An unbounded number of free calls is still an unbounded loop.

---

## 0031 — The baseline measures through the same tools as the agent (2026-07-26)

**Decision.** `RulesEngine` no longer implements its own arithmetic. It takes a
`ToolContext` — the same object the agent gets — and calls the same eighteen
tools through the same registry, then applies explicit thresholds to their
ledgers.

**Why.** The comparison is the deliverable. If the baseline measured
differently, a gap between the two would be a gap in *measurement quality* and
would say nothing about reasoning. Measuring identically isolates the only
variable worth testing: what gets done with the numbers. It also means every
improvement to a tool improves both engines, so the baseline cannot go stale
and quietly become a straw man.

Its structural limits are untouched and are the point: a fixed sequence in a
fixed order, commit to the first rule that fires, no way to hold two causes
open.

**Two bugs this immediately exposed.**

*A failed measurement read as a passing one.* The engine defaulted a missing
completeness figure to 1.0, so a window where the performance ratio could not be
computed **at all** passed the data-quality check and was then diagnosed as a
string fault. Six telemetry-gap cases per split were mislabelled on that alone.
The gap injector blanks the power channel while the logger keeps writing rows,
so row coverage stays at 100% — `profile_data_quality` now also reports
`lit_interval_completeness`, which is the number the question actually needs.

*The first threshold fired on every case ever measured.* A per-string deficit
threshold of 0.10 sits below combiner 7's ~12% standing anomaly, so all 86 cases
came back as string faults and the false-alarm rate was 1.000. Raised to 0.22.

**On the tuning of that threshold.** A sweep on the tuning split says 0.18
maximises macro-F1. 0.22 was chosen instead: 0.18 buys 0.012 of overall accuracy
and *doubles* the false-alarm rate, which is the number the field cares about,
and it sits close enough to the standing anomaly that a slightly different
window would push it over. Recorded because "we picked the best threshold" and
"we picked the threshold that fails safely" are different claims.

---

## 0032 — The comparison is built to be losable (2026-07-26)

**Decision.** `eval/compare.py` scores both engines with the same `aggregate()`,
orders the headline metrics with the false-alarm rate and the correct-abstention
rate **above** overall accuracy, prints signed differences, and lists every case
the two answered differently with which engine got it right.

**Why.** A comparison that quietly favours the thing being evaluated is worse
than no comparison. Three specifics:

- **The headline is not accuracy.** A detector that alarms on any deficit scores
  well on clean faults and is useless in the field; only the look-alike column
  exposes that.
- **Differences are signed.** The table has to be readable when the agent loses.
  Tests assert the arithmetic in both directions.
- **The per-case disagreement list is the useful output.** An aggregate
  difference of a few points says nothing about *why*; the list of cases that
  moved says all of it. Winners are decided on **cause**, not category, because
  two causes in one category can lead to opposite actions — "wash the array" and
  "wipe the sensor" are both interventions and only one is on the plant.

**Both ablations are flags, not plans.** `--no-review` and `--no-knowledge` run
the identical loop with one layer removed. An ablation that cannot be run is a
claim rather than a measurement, and a test asserts both flags exist.

---

## 0033 — `docs/FINDINGS.md` states the agent has not been run (2026-07-26)

**Decision.** The findings document leads with the fact that no
`ANTHROPIC_API_KEY` was available, so every number in it is baseline, physics or
injector. The agent column is empty.

**Why.** A half-finished evaluation is exactly the kind of thing that gets
quietly presented as a whole one. The harness, the golden set, the metrics and
the comparison are complete and tested; the agent column is empty because it has
not been run, not because it is pending analysis, and the difference matters.

The document also records the bugs the evaluation found in itself, each of which
would have made a published accuracy figure meaningless. That section is
deliberately not in an appendix.

---

## 0034 — The findings store is an append-only log, and identity is not the text (2026-07-26)

**Decision.** Nothing is ever mutated or deleted. Every state change appends a
record and the current view is a fold over the log. A finding's identity is
`(scope, cause)` when settled and `(scope, sorted set of surviving causes)` when
not.

**Why.** Without lifecycle you get the same three findings shouted at you every
morning until you stop reading them, and a monitoring system nobody reads is
worse than none because it provides cover. So the store's real job is knowing
that the thing it saw today is the thing it saw yesterday.

Identity cannot be the wording: two runs of the same investigation produce
different prose, and matching on text would open a new finding every sweep.
Identifying an unsettled finding by its *set* of survivors means narrowing four
candidates to two correctly opens a new finding — that is a different claim.

**Two things the tests forced out.**

*A quiet sweep has to be recorded.* A sweep that writes nothing leaves no trace,
so counting distinct log timestamps meant a finding was only aged when some
*other* finding happened to be active. On a plant that had gone quiet, a fixed
fault would sit open forever. The miss count is now carried on the record, and
`watcher.py` tells the store about every sweep including the empty ones.

*Ageing records are not news.* "Still not seen, second sweep" is real history
and belongs in the log, but a what's-new-since-yesterday list padded with it is
one nobody reads, so `since()` excludes them by default.

**Resolve after three quiet sweeps, not one.** One is too eager — a cloudy day
can hide a real string fault from a detector. This is the knob that decides
whether the interface cries wolf or goes quiet on something real.

**Suppressed stays suppressed.** Re-raising a finding an operator has dismissed
is exactly how they learn to ignore the whole interface.

---

## 0035 — Detectors decide when to look, never why (2026-07-26)

**Decision.** `DeficitSignal` carries a scope, a window, a measured size and the
name of the measurement that produced it. It has no cause field and no field one
could be smuggled into. The diagnostic — rules engine or agent — is a separate
step.

**Why.** The sweep is the only thing that runs unprompted, so if it named causes
the Watcher would be a rules engine on a timer and the agent would be reduced to
writing up a conclusion already reached.

Detectors overlap deliberately: a window can trip three at once, and an
investigation starting from three independent signals is better grounded than
one starting from whichever fired first. Signals are ordered by how far past
their threshold they sit, so the ranking is stable rather than alphabetical.

**The trailing window ends the day before "now".** The current day is still
accumulating, and scoring a half-finished day produces a deficit every single
morning.

**The performance detector compares against the preceding month, not an absolute
level.** A plant that has run at 0.78 for two years is not developing a fault,
and an absolute threshold would open the same investigation every day forever.

---

## 0036 — The replay clock now starts inside the ingested record (2026-07-26)

**Decision.** `config/site_defaults.yaml` moves the clock start from
2019-01-01 to 2017-02-01.

**Why.** The data covers 2016–2017. A start date the data does not cover leaves
every trailing window empty, so the Watcher sweeps forever and finds nothing —
silently, with a zero exit code. A test now asserts the configured start falls
inside the record.

**What it found once pointed at real data.** Sweeping mid-2016 with
`--engine rules` picks up the **real 35-day telemetry gap** in this dataset and
carries it through the full lifecycle: new, ongoing, ongoing, then resolved
after three quiet sweeps. That is an unlabelled event found by the detectors
rather than by a label, and it is the only ground truth in the project that did
not come from an injector.
