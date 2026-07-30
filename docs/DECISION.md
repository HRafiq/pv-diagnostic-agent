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

---

## 0037 — Retrieval reports reciprocal rank, not "right document in top 10" (2026-07-26)

**Decision.** `RetrievalReport` detects when `k` is a large fraction of the
corpus and switches its headline metric. The ablation table drops the saturated
column rather than printing it.

**Why.** The intended headline is recall@10. On a 23-chunk corpus it returns 43%
of everything and scores **1.000 for every configuration**, including ones that
have learned nothing. Printing that table would produce the most flattering and
least honest number in the project, and it would be quoted.

The threshold is `k / N > 0.2`. Arbitrary, and stated as arbitrary; what is not
arbitrary is that some such guard has to exist, because the corpus is small by
circumstance and a future reader has no way to know that from the number alone.

**What the ablation says once the saturated column is gone.** Fusion earns its
place — hybrid beats either stage alone by ~0.09 of reciprocal rank. The
reranker adds 0.009, which on 18 queries is one query moving one position, so on
this evidence it is close to cost without benefit and stays only because the
corpus is too small to conclude either way.

**The dense stage is not semantic, and the queries say so.** No embedding API is
reachable from this environment and no local transformer is installed, so the
"vector" stage is character n-gram TF-IDF. It is a genuinely different signal
from BM25 — it matches sub-word overlap — but it will not match "the array is
dirty" to "soiling". The golden set includes paraphrase queries specifically so
that limitation appears as a number (0.657 against 0.917 on direct queries)
rather than being taken on trust.

---

## 0038 — Saved analyses are level 1 only, and cannot be saved unevaluated (2026-07-26)

**Decision.** An `AnalysisSpec` is a question plus a selection over existing
tools. Not a new tool, not new orchestration, not generated code. Saving is
refused unless every named tool exists and every cited golden case exists.

**Why.** `golden_case_ids` being non-empty is already a model validator; this
adds that the ids are real. "Evaluated against case G-999" is unevaluated with
extra steps, and an agent factory that produces unevaluated agents defeats the
whole point of the project — it manufactures things that look like diagnostics
and have never been scored.

Level 2 (new tools) and level 3 (new orchestration) are deliberately absent. Each
would need its own evaluation for every entry, and there is no honest way to
build that at this scale.

---

## 0039 — LangGraph is kept, and checkpointing is why (2026-07-26)

**Decision.** Both loops ship. `loop_plain` is the reference implementation and
what the tests are written against; `loop_graph` is the port, gated by sixteen
exact-equality tests in `tests/test_langgraph_port.py`.

**Why the port at all.** Writing the plain loop first (DECISION 0018) made this
answerable. The full argument is in `docs/LANGGRAPH_TRADEOFF.md`; the short
version is that the declared graph and named branches are real but modest gains,
the restated bound and the copied state are real but modest costs, and
**checkpointing is what tips it** — an 86-case evaluation that dies at case 60
currently starts again from case 1.

**The bug the equivalence test caught immediately.** LangGraph copies state
between nodes rather than threading one object through, so the port read its
local `AgentState` back after `invoke()` and reported an untouched run: no tools
called, no cycles used. Every equivalence test failed at once. In a hand-written
loop that bug does not exist to be made, and without the plain loop to compare
against it would have been invisible — the graph would simply have produced
plausible-looking, wrong agency metrics.

That is the case for having written both, in that order, more than any argument
about ergonomics.

---

## 0040 — `.gitignore` patterns are anchored, and a test enforces it (2026-07-26)

**The bug.** `.gitignore` carried a bare `findings/` line, written for the
runtime findings output directory at the repo root. A directory pattern with no
leading slash matches at *every* depth, so it also matched `src/findings/`.
`src/findings/build.py` and `src/findings/store.py` were never committed —
through step 8, step 9, step 10, step 12 and step 13, across five pushes.

**Why nothing caught it.** Every local signal was green. `git status` was clean,
because the files were ignored rather than untracked. The full suite passed,
because the files were on disk. The pushes succeeded. The repository was broken
only for someone cloning it, which is exactly the population that cannot report
the failure until they hit it — here, a fresh clone that failed to import
`src.findings.build` while `python -m pip list` showed every dependency present.

**Decision.** Two changes:

1. Project-specific directory patterns are anchored to the repo root —
   `/findings/`, `/traces/`. The upstream Python `.gitignore` block is left
   alone; its bare `build/`, `lib/`, `dist/` entries carry the same hazard and
   are now covered by the guard below rather than by editing a vendored file.
2. `tests/test_repo_tracking.py` shells out to `git check-ignore` over every
   source file under `src`, `simulator`, `eval`, `dashboard`, `tests` and
   `config`, and fails if any is excluded. A second test catches the adjacent
   case — a file that escapes `.gitignore` but was never `git add`ed, inside a
   package that is otherwise committed.

**Alternative considered and rejected.** A blanket `!/src/**` negation at the
end of `.gitignore`. It would have re-included `src/**/__pycache__` too, and it
fixes the symptom in one directory while leaving `eval/` and `dashboard/`
exposed. The test covers all of them and names the offending pattern in its
failure message.

**The general lesson, which is the reason this gets an entry.** The project's
other mechanical guard — `test_no_wall_clock.py` — exists for the same class of
defect: a mistake that produces no error, only a quietly wrong result. Ignored
source files belong in that class. Green tests say the code on this disk works;
they say nothing about whether that code is what the repository contains.

---

## 0041 — No test reads the ingested dataset from disk (2026-07-26)

**The bug.** Four tests — three in `test_watcher_cli.py`, one in
`test_experiments.py` — began by loading system 4902 out of `data/raw/`. That
directory is gitignored and the ingest is a download, so the tests passed on a
machine where `python -m src.data.cli ingest` had been run and failed on every
other one. Found the same way as DECISION 0040: a fresh clone on a second
machine.

**Why it was wrong on its own terms, not just inconvenient.** A unit test states
a property of the code. These stated a property of one laptop's filesystem. The
three watcher tests assert CLI plumbing — the clock advances, the window is
printed, `--dry-run` writes nothing — and none of that needs eight megabytes of
measured irradiance. They reached for the real dataset only because
`sweep_once` had no way to be pointed anywhere else.

**Decision.**

- `watcher` gains `--data-dir`. Useful beyond the tests — a sweep can run
  against an alternate ingest — and it is what lets the tests supply their own
  plant instead of borrowing the machine's.
- `tests/conftest.py` gains `write_ingested_plant()` and an `ingested_plant`
  fixture: the existing synthetic frame written to a tmpdir in the ingest
  layout, parquet plus manifest. It is a *test article* for CLI plumbing, and
  the file's opening note still holds — nothing scored ever runs on it.
  Evaluation cases remain real measured data with physics-level injections.
- The failure path is now asserted rather than merely suffered:
  `test_a_missing_ingest_is_reported_not_raised` pins that a sweep with no data
  prints the remedy and exits 1.

**A live-API hazard found while fixing it.** The experiments test asserted only
`main(...) == 1`. It passed for the wrong reason on a clean machine — no
ingested data, not no key. Asserting on the message instead exposed the real
problem: `Settings.from_env()` calls `load_dotenv`, so deleting the environment
variable does not produce a keyless run once a developer has a `.env`. On such
a machine that test would have launched a live 43-case evaluation against the
paid API. The test now patches `Settings.from_env` itself.

`_cmd_experiments` was also changed to require the key *before* loading cases.
Four ablations over 43 cases each load a plant and materialise an injection
before they reach a client, so a missing key surfaced minutes in, underneath a
header that looked like a run in progress. `test_the_key_is_checked_before_any_case_is_loaded`
pins the ordering.

**And the structural fix: CI.** `.github/workflows/ci.yml` runs ruff, mypy and
the suite on 3.11 and 3.12 from a clean checkout with no data directory and no
API key. Both this and DECISION 0040 were invisible locally and would have been
caught on the first push. A job step fails the build if `data/raw/` ever
contains a parquet file, so the tests cannot quietly reacquire the dependency
they just shed.

---

## 0042 — uv and a committed lockfile; pyarrow was never declared (2026-07-26)

**The bug.** `pyproject.toml` never listed `pyarrow`. pandas needs an engine to
read or write parquet and depends on neither pyarrow nor fastparquet itself,
and every ingested system is stored as parquet — so `src/data/ingest.py`,
`load_plant`, the Watcher, both evaluation engines and the dashboard all sit on
top of a dependency the project did not declare. It was pre-installed in the
development container (`Required-by:` empty — nothing pulled it in), so it
worked there and raised `ImportError` on every clean install.

**Why it took three attempts to find.** This is the third bug in a row that was
invisible locally: uncommitted source (0040), tests reading a gitignored
dataset (0041), and now an undeclared dependency. Each time the development
environment had state a fresh clone does not, and each time a green local suite
was read as evidence about the repository. It is not; it is evidence about one
disk.

Note that an import scanner would **not** have caught this one. Nothing in
`src/` imports pyarrow — pandas loads it at runtime inside `to_parquet`. The
only check that finds it is installing from the declared dependencies alone and
running everything. That is a property of the *process*, not of any test.

**Decision.**

- `pyarrow==19.0.1` is a core dependency. Stated in `pyproject.toml` with the
  reason, because "pandas needs it" is not obvious from any import.
- Dependencies are managed with **uv** against a committed `uv.lock`. Pinned
  versions in `pyproject.toml` already fixed direct dependencies; the lockfile
  fixes the transitive ones too, so a clone in six months resolves what this
  one did. `uv sync --extra dev` is the documented setup.
- `tests/test_data_layer.py::test_the_parquet_engine_is_installed` round-trips
  a frame through `load_dataset`. A missing engine now fails as one named test
  rather than as three ImportErrors inside fixture setup.
- `tests/test_repo_tracking.py` gains `ROOT_FILES`, which pins that `uv.lock`,
  `pyproject.toml`, `.gitignore` and the CI workflow are present, un-ignored
  and tracked. The stock Python `.gitignore` ships a commented-out `uv.lock`
  line; uncommenting it would silently return the project to unlocked
  resolution, which is DECISION 0040 in a new costume.
- CI runs `uv lock --check`, so a dependency added to `pyproject.toml` without
  re-running `uv lock` fails the build instead of installing fine on the author's
  machine and being missing for everyone else.

**Why CI at all — the point of these three entries.** The project has strong
guarantees about *reasoning*: the numeric grounding check, the critic's
structured verdict, the anti-circularity rules, the wall-clock scan. It had no
guarantee at all that the thing being reasoned about was installable. CI is a
machine that starts from nothing on every push — clean checkout, no data
directory, no API key, no packages except what the lockfile names — on both
ends of the `requires-python` range. All three bugs above would have been
caught by its first run, in minutes, instead of one at a time by a person on a
laptop.

---

## 0043 — A fetch failure is not an archive gap (2026-07-26)

**How it surfaced.** A second machine's ingest produced 58,752 rows ending
2017-09-04. The development machine's had 70,176 ending 2018-01-01. Same
command, same system, same years. The short record then crashed the rules
engine inside `weather_context` with a pydantic ValidationError — an error
about NaN, thirty cases into an evaluation, with nothing pointing at the
dataset.

**The root cause.** `_read_day` caught `HTTPError`, `URLError` and
`TimeoutError` in one place and returned `None` for all of them. A 404 and a
dead resolver were recorded identically, as `missing_days`. The machine in
question had been failing DNS lookups minutes earlier. Roughly four months of
days silently failed to fetch, were logged as holes in the archive, and the
command printed a success summary.

These are not the same fact:

- **404** — the archive does not have that day. A property of the dataset,
  identical for everyone, and legitimately drawn in the gap calendar.
- **network failure** — we could not ask. A property of one download attempt on
  one machine. Nothing was learned about that day at all.

**Decision.**

- `_read_day` returns a status: `ok`, `gap`, or `failed`. 404 is a gap and is
  not retried, because a day the archive lacks will not appear on a second
  request. Everything else — DNS, timeout, connection reset, 5xx — is retried
  three times with 1/2/4s backoff and then reported as `failed`.
- `ingest_system` **refuses to write anything** if any day failed, unless
  `--allow-partial` is passed. A short record is not a smaller version of a
  complete one: every measurement in this project is relative to a trailing
  baseline or to the same calendar span in another year, so dropped days do not
  degrade the evaluation, they silently change what it measures.
- The check runs *before* the empty-record check. A machine that is simply
  offline fails every day, and "no PVDAQ data for system 4902" reads as a claim
  about the archive when the truth is a claim about the network.
- The manifest records `missing_days` and `fetch_failures` separately. Only the
  first belongs in a reproducibility record.

**The second bug, found by the first.** `weather_context` computed
`day_to_day_spread` as `Series.std()`, which uses ddof=1 and returns NaN — not
an exception — on a single element. Any window holding one day produced a NaN
ledger value. It had been there since step 5 and was invisible only because the
development record ran four months longer, so no case window ever landed near
the end of it.

The fix omits the key rather than raising: the spread is genuinely undefined on
one day, but the mean insolation and the seasonal comparison are still real, so
discarding the whole measurement would be its own dishonesty. An absent key
reads as "not measured"; a zero would read as "perfectly steady", which is the
opposite of what one day tells you. A caveat says so in words.

A third bug fell out of the same investigation: a record whose irradiance
channel reads zero throughout divided by a zero seasonal norm in the summary
string — guarded in the ledger, unguarded in the prose. It now raises
`ToolError`, because there is no honest statement to make about weather from a
dead pyranometer.

**The guard.** `tests/test_tool_degenerate_windows.py` runs all eighteen tools
over six windows a real record can genuinely produce — one day, two intervals,
the tail of the record, night only, every channel constant, irradiance
identically zero — and asserts each either returns finite values or raises
`ToolError`. Never a ValidationError, never a NaN. It reproduced the reported
bug five ways before the fix and found the divide-by-zero, which nobody had
reported. The other seventeen tools were already honest.

**And a legibility fix.** With the NaN repaired, a truncated record fails
differently: `materialise` raises "case G-007: window has no data", one case at
a time, deep in a run. `eval.runner` now checks every case window against the
record before the first case and reports the whole set at once — which cases,
what the record actually covers, and the command to fix it.

**The pattern across 0040–0043.** Four bugs, all invisible on the machine that
wrote the code, all found by a second machine: uncommitted source, tests
reading gitignored data, an undeclared dependency, and a silently truncated
download. Each was a case of local state standing in for the repository. CI
covers the first three. This one it cannot — a network failure is not
reproducible on demand — so the defence is that the ingest now refuses to
produce the bad state at all.

---

## 0044 — The pinned SDK predated the API the code targets (2026-07-26)

**The bug.** `pyproject.toml` pinned `anthropic==0.62.0`. Every request in
`src/agent/llm.py` sends `output_config`, which carries both the `effort` level
and the structured-output `format`. That parameter did not exist on
`messages.create` until **0.77.0** — fifteen releases later. The first real
agent run, with an API key finally available, died thirty seconds in on
`TypeError: Messages.create() got an unexpected keyword argument 'output_config'`.

**Why nothing caught it.** Because nothing had ever called it. The project was
built without an API key, so the entire LLM path — planner, router,
synthesiser, critic — had never executed once. `ScriptedClient` covers the
orchestration with no network and no key, which is the right design and is why
567 tests could pass over thirteen build steps while the real client was
unrunnable. A scripted client is the one thing that cannot check whether the
*real* client's payload is valid.

The request code was never wrong. It was written against the current API, and
`docs/DECISION.md` 0004 records the reasoning for using `effort` instead of the
removed `temperature`. The pin was simply frozen at a version that predated the
API the code correctly targeted, and no test related the two.

**Decision.**

- `anthropic` is pinned to `0.120.0`, with `>= 0.77.0` recorded inline as a
  hard floor and the reason stated — "pandas needs an engine" (0042) and this
  are the same failure: a dependency whose necessity is invisible from the
  import graph.
- The payload construction moved out of `AnthropicClient.complete` into a pure
  `request_kwargs(cfg, system, user, schema)`. Not a refactor for its own sake:
  it is what makes the request checkable without a key, a network call, or a
  cent of spend.
- `tests/test_llm_request.py` checks every `(profile, node)` pair in
  `config/models.yaml` against the installed SDK's own signature and types —
  every parameter is accepted; `effort` and `format` are nested inside
  `output_config` rather than top-level; the removed `temperature` / `top_p` /
  `top_k` are absent; `thinking` is `adaptive` or `disabled` and never carries
  `budget_tokens`; and the configured effort levels are read from the SDK's
  `OutputConfigParam` rather than restated, so a level the API adds or drops
  surfaces on the next bump instead of as a 400.

  Two of the tests guard the guard: one fails if `messages.create` ever grows
  `**kwargs` (which would make an unknown-parameter check meaningless), and one
  asserts an obviously-fake parameter name is rejected.

- Verified by reinstalling `0.62.0` and confirming the suite reproduces the
  reported `TypeError` as a named assertion failure, then restoring the pin.

**The pattern this closes.** 0040 through 0043 were all local state standing in
for the repository. This one is different and worth separating: it is an
**untested path**, not an untested environment. CI cannot fix it, because CI
has no API key either — and it should not have one. The fix had to be a test
that validates the request *without* making it. Where a path cannot be executed
in CI, the obligation is to find the part of it that can be checked offline and
check that, rather than let "no key available" stand in for "no coverage
possible."

---

## 0045 — Structured outputs take a subset of JSON Schema (2026-07-26)

**The bug.** With the SDK pin fixed (0044), the request reached the API and was
rejected:

```
400 output_config.format.schema: For 'array' type, 'minItems' values other
than 0 or 1 are not supported (got: [2, 5])
```

`PLAN_SCHEMA` set `minItems: 2, maxItems: 8` on the planner's hypotheses list.
Structured outputs accept a *subset* of JSON Schema: array-length bounds beyond
0/1, numeric bounds, string-length bounds, `pattern` and `multipleOf` are all
rejected. `SYNTHESIS_SCHEMA` carried three more (`minimum`/`maximum` on
confidence, `minimum` on energy at stake) and would have produced the next 400
immediately after the first was fixed.

**Why deleting the constraints was not an option.** `minItems: 2` on the
hypothesis list is not decoration — a "differential diagnosis" that enumerates
one cause is a guess with extra steps, and the whole project rests on the
distinction. A confidence outside [0, 1] is meaningless. The constraints had to
survive; only their enforcement point could move.

**Decision.** `sanitize_schema()` strips what the API rejects and appends each
removed constraint to the field's `description`, so the model is still *told*
("At least 2 entries. At most 8 entries."). `violations()` re-checks the same
constraints against the parsed reply, and `AnthropicClient.complete` raises when
one is broken. The requirement is unchanged; it is now enforced in this file
rather than upstream, and stated in both places.

`minItems` of 0 or 1 **is** accepted, so it stays on the wire and stays
machine-enforced — `planned_tools` keeps its `minItems: 1`.

**The guard.** `tests/test_llm_request.py` walks every schema the agent sends
and fails on any keyword the API rejects, checks that sanitising removes only
constraints (never a property, type or enum), that a removed constraint still
appears in the description, and that `violations()` catches the reply that
breaks it. Verified by disabling the sanitiser and confirming it reproduces the
reported 400 as an assertion — which also surfaced the synthesizer's three,
before they cost a second run.

## 0046 — One bad case must not destroy the other forty-two (2026-07-26)

**The problem this session made unmissable.** A 43-case agent evaluation costs
real money and takes a long time. Both bugs above surfaced as an exception out
of `investigate()` on the *first* case, which propagated through
`run_agent_engine` and aborted the entire run. Twice. Had either failed on case
40 instead, thirty-nine successful investigations — and what they cost — would
have been discarded with them.

**Decision.** `run_agent_engine` catches per case. A failed investigation is
recorded as what it is — an answer the agent could not produce: unsettled, no
category, no tools called — and stays in the denominator, because dropping it
would flatter the engine by removing its failures from the score. The run then
continues. At the end the failures are listed explicitly, with the reason, so a
reader can tell "the agent answered badly" from "the agent did not answer".

`BudgetExceeded` is deliberately **not** caught. It means the loop is not
terminating, which is a defect in the agent rather than a bad case, and
swallowing it would repeat the same runaway forty-three times.

**Note on what this is not.** Catching broadly around a whole investigation is
usually a smell — it can hide defects behind a tidy score. It earns its place
here only because the failure is *reported by case ID with its exception text*
and *counted as a failure*, never silently absorbed. A quiet `except Exception:
pass` in this position would be strictly worse than the crash it replaced.

---

## 0047 — Objects must be closed, and a 400 is never one case's problem (2026-07-26)

**The bug.** After 0045, the next call returned:

```
400 output_config.format.schema: For 'object' type, 'additionalProperties: true'
is not supported. Please set 'additionalProperties' to false
```

`ROUTE_SCHEMA`'s tool-argument object was deliberately open — it holds whatever
arguments the chosen tool takes, which differ per tool. Structured outputs
cannot express that: every object must carry `additionalProperties: false`.

**Why the 0045 guard missed it.** That test checked what a schema must *not*
carry — rejected keywords. This is the complement: what it must carry. Two
different failures, and only the first had a guard. The lesson is narrow and
worth stating: a validity checker written from an error message covers that
error message.

**Decision.**

- The router's `args` object is built from the tool registry — the union of
  every tool's argument fields, all nullable, all listed in `required`, and the
  object closed. This is **better than the open object it replaces**: the
  router now sees the real argument vocabulary instead of being free to invent
  a name that `run_tool` would reject downstream. `_clean_args` drops the nulls
  before a tool sees them, so "null" and "not supplied" mean the same thing.
  A test asserts the schema's field set equals the registry's, so the two
  cannot drift into a silent capability loss.
- `sanitize_schema` now sets `additionalProperties: false` on every object it
  passes. Enforced structurally rather than left to each schema author, because
  the failure is a 400 on a request already paid for in latency and money.

## 0048 — Per-case isolation was right; applying it to a 400 was not (2026-07-26)

**What went wrong with the fix from 0046.** Isolating failures per case is
correct for a bad case. It is exactly wrong for a malformed request: the run
produced **forty-three identical 400s**, one round trip each, and the only
informative one was the first. Isolation turned a fast failure into a slow one.

**Decision.** `is_systemic_request_error()` distinguishes the two. A 400
`invalid_request_error` — including one wrapped by a node on its way up — means
the *request* is malformed, which is a defect in this codebase and identical
for every case; the run stops immediately and says so, naming the case and the
error and stating that the remaining cases would reproduce it. Rate limits,
overloads and timeouts stay isolated per case, because the next case may well
succeed.

**The principle.** Continue past what varies between cases; stop on what does
not. Retrying a deterministic failure N times is not resilience, it is N times
the cost for one bit of information.

---

## 0049 — Why a reply ended is part of the reply (2026-07-26)

**The bug.** The first run that got through the plumbing failed with:

```
ValueError: synthesizer was asked for structured output but returned
unparseable text: '{"settled": true, "category": "not_the_plant", "cause":
"String 3's current channel is a stuck/frozen sensor reporting a fake zero ...
```

The JSON was not malformed. It was **cut off mid-sentence** — the synthesiser
hit its 8000-token cap. `max_tokens` bounds thinking *and* the reply together,
and a synthesis at `effort: high` with adaptive thinking, a title, a summary, a
full answer, an evidence ledger and candidate causes does not fit in 8000.

The message was actively misleading. It blamed the model for bad JSON, which is
the kind of error that sends someone to rewrite a prompt for an afternoon. The
defect was a number in a config file.

**Decision.**

- `complete()` reads `stop_reason` before parsing. `max_tokens` raises
  `TruncatedReply` naming the node, its cap, and where to change it; `refusal`
  raises `RefusedReply` with the safety category. Both subclass `ValueError`, so
  existing handling still catches them, but the message says what happened.
  Genuinely malformed output still reports as unparseable — a test pins that the
  new check does not swallow it.
- Every node's `max_tokens` is raised: synthesiser 8000 → 32000, planner and
  critic to 16000, with the `quality` profile scaled the same way. **A ceiling
  costs nothing when it is not reached** — billing tracks tokens generated, not
  tokens allowed — so a tight cap buys no saving and risks exactly this. The
  original numbers were sized by guess, because nothing had ever run.
- A test asserts a floor per node, so the caps cannot quietly drift back down.

**The pattern, one more time.** Every failure in this sequence — 0044 through
0049 — is a first-execution failure of a path that thirteen build steps never
ran. The stale pin, the rejected schema keywords, the open object, and now the
token budget were all invisible to a suite that mocks the model. That is not an
argument against mocking it: `ScriptedClient` is what makes the orchestration
testable at all. It is an argument that **the parts of a request that can be
validated offline should be**, and that a config number nobody has exercised is
a guess until a run says otherwise.

---

## 0050 — Live progress, hooked at the trace writer (2026-07-26)

**The problem.** An agent evaluation printed nothing until a case finished. On
a run where every case was going to fail the same way, that meant three or four
minutes of silence before the first line of evidence — and no way to tell a
working run from a hung one.

**Where to put it.** Not prints scattered through the nodes. Every step already
passes through one place — `TraceWriter.write`, which the dashboard tails for
the same reason. `TraceWriter` gains an optional `on_step` callback, `investigate`
and `investigate_with_graph` take and forward it, and `eval/progress.py` decides
how to render.

The split matters: `src/` stays UI-agnostic (CLAUDE.md), emitting steps and
saying nothing about display. `eval/` prints. The dashboard makes a different
choice from the same data. Adding this to `src/` as a print would have been
three fewer lines and a rule broken.

**Two things it deliberately does.**

- Exceptions from the callback are suppressed, in both loops and the writer. A
  progress display runs inside a paid evaluation; a bug in a *view* must never
  take down the run it is describing. A test asserts the step is still recorded
  when the display raises.
- The `--quiet` counter checks `isatty()`. `\r` overwrites on a terminal and
  concatenates into one unreadable line in a CI log or a redirect, so off a
  terminal it goes quiet and only the per-case summaries remain.

**Two checks the type system and the model made for me.** mypy rejected the
first label map because it invented step kinds (`route`, `review`) that
`StepKind` does not define — a label for an impossible kind is dead code that
reads as coverage. And `TraceStep`'s own validator rejected a test fixture for
an unplanned step with no `reason_for_choosing`, which is the rule that makes
agency visible rather than asserted. Both are the project's guards working on
new code written months later, which is the whole point of having them.

**Also fixed in passing.** `investigate_with_graph` had to take the parameter
too — the LangGraph equivalence test compares signatures, and a port that
cannot show progress is not the same thing being compared.

---

## 0051 — Three failures from the first run that reached the agent (2026-07-26)

The first run where the plumbing held long enough to watch the agent think
exposed three things. Two were regressions from the previous two entries; one
was a real defect in the agent that had never been reachable before.

**1. Raising `max_tokens` required streaming (my regression, 0049).** The SDK
refuses a *non*-streaming request whose `max_tokens` it estimates could run past
ten minutes — an idle HTTP connection would drop first — and raises before
sending anything:

```
ValueError: Streaming is required for operations that may take longer than
10 minutes.
```

Raising the ceilings crossed that threshold on every node, so every case failed.
`AnthropicClient` now streams and takes `get_final_message()`, which returns the
same object `create()` would; streaming removes the limit on how *long* a reply
may take without lowering how *long* it may be.

Two follow-ons. The test doubles stubbed `messages.create`, so they passed while
the real path was broken — they now mirror the streaming shape, and a test
fails if `create` is used at all. And `is_systemic_request_error` only knew
about `BadRequestError`; this is a plain `ValueError` raised client-side, so it
slipped the net and failed all forty-three cases one at a time. It is now
recognised, and the run stops on the first.

**2. The shared argument vocabulary needed narrowing (my regression, 0047).**
Closing the router's `args` object meant the router chooses from the union of
every tool's fields. Tool argument models are `extra="forbid"`, so a field
belonging to a different tool is a hard validation error — and the trace showed
it repeatedly: "could not take this measurement — invalid arguments".

`narrow_args` drops fields the chosen tool does not accept, at the router
boundary. **Not** inside `run_tool`, which stays strict: a wrong argument passed
from code must still fail loudly, and a test pins that it does. What is dropped
is only the cost of sharing one vocabulary across eighteen tools.

**3. The router could loop on lookups (a real defect).** Case G-007 spent
*seventeen consecutive calls* asking what separates the same two causes. The
mechanism: a lookup whose answer is already in the brief adds nothing the router
can see, and the `look_up` branch costs an LLM call but deliberately does not
count against the per-cycle measurement cap — so nothing stopped it but the
per-investigation budget, minutes and dollars later.

Fixed on both sides at once. An unproductive lookup is counted; the router is
*told*, in the brief, that this lookup returns nothing new and it should measure
or stop; and after three the cycle ends and the synthesiser answers from what
was actually measured. An honest "not enough evidence" is a better outcome than
a budget drained in a loop. The LangGraph port gets the same guard as a labelled
transition, because the equivalence test compares the two run for run.

**What the run showed that was not a bug.** The trace is worth reading on its
own terms: the agent enumerated six or seven candidate causes per case, checked
data quality and weather before performance, went to per-string current when the
whole-plant ratio was ambiguous, and — on G-001 — took an *unplanned* lookup to
separate `string_outage` from `shading`, settled, was sent back by the critic,
and re-planned with three narrower causes. That is differential diagnosis, and
it is the first direct evidence the design works. None of it was scoreable yet,
because every case died before the answer was filed.

---

## 0052 — The critic could not accept (2026-07-26)

**The evidence.** The first four scored cases produced **ten reviews and zero
accepts**. Every case ran to the four-cycle cap. Ground truth against what the
agent said in its *first* cycle:

| case | truth | cycle-1 answer | final | outcome |
|---|---|---|---|---|
| G-001 | fault / string_outage | correct | not enough evidence | **right turned into wrong** |
| G-002 | fault / string_outage | correct | correct | 3× the cost, nothing changed |
| G-003 | fault / shading | correct (`shading`) | correct | 3× the cost, nothing changed |
| G-004 | recoverable / soiling | wrong | moved to soiling | **the critic helped** |

The critic is not the villain — on G-004 it pushed a wrong string-fault answer
onto the right soiling answer, which is exactly its job. The defect is that it
could not *stop*.

**The cause.** `accept` was gated on the intersection of look-alikes *measured*
and look-alikes the critic *named*, in two places: the verdict override in
`review()`, and `CriticVerdict`'s own validator. A reviewer looking at a string
fault names the three or four look-alikes that bear on it; it does not recite
all seven. One unnamed item forced `send_back` however good the answer was.

`test_accept_survives_a_clean_answer` never caught it because the fixture
defaults `lookalikes_considered` to the entire checklist — it recited all seven
on every call, which is precisely what a real critic does not do.

**Decision.** The critic's list is ignored entirely; the verdict is gated on
what the tool registry says was measured. The reasoning that justified the
intersection — "a model will happily say all seven" — is correct and led to the
wrong conclusion: **a claim nobody can verify carries no information, so
intersecting with it cannot remove a false positive, only a true one.** The
measurement is the fact, computed from `spec.discriminates`, and it cannot be
inflated by anything the model says. The guarantee is unchanged.

Checked against the observed traces: G-001 cycle 1 measured 7/7 (its correct
answer would now be accepted); G-002 measured 7/7 (two cycles and ~$0.85
saved); **G-003 cycle 1 measured only 5/7** — missing curtailment and seasonal
derating, which it went on to measure in cycle 2. That send-back was
legitimate, and it still happens. The critic keeps its teeth.

## 0053 — A cycle that measures nothing cannot change the evidence (2026-07-26)

G-001 spent cycles 3 and 4 on `plan -> look up -> answer` with no measurement
between them — roughly five minutes and half the case's budget re-reading the
same ledger and being rejected for the same reasons. The critic was right each
time; the loop was wrong to ask again.

Both loops now end the investigation when a review cycle took no measurement,
and say so in `stopped_because`. A cycle that *did* measure still replans — a
test pins that, because a guard that turned every `send_back` into a stop would
remove the critic's only means of correcting an answer, which is what rescued
G-004.

## 0054 — `cause` is a label; the explanation goes in `answer` (2026-07-26)

`CaseScore.cause_correct` is exact string equality against a canonical token
(`string_outage`). The synthesis schema left `cause` as free text, so the model
used the field both ways — `shading` on one case, *"A low-angle morning
obstruction (trees, structure, or an adjacent row) is shading strings 1, 2
and…"* on the next. The first scores; the second scores zero however right it
is.

The rules engine picks from a fixed vocabulary, so it was unaffected. **The
rules-vs-agent comparison was being decided by formatting** — which would have
been published as a finding about reasoning.

`cause` is now an enum, read from the knowledge base's signature names rather
than restated, so the vocabulary cannot drift from the signatures the agent is
shown. A test asserts every `expected_cause` in the golden set is in it: ground
truth naming a cause the agent has no way to say is a case that cannot be won.
`answer`, `summary` and `title` stay free text — the explanation was never the
problem, the label doing two jobs was.

---

## 0055 — A review is not a veto (2026-07-26)

**What the traces showed.** DECISION 0052 fixed a real bug and missed the
binding one. Reading the actual critic verdicts from a G-001 trace:

| | `unchecked` | `unsupported` |
|---|---|---|
| V1 | `snow_or_dust_event` | 2 prose observations |
| V2 | — | 2 prose observations |
| V3 | — | — |
| V4 | 3 items | 2 prose observations |

Every send_back carried at least one *other* blocking reason, so the 0052 fix
would not have changed a single verdict on that case. It was still worth doing
— V1's and V4's `unchecked` entries were pure naming artefacts, since
`weather_context`, `profile_data_quality` and `check_clearsky_consistency` had
all run — but it was not the constraint.

**The constraint.** `review()` merged two different things into one list:

```python
unsupported = [*grounding.as_claims(), *payload.get("unsupported_claims", [])]
```

and `CriticVerdict` refuses `accept` when that list is non-empty. So the
critic's own prose observations became hard vetoes. These are not fabricated
numbers — they are reviews:

> *"there is no matching rise in performance ratio which would be the signature
> of a bad sensor" — no tool measured or reported a correlation between
> clear-sky ratio and PR trend; this is an inference presented as evidence.*

That is a good critique, and it disqualified the answer permanently, because a
competent reviewer always finds something to say. **The critic blocked itself
by doing its job well.** One of its observations independently identified the
fault-injector defect recorded in DECISION 0051 — arguing from physics that a
dead string and a frozen sensor produce the same noiseless zero — which is
exactly why these must inform the next cycle rather than end the investigation.

**Decision.** The two are separated.

- `unsupported_claims` is what the *arithmetic* found: a figure in the prose
  that appears in no tool's provenance ledger. Objective, verifiable, and still
  a hard veto — "no fabricated numerics reach the interface" is the guarantee
  this node exists for and it does not weaken.
- `observations` is what the *reviewer* said. Recorded on the verdict, shown,
  and used as the revision request when one is sent back — the reviewer's own
  words are a better instruction than boilerplate — but it cannot veto alone.

Replaying the four real verdicts through the new code: all four now accept,
where all four sent back.

**One thing the trace cannot settle.** It records the verdict, not what the
model *wanted*. If these send_backs were the model's own choice rather than the
override's, this changes nothing for them — V3, with nothing flagged at all,
certainly was. The next run distinguishes the two: with observations no longer
vetoing, a persistent send_back is the model's judgement, and the fix would
then be in the critic's prompt rather than in this gate.

## 0056 — Transient failures are retried; a dead network stops the run (2026-07-26)

Four evaluation runs died to `APIConnectionError` and `overloaded_error`,
losing twenty-plus cases each at zero work apiece. Two changes, at two levels.

**In the client:** `overloaded_error`, dropped connections and rate limits are
retried in place — three attempts, 2s/4s backoff. The API saying "not now" is
not "not ever", and abandoning a case on the first one throws away the minutes
and dollars already spent on it. A `BadRequestError` is deliberately *not*
retried: it fails identically every time, and `is_systemic_request_error` stops
the whole run on it instead (DECISION 0048).

**In the runner:** three consecutive case failures end the run with a count of
what completed and what remains. One case can legitimately die on its own
content; three in a row *after* the client's retries is the environment, and
continuing burns the case list for nothing.

---

## 0057 — Subsets are supported, and labelled (2026-07-26)

**The problem.** 43 cases at a few minutes and roughly a dollar each is the
right price for a *result* and the wrong price for "did the critic fix work?",
which two cases answer. Iterating on the full split also means every network
blip costs the whole run — four runs died that way.

**The trap.** The obvious subset is "the first five", and the golden set is
ordered by fault type. The first five are three faults, one recoverable and one
by-design: **no unresolvable cases and no `not_the_plant`**. So
`correct_abstention_rate` cannot be measured at all, macro-F1 averages over
three classes instead of four, and neither number is comparable to the rules
baseline's — while both still print, looking exactly like a score.

**Decision.** Three ways to narrow, and a sentence saying what each costs.

- `--sample N` draws **stratified by category and settledness**, with a
  largest-remainder allocation and a floor of one per stratum so the single
  `by_design` case in forty-three cannot round away and take its share of
  macro-F1 with it. Eight cases keep all four categories, four look-alikes and
  one unresolvable — every headline metric stays computable. Seeded, so two
  runs draw the same cases and are comparable.
- `--only G-001,G-007` runs exactly those, for debugging one thing. An unknown
  id is an error: silently running forty-two cases when one was asked for is
  worse than a stack trace.
- `--limit N` takes the first N. Cheap, deterministic, and not a cross-section.

**The half that makes the other half safe.** Every subset prints, *above* the
results, which metrics it cannot support and that it is not a score. Above,
because a number is quoted far more often than the paragraph under it. This
extends the rule already in `eval/metrics.py` — a metric with no cases behind it
is reported absent rather than zero — from after the run to before it.

`--sample` is the one to reach for. `--limit 5` is honest about being a
straight line through a sorted file, which is exactly what it is.

---

## 0058 — Retry by status code, not by exception class (2026-07-26)

**The bug in the previous fix.** DECISION 0056 added transient-failure retries
with a list of exception classes:

```python
transient = (APIConnectionError, RateLimitError, InternalServerError)
```

A run then died on `overloaded_error` regardless. The reason is in the SDK's
own mapping: **529 maps to `OverloadedError`, checked *before* the `>= 500`
branch**, and `OverloadedError` is a *sibling* of `InternalServerError` under
`APIStatusError` — not a subclass. The retry never saw it.

**The shape of the mistake, which is the point.** Enumerating classes is the
wrong shape for "is this worth trying again". The SDK grows new ones —
`OverloadedError` and `RequestTooLargeError` are both recent — and every
addition silently reopens the hole, with no failing test, because the code that
would have caught it is a list of names that still looks complete.

Retryability is a property of the *response*, and the SDK's own policy is
expressed in status codes: 408, 409, 429 and anything ≥ 500. `_is_transient`
now checks those, plus `APIConnectionError` (which covers timeouts), and a test
asserts an unmapped 5xx is retried too — the rule is "5xx means try again", not
a list of blessed classes.

**Also changed while here.** Retries go from three to five attempts with 2/4/8/16s
backoff, about half a minute. Three attempts over six seconds rides out a blip;
a real overload episode lasts longer than that, and the SDK has already retried
twice before raising, so a call reaching the last attempt has been tried a
dozen times. `retry-after` is honoured when the API sends one, bounded at sixty
seconds so a server asking for ten minutes fails the case rather than stalling
the run.

---

## 0059 — Grounding checks quotation, not just the ledger (2026-07-28)

**The failure.** G-005's true cause is `seasonal_temperature_derating`. The
agent named it correctly on cycle 1, was sent back, spent two more cycles and
574 seconds, and finished on "not enough evidence" — a right answer turned into
a wrong abstention. The run printed its own reason:

```
UNGROUNDED FIGURES: ['2014', '2014', '2014', '-3.53642', '6.90082']
```

`unsupported_claims` is a hard veto (DECISION 0057 made it one deliberately, to
keep invented numbers off the interface). So a false positive here does not
merely annoy — it costs a correct answer, every time.

**What was actually wrong.** The check asked "is this figure in some tool's
provenance ledger?" That is narrower than the guarantee it stands for. The
guarantee is *no invented numbers*; the ledger is one of several places a
legitimate number comes from. Three sources were being shown to the agent and
then scored as fabrication when it used them. All three were reproduced offline,
without an API key, by running the tool registry against the test plant:

1. **Bare years.** `_MASKS` strips ISO dates, so `2014-05-01` is structure —
   but a sentence saying "compared with 2014" yields the integer 2014, which
   clears the free-integer ceiling of 24 and gets checked. A seasonal comparison
   is *about* comparing years, so the check punished hardest on exactly the
   reasoning the case required.
2. **Tool prose.** `ToolResult.digest()` renders `summary`, `caveats` and
   `labels`, and those carry figures the ledger does not: "a healthy inverter
   sits near 0.96-0.98 at load" puts two unquotable numbers in front of the
   model. The probe found two such traps in the synthetic plant alone.
3. **The knowledge base.** The `seasonal_temperature_derating` signature states
   that output falls about 0.4% per °C above 25°C — four figures, on the one
   signature that matters for this case. Quoting the physics it was handed was
   scored as making it up.

A fourth, smaller bug fell out of the same probe: `0.96-0.98` was parsed as
`0.96` and *minus* `0.98`. The check was inventing a negative figure and then
flagging it — manufacturing its own violation.

**The fix.** A second reference source, `quoted_from`: the prose the agent was
shown. Numbers are pulled from it *without* the masks, so a window written
`2014-05-01` grounds a sentence that says "2014", while a year the record does
not contain is still caught. `_NUMBER` gained a lookbehind so a hyphen between
two digits is a range rather than a sign.

**Why the material is carried on `Synthesis` rather than rebuilt.** The critic
is not passed the knowledge text. A reconstruction in `critic.py` would quietly
omit it and re-flag exactly the figures the synthesiser was entitled to quote —
the same class of divergence CLAUDE.md's "the dashboard must not reimplement a
computation" rule exists to prevent. `Synthesis.citable` holds the numbers the
model was shown when it wrote the draft, and every later check reads that.

**What did not change.** Arithmetic is still flagged: both operands shown, their
difference is not quotable, and a test asserts it. A value interpolated inside a
quoted range (`0.97` between `0.96` and `0.98`) is arithmetic too, and stays
flagged. Invented decimals, invented energies, and years outside the record are
all still caught — verified against the same probe after the change.

**The lesson worth keeping.** `tests/test_grounding.py` had already written the
failure mode into its own docstring: "a false positive buries the real finding,
and a check nobody reads is a check that is not running." Every test in it
tested the false-negative direction against a hand-written ledger. None ran the
check against the material the agent is actually shown. The bug was not in the
reasoning; it was in never pointing the check at real inputs.

---

## 0060 — A send_back says why, while the run is still going (2026-07-28)

The critic's trace step already carried `unsupported_claims`,
`unchecked_lookalikes` and `revision_request`. The progress display printed
`send_back` and nothing else, so diagnosing why a run was burning three cycles
per case meant a `jq` query over the trace *after* paying for it.

A send_back costs an extra cycle — minutes, and roughly a third of a case's
budget. Across 43 cases that is the difference between a debugging run and a
wasted afternoon, and the information to stop it was already being written to
disk. The display now shows the blocking reason in the order the critic applies
them: unsupported figure, then unweighed look-alike, then the reviewer's prose.

This is a view, not a measurement — `eval/progress.py`, not `src/` — and it
stays inside the existing guarantee that a display failure cannot take down a
paid run.

---

## 0061 — A review that will not commit must un-commit the answer (2026-07-28)

**The failure.** G-039's ground truth is that the question is *not decidable*
from this plant's telemetry: a ceiling at 185 kW against a 260 kW inverter is
clipping or curtailment, and the golden case says "committing is wrong either
way". The run reads:

```
[G-039] answer  settled: curtailment
[G-039] review  not_enough_evidence
-> G-039 ... settled
```

The critic got it right. The loop published the commitment anyway.

**The bug.** `loop_plain.py` treated the verdict as a *stop* signal only:

```python
if verdict.verdict in ("accept", "not_enough_evidence"):
    break
```

Stopping was correct. Leaving `out.synthesis` settled was not — the reviewer's
refusal ended the investigation and shipped the very answer it rejected. The
one guarantee the critic exists to provide was inverted by the code consuming
it, and `correct_abstention_rate: 0.0` on the only unresolvable case in the
sample is entirely this.

**Why no test caught it.** `test_not_enough_evidence_ends_the_loop_as_a_success`
existed and passed. It fed the loop an **unsettled** draft, so there was nothing
to withdraw and the missing withdrawal was invisible. The test asserted the
loop's cheap property (it stops) and never the expensive one (what it publishes).
The new test uses a settled draft and fails against the old code with
`settled=True, cause='string_outage'` — which is the bug, printed.

**The fix.** `withdraw_commitment` rebuilds the draft as the abstention the
review asked for: cause and confidence stripped before `Finding` construction
(the same order `synthesize` already uses), surviving causes taken from the
critic's `hypotheses_still_standing`, and their operational consequence and the
resolving measurement read from the hypotheses the *planner* wrote. Re-deriving
those here would be a second opinion about work already done.

Mirrored into `loop_graph.py` with an equivalence test, because a behaviour that
lives in only one loop is the same bug waiting in the half nobody reads.

---

## 0062 — The ledger namespaces by call, not by tool (2026-07-28)

`ledger_of` keyed on `f"{result.tool}.{key}"`, so a tool run twice silently
overwrote its own earlier measurements. That is not a rare path — re-running a
measurement on a narrower window is the router's whole job. One case ran
`per_mppt_current_balance` four times over different windows, keeping only the
fourth, and the six figures its answer quoted from the first two were reported
as fabricated.

The shape of the failure is the worst available: it makes the *most* thoroughly
measured investigations look the least grounded, and since `unsupported_claims`
is a hard veto (DECISION 0057), the punishment for measuring twice was a
rejected answer. This is also the mechanism behind the `-3.53642` / `6.90082`
figures in DECISION 0059 that the quotation fix covered without explaining.

Repeat calls now get a `#2`, `#3` suffix. Keys are provenance labels and are
never parsed, so the shape is free to say which call a figure came from.

---

## 0063 — The critic may write as much as the synthesiser (2026-07-28)

`TruncatedReply: critic hit its 16000-token cap` failed a whole case — G-006,
which the agent had answered **correctly, twice**. The critic re-reads every
measurement, the draft and the look-alike checklist, then writes an exclusion
with reasoning for each; at `effort=high` its thinking alone can outrun what the
draft cost to produce. DECISION 0055 raised the synthesiser to 32000 and left
the critic at 16000, which put the ceiling on the wrong node — the one that
*reads* the long output was capped below the one that writes it.

Both profiles now give the critic the synthesiser's ceiling, and a test asserts
the relation rather than the number, so the next person to raise one is made to
raise the other.

---

## 0064 — The router may only look up causes that exist (2026-07-28)

`look_up_causes` was `{"type": "string"}`. The router used it to ask what
separates "H4 and H1", and once a whole sentence. The knowledge base is keyed by
cause name, so those lookups could not have matched anything — nine of them
across eight cases, each a paid round trip returning "nothing known about those
causes".

DECISION 0053 closed the synthesiser's `cause` to a vocabulary for exactly this
reason and stopped there. This is the other half. The vocabulary itself moved to
`src.knowledge.cause_vocabulary()`, since the router needing it made reaching it
through the synthesiser look like a synthesiser question when it is a knowledge
question.

---

## 0065 — A crash is not an abstention, and an unrun target is not a failure
(2026-07-28)

Two reporting bugs, one shape: the evaluation claiming to know things it had not
measured.

**Agency counted crashes as choices.** `self_initiated_abstentions` was
`sum(1 for p in predictions if not p.settled)`. A case that died on a 529 scores
as unsettled — correctly, that is what it produced — and was then reported as
the agent *deciding* to abstain. Eight cases with three API failures reported
five self-initiated abstentions when the agent had chosen twice. The same
contamination ran through every agency number: a crash contributes an empty
trajectory, zero tools, zero cost and zero cycles, all of which flatter.

`Prediction.failed_with` now marks a case that never produced an answer.
Scoring is unchanged — the failures stay in the denominator, because dropping
them would flatter the engine by removing its failures. Agency is measured over
the runs that ran, with `completed_runs` and `failed_runs` reported beside it so
a reader can see how much of the sample survived.

**Targets reported FAIL for never-assessed.** `meets_v1_targets` read
`by_split["heldback"]`, which is `{}` on a tuning-only run, so
`float(None or 0.0) >= 0.75` came out False four times. A debugging run printed
`false alarms 0.000` and `FAIL false_alarm_rate_at_most_0.15` on the same
screen. Both lines were computed correctly and together they said nothing true.

The method is three-valued now: met, missed, **not measured**. The distinction
it protects is real and survives — a split that *was* run but holds no
look-alikes still **fails** the false-alarm bar, because you do not clear a bar
by bringing no evidence to it. A split that was never run has not been assessed,
and calling that a failure is the same category error pointing the other way.
`eval/metrics.py` already omits a metric with no cases behind it; the targets
block now obeys its own rule.

**What the eight-case run actually measured.** Three of eight cases died on
infrastructure. Of the six that answered, every one had the right answer
somewhere in its trace and two survived to be scored. macro-F1 0.213 is not a
measurement of diagnostic ability; it is a measurement of the loop's ability to
keep an answer it already had. That is worth writing down because the number
looks like the former and would have been quoted as it.

---

## 0066 — The checklist is stated in the terms it is enforced in (2026-07-28)

**The gap.** The brief told every node:

```
LOOK-ALIKES THAT MUST BE CONSIDERED ON EVERY INVESTIGATION
weather, seasonal_temperature_derating, clipping, curtailment, ...
```

The critic hard-vetoed `accept` unless a tool from the registry's
`discriminates` had actually run for each of the seven. Those are two different
requirements in two vocabularies, and an agent can satisfy the first completely
while failing the second: consider clipping perfectly well from a time-of-day
profile already in hand, and still be sent back for never calling
`check_ac_ceiling`. G-017 took fifteen measurements, missed exactly one item —
`telemetry_gap` — and could not be accepted however good its answer was.

Coverage is also lopsided in a way the bare list hid. Five of the seven have
three or more tools that satisfy them; `curtailment` and `snow_or_dust_event`
have exactly one each. "Run something relevant" cannot satisfy the checklist by
luck, and nothing said so.

**The fix.** `lookalike_coverage()` inverts the registry into
`{look-alike: [tools that settle it]}`, and it is now the **one** definition:
`lookalikes_measured` checks against it, `lookalike_coverage_text` shows it in
the brief, and the critic's own prompt restates it from the same call. A tool
whose `discriminates` changes moves the instruction and the enforcement
together; neither can quietly become stricter than the other. The planner prompt
says the opening plan must cover every line, and why: a plan that leaves one out
has committed the investigation to a second round before it starts.

`lookalikes_actually_checked` — superseded by DECISION 0057 and since then dead
production code kept alive by its own tests — is deleted. It was a second,
divergent walk of `spec.discriminates` sitting beside the one just made
canonical, which is the exact hazard this entry closes.

**Why C rather than running the tools automatically.** The alternative was a
deterministic pre-flight: `profile_data_quality` and `detect_stuck_channels` are
pure functions, so covering `telemetry_gap` costs no LLM call at all. It was
rejected for now because injecting five tools into every run converges the
trajectories, and `distinct_tool_trajectories` and `unplanned_measurement_rate`
are how §5.5 measures whether this is an agent or a pipeline that narrates.
Buying accepts by having the harness take the measurements would improve the
score by removing the agency it claims to be scoring. If instruction alone does
not close the gap, the pre-flight returns — with the injected calls marked in
the trace so agency can exclude them.

**Cost.** The brief is in the user message, not the cached system prompt, so
this adds roughly 250 input tokens to every LLM call — on the order of a cent
per case against a median of $1.00. The rejected alternative was cheaper on
tokens and more expensive on the thing being measured.

**Not a signature catalogue.** CLAUDE.md forbids handing the planner "string
outage looks like X". This says which *tool* addresses which look-alike, never
what one looks like, and it discloses nothing new — the tool catalogue already
prints "check_ac_ceiling helps separate: clipping, curtailment". It is the same
relation indexed the other way, so a requirement stated per look-alike can be
acted on per look-alike.

**What it does not fix.** Whether the checklist is actually what is blocking
`accept` on most cases is still unmeasured. It was hand-verified on G-017 only.
DECISION 0060 now prints the blocking reason live — `send_back: look-alikes not
weighed: telemetry_gap` — so the next run answers that question directly instead
of by inference.

---

## 0067 — Look-alike coverage was crowding out the answer (2026-07-28)

**The observation.** G-017 re-run with DECISION 0066 in place. Every mechanical
check passed: no dead knowledge look-ups, `profile_data_quality` planned rather
than adaptive, ungrounded figures 6 → 1, **zero send_backs for the first time in
the project's history**, cost $1.09 → $0.65, latency 644s → 402s. And the case
came out *worse*: `not enough evidence` against a ground truth of
`fault / shading`, which an earlier run had answered correctly.

**The cause, which is arithmetic and not luck.**

| | measurements | checklist coverage | answer |
| --- | --- | --- | --- |
| before 0066, cycle 1 | 7 | 4/7 | `settled: shading` ✅ |
| after 0066 | 8 | **7/7** | not enough evidence ❌ |

`max_tools_per_cycle` was 8. Full coverage costs five well-chosen measurements —
exactly as 0066's own brief promises — leaving three for the question the plant
manager actually asked. So the run spent them on `check_ac_ceiling`,
`compute_temp_corrected_pr` and `profile_data_quality`, and **dropped**
`check_night_offset` and `characterize_onset`, the two onset measurements that
had produced `shading` before.

0066 worked precisely as designed. Inside a fixed measurement budget it turned
coverage and discrimination into a zero-sum trade, and coverage won because it
is the one with a hard veto behind it. A cap chosen before the checklist was
enforceable stopped being a safety net and became the binding constraint on
whether the agent could answer at all.

**Fix, part one: the budget lives in one place and is larger.** 8 → 12, moved
into `limits.max_tools_per_cycle`. It had been hardcoded in `loop_plain`,
`loop_graph` *and* `eval/runner` — three copies of one number, and a test now
asserts all three read the config instead.

**Fix, part two: the circuit breakers had to move with it.** These are not
budgets, and conflating the two was its own bug. `BudgetExceeded` aborts the
entire evaluation by design, because it normally means the loop is not
terminating. So a per-investigation cap set *below* the intended operating point
does not save money — it converts an ordinary expensive case into a dead run.
The old $1.00 was set when the design target was $0.15 per question; one measured
cycle is $0.65, and the two 2-cycle runs that legitimately cost $1.09 and $1.11
were already over it. Twelve measurements is ~18 calls and ~$0.70 a cycle, so
the §3.5 four-cycle allowance permits ~72 calls and ~$2.80 *by construction*.
Breakers now sit above that at 80 calls and $3.50, and a test checks the relation
arithmetically so raising either allowance forces them to be revisited.

This is a spend ceiling, not a spend plan. Median cost is not managed by the
breaker; it is managed by making extra cycles unnecessary.

---

## 0068 — An abstention may not name a measurement the agent could have taken
(2026-07-28)

G-017 declined to answer and gave its reason as:

> Run `string_onset_scan` to see whether the morning shortfall's start/end times
> track the sun

`string_onset_scan` is in its own registry. It had run it on this case in an
earlier attempt. The abstention was not a judgement that the telemetry cannot
decide — it was the per-cycle measurement cap, reported as if it were.

CLAUDE.md makes `not_enough_evidence` a first-class successful outcome, and it
has to stay one: G-039's resolving measurement is the grid operator's dispatch
log, genuinely outside the plant's data, and declining there is the correct
answer. The two cases produce identical-looking output and are opposite in
kind. Nothing in the system told them apart.

`unrun_tools_named_in` does, deterministically: match the registry's tool names
against the resolving measurement and subtract what was already run. Verified
against the real text from both runs —

```
G-017: "Run string_onset_scan to see whether..."   -> ['string_onset_scan']
G-039: "Pull the grid operator's dispatch log..."  -> []
```

A substring match suffices because tool names are long and snake_cased;
`string_onset_scan` does not appear in a sentence by accident.

An abstention naming an unrun tool is now a `send_back` whose request names the
tool. It is applied to `accept` as well as to `not_enough_evidence`: a reviewer
that waves through an abstention with an available next step has made the same
mistake as the node that wrote it. The revision request *replaces* the model's
own rather than deferring to it, because the deterministic reason is what the
next cycle has to act on.

The synthesiser prompt now states the rule where it is enforced — the same
pattern as 0066. "Reserve it for evidence that is genuinely out of reach: a grid
operator's dispatch log, an inverter's configuration, someone walking the array.
This is checked against the tool registry, not taken on trust."

**Caveat carried forward.** n=1, and LLM runs are not deterministic; this run and
the previous G-017 differ by more than the fixes between them. The budget
arithmetic in 0067 is structural and holds regardless. Whether these two changes
recover the answer is unmeasured until the next run.

---

## 0069 — A streamed error arrives inside a 200 (2026-07-28)

**The failure.** G-017, re-run with 0067 and 0068 in place, died after six
measurements on `overloaded_error` — the exact failure DECISION 0058 was written
to retry. Five attempts over thirty seconds should have ridden it out. None
happened.

**The clue was in the type name.** The log said `APIStatusError`, not
`OverloadedError`. The SDK maps 529 to `OverloadedError`, so a bare base-class
instance means the status code was not 529.

**The mechanism, read out of the SDK and reproduced offline.**
`_streaming.py` handles a mid-stream failure like this:

```python
if sse.event == "error":
    raise self._client._make_status_error(err_msg, body=body, response=self.response)
```

`self.response` is the **stream's** HTTP response, and it succeeded — status
**200**. `_make_status_error` dispatches purely on `response.status_code`, so
every branch misses (400, 401, 403, 404, 409, 413, 422, 429, 529, `>= 500`) and
it falls through to a plain `APIStatusError` whose `status_code` is 200. The real
reason survives only in the body.

`_is_transient` asked the status code. 200 is not retryable and not `>= 500`, so
the overload was classified as permanent and the case was abandoned.

**Two correct fixes with a blind spot between them.** DECISION 0058 replaced
exception-class matching with status-code matching, and the reasoning there still
holds — enumerating SDK classes is the wrong shape for "is this worth trying
again". DECISION 0049 moved every call onto `messages.stream()` because raising
the token ceilings made non-streaming requests illegal. Each was right on its own.
Together they left every API error travelling by a route where the discriminator
is meaningless. Neither entry could have anticipated it; nothing tested the two
in combination, because no test ever built the exception the streaming path
actually raises.

**The same hole, pointing the other way.** `is_systemic_request_error` also keyed
on the status: it recognises `BadRequestError`, which a streamed
`invalid_request_error` never becomes. A schema the API rejects — a defect
identical for every case, and the reason that check exists — would have been
treated as a per-case failure and reproduced once per case. The bug that entry was
written to prevent was reachable again through the new transport, and silently.

**The fix.** Classify on the error type the API reports, consulted *before* the
status code, with the status-code rule kept as the fallback:

```
200 overloaded_error       APIStatusError    transient=True   systemic=False
529 overloaded_error       OverloadedError   transient=True   systemic=False
200 invalid_request_error  APIStatusError    transient=False  systemic=True
400 invalid_request_error  BadRequestError   transient=False  systemic=True
200 some_future_error      APIStatusError    transient=False  (falls back to status)
502 some_future_error      APIStatusError    transient=True   (falls back to status)
```

Systemic types are checked first, so a malformed request cannot be retried
however it is transported. An unrecognised type still falls back to 0058's rule,
so a new transient class the API introduces on a 5xx keeps being retried without
this list being updated.

**What the run does and does not tell us.** Nothing about the agent: it never
reached an answer. The metrics fixes from DECISION 0065 did exactly their job —
`completed_runs 0`, `failed_runs 1`, and `note: every case failed before
answering; nothing to measure`, with no agency figures invented from a crash. The
`macro_f1 0.000` and `missed real faults 1.000` lines are scoring, where failures
stay in the denominator on purpose; on one case they are noise, not a result.

Whether 0067 and 0068 recover G-017 remains unmeasured. Six measurements is one
past the old cap of eight would have allowed for discrimination, which is
suggestive of nothing yet.

---

## 0070 — Convergence is arithmetic, not an opinion (2026-07-29)

**The failure this closes, stated plainly.** Four cases reached the correct
cause and were talked out of it: G-001, G-005, G-006, G-017. G-017's last run
committed to `shading` in cycle 1, again in cycle 2 after three more
measurements, and was sent back both times; cycle 3 abandoned it for "not
enough evidence". Two thirds of a ten-minute run were spent making the answer
worse.

**The rule.** If the previous cycle committed to a cause, this cycle took more
measurements in response to the review, and the answer is *still* that cause,
the investigation has converged. More evidence did not move it, which is the
strongest signal available that another cycle will not either. No model is asked.

**What it does not skip.** `inspect_draft` — factored out of `review` for this
purpose — is the deterministic half of a review: no fabricated numerics, every
look-alike weighed, no abstention naming an unrun tool. An answer accepted by
convergence has passed every check that does not require judgement. What it
skips is the judgement, and the judgement is what was destroying correct
answers. A test asserts that a draft carrying an invented figure is *not*
accepted by this path however many times it repeats.

Writing the tests found the same edge honestly: the first version did not weigh
the look-alikes and therefore did not converge. The test was wrong and the code
was right, which is the correct way round for once.

It also bounds latency. A converged answer stops at cycle 2 rather than running
to the cap — roughly two thirds of the wall clock on the runs measured so far.

---

## 0071 — The critic could not see its own previous review (2026-07-29)

**The core issue, after five rounds of fixing symptoms.** The critic's prompt
held the brief, the candidate causes, every measurement, the draft and the
look-alike checklist. It held **nothing about its own previous review**.
Meanwhile `plan()` and `synthesize()` both receive `revision_request`. So
information flowed one way — critic to planner and synthesiser — and nothing
came back.

Every review was therefore a fresh reviewer meeting the case for the first
time, with unlimited standards and no memory. The consequences are the entire
observed history:

* It could not say "you addressed my concern", because it did not know it had
  one.
* It re-raised the same objection in new words. G-017: "reconcile the sharp
  onset detected by characterize_onset", then "split the string-1 evidence
  around the 2017-03-16" — the same point, after the first had been answered
  with three more measurements.
* It could not notice the answer had not changed, which is the strongest
  evidence of convergence available to it.
* A competent reviewer always finds something and nothing priced another cycle,
  so `send_back` was the equilibrium rather than an accident.

The earlier fixes — grounding quotation (0059), the ledger collision (0062),
the checklist vocabulary (0066) — were all real bugs, and clearing them is why
this became visible: the send_backs in the final G-017 run were pure critic
judgement with no deterministic veto masking them. But they were repeatedly
described as "the reason it never accepts", and they were not.

**The fix.** `review()` now receives its previous verdict, the cause the last
cycle committed to, and how many measurements were taken in response, and the
prompt says what to do with them: *"If what you asked for was done and the
answer is unchanged, that is evidence the investigation has converged and
should be accepted — not a reason to find something new."* A structured
`previous_request_addressed` field makes it answer rather than pass over.

**What deliberately did not change.** The critic still tightens and never
loosens; `previous_request_addressed: "yes"` does not override a `send_back`
into an `accept`. Giving a model the power to talk itself into accepting is the
opposite of the guarantee this node exists for. The *deterministic* convergence
stop in DECISION 0070 is where an accept can be granted without being asked
for, and it is arithmetic.

---

## 0072 — The graph becomes the production path, and the checkpointing claim
gets corrected (2026-07-29)

`loop_graph.py` was written, equivalence-tested and then called by nothing:
both `eval/runner.py` and `watcher.py` imported `loop_plain.investigate`. Every
run this project has ever done used the plain loop. A second implementation
nobody exercises is a liability, not a safety net, so the runner and the
watcher now import the graph. The plain loop remains the specification and the
equivalence test remains the gate.

**And the claim that justified the port was false.** This file's own docstring
said *"Checkpointing comes free... a run that dies at case 60 currently starts
again."* There was no checkpointer — no saver, no thread id, only a recursion
limit. It sat unexamined for as long as the port sat unused.

One is wired now (`InMemorySaver`, thread id per investigation), and the
docstring says what it actually buys. Be precise: an in-memory saver records
every node boundary inside one investigation and **dies with the process**. The
failures that actually cost this project cases — an API overload at case 6, an
exhausted credit balance mid-answer — take the interpreter with them. Durable
within-run resume needs `langgraph-checkpoint-sqlite`, which is not a
dependency here.

**So the resumability that matters is the runner's, and it is not LangGraph's.**
What is expensive to lose is the twenty cases already paid for, and only a file
survives that. `--resume FILE` writes one JSON object per case as it finishes
and skips cases already recorded. Append-only and flushed per case, so a
process killed mid-write loses at most the case in flight — and a half-written
final line is skipped rather than refusing the whole resume, because refusing
would discard exactly what the journal exists to save. The `failed_with` marker
survives the round trip, so a resumed run still does not count a crash as a
chosen abstention.

The honest ledger on the framework, for `docs/LANGGRAPH_TRADEOFF.md`: it makes
within-run resume *possible* where the plain loop makes it impossible. It did
not deliver the thing the docstring promised, and the thing that was actually
needed took thirty lines of `json.dumps` in the runner.

---

## 0073 — Measure the run instead of estimating it (2026-07-29)

Every speed decision in this project was made from an estimate: count the calls
in a run log, multiply by a guessed per-call latency, argue from the product.
"About 80% of the wait is the eight Sonnet calls" was said out loud on exactly
that basis. It may well be right. It was never checked, and it did not need a
new run to check — `TraceStep` has recorded `latency_ms` and `cost_usd` since
step 3, and there were seven runs on disk.

`eval/runner.py profile` reads them. Per node: calls, seconds, share, dollars;
per run and in total; plus the number the latency argument turns on — **how much
was spent after cycle 1**, which is the re-review overhead that four cases paid
for an answer they already had.

**Two things it refuses to do, because both would flatter the result.**

A trace with no recorded timing is *skipped and named*, not averaged in. Scripted
and replayed traces record zero latency, and including them would halve every
figure. The report says "no timing recorded, skipped" and gives the count of
timed runs against the total.

A `tool` step's latency is the **router turn that chose the measurement**, not
the measurement. Tools are pure functions and take milliseconds. The report says
so in as many words, because a reader who concluded the physics was slow would
draw exactly the wrong lesson about where to optimise.

**Two gaps this found in the trace itself, which is the point of building it.**
The critic's cost and latency were accrued to the run total and never written
onto its trace step, so review — the node whose value is least established —
appeared free to anything reading the tape. That is fixed, and older traces are
flagged as under-reporting review rather than silently averaged. The repository
guard from DECISION 0043 also caught `eval/profile.py` as untracked before it
could become a second `src/findings/build.py`.

---

## 0074 — A `fast` profile, added as a lever rather than pulled (2026-07-29)

The obvious response to a ten-minute run is to lower `effort` on the expensive
nodes. That is a quality trade, and making it by editing `default` would repeat
the mistake DECISION 0073 exists to stop.

So it is a third profile beside `default` and `quality`, selectable with
`PV_MODEL_PROFILE=fast`, to be measured rather than assumed. Planner and critic
drop to `effort: medium`; the synthesiser stays at `high` because it writes the
answer a plant manager reads and is the node least worth degrading. The critic
drops because it is the node whose value is least established — one help, four
correct answers destroyed — so it is the cheapest place to spend less while that
is being settled.

The eval output now prints the active profile in its header. Two runs at
different effort settings were previously indistinguishable in the output, which
makes a comparison between them not a comparison.

---

## 0075 — The README said things that were not true (2026-07-29)

Three corrections, all of them the same kind of error: a design target reported
as though it were a measurement.

**Cost.** "86 investigations at roughly $0.15 each… budget ~$40… `--split
tuning` for about $7." Measured, a case is $0.65–$1.10 and 7–12 minutes, so the
real figures are ~$200 and ~$30. The target is now stated as a target, followed
by the measurement and the word "not met".

**Latency and shape.** The README implied a question-and-answer tool. Nobody
waits ten minutes at a prompt. The honest shape is `watcher.py` — an unattended
sweep leaving a ranked findings queue with evidence, triaged in seconds — and the
README now says that rather than letting a reader assume something the system
does not do well. The alternative being compared against is an engineer spending
an afternoon, not a chatbot answering instantly.

**RAG.** The retrieval numbers were reported without the two facts that decide
what they mean: the corpus is 23 chunks and *is the project's own knowledge
base*, and `CorpusRetriever` is not wired into any run — `investigate` defaults
to a dict lookup on cause names. Every "Looked up what is known about…" line in
every run log is that lookup. The stack is built, tested, measurably better than
BM25 alone, and unused. Saying so costs a talking point and is the only version
that survives a reader checking.

The pattern across all three is worth naming: each number was written when it was
a plan, and stayed after it became false. Nothing re-checked them because nothing
had run end to end.

---

## 0076 — Review splits into a reviewer and a set of checks (2026-07-29)

**The measurement.** G-017 with `--no-review`, the first end-to-end correct
result this project has produced:

| | answer | cost | wall clock |
| --- | --- | --- | --- |
| with the LLM reviewer | not enough evidence ✗ | $1.09–$1.31 | 644–900s |
| `--no-review` | **`settled: shading`** ✓ | **$0.288** | **165s** |

Correct, four times cheaper, four times faster. n=1, and it is one case the
reviewer had already failed three times, so it is the least surprising case for
this to happen on. It is still the sharpest evidence available.

**The conclusion is not "delete the critic".** Reading the ablation carefully
shows what it actually removed: `critic=False` broke *before* the deterministic
checks as well as the reviewer, so that correct answer carried no guarantee its
figures were grounded or its look-alikes weighed. It happened to have done both
— eight measurements covering all seven checklist lines — but a configuration
worth shipping cannot rest on happening to.

The critic node does two separable things:

1. **Checks that need no model.** No fabricated numerics, every look-alike
   weighed, no abstention naming a tool the agent owns and did not run. Free,
   deterministic, already factored out as `inspect_draft`.
2. **A judgement.** Expensive, slow, and with a record of four correct answers
   destroyed against one improved.

The evidence points at keeping 1 and dropping 2. So review now has three modes:
`None` (reviewer plus checks), `"checks"` (checks alone, with a repair cycle the
arithmetic specifies through `MechanicalObjections.as_request`), and `False`
(nothing).

**`False` stays exactly as it was**, and that matters more than it looks. It is
the control the ablation is measured against, and a control that quietly does
some of the work is not a control. An earlier version of this change redefined
`False` to include the checks; it broke eight tests, and the tests were right.

The cap is tested *after* the cycle increment in the new mode, matching the
reviewer path, so `--checks-only` and the default run on the same cycle budget.
An ablation comparing two different budgets compares nothing.

---

## 0077 — A dependency's warning is not the operator's problem (2026-07-29)

LangGraph's checkpoint package emits a `LangChainPendingDeprecationWarning`
about an `allowed_objects` serialiser default that this project neither
constructs nor configures. It began appearing in CLI output the moment the
runner started importing the graph (DECISION 0072), where it sits above the run
and reads as something the operator did wrong.

Filtered in `eval/runner.main`, by **message** rather than by category or
module, so a genuine deprecation from anywhere else still surfaces. Verified
both ways: absent from CLI output, still raised when the filter is not applied.

Suppressing a warning is usually the wrong instinct and worth justifying when it
is not. The test here is whether the reader can act on it: this one names a
parameter of a class in a transitive dependency, on a code path this repository
does not touch. Nobody reading a run summary can do anything with it, and a
warning nobody can act on trains people to ignore the ones they can.

---

## 0078 — What the profiler measured, and what it corrected (2026-07-29)

The first real answer to "where do the ten minutes go", from 13 timed runs
totalling 13,625 seconds and $21.87:

| node | calls | seconds | share | usd |
| --- | --- | --- | --- | --- |
| synthesiser | 59 | 6833 | **50.2%** | 10.99 |
| planner | 91 | 3052 | 22.4% | 5.72 |
| router + measure | 397 | 1988 | 14.6% | 3.66 |
| look up | 164 | 1321 | **9.7%** | 0.78 |
| review | 44 | 0 | — | — |

**The estimate was directionally right and wrong in the detail.** "About 80% is
the Sonnet nodes" — measured, planner plus synthesiser is 72.6%. But the
synthesiser *alone* is half of everything, which no estimate had said, and it is
the node nobody proposed touching. Router turns came in at 14.6% against a
guessed ~15%, the one number the estimate got right.

**Look-ups are not free.** 164 calls and 1321 seconds — 9.7%, not the rounding
error they were assumed to be. G-007 is the extreme: 20 look-ups, 990 seconds,
**89.8% of that entire run**. Each is a router turn, and repeats are common.

**Review is still unmeasured, not zero.** All 44 review steps predate the fix
that writes review cost onto the step, so the table reports 0.0s with a flag
rather than a figure. The claim "review is roughly a third of the run" remains
unverified — the profiler's job here was to say so rather than to average zeros
into a total.

**48% of everything was spent after cycle 1** — 6490 of 13625 seconds, $10.72 of
$21.87 — on runs where four cases had the right answer in cycle 1 and were sent
back. That is the convergence-stop argument, now with a number behind it.

**The correction the profiler needed.** `TraceWriter` opens with mode `"a"` and
the evaluation names every trace after the case, so re-running a case appends to
the file it wrote last time. `INV-G-004` reported 59 measurements, 4 cycles and
$3.37 while the run that wrote its last entries took 11 measurements and cost
$0.34: every per-run figure was a session of attempts summed together.
`split_attempts` cuts on `step_index` restarting at zero, which is exact rather
than inferred, and repeated attempts are labelled `#1`, `#2`. The totals were
always right; the per-run rows were not.

---

## 0079 — An even share is not this array's baseline (2026-07-29)

**The false alarms had one cause.** Across every log: string 7 at 0.125, 0.116,
0.124, 0.124, 0.126 — in five different months, against an even 1/7 = 0.143.
That is the plant, not a fault. `per_mppt_current_balance` measured every window
against the theoretical even share, so string 7 read as 12–19% deficient every
time it was scored, and the summary led with *"the lowest share is string 7 at
0.116"*.

G-004 and G-005 both read that sentence and committed to `string_outage`. Their
true causes were soiling and seasonal derating, and G-005 is a look-alike, so it
is a false alarm — a crew dispatched to a healthy array, which is the failure
this project exists to prevent. Two of the three cause errors in the eight-case
run, from one sentence.

**The tool already knew.** Its own source comment says the plant "has a string
that sits persistently ~12% below even in untouched data" and computes
`below_10` and `below_25` so the agent can tell a standing offset from a
failure. The information was in `values` and absent from the sentence the agent
reads. Knowing something in a comment is not the same as saying it.

**The fix is a second measurement, not a threshold.** Each string's share is now
compared against its own share over the record *before* the window, and the
summary reports the change:

```
standing offset:  string 7 carried 0.125 before this window, so it has moved
                  +0.0000. No string sits below its own baseline.
real outage:      string 3 carried 0.146 before this window, so it has moved
                  -0.0725. The largest fall from baseline is string 3, down 0.0725.
```

No classification, no rule — the tool reports the change and the agent decides.
The even share is still reported, because a genuine outage is visible in both.

Where there is not enough history the tool says so in a caveat rather than
reporting no change: "no baseline" and "no change" must not look the same.

The "largest fall" claim is compared at the precision the sentence prints. A
share that moved by 1e-9 is float noise, and *"the largest fall is string 1, down
0.0000"* asserts a fall while displaying none.

---

## 0080 — A cache hit is not a price (2026-07-29)

G-017 in the eight-case run reported **`$0.288, 0s`**. Both halves are wrong
together: its prompts matched an earlier run, so every call was replayed from
the response cache. It spent nothing, took no wall clock, and reported the
original run's cost.

`LLMResponse.cached` existed and nothing downstream read it, so the cache
docstring's promise that "cost figures stay honest" was half true — the flag was
set and never consulted.

Now `InvestigationResult` carries `cached_calls` and `cached_cost_usd`, the
per-case line says `[11 of 11 call(s) replayed from cache — $0.288 not spent
again]`, and `measure_agency` computes cost and latency medians over the runs
that actually spent something. A run where *every* call was replayed is counted
in `runs_served_from_cache` and excluded from those two medians; a partially
cached run still counts as priced, because it did pay.

When every run was cached the medians report `None` rather than 0.0. "Free" and
"not measured" are different claims, and this module has now made that mistake
twice — once with the v1 targets (DECISION 0065) and once here.

---

## 0081 — The category is a lookup, so stop asking for it (2026-07-30)

The eight-case run scored **0.875 on cause and 0.750 on category**. The whole
gap is one answer that named the right cause and filed it in the wrong bucket.

That is not a reasoning failure. `fault_signatures.yaml` already assigns a
category to every cause, and the mapping is a dictionary:

```
soiling      -> recoverable      sensor_drift  -> not_the_plant
shading      -> fault            clipping      -> by_design
```

The synthesiser was choosing both independently, so it could contradict a file
it had been shown. CLAUDE.md keeps the LLM out of arithmetic, thresholds,
scoring and data scope; a lookup on a value the model itself just chose belongs
on that list. `category_for(cause)` now decides it.

**The model is still asked**, and the answer still recorded, as
`Synthesis.category_as_written` and in the trace. The knowledge base wins, but a
disagreement is a signal — either the model has misunderstood the taxonomy or
the taxonomy is wrong — and silently overwriting it would throw that away. Same
shape as the critic recording what the reviewer said beside what the arithmetic
found.

---

## 0082 — `explain`, because the console clips the evidence (2026-07-30)

G-004 is the one case the eight-case run still gets wrong: it commits to
`string_outage` against a ground truth of soiling, having *measured* the soiling
signature — performance falling 0.00627 per day across five dry stretches, which
is dust accumulating between rain. Why it preferred the string reading is in the
trace and was never readable, because `StepPrinter` clips every line to 96
characters and that is where the sentence ends.

The display is right to clip: it is for watching a run. It is the wrong tool for
diagnosing one, and there was no other. `eval/runner.py explain G-004` prints one
attempt in full — the plan's candidate causes and what each would cost to act on,
every measurement's whole summary, why each unplanned tool was chosen, and the
final answer with its category, confidence and ungrounded figures.

No API calls: it reads a trace already on disk. It reuses `split_attempts` from
DECISION 0078, so `--attempt` selects among the runs concatenated into one file
and the default is the most recent.

The general point, which this project keeps rediscovering: the evaluation
harness is not only for producing scores. Half the bugs found in the last week —
the ledger collision, the standing offset, the concatenated traces — were
invisible until something was built to look at what had already been recorded.

---

## 0083 — What `explain` found on G-004 (2026-07-30)

The first case read in full rather than through a 96-character clip, and it
changes what the remaining error is about.

**The reasoning is not the problem.** The router's `reason_for_choosing` fields
are real differential diagnosis, unprompted:

> *"soiling should show smooth decline with occasional rain-day jumps, but a
> string outage shows exactly one step."*

That is the correct discriminator. It then **measured that discriminator three
times and cited none of the results**:

| measurement | reading |
| --- | --- |
| `soiling_recovery_pattern` | −0.00627/day across 5 dry stretches, **+0.0017 after 7 low-insolation days** |
| `daily_performance_trend` | R² **0.02**, with **10 single-day recoveries of ≥0.01** |
| `characterize_onset` | **14 days spent between the two levels** |

Ten recoveries and a fortnight-long transition. By its own stated test, that is
soiling and it is not close.

**What it cited instead were two artefacts.**

*"String 7 is running at only 0.791 of the array's median level"* —
`compare_string_profiles`, which had the identical standing-offset hole DECISION
0079 fixed in `per_mppt_current_balance`. Half a fix: two tools make the same
comparison and only one of them was given a baseline. This is the number the
wrong answer rested on.

*"its current share sits below every other string (0.125 against an even
0.143)"* — it quoted the even-share clause and skipped the baseline clause **in
the same sentence**, which read "string 7 carried 0.127 before this window, so
it has moved −0.0017". The 0079 fix worked; the disconfirming evidence was
present and correctly computed, and the agent preferred the framing that
confirmed its hypothesis.

That last point is worth keeping separate from the fixes. Correcting the tools
raises the floor. It does not establish that the reasoning is sound, and G-004
is evidence that it is not reliably so.

---

## 0084 — `characterize_onset` printed a fall as a rise (2026-07-30)

```
averages 0.891 before 2017-05-28 and 0.794 after, a change of +0.097
```

`drop = before - after`, emitted as a signed "change". The words say it fell,
the sign says it rose, and the field carrying it is called `absolute_change` —
which any reader takes as after minus before. Reproduced offline: 0.854 → 0.768
printed `+0.085`, with `absolute_change = +0.0854` and `relative_change = +0.1`.

`absolute_change` and `relative_change` are now after minus before, so a fall is
negative; `change_magnitude` keeps the unsigned size for anything that wants
"how big was the step". The sentence says "fell by" or "rose by" rather than
handing the reader a sign to interpret.

Its test asserted `absolute_change > 0.1` on a *drop*, so the test agreed with
the bug — which is how it survived. There is now a companion test for a rise,
because half the golden cases ask "has something got better?" and a recovery
reported as a fall is the same bug pointing the other way.

---

## 0085 — Two tools disagreed by 5× and neither said why (2026-07-30)

On G-004:

* `string_onset_scan` — string 7 moved **−0.0090**, 16.9× its own scatter
* `per_mppt_current_balance` — string 7 moved **−0.0017** from baseline

Both correct. They measure different things: the onset scan compares before and
after a change-point *inside* the window, while the share compares the
whole-window mean against the record before it, so a step part-way through is
diluted by the days either side of it. Nothing in either summary said so, and
the agent was left to reconcile a 5× gap with no basis for doing it.

The share tool now says it is a whole-window average and names
`string_onset_scan` as the instrument for locating a change-point — including
the inference that matters: a larger figure there than here means the change is
recent rather than absent.

This is a general hazard in a tool set built to be composed. Each tool is
individually honest and the composition can still mislead, because a reader
cannot know which of two disagreeing measurements answers their question unless
the tools say what question they answer.

---

## 0086 — Correcting DECISION 0019's headline: checkpointing (2026-07-30)

This file is append-only, so DECISION 0019 stands as written. It should be read
with this next to it, because its central claim was wrong.

0019 decided both loops ship, and gave the reason:

> **checkpointing is what tips it** — an 86-case evaluation that dies at case 60
> currently starts again from case 1.

Two things are wrong with that.

**There was no checkpointer.** `build_graph` called `graph.compile()` with no
saver and no thread id. The capability the port was justified by had never been
wired up, and nothing noticed for as long as nothing imported the port — every
run this project made used the plain loop until DECISION 0072.

**Wiring one in does not deliver the sentence.** `InMemorySaver` records node
boundaries *within one investigation* and dies with the process. The failures
that actually cost cases — an overload at case 6, an exhausted credit balance
mid-answer — take the interpreter with them. A run that dies at case 60 still
starts again. What survives it is `eval/runner.py --resume`: one JSON object per
case, appended as it finishes. Thirty lines, no framework.

Durable within-run resume is reachable — `langgraph-checkpoint-sqlite` is one
dependency away — but that is a smaller claim than the one made.

**Two costs 0019 could not have seen**, because it was written against a loop
that had stopped changing:

*Every behaviour change now costs two edits.* Convergence detection, the three
review modes, the withdrawn commitment, the critic's own history — each written
twice and kept identical, with the equivalence test as the only thing between
"ported" and "diverged". On a loop under active change this is the largest cost
by some distance.

*A dependency's warnings become yours.* LangGraph's checkpoint package emits a
pending-deprecation notice about a serialiser this project neither constructs
nor configures, which began appearing above every evaluation run the moment the
runner imported the graph (DECISION 0077).

**What does not change.** Both loops still ship, and the port is still what
runs — a second implementation nobody exercises is worse than either choice,
which is precisely how its own justification went unchecked for so long. And the
argument for writing the plain loop first is *strengthened*: "did the framework
buy anything?" became answerable, and the answer was less than the document
claiming to answer it said.

The uncomfortable part is worth stating plainly. `docs/LANGGRAPH_TRADEOFF.md`
exists specifically to hold a framework to a measured standard, and it asserted
the framework's headline benefit without ever exercising it — in a paragraph
that congratulated the project for not doing exactly that. Writing "measured
claim rather than assumption" does not make it one.
