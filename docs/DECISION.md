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
