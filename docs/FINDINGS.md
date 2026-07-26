# Findings

What the evaluation actually showed, published whichever way it fell.

Everything here is reproducible from a clean checkout:

```bash
python -m src.data.cli ingest --system 4902 --years 2016 2017
python -m eval.runner build
python -m eval.runner run --engine rules --split both
python -m eval.runner compare --split heldback
```

---

## Status

**The agent has not been scored yet.** No `ANTHROPIC_API_KEY` was available in
the environment this was built in, so every number below comes from the rules
baseline, the physics, or the fault injector — all of which run without one.

That is stated first because a half-finished evaluation is exactly the kind of
thing that gets quietly presented as a whole one. The harness, the golden set,
the metrics and the comparison are complete and tested; the agent column is
empty because it has not been run, not because it is pending analysis.

To fill it in:

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY
python -m eval.runner compare --split heldback --out docs/comparison.json
```

---

## The dataset, and what it turned out to be

**NREL PVDAQ system 4902, NIST_Ground_1**, Gaithersburg MD. 270.7 kW, 1152
Sharp NU-U235F2 modules, fixed 20° tilt due south, 2016–2017 at 15 minutes.

Three things about it were only discovered by looking:

**The logger runs on Eastern *Standard* Time all year.** Recovered by scoring
every half-hour offset against a modelled clear-sky curve on the clearest days:
UTC−5 at r = 0.9972, beating the runner-up by 0.026. Nothing in the metadata
says so. Assuming UTC would have put every time-of-day measurement five hours
out, which is the difference between a morning shadow and an afternoon one.

**Two of the three POA channels are unusable.** `irradiance_poa_o_2203` reads a
maximum of 12.38 where the working channel reads 1360 — a factor of 115.
Selecting on name alone makes every performance ratio wrong by that factor while
still looking plausible. Channel resolution is done on physical plausibility for
this reason, and it also rejected a wind channel reading 292 m/s.

**There are no fault logs.** PVDAQ ships telemetry and equipment metadata only;
there is no maintenance, outage or event table anywhere in the archive, and the
metadata's `comments` field is empty. Ground truth is therefore injection-only,
which is why injecting at the physics layer rather than the signature layer
carries so much weight — it is the only truth the evaluation has.

**One string has been under-producing the whole time.** Combiner 7 carries about
12.5% of array current against an even share of 14.3%, consistently, in
untouched data across both years. Either it is a smaller combiner or it is a
real long-standing fault, and without a maintenance log it cannot be settled —
which is a fair illustration of the problem this project is about. It is also a
practical nuisance: any per-string threshold below 12% fires on every window
ever measured (see below).

---

## Physics validation

Over the full two years, with no injection:

| | |
| --- | --- |
| Performance ratio, as measured | 0.857 |
| Performance ratio, corrected to 25 °C | 0.893 |
| Data completeness (daylight intervals with power) | 91.3% |
| Deficit against the coarse expectation model | 2.58% |
| Intervals clipped | 0 (plant peaks at 254.9 kW against a 260 kW inverter) |

The seasonal signature is textbook and is the thing the whole project exists to
handle: **raw PR swings 0.80 to 0.95 across the year while the corrected figure
stays flat between 0.86 and 0.92.** Every one of those raw summer troughs is a
false alarm waiting to be dispatched on.

### A real bug the physics found

Scored naively, July 2016 reads **PR = 0.098** — a 90% loss that never happened.
The plant loses its AC power channel for 35 days while the weather station keeps
logging, and integrating all the insolation against only the surviving energy
manufactures a catastrophic phantom deficit. The numerator and the denominator
now cover identical intervals, and `data_completeness` is reported alongside
every performance ratio so a figure resting on 14% of the record cannot be
presented as one resting on 98%.

---

## The rules baseline

The baseline calls **the same eighteen tools the agent calls**, through the same
registry, with the same argument validation. A gap between the two is therefore
a gap in reasoning rather than in measurement quality, which is the only
variable worth isolating.

Thresholds are set on the tuning split only. The held-back split was generated
from a different base seed and different windows and was not inspected while
setting them.

| | tuning | held back |
| --- | --- | --- |
| Overall accuracy (macro-F1) | 0.345 | 0.469 |
| Category accuracy | 0.488 | 0.581 |
| Cause accuracy | 0.349 | 0.372 |
| **False alarms on look-alikes** | **0.056** | **0.056** |
| **Correct "not enough evidence"** | **0.000** | **0.000** |
| Missed real faults | 0.375 | 0.438 |

43 cases per split, 18 look-alikes (41.9%), 5 unresolvable cases.

### What the numbers say

**Correct abstention is 0.000 and no amount of tuning changes it.** The engine
commits to the first rule that fires and has no way to represent "two causes
survive and here is the test that separates them". On all five unresolvable
cases per split it confidently answers "clipping" — and on the curtailment ones
it is confidently wrong in the direction that costs money, because a plant
curtailed by the network operator and recorded as clipped never gets claimed
for. This is the gap the agent exists to fill, and it is the single number to
watch when the agent column is filled in.

**The false-alarm rate is deliberately not minimised.** A threshold of 0.18 on
the per-string deficit scores 0.012 higher on macro-F1 and *doubles* the
false-alarm rate. 0.22 was chosen instead. Crews dispatched to healthy plant is
the number the field cares about, and 0.18 also sits uncomfortably close to
combiner 7's 12% standing anomaly — a slightly different window would push it
over.

**Held-back scores better than tuning, by 0.12.** Unusual, and the honest
reading is that at 43 cases per split this is noise rather than evidence of
anything. The two splits use different windows and different seasons; the
tuning split happens to contain more of the marginal cases. It is reported
because a gap in the *other* direction over 0.10 would be called overfitting,
and it would be dishonest to only report the gap when it flatters.

**Agency metrics on the baseline read exactly as they should**: one distinct
tool trajectory across all 86 runs, an unplanned-measurement rate of 0.000, and
zero self-initiated abstentions. That is the metric correctly identifying a
pipeline, which is what the baseline is. If the agent's numbers come back
looking like this, the agent is a pipeline too and the honest conclusion is that
the orchestration bought nothing.

---

## Bugs the evaluation found in itself

Worth recording because each was invisible until something else was built on
top of it, and each would have made a published accuracy figure meaningless.

**Soiling left every string current untouched** while cutting plant power. Dust
sits on the glass, upstream of everything electrical, so it costs every string
the same fraction — and that uniformity is precisely the signature that
separates soiling from a string fault. The case's own stated reasoning element
("affects all strings equally, so it is not a string fault") was vacuous: the
strings were flat because the injector had forgotten them.

**Clipping and curtailment capped AC only**, leaving DC at full. That implies an
inverter running at 58% efficiency all midday, which no inverter does. It is a
fingerprint of the simulator rather than of a power limit, and an agent could
have learned it instead of the physics.

**Shading one string of seven for three hours** removed 1.2% of the window's
energy — under the noise floor of every tool in the set. The case was
unlearnable in principle and would have scored coin flips.

**The unresolvable case was resolvable.** Its ceiling sat at 150 kW against a
260 kW inverter — 58% of nameplate, which no inverter does to itself. "Clipping"
was not a live explanation, so the pair the case exists to test was quietly
decidable and the case was not measuring what it claimed.

**The diagnostic only ever saw the investigation window.** Two of the eighteen
tools — "is this new?" and "was the weather unusual?" — raised on every single
case, silently, and would have been scored as failures of the agent.

**A failed measurement read as a passing one.** The rules engine defaulted
missing completeness to 1.0, so a window where the performance ratio *could not
be computed at all* passed the data-quality check and was diagnosed as a string
fault. Six telemetry-gap cases per split were mislabelled on that alone.

---

## Still open

- **The agent has not been run.** Everything above is baseline and physics.
- **Retrieval has not been ablated.** The knowledge layer is switchable
  (`--no-knowledge`) and untested against accuracy.
- **The critic has not been priced.** `--no-review` runs the identical loop
  without it.
- **Combiner 7.** Unresolvable without a maintenance log, and a live example of
  why `not_enough_evidence` needs to be a first-class answer.
