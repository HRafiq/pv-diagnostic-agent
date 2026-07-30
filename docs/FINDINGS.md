# Findings

What the evaluation actually showed, published whichever way it fell.

The baseline, the physics, the fault injector and the retrieval table are
reproducible bitwise from a clean checkout, with no API key:

```bash
python -m src.data.cli ingest --system 4902 --years 2016 2017
python -m eval.runner build
python -m eval.runner run --engine rules --split both
python -m eval.runner retrieval
python -m eval.runner compare --split heldback
```

The agent numbers are not, and it would be convenient to leave that unsaid.
They need a key, and `CLAUDE.md#determinism` scopes what reproducibility means
here: physics, tools, injection and retrieval are bitwise identical for the same
seed; the planner, router, synthesiser and critic are **not**, and never will be
— `temperature` no longer exists on current Claude models. Re-running the agent
gives a similar answer, not the same one. Every agent figure below is one run
per case unless it says otherwise, and that is stated wherever such a figure
appears.

---

## Status

**The agent has been run on eight tuning cases. It has not been scored.** Those
are different claims and the difference is the whole point of this section.

| | eight tuning cases, `--checks-only` |
| --- | --- |
| Overall accuracy | 0.700 |
| Cause accuracy | 0.875 |
| **False alarms on look-alikes** | **0.000** |
| Correct "not enough evidence" | 1.000 |
| Missed real faults | 0.000 |
| Median cost / latency | $0.36 / 196s |

**Do not quote those.** Three independent reasons, each sufficient on its own:

1. **Eight cases is a subset**, and `eval/subset.py` prints so above every run.
2. **The tools were changed in response to these eight failing.** The
   string-baseline fix below came directly from watching G-004 and G-005 go
   wrong. That is tuning, and a number measured on the cases you tuned against
   is optimistic by construction. `CLAUDE.md` forbids tuning against the
   held-back set for exactly this reason; the tuning set is where tuning is
   *supposed* to happen, and the cost is that its number stops being a score.
3. **Only one of the two improvements is attributable.** G-005 read the
   corrected string figure and moved off `string_outage`, which is the predicted
   mechanism. G-029 also improved, but took a different measurement path —
   it ran `check_night_offset`, which it had skipped before — so it may simply
   have varied. LLM nodes are not deterministic (`CLAUDE.md#determinism`) and
   one run per case cannot separate a fix from a coin.

What would make it a score: the full tuning split, then the held-back split,
reported separately with the gap stated.

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY
python -m eval.runner run --engine agent --split tuning \
    --checks-only --resume runs/tuning.jsonl
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

### The second batch, found once the agent actually ran

Everything above was found before a single LLM call was made. Running the agent
found a second set, and they share a shape worth naming: **thirteen build steps
produced a suite that passed while the LLM path had never executed once.** A
green suite says the code on that disk works. It does not say the repository
does.

**An even share is not this array's baseline.** Combiner 7 — already flagged
under *Still open* as unresolvable without a maintenance log — carries about 12%
less than its neighbours across the entire record. Two tools measured against a
theoretical even 1/7 and reported that standing characteristic as a deficit in
every window ever scored. G-004 and G-005 both read "the lowest share is string
7" and committed to `string_outage`; their true causes were soiling and seasonal
derating, and the second is a false alarm on a look-alike. Both tools now
compare each string against its own history: a permanent offset reads −0.003
where a real fault on the same string reads −0.297.

**The provenance ledger overwrote itself.** Keys were `{tool}.{value}`, so a
tool run twice kept only the last call. One case ran `per_mppt_current_balance`
four times over different windows and had the six figures its answer quoted from
the first two reported as fabricated — the *more* thoroughly a case was
measured, the less grounded it looked, and unsupported claims are a hard veto.

**A fall printed as a rise.** `characterize_onset` computed `before - after` and
emitted it as a signed "change", so a metric going 0.891 → 0.794 read "a change
of +0.097" in a field named `absolute_change`. Its own test asserted `> 0.1` on
a drop, so the test agreed with the bug.

**A streamed error arrives inside a 200.** The SDK raises a mid-stream failure
against the *stream's* HTTP response, which succeeded, so an `overloaded_error`
reached the retry logic as a bare `APIStatusError` with `status_code == 200` and
was classified as permanent. Two correct changes — retry by status code, and
stream every call because the token ceilings made non-streaming illegal — left a
blind spot between them that nothing tested, because no test built the exception
the streaming path actually raises.

**The critic could not see its own previous review.** Its prompt held the brief,
the causes, the evidence, the draft and the look-alike checklist, and nothing
about what it had asked for last time — while the planner and synthesiser both
received the revision request. Information flowed one way. Every review was a
fresh reviewer with unlimited standards and no memory, so it re-raised answered
objections and could not notice the answer had stopped changing. Four cases
reached the right cause and were talked out of it.

**Traces concatenated across runs.** `TraceWriter` opens with mode `"a"` and the
evaluation names traces after the case, so re-running appends. `INV-G-004`
reported 59 measurements and $3.37 while the run that wrote its last lines took
11 and cost $0.34.

**A cache hit was quoted as a price.** A replayed run reported the original
call's cost against zero seconds of wall clock. `LLMResponse.cached` existed and
nothing downstream read it.

**Review cost was never written to the tape.** It was accrued to the run total
and omitted from the trace step, so the node whose value is most in question
appeared free to anything reading a trace.

**The category was an LLM decision that is a dictionary lookup.**
`fault_signatures.yaml` assigns a category to every cause; the synthesiser chose
both independently and could contradict a file it had been shown. One case in
eight named the right cause and filed it in the wrong bucket.

The general lesson, since it recurred: **half of these were invisible until
something was built to look at data that had already been recorded.** The
evaluation harness is not only for producing scores — `profile` and `explain`
both exist because a number was wrong and nothing could say why.

---

## Still open

- **The agent has not been scored.** Eight tuning cases have been run, and they
  are the cases the tools were tuned against. See *Status*.
- **Retrieval has not been ablated, and is not in the loop.** `investigate`
  defaults to the hand-written `KnowledgeBase` — a dictionary lookup on cause
  names — so every "looked up what is known about…" line in every run log is
  that lookup, not retrieval. `CorpusRetriever` is built, tested, measurably
  better than BM25 alone, and constructed by nothing. The corpus is also 23
  chunks and *is* the knowledge base. Making the claim real needs real
  documents behind it.
- **The LLM reviewer has been priced on one case, not measured.** G-017 with it
  took 644–900s and $1.09–1.31 and ended on "not enough evidence"; with
  `--no-review` it answered correctly in 165s for $0.29. That is n=1 on the case
  the reviewer had already failed three times. Across every case seen so far it
  has improved one answer and talked four correct ones out of themselves, which
  is why `--checks-only` exists — the deterministic half of review kept, the
  judgement dropped.
- **Whether the reasoning is sound, as opposed to the evidence being correct.**
  G-004 is the open case, and reading it in full is uncomfortable: the agent
  stated the right discriminator unprompted ("a string outage shows exactly one
  step"), measured it three times, got three answers pointing at soiling, and
  cited none of them — quoting instead a figure from the same sentence as the
  disconfirming one. Fixing tools raises the floor. It does not establish that
  the reasoning above the floor is reliable.
- **Combiner 7.** Unresolvable without a maintenance log, and a live example of
  why `not_enough_evidence` needs to be a first-class answer. It is also the
  direct cause of the standing-offset bug above: a real plant characteristic
  that two tools reported as a fault.

---

## Retrieval (step 9)

```bash
python -m eval.runner retrieval
```

18 golden queries of three kinds over a 23-chunk corpus. **The corpus is the
project's own knowledge base**: no third-party documents were reachable from
this environment, and the repository ships checksums rather than text in any
case (`corpus/README.md`).

| configuration | top-1 | reciprocal rank | direct | discriminating | paraphrase |
| --- | --- | --- | --- | --- | --- |
| BM25 only | 0.389 | 0.625 | 0.806 | 0.611 | 0.458 |
| vectors only | 0.444 | 0.626 | **1.000** | 0.389 | 0.489 |
| hybrid, no rerank | 0.556 | 0.719 | 0.917 | 0.611 | 0.631 |
| **hybrid + rerank** | **0.556** | **0.728** | 0.917 | 0.611 | **0.657** |

### "Right document in top 10" is not reported, on purpose

It scores **1.000 for every configuration** — including ones that have learned
nothing — because 10 of 23 chunks is 43% of the corpus. Quoting it would be the
most flattering and least honest number in the project. `RetrievalReport`
detects the saturation, the ablation table drops the column, and reciprocal rank
is used instead. Growing the corpus is what makes the intended metric usable,
and the ingest path exists for exactly that.

### What the ablation says

**Fusion earns its place; the reranker barely does.** Hybrid beats either stage
alone by about 0.09 of reciprocal rank — a real margin. The reranker adds 0.009
on top, which on 18 queries is one query moving one position. On this evidence
the reranker is close to cost without benefit, and it stays in only because the
corpus is too small to conclude either way. That is a finding, not a hedge.

**The two stages fail in opposite directions, which is why fusing works.** The
vector stage is perfect on direct queries (1.000) and worst on discriminating
ones (0.389); BM25 is the reverse. Neither ordering is an accident: character
n-grams match a named cause almost exactly, and lose badly when the question is
"how do I tell these two apart" and the words are shared between both answers.

**Paraphrase queries are the weakest column and that is an honest limitation.**
No embedding API was reachable and no local transformer was available, so the
dense stage is character n-gram TF-IDF — a genuinely different signal from BM25
but *not* a semantic one. It will not match "the array is dirty" to "soiling"
the way a sentence embedding would. The paraphrase queries exist specifically so
that gap shows up as 0.657 rather than being taken on trust.

### What is still not measured

**Whether retrieval changes the agent's accuracy at all.** That is the ablation
that matters, it needs an API key, and `--no-knowledge` runs the identical loop
without it. The numbers above say retrieval finds the right passage; they say
nothing about whether the agent diagnoses better for having read it. If it turns
out not to, the layer should come out.

**And a larger caveat, which belongs here rather than in a footnote:
`CorpusRetriever` is not wired into any run.** `investigate` defaults to the
hand-written `KnowledgeBase`, so the ablation as it stands would compare a
dictionary lookup against no dictionary lookup. The corpus is also 23 chunks and
those chunks *are* the knowledge base, so the table above measures finding the
right entry among 23 items that could equally be fetched by key. The stack works
and is measurably better than BM25 alone; it is not doing the job the word RAG
implies, and saying so costs a talking point.

---

## The two experiments (step 11)

```bash
python -m eval.runner experiments --runs 3 --split heldback
```

**Not run at scale.** The harness is complete and tested, and one case has been
run through the "no review" arm by hand (see *Still open*). What follows is what
the experiments will report and why they are built that way.

Since they were designed, review gained a third mode. `--no-review` remains the
control — a clean single pass, because a control that quietly does some of the
work is not a control — and `--checks-only` keeps the deterministic half (no
fabricated numerics, every look-alike weighed, no abstention naming a tool the
agent owns and did not run) while dropping the LLM judgement. The interesting
comparison is now three-way rather than two.

### Reported as mean ± spread, never as one number

`CLAUDE.md` scopes determinism: physics, tools, injection and retrieval are
bitwise reproducible; the planner, router, synthesiser and critic are not. A
single figure from the non-deterministic half is not a result, so every headline
metric is the mean over N fresh runs with its sample spread beside it.

N = 3 is the brief's number and it is small — three runs give a spread that is
indicative, not a confidence interval. A run count of 1 prints a spread of
0.000, and the harness prints a note saying that means "measured once", not
"perfectly stable".

### The four configurations

| | review | knowledge |
| --- | --- | --- |
| full | on | on |
| no review | **off** | on |
| no knowledge | on | **off** |
| neither | off | off |

Each row removes exactly one thing from `full`, and a test asserts it: an
ablation that changes two things at once attributes nothing. All four run the
*identical* loop — `critic=False` and an empty `KnowledgeBase` are constructor
arguments, not separate code paths — which is what makes the difference between
two rows attributable to the layer rather than to the implementation.

### What each one answers

**"no review" prices the critic.** The critic is the one whose value is easiest
to assert and hardest to show. The number to read is `correct_abstention_rate`:
its whole job is refusing to accept an answer that has not excluded its
look-alikes, so if abstention does not move when it is removed, it is
decoration.

The measured profile complicates the premise. Across 13 timed runs the
synthesiser is **50.2%** of all latency and the planner **22.4%**; review's own
cost was never written to the trace, so it is unmeasured rather than small. What
*is* measured is that **48% of all time and half of all spend went after cycle
1** — the re-review loop — on runs where four cases had the right answer in
cycle 1 and were sent back.

**"no knowledge" prices retrieval.** Step 9 established that the right passage
comes back. This establishes whether the agent diagnoses better for having read
it. **If it does not, the layer should come out** — and that would be the more
interesting result of the two.

**"neither" checks the two are not substituting for each other.** If removing
either alone costs little but removing both costs a lot, they are covering the
same failure and one of them is redundant.

### The number to read first, in every table

`correct_abstention_rate`. The rules baseline scores **0.000** on it
structurally, so it is the one column where an agent has somewhere to be better
rather than merely different. Overall accuracy is reported and is deliberately
not the headline: a detector that alarms on any deficit scores well on clean
faults and is useless in the field.
