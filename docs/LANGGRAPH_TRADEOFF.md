# Plain loop or LangGraph?

Both exist. `src/agent/loop_plain.py` is ~790 lines of explicit Python;
`src/agent/loop_graph.py` is the same loop as a LangGraph state machine, and
every node in it calls the identical function the plain loop calls.

`tests/test_langgraph_port.py` is the gate: given the same scripted replies the
two must produce the same tools, the same tape, the same costs and the same
finding. Eighteen tests, exact equality, no tolerance. **The plain loop is the
specification.** If they diverge, the graph is wrong.

That comparison is only possible because `ScriptedClient` makes a run
deterministic. Against a live model the two would differ for reasons that have
nothing to do with orchestration, and this document would be an opinion.

---

## A correction, because this document got its own headline wrong

The first version of this file said:

> **Checkpointing, and it is the one that matters.** An 86-case evaluation that
> dies at case 60 currently starts again from case 1. LangGraph gets resumable
> state for free […] it is the only capability on this list the plain loop does
> not already have.

**That was false in two separate ways, and it was the entire justification for
keeping the port.**

*First, there was no checkpointer.* `build_graph` called `graph.compile()` with
no saver, no thread id, only a recursion limit. The capability the port was
justified by had never been wired up, and the file sat unread for as long as the
port sat unused — nothing imported `investigate_with_graph` except its own
tests, so every run this project ever did used the plain loop.

*Second, and worse: wiring it in does not deliver what the sentence promised.*
The saver now in place is `InMemorySaver`. It records every node boundary
**within one investigation** and dies with the process. The failures that
actually cost this project cases — an API overload at case 6, an exhausted
credit balance mid-answer — take the interpreter with them, and nothing in
memory outlives that. A run that dies at case 60 still starts again.

What does survive is `eval/runner.py --resume`: a journal that appends one JSON
object per case as it finishes and skips the ones already recorded. Thirty
lines. No framework.

Durable within-investigation resume is available — it needs
`langgraph-checkpoint-sqlite`, which is not a dependency here — but that is a
different and much smaller claim than the one this file made.

**The lesson is the one the project keeps relearning**, and it is more pointed
here than anywhere else: a document written specifically to hold a framework to
a measured standard asserted the framework's headline benefit without ever
exercising it. Writing "measured claim rather than assumption" does not make it
one.

---

## What the framework actually bought

With checkpointing corrected, the list is shorter and more modest.

**The graph is data, not control flow.** `plan → retrieve → route → execute` is
an object that can be drawn and inspected. In the plain loop that structure
exists only as code you have to read to know. For a reviewer, an operator, or a
diagram, that is a genuine difference.

**Branches are named rather than implied.** `after_route` returning `"execute"`,
`"look_up"` or `"synthesise"` is exactly the `if` the plain loop contains, but as
a labelled transition. Three named outcomes are easier to reason about — and
easier to test in isolation — than three `break`s at different indentation.

**Within-run resume is possible where the plain loop makes it impossible.** Not
delivered, but reachable: a durable saver is a dependency away, and the plain
loop would need it written by hand. This is the honest residue of the original
checkpointing claim.

---

## What it cost

**The bound moved and had to be restated.** The plain loop's `for` over the
per-cycle allowance became a self-edge with the cap enforced in a conditional
edge, plus a recursion limit as a backstop. Two mechanisms where there was one,
and they have to be kept consistent or the two loops stop for different reasons
— which the tests check, because the first version of the port got it wrong.

**State stopped being one object.** LangGraph copies state between nodes rather
than threading one instance through. The port initially read its local
`AgentState` back after `invoke()` and reported an untouched run: no tools
called, no cycles used. Every equivalence test failed at once, which is the
system working — but in a hand-written loop the bug does not exist to be made.

**Every behaviour change now costs two edits.** Convergence detection, the
three review modes, the withdrawn commitment on `not_enough_evidence`, the
critic's own history — each had to be written twice and kept identical, with the
equivalence test as the only thing standing between "ported" and "diverged". On
a loop under active change this is the largest cost by some distance, and it was
not visible when the port was written against a loop that had stopped moving.

**The inner loop is harder to read.** A `for _ in range(cap)` with three `break`s
is more legible than four nodes and a conditional edge, at this size.

**Stack traces pass through the framework.** A tool error in the plain loop
points at the tool. In the graph it points at the tool via two frames of
LangGraph.

**A dependency's warnings become yours.** LangGraph's checkpoint package emits a
pending-deprecation notice about a serialiser default this project neither
constructs nor configures. It began appearing above every evaluation run the
moment the runner started importing the graph, where it reads as something the
operator did wrong, and had to be filtered by message in `eval/runner.main`.

---

## The finding

**On a loop this small, without checkpointing, the plain loop wins on
legibility — and the port is what runs anyway.**

That reads like a contradiction and is not. Both are true:

- The framework's headline benefit was never delivered. Named branches and a
  declared graph are real gains and would not, alone, have justified a second
  implementation.
- A second implementation that nobody runs is worse than either choice. It rots,
  its claims go unchecked — as this file demonstrates — and the equivalence test
  passing tells you only that two things nobody uses still agree. So the runner
  and the watcher now import the graph, and the plain loop remains the
  specification and the reference the tests are written against.

If the project were starting again knowing this, the honest answer is **plain
loop only**, plus the thirty-line journal, until something genuinely needs
durable within-run resume. The value of having built both is not the graph. It
is that "did the framework buy anything?" became answerable, and the answer
turned out to be *less than the file claiming to answer it said*.

That is the argument for writing the plain version first (DECISION 0018),
strengthened rather than weakened by the correction above. Had the project
started with LangGraph, "the framework handles it" would have remained an
assumption — exactly as it did, in writing, in this document, for as long as
nobody checked.

**Both are kept.** `loop_plain` is the specification; `loop_graph` is what runs.
Keeping both costs one test file and a second edit on every behaviour change,
and that test file is what makes either of them trustworthy.
