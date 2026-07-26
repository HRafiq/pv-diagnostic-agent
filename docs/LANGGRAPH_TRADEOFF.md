# Plain loop or LangGraph?

Both exist. `src/agent/loop_plain.py` is ~330 lines of explicit Python;
`src/agent/loop_graph.py` is the same loop as a LangGraph state machine, and
every node in it calls the identical function the plain loop calls.

`tests/test_langgraph_port.py` is the gate: given the same scripted replies the
two must produce the same tools, the same tape, the same costs and the same
finding. Sixteen tests, exact equality, no tolerance. **The plain loop is the
specification.** If they diverge, the graph is wrong.

That comparison is only possible because `ScriptedClient` makes a run
deterministic. Against a live model the two would differ for reasons that have
nothing to do with orchestration, and this document would be an opinion.

---

## What the framework actually bought

**Checkpointing, and it is the one that matters.** An 86-case evaluation that
dies at case 60 currently starts again from case 1. LangGraph gets resumable
state for free; the plain loop would need it written by hand. This is not a
hypothetical: the run costs real money and takes real time, and it is the only
capability on this list the plain loop does not already have.

**The graph is data, not control flow.** `plan → retrieve → route → execute` is
an object that can be drawn, inspected and checkpointed. In the plain loop that
structure exists only as code you have to read to know. For a reviewer, an
operator, or a diagram in a README, that is a genuine difference.

**Branches are named rather than implied.** `after_route` returning `"execute"`,
`"look_up"` or `"synthesise"` is exactly the `if` the plain loop contains, but as
a labelled transition. Three named outcomes are easier to reason about — and
easier to test in isolation — than three `break`s at different indentation.

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

**The inner loop is harder to read.** A `for _ in range(cap)` with three `break`s
is more legible than four nodes and a conditional edge, at this size.

**Stack traces pass through the framework.** A tool error in the plain loop
points at the tool. In the graph it points at the tool via two frames of
LangGraph.

---

## The finding

**On a loop this small it is close to an even trade, and checkpointing is what
tips it.** The declared graph and named branches are real but modest gains; the
extra bound and the copied state are real but modest costs. If the evaluation
did not need to survive a crash halfway through, the plain loop would win on
legibility alone.

Which is why writing the plain version first was the right order (DECISION
0018). Had the project started with LangGraph, none of the above would be
knowable — "the framework handles it" would be an assumption rather than a
measured claim, and the copied-state bug would have been invisible because there
would have been nothing to compare against.

**Both are kept.** `loop_plain` is the reference implementation and what the
tests are written against; `loop_graph` is what an evaluation run over 86 cases
should use once it needs to resume. Keeping both costs one test file, and that
test file is what makes either of them trustworthy.
