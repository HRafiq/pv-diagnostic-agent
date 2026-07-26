# Corpus

**Nothing in here is committed except this file and `manifest.json`.**

Third-party PV engineering documents — IEC 61724, NREL technical reports,
manufacturer application notes — make the best corpus for this system and are
copyrighted, so the repository ships checksums rather than text (CLAUDE.md,
repository hygiene). Drop `.txt` or `.md` files here and they join the corpus
automatically; `python -m eval.runner retrieval` reports which corpus it
measured and its hash.

With no documents present the corpus is the project's own knowledge base, which
is enough to run and to score but **too small for "right document in top 10" to
mean anything** — 10 of 23 chunks is 43% of everything. The report flags that
and falls back to reciprocal rank. Growing the corpus is what makes the
intended headline metric usable.
