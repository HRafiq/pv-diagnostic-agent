"""Scoring retrieval: the golden queries, the metrics, and the ablation.

The intended headline is **"right document in top 10"** — recall@10 in the
literature, and never called that in the interface (CLAUDE.md). It matches what
retrieval is for here: the agent reads what comes back, so a relevant chunk
anywhere in the returned set has done its job.

**On this corpus it measures nothing, and the report says so.** With 23 chunks,
top-10 returns 43% of everything and scores 1.000 for every configuration
including ones that have learned nothing. `RetrievalReport.top_k_is_saturated`
flags it and the ablation table drops the column rather than printing a row of
1.000s a reader might quote. Reciprocal rank and top-1 are the numbers to read
until the corpus is large enough for top-10 to discriminate — which is what the
document ingest is for.

The queries are deliberately of three kinds, and the split between them is the
interesting result:

* **direct** — the cause is named. Any lexical matcher gets these.
* **paraphrase** — the same question in a plant manager's words. This is where
  a character n-gram index and a real sentence embedding come apart, and the
  gap is the honest cost of having no embedding model available.
* **discriminating** — "how do I tell these two apart". The hardest, and the
  one the agent most needs, because a fault is only ever diagnosed against its
  nearest look-alike.

Nothing here is tuned against these queries. They were written from the domain,
not from watching the retriever fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.rag.index import HybridIndex, RetrievalConfig

__all__ = [
    "GOLDEN_QUERIES",
    "RetrievalQuery",
    "RetrievalReport",
    "ablation",
    "score_retrieval",
]


@dataclass(frozen=True)
class RetrievalQuery:
    """One query and the chunks that would answer it.

    `relevant` holds chunk ids. Ids are stable across corpus rebuilds by
    construction, so these labels survive a paragraph being added.
    """

    id: str
    query: str
    relevant: tuple[str, ...]
    kind: str  # direct | paraphrase | discriminating
    note: str = ""


GOLDEN_QUERIES: tuple[RetrievalQuery, ...] = (
    # --- direct: the cause is named -------------------------------------
    RetrievalQuery(
        "R-001",
        "string outage signature",
        ("signature:string_outage",),
        "direct",
    ),
    RetrievalQuery(
        "R-002",
        "soiling on the modules",
        ("signature:soiling",),
        "direct",
    ),
    RetrievalQuery(
        "R-003",
        "inverter clipping at its power limit",
        ("signature:clipping",),
        "direct",
    ),
    RetrievalQuery(
        "R-004",
        "telemetry gap in the logger",
        ("signature:telemetry_gap",),
        "direct",
    ),
    RetrievalQuery(
        "R-005",
        "seasonal temperature derating",
        ("signature:seasonal_temperature_derating",),
        "direct",
    ),
    RetrievalQuery(
        "R-006",
        "irradiance sensor drift",
        ("signature:sensor_drift",),
        "direct",
    ),
    # --- paraphrase: the same question, no jargon ------------------------
    RetrievalQuery(
        "R-101",
        "one part of the array is producing nothing while the rest is fine",
        ("signature:string_outage",),
        "paraphrase",
        "No cause named. A lexical matcher has to work from 'part of the array'.",
    ),
    RetrievalQuery(
        "R-102",
        "output drops for a few hours every morning then recovers",
        ("signature:shading",),
        "paraphrase",
        "The time-of-day signature described without the word 'shading'.",
    ),
    RetrievalQuery(
        "R-103",
        "performance has been slipping for weeks but jumps back up after rain",
        ("signature:soiling",),
        "paraphrase",
        "The sawtooth described in plain words.",
    ),
    RetrievalQuery(
        "R-104",
        "the performance ratio is going up, has the plant improved",
        ("signature:sensor_drift",),
        "paraphrase",
        "A rising ratio is the one thing a real loss cannot do.",
    ),
    RetrievalQuery(
        "R-105",
        "power stops at the same level every clear midday",
        ("signature:clipping", "signature:curtailment"),
        "paraphrase",
        "Deliberately ambiguous: both answers are correct.",
    ),
    RetrievalQuery(
        "R-106",
        "energy is down this month but nothing seems broken",
        ("signature:weather", "signature:seasonal_temperature_derating"),
        "paraphrase",
    ),
    # --- discriminating: the question the agent actually asks ------------
    RetrievalQuery(
        "R-201",
        "how do I tell soiling from a drifting irradiance sensor",
        ("distinguish:soiling__sensor_drift",),
        "discriminating",
        "The pair that gets a wash crew sent to a clean array.",
    ),
    RetrievalQuery(
        "R-202",
        "telling a shadow apart from a failed string",
        ("distinguish:string_outage__shading",),
        "discriminating",
    ),
    RetrievalQuery(
        "R-203",
        "can clipping and curtailment be separated from plant data",
        ("distinguish:clipping__curtailment",),
        "discriminating",
        "The answer is no, and the corpus has to say so.",
    ),
    RetrievalQuery(
        "R-204",
        "is this a loss at the inverter or on the array",
        ("distinguish:inverter_fault__string_outage",),
        "discriminating",
    ),
    RetrievalQuery(
        "R-205",
        "distinguishing a data gap from a real fault",
        ("distinguish:telemetry_gap__string_outage",),
        "discriminating",
    ),
    RetrievalQuery(
        "R-206",
        "summer decline: heat or dirt",
        ("distinguish:seasonal_temperature_derating__soiling",),
        "discriminating",
    ),
)


@dataclass
class RetrievalReport:
    """Per-kind and overall retrieval quality for one configuration."""

    configuration: str
    corpus_hash: str = ""
    queries: int = 0
    corpus_chunks: int = 0
    k: int = 10
    by_kind: dict[str, dict[str, float]] = field(default_factory=dict)
    overall: dict[str, float] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)

    @property
    def top_k_is_saturated(self) -> bool:
        """Is "right document in top k" measuring anything on this corpus?

        Returning k of N chunks when k is a large fraction of N is close to
        returning the corpus. On a 23-chunk corpus, top-10 hands back 43% of
        everything and scores 1.000 for a retriever that has learned nothing.
        Quoting that as evidence would be the single most flattering and least
        honest number in the project.
        """
        return self.corpus_chunks > 0 and self.k / self.corpus_chunks > 0.2

    @property
    def headline_metric(self) -> str:
        """Which number to actually read for this corpus size."""
        return "reciprocal_rank" if self.top_k_is_saturated else f"top_{self.k}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration,
            "corpus_hash": self.corpus_hash,
            "corpus_chunks": self.corpus_chunks,
            "queries": self.queries,
            "overall": self.overall,
            "by_kind": self.by_kind,
            "queries_with_nothing_relevant_returned": self.misses,
            "headline_metric": self.headline_metric,
            "top_k_saturated": self.top_k_is_saturated,
        }


def _hit_at_k(returned: list[str], relevant: tuple[str, ...], k: int) -> float:
    return 1.0 if set(returned[:k]) & set(relevant) else 0.0


def _reciprocal_rank(returned: list[str], relevant: tuple[str, ...]) -> float:
    for position, chunk_id in enumerate(returned, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def score_retrieval(
    index: HybridIndex,
    queries: tuple[RetrievalQuery, ...] = GOLDEN_QUERIES,
    k: int = 10,
) -> RetrievalReport:
    """Score one index configuration over the golden queries."""
    report = RetrievalReport(
        configuration=index.config.label,
        corpus_hash=index.snapshot_hash[:12],
        queries=len(queries),
        corpus_chunks=len(index),
        k=k,
    )
    rows: list[dict[str, Any]] = []
    for query in queries:
        returned = [hit.chunk.id for hit in index.search(query.query, limit=k)]
        row = {
            "kind": query.kind,
            "top_10": _hit_at_k(returned, query.relevant, k),
            "top_3": _hit_at_k(returned, query.relevant, 3),
            "top_1": _hit_at_k(returned, query.relevant, 1),
            "reciprocal_rank": _reciprocal_rank(returned, query.relevant),
        }
        rows.append(row)
        if row["top_10"] == 0.0:
            report.misses.append(f"{query.id}: {query.query}")

    frame = pd.DataFrame(rows)
    if frame.empty:
        return report

    metrics = ["top_10", "top_3", "top_1", "reciprocal_rank"]
    report.overall = {m: round(float(frame[m].mean()), 4) for m in metrics}
    for kind, group in frame.groupby("kind"):
        report.by_kind[str(kind)] = {
            **{m: round(float(group[m].mean()), 4) for m in metrics},
            "queries": float(len(group)),
        }
    return report


# The configurations the ablation compares. Each removes one stage, so the
# difference between two rows is attributable to that stage and nothing else.
ABLATION_CONFIGS: tuple[tuple[str, RetrievalConfig], ...] = (
    ("bm25 only", RetrievalConfig(use_vectors=False, use_rerank=False)),
    ("vectors only", RetrievalConfig(use_bm25=False, use_rerank=False)),
    ("hybrid, no rerank", RetrievalConfig(use_rerank=False)),
    ("hybrid + rerank", RetrievalConfig()),
)


def ablation(chunks: list[Any], k: int = 10) -> pd.DataFrame:
    """Score every configuration over the same corpus and queries.

    The table is the deliverable. If hybrid does not beat BM25 on this corpus,
    that is the finding and it gets published — the extra stage is cost without
    benefit and should come out.
    """
    rows: list[dict[str, Any]] = []
    saturated = False
    for label, config in ABLATION_CONFIGS:
        report = score_retrieval(HybridIndex(chunks, config), k=k)
        saturated = saturated or report.top_k_is_saturated
        row: dict[str, Any] = {"configuration": label, **report.overall}
        for kind, values in sorted(report.by_kind.items()):
            row[f"{kind}_rr"] = values["reciprocal_rank"]
        row["missed"] = len(report.misses)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if saturated:
        # Drop the saturated column rather than print a table of 1.000s that a
        # reader might quote. What is not measurable on this corpus should not
        # appear as though it were.
        frame = frame.drop(columns=[f"top_{k}"], errors="ignore")
        frame.attrs["note"] = (
            f"top-{k} omitted: it returns {k} of {len(chunks)} chunks, so it "
            "scores 1.000 for any retriever and measures nothing. Read "
            "reciprocal rank and top-1."
        )
    return frame
