"""Retrieval: BM25, a vector index, reciprocal-rank fusion, and a reranker.

Built as four separable stages because the point of step 9 is to *measure*
whether retrieval is worth anything, and a monolith cannot be ablated. Each
stage can be switched off and the golden set re-scored, so "hybrid beats BM25"
is a number rather than an assertion.

**On the vector index.** There is no embedding API reachable from this
environment and no local transformer, so the dense stage is character n-gram
TF-IDF with cosine similarity — a genuinely different retrieval signal from
BM25 (it matches on sub-word overlap, so "pyranometer" finds "pyranometers" and
"soiled sensor" finds "sensor soiling") but *not* a semantic one. It will not
match "the array is dirty" to "soiling" the way a sentence embedding would.
That limitation is real, it is stated here rather than buried, and the
retrieval golden set includes paraphrase queries specifically so the gap shows
up in the numbers instead of being taken on trust.

Everything here is deterministic: identical corpus, identical query, identical
ranking, every time. That is what lets the retrieval metric be a fixed number
while the LLM path around it is not (CLAUDE.md).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from src.determinism import stable_hash

__all__ = [
    "Chunk",
    "Hit",
    "HybridIndex",
    "RetrievalConfig",
    "tokenize",
]

_WORD = re.compile(r"[a-z0-9]+")
# Words that appear in nearly every chunk of a PV corpus carry no signal and
# actively hurt BM25 by inflating short chunks that happen to contain them.
# fmt: off
_STOPWORDS = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "do", "does", "for", "from", "had", "has", "have", "if", "in",
    "into", "is", "it", "its", "may", "more", "most", "no", "not", "of", "on",
    "or", "that", "the", "their", "then", "there", "these", "they", "this",
    "to", "was", "were", "what", "when", "which", "while", "who", "will",
    "with", "would",
])
# fmt: on


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]


def _char_ngrams(text: str, n: int = 4) -> Counter[str]:
    cleaned = re.sub(r"\s+", " ", text.lower())
    return Counter(cleaned[i : i + n] for i in range(max(0, len(cleaned) - n + 1)))


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage, with enough provenance to cite it."""

    id: str
    text: str
    source: str
    title: str = ""
    causes: tuple[str, ...] = ()

    @property
    def citation(self) -> str:
        return f"{self.source}" + (f" — {self.title}" if self.title else "")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    stage: str = "hybrid"


@dataclass(frozen=True)
class RetrievalConfig:
    """Which stages run. Every flag exists so it can be turned off and scored."""

    use_bm25: bool = True
    use_vectors: bool = True
    use_rerank: bool = True
    top_k: int = 10
    candidates: int = 30
    # BM25's usual defaults. k1 controls term-frequency saturation, b how much
    # length normalisation is applied.
    k1: float = 1.5
    b: float = 0.75
    rrf_k: int = 60

    @property
    def label(self) -> str:
        parts = [
            name
            for name, on in (
                ("bm25", self.use_bm25),
                ("vectors", self.use_vectors),
                ("rerank", self.use_rerank),
            )
            if on
        ]
        return "+".join(parts) or "none"


class HybridIndex:
    """BM25 and a character n-gram vector index, fused and optionally reranked.

    Built once from a corpus snapshot. `snapshot_hash` identifies that corpus
    exactly, so a retrieval metric can be quoted alongside the corpus it was
    measured on — a number from a different snapshot is a different number.
    """

    def __init__(self, chunks: Iterable[Chunk], config: RetrievalConfig | None = None):
        self.chunks: list[Chunk] = list(chunks)
        self.config = config or RetrievalConfig()
        self._tokens: list[list[str]] = [tokenize(c.text) for c in self.chunks]
        self._lengths = [len(t) for t in self._tokens]
        self._avg_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        self._term_frequencies = [Counter(t) for t in self._tokens]
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            self._document_frequency.update(set(tokens))
        self._ngrams = [_char_ngrams(c.text) for c in self.chunks]
        self._norms = [
            math.sqrt(sum(v * v for v in grams.values())) or 1.0
            for grams in self._ngrams
        ]

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def snapshot_hash(self) -> str:
        """Identifies the corpus this index was built from."""
        return stable_hash(*[f"{c.id}:{c.text}" for c in self.chunks])

    # ------------------------------------------------------------------
    def bm25(self, query: str, limit: int) -> list[Hit]:
        terms = tokenize(query)
        total = len(self.chunks)
        if not terms or not total:
            return []
        scored: list[Hit] = []
        for index, chunk in enumerate(self.chunks):
            frequencies = self._term_frequencies[index]
            length = self._lengths[index] or 1
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                idf = math.log(
                    1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self.config.k1 * (
                    1
                    - self.config.b
                    + self.config.b * length / (self._avg_length or 1.0)
                )
                score += idf * frequency * (self.config.k1 + 1) / denominator
            if score > 0:
                scored.append(Hit(chunk, score, "bm25"))
        return _stable_top(scored, limit)

    def vectors(self, query: str, limit: int) -> list[Hit]:
        grams = _char_ngrams(query)
        norm = math.sqrt(sum(v * v for v in grams.values())) or 1.0
        scored: list[Hit] = []
        for index, chunk in enumerate(self.chunks):
            other = self._ngrams[index]
            shared = grams.keys() & other.keys()
            if not shared:
                continue
            dot = sum(grams[g] * other[g] for g in shared)
            scored.append(Hit(chunk, dot / (norm * self._norms[index]), "vectors"))
        return _stable_top(scored, limit)

    # ------------------------------------------------------------------
    def search(self, query: str, limit: int | None = None) -> list[Hit]:
        """Run the configured stages and return the ranked result."""
        k = limit or self.config.top_k
        candidates = max(self.config.candidates, k)

        rankings: list[list[Hit]] = []
        if self.config.use_bm25:
            rankings.append(self.bm25(query, candidates))
        if self.config.use_vectors:
            rankings.append(self.vectors(query, candidates))
        if not rankings:
            # No retrieval stage enabled. Returning the first k chunks would be
            # a silent lie about what was retrieved, so return nothing.
            return []

        fused = _reciprocal_rank_fusion(rankings, self.config.rrf_k)
        if self.config.use_rerank:
            fused = self._rerank(query, fused)
        return fused[:k]

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """A cheap lexical reranker over the fused candidates.

        Not a cross-encoder — there is no model here. It rescores on exact
        phrase overlap and on how much of the query's vocabulary a chunk
        covers, which is precisely the signal rank fusion throws away when it
        reduces each list to a position. Whether that is worth its place is a
        question for the ablation, not for this docstring.
        """
        terms = set(tokenize(query))
        lowered = query.lower().strip()
        rescored: list[Hit] = []
        for position, hit in enumerate(hits):
            text = hit.chunk.text.lower()
            coverage = (
                len(terms & set(tokenize(hit.chunk.text))) / len(terms)
                if terms
                else 0.0
            )
            phrase = 1.0 if lowered and lowered in text else 0.0
            # Keep the fusion order as a tie-breaker rather than discarding it.
            prior = 1.0 / (1 + position)
            rescored.append(
                Hit(hit.chunk, 2.0 * coverage + 1.5 * phrase + prior, "rerank")
            )
        return _stable_top(rescored, len(rescored))

    def brief(self, query: str, limit: int | None = None) -> str:
        """Retrieved passages, formatted for a prompt, each with its citation."""
        hits = self.search(query, limit)
        if not hits:
            return "(nothing retrieved)"
        return "\n\n".join(
            f"[{hit.chunk.id}] {hit.chunk.citation}\n{hit.chunk.text.strip()}"
            for hit in hits
        )


def _stable_top(hits: list[Hit], limit: int) -> list[Hit]:
    """Highest score first, ties broken by chunk id.

    The tie-break is not cosmetic: without it two chunks with identical scores
    swap places between runs depending on dict ordering, and a retrieval metric
    that moves without the corpus moving is unusable as a regression check.
    """
    return sorted(hits, key=lambda h: (-h.score, h.chunk.id))[:limit]


def _reciprocal_rank_fusion(rankings: list[list[Hit]], k: int) -> list[Hit]:
    """Combine several rankings by position rather than by score.

    BM25 scores and cosine similarities are not on the same scale and cannot be
    added. Fusing on rank sidesteps the question entirely, which is why it beats
    score normalisation in practice — there is no scale to get wrong.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for position, hit in enumerate(ranking):
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + 1.0 / (
                k + position + 1
            )
            chunks[hit.chunk.id] = hit.chunk
    fused = [Hit(chunks[cid], score, "fused") for cid, score in scores.items()]
    return _stable_top(fused, len(fused))


@dataclass
class CorpusStats:
    chunks: int = 0
    sources: int = 0
    tokens: int = 0
    by_source: dict[str, int] = field(default_factory=dict)


def corpus_stats(chunks: list[Chunk]) -> CorpusStats:
    by_source: dict[str, int] = {}
    tokens = 0
    for chunk in chunks:
        by_source[chunk.source] = by_source.get(chunk.source, 0) + 1
        tokens += len(tokenize(chunk.text))
    return CorpusStats(
        chunks=len(chunks),
        sources=len(by_source),
        tokens=tokens,
        by_source=by_source,
    )
