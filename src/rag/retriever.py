"""Adapters that let the loop take its evidence from either knowledge source.

The agent asks for two things — "what is known about these causes" and "what
separates this pair" — and it should not care whether the answer came from a
hand-written YAML file or from retrieval over a document corpus. Both satisfy
the same two-method interface, so swapping them is a constructor argument and
the step 9 ablation is a real comparison rather than two different code paths.

`KnowledgeBase` already implements the interface. `CorpusRetriever` wraps a
`HybridIndex` in it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.rag.corpus import build_corpus
from src.rag.index import HybridIndex, RetrievalConfig

__all__ = ["CorpusRetriever", "Retriever", "build_retriever"]


@runtime_checkable
class Retriever(Protocol):
    """What the loop needs from a knowledge source."""

    def match(self, phrases: list[str], limit: int = 6) -> list[str]:
        """Map free-text candidate causes onto retrievable keys."""
        ...

    def brief_for(self, causes: list[str]) -> str:
        """Evidence about those causes, as text the agent reads."""
        ...


class CorpusRetriever:
    """A `HybridIndex` behind the knowledge interface.

    `match` returns chunk ids rather than cause names, and `brief_for` looks
    those ids back up. That indirection is what keeps the loop identical across
    both sources: it passes whatever `match` gave it straight back, and never
    inspects the strings.
    """

    def __init__(self, index: HybridIndex, per_cause: int = 3) -> None:
        self.index = index
        self.per_cause = per_cause
        self._by_id = {chunk.id: chunk for chunk in index.chunks}

    def match(self, phrases: list[str], limit: int = 6) -> list[str]:
        seen: list[str] = []
        for phrase in phrases:
            for hit in self.index.search(phrase, limit=self.per_cause):
                if hit.chunk.id not in seen:
                    seen.append(hit.chunk.id)
        return seen[:limit]

    def brief_for(self, causes: list[str]) -> str:
        chunks = [self._by_id[c] for c in causes if c in self._by_id]
        if not chunks:
            return "(no knowledge for these causes)"
        return "\n\n".join(f"[{c.id}] {c.citation}\n{c.text.strip()}" for c in chunks)

    def unresolvable_pairs(self) -> tuple[tuple[str, str], ...]:
        return ()


def build_retriever(config: RetrievalConfig | None = None) -> CorpusRetriever:
    """Build a retriever over the whole corpus, knowledge base included."""
    chunks, _ = build_corpus()
    return CorpusRetriever(HybridIndex(chunks, config))
