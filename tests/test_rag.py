"""Retrieval: the index, the corpus, and the ablation that prices it.

The point of step 9 is to *measure* whether retrieval is worth anything, so
most of these tests are about whether the measurement is trustworthy — stable
ranking, stable chunk ids, an honest headline metric — rather than about
whether any particular query happens to work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.retrieval import (
    GOLDEN_QUERIES,
    RetrievalReport,
    ablation,
    score_retrieval,
)
from src.agent.llm import ScriptedClient
from src.agent.loop_plain import investigate
from src.rag.corpus import build_corpus, chunk_document, knowledge_chunks
from src.rag.index import Chunk, HybridIndex, RetrievalConfig, corpus_stats, tokenize
from src.rag.retriever import CorpusRetriever, Retriever, build_retriever


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    built, _ = build_corpus()
    return built


@pytest.fixture(scope="module")
def index(chunks: list[Chunk]) -> HybridIndex:
    return HybridIndex(chunks)


# ===========================================================================
# The corpus
# ===========================================================================
def test_the_knowledge_base_is_always_available_as_a_corpus() -> None:
    """A clone with no ingested documents must still retrieve something."""
    assert len(knowledge_chunks()) > 10


def test_chunk_ids_are_stable_across_rebuilds() -> None:
    """The retrieval golden set refers to chunks by id.

    An id that shifts when a paragraph is added silently invalidates every
    label in that set, and the metric quietly starts measuring nothing.
    """
    assert [c.id for c in knowledge_chunks()] == [c.id for c in knowledge_chunks()]


def test_every_golden_query_points_at_a_chunk_that_exists(
    chunks: list[Chunk],
) -> None:
    """A label pointing at a missing chunk scores zero forever and looks like a
    retrieval failure."""
    available = {c.id for c in chunks}
    for query in GOLDEN_QUERIES:
        for chunk_id in query.relevant:
            assert chunk_id in available, f"{query.id} labels a missing {chunk_id}"


def test_the_corpus_names_no_golden_case(chunks: list[Chunk]) -> None:
    """The corpus must not be the answer key.

    If a chunk said "case G-001 is a string outage", retrieval would score
    perfectly and measure nothing.
    """
    for chunk in chunks:
        assert "G-0" not in chunk.text and "H-0" not in chunk.text


def test_chunking_overlaps_so_an_argument_is_not_cut_in_half() -> None:
    text = "\n\n".join(f"Paragraph {n} " + "word " * 60 for n in range(6))
    pieces = chunk_document(text, source="doc", overlap_words=20)
    assert len(pieces) > 1
    tail = pieces[0].text.split()[-20:]
    assert " ".join(tail) in pieces[1].text


def test_corpus_stats_counts_sources(chunks: list[Chunk]) -> None:
    stats = corpus_stats(chunks)
    assert stats.chunks == len(chunks)
    assert stats.sources >= 1
    assert stats.tokens > 0


def test_an_ingested_document_joins_the_corpus(tmp_path: Path) -> None:
    (tmp_path / "iec_61724.txt").write_text(
        "\n\n".join("Performance ratio is defined as " + "x " * 50 for _ in range(4))
    )
    built, files = build_corpus(tmp_path, include_knowledge=False)
    assert built and files
    assert files[0].sha256 and files[0].chunks == len(built)


def test_the_manifest_ships_checksums_not_documents(tmp_path: Path) -> None:
    """No third-party PDFs are committed (CLAUDE.md)."""
    from src.rag.corpus import write_manifest

    (tmp_path / "doc.txt").write_text("some public domain text about pyranometers")
    _, files = build_corpus(tmp_path, include_knowledge=False)
    path = write_manifest(files, tmp_path / "manifest.json")
    payload = path.read_text()
    assert "sha256" in payload
    assert "pyranometers" not in payload


# ===========================================================================
# The index
# ===========================================================================
def test_tokenising_drops_words_that_carry_no_signal() -> None:
    assert "the" not in tokenize("the array and the inverter")
    assert "inverter" in tokenize("the array and the inverter")


def test_bm25_finds_the_obviously_relevant_chunk(index: HybridIndex) -> None:
    top = index.bm25("string outage combiner fuse", limit=3)
    assert any(h.chunk.id == "signature:string_outage" for h in top)


def test_the_vector_stage_matches_sub_word_overlap(index: HybridIndex) -> None:
    """Character n-grams find 'pyranometers' from 'pyranometer' — which BM25,
    matching whole tokens, does not."""
    top = [h.chunk.id for h in index.vectors("pyranometers drifting", limit=5)]
    assert "signature:sensor_drift" in top


def test_ranking_is_stable_across_runs(index: HybridIndex) -> None:
    """A metric that moves without the corpus moving is unusable as a
    regression check."""
    first = [h.chunk.id for h in index.search("soiling versus sensor drift")]
    second = [h.chunk.id for h in index.search("soiling versus sensor drift")]
    assert first == second


def test_ties_break_on_chunk_id_not_dict_order() -> None:
    chunks = [
        Chunk(id="b", text="identical text here", source="s"),
        Chunk(id="a", text="identical text here", source="s"),
    ]
    hits = HybridIndex(chunks).search("identical text here")
    assert [h.chunk.id for h in hits] == ["a", "b"]


def test_the_snapshot_hash_changes_with_the_corpus() -> None:
    one = HybridIndex([Chunk(id="a", text="alpha", source="s")])
    two = HybridIndex([Chunk(id="a", text="beta", source="s")])
    assert one.snapshot_hash != two.snapshot_hash


def test_disabling_every_stage_returns_nothing_rather_than_the_first_k(
    chunks: list[Chunk],
) -> None:
    """Returning the head of the corpus would be a silent lie about what was
    retrieved."""
    off = RetrievalConfig(use_bm25=False, use_vectors=False, use_rerank=False)
    assert HybridIndex(chunks, off).search("anything") == []


def test_an_empty_index_returns_nothing() -> None:
    assert HybridIndex([]).search("string outage") == []


def test_the_brief_carries_a_citation(index: HybridIndex) -> None:
    """An agent quoting a document has to be able to say which one."""
    brief = index.brief("how do I tell soiling from sensor drift", limit=2)
    assert "project knowledge base" in brief
    assert "[" in brief  # chunk id, so a claim can be traced back


def test_the_configuration_label_names_the_stages() -> None:
    assert RetrievalConfig().label == "bm25+vectors+rerank"
    assert RetrievalConfig(use_vectors=False, use_rerank=False).label == "bm25"


# ===========================================================================
# The metric, and its honesty
# ===========================================================================
def test_top_k_is_flagged_as_saturated_on_a_small_corpus(
    index: HybridIndex,
) -> None:
    """Top-10 over 23 chunks returns 43% of everything and scores 1.000 for a
    retriever that has learned nothing. Quoting it would be the most flattering
    and least honest number in the project.
    """
    report = score_retrieval(index)
    assert report.top_k_is_saturated
    assert report.headline_metric == "reciprocal_rank"


def test_top_k_is_not_flagged_on_a_large_corpus() -> None:
    many = [
        Chunk(id=f"c{n:03d}", text=f"chunk about topic {n}", source="s")
        for n in range(200)
    ]
    report = score_retrieval(HybridIndex(many), queries=GOLDEN_QUERIES[:1])
    assert not report.top_k_is_saturated
    assert report.headline_metric == "top_10"


def test_the_ablation_drops_the_saturated_column(chunks: list[Chunk]) -> None:
    table = ablation(chunks)
    assert "top_10" not in table.columns
    assert "reciprocal_rank" in table.columns
    assert "measures nothing" in table.attrs["note"]


def test_the_ablation_compares_every_configuration(chunks: list[Chunk]) -> None:
    table = ablation(chunks)
    assert set(table["configuration"]) == {
        "bm25 only",
        "vectors only",
        "hybrid, no rerank",
        "hybrid + rerank",
    }


def test_fusing_two_stages_beats_either_alone_on_this_corpus(
    chunks: list[Chunk],
) -> None:
    """The result the ablation exists to produce. If it ever stops holding,
    the extra stage is cost without benefit and should come out."""
    table = ablation(chunks).set_index("configuration")["reciprocal_rank"]
    assert table["hybrid + rerank"] > table["bm25 only"]
    assert table["hybrid + rerank"] > table["vectors only"]


def test_paraphrase_queries_score_worse_than_direct_ones(
    index: HybridIndex,
) -> None:
    """The honest cost of having no embedding model available.

    A character n-gram index matches sub-word overlap, not meaning, so a
    question asked in a plant manager's words scores lower than one that names
    the cause. Stated as a test so it stays visible rather than being taken on
    trust.
    """
    by_kind = score_retrieval(index).by_kind
    assert (
        by_kind["paraphrase"]["reciprocal_rank"] < by_kind["direct"]["reciprocal_rank"]
    )


def test_the_report_records_which_corpus_it_measured(index: HybridIndex) -> None:
    """A retrieval number from a different snapshot is a different number."""
    report = score_retrieval(index)
    assert report.corpus_hash
    assert report.corpus_chunks == len(index)
    assert report.to_dict()["corpus_hash"] == report.corpus_hash


def test_an_empty_report_serialises() -> None:
    assert RetrievalReport(configuration="none").to_dict()["queries"] == 0


def test_the_golden_queries_cover_all_three_kinds() -> None:
    kinds = {q.kind for q in GOLDEN_QUERIES}
    assert kinds == {"direct", "paraphrase", "discriminating"}


# ===========================================================================
# The retriever behind the loop's interface
# ===========================================================================
def test_a_corpus_retriever_satisfies_the_same_interface_as_the_knowledge_base() -> (
    None
):
    """What makes the ablation a comparison rather than two code paths."""
    from src.knowledge import load_knowledge

    assert isinstance(build_retriever(), Retriever)
    assert isinstance(load_knowledge(), Retriever)


def test_the_retriever_returns_ids_the_brief_can_look_up() -> None:
    retriever = build_retriever()
    ids = retriever.match(["a string has failed", "dust on the modules"])
    assert ids
    brief = retriever.brief_for(ids)
    assert brief != "(no knowledge for these causes)"
    assert ids[0] in brief


def test_the_retriever_says_so_when_it_has_nothing() -> None:
    empty = CorpusRetriever(HybridIndex([]))
    assert empty.match(["anything"]) == []
    assert empty.brief_for(["nothing"]) == "(no knowledge for these causes)"


def test_the_loop_runs_against_the_corpus_retriever(ctx: object, clock: object) -> None:
    from tests.test_agent_loop import a_call, a_plan, a_settled_answer, a_stop

    client = ScriptedClient(
        replies={
            "planner": [
                a_plan(
                    ["compute_temp_corrected_pr"],
                    [("H1", "a string has failed"), ("H2", "dust on the modules")],
                )
            ],
            "router": [a_call("compute_temp_corrected_pr"), a_stop()],
            "synthesizer": [a_settled_answer()],
        }
    )
    out = investigate(
        "why is output down?",
        ctx,  # type: ignore[arg-type]
        client,
        clock,  # type: ignore[arg-type]
        investigation_id="INV-RAG",
        knowledge=build_retriever(),
        critic=False,
    )
    lookup = next(s for s in out.steps if s.kind == "retrieval")
    assert lookup.args is not None and lookup.args["matched_causes"]
    router_prompt = next(u for node, _, u in client.calls if node == "router")
    assert "WHAT IS KNOWN ABOUT THESE CAUSES" in router_prompt
