"""Building the corpus, and where its text is allowed to come from.

Two hard constraints shape this file.

**No third-party PDFs are committed** (CLAUDE.md). The IEC and NREL documents
that would make the best corpus are copyrighted, so the repository ships an
*ingest* for them and a manifest of SHA-256 checksums, never the documents. A
clone without those files still works — retrieval falls back to the knowledge
YAML — and the ablation reports which corpus it measured.

**The corpus must not be the answer key.** Chunks describe physics and
signatures; none of them names a golden case, a threshold, or an injector
parameter. If the corpus contained "case G-001 is a string outage", retrieval
would score perfectly and would be measuring nothing.

Chunk ids are stable across rebuilds — `source:section:n` — because the
retrieval golden set refers to them by id. An id that shifts when a paragraph
is added silently invalidates every label in that set.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.config import REPO_ROOT
from src.knowledge import load_knowledge
from src.rag.index import Chunk

__all__ = [
    "CORPUS_DIR",
    "build_corpus",
    "chunk_document",
    "knowledge_chunks",
    "write_manifest",
]

CORPUS_DIR = REPO_ROOT / "corpus"
MANIFEST = CORPUS_DIR / "manifest.json"

# Long enough that a chunk answers a question on its own, short enough that a
# hit is specific. Retrieval over whole documents returns "the document is
# relevant", which the agent cannot act on.
TARGET_WORDS = 160
OVERLAP_WORDS = 30


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


def chunk_document(
    text: str,
    source: str,
    title: str = "",
    target_words: int = TARGET_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[Chunk]:
    """Split a document into overlapping passages on paragraph boundaries.

    Overlap matters more than it looks. A signature and the observation that
    separates it from its look-alike often sit either side of a paragraph break,
    and a hard split puts the answer in one chunk and the question in another.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    buffer: list[str] = []
    count = 0

    def flush() -> None:
        nonlocal buffer, count
        if not buffer:
            return
        body = "\n\n".join(buffer)
        chunks.append(
            Chunk(
                id=f"{_slug(source)}:{len(chunks):03d}",
                text=body,
                source=source,
                title=title,
            )
        )
        # Carry the tail forward so a boundary never cuts an argument in half.
        tail = body.split()[-overlap_words:] if overlap_words else []
        buffer = [" ".join(tail)] if tail else []
        count = len(tail)

    for paragraph in paragraphs:
        words = len(paragraph.split())
        if count and count + words > target_words:
            flush()
        buffer.append(paragraph)
        count += words
    flush()
    return chunks


def knowledge_chunks() -> list[Chunk]:
    """The always-available corpus: the project's own domain knowledge.

    Chunked one signature or one distinguishing test at a time, because that is
    the unit a query actually wants. A chunk holding three signatures is a hit
    that makes the agent read two irrelevant ones.
    """
    knowledge = load_knowledge()
    chunks: list[Chunk] = []

    for key, signature in knowledge.signatures.items():
        chunks.append(
            Chunk(
                id=f"signature:{key}",
                text=signature.as_text(),
                source="project knowledge base",
                title=f"{key} — {signature.plain_name}",
                causes=(key,),
            )
        )
    for test in knowledge.tests:
        first, second = test.between
        chunks.append(
            Chunk(
                id=f"distinguish:{first}__{second}",
                text=test.as_text(),
                source="project knowledge base",
                title=f"telling {first} from {second}",
                causes=(first, second),
            )
        )
    return chunks


@dataclass(frozen=True)
class CorpusFile:
    path: Path
    sha256: str
    bytes: int
    chunks: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def build_corpus(
    corpus_dir: Path | None = None, include_knowledge: bool = True
) -> tuple[list[Chunk], list[CorpusFile]]:
    """Assemble every chunk, from the knowledge base and any ingested documents.

    Returns the chunks and a record of the files they came from, so a retrieval
    number can be quoted alongside the corpus that produced it.
    """
    root = corpus_dir or CORPUS_DIR
    chunks: list[Chunk] = knowledge_chunks() if include_knowledge else []
    files: list[CorpusFile] = []

    if root.exists():
        for path in sorted(root.rglob("*.txt")) + sorted(root.rglob("*.md")):
            if path.name == "README.md":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            document = chunk_document(text, source=path.stem, title=path.stem)
            chunks.extend(document)
            files.append(
                CorpusFile(path, _sha256(path), path.stat().st_size, len(document))
            )
    return chunks, files


def write_manifest(files: list[CorpusFile], path: Path | None = None) -> Path:
    """Commit checksums, never the documents themselves.

    This is what makes the corpus reproducible without shipping copyrighted
    text: anyone can re-download the sources and verify byte-for-byte that they
    have the same corpus the published retrieval numbers were measured on.
    """
    target = path or MANIFEST
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "name": f.path.name,
                        "sha256": f.sha256,
                        "bytes": f.bytes,
                        "chunks": f.chunks,
                    }
                    for f in sorted(files, key=lambda f: f.path.name)
                ],
                "note": (
                    "Checksums only. The documents are third-party and are not "
                    "committed; re-download them to reproduce the corpus."
                ),
            },
            indent=2,
        )
    )
    return target
