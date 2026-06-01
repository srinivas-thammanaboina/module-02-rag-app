"""
Stage 4: Vector storage with Chroma.

Persists (id, document, vector, metadata) rows for every chunk so retrieval
has a single addressable index to query. Wraps Chroma behind a VectorStore
ABC for the same reason we wrapped the embedder — to make the backend swap
a one-file change.

Design choices (full rationale in notes/store-chroma-notes.md):

  1. Single collection named "filings" with `ticker` as a metadata field.
     One collection scales to N tickers for free; per-ticker collections
     would require code changes to add a new company.

  2. `upsert` (not `add`) so re-runs are idempotent. Same chunk id
     overwrites; rebuilding from data/chunks/*.jsonl ten times produces the
     same state as building it once.

  3. Cosine distance configured at collection creation. Vectors are
     L2-normalized by the embedder, so cosine = dot product. We report
     `similarity = 1 - distance` everywhere so the language stays
     consistent across the codebase (higher = closer, always).

  4. Constructor injection of the Embedder. The store doesn't construct an
     embedder; it accepts one. This makes A/B testing trivial.

  5. Query/get results normalized to list[dict]. Chroma's native response
     is parallel arrays under nested keys; we hide that here so every
     caller sees the same simple shape.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import config
from app.embed import Embedder, get_embedder


# Collection name. Single collection for all tickers, with `ticker` as
# a metadata field. See "Chosen design" in notes/store-chroma-notes.md for why.
COLLECTION_NAME = "filings"

# Metadata fields propagated into Chroma. Listed explicitly so the contract
# is visible in code, not implicit from whatever the chunker happened to
# emit. Chroma requires scalar values only (str / int / float / bool).
_METADATA_FIELDS = (
    "ticker",
    "company_name",
    "section",
    "filing_date",
    "cik",
    "accession_number",
    "source_url",
    "chunk_index",
    "char_start",
    "char_end",
)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class VectorStore(ABC):
    """Contract every vector-store implementation must satisfy.

    Four methods cover Stages 4–6: upsert (write), query (top-k read with
    optional filter), count (sanity check), peek (inspection / debugging).
    """

    @abstractmethod
    def upsert_chunks(self, chunks: list[dict]) -> int:
        """Embed and upsert chunks. Returns rows written."""

    @abstractmethod
    def query(self, query_text: str, k: int, where: dict | None = None,
              include_embeddings: bool = False) -> list[dict]:
        """Top-k by similarity. Each row: {id, document, similarity, metadata}.

        `include_embeddings=True` adds the chunk's stored vector under `embedding`
        (needed by MMR, which scores chunk-to-chunk similarity — see app/mmr.py).
        """

    @abstractmethod
    def count(self, where: dict | None = None) -> int:
        """Row count, optionally restricted by a metadata filter."""

    @abstractmethod
    def peek(self, n: int = 1) -> list[dict]:
        """Return n rows in arbitrary order (for debugging / inspection)."""


# ---------------------------------------------------------------------------
# Chroma implementation
# ---------------------------------------------------------------------------


class ChromaVectorStore(VectorStore):
    """Chroma-backed vector store, persisted to disk under config.chroma_dir.

    The collection is created on first access with cosine distance configured
    (`hnsw:space=cosine`). The HNSW index is Chroma's default; we don't tune
    it because at 678 chunks brute force is already faster than HNSW anyway.
    """

    def __init__(
        self,
        embedder: Embedder,
        persist_dir: Path | None = None,
        collection_name: str = COLLECTION_NAME,
    ):
        # Lazy import: chromadb pulls in onnxruntime + sqlite extras. Importing
        # it at the top of the module would slow down every CLI subcommand.
        import chromadb

        path = persist_dir or config.chroma_dir
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        # `hnsw:space=cosine` configures the index for cosine distance.
        # Without this, Chroma defaults to squared L2, which is wrong for
        # L2-normalized embeddings (the rank order works out, but the
        # `distance` values are misleading).
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder
        self._collection_name = collection_name

    # --- writes ---------------------------------------------------------

    def upsert_chunks(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0
        ids = [c["chunk_id"] for c in chunks]
        docs = [c["text"] for c in chunks]
        metas = [self._chunk_metadata(c) for c in chunks]

        # Embed in one batch — the embedder handles internal batching by 32.
        vecs = self._embedder.embed_documents(docs)

        # Chroma expects lists of floats, not numpy arrays.
        self._collection.upsert(
            ids=ids,
            documents=docs,
            embeddings=vecs.tolist(),
            metadatas=metas,
        )
        return len(ids)

    # --- reads ----------------------------------------------------------

    def query(self, query_text: str, k: int, where: dict | None = None,
              include_embeddings: bool = False) -> list[dict]:
        q_vec = self._embedder.embed_query(query_text).tolist()
        # Chroma omits embeddings by default (they're large); request them only
        # when a caller (MMR) actually needs the vectors.
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include = include + ["embeddings"]
        raw = self._collection.query(
            query_embeddings=[q_vec],
            n_results=k,
            where=where,
            include=include,
        )
        return self._normalize_query_response(raw)

    def count(self, where: dict | None = None) -> int:
        if where is None:
            return self._collection.count()
        # collection.count() ignores `where`; fall back to get() with no
        # included fields (cheaper — we just want the matching ids).
        rows = self._collection.get(where=where, include=[])
        return len(rows["ids"])

    def peek(self, n: int = 1) -> list[dict]:
        raw = self._collection.peek(limit=n)
        return self._normalize_get_response(raw)

    # --- helpers --------------------------------------------------------

    @staticmethod
    def _chunk_metadata(chunk: dict) -> dict:
        """Pull only the documented metadata fields out of a chunk dict.

        Defensive: if a field is missing, skip it (Chroma will reject `None`).
        """
        meta = chunk.get("metadata", {}) or {}
        return {f: meta[f] for f in _METADATA_FIELDS if f in meta}

    @staticmethod
    def _normalize_query_response(raw: dict) -> list[dict]:
        """Convert Chroma's parallel-arrays query response to list[dict].

        Chroma returns
            {"ids": [[...]], "documents": [[...]],
             "distances": [[...]], "metadatas": [[...]]}
        for a single query (one outer list per query). We pull the first
        (only) query's results and pivot to a row-oriented shape.
        """
        ids = raw["ids"][0]
        docs = raw["documents"][0]
        dists = raw["distances"][0]
        metas = raw["metadatas"][0]
        # Embeddings present only when the caller asked for them. Use an identity
        # check (not truthiness) — these are numpy arrays.
        embs = raw.get("embeddings")
        embs = embs[0] if embs is not None else None
        rows = []
        for i in range(len(ids)):
            row = {
                "id": ids[i],
                "document": docs[i],
                # cosine distance → cosine similarity. With L2-normalized
                # vectors and hnsw:space=cosine, distance = 1 - cos_sim.
                "similarity": 1.0 - float(dists[i]),
                "metadata": metas[i] or {},
            }
            if embs is not None:
                row["embedding"] = embs[i]
            rows.append(row)
        return rows

    @staticmethod
    def _normalize_get_response(raw: dict) -> list[dict]:
        """Convert Chroma's get/peek response to list[dict] (no distances)."""
        ids = raw["ids"]
        docs = raw.get("documents") or [None] * len(ids)
        metas = raw.get("metadatas") or [None] * len(ids)
        return [
            {
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i] or {},
            }
            for i in range(len(ids))
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_vector_store(embedder: Embedder | None = None) -> VectorStore:
    """Return the configured vector store.

    Optionally accepts a pre-built embedder so tests / experiments can inject
    a fake. Defaults to the production embedder via `get_embedder()`.
    """
    if embedder is None:
        embedder = get_embedder()
    return ChromaVectorStore(embedder=embedder)


# ---------------------------------------------------------------------------
# CLI: build / store / inspect
# ---------------------------------------------------------------------------


def _read_chunks_jsonl(path: Path) -> list[dict]:
    """Load chunks from a JSONL file (one JSON object per line)."""
    chunks: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def _chunks_dir() -> Path:
    return config.root / "data" / "chunks"


def run_build_cli(args) -> None:
    """`python cli.py build` — full index build from all data/chunks/*.jsonl."""
    chunks_dir = _chunks_dir()
    files = sorted(chunks_dir.glob("*.jsonl"))
    if not files:
        print(f"[build] no chunk files in {chunks_dir}. Run `chunk` first.")
        return

    print()
    print(f"  chunks dir   : {chunks_dir}")
    print(f"  chroma path  : {config.chroma_dir}")
    print(f"  files        : {[f.name for f in files]}")
    print()

    store = get_vector_store()
    t0 = time.time()
    total = 0
    for path in files:
        chunks = _read_chunks_jsonl(path)
        n = store.upsert_chunks(chunks)
        total += n
        print(f"  {path.name:18} → upserted {n} chunks")
    elapsed = time.time() - t0

    print()
    print(f"  total upserted  : {total}")
    print(f"  collection size : {store.count()}")
    print(f"  elapsed         : {elapsed:.1f}s")
    print()


def run_store_cli(args) -> None:
    """`python cli.py store --ticker TSLA` — single-ticker rebuild for iteration."""
    ticker = args.ticker.upper()
    path = _chunks_dir() / f"{ticker}.jsonl"
    if not path.exists():
        print(f"[store] {path} not found. Run `chunk --ticker {ticker}` first.")
        return

    print()
    print(f"  ticker       : {ticker}")
    print(f"  chunks file  : {path}")
    print()

    chunks = _read_chunks_jsonl(path)
    store = get_vector_store()
    t0 = time.time()
    n = store.upsert_chunks(chunks)
    elapsed = time.time() - t0

    print(f"  upserted        : {n} chunks in {elapsed:.1f}s")
    print(f"  collection size : {store.count()}")
    print()


def run_inspect_cli(args) -> None:
    """`python cli.py inspect` — print collection stats and a sample row.

    Implements the three sanity-check properties from notes/store-chroma-notes.md:
      1. Total row count matches the chunker's output.
      2. Per-ticker filter counts match per-file chunk counts.
      3. A sample row is well-formed (embedding, document, metadata together).
    """
    store = get_vector_store()
    total = store.count()
    if total == 0:
        print("[inspect] collection is empty. Run `build` first.")
        return

    print()
    print(f"  collection size : {total}")
    print(f"  by ticker       :")
    for tk in config.tickers:
        n = store.count(where={"ticker": tk})
        print(f"      {tk:6} : {n} chunks")
    print()

    sample = store.peek(n=1)
    if not sample:
        return

    row = sample[0]
    doc = row.get("document") or ""
    head = doc[:140].replace("\n", " | ")
    print("  --- sample row ---")
    print(f"    id           : {row['id']}")
    print(f"    document len : {len(doc)} chars")
    print(f"    head         : {head!r}")
    print(f"    metadata     :")
    for k, v in (row.get("metadata") or {}).items():
        sv = str(v)
        if len(sv) > 80:
            sv = sv[:77] + "..."
        print(f"      {k:18} = {sv}")
    print()
