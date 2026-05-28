"""
Stage 3: Embedding.

Turns text into 384-dimensional vectors so chunks can be compared by
geometric distance instead of keyword overlap.

Two design choices that matter:

  1. **Embedder is an interface, not a class.** Today we ship a local
     sentence-transformers implementation. Tomorrow we might swap in an
     OpenAI/Voyage/etc. API behind the same interface. Stages 5 and 6
     never touch the concrete embedder — they call into the interface.

  2. **The BGE query/document asymmetry is enforced inside the embedder.**
     BGE-family models were trained with the prefix
     "Represent this sentence for searching relevant passages: " on
     queries only. Documents have no prefix. The caller doesn't need to
     remember this — embed_query() prepends; embed_documents() doesn't.

All vectors returned from this module are L2-normalized, so cosine
similarity reduces to a dot product. That keeps downstream code simple.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.config import config


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class Embedder(ABC):
    """The contract every embedder must satisfy.

    `embed_query` and `embed_documents` are deliberately separate methods —
    not just because BGE needs different prefixes, but because the SHAPE of
    the inputs differs (one query vs many documents) and batching matters
    for throughput on the documents side.
    """

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single user query. Returns a 1-D ndarray of shape (dim,)."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents (e.g. chunks). Returns 2-D (n, dim)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension. Used to size the vector store."""


# ---------------------------------------------------------------------------
# Local implementation — sentence-transformers on CPU
# ---------------------------------------------------------------------------


class LocalSentenceTransformerEmbedder(Embedder):
    """Wraps a sentence-transformers model for local on-CPU inference.

    The model loads once at construction (~130 MB download on first use to
    `~/.cache/huggingface/hub/`) and is held in memory for the process
    lifetime. Subsequent calls just run forward passes.

    Encoded vectors are L2-normalized by sentence-transformers so cosine
    similarity = dot product. Chroma expects normalized vectors too when
    we configure it with cosine distance, so this lines up.
    """

    # Recommended on the model card: this exact string was used at training
    # time to mark inputs as "queries". Documents got no prefix.
    # Reference: https://huggingface.co/BAAI/bge-small-en-v1.5
    _QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str):
        # Lazy import keeps `python cli.py --help` snappy and avoids importing
        # heavy ML libraries in stages that don't need them (e.g. ingest, chunk).
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        self._model_name = model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_query(self, text: str) -> np.ndarray:
        # IMPORTANT: prefix is required for BGE-family queries. Skipping it
        # silently degrades retrieval accuracy by 5-10% (no error raised).
        prefixed = self._QUERY_PREFIX + text
        vec = self._model.encode(prefixed, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        # batch_size=32 is fine on CPU for ~1KB chunks. Larger batches don't
        # speed things up much once the BLAS lib saturates a few cores.
        mat = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return np.asarray(mat, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory — one line to swap implementations
# ---------------------------------------------------------------------------


def get_embedder() -> Embedder:
    """Single seam for swapping embedder implementations.

    Today this returns the local model. To A/B test against an API model,
    add a new class implementing Embedder and dispatch here on a config
    field. Nothing else in the codebase changes.
    """
    return LocalSentenceTransformerEmbedder(config.embedding_model)


# ---------------------------------------------------------------------------
# CLI entry point — `python cli.py embed --text "..."` (one or more)
# ---------------------------------------------------------------------------


def run_cli(args) -> None:
    """Demonstrate the embedder.

    With one --text: print vector stats only (dim, norm, first few values).
    With two or more: treat the FIRST as a query, the rest as documents,
    and print cosine similarity (= dot product since vectors are normalized)
    for each. This is the sanity-check experiment from chunking-notes.md.
    """
    embedder = get_embedder()
    texts: list[str] = args.text

    print()
    print(f"  model          : {embedder.model_name}")
    print(f"  dimension      : {embedder.dim}")
    print()

    if len(texts) == 1:
        vec = embedder.embed_query(texts[0])
        print(f"  text           : {texts[0]!r}")
        print(f"  vector[:8]     : {[round(float(v), 4) for v in vec[:8]]}")
        print(f"  L2 norm        : {float(np.linalg.norm(vec)):.6f}  (should be ~1.0)")
        return

    # Multi-text: query vs documents.
    query = texts[0]
    docs = texts[1:]
    q_vec = embedder.embed_query(query)
    d_vecs = embedder.embed_documents(docs)

    # Cosine sim = dot product because we're L2-normalized.
    sims = d_vecs @ q_vec  # shape (n_docs,)

    print(f"  query          : {query!r}")
    print()
    print(f"  similarity scores (higher = closer in meaning):")
    print()
    # Print sorted by sim descending — easier to eyeball ranking.
    order = np.argsort(-sims)
    for rank, i in enumerate(order, start=1):
        bar = "#" * int(round(float(sims[i]) * 40))
        print(f"    {rank}. sim={sims[i]:.4f}  {bar}")
        print(f"       {docs[i]!r}")
        print()
