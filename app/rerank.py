"""
Advanced stage: reranking (retrieve wide → rerank narrow).

The base Retriever is a BI-ENCODER: it embeds the question and each chunk
separately (chunks were embedded back in Stage 3) and compares vectors. Fast,
but it never sees the question and a chunk *together*, so it ranks imperfectly.

A CROSS-ENCODER reads `(question, chunk)` as one joined input and outputs a
single relevance score — much better at judging fit, but too slow to run over
the whole corpus. The standard fix: use the cheap bi-encoder to fetch a wide
candidate pool, then the expensive cross-encoder to re-score and re-sort just
that pool. We pay the slow model on ~50 chunks, not 678.

Composition, per the advanced-stage structure decision: `RerankingRetriever`
WRAPS the base `Retriever` and exposes the same `.retrieve(question, k, company)`
interface — so `eval` and `ask` accept it with zero changes. The naive v1
retriever is untouched. Design + the depth-sweep that justified pool=50 are in
notes/advanced/eval-notes.md (Finding 4).
"""

from __future__ import annotations

# Cross-encoder choices. `minilm` = small/fast MS MARCO web-QA model (default);
# `bge` = larger, longer-passage model that pairs with our BGE embedder.
RERANKER_MODELS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge": "BAAI/bge-reranker-base",
}
DEFAULT_RERANKER_MODEL = RERANKER_MODELS["minilm"]

# How many bi-encoder candidates to re-score. The depth sweep showed recall@50
# = 1.00 on this corpus, so a 50-pool gives the cross-encoder every relevant
# chunk to work with (Finding 4).
DEFAULT_CANDIDATE_POOL = 50


class Reranker:
    """Wraps a cross-encoder model that scores (question, document) pairs."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self._model_name = model_name
        self._model = None  # lazy — the model pulls in torch (see embed.py)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def score(self, question: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document (higher = more relevant)."""
        if not documents:
            return []
        pairs = [(question, doc) for doc in documents]
        return [float(s) for s in self._get_model().predict(pairs)]


class RerankingRetriever:
    """Retrieve a wide pool with the base retriever, then cross-encoder re-rank.

    Honors the same `.retrieve()` contract as the base `Retriever`, so it is a
    drop-in anywhere a retriever is expected (eval, ask). Returned dicts keep the
    original cosine as `similarity` and add `rerank_score`; the list is ordered
    by `rerank_score`.
    """

    def __init__(self, base, reranker: Reranker | None = None,
                 candidate_pool: int = DEFAULT_CANDIDATE_POOL):
        self._base = base
        self._reranker = reranker or Reranker()
        self._pool = candidate_pool

    def retrieve(self, question: str, k: int = 5, company: str | None = None) -> list[dict]:
        # Stage 1: wide, cheap bi-encoder recall.
        candidates = self._base.retrieve(question, k=self._pool, company=company)
        if not candidates:
            return []

        # Stage 2: expensive cross-encoder re-scores every candidate.
        scores = self._reranker.score(question, [c["document"] for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)

        # Stage 3: narrow to top-k, carrying the new score alongside the cosine.
        return [{**c, "rerank_score": s} for c, s in ranked[:k]]
