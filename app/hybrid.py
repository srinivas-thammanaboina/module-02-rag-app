"""
Advanced stage: hybrid retrieval (dense + sparse BM25, fused with RRF).

The base Retriever is DENSE: it embeds the question and compares vectors. It
wins on meaning/paraphrase but whiffs on opaque tokens the embedder never
learned a good vector for — acronyms (FDDEI, GDPR), named acts (GAIN AI Act),
foreign entities (TSMC). The golden-set v2 `lexical` category exists to expose
exactly that gap (recall@5=0.30 baseline; see notes/advanced/eval-notes.md).

The fix is a SECOND retriever that scores on literal tokens — BM25 — run in
parallel, then FUSE the two ranked lists. BM25 gives a rare token like "FDDEI"
a huge weight precisely because it's rare, so the answer chunk it returns at
rank 1 survives fusion into the top-k even though dense never saw it.

We hand-roll BM25 (≈40 lines) rather than pulling in `rank_bm25`: the whole
point of this project is understanding the mechanics from first principles, and
the TF/IDF/length-normalization math is the lesson. Theory companion:
ai-engineering-notes/02-rag/hybrid-retrieval.md.

Composition, per the advanced-stage convention: `HybridRetriever` WRAPS the base
`Retriever` and exposes the same `.retrieve(question, k, company)` interface, so
`eval` accepts it with no changes and the naive v1 retriever stays untouched.
"""

from __future__ import annotations

import json
import math
import re
from itertools import zip_longest

from app.config import config

# --- BM25 hyperparameters (Okapi BM25 defaults) ----------------------------
# K1 controls TF saturation: how fast extra occurrences of a term stop helping.
# B controls length normalization: 0 = ignore chunk length, 1 = fully normalize.
# These are the textbook defaults and we don't tune them — at 678 chunks there's
# nothing to overfit to, and the lesson is the algorithm, not the knobs.
BM25_K1 = 1.5
BM25_B = 0.75

# RRF: a document highly ranked in EITHER lane gets a good fused score; the
# constant softens how much the very top ranks dominate. 60 is the canonical
# value from the original RRF paper (and what Elastic/Qdrant/Weaviate ship).
DEFAULT_RRF_K = 60

# Per-lane candidate depth. We pull this many from EACH of dense and BM25 before
# fusing. Wide enough that a relevant chunk strong in one lane is in the pool;
# the depth sweep (eval-notes Finding 4) showed recall@50=1.00 here.
DEFAULT_POOL = 50

# How to combine the two ranked lists.
#   "rrf"        — Reciprocal Rank Fusion: consensus scoring. Protects categories
#                  dense already wins, but its one-lane cap can't rescue a chunk
#                  dense is BLIND to (the opaque-token case). See note §3c.
#   "interleave" — round-robin: guaranteed slots per lane, so a one-lane answer
#                  survives — at the risk of injecting BM25 noise into semantic
#                  questions. The tradeoff is empirical; the eval settles it.
FUSION_MODES = ("rrf", "interleave")
DEFAULT_FUSION = "rrf"


# English function-word stopwords for cleaning the BM25 *query* (never the
# documents, never the dense lane). These carry no lexical signal but — because
# 10-K prose almost never uses conversational scaffolding like "what does …
# disclose" — they're actually RARE in this corpus, so BM25's IDF wrongly treats
# them as high-value tokens and pulls generic chunks up the sparse lane (which
# RRF then rewards for spurious cross-lane agreement). The list is a property of
# the *language*, not of our corpus or eval set: it's a standard function-word
# set, chosen on linguistic principle and identical in dev and prod. We never
# add/remove words by inspecting which golden questions improve — that would
# leak the held-out eval into the system. Statistical (corpus-common) stopwords
# are already handled automatically by BM25's IDF; this list handles the
# orthogonal problem of query phrasing. See ai-engineering-notes/02-rag/
# hybrid-retrieval.md and eval-notes.md "Golden set v2".
STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does doing for from
had has have having how in into is it its of on or our the their this that
these those to was were what when where which who whom whose why will with
would you your about above after again against am any because before below
between both during each few further here if more most no nor not only other
over own same so some such than then there through under until up very
""".split())


def _tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens.

    `[a-z0-9]+` keeps digits (so "Section 232" → ["section", "232"]) and snaps
    acronyms out of surrounding punctuation ("(FDDEI)" → ["fddei"]) — the exact
    tokens BM25's exact-match advantage rides on. No stemming: stemming would
    fold "Act"/"acts" together but also risks mangling the rare named tokens we
    most care about, and at this corpus size we don't need the recall boost.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_query(question: str) -> str:
    """Strip function-word stopwords from a question for the BM25 lane.

    Returns the content tokens space-joined. The dense lane keeps the FULL
    natural-language question (embeddings use the context); only the sparse,
    bag-of-words lane wants keywords. Falls back to the original tokens if the
    question is all stopwords (so we never hand BM25 an empty query).
    """
    content = [t for t in _tokenize(question) if t not in STOPWORDS]
    return " ".join(content) if content else " ".join(_tokenize(question))


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------


class BM25Index:
    """An in-memory Okapi BM25 index over the chunk corpus.

    Built once from data/chunks/*.jsonl (the same source `build` embeds). Stores
    an inverted index (term → postings) so a query only touches chunks that
    actually contain its terms, plus per-chunk length and metadata for scoring
    and for reconstructing result rows.
    """

    def __init__(self, chunks: list[dict]):
        # Parallel arrays indexed by an internal integer doc index.
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._metas: list[dict] = []
        self._tickers: list[str] = []
        self._doc_len: list[int] = []

        # Inverted index: term -> list of (doc_index, term_frequency).
        self._postings: dict[str, list[tuple[int, int]]] = {}
        # Document frequency: term -> number of chunks containing it.
        self._df: dict[str, int] = {}

        for c in chunks:
            idx = len(self._ids)
            meta = c.get("metadata", {}) or {}
            text = c.get("text", "") or ""
            tokens = _tokenize(text)

            self._ids.append(c["chunk_id"])
            self._docs.append(text)
            self._metas.append(meta)
            self._tickers.append(meta.get("ticker", ""))
            self._doc_len.append(len(tokens))

            # Term frequencies within this chunk.
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for t, f in tf.items():
                self._postings.setdefault(t, []).append((idx, f))
                self._df[t] = self._df.get(t, 0) + 1

        self._n = len(self._ids)
        self._avgdl = (sum(self._doc_len) / self._n) if self._n else 0.0

        # Precompute IDF per term. The +1 smoothing (ln(1 + …)) keeps IDF
        # strictly positive even for terms in most documents — the plain Okapi
        # IDF can go negative for very common words, which distorts fusion.
        self._idf: dict[str, float] = {
            t: math.log(1 + (self._n - df + 0.5) / (df + 0.5))
            for t, df in self._df.items()
        }

    def search(self, query: str, k: int, ticker: str | None = None) -> list[tuple[str, float]]:
        """Return the top-k (chunk_id, bm25_score), highest score first.

        `ticker` applies the same company filter the dense lane gets via Chroma's
        `where` — BM25 has no native metadata filter, so we restrict scoring to
        chunks whose metadata ticker matches. Both lanes MUST honor the filter
        identically or the comparison is dishonest.
        """
        q_terms = _tokenize(query)
        scores: dict[int, float] = {}

        for t in q_terms:
            postings = self._postings.get(t)
            if not postings:
                continue
            idf = self._idf[t]
            for doc_idx, f in postings:
                if ticker and self._tickers[doc_idx] != ticker:
                    continue
                # Okapi BM25 term contribution: IDF × saturated, length-normalized TF.
                dl = self._doc_len[doc_idx]
                denom = f + BM25_K1 * (1 - BM25_B + BM25_B * dl / self._avgdl)
                contribution = idf * (f * (BM25_K1 + 1)) / denom
                scores[doc_idx] = scores.get(doc_idx, 0.0) + contribution

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(self._ids[i], s) for i, s in ranked]

    def row(self, chunk_id: str) -> dict | None:
        """Reconstruct a result dict (document + metadata) for a chunk id.

        Used for chunks BM25 surfaced that the dense lane never saw — we still
        need their text/metadata to hand the generator (and to display).
        """
        try:
            i = self._ids.index(chunk_id)
        except ValueError:
            return None
        return {"id": chunk_id, "document": self._docs[i], "metadata": self._metas[i]}


# Module-level cache: build the index once per process, not per query.
_BM25_INDEX: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    """Load (and cache) the BM25 index from data/chunks/*.jsonl."""
    global _BM25_INDEX
    if _BM25_INDEX is None:
        chunks: list[dict] = []
        for path in sorted((config.root / "data" / "chunks").glob("*.jsonl")):
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
        _BM25_INDEX = BM25Index(chunks)
    return _BM25_INDEX


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    dense_ids: list[str],
    sparse_ids: list[str],
    rrf_k: int = DEFAULT_RRF_K,
) -> dict[str, float]:
    """Reciprocal Rank Fusion: combine two rankings by POSITION, not score.

    Each list contributes 1/(rrf_k + rank) to every id it ranks (rank 0-indexed).
    Rank-only fusion sidesteps the cosine-vs-BM25 scale mismatch entirely — there
    is nothing to normalize. An id strong in either lane scores well; an id in
    both gets the sum (cross-lane agreement is rewarded). Returns id -> score.
    """
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def interleave_ids(dense_ids: list[str], sparse_ids: list[str]) -> list[str]:
    """Round-robin merge: dense#1, sparse#1, dense#2, sparse#2, … (dedup).

    The guaranteed-slot alternative to RRF (the round_robin_merge pattern from
    decompose.py, at the id level). Each lane gets slots in turn regardless of
    the other lane's opinion, so a chunk only ONE lane found — e.g. the opaque
    token dense is blind to — still survives, which RRF's consensus math caps
    out of the top-k (notes §3c). Dense leads (it's the primary retriever), so a
    one-lane BM25 answer lands at position 2.
    """
    out: list[str] = []
    seen: set[str] = set()
    for d_id, s_id in zip_longest(dense_ids, sparse_ids):
        for cid in (d_id, s_id):
            if cid is not None and cid not in seen:
                seen.add(cid)
                out.append(cid)
    return out


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Dense + BM25, fused with RRF. Same `.retrieve()` contract as Retriever.

    Each returned dict keeps the base shape (id, document, similarity, metadata)
    and adds provenance for inspection: `rrf_score`, `dense_rank`, `bm25_rank`,
    `bm25_score` (rank fields are 1-indexed, or None if that lane didn't surface
    the chunk). `similarity` carries the real dense cosine when the chunk was in
    the dense pool, else 0.0 — a chunk BM25 found that dense ranked outside its
    pool has no cosine signal, and 0.0 says exactly that.
    """

    def __init__(self, base, pool: int = DEFAULT_POOL, rrf_k: int = DEFAULT_RRF_K,
                 fusion: str = DEFAULT_FUSION, bm25_index: BM25Index | None = None):
        if fusion not in FUSION_MODES:
            raise ValueError(f"fusion must be one of {FUSION_MODES}, got {fusion!r}")
        self._base = base
        self._pool = pool
        self._rrf_k = rrf_k
        self._fusion = fusion
        self._bm25 = bm25_index or get_bm25_index()

    def retrieve(self, question: str, k: int = 5, company: str | None = None) -> list[dict]:
        # Lane 1: dense recall (wide pool). Carries cosine + document + metadata.
        dense = self._base.retrieve(question, k=self._pool, company=company)
        dense_ids = [r["id"] for r in dense]
        dense_by_id = {r["id"]: r for r in dense}
        dense_rank = {rid: i + 1 for i, rid in enumerate(dense_ids)}

        # Lane 2: sparse BM25 recall, same company filter (ticker). BM25 gets the
        # stopword-stripped query — function words are noise to a bag-of-words
        # matcher and (being rare in 10-K prose) actively mislead it here.
        sparse = self._bm25.search(bm25_query(question), k=self._pool, ticker=company)
        sparse_ids = [cid for cid, _ in sparse]
        bm25_rank = {cid: i + 1 for i, cid in enumerate(sparse_ids)}
        bm25_score = {cid: s for cid, s in sparse}

        # RRF scores are always computed (cheap) — they order the list in "rrf"
        # mode and ride along as provenance in both modes.
        rrf_scores = reciprocal_rank_fusion(dense_ids, sparse_ids, rrf_k=self._rrf_k)
        if self._fusion == "rrf":
            ordered_ids = [cid for cid, _ in sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)]
        else:  # interleave — guaranteed slots per lane (notes §3c)
            ordered_ids = interleave_ids(dense_ids, sparse_ids)
        ordered_ids = ordered_ids[:k]

        results: list[dict] = []
        for cid in ordered_ids:
            base_row = dense_by_id.get(cid)
            if base_row is not None:
                row = dict(base_row)  # real cosine in `similarity`
            else:
                # BM25-only: dense never ranked it in the pool, so no cosine.
                reconstructed = self._bm25.row(cid) or {"id": cid, "document": "", "metadata": {}}
                row = {**reconstructed, "similarity": 0.0}
            row["fusion"] = self._fusion
            row["rrf_score"] = rrf_scores.get(cid)
            row["dense_rank"] = dense_rank.get(cid)
            row["bm25_rank"] = bm25_rank.get(cid)
            row["bm25_score"] = bm25_score.get(cid)
            results.append(row)
        return results
