"""
Advanced stage: MMR (Maximal Marginal Relevance) selection.

Plain top-k runs ONE similarity competition and takes the 5 most similar chunks
— so a multi-aspect question ("revenue beyond vehicle sales") fills the window
with near-duplicates of the single dominant aspect and starves the others (the
enumeration failure, eval-notes Finding 3). MMR fixes the SELECTION, not the
retrieval: it builds the result greedily, each pick trading query-relevance
against redundancy versus what's already chosen, so once one used-vehicle chunk
is in, its siblings are penalized and a different aspect can win the next slot.

    first pick = argmax sim(q, c)
    next pick  = argmax  λ·sim(q, c)  −  (1−λ)·max sim(c, s) for s in selected
                         └ relevance ┘     └ redundancy penalty ┘

KEY LIMIT (the reason this can't fix every enumeration): MMR only REORDERS the
candidate pool — it weights relevance, so a chunk dense buried because it has
LOW query-similarity (Q24's per-market subsections, which don't look like "end
markets") won't be promoted no matter how wide the pool. MMR can't manufacture
relevance; that's retrieve-then-expand's job. See notes/advanced (enumeration).

Composition: `MMRRetriever` wraps the base `Retriever` and honors the same
`.retrieve(question, k, company)` contract. It needs the candidate VECTORS
(chunk-to-chunk cosine), so it asks the base retriever for `include_embeddings`
— the one capability plain dense retrieval didn't expose before.
"""

from __future__ import annotations

import numpy as np

# λ: relevance-vs-diversity dial. 0.7 leans on relevance (a 10-K answer must
# stay on-topic; we want modest diversification, not a grab-bag). Set on
# PRINCIPLE, not tuned against the golden set — that would leak the held-out
# eval into the system. Exposed via --mmr-lambda for transparent exploration only.
DEFAULT_LAMBDA = 0.7

# Candidate pool MMR diversifies within. It only reorders the pool, so a wider
# pool extends reach — but see the KEY LIMIT above: relevance, not pool size,
# is usually the binding constraint. recall@50 = 1.00 on v1 motivates 50.
DEFAULT_POOL = 50


class MMRRetriever:
    """Diversity-aware re-selection of a dense candidate pool."""

    def __init__(self, base, pool: int = DEFAULT_POOL, lambda_: float = DEFAULT_LAMBDA):
        self._base = base
        self._pool = pool
        self._lambda = lambda_

    def retrieve(self, question: str, k: int = 5, company: str | None = None) -> list[dict]:
        candidates = self._base.retrieve(
            question, k=self._pool, company=company, include_embeddings=True
        )
        if len(candidates) <= k:
            # Nothing to diversify away — return what we have, labeled.
            return [self._clean(c, rank) for rank, c in enumerate(candidates, 1)]

        # Embeddings are L2-normalized (bge), so cosine = dot product.
        embs = np.asarray([c["embedding"] for c in candidates], dtype=float)
        sim_q = np.asarray([c["similarity"] for c in candidates], dtype=float)

        selected: list[int] = []
        remaining = list(range(len(candidates)))

        # First pick: pure relevance (== plain top-1).
        first = max(remaining, key=lambda i: sim_q[i])
        selected.append(first)
        remaining.remove(first)

        # Greedy MMR for the rest.
        while len(selected) < k and remaining:
            sel = embs[selected]                       # (s, d)
            rem = embs[remaining]                      # (r, d)
            # redundancy[j] = max cosine of remaining-j to any already-selected
            redundancy = (rem @ sel.T).max(axis=1)     # (r,)
            mmr = self._lambda * sim_q[remaining] - (1.0 - self._lambda) * redundancy
            best = remaining[int(np.argmax(mmr))]
            selected.append(best)
            remaining.remove(best)

        return [self._clean(candidates[i], rank) for rank, i in enumerate(selected, 1)]

    @staticmethod
    def _clean(c: dict, rank: int) -> dict:
        """Drop the bulky embedding from the returned dict; tag the MMR position."""
        row = {key: val for key, val in c.items() if key != "embedding"}
        row["mmr_rank"] = rank
        return row
