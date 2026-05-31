"""
Advanced stage: decomposition / round-robin retrieval (Experiment 7, Phase A).

A single top-k retrieval is ONE competition with one scoreboard: every chunk
competes on a single similarity score, so the most-similar sub-topic wins all
the slots and starves the others. For a "compare A and B" question that's a
structural coverage failure — the dominant company fills the window, the other
is missed (cross-company recall@5 = 0.67 on our golden set). No reranker fixes
this; a cross-encoder concentrates HARDER on the dominant side.

Phase A splits the one competition into several: detect the companies named in
the question, run one FILTERED retrieval per company, then round-robin merge so
each company is guaranteed slots — balanced by construction.

Composition, per the advanced-stage structure decision: `DecompositionRetriever`
WRAPS the base `Retriever` and honors the same `.retrieve(question, k, company)`
contract, so `eval`/`ask` accept it unchanged. Design + predictions + the honest
limitations are in notes/advanced/decomposition-notes.md.

Scope (Phase A): the CROSS-COMPANY case only. The aspect-enumeration case (Q7,
"revenue beyond vehicle sales") has no company keywords to detect, so it falls
through to plain retrieval — that's Phase B (LLM query decomposition).
"""

from __future__ import annotations

from app.retrieve import detect_companies_in_question


def round_robin_merge(per_company: dict[str, list[dict]], k: int) -> list[dict]:
    """Interleave per-company ranked lists by rank: A#1, B#1, A#2, B#2, ... to k.

    Balanced by construction — every company gets slots in turn, so a comparison
    question can't be monopolized by the more-similar company. Dedup by id is
    belt-and-suspenders (per-company filters are disjoint tickers, so lists don't
    actually overlap), kept for generality. Result dicts pass through untouched.
    """
    lists = [per_company[c] for c in sorted(per_company)]
    if not lists:
        return []
    merged: list[dict] = []
    seen: set[str] = set()
    for rank in range(max(len(lst) for lst in lists)):
        for lst in lists:
            if rank < len(lst):
                item = lst[rank]
                if item["id"] not in seen:
                    merged.append(item)
                    seen.add(item["id"])
                    if len(merged) == k:
                        return merged
    return merged


class DecompositionRetriever:
    """Cross-company decomposition wrapped around a base retriever.

    Dispatch (the safety property): decompose ONLY when the call is unfiltered
    AND the question names >=2 companies. Otherwise passthrough to the exact
    baseline path — so single-company and semantic questions are provably
    unchanged, and the only category that can move is cross-company. Honors the
    base `.retrieve()` contract, so it is a drop-in for eval/ask.
    """

    def __init__(self, base):
        self._base = base

    def retrieve(self, question: str, k: int = 5, company: str | None = None) -> list[dict]:
        # Caller already scoped to one company -> nothing to decompose.
        if company is not None:
            return self._base.retrieve(question, k=k, company=company)

        companies = detect_companies_in_question(question)
        # Single-topic (or no company named) -> plain retrieval, identical to baseline.
        if len(companies) < 2:
            return self._base.retrieve(question, k=k, company=None)

        # Cross-company: one filtered retrieval per company, then round-robin merge.
        # Fetch k per company so the merge has enough to fill k balanced slots.
        per_company = {
            ticker: self._base.retrieve(question, k=k, company=ticker)
            for ticker in sorted(companies)
        }
        return round_robin_merge(per_company, k)
