"""
Stage 5: Retrieval — top-k chunks for a question.

Mechanically: embed the question, query the vector store, return top-k.
The store handles cosine similarity and metadata filtering; this module
orchestrates and formats. The Retriever class is intentionally thin —
most of the work is already in app/store.py and app/embed.py.

Two pressure-test mitigations live here (full discussion in
notes/retrieval-notes.md):

  1. **Company-mismatch warning.** If the question text mentions a ticker
     that isn't the one passed via --company, the CLI warns the user. The
     filter still wins — that's the contract — but the contradiction is
     made visible instead of silent.

  2. **Top-1 similarity surfaced prominently.** Per notes/embedding-notes.md,
     BGE-small's noise floor on prose is ~0.50–0.55. The CLI labels the
     confidence band of the top result so the user can tell when the
     corpus probably doesn't contain a good answer. Stage 6's prompt
     will act on this signal; Stage 5's job is to make it visible.

Pure top-k for now. MMR / cross-encoder rerank / HyDE / hybrid retrieval
are queued as future experiments in notes/retrieval-notes.md.
"""

from __future__ import annotations

import re
import time

from app.store import VectorStore, get_vector_store


# Company-name detection. Conservative — three known tickers and their
# lowercase company names. A production system would use a NER model or
# a one-shot LLM router; at three tickers neither is justified.
_COMPANY_PATTERNS: dict[str, tuple[str, ...]] = {
    "TSLA": ("tesla",),
    "AAPL": ("apple",),
    "NVDA": ("nvidia",),
}


# Confidence bands for BGE-small on English prose. Source:
# notes/embedding-notes.md "Score interpretation bands". CLI-only labelling —
# the retriever never gates on these; that's Stage 6's job.
def _confidence_label(top_sim: float) -> str:
    if top_sim >= 0.75:
        return "very high (direct paraphrase)"
    if top_sim >= 0.65:
        return "high (clearly relevant)"
    if top_sim >= 0.58:
        return "moderate (likely relevant)"
    if top_sim >= 0.52:
        return "low (near BGE noise floor — corpus may not contain a good match)"
    return "very low (likely no good match)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def detect_companies_in_question(question: str) -> set[str]:
    """Return the set of tickers mentioned in the question.

    Matches either the ticker symbol (word-bounded, case-insensitive) or
    a known company-name pattern. Conservative on purpose — a false
    positive triggers a spurious warning (annoying), but a false negative
    re-introduces the silent-failure mode the warning exists to prevent.
    """
    q_upper = question.upper()
    q_lower = question.lower()
    found: set[str] = set()
    for ticker, name_patterns in _COMPANY_PATTERNS.items():
        if re.search(rf"\b{re.escape(ticker)}\b", q_upper):
            found.add(ticker)
            continue
        for pat in name_patterns:
            if re.search(rf"\b{re.escape(pat)}\b", q_lower):
                found.add(ticker)
                break
    return found


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """Thin wrapper around VectorStore.query.

    Translates --company into a `where` filter and forwards the rest.
    Deliberately does NOT inspect question text or apply confidence
    thresholds — those are UX / prompt-side concerns and would couple
    this class to layers that should stay independent.
    """

    def __init__(self, store: VectorStore):
        self._store = store

    def retrieve(
        self,
        question: str,
        k: int = 5,
        company: str | None = None,
    ) -> list[dict]:
        where = {"ticker": company.upper()} if company else None
        return self._store.query(question, k=k, where=where)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_cli(args) -> None:
    """`python cli.py retrieve --question "..." [--company TSLA] [--k N] [--compare]`."""
    question: str = args.question
    k: int = args.k
    company: str | None = args.company.upper() if args.company else None

    # Pressure-test mitigation 1: warn on detected-vs-filter mismatch.
    mentioned = detect_companies_in_question(question)
    if company and mentioned and company not in mentioned:
        print()
        print(f"  WARNING: question mentions {sorted(mentioned)}, but --company={company} was set.")
        print(f"  The filter will win — retrieval will return {company} content.")
        print(f"  If you want different content, change or drop --company.")

    store = get_vector_store()
    retriever = Retriever(store)

    print()
    print(f"  question     : {question!r}")
    print(f"  top-k        : {k}")

    if args.compare:
        if not company:
            print()
            print("  --compare requires --company (it specifies the filtered arm).")
            return
        _show_results(
            label=f"WITH FILTER (--company {company})",
            results=_run_one(retriever, question, k, company),
        )
        print()
        _show_results(
            label="WITHOUT FILTER",
            results=_run_one(retriever, question, k, None),
            include_ticker_breakdown=True,
        )
    else:
        label = f"FILTERED on {company}" if company else "UNFILTERED"
        _show_results(label=label, results=_run_one(retriever, question, k, company))


def _run_one(
    retriever: Retriever,
    question: str,
    k: int,
    company: str | None,
) -> list[dict]:
    t0 = time.time()
    results = retriever.retrieve(question, k=k, company=company)
    elapsed_ms = (time.time() - t0) * 1000
    print(f"  elapsed      : {elapsed_ms:.0f} ms")
    return results


def _show_results(
    label: str,
    results: list[dict],
    include_ticker_breakdown: bool = False,
) -> None:
    print()
    print(f"  === {label} ===")
    if not results:
        print("    (no results)")
        return

    # Pressure-test mitigation 2: top-1 similarity with confidence band.
    top_sim = results[0]["similarity"]
    print(f"  top-1 sim    : {top_sim:.4f}  ({_confidence_label(top_sim)})")

    if include_ticker_breakdown:
        counts: dict[str, int] = {}
        for r in results:
            tk = r["metadata"].get("ticker", "?")
            counts[tk] = counts.get(tk, 0) + 1
        breakdown = ", ".join(f"{tk}×{n}" for tk, n in sorted(counts.items()))
        print(f"  tickers      : {breakdown}")

    print()
    for rank, r in enumerate(results, start=1):
        meta = r["metadata"]
        section = meta.get("section", "?")
        ticker = meta.get("ticker", "?")
        chunk_id = r["id"]
        doc = r["document"] or ""
        # Word-boundary snap so the preview doesn't start mid-word.
        head = doc[:200]
        head_clean = head[: head.rfind(" ")] if " " in head[160:] else head
        head_clean = head_clean.replace("\n", " | ")
        print(f"    rank {rank}  sim={r['similarity']:.4f}  {ticker} | {section}")
        print(f"      id  : {chunk_id}")
        print(f"      head: {head_clean!r}")
        print()
