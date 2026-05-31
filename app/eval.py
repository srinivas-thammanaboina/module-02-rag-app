"""
Advanced stage: retrieval evaluation harness.

Turns "retrieval feels better" into numbers. For each question in the golden
set (eval/golden.jsonl) we know the chunk id(s) that actually answer it; we run
the question through a retriever and score how well the returned ranking
recovers them. Full design + metric decisions in notes/advanced/eval-notes.md.

Key properties (so this stays interpretable, not just runnable):
  - Reads ANY retriever object with a `.retrieve(question, k, company)` method.
    The baseline is the naive dense `Retriever`; later we pass a
    `RerankingRetriever`/`HybridRetriever` and run the exact same scoring. A/B
    is "evaluate(config A) vs evaluate(config B)".
  - Recall is reported BOTH ways: hit-rate@k (≥1 relevant in top-k) and
    fraction recall@k (|retrieved ∩ relevant| / |relevant|).
  - Headline @5 (the generator's real window) + an @10 diagnostic (so near
    misses just outside the window are visible — that's what reranking fixes).
  - MRR = reciprocal rank of the first relevant chunk in the returned list.
  - The negative control (empty relevant_ids) is scored SEPARATELY, never
    averaged in (recall/MRR are undefined with no relevant chunks).
  - Aggregates are broken out per category, because the point is seeing WHICH
    category each future pattern improves.
"""

from __future__ import annotations

import json

from app.config import config
from app.retrieve import Retriever
from app.store import get_vector_store

GOLDEN_PATH = config.root / "eval" / "golden.jsonl"

# How deep to retrieve during eval. The generator's real window is 5, but we
# pull 10 so recall@10 / MRR can see relevant chunks sitting just outside the
# top-5 — exactly the near-misses reranking is meant to rescue.
EVAL_DEPTH = 10


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_golden(path=GOLDEN_PATH) -> list[dict]:
    """Read the golden set (one JSON object per line)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Metrics (pure functions over id lists — easy to reason about and test)
# ---------------------------------------------------------------------------


def recall_at(retrieved_ids: list[str], relevant: set[str], k: int) -> tuple[float, int]:
    """Return (fraction_recall@k, hit@k) over the top-k of `retrieved_ids`.

    fraction = how many of the relevant chunks landed in the top-k, as a ratio.
    hit      = 1 if at least one relevant chunk is in the top-k, else 0.
    """
    topk = retrieved_ids[:k]
    found = sum(1 for r in relevant if r in topk)
    fraction = found / len(relevant)
    hit = 1 if found > 0 else 0
    return fraction, hit


def mrr(retrieved_ids: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the FIRST relevant chunk in the returned list (0 if none)."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(retriever, golden: list[dict], depth: int = EVAL_DEPTH) -> list[dict]:
    """Score every golden question against `retriever`. Returns per-question rows.

    Each row carries the question's metrics (or, for the negative control with
    no relevant chunks, just the top-1 similarity). The retriever is any object
    exposing `.retrieve(question, k, company)`.
    """
    rows = []
    for g in golden:
        relevant = set(g["relevant_ids"])
        results = retriever.retrieve(g["question"], k=depth, company=g.get("company"))
        retrieved_ids = [r["id"] for r in results]
        top1 = results[0]["similarity"] if results else 0.0

        row = {
            "id": g["id"],
            "category": g["category"],
            "question": g["question"],
            "top1": top1,
        }

        # Negative control: no relevant chunk exists. Recall/MRR are undefined,
        # so we don't compute them — the meaningful signal is the top-1 sim.
        if not relevant:
            row["control"] = True
            rows.append(row)
            continue

        f5, h5 = recall_at(retrieved_ids, relevant, 5)
        # recall over the FULL retrieved pool (= the ceiling for reranking at this
        # depth — reranking can only reorder what was retrieved).
        fpool, _ = recall_at(retrieved_ids, relevant, depth)
        row.update(
            control=False,
            # Is `relevant_ids` the COMPLETE answer set (fractional recall is fair)
            # or just a representative sample (recall has a fake denominator —
            # judge on hit@k/MRR only)? Set per-question in golden.jsonl; default
            # True for backward compatibility. See notes/advanced/eval-audit.md.
            recall_reliable=g.get("recall_reliable", True),
            n_relevant=len(relevant),
            recall5=f5,
            hit5=h5,
            recall_pool=fpool,
            mrr=mrr(retrieved_ids, relevant),
            # which relevant chunks we missed in the top-5 (diagnostic)
            missed5=sorted(relevant - set(retrieved_ids[:5])),
        )
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Mean metrics overall and per category, excluding the negative control.

    Fraction recall is averaged ONLY over `recall_reliable` questions — those
    whose `relevant_ids` is the complete answer set, so the denominator is real.
    hit@5 and MRR are averaged over ALL scored questions (they're valid even when
    the label set is only representative). So recall and hit/MRR have DIFFERENT
    denominators (`n_rel` vs `n`), both reported. See notes/advanced/eval-audit.md.
    """
    scored = [r for r in rows if not r.get("control")]

    def _means(subset: list[dict]) -> dict | None:
        if not subset:
            return None
        n = len(subset)
        reliable = [r for r in subset if r.get("recall_reliable", True)]
        nr = len(reliable)
        return {
            "n": n,                # hit@5 / MRR denominator (all scored)
            "n_rel": nr,           # fraction-recall denominator (reliable only)
            "hit5": sum(r["hit5"] for r in subset) / n,
            "mrr": sum(r["mrr"] for r in subset) / n,
            "recall5": (sum(r["recall5"] for r in reliable) / nr) if nr else None,
            "recall_pool": (sum(r["recall_pool"] for r in reliable) / nr) if nr else None,
        }

    by_cat = {}
    for cat in sorted({r["category"] for r in scored}):
        by_cat[cat] = _means([r for r in scored if r["category"] == cat])
    return {"overall": _means(scored), "by_category": by_cat}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_report(rows: list[dict], agg: dict, label: str, depth: int) -> None:
    pool_col = f"rec@{depth}"
    print()
    print(f"  === RETRIEVAL EVAL — {label} ===")
    print(f"  (retrieved depth {depth}; headline @5, ceiling @{depth})")
    print()
    print(f"  {'Q':>3}  {'category':<15} {'hit@5':>5} {'rec@5':>6} {pool_col:>7} {'MRR':>5}  question")
    print(f"  {'-'*3}  {'-'*15} {'-'*5} {'-'*6} {'-'*7} {'-'*5}  {'-'*40}")
    for r in rows:
        q = r["question"][:46]
        if r.get("control"):
            print(f"  {r['id']:>3}  {r['category']:<15} {'—':>5} {'—':>6} {'—':>7} {'—':>5}  {q}")
            continue
        if r.get("recall_reliable", True):
            rec5 = f"{r['recall5']:>6.2f}"
            recp = f"{r['recall_pool']:>7.2f}"
            flag = "" if r["recall5"] == 1.0 else "  ← misses in top-5: " + ",".join(
                m.split("-")[-1] for m in r["missed5"]
            )
        else:
            # representative labels: show the recall number in parens to signal
            # it is NOT aggregated (its denominator is a sample, not the answer).
            rec5 = f"({r['recall5']:.2f})".rjust(6)
            recp = f"({r['recall_pool']:.2f})".rjust(7)
            flag = "  (representative labels — recall not aggregated)"
        print(
            f"  {r['id']:>3}  {r['category']:<15} {r['hit5']:>5} {rec5} "
            f"{recp} {r['mrr']:>5.2f}  {q}{flag}"
        )

    print()
    print("  === AGGREGATE (excluding negative control) ===")

    def _agg_line(name: str, m: dict) -> str:
        r5 = f"{m['recall5']:.2f}" if m["recall5"] is not None else "—"
        rp = f"{m['recall_pool']:.2f}" if m["recall_pool"] is not None else "—"
        # recall uses the reliable-only denominator (n_rel); hit/MRR use all (n)
        return (
            f"  {name:<16} recall@5={r5} recall@{depth}={rp} (n_rel={m['n_rel']})"
            f"  ·  hit@5={m['hit5']:.2f} MRR={m['mrr']:.2f} (n={m['n']})"
        )

    print(_agg_line("overall", agg["overall"]))
    for cat, m in agg["by_category"].items():
        print(_agg_line(cat, m))

    # Negative control reported on its own.
    controls = [r for r in rows if r.get("control")]
    if controls:
        print()
        print("  --- negative control (scored separately) ---")
        for r in controls:
            band = "noise floor — corpus has no good match (expected)" if r["top1"] < 0.58 else "WARN: unexpectedly high"
            print(f"  Q{r['id']} top-1 sim={r['top1']:.4f}  ({band})")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_cli(args) -> None:
    """`python cli.py eval` — score the naive dense retriever against the golden set.

    This is the baseline every advanced pattern must beat. Later patterns plug
    in by constructing a different retriever and calling evaluate() on it.
    """
    depth = getattr(args, "depth", None) or EVAL_DEPTH
    golden = load_golden()
    retriever = Retriever(get_vector_store())
    label = "baseline (naive dense)"

    # Optional: wrap the base retriever in the cross-encoder reranker (A/B).
    if getattr(args, "rerank", False):
        from app.rerank import RerankingRetriever, Reranker, RERANKER_MODELS, DEFAULT_CANDIDATE_POOL

        pool = getattr(args, "candidates", None) or DEFAULT_CANDIDATE_POOL
        key = getattr(args, "reranker", None) or "minilm"
        model_name = RERANKER_MODELS.get(key, key)  # accept a shorthand or a full model name
        retriever = RerankingRetriever(retriever, reranker=Reranker(model_name), candidate_pool=pool)
        label = f"reranked ({key}, pool={pool})"

    # Optional: wrap in the cross-company round-robin decomposer (Experiment 7, Phase A).
    if getattr(args, "decompose", False):
        from app.decompose import DecompositionRetriever

        retriever = DecompositionRetriever(retriever)
        label = "decomposed (round-robin)" if label == "baseline (naive dense)" else f"{label} + decomposed"

    # Optional: wrap in the LLM query decomposer (Experiment 7, Phase B).
    if getattr(args, "llm_decompose", False):
        from app.llm_decompose import LLMDecompositionRetriever, DECOMPOSER_MODELS

        key = getattr(args, "decomposer", None) or "haiku"
        model_name = DECOMPOSER_MODELS.get(key, key)  # shorthand or a full model name
        sub_filter = getattr(args, "sub_filter", False)
        retriever = LLMDecompositionRetriever(retriever, model=model_name, filter_subqueries=sub_filter)
        tag = f"llm-decomposed ({key}{'+filter' if sub_filter else ''})"
        label = tag if label == "baseline (naive dense)" else f"{label} + {tag}"

    rows = evaluate(retriever, golden, depth=depth)
    agg = aggregate(rows)
    _print_report(rows, agg, label=label, depth=depth)
