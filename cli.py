"""
Single CLI entry point.

Each pipeline stage is a subcommand so they can be run independently:

    python cli.py ingest    --ticker TSLA
    python cli.py chunk     --ticker TSLA
    python cli.py embed     --text "supply chain risk"
    python cli.py build                                 # ingest+chunk+embed+store for all tickers
    python cli.py retrieve  --question "..." [--company TSLA]
    python cli.py ask       --question "..." [--company TSLA]

Stages register themselves below. Subcommands not yet implemented print
a friendly "coming next stage" message.
"""

from __future__ import annotations

import argparse
import sys

from app import ingest    # Stage 1
from app import chunking  # Stage 2
from app import embed     # Stage 3
from app import store     # Stage 4
from app import retrieve  # Stage 5
from app import generate  # Stage 6
from app import eval as evaluation  # Advanced stage: retrieval eval harness


def _not_yet(stage_name: str):
    def _run(_args):
        print(f"[stage not implemented yet: {stage_name}]")
        sys.exit(1)
    return _run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-app",
        description="Citation-grounded Q&A copilot over SEC 10-K filings.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- Stage 1: ingest ---
    p_ingest = sub.add_parser(
        "ingest",
        help="Fetch & clean a 10-K from EDGAR for one ticker.",
    )
    p_ingest.add_argument("--ticker", required=True, help="e.g. TSLA")
    p_ingest.set_defaults(func=ingest.run_cli)

    # --- Stage 2: chunk ---
    p_chunk = sub.add_parser("chunk", help="Split a cleaned filing into chunks.")
    p_chunk.add_argument("--ticker", required=True, help="e.g. TSLA")
    p_chunk.set_defaults(func=chunking.run_cli)

    # --- Stage 3: embed ---
    p_embed = sub.add_parser(
        "embed",
        help="Embed text. One --text shows vector stats; two+ shows cosine similarity (first is the query).",
    )
    p_embed.add_argument(
        "--text",
        required=True,
        action="append",
        help="Pass --text once for vector stats, or multiple times: first is query, rest are documents.",
    )
    p_embed.set_defaults(func=embed.run_cli)

    # --- Stage 4: build (full index from data/chunks/*.jsonl) ---
    p_build = sub.add_parser(
        "build",
        help="Embed all chunks and upsert into the Chroma index.",
    )
    p_build.set_defaults(func=store.run_build_cli)

    # --- Stage 4: store (single-ticker rebuild for iteration) ---
    p_store = sub.add_parser(
        "store",
        help="Embed one ticker's chunks and upsert. For development iteration.",
    )
    p_store.add_argument("--ticker", required=True, help="e.g. TSLA")
    p_store.set_defaults(func=store.run_store_cli)

    # --- Stage 4: inspect (sanity check on the persisted collection) ---
    p_inspect = sub.add_parser(
        "inspect",
        help="Print collection size, per-ticker counts, and a sample row.",
    )
    p_inspect.set_defaults(func=store.run_inspect_cli)

    # --- Stage 5: retrieve ---
    p_retrieve = sub.add_parser(
        "retrieve",
        help="Retrieve top-k chunks for a question.",
    )
    p_retrieve.add_argument("--question", required=True)
    p_retrieve.add_argument("--company", default=None, help="Optional ticker filter, e.g. TSLA")
    p_retrieve.add_argument("--k", type=int, default=5, help="Top-k (default: 5)")
    p_retrieve.add_argument(
        "--compare",
        action="store_true",
        help="Run both filtered (uses --company) and unfiltered retrieval, side by side.",
    )
    p_retrieve.set_defaults(func=retrieve.run_cli)

    # --- Stage 6: ask ---
    p_ask = sub.add_parser("ask", help="(Stage 6) Full RAG: retrieve + generate with citations.")
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--company", default=None, help="Optional ticker filter, e.g. TSLA")
    p_ask.add_argument("--k", type=int, default=None, help="Top-k chunks to ground in (default: config.top_k)")
    p_ask.set_defaults(func=generate.run_cli)

    # --- Advanced stage: eval (retrieval quality vs golden set) ---
    p_eval = sub.add_parser(
        "eval",
        help="Score retrieval (recall@k + MRR) against eval/golden.jsonl.",
    )
    p_eval.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Retrieval pool depth (default 10). recall@depth = the ceiling a reranker on this pool could reach.",
    )
    p_eval.add_argument(
        "--rerank",
        action="store_true",
        help="Wrap the dense retriever in the cross-encoder reranker (retrieve wide -> rerank narrow).",
    )
    p_eval.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="Reranker candidate pool size (default 50). Only used with --rerank.",
    )
    p_eval.add_argument(
        "--reranker",
        default="minilm",
        help="Cross-encoder for --rerank: 'minilm' (default) or 'bge', or a full model name.",
    )
    p_eval.add_argument(
        "--hybrid",
        action="store_true",
        help="Wrap the dense retriever in hybrid (dense + BM25, fused with RRF). Targets the lexical/opaque-token category.",
    )
    p_eval.add_argument(
        "--fusion",
        default=None,
        choices=["rrf", "interleave"],
        help="Hybrid fusion: 'rrf' (consensus, default) or 'interleave' (round-robin, guaranteed slots per lane). Only used with --hybrid.",
    )
    p_eval.add_argument(
        "--rrf-k",
        dest="rrf_k",
        type=int,
        default=None,
        help="RRF constant (default 60). Larger = top ranks dominate less. Only used with --hybrid --fusion rrf.",
    )
    p_eval.add_argument(
        "--hybrid-gate",
        dest="hybrid_gate",
        action="store_true",
        help="Engage BM25 only when the query carries an opaque-identifier token; pass through to pure dense otherwise. Only used with --hybrid.",
    )
    p_eval.add_argument(
        "--decompose",
        action="store_true",
        help="Wrap the dense retriever in the cross-company round-robin decomposer (Experiment 7, Phase A).",
    )
    p_eval.add_argument(
        "--llm-decompose",
        dest="llm_decompose",
        action="store_true",
        help="Wrap in the LLM query decomposer (Experiment 7, Phase B — general, cached, costs LLM calls).",
    )
    p_eval.add_argument(
        "--decomposer",
        default=None,
        help="LLM for --llm-decompose: 'haiku' (default), 'sonnet', 'opus', or a full model name.",
    )
    p_eval.add_argument(
        "--sub-filter",
        dest="sub_filter",
        action="store_true",
        help="Phase B+: hard-filter each LLM sub-query that names one company to that ticker. Only used with --llm-decompose.",
    )
    p_eval.set_defaults(func=evaluation.run_cli)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
