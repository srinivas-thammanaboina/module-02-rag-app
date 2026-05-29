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

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
