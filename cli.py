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
# Future imports (added stage by stage):
# from app import store, retrieve, generate


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

    # --- Stage 4: build (full index pipeline) ---
    p_build = sub.add_parser("build", help="(Stage 4) Run ingest+chunk+embed+store for all tickers.")
    p_build.set_defaults(func=_not_yet("build"))

    # --- Stage 5: retrieve ---
    p_retrieve = sub.add_parser("retrieve", help="(Stage 5) Retrieve top-k chunks for a question.")
    p_retrieve.add_argument("--question", required=True)
    p_retrieve.add_argument("--company", default=None, help="Optional ticker filter, e.g. TSLA")
    p_retrieve.set_defaults(func=_not_yet("retrieve"))

    # --- Stage 6: ask ---
    p_ask = sub.add_parser("ask", help="(Stage 6) Full RAG: retrieve + generate with citations.")
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--company", default=None)
    p_ask.set_defaults(func=_not_yet("ask"))

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
