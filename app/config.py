"""
Central configuration.

Every knob the spec marked "must be configurable" lives here:
  - SEC User-Agent (loaded from .env, refuses placeholder)
  - tickers, embedding model, chunk size/overlap, top-k, generation model

Why one config module: business logic (ingest, chunk, embed...) imports a single
object. Nothing reads env vars directly. Makes "swap the model" / "change top-k"
a one-line edit and makes the pipeline easy to reason about.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of the `app/` package.
ROOT = Path(__file__).resolve().parent.parent

# Load .env once at import time. Safe if .env doesn't exist.
load_dotenv(ROOT / ".env")


# A sentinel value that means "user never filled this in". We fail loudly
# rather than send the placeholder to SEC (it would get blocked).
_USER_AGENT_PLACEHOLDER = "Your Name your_email@example.com"


@dataclass
class Config:
    # --- Paths ---
    root: Path = ROOT
    raw_dir: Path = ROOT / "data" / "raw"
    clean_dir: Path = ROOT / "data" / "clean"
    chroma_dir: Path = ROOT / "data" / "chroma"

    # --- Ingest ---
    # Tickers we'll index. Spec asked for 2-3 with Tesla + Apple + NVDA.
    tickers: list[str] = field(default_factory=lambda: ["TSLA", "AAPL", "NVDA"])
    # SEC rate limit is ~10 req/s. 0.15s between requests keeps us well under.
    sec_rate_limit_delay: float = 0.15
    sec_user_agent: str = ""  # filled in from env below

    # --- Chunk ---
    # Recursive/structure-aware chunker config (Experiment 1 in notes/chunking-notes.md).
    # Tune these and observe the effect on retrieval — that's the whole point
    # of having them here.
    chunk_size: int = 1000       # max chars per chunk (budget, not target)
    chunk_overlap: int = 150     # chars of overlap copied from previous chunk
    chunk_min_size: int = 200    # drop trailing fragments shorter than this

    # --- Embed ---
    # bge-small-en-v1.5: 384 dims, fast on CPU, strong on English.
    # Hidden behind the Embedder interface so we can swap to an API later.
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Retrieve ---
    top_k: int = 5

    # --- Generate ---
    # Opus 4.8 (bumped from 4.6 at the start of Stage 6 — stronger on the exact
    # Stage 6 stressors: instruction-following on the citation contract, clean
    # refusals, and resisting chunk-boundary prompt injection).
    anthropic_model: str = "claude-opus-4-8"
    anthropic_api_key: str = ""

    # Decomposer model (advanced stage, decomposition Phase B). Query-splitting is
    # a cheap, simple task — Haiku, not the Opus answer-model. Used by
    # app/llm_decompose.py; see notes/advanced/decomposition-notes.md.
    decomposer_model: str = "claude-haiku-4-5"

    # Refusal gate thresholds for the hybrid refusal policy (Stage 6).
    # CALIBRATED FOR bge-small-en-v1.5 ONLY — these are top-1 cosine bands from
    # notes/embedding-notes.md. Swap the embedding model and BOTH numbers must be
    # re-derived against the new model's noise floor (see retrieval-notes.md
    # Finding 5), or the gate silently misfires.
    #   top-1 < refuse_floor            -> hard refusal, no API call
    #   refuse_floor <= top-1 < grey    -> call the model, warn it retrieval was weak
    #   top-1 >= grey                   -> answer normally
    refuse_floor: float = 0.52
    refuse_grey: float = 0.58

    def __post_init__(self) -> None:
        self.sec_user_agent = os.getenv("SEC_USER_AGENT", "").strip()
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    def require_sec_user_agent(self) -> str:
        """Call before any EDGAR request. Fails clearly if .env isn't set."""
        if not self.sec_user_agent or self.sec_user_agent == _USER_AGENT_PLACEHOLDER:
            raise RuntimeError(
                "SEC_USER_AGENT is not set in .env. SEC requires a "
                "'Your Name your_email@example.com' header on every request."
            )
        return self.sec_user_agent

    def require_anthropic_key(self) -> str:
        """Call before any Anthropic request. Fails clearly if .env isn't set."""
        if not self.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in .env.")
        return self.anthropic_api_key


# Single shared instance. Modules do: `from app.config import config`
config = Config()
