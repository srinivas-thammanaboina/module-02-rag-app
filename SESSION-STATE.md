# Session State — module-02-rag-app

> Paste this file at the start of the next session to resume.

## Where we are

Building a **citation-grounded Q&A copilot over SEC 10-K filings** (per `prompt-instructions.md`). Sequential stage-by-stage build, pausing for review after each stage.

## Confirmed decisions

- **Generation:** Anthropic, model `claude-opus-4-6`
- **Embedding:** local `BAAI/bge-small-en-v1.5` behind an `Embedder` interface
- **Vector DB:** Chroma, persisted to `data/chroma/`
- **Tickers:** TSLA, AAPL, NVDA
- **SEC User-Agent:** placeholder in `.env` (user will fill in)
- **CLI:** single `cli.py` with subcommands
- **Cache raw HTML:** yes, under `data/raw/`
- **No hybrid search or re-ranking yet** — pure dense retrieval first
- **No clever abstractions** — mechanism must stay visible
- Code comments: high-level on classes/methods + important logic only

## Build order + status

| # | Stage | Status |
|---|---|---|
| 0 | Scaffold (`.env.example`, `.gitignore`, `requirements.txt`, `app/config.py`, `cli.py`) | done |
| 1 | Ingest — EDGAR client, ticker→CIK→latest 10-K, clean HTML, section split | **done** — iterated regex 3 times, all 3 tickers parse cleanly |
| 2 | Chunk — structure-aware splitter, metadata propagation | **designed, approved, not yet coded** |
| 3 | Embed — `Embedder` interface + local `bge-small` | pending |
| 4 | Store — Chroma persistence + full index build | pending |
| 5 | Retrieve — top-k + metadata filter | pending |
| 6 | Generate — Anthropic call with citation prompt | pending |
| 7 | `WHY.md` + `README.md` | pending |

## Stage 1 result snapshot (final)

| Section | TSLA | AAPL | NVDA |
|---|---|---|---|
| Item 1 (Business) | 45,455 | 16,053 | 48,241 |
| Item 1A (Risk Factors) | 83,740 | 68,047 | 114,916 |
| Item 3 (Legal) | — | 5,401 | — |
| Item 7 (MD&A) | 55,454 | 18,020 | 34,154 |
| Item 7A (Market Risk) | 1,625 | 3,023 | 4,253 |

TSLA and NVDA file legal proceedings as a cross-reference to financial-statement notes (one sentence, below our 500-char floor). Legitimately absent, not a bug.

## Stage 2 plan (approved, ready to code on next session)

Detailed in `chunking-notes.md`. Summary:

- **Algorithm:** paragraph-pack-then-overflow-split. Split section into paragraphs, greedy-pack into chunks under `max_size`, recursively split oversized paragraphs at sentence → line → word → hard cut. Verbatim overlap copied from previous chunk, snapped back to a sentence boundary.
- **Parameters (Experiment 1):** `max_size=1000`, `overlap=150`, `min_size=200`.
- **Metadata per chunk:** `ticker`, `company_name`, `section`, `filing_date`, `cik`, `accession_number`, `source_url`, `chunk_index`, `char_start`, `char_end`.
- **No LangChain** — we write the ~60-line splitter ourselves so every decision is visible.
- **CLI:** `python cli.py chunk --ticker TSLA` prints count, length distribution, and 2-3 sample chunks with full metadata. We'll also need a `chunk-all` to run the full corpus.

## Files to create / edit next session

1. `app/chunking.py` — new module with the splitter + a `chunk_filing(filing_dict)` orchestration function.
2. `cli.py` — add `chunk` subcommand wired to `app.chunking.run_cli`.
3. `data/chunks/` — new output directory (alongside `data/clean/`) for `{TICKER}.jsonl` files.
4. After running, fill in Experiment 1's "Results placeholder" in `chunking-notes.md`.

## Files on disk now

```
module-02-rag-app/
├── .env.example
├── .gitignore
├── requirements.txt
├── prompt-instructions.md
├── SESSION-STATE.md          ← this file
├── ingest-observation.md     ← Stage 1 diagnostic playbook (3-iteration regex story)
├── chunking-notes.md         ← Stage 2 design + Experiment 1 record (NEW)
├── cli.py                    ← `ingest` wired up; `chunk` to be added
├── app/
│   ├── __init__.py
│   ├── config.py
│   └── ingest.py             ← three-stacked-signal regex; final
└── data/
    ├── raw/                  ← cached 10-K HTML for TSLA/AAPL/NVDA
    ├── clean/                ← parsed section JSON for all three
    └── chroma/.gitkeep
```

## What I should do at the start of next session

1. Re-read `SESSION-STATE.md` (this file) — recover full context.
2. Re-read `chunking-notes.md` — confirm Experiment 1 parameters are still the approved plan.
3. Re-read `ingest-observation.md` for the regex iteration story if the user mentions Stage 1.
4. Write `app/chunking.py` and add the CLI subcommand.
5. Run on all three tickers, fill in Experiment 1's results in `chunking-notes.md`.
6. Show the user sample chunks for inspection before moving on.

## Open teaching threads still to revisit

- **Stage 3:** why we hide embedder behind an interface; concrete "swap to API" walkthrough.
- **Stage 4:** what a Chroma row contains (vector + doc + metadata); metadata filter as the cheapest accuracy win — prove it in Stage 5.
- **Stage 5:** developing intuition for similarity scores (what 0.61 vs 0.84 *feels* like).
- **Stage 6:** the citation prompt is the whole game; behavior when chunks don't contain the answer.

## What's on disk now

```
module-02-rag-app/
├── .env.example          ← created (user copies to .env and fills in)
├── .gitignore
├── requirements.txt
├── prompt-instructions.md
├── SESSION-STATE.md      ← this file
├── cli.py                ← subcommands registered; only `ingest` wired up
├── app/
│   ├── __init__.py
│   ├── config.py         ← single Config dataclass; everything tunable
│   └── ingest.py         ← Stage 1: EDGARClient + FilingFetcher + FilingParser
└── data/
    ├── raw/.gitkeep
    ├── clean/.gitkeep
    └── chroma/.gitkeep
```

## What user is supposed to do before next session (optional)

```bash
cd /Users/srinivasthammanaboina/Projects/module-02-rag-app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # then edit SEC_USER_AGENT
python cli.py ingest --ticker TSLA
# repeat for AAPL and NVDA if Tesla looks right
```

Expected output: company name, CIK, accession, EDGAR URL, plus 4–5 sections (Item 1, 1A, 3, 7, 7A) with char counts and previews. Files land in `data/clean/<TICKER>.json`.

## Open question I left the user with (interview-style)

> Why are we keeping `source_url`, `accession_number`, and `cik` separately on every record, when we could just compute the URL?
>
> Hint: think about Stage 6 — when the LLM cites a chunk, what does the user click? What if EDGAR changes its URL scheme?

## Teaching threads to revisit

- **Stage 1 quirks already covered:** EDGAR User-Agent rule, ~10 req/s rate limit, `company_tickers.json` is keyed by integer not ticker, submissions endpoint uses parallel arrays, inline-XBRL bloat, "every Item appears twice (TOC + body) — keep the longest" trick.
- **Stage 2 to cover:** why recursive/structure-aware beats fixed-size on filings; how metadata travels with each chunk; the chunk-size/overlap tradeoff and how to *see* its effect.
- **Stage 3 to cover:** why we hide embedder behind an interface; concrete "swap to API" walkthrough.
- **Stage 4 to cover:** what a Chroma row contains (vector + doc + metadata); metadata filter as the cheapest accuracy win — prove it in Stage 5.
- **Stage 5 to cover:** developing intuition for similarity scores (what 0.61 vs 0.84 *feels* like).
- **Stage 6 to cover:** the citation prompt is the whole game; behavior when chunks don't contain the answer.

## Curriculum context (do not lose)

This is the project for **Module 02 (RAG)** of the AI engineering curriculum at `~/Projects/ai-engineering-notes/`. Theory phase is done; notes are in `02-rag/`. The user values genuine understanding they can defend in an interview, not just a working pipeline. Teach, don't just tell. Be direct.
