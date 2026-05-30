# module-02-rag-app

A citation-grounded Q&A copilot over SEC 10-K filings — built as the project for Module 02 (RAG) of an AI engineering curriculum.

The goal is genuine, defendable understanding of how a retrieval-augmented generation system fits together end to end. The codebase deliberately favors transparency over cleverness: each stage is a separate module, each subcommand is independently runnable, and the major design decisions are explained in companion `*-notes.md` files alongside the code.

## What this project does

Given a question like *"What are Tesla's main risks?"*, the system:

1. **Ingests** the latest 10-K filings from SEC EDGAR for a small set of tickers (TSLA, AAPL, NVDA) and cleans them into per-section text.
2. **Chunks** each section into ~1000-character pieces with structural awareness (paragraph and sentence boundaries respected, deterministic IDs).
3. **Embeds** every chunk into a 384-dimensional vector using `BAAI/bge-small-en-v1.5` running locally on CPU.
4. **Stores** the chunks (text + vector + metadata) in a persistent Chroma collection.
5. **Retrieves** the top-k chunks most similar to the user's question, with optional company-level metadata filtering.
6. **Generates** a grounded answer with inline citations to the source filings using Anthropic Claude, refusing cleanly when the retrieved chunks don't support an answer.

## Quick start

```bash
# 1. Clone the repo and enter it
cd module-02-rag-app

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure secrets — copy the template and fill in two values
cp .env.example .env
#  Fill in:
#   SEC_USER_AGENT="Your Name your_email@example.com"   (SEC requires this)
#   ANTHROPIC_API_KEY="sk-ant-..."                       (for Stage 6)

# 4. Build the index from scratch
python cli.py ingest --ticker TSLA   # repeat for AAPL, NVDA
python cli.py chunk  --ticker TSLA   # repeat for AAPL, NVDA
python cli.py build                  # embed + store all chunks at once

# 5. Try a retrieval
python cli.py retrieve --question "What are the main risks Tesla faces?" --company TSLA
```

## Pipeline overview

Each stage produces an artifact that the next stage consumes. Every stage is independently runnable and inspectable.

```
EDGAR (10-K HTML)
        │
        ▼  ingest      → data/raw/{TICKER}.html       (cached download)
                       → data/clean/{TICKER}.json     (per-section cleaned text)
        │
        ▼  chunk       → data/chunks/{TICKER}.jsonl   (~250 chunks/filing, deterministic IDs)
        │
        ▼  embed       (384-d BGE vectors, normalized)
        │
        ▼  store       → data/chroma/                 (Chroma collection: 678 rows total)
        │
        ▼  retrieve    (top-k cosine, optional ticker filter, confidence label on top-1)
        │
        ▼  generate    (Claude: grounded-citation prompt + hybrid refusal + citation audit)
```

## CLI reference

All subcommands are exposed via a single entry point: `python cli.py`.

```bash
# Stage 1 — fetch + clean one 10-K from EDGAR
python cli.py ingest --ticker TSLA

# Stage 2 — chunk one cleaned filing
python cli.py chunk --ticker TSLA

# Stage 3 — embedder sanity check
python cli.py embed --text "supply chain risk"
python cli.py embed --text "supply chain risk" \
                    --text "we depend on third-party vendors" \
                    --text "the company logo is red"

# Stage 4 — build / store / inspect
python cli.py build                       # embed + upsert all chunks
python cli.py store --ticker TSLA         # single-ticker rebuild (dev iteration)
python cli.py inspect                     # collection count + per-ticker counts + sample row

# Stage 5 — retrieve
python cli.py retrieve --question "..."                       # unfiltered
python cli.py retrieve --question "..." --company TSLA        # filtered
python cli.py retrieve --question "..." --company TSLA --k 8  # custom top-k
python cli.py retrieve --question "..." --company TSLA --compare   # filtered vs unfiltered side by side

# Stage 6 — full RAG: retrieve + generate with citations
python cli.py ask --question "..."                       # unfiltered
python cli.py ask --question "..." --company TSLA         # filtered
python cli.py ask --question "..." --company TSLA --k 8   # custom top-k
```

## Repository layout

```
module-02-rag-app/
├── README.md                  ← this file
├── WHY.md                      ← cross-cutting design rationale (why the system is shaped this way)
├── prompt-instructions.md     ← original project spec
├── .env.example               ← config template (commit-safe)
├── .env                       ← real config (gitignored)
├── requirements.txt
├── cli.py                     ← single entry point, all subcommands

├── app/
│   ├── config.py              ← central configuration (paths, tickers, model names, knobs)
│   ├── ingest.py              ← Stage 1: EDGAR client, section-aware parser
│   ├── chunking.py            ← Stage 2: recursive structure-aware chunker
│   ├── embed.py               ← Stage 3: Embedder ABC + local BGE implementation
│   ├── store.py               ← Stage 4: VectorStore ABC + Chroma wrapper
│   ├── retrieve.py            ← Stage 5: top-k orchestration + UX mitigations
│   └── generate.py            ← Stage 6: Generator (citation contract + hybrid refusal + audit)

├── data/                      ← (gitignored) all build artifacts
│   ├── raw/                   ← cached 10-K HTML
│   ├── clean/                 ← parsed section JSON
│   ├── chunks/                ← chunked JSONL
│   └── chroma/                ← persisted vector store

# Design notes — one per stage
├── notes/
│   ├── ingest-observation.md  ← Stage 1: 3-iteration regex diagnostic story
│   ├── chunking-notes.md      ← Stage 2: design + experiments + implementation decisions
│   ├── embedding-notes.md     ← Stage 3: intuition + BGE quirks + score-compression lesson
│   ├── store-chroma-notes.md  ← Stage 4: numpy-vs-vectorDB framing + design decisions
│   ├── retrieval-notes.md     ← Stage 5: filtered-vs-unfiltered + pressure tests + findings
│   └── generation-notes.md    ← Stage 6: citation contract + hybrid refusal + injection defense

# Session continuity
└── SESSION-STATE.md           ← state-of-the-build for resuming across sessions
```

## Build status

| Stage | Status | Result snapshot |
|---|---|---|
| 0 — Scaffold | done | `.env.example`, `requirements.txt`, `config.py`, `cli.py` skeleton |
| 1 — Ingest | done | 3 filings parsed cleanly; 4–5 sections per filing |
| 2 — Chunk | done | 678 chunks total (TSLA 251, AAPL 149, NVDA 278); median ~890 chars |
| 3 — Embed | done | BGE-small-en-v1.5 wrapper + sanity-check CLI |
| 4 — Store | done | Chroma collection `filings` with 678 rows; metadata filter verified |
| 5 — Retrieve | done | Top-k with optional `--company` filter; warning + confidence label mitigations verified |
| 6 — Generate | done | Claude Opus 4.8: citation contract + hybrid refusal gate + citation audit; 5-question run, 0 hallucinated citations |
| 7 — Polish | done | `WHY.md` design rationale + README pass |

## Where to read for depth

The companion `*-notes.md` files are the real documentation. Each one captures intuition, design decisions, and lessons learned for one stage:

| File | Read when you want to understand… |
|---|---|
| `notes/ingest-observation.md` | how text-parsing heuristics get refined iteratively against failure modes (3 regex iterations on real 10-K data) |
| `notes/chunking-notes.md` | why structure-aware chunking matters, how chunk size and overlap interact with retrieval, and what went wrong in Experiment 1 |
| `notes/embedding-notes.md` | what an embedding actually is, BGE's query-prefix quirk, why model-specific score calibration matters more than you'd think |
| `notes/store-chroma-notes.md` | when a vector database earns its keep vs. plain numpy, why metadata filtering is the cheapest accuracy lever |
| `notes/retrieval-notes.md` | pressure-tests of the retrieval design, real failure modes observed, why cross-company questions break naive top-k |
| `notes/generation-notes.md` | the citation contract and how it's enforced, the hybrid refusal gate, prompt-injection defense by role discipline, why refusal is three-state |
| `WHY.md` | the horizontal view — the cross-cutting design principles and "why X not Y" decisions that span all six stages |

Together with the inline code comments, these notes are the design document. Reading them in stage order builds the same picture the code does, from a teaching perspective rather than an implementation one.

## Design philosophy

A few principles run through the codebase:

- **One module per stage.** Each can be run, tested, and inspected on its own. No stage knows about more than the immediately adjacent ones.
- **Interfaces at every swap point.** `Embedder` (Stage 3) and `VectorStore` (Stage 4) are abstract classes with concrete defaults. Adding an OpenAI embedder or a Qdrant backend should be one file, not a refactor.
- **Mechanism stays visible.** No clever abstractions hiding the math. The chunker is ~120 lines; the retriever is ~30. If you can't read every step of what happens to your data, you can't debug it.
- **Notes before code.** Every stage was designed in its `*-notes.md` file first, then implemented, then the notes were updated with what actually happened on a real run. This is a deliberate practice — design, build, measure, write — not after-the-fact documentation.
- **Honest about limitations.** Real failure modes (cross-company comparison questions, score compression in BGE, redundant filters when the query encodes the company) are documented as observations, not papered over. The fix queue is explicit.

## License

Personal learning project. No license. Don't depend on this for anything that matters.
