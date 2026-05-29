# Session State — module-02-rag-app

> Paste this file at the start of the next session to resume. Say: *"continue module-02-rag-app — read SESSION-STATE.md"*.

## Where we are

Building a **citation-grounded Q&A copilot over SEC 10-K filings** (per `prompt-instructions.md`). Sequential, stage-by-stage build with a pause for review after each stage.

**Stages 1 → 3 are complete.** Stage 4 (Chroma vector store) is next.

## Confirmed decisions (durable)

- **Generation:** Anthropic, model `claude-opus-4-6`
- **Embedding:** local `BAAI/bge-small-en-v1.5` behind an `Embedder` interface (384-dim, L2-normalized)
- **Vector DB:** Chroma, persisted to `data/chroma/`
- **Tickers:** TSLA, AAPL, NVDA
- **SEC User-Agent:** placeholder in `.env` (filled in by user)
- **CLI:** single `cli.py` with subcommands
- **Cache raw HTML:** yes, under `data/raw/`
- **No hybrid search or re-ranking yet** — pure dense retrieval first
- **No clever abstractions** — mechanism must stay visible
- **Code comments:** high-level on classes/methods + important logic only

## Build order + status

| # | Stage | Status |
|---|---|---|
| 0 | Scaffold (`.env.example`, `.gitignore`, `requirements.txt`, `app/config.py`, `cli.py`) | **done** |
| 1 | Ingest — EDGAR client, ticker→CIK→latest 10-K, clean HTML, section split | **done** — three regex iterations, all three tickers parse cleanly |
| 2 | Chunk — structure-aware splitter, metadata propagation | **done** — Experiment 1 (buggy) → Experiment 2 (fixed), 678 chunks total |
| 3 | Embed — `Embedder` interface + local `bge-small` + sanity-check CLI | **done** — vectors normalized, rank ordering correct, score-compression lesson captured |
| 4 | Store — Chroma persistence + full index build | **done** — 678 rows in `filings` collection, ticker filter verified, 6.7s build |
| 5 | Retrieve — top-k + metadata filter + company-mismatch warning + top-1 confidence labels | **done** — five-question sanity check passed, two pressure-test mitigations verified, three findings recorded in `notes/retrieval-notes.md` |
| 6 | Generate — Anthropic call with citation prompt | **next** |
| 6 | Generate — Anthropic call with citation prompt | pending |
| 7 | `WHY.md` + `README.md` | pending |

## Stage 1 result snapshot (cleaned section sizes)

| Section | TSLA | AAPL | NVDA |
|---|---|---|---|
| Item 1 (Business) | 45,455 | 16,053 | 48,241 |
| Item 1A (Risk Factors) | 83,740 | 68,047 | 114,916 |
| Item 3 (Legal) | — | 5,401 | — |
| Item 7 (MD&A) | 55,454 | 18,020 | 34,154 |
| Item 7A (Market Risk) | 1,625 | 3,023 | 4,253 |

TSLA and NVDA file legal proceedings as a cross-reference to financial-statement notes (one sentence, below our 500-char floor). Legitimately absent, not a bug.

## Stage 2 result snapshot (Experiment 2, post-bug-fix)

```
Total chunks: 678   (TSLA 251 | AAPL 149 | NVDA 278)
By section:   Item 1   152
              Item 1A  369
              Item 3     7   (AAPL only)
              Item 7   138
              Item 7A   12

Chunk length: TSLA — min 236 | median 887 | p95 989 | max 1000
              AAPL — min 202 | median 891 | p95 994 | max 1058
              NVDA — min 226 | median 882 | p95 994 | max 1000

Over budget (> 1000): 1 / 678  (intentional absorption guard, AAPL Item 3)
Under floor (< 200):  0 / 678
```

Output is at `data/chunks/{TSLA,AAPL,NVDA}.jsonl` — one chunk per line.

## Stage 3 result snapshot (sanity-check experiment)

```
Model: BAAI/bge-small-en-v1.5
Dim:   384
Norm:  ~1.0  (L2-normalized as expected)

Query: "supply chain risk from foreign suppliers"

  rank 1  sim=0.5870  "we depend on third-party component vendors"           (paraphrase)
  rank 2  sim=0.5572  "our cost of goods sold increased due to inflation"    (same domain)
  rank 3  sim=0.5127  "the company logo and brand identity..."               (unrelated)
```

**Key lesson captured in `notes/embedding-notes.md`:** rank order correct, but BGE's absolute score range is compressed to ~0.45–0.90. Bands like "0.7 is relevant" do not transfer between embedders — calibrate per-model. Initial predicted bands in `notes/embedding-notes.md` were wrong for BGE; they've been corrected with the actual run's evidence.

## Files on disk now

```
module-02-rag-app/
├── .env.example
├── .env                       ← user-filled (gitignored)
├── .gitignore
├── requirements.txt
├── README.md                  ← project entry point + CLI reference + stage status
├── SESSION-STATE.md           ← this file
├── prompt-instructions.md     ← original project spec
├── cli.py                     ← Stages 1–5 wired (ingest / chunk / embed / build·store·inspect / retrieve); ask pending
├── notes/                     ← stage-by-stage design notes (moved into folder at end of Stage 5 session)
│   ├── ingest-observation.md  ← Stage 1: 3-iteration regex diagnostic story
│   ├── chunking-notes.md      ← Stage 2: design + Experiment 1/2 + implementation decisions
│   ├── embedding-notes.md     ← Stage 3: intuition + BGE quirks + score-compression lesson
│   ├── store-chroma-notes.md  ← Stage 4: numpy-vs-vectorDB framing + design decisions
│   └── retrieval-notes.md     ← Stage 5: pressure tests + 5-question results + Experiment 7 queued
├── app/
│   ├── __init__.py
│   ├── config.py              ← central configuration (paths, tickers, model names, knobs)
│   ├── ingest.py              ← Stage 1: EDGAR client + section-aware parser
│   ├── chunking.py            ← Stage 2: RecursiveChunker (absorption guard + budget reseed)
│   ├── embed.py               ← Stage 3: Embedder ABC + LocalSentenceTransformerEmbedder
│   ├── store.py               ← Stage 4: VectorStore ABC + ChromaVectorStore
│   └── retrieve.py            ← Stage 5: Retriever + company-mismatch warning + confidence labels
└── data/                      ← gitignored build artifacts
    ├── raw/                   ← cached 10-K HTML for TSLA / AAPL / NVDA
    ├── clean/                 ← parsed section JSON for all three
    ├── chunks/                ← 678-chunk JSONL (TSLA 251, AAPL 149, NVDA 278)
    └── chroma/                ← persisted vector store (8.3 MB on disk)
```

## Carry-forward TODOs (small, deliberately deferred)

1. **`get_sentence_embedding_dimension` FutureWarning** in `app/embed.py:89`. The method was renamed to `get_embedding_dimension` in a recent sentence-transformers release; one-line fix. Cosmetic only — no functional impact.
2. **CLI tail-preview cropping** in `app/chunking.py:_print_sample_chunk`. The tail slice doesn't snap to a word boundary, so sample chunks display previews that *appear* to start mid-word. The chunk content is correct; only the display is ugly. Will fix on next CLI touch.

## Stage 6 plan (next session)

**What:** wire Claude Opus 4.6 into a `generate` subcommand that takes a question, runs Stage 5 retrieval, formats the chunks into a grounded prompt, and returns an answer with inline `[chunk-id]` citations.

**Why it matters:** Stage 6 is where the prompt becomes the whole game. Every claim in the answer must be backed by a chunk from retrieval, every chunk must be cite-able, and when retrieval is weak (top-1 < 0.58 per `notes/embedding-notes.md` bands) the model must refuse rather than fabricate. The architecture work is small; the prompt design is the lesson.

**Anticipated structure:**

- `app/generate.py` — new module wrapping the Anthropic client. Public surface:
  - `Generator.answer(question, chunks, top_sim) -> dict` — takes Stage 5 output + the confidence signal, returns a structured response (answer text, citations, refused flag)
- `cli.py` — wire `python cli.py ask --question "..." [--company TSLA]` to combine Stage 5 + Stage 6
- `notes/generation-notes.md` — new design notes following the established pattern

**What I'll teach during Stage 6:**

- **The citation contract** — how to make the model emit `[chunk-id]` inline, and how to verify it
- **Refusing cleanly when retrieval is weak** — using Stage 5's top-1 confidence label to decide when to refuse rather than fabricate
- **Prompt-injection defense at the chunk boundary** — filings contain arbitrary text; the prompt must format chunks so they can't escape their data role
- **System prompt vs per-turn prompt division**
- **Output structure that the user can trust and verify**

## What to do at the start of next session

1. Re-read `README.md` (project entry point) and this file.
2. Skim `notes/retrieval-notes.md` — Stage 6's prompt depends on the confidence label and chunk metadata defined there.
3. Skim `notes/embedding-notes.md` — refresh the BGE noise-floor bands; Stage 6's refusal threshold sits there.
4. Create `notes/generation-notes.md` first (matching the pattern), then whiteboard the prompt design, then write `app/generate.py`.
5. Wire `ask` subcommand in `cli.py`.
6. Run the five Stage 5 questions through the full pipeline; record answers in `notes/generation-notes.md`.

## Open teaching threads still to revisit

- **Stage 6:** citation prompt, refusal handling, injection defense at the chunk boundary, system-vs-turn prompt division.
- **Cosmetic TODOs** (above) — knock out on the next CLI touch.
- **Future experiments queued** (in `notes/embedding-notes.md` and `notes/retrieval-notes.md`):
  - Larger BGE (`bge-base-en-v1.5`)
  - Domain-tuned embedder (`voyage-finance-2`)
  - Hybrid retrieval (dense + BM25)
  - Cross-encoder re-rank
  - HyDE
  - Per-section context prefix on chunks before embedding
  - Cross-company round-robin retrieval (Experiment 7 — motivated by Stage 5 Finding 2)

## Curriculum context (do not lose)

This is the Module 02 (RAG) project of the AI engineering curriculum at `~/Projects/ai-engineering-notes/`. Theory phase is done; notes are in `02-rag/`. User values genuine understanding they can defend in an interview, not just a working pipeline. Teach, don't just tell. Be direct.
