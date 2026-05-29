# Session State — module-02-rag-app

> Paste this file at the start of the next session to resume. Say: *"continue module-02-rag-app — read SESSION-STATE.md"*.

## Where we are

Building a **citation-grounded Q&A copilot over SEC 10-K filings** (per `prompt-instructions.md`). Sequential, stage-by-stage build with a pause for review after each stage.

**Stages 1 → 6 are complete.** Stage 7 (WHY.md + README polish) is next.

## Confirmed decisions (durable)

- **Generation:** Anthropic, model `claude-opus-4-8` (bumped from `4-6` at the start of Stage 6 — stronger on citation-following, clean refusals, and injection resistance; the exact Stage 6 stressors). Note: Opus 4.8 **deprecates the `temperature` parameter** — the API rejects it, so `generate.py` omits it.
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
| 6 | Generate — Anthropic call with citation prompt + hybrid refusal + citation audit | **done** — 5-question run; 0 hallucinated citations; Q4/Q5 refusals; findings in `notes/generation-notes.md` |
| 7 | `WHY.md` + `README.md` | **next** |

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

## Stage 6 result snapshot (five-question end-to-end run, `claude-opus-4-8`)

```
Q1  Tesla risks (TSLA)         top-1 0.7722  ANSWERED  5 chunks cited
Q2  Apple supply chain (AAPL)  top-1 0.6901  ANSWERED  2 chunks cited
Q3  Tesla+NVIDIA AI (nofilter) top-1 0.7625  PARTIAL   NVDA-only (reproduces Finding 2)
Q4  Tesla risks (AAPL filter)  top-1 0.6812  REFUSED   chunks are Apple's, Q asks Tesla
Q5  CEO home address (TSLA)    top-1 0.5656  REFUSED   grey-band prompt path

Hallucinated citations across all 5: 0  (citation audit clean every run)
```

**Design shipped:** hybrid refusal (hard-gate `<0.52` no-API, grey-band `0.52–0.58` prompt-decides, thresholds in `config`), citation audit (regex-extract `[id]`, split known/unknown), injection defense by role discipline (rules in system, chunks fenced + declared inert in user turn). Full rationale + per-question grading in `notes/generation-notes.md`.

**Two findings worth carrying forward (detail in generation-notes.md):**
- **Finding B:** a high similarity score is *not* a license to answer. Q4 refused at 0.68 because the chunks answered a different question (Apple's risks, not Tesla's). The confidence gate guards weak retrieval; grounding-to-the-question guards confidently-wrong-company retrieval.
- **Finding C (queued fix):** the `refused` flag is exact-match, so Q3's "lead with refusal sentence then partially answer" prints as ANSWER with self-contradictory prose. Tighten system rule 3 so the canned refusal line is reserved for *total* refusal; partial answers should answer what's there and state the gap. Deferred — this run is the "before" evidence.

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
├── cli.py                     ← Stages 1–6 wired (ingest / chunk / embed / build·store·inspect / retrieve / ask)
├── notes/                     ← stage-by-stage design notes (moved into folder at end of Stage 5 session)
│   ├── ingest-observation.md  ← Stage 1: 3-iteration regex diagnostic story
│   ├── chunking-notes.md      ← Stage 2: design + Experiment 1/2 + implementation decisions
│   ├── embedding-notes.md     ← Stage 3: intuition + BGE quirks + score-compression lesson
│   ├── store-chroma-notes.md  ← Stage 4: numpy-vs-vectorDB framing + design decisions
│   ├── retrieval-notes.md     ← Stage 5: pressure tests + 5-question results + Experiment 7 queued
│   └── generation-notes.md    ← Stage 6: citation contract + hybrid refusal + injection defense + 5-question run
├── app/
│   ├── __init__.py
│   ├── config.py              ← central configuration (paths, tickers, model names, knobs)
│   ├── ingest.py              ← Stage 1: EDGAR client + section-aware parser
│   ├── chunking.py            ← Stage 2: RecursiveChunker (absorption guard + budget reseed)
│   ├── embed.py               ← Stage 3: Embedder ABC + LocalSentenceTransformerEmbedder
│   ├── store.py               ← Stage 4: VectorStore ABC + ChromaVectorStore
│   ├── retrieve.py            ← Stage 5: Retriever + company-mismatch warning + confidence labels
│   └── generate.py            ← Stage 6: Generator (hybrid refusal + citation audit) + ask CLI
└── data/                      ← gitignored build artifacts
    ├── raw/                   ← cached 10-K HTML for TSLA / AAPL / NVDA
    ├── clean/                 ← parsed section JSON for all three
    ├── chunks/                ← 678-chunk JSONL (TSLA 251, AAPL 149, NVDA 278)
    └── chroma/                ← persisted vector store (8.3 MB on disk)
```

## Carry-forward TODOs (small, deliberately deferred)

1. **`get_sentence_embedding_dimension` FutureWarning** in `app/embed.py:89`. The method was renamed to `get_embedding_dimension` in a recent sentence-transformers release; one-line fix. Cosmetic only — no functional impact. (Still firing — seen again during the Stage 6 run.)
2. **CLI tail-preview cropping** in `app/chunking.py:_print_sample_chunk`. The tail slice doesn't snap to a word boundary, so sample chunks display previews that *appear* to start mid-word. The chunk content is correct; only the display is ugly. Will fix on next CLI touch.
3. **Stage 6 Finding C — refusal-contract refinement.** The exact-match `refused` flag mishandles partial answers (Q3 led with the canned refusal sentence then answered). Tighten system rule 3 in `app/generate.py` so the refusal sentence is reserved for *total* refusal; partial answers should answer what's available and state the gap in their own words. This run is the deliberate "before" evidence.

## Stage 6 — DONE (summary)

Shipped `app/generate.py` (`Generator.answer(question, chunks, top_sim) -> dict`) + the `ask` subcommand. Hybrid refusal gate (hard-gate `<0.52` no-API, grey-band `0.52–0.58` prompt-decides, thresholds in `config.refuse_floor`/`refuse_grey`), citation audit (extract `[id]`, split known/unknown), injection defense by role discipline (rules in system prompt; chunks fenced + declared inert in user turn). Five-question run passed: 0 hallucinated citations, Q4+Q5 refused correctly, Q5 via the grey-band path. Full design + grading: `notes/generation-notes.md`. Two findings (B: high sim ≠ answerable; C: refusal-flag refinement queued) carried into the TODO list above.

## Stage 7 plan (next session)

**What:** `WHY.md` (design-rationale narrative tying the stages together) + a final `README.md` polish pass now that the pipeline is end-to-end complete.

**Anticipated content for `WHY.md`:** the through-line of *why* each stage's key decision was made — structure-aware chunking, the Embedder/VectorStore interfaces, metadata filtering as the cheapest accuracy lever, the two-layer "retrieval reports, prompt acts" confidence design, and the hybrid refusal gate. Pull the "interview-defensible" framings already scattered across the `*-notes.md` files into one narrative.

## What to do at the start of next session

1. Re-read `README.md` (project entry point) and this file.
2. Skim `notes/generation-notes.md` — the Stage 6 findings (esp. Finding C) feed the WHY.md narrative and may prompt the refusal-contract refinement first.
3. Decide: knock out Finding C (refusal-contract refinement) + the two cosmetic TODOs before or after writing WHY.md.
4. Update README build-status table + pipeline diagram to mark Stage 6 done.
5. Write `WHY.md`.

## Open teaching threads still to revisit

- **Stage 6 Finding C** — refusal-contract refinement (queued in TODOs).
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
