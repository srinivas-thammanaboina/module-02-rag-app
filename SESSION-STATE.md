# Session State — module-02-rag-app

> Paste this file at the start of the next session to resume. Say: *"continue module-02-rag-app — read SESSION-STATE.md"*.

## Where we are

Building a **citation-grounded Q&A copilot over SEC 10-K filings** (per `prompt-instructions.md`). Sequential, stage-by-stage build with a pause for review after each stage.

**Stages 1 → 7 are complete.** The pipeline is feature-complete end to end (ingest → chunk → embed → store → retrieve → generate), with `WHY.md` design rationale written and the README polished. Remaining work is optional experiments + small TODOs, not core build.

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
| 7 | `WHY.md` + `README.md` | **done** — cross-cutting design rationale (5 principles + decision log + eval-gap + experiment queue, with woven self-test Q&A); README build-status/diagram/layout updated |

## Advanced stage (in progress — post Stage 7)

The naive 7-stage pipeline is complete. We've entered the **advanced-RAG stage**, run **depth-first and eval-first** (decided with the user): build a measurement harness, then add each advanced pattern as a *measured* experiment rather than a vibe.

**Structure convention (decided):** advanced patterns are added as **new capability files composed behind the existing interfaces** — e.g. a `RerankingRetriever` that *wraps* the base `Retriever`, a `HybridRetriever` that fuses a dense + a BM25 retriever via RRF. The naive v1 modules stay **untouched and runnable** as the baseline. **No `v2` copies of existing files.** Advanced notes live under `notes/advanced/`.

**Sequence:** (1) eval harness [in progress] → (2) reranking → (3) hybrid (BM25+RRF) → (4) decomposition / round-robin (Experiment 7). Each measured against the golden set.

**Eval harness status:** design captured in `notes/advanced/eval-notes.md`. Metrics: **recall@k + MRR**, retrieval-only (not faithfulness — that's Module 05). Baseline run done (recall@5=0.79, MRR=0.86). Plain-terms metric explainer for the user in `notes/advanced/reading-eval-metrics.md` (Q3 worked fully; Q4/Q12/Q13/Q7/Q16 queued as worked examples — currently walking through these one at a time with the user before resuming the build).

### ⏸ RESUME HERE (reranking PARKED; next = eval audit/repair)

**Done so far in the advanced stage:**
- Golden set: 17 Qs labeled → `eval/golden.jsonl`. Eval harness: `app/eval.py` + `cli.py eval` (recall@5 hit+fraction, recall@depth, MRR, per-category, control). Baseline **recall@5=0.79, MRR=0.86**.
- Depth sweep: **recall@50 = 1.00** → problem is *ranking* not retrieval; chose pool N=50. (`reranking-pool-sweep.md`)
- Reranking built: `app/rerank.py` (`Reranker` + `RerankingRetriever`, pool=50), eval flags `--rerank`/`--candidates`/`--reranker {minilm,bge}`.
- **Reranking measured — both models REGRESSED:** minilm 0.79→0.63, bge 0.79→**0.17**. Diagnosed thoroughly (`reranking-results.md`).

**KEY OUTCOME — the eval is not yet trustworthy.** The reranking runs acted as an adversarial audit and exposed: (1) Q12 ("does Apple pay a dividend?") has **no real answer chunk** in the corpus — all "dividend" mentions are risk/tax-framed; its labels were charitable term-matching. (2) Chunk `0116` is **cut mid-sentence** and is topically a stock-volatility-risk chunk — a chunking flaw. (3) **Label selection bias**: golden set was seeded from the cosine bi-encoder, so it's biased toward cosine-retrieved chunks — a stronger reranker that diverges from cosine scores *lower*, which is why bge (better model) scored worse. **A reranker improvement is indistinguishable from eval bias on this golden set.** Reranking is **parked, not concluded.**

**THE VERY NEXT STEP — eval audit/repair (whiteboard-first):**
1. **Reclassify Q12** — no real answer in corpus → make it a control or drop it.
2. **Broad-question labels** (Q1/Q5/Q6 etc.) — expand `relevant_ids` to the full valid set, OR mark them recall-unreliable, OR judge precise-only.
3. **Chunking triage** — note the `0116`-style mid-sentence boundary issue (revisit chunker? or just re-label).
4. **Consider LLM-as-judge eval** — score whatever is returned for relevance (no fixed key). Kills the cosine-seeding bias; bridges to Module 05. Likely the real fix.
5. **THEN** resume reranking/hybrid/decomposition on an eval we can trust.

**Advanced notes:** `eval-notes.md` (harness + findings 1–4), `reading-eval-metrics.md` (metric explainer), `reranking-pool-sweep.md` (depth sweep), `reranking-results.md` (both reranker runs + the eval-audit findings + the "stronger model scores worse = biased eval" insight).

**Note:** curriculum reframe (interview/career → deep-learning focus) is **done + validated clean** across both repos; `06-career/` → `06-ai-native/`.

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
- **Finding C (RESOLVED):** refusal in RAG is three-state (answer / partial / refuse), not binary. Q3 fell through the missing "partial" slot and self-contradicted. Fixed by rewriting system rule 3 into three branches keyed to "does the chunk answer the part asked?". Re-run confirmed Q3 fixed + Q5 not regressed. See `notes/generation-notes.md` Finding C.

## Files on disk now

```
module-02-rag-app/
├── .env.example
├── .env                       ← user-filled (gitignored)
├── .gitignore
├── requirements.txt
├── README.md                  ← project entry point + CLI reference + stage status
├── WHY.md                      ← cross-cutting design rationale (the horizontal view) + self-test Q&A
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
3. ~~Stage 6 Finding C — refusal-contract refinement.~~ **DONE.** Rewrote system rule 3 into three keyed branches (answer / partial / refuse), keyed to "does the chunk answer the part asked?" not "is there related content?". Re-run verified: Q3 now answers NVIDIA + states the Tesla gap in its own words (no canned sentence, 5 cites, audit clean); Q5 stayed a clean refusal (no regression). No code beyond the prompt string. Details in `notes/generation-notes.md` Finding C. **Queued follow-up:** promote `partial` to a first-class return signal (model emits a structured tag) — deferred, not smuggled in.

## Stage 6 — DONE (summary)

Shipped `app/generate.py` (`Generator.answer(question, chunks, top_sim) -> dict`) + the `ask` subcommand. Hybrid refusal gate (hard-gate `<0.52` no-API, grey-band `0.52–0.58` prompt-decides, thresholds in `config.refuse_floor`/`refuse_grey`), citation audit (extract `[id]`, split known/unknown), injection defense by role discipline (rules in system prompt; chunks fenced + declared inert in user turn). Five-question run passed: 0 hallucinated citations, Q4+Q5 refused correctly, Q5 via the grey-band path. Full design + grading: `notes/generation-notes.md`. Two findings (B: high sim ≠ answerable; C: refusal-flag refinement queued) carried into the TODO list above.

## Stage 7 — DONE (summary)

Wrote `WHY.md`: the horizontal design-rationale doc (distinct from the vertical per-stage notes). Five cross-cutting principles (interfaces at swap points; mechanism stays visible; retrieval reports / prompt acts; trust rank, calibrate score per-model; honest about limitations), a "why X not Y" decision-log table, a "trusting quality without a full eval harness" section (eyeballing skills + citation audit + the named eval gap), and the experiment queue framed as a roadmap. Prose-first with one table; **learner self-test Q&A woven after each principle** (concept questions, not interview prep — per the curriculum's learning reframe). README updated: build-status table (Stage 6+7 done), pipeline diagram, repo layout (+WHY.md, +generate.py, +generation-notes.md), CLI reference (`ask` live), and "where to read for depth" (+generation-notes, +WHY.md).

## What to do at the start of next session

The core 7-stage build is complete. Options, all optional:

1. **Experiments** (queued below) — Experiment 7 (round-robin retrieval) is the highest-value next concept; it makes cross-company comparison questions actually answerable.
2. **Finding C follow-up** — promote `partial` to a first-class return signal (model emits a structured tag).
3. **Cosmetic TODOs** — `embed.py` FutureWarning + chunk tail-preview cropping.
4. **Move to Module 03 (agents)** in the curriculum.

Whichever: whiteboard-first, teach scenarios, user runs the tests (see CLAUDE.md working agreement).

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

This is the Module 02 (RAG) project of the AI engineering curriculum at `~/Projects/ai-engineering-notes/`. Theory phase is done; notes are in `02-rag/`. User values deep, first-principles understanding they can reason from, not just a working pipeline. Teach, don't just tell. Be direct.
