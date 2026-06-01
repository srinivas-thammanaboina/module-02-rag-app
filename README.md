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

# Advanced stage — retrieval eval (recall@k + MRR vs eval/golden.jsonl)
python cli.py eval                                       # baseline (naive dense)
python cli.py eval --depth 50                            # sweep the candidate-pool ceiling
python cli.py eval --rerank --reranker minilm            # A/B: cross-encoder reranking
python cli.py eval --decompose                           # A/B: cross-company round-robin
python cli.py eval --hybrid --fusion interleave          # A/B: dense + BM25 (the win)
python cli.py eval --hybrid --fusion rrf                 # A/B: RRF fusion (the documented wash)
python cli.py eval --mmr                                 # A/B: diversity re-selection (dead-end for enumeration)
python cli.py eval --expand                              # A/B: grounded aspect-decomposition (the enumeration win)
python cli.py eval --hybrid --fusion interleave --hybrid-gate --decompose --expand   # the full shipped stack
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
│   ├── generation-notes.md    ← Stage 6: citation contract + hybrid refusal + injection defense
│   └── advanced/              ← advanced stage: measured retrieval experiments
│       ├── eval-notes.md      ← the eval harness (recall@k + MRR) + golden set design
│       ├── eval-audit.md      ← repairing the eval after reranking exposed its flaws
│       ├── reranking-pool-sweep.md ← candidate-pool depth sweep (ranking vs retrieval problem)
│       ├── reranking-results.md    ← reranking re-judged: wash/trade + the bge harness bug
│       ├── decomposition-notes.md  ← cross-company round-robin — the first pattern to beat baseline
│       ├── hybrid-notes.md         ← dense + BM25: RRF (wash) vs interleave (win), the gate, composition
│       └── enumeration-notes.md    ← MMR (dead-end) vs grounded retrieve-then-expand (win); the full stack

├── eval/
│   ├── golden.jsonl           ← hand-labeled question → relevant-chunk golden set
│   └── debug_*.py             ← read-only reranker diagnostics

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

## Advanced stage — measured retrieval experiments

After the naive pipeline was complete, the project entered an **eval-first** advanced stage: build a measurement harness, then add each advanced-RAG pattern as a *measured* experiment rather than a vibe. Advanced patterns are composed **behind the existing `Retriever` interface** (e.g. a `DecompositionRetriever` that wraps the base retriever) — the v1 pipeline stays pristine as the baseline. Notes live in `notes/advanced/`.

**The arc, one line each — this is the whole story of the stage:**

1. **Built a measurement harness** — recall@k + MRR over a 17-question hand-labeled golden set, so every later change is measured, not guessed. First baseline: recall@5 = 0.79, MRR = 0.86.
2. **Reranking "regressed"** — adding a cross-encoder *lowered* the score. But the regression was the clue, not the conclusion: it acted as an adversarial audit of the eval itself.
3. **Audited and repaired the eval** — found a mislabeled golden answer (the dividend question credited the wrong chunks) and broad questions whose recall denominators were fictional; quarantined unreliable recall behind a per-question flag. Trustworthy baseline: recall@5 = 0.79, MRR = 0.91.
4. **Re-judged reranking on the fixed eval** — `minilm` is a wash/trade (wins within-company, loses cross-company, no net gain); the second cross-encoder (`bge`) was a *broken measurement*, not a bad model. The original "reranking regressed" headline was an eval artifact all along.
5. **Decomposition (round-robin), Phase A — the first real win** — splitting *cross-company* questions by a deterministic keyword match, retrieving per company with a hard filter, and merging round-robin lifted recall@5 **0.79 → 0.88** with MRR flat, exactly where the eval predicted, with zero collateral damage.
6. **Decomposition Phase B (LLM query decomposition) — an instructive loss** — replacing the keyword split with a general LLM splitter *dropped* recall to 0.76 (below baseline). The cache showed why: it under-split the hard enumeration question, and its filterless text sub-queries underperformed the hard metadata filter. Generality lost to 30 deterministic lines, head-to-head.
7. **Diversified the golden set (v2) so a *lexical* win could even be seen** — the v1 set had `hit@5 = 1.00` on every question (dense always found *something*) — structurally blind to a keyword-retrieval win. Added a `lexical` category of opaque-token questions (TSMC, GAIN AI Act, GDPR, FDDEI…). Dense baseline on v2: recall@5 = 0.59, with lexical at **0.30** — the gap finally visible.
8. **Hybrid retrieval (dense + BM25) — the lexical cure, with two twists** — a hand-rolled BM25 lane finds the opaque tokens dense is *blind* to (lexical recall **0.30 → 0.70**). But (a) the textbook fusion **RRF was a wash** — its one-lane cap can't surface a chunk that lives in only the sparse lane; plain **round-robin interleave** won; and (b) a df dispatch **gate looked like a wash standalone but proved *load-bearing in composition*** — it keeps BM25 off decomposition's semantic branches.
9. **Enumeration via retrieve-then-expand — the last category, and the first LLM win** — multi-aspect questions ("revenue beyond X") collapse onto one aspect. **MMR** (deterministic diversity) was a measured dead-end (embedding spread ≠ the semantic aspects). **Grounded** LLM aspect-decomposition won — seed-retrieve, let the LLM name the aspects *from the chunks*, re-query each: enumeration **0.12 → 0.50**, hit@5 **0.50 → 1.00**. The first LLM tool in the stage to beat its target — because the *same model that failed this question blind* (Phase B) *won it grounded*.

The full shipped stack — `Expand(Decomposition(Hybrid(interleave, gated)))` — measures **overall recall@5 0.59 → 0.84, hit@5 0.78 → 1.00** (a relevant chunk for *every* golden question), with an emergent twist: expand's focused aspect queries un-dilute opaque tokens for hybrid's BM25, so **Q18 TSMC — dead in every other config — becomes a hit**, lifting lexical to **0.90**.

**The throughline:** most of the work was making the *measurement* trustworthy, then trusting it over intuition. "The model is bad" repeatedly turned out to be "the eval is wrong" (a mislabel, then a harness bug) — or "the *measurement was on the wrong bench*" (the gate and expand×hybrid, each judged on a bench but load-bearing/multiplicative only once composed). Every time the more powerful/celebrated component was tried *as a drop-in* — a SOTA cross-encoder, a 10× LLM decomposer, RRF fusion, MMR — the eval said *no*; the wins were cheap and deterministic, **except** the one place an LLM finally earned its keep: enumeration, where it won not by raw capability but by being *grounded* in a retrieval pass first.

| Pattern | recall@5 (vs trustworthy baseline 0.79) | verdict |
|---|---|---|
| Reranking — `minilm` cross-encoder | 0.80 | wash/trade — no net gain on this corpus |
| Reranking — `bge` cross-encoder | 0.19 | broken measurement (harness issue), not a model verdict |
| **Decomposition Phase A — cross-company round-robin** | **0.88** | **the winner; cross-company 0.67 → 0.94** |
| Decomposition Phase B — LLM query decomposition | 0.76–0.78 | instructive loss — filterless retrieval; a 10× model (Opus) bought +0.02 |
| Decomposition Phase B+ — LLM split + per-sub-query filter | 0.81 | beat baseline but still < Phase A — the LLM's reworded queries rank worse than the original |

Then the golden set was diversified (v2) to expose the *lexical* gap, and hybrid retrieval measured against it (dense baseline recall@5 = **0.59**, lexical **0.30**):

| Config (golden-set v2) | overall recall@5 | lexical recall@5 | verdict |
|---|---|---|---|
| Hybrid — RRF fusion | 0.59 | 0.40 (hit@5 stuck at 0.33) | **wash** — one-lane cap can't surface a dense-blind answer |
| Hybrid — round-robin interleave | 0.68 | **0.70** | **the win** — guaranteed slots rescue the opaque-token chunk |
| Hybrid — interleave + df gate | 0.68 | 0.70 | a wash *standalone* (gain/loss cancel) |
| Composition — `Decomposition(Hybrid(interleave, gated))` | 0.73 | 0.70 | gate *load-bearing here*; cross-company 0.67 → 0.94 |
| MMR — diversity re-selection | 0.61 | — | **dead-end** for enumeration (0.12 → 0.12); embedding spread ≠ the aspects |
| Expand — grounded aspect-decomposition | 0.64 | — | enumeration **0.12 → 0.50**, hit@5 → 1.00; first *LLM* win (because grounded) |
| **Full stack — `Expand(Decomposition(Hybrid(interleave, gated)))`** | **0.84** | **0.90** | **shipped; hit@5 = 1.00**; expand×hybrid emergently fix Q18 TSMC |

**The spine of the advanced stage** — every time the more powerful, more "obviously better" tool was tried *as a drop-in*, the trustworthy eval said **no**:

- reranking (cross-encoder) > dense → **no** (a wash)
- bge (SOTA reranker) > minilm → **no** (a broken measurement, not a better model)
- Opus > Haiku as the decomposer → **no** (+0.02 for ~10× the cost)
- LLM decomposition > a deterministic keyword split → **no** (lost even when handed the same filter)
- RRF > plain round-robin interleave (hybrid fusion) → **no** (surfaced none of the dense-blind answers)
- MMR (diversity) for enumeration → **no** (geometry ≠ the semantic aspects; a measured dead-end)

…with **one telling exception**: enumeration, where *grounded* LLM aspect-decomposition won — and the same Haiku that *failed it blind* (Phase B) *won it grounded*. The bigger tool earned its keep not by raw capability but by being given what it needed.

The changes that moved retrieval were cheap and deterministic — **repairing the eval's labels**, a **round-robin merge** (cross-company *and* hybrid fusion), a **BM25 lane** for opaque tokens — plus that one grounded-LLM win. Three lessons, earned rather than asserted: **(1)** don't reach for the bigger tool until a trustworthy measurement says the simpler one isn't enough — and when a result surprises you, suspect the measurement before the model; **(2)** a component's value is **context-dependent** — the dispatch gate was a wash alone but load-bearing composed, and expand×hybrid fixed a residual neither could alone, so judge a part *in the stack it will run in*; **(3)** when an LLM *does* win, it's usually because it was **grounded** in retrieval, not because it's bigger.

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
| `notes/advanced/eval-notes.md` | how to turn "retrieval feels better" into recall@k + MRR, and why the golden set is the real engineering |
| `notes/advanced/eval-audit.md` | how a "regression" exposed a broken eval, and the discipline of repairing labels before trusting numbers (the stage's core lesson) |
| `notes/advanced/reranking-results.md` | why a cross-encoder was a wash here, how a SOTA model turned out to be a broken measurement, and how to tell those apart |
| `notes/advanced/decomposition-notes.md` | why cross-company questions break naive top-k structurally, and how round-robin decomposition fixes it (the first real win) |
| `notes/advanced/hybrid-notes.md` | why dense is *blind* to opaque tokens, why RRF fusion can't rescue them but round-robin interleave can, and how the dispatch gate flips from wash to load-bearing in composition |
| `notes/advanced/enumeration-notes.md` | why multi-aspect questions break dense, why MMR's diversity is the wrong tool, and how *grounded* LLM aspect-decomposition wins where blind decomposition failed (the first LLM win) |

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
