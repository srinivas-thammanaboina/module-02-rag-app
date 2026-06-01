# Hybrid retrieval (dense + BM25, fused) — Experiment 8

**Takeaway:** Dense retrieval is *blind* — not just weak — to opaque tokens it never learned a vector for (acronyms, named acts, foreign entities: FDDEI, TSMC, GAIN AI Act, GDPR). No reranker or decomposer can fix that: the chunk isn't in the candidate pool to begin with. A second, lexical retriever (BM25) that scores on literal tokens *finds* those chunks; the engineering is all in **how you fuse the two lists**. The headline findings: the textbook fusion (RRF) was a **wash** (its one-lane cap can't surface a blind-lane answer) — plain round-robin **interleave** won. A df dispatch gate looked like a wash *standalone* but proved **load-bearing in composition** (it keeps BM25 off decomposition's semantic branches). Shipped: `Decomposition(Hybrid(interleave, gated))` — overall recall@5 **0.59 → 0.73**, hit@5 **0.78 → 0.91**, lexical **0.30 → 0.70**, cross-company **0.67 → 0.94**.

> Advanced-stage convention (see `eval-notes.md`): a new capability file `app/hybrid.py` — `HybridRetriever` *wraps* the base `Retriever` and honors the same `.retrieve(question, k, company)` contract, so `eval`/`ask` accept it unchanged. Naive v1 stays untouched. Same pattern as `RerankingRetriever`/`DecompositionRetriever`. Theory companion: `ai-engineering-notes/02-rag/hybrid-retrieval.md`.

## Why we're building this (and what made it measurable)

Until golden-set v2, the eval *couldn't see* a hybrid win — every question was one dense already answered (`hit@5 = 1.00` everywhere, the ceiling problem). v2 added the `lexical` category — 6 opaque-token questions — and the dense baseline finally showed a gap:

- **lexical** recall@5 = **0.30**, hit@5 = **0.33** (n=6) — dense whiffs on 4 of 6 (TSMC, GAIN AI Act, GDPR, FDDEI all hit@5=0).

That 0.30 is the number hybrid has to beat. See `eval-notes.md` "Golden set v2".

## Intuition / mental model

Dense and sparse fail in *opposite* ways, which is exactly why combining them works:
- **Dense (embeddings)** — matches meaning/paraphrase; blind to rare/novel tokens (their vectors are near-noise).
- **Sparse (BM25)** — matches literal tokens; weights a rare term *more* (IDF), so a 2-of-678 token like "FDDEI" rockets its chunk to the top; blind to synonyms.

The win case is a **disagreement**: the answer is the chunk only BM25 can see. That detail turns out to decide the whole fusion design (below).

## Why the naive approach fails — concrete, from real data

`"What does NVIDIA disclose about FDDEI?"` — the answer chunk `0217`:
- Dense ranks it **#87 of 278** (NVDA-filtered) — *blind*, not merely low.
- BM25 (clean query) ranks it **#1**.

Dense can't be fixed by reranking (87 is far outside any pool) or decomposition (no sub-parts). Only a lexical lane finds it.

## Chosen design

`HybridRetriever(base, pool=50, fusion="interleave", gated=False)`:

```
retrieve(question, k, company):
    dense  = base.retrieve(question, k=pool, company=company)     # full NL question
    sparse = bm25.search(bm25_query(question), k=pool, ticker=company)  # stopword-stripped
    fuse(dense, sparse) -> top-k
```

Three sub-decisions, each forced by a measured failure:

### 1. Hand-rolled Okapi BM25 (`BM25Index`)
~40 lines: inverted index (term→postings), `df`/`idf` (smoothed `ln(1 + (N-df+0.5)/(df+0.5))`, always positive), Okapi score with `k1=1.5, b=0.75`. Built once from `data/chunks/*.jsonl`. **No `rank_bm25` dependency** — the TF/IDF/length-norm math *is* the lesson (the faster `pip install` path was offered and declined). Company filter = restrict scoring to chunks whose metadata ticker matches (BM25 has no native `where`).

### 2. Stopword-stripped BM25 query — NECESSARY but not sufficient
First failure: the naive full-sentence BM25 query returned generic chunks. Root cause, from the IDF column: conversational words are *rare in 10-K prose* — `"what"` (idf 4.8), `"disclose"` (idf 5.3), `"does"` (idf 3.7) — so BM25 treats them as high-value tokens and pulls generic chunks up the lane. Fix: strip function words **and reporting verbs** (say/describe/disclose/…) from the **BM25 query only** (dense keeps the full NL question — embeddings use the context). The stopword list is a **language-level** set chosen on linguistic principle, identical dev↔prod, **never derived from the corpus or the eval** (that would leak the held-out judge). After stripping, the answer chunks hit BM25 rank 1–2. But it still wasn't enough on its own — see fusion.

### 3. Fusion: interleave, NOT RRF (the load-bearing finding)
Even with the answer at BM25 rank 1, RRF buried it. **RRF rewards lane-*count* over rank-*position*:**

```
ANSWER (FDDEI):  BM25 #1, dense absent  →  RRF = 1/(60+0)            = 0.0167  (one lane)
generic chunk:   dense #1 + BM25 #5     →  RRF = 1/60 + 1/(60+4)     = 0.0323  (two lanes)
```

The answer earns the *best score a one-lane chunk can* (1/60) and still loses — any two-lane chunk adds a second term on top. Since dense is *structurally blind* to the token, the answer can never become a two-lane chunk, so RRF's one-lane cap buries it at every `k` (lowering `k` doesn't help — the two-lane chunk keeps its extra term). **Round-robin interleave** (dense#1, sparse#1, dense#2, … dedup) gives each lane *guaranteed slots*, so a one-lane answer survives — FDDEI's `0217` lands at position 2. Full reasoning: theory note §3c.

## Design decisions baked into the code

1. **`pool = 50` per lane** (matches reranking's pool; depth sweep showed recall@50=1.00 here).
2. **`fusion` is a knob** (`--fusion {rrf,interleave}`), default `rrf` in code but **interleave is the shipped choice** — RRF is kept as the instructive negative result, not deleted.
3. **Asymmetric query**: dense gets the full question, BM25 gets `bm25_query()` (stripped). Each retriever gets the query form it wants.
4. **Provenance on every result dict**: `fusion`, `rrf_score`, `dense_rank`, `bm25_rank`, `bm25_score`. Sparse-only chunks (dense-blind) carry `similarity = 0.0` — honest "no dense signal".
5. **Optional gate** (`gated`, `--hybrid-gate`): engage BM25 only when the query carries a rare token (df ≤ 1% of corpus); else pass through to pure dense. **Measured a wash** — kept behind the flag, off by default. See below.

## Sanity-check experiment — RESULTS

Baseline to beat (dense, golden v2): overall recall@5 **0.59**, hit@5 **0.78**; lexical **0.30 / 0.33**.

| config | overall rec@5 | overall hit@5 | lexical rec@5 | lexical hit@5 | semantic | cross-co | enum |
|---|---|---|---|---|---|---|---|
| dense | 0.59 | 0.78 | 0.30 | 0.33 | 1.00 | 0.67 | 0.12 |
| **+ RRF** | 0.59 | 0.78 | 0.40 | **0.33** | 0.75 | 0.47 | 0.25 |
| **+ interleave** | 0.68 | 0.91 | **0.70** | **0.83** | 0.75 | 0.58 | 0.25 |
| + interleave, gated | 0.68 | 0.91 | 0.70 | 0.83 | 0.75 | 0.67 | 0.12 |
| compose: decomp(interleave) | 0.69 | 0.91 | 0.70 | 0.83 | 0.75 | 0.64 | 0.25 |
| **compose: decomp(interleave,gated)** ← SHIP | **0.73** | **0.91** | **0.70** | **0.83** | 0.75 | **0.94** | 0.12 |

- **RRF = wash.** Overall 0.59→0.59, hit 0.78→0.78. The tell: lexical hit@5 stayed at dense's **0.33** — it surfaced *zero* new opaque-token answers (the one-lane cap, live). It even hurt cross-company (0.67→0.47).
- **Interleave = the win.** Lexical recall 0.30→0.70, hit 0.33→0.83 (GAIN AI Act / GDPR / Section 232 / FDDEI all went hit 0 → hit 1). Overall hit@5 0.78→**0.91**.
- **Gate = wash.** It only *moved* collateral: recovered cross-company (0.58→0.67) but lost enumeration (0.25→0.12, suppressed Q7) and **failed to fix the semantic dip it was built for** (Q2 false-fired on `concentration(df5)` — a rare *ordinary* word the df-signal can't tell from an identifier). Net aggregate identical to ungated.

### Predictions vs reality (honest log)
- **Wrong (twice):** predicted RRF would lift lexical to ~0.80 — it did ~nothing. Then predicted stopword-cleaning would fix FDDEI — necessary but insufficient (RRF still buried it). Both owned in-thread; the fix (interleave) came from reading the RRF arithmetic on real data.
- **Right:** predicted the gate would be a wash (the df-signal conflates rare-ordinary with opaque) — confirmed exactly.

## Composition with decomposition — the gate flips from wash to load-bearing

Standalone, the gate was a wash (above) and I concluded "ship plain ungated interleave; the gate is redundant with decomposition." **The composition overturned that.** `ask` runs `Decomposition(Hybrid(dense))` — the splitter is **outermost** so each scoped sub-query gets its own BM25 lane (the reverse order runs ONE global BM25 that re-injects the cross-company imbalance decomposition just removed — see hybrid-retrieval theory note / the composition whiteboard).

- **Ungated composition** held lexical (0.70) but cross-company recall@5 *dropped to 0.64* — below dense's 0.67. Cause: each decomposition branch is a full hybrid, so on a **semantic** comparison sub-query ("supply chain risk") the BM25 lane injects noise that displaces the relevant dense chunks (Q14 0.75→0.50). **Hybrid's collateral leaked into decomposition's branches.**
- **Gated composition** fixed it: the gate keeps BM25 *out* of the semantic branches (they pass through to clean dense per company), so decomposition's rebalancing runs unpolluted — **cross-company 0.64 → 0.94** (its standalone best), lexical still 0.70, **overall recall@5 0.69 → 0.73** (the best of the whole arc).

The one cost: **enumeration 0.25 → 0.12** — Q7 has no rare token, so the gate routes it to dense, losing hybrid's incidental help. Acceptable: aspect-enumeration (Q7/Q24) is **decomposition Phase B's** job, unbuilt, not hybrid's — we forfeit nothing hybrid owned.

## Decision: SHIP `Decomposition(Hybrid(dense, fusion="interleave", gated=True))`

The measured best config: overall recall@5 **0.59→0.73**, hit@5 **0.78→0.91**, lexical **0.30→0.70**, cross-company **0.67→0.94**, every category at/near its best at once. Wired into `ask` (generate.py). RRF stays behind its flag as the documented negative; **the gate ships** (default-off in `HybridRetriever`, but on in the `ask` composition). The residual semantic dip (1.00→0.75, n_rel=2, Q2 false-fire) is hit@5-invisible — the generator never loses an answer.

## Lessons to carry forward

- **"Measure before you reach for the bigger tool."** RRF (the celebrated fusion) lost to plain round-robin; the eval, per-category, is the only reason we know.
- **A component's value is context-dependent — measure it in the composition, not just standalone.** The df-gate was a wash *alone* (its cross-company gain cancelled an enumeration loss), so I twice called it redundant. In the stack it's **load-bearing**: the same "keep BM25 off semantic queries" behavior protects decomposition's comparison branches from hybrid's collateral (cross-company 0.64→0.94). Judging it in isolation gave the wrong verdict. This is the sharpest new lesson of the experiment.
- **RRF's blind-spot is structural, not a tuning miss.** It rewards lane-agreement; when one retriever is *blind* to a query class, agreement can't form, and no `k` rescues it. Reach for guaranteed-slot fusion (interleave) for the blind-lane case.
- **Composition order: the query splitter goes OUTERMOST.** So every downstream lane (including sparse) operates on the already-scoped sub-query. `Hybrid(Decomposition)` runs one global BM25 that undoes the split; `Decomposition(Hybrid)` is correct.
- **A query's phrasing decides its BM25-favorability.** Conversational scaffolding is rare in formal corpora → high IDF → misleads BM25. Strip it from the sparse query (only).
- **A cheap df-gate can't cleanly separate "lexical" from "semantic"** (it false-fires on rare ordinary words like `concentration`). Clean separation is essentially NER → LLM query-understanding (theory note §6). But "clean" isn't required here: in the composition the gate only needs to keep BM25 off the *obviously*-semantic comparison branches, which it does well enough to be worth shipping.
- **Never tune the cleaner/gate on the eval.** Stopword list and df-threshold were set on principle; the eval *measured* them, never *designed* them.

## Future experiments queue

- **Q18 TSMC residual** — still hit@5=0 under interleave: `0038` is only BM25 rank 8 because content words ("reliance/manufacture/chips") dilute the one opaque token "tsmc". Future: entity-weighting / tighter keywording, or LLM query-understanding.
- **Compose hybrid + decomposition** — order matters (`Decomposition(Hybrid(dense))` vs `Hybrid(Decomposition(dense))`); whiteboard before shipping to `ask`.
- **LLM query-understanding** — the heavyweight version of stopword-strip + gate, if a static signal ever proves too blunt.
- **bge reranker root-cause** (carried from reranking-results.md) — still open.
