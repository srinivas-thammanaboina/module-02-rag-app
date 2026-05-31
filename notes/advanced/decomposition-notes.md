# Decomposition / round-robin retrieval (Experiment 7) — Phase A

**Takeaway:** A single top-k retrieval is *one competition with one scoreboard* — the most-similar sub-topic wins all 5 slots and starves the others. For a multi-part question (compare A and B; enumerate X/Y/Z) that's a structural failure no reranker can fix (a cross-encoder concentrates *harder*). Decomposition splits the one big competition into several small ones — retrieve per part, then merge so every part is *guaranteed* representation. Phase A handles the **cross-company** case deterministically; the aspect-enumeration case (Q7) is Phase B.

> Advanced-stage convention (see `eval-notes.md`): added as a new capability file composed behind the existing `Retriever` interface — a `DecompositionRetriever` that *wraps* the base retriever and honors the same `.retrieve(question, k, company)` contract, so `eval`/`ask` accept it with zero changes. The naive v1 retriever stays untouched. This is the same composition pattern as `RerankingRetriever`.

## Why we're building this (and not more reranking)

The repaired eval (`eval-audit.md`) isolated the two real weak spots, and they're the *same* failure:
- **cross-company** Q13–15 → recall@5 = **0.67** (n_rel=3)
- **Q7** "revenue beyond vehicle sales" → recall@5 = **0.25** — an *enumeration* failure (Finding 3 in `eval-notes.md`)

Both are "dense collapses onto the dominant sub-topic." And the reranking re-run proved this is *unreachable* by reranking: a cross-encoder maximizes per-chunk relevance, so on a comparison it concentrated on the dominant company even harder (Q14/Q15 dropped 0.75→0.50 under minilm). So the eval points here, measured, not assumed.

## Intuition / mental model

Top-k retrieval ranks every chunk on a **single** similarity score and takes the best 5. That's correct for a single-topic question. It breaks when a question has **multiple parts that live on different similarity scales** — one part is simply more similar to the query embedding, wins the slots, and the other parts never appear. The fix is not a better ranker; it's **not making the parts compete in the first place** — give each part its own retrieval and its own guaranteed slots.

## Why the naive approach fails — concrete, from real data

- **Q13** "How do Tesla *and* NVIDIA describe their AI investments?" — unfiltered dense returns **NVDA-heavy** (NVIDIA's AI chunks out-score Tesla's against "AI investments"), so Tesla's `0000/0183/0197` are starved → recall 0.50.
- **Q14 / Q15** — same shape: one company's side fills the window, the other's relevant chunk (`0051` / `0045`) is missed → recall 0.75 each.
- **Q7** — multi-*aspect* (used cars / energy / leasing / services); dense collapses onto the single most-similar aspect (`0020`) → recall 0.25. Same mechanism, no company keywords → **not** solvable by Phase A.

## Chosen design

`DecompositionRetriever(base)` — dispatch then (maybe) split + merge:

```
retrieve(question, k=5, company=None):
    if company is not None:
        return base.retrieve(question, k, company)        # caller already scoped → passthrough
    companies = detect_companies_in_question(question)     # REUSE the Stage-5 primitive
    if len(companies) < 2:
        return base.retrieve(question, k, company=None)    # single-topic → plain (== baseline)
    per = {c: base.retrieve(question, k, company=c) for c in sorted(companies)}
    return round_robin_merge(per, k)                       # interleave by rank
```

```
round_robin_merge(per_company, k):
    lists = [per_company[c] for c in sorted(per_company)]
    merged, seen = [], set()
    for i in range(max_len):                  # TSLA#1, NVDA#1, TSLA#2, NVDA#2, ...
        for lst in lists:
            if i < len(lst) and lst[i]["id"] not in seen:
                merged.append(lst[i]); seen.add(lst[i]["id"])
                if len(merged) == k: return merged
    return merged
```

**Key reuse:** `detect_companies_in_question()` already exists (retrieve.py:65) — built in Stage 5 for the company-mismatch warning, returns the set of tickers named in a question (ticker symbol or company-name match). Phase A is therefore mostly a *merge function + thin wrapper + eval flag*, not new "knows-the-answer" detection code. The detector was independently justified before this experiment existed.

## Design decisions baked into the code (CONFIRMED with user)

1. **Dispatch rule.** Decompose **only** when the call is unfiltered (`company is None`) **and** ≥2 companies are detected in the question. Otherwise passthrough to the exact baseline path. → single-company and semantic questions are *provably unchanged*; the only category that can move is cross-company. This is the safety property that makes it a clean A/B (unlike reranking, which touched everything).
2. **Round-robin merge** (not RRF). Interleave by rank → the window is **balanced by construction**, which is exactly what symmetric "compare A and B" questions need. Maximally visible mechanism. RRF (weights by rank across lists) is deferred as a later refinement if merge quality ever demands it.
3. **Fetch k per company, then merge to k.** With 2 companies and k=5, each filtered list of 5 is plenty to fill 5 balanced slots.
4. **Result dicts pass through untouched** (`id`/`document`/`similarity`/`metadata`), so the eval scores a decomposed list identically — true drop-in. Reported top-1 sim = the first merged chunk's cosine.
5. **New eval flag** `cli.py eval --decompose`, label `decomposed (round-robin)`, parallel to `--rerank`. Pure-Python, deterministic, **no LLM / no API cost**.

## Sanity-check experiment — PREDICTIONS (fill actuals after running)

Run: `python cli.py eval --decompose`. Baseline to beat: `recall@5 = 0.79 (n_rel=10)`, `MRR = 0.91 (n=16)`.

| Q | relevant | baseline rec@5 | **predicted** | reasoning |
|---|---|---|---|---|
| Q13 TSLA+NVDA AI | 3 TSLA + 3 NVDA | 0.50 | **~0.83** (cap) | TSLA gets guaranteed slots; 6 rel / 5 slots caps recall@5 at 0.83 |
| Q14 AAPL+TSLA supply | 2 + 2 | 0.75 | **~1.00** | balancing recovers the missed AAPL `0051` |
| Q15 TSLA+AAPL reg/legal | 2 + 2 | 0.75 | **0.75–1.00** | depends whether TSLA `0045` ranks top-2 *when filtered* — least certain |
| **cross-company** (n_rel=3) | | **0.67** | **~0.83–0.94** | clear win |
| overall (n_rel=10) | | 0.79 | **~0.84–0.88** | cross-company lifts; rest flat |
| semantic (n_rel=3) | | 0.75 | **0.75 (unchanged)** | passthrough — dispatch guarantees it |
| exact-term (n_rel=4) | | 0.92 | **0.92 (unchanged)** | passthrough |
| MRR (n=16) | | 0.91 | **~flat** | cross-company MRR already 1.00 (dominant company ranks #1) |

**Falsification check:** if overall doesn't reach ~0.84+, OR if semantic/exact-term move *at all*, the dispatch is buggy → debug before believing the cross-company number.

## Sanity-check experiment — ACTUALS (`python cli.py eval --decompose`)

```
overall        recall@5=0.88 recall@10=0.95 (n_rel=10)  ·  hit@5=1.00 MRR=0.91 (n=16)
cross-company  recall@5=0.94 recall@10=1.00 (n_rel=3)   ·  hit@5=1.00 MRR=1.00 (n=3)
exact-term     recall@5=0.92 recall@10=1.00 (n_rel=4)   ·  hit@5=1.00 MRR=1.00 (n=5)
semantic       recall@5=0.75 recall@10=0.83 (n_rel=3)   ·  hit@5=1.00 MRR=0.81 (n=8)
control  Q16 top-1 sim=0.5656 (noise floor — expected)
```

| | baseline | predicted | **actual** |
|---|---|---|---|
| Q13 TSLA+NVDA AI | 0.50 | ~0.83 (cap) | **0.83** (misses only `0183`, at the 6/5 cap) |
| Q14 AAPL+TSLA supply | 0.75 | ~1.00 | **1.00** (recovered `0051`) |
| Q15 TSLA+AAPL reg/legal | 0.75 | 0.75–1.00 | **1.00** (recovered `0045`) |
| **cross-company** (n_rel=3) | 0.67 | 0.83–0.94 | **0.94** |
| **overall** (n_rel=10) | 0.79 | 0.84–0.88 | **0.88** |
| semantic (n_rel=3) | 0.75 | unchanged | **0.75** (byte-identical) |
| exact-term (n_rel=4) | 0.92 | unchanged | **0.92** (byte-identical) |
| MRR (n=16) | 0.91 | ~flat | **0.91** |

**Predicted to the decimal, and the falsification check PASSED.** semantic + exact-term are byte-identical to baseline (Q9 still misses `0005`, Q7 still 0.25), proving the dispatch passthrough is clean — decomposition touched *only* cross-company, **zero collateral damage**. This is the first advanced pattern to actually beat the trustworthy baseline (0.79 → 0.88, MRR flat), and it cost a deterministic 30-line merge — where a SOTA cross-encoder (reranking) bought nothing. The eval diagnosed the failure (MRR-high/recall-low = coverage, not ordering), predicted *where* the fix would land (cross-company), and the win landed exactly there.

**Both stated limitations held exactly:** Q7 stayed 0.25 (aspect-split → passthrough → Phase B); Q13 capped at 0.83 (6 rel / 5 slots; its miss `0183` sits at recall@10=1.00 — in the pool, just outside the window, not a flaw).

## Honest limitations (carry forward)

1. **Keyword detection works because our questions name the companies.** Reused primitive or not, "compare the two EV makers" wouldn't trigger it. That gap is Phase B's (LLM decomposition) reason to exist.
2. **Q7 stays at 0.25.** Aspect-split, no tickers → dispatch passthrough → unchanged. Phase B.
3. **Forced balance assumes symmetric questions.** Round-robin gives every company equal slots; a question genuinely weighted toward one company could get a weaker chunk injected. Our compare-questions are symmetric, so it's right here — but the assumption is real (RRF is the nuance later).
4. **Small-corpus caveat persists.** Q13 capped at 0.83 (6 rel, 5 slots) — read its score against 0.83, not 1.0.

---

# Phase B — LLM query decomposition

**Takeaway:** Phase A split on a *known, detectable* axis (company, via keyword match) — deterministic, free, provably safe. Phase B handles questions whose split axis is *semantic and unknown ahead of time* (aspects), so we ask an LLM to find the axis and split on it. It's **query understanding, not retrieval** — and it trades away all three of Phase A's virtues (determinism, zero-cost, the safety guarantee) for generality. Whether that trade pays off is exactly what the eval is for.

## The target (and why Phase A can't touch it)

**Q7** "How does Tesla generate revenue beyond vehicle sales?" — still **0.25**, the worst reliable question. Relevant: 0020 (used cars), 0012 (energy), 0224 (leasing), 0021 (services). Dense collapses onto the single most-similar aspect (0020); Phase A sees one company → passthrough → no help. An *enumeration* failure with no tickers to split on.

## Design — `LLMDecompositionRetriever` (general; a candidate REPLACEMENT for Phase A)

```
retrieve(question, k, company):
    subs = llm_decompose(question)              # LLM → 1..N focused sub-queries (cached)
    if len(subs) <= 1: return base.retrieve(question, k, company)   # atomic → passthrough
    per = {sub: base.retrieve(sub, k, company) for sub in subs}     # text sub-queries, caller's filter
    return round_robin_merge(per, k)            # REUSE Phase A merge; dedup now load-bearing
```

The LLM decomposer is **strictly more general than Phase A**: "Tesla and NVIDIA AI investments" → split per company; Q7 → split per aspect. So Phase B isn't "A + Q7" — it's a candidate replacement, and the real question is *does a general LLM splitter match the cheap deterministic one on cross-company AND fix Q7, at an acceptable cost?*

## Design decisions (CONFIRMED with user)

1. **Decomposer model = Haiku** (`config.decomposer_model`, `claude-haiku-4-5`). Splitting is a cheap, simple task; Opus (the *answer* model) is overkill. Reuses the lazy-Anthropic-client pattern from `generate.py`.
2. **Standalone general decomposer**, measured head-to-head against Phase A — not layered on top of it. The cleaner experiment: does generality beat the deterministic special-case?
3. **Cache decompositions to disk** (`data/decomp_cache.json`, model-keyed) — preserves eval reproducibility (the "predicted to the decimal" discipline), avoids re-billing the same 17 questions every run. Stale on model change → rebuilt.
4. **Pure text sub-queries, NO per-sub-query company filter.** Phase A hard-filtered per ticker; Phase B retrieves on sub-query *text* only. Chosen to **isolate the LLM-decomposition variable** for a clean A-vs-B comparison. Risk: text-steering may not isolate a company as hard as a filter → cross-company could land *below* A's 0.94. Per-sub filtering is a queued refinement, not baked in.
5. **Structured output via tool-use** (`submit_subqueries`), not free-text parsing; **fallback to `[question]`** (= baseline) on any malformed/failed decomposition. Robust failure surface.
6. **No `temperature`** (codebase convention; Opus 4.8 deprecates it, and we don't branch per-model). Determinism comes from the cache, not sampling control.

## The tradeoffs — the whole lesson of Phase B

| | Phase A (deterministic) | Phase B (LLM) |
|---|---|---|
| safety | non-target questions *provably* untouched | atomic questions *should* be untouched — now a RISK to measure, not a guarantee |
| determinism | exact, reproducible | nondeterministic → mitigated by the cache |
| cost | 0 API calls | 1 LLM call per (uncached) question |
| failure surface | none | malformed output, drift, over/under-split → structured output + fallback |
| reach | cross-company only | cross-company *and* aspect (general) |

## Sanity-check experiment — PREDICTIONS (fill actuals after running)

Run: `python cli.py eval --llm-decompose`. Baselines: naive 0.79; Phase A 0.88.

| | baseline | Phase A | **Phase B predicted** |
|---|---|---|---|
| Q7 (aspect) | 0.25 | 0.25 | **~0.75–1.00** (LLM-dependent — less certain than A) |
| cross-company (n_rel=3) | 0.67 | 0.94 | **~0.83–0.94** (only if text sub-queries isolate the company as well as A's filter did) |
| atomic (Q2/8/9/10/12/17) | — | unchanged | **should be ~unchanged — RISK of over-split collateral** |
| overall (n_rel=10) | 0.79 | 0.88 | **~0.88–0.92** (upside from Q7; downside from over-splitting atomics or weak company-steering) |

**How to read the result:** unlike A, B has real upside (Q7) *and* real downside risk. Watch three things: (1) did Q7 jump? (2) did cross-company hold near A's 0.94, or did losing the hard filter cost us? (3) did any atomic question move (over-split collateral)? Also eyeball `data/decomp_cache.json` to see *how* the LLM actually split each question — the sub-queries are the real artifact.

## Sanity-check experiment — ACTUALS (`python cli.py eval --llm-decompose`)

```
overall        recall@5=0.76 recall@10=0.88 (n_rel=10)  ·  hit@5=1.00 MRR=0.91 (n=16)
cross-company  recall@5=0.56 recall@10=0.75 (n_rel=3)   ·  hit@5=1.00 MRR=1.00 (n=3)
exact-term     recall@5=0.92 recall@10=1.00 (n_rel=4)   ·  hit@5=1.00 MRR=1.00 (n=5)
semantic       recall@5=0.75 recall@10=0.83 (n_rel=3)   ·  hit@5=1.00 MRR=0.81 (n=8)
```

**Phase B LOST: `baseline 0.79 → Phase A 0.88 → Phase B 0.76`** — below baseline, 0.12 under the deterministic version, and Q7 never moved. The honest answer to "does generality beat the special-case?" is **no**, and the cache (`data/decomp_cache.json`) shows two distinct causes:

**Failure 1 — Q7 under-decomposition.** Haiku returned Q7 as a SINGLE sub-query ("How does Tesla generate revenue beyond vehicle sales") — it never split it. Decomposing an *implicit* enumeration requires inferring Tesla's specific non-vehicle streams (used cars / energy / leasing / services); the model didn't. The target failure was never attempted → 0.25 unchanged.

**Failure 2 — cross-company degraded (0.67 → 0.56, below baseline).** The LLM split *correctly* by company (e.g. Q15 → "Tesla regulatory/legal risk" + "Apple regulatory/legal risk"), but these are **pure-text sub-queries with no hard filter.** Run unfiltered, "Tesla regulatory risk" doesn't guarantee Tesla chunks, and splitting the 5-slot window across two un-partitioned queries gave fewer slots to worse-partitioned results than the single baseline query. Q15 cratered 0.75 → 0.25.

**Attribution is clean:** the entire 0.79 → 0.76 drop is cross-company degradation. Atomics held (exact-term 0.92, semantic 0.75 — every atomic question returned one sub-query, cache-confirmed). **The over-split collateral risk never materialized; Haiku *under*-split if anything.**

**The lesson that matters:** this isolates what actually made Phase A work — **not the round-robin merge, the hard company filter.** A's `ticker=` filter guaranteed each company's chunks came from a disjoint, correctly-partitioned pool. B kept the merge, dropped the filter, and underperformed even the undecomposed baseline. **Decomposing without a partition guarantee can be worse than not decomposing.** And the curriculum-level lesson: a more powerful/general tool (LLM) is not automatically better — measured head-to-head, it lost to 30 deterministic lines. Only knowable because the eval is trustworthy.

**Verdict: pure LLM decomposition is parked.** The fix is Phase B+ (below) — but Phase A remains the shipped cross-company solution.

### Model sweep — capability is NOT the lever (`--decomposer opus` vs `haiku`)

Swapped the decomposer Haiku → Opus (`--decomposer {haiku,opus}`, splits cached per-model in `data/decomp_cache.json`) to test whether Q7's under-split was a *model-capability* gap.

| | Haiku | Opus | baseline | Phase A |
|---|---|---|---|---|
| overall recall@5 | 0.76 | 0.78 | 0.79 | **0.88** |
| cross-company | 0.56 | 0.64 | 0.67 | **0.94** |
| Q7 | 0.25 | **0.25** | 0.25 | 0.25 |

**A ~10×-pricier model moved the headline +0.02, and still lost to baseline and to the 30-line Phase A.** The cache shows why, and it fully disentangles the two failures:

- **Q7 is not capability-bound.** Opus echoed Q7 unchanged (`["How does Tesla generate revenue beyond vehicle sales?"]`) — exactly like Haiku. Opus surely *knows* Tesla's segments from training, but with the question reading as one coherent ask, no instruction to proactively enumerate, and **no corpus grounding**, the safe move is to echo. → fix is **prompting/grounding** (retrieve-then-expand / feed the aspects), not a bigger model.
- **Cross-company is not capability-bound.** Both models split correctly by company; Opus did marginally better (Q15 0.25→0.50) *only because its sub-query phrasing was cleaner* (Haiku appended noise like "in its 10-K filing"). Still 0.64 < baseline — the missing hard filter is the structural cause, untouched by model choice. → fix is the **mechanism** (per-sub-query filter).

**Stage meta-lesson, third confirmation:** the fancier/pricier component keeps NOT winning the measurement — reranking>dense (no), LLM-decompose>keyword-decompose (no), Opus>Haiku (+0.02, still a loss). The real levers here are cheap: a hard partition filter (mechanism) and grounding/prompting (task design), not raw model power. (Minor side-finding: terse sub-query phrasing beat verbose "…in its 10-K filing" phrasing — sub-query *style* matters a little.)

## Future experiments queue

- **Phase B+ = LLM split + per-sub-query company filter (NOW DATA-JUSTIFIED).** Reuse `detect_companies_in_question()` on each sub-query; if it names exactly one company, apply Phase A's hard `ticker=` filter. Predicted: restores cross-company to ~Phase A's 0.94 (the filter — the load-bearing part — comes back) while keeping LLM generality for aspect-splits. Leaves Q7 unfixed (that's decomposition *quality*, not retrieval).
- **Q7 / implicit-enumeration fix** — Haiku under-split it. Options: a stronger decomposer (Opus) for splitting; few-shot examples of aspect-splits in the prompt; or **retrieve-then-expand** (a first retrieval pass surfaces the aspects, then decompose) — the principled but heavier route.
- **RRF merge** as an alternative to round-robin if/when a question needs rank-weighted (not equal) balance.
- **Apply decomposition inside `ask`** (not just `eval`) so cross-company *answers* improve, not just retrieval — and re-check Stage 6 Finding 2 (the cross-company partial-answer case).

## How to think about this, generally

Reranking and decomposition fix *different* failures: reranking improves **ordering within one competition**; decomposition **changes the competition structure** when the question is secretly several questions. Diagnose which failure you have (our eval did: cross-company/enumeration recall is a *coverage* failure, not an *ordering* one) before reaching for a tool. The eval is what tells them apart — MRR-high-but-recall-low (Finding 1) is the fingerprint of "decomposition, not reranking."
