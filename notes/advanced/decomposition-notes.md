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

## Future experiments queue

- **Phase B — LLM query decomposition** for the aspect-enumeration case (Q7) and the general "entities not named" case. Adds an LLM call + latency + nondeterminism; measure whether the generality is worth it.
- **RRF merge** as an alternative to round-robin if/when a question needs rank-weighted (not equal) balance.
- **Apply decomposition inside `ask`** (not just `eval`) so cross-company *answers* improve, not just retrieval — and re-check Stage 6 Finding 2 (the cross-company partial-answer case).

## How to think about this, generally

Reranking and decomposition fix *different* failures: reranking improves **ordering within one competition**; decomposition **changes the competition structure** when the question is secretly several questions. Diagnose which failure you have (our eval did: cross-company/enumeration recall is a *coverage* failure, not an *ordering* one) before reaching for a tool. The eval is what tells them apart — MRR-high-but-recall-low (Finding 1) is the fingerprint of "decomposition, not reranking."
