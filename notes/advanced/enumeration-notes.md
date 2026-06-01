# Enumeration retrieval (Q7/Q24) — Experiment 9

**Takeaway:** Some questions ask for an *enumeration of implicit aspects* that each live in a different chunk ("revenue beyond vehicle sales" → used cars / energy / leasing / services). Dense runs one similarity competition and collapses onto the single dominant aspect; the rest are starved. It's the last red category (enumeration recall@5 ≈ 0.12). **MMR — the deterministic diversity tool, tried first per the spine — was a measured dead-end** (embedding-geometry spread ≠ the semantic aspects the question wants). **Retrieve-then-expand WON:** ground an LLM in a seed retrieval, let it name the aspects from the evidence, re-query each → enumeration recall@5 **0.12 → 0.50**, hit@5 **0.50 → 1.00**, no aggregate collateral. It is the **first LLM tool in the whole advanced stage to beat its target** — because *grounding* gave it what blind Phase B lacked.

> Advanced-stage convention: new capability composed behind the existing `Retriever` interface; naive v1 untouched. Same as reranking/decomposition/hybrid.

## The problem (recap, one mechanism)

Two questions, identical failure:
- **Q7** "How does Tesla generate revenue *beyond* vehicle sales?" → used cars (0020), energy (0012), leasing (0224), services (0021).
- **Q24** "What end markets does NVIDIA serve, and what does *each* cover?" → Data Center (0013), Gaming (0019), Pro Viz (0020), Automotive (0021).

Dense takes the single most-similar aspect and misses the rest: Q7 = 0.25 (0020 only), Q24 = 0.00 (returns the *overview* chunk, none of the per-market subsections — not even in the top 10). This is eval-notes **Finding 3**. The shipped stack (`Decomposition(Hybrid)`) leaves enumeration at **0.12** — decomposition Phase A splits by *company* (no aspect keywords here), and hybrid needs an *opaque token* (none here). So enumeration is genuinely unaddressed.

## Why MMR first (the spine: measure the simple deterministic tool before the LLM)

MMR (Maximal Marginal Relevance) directly targets "dense fills the window with near-duplicates of one aspect." It re-selects the candidate pool greedily, each pick trading query-relevance against redundancy vs what's already chosen:

```
next pick = argmax  λ·sim(q, c)  −  (1−λ)·max sim(c, s) for s in selected
```

It's deterministic, no LLM, no API cost — exactly the kind of tool the project reaches for *before* the expensive one. The hypothesis: once one used-vehicle chunk is picked, its siblings are penalized, so a different revenue aspect wins the next slot. (`app/mmr.py`, `eval --mmr --mmr-lambda`; needed one plumbing change — `include_embeddings` through `store.query`/`Retriever.retrieve`, since MMR scores chunk-to-chunk cosine.)

λ = 0.7 (relevance-leaning) and pool = 50, both **set on principle, not tuned on the eval**.

## What we observed — MMR fails, and the diagnosis is the real lesson

**It's not a reach problem.** All four Q7 aspects are *in* the pool: used cars (rank 1, sim 0.737), leasing (rank 6, 0.705), services (rank 12, 0.678), energy (rank 36, 0.646). MMR had every aspect available and **chose not to pick them** — it took non-golden chunks (0226/0074/0223/0023), only the rank-1 chunk golden.

**Why:** the four aspects sit in one tight embedding cluster (all "Tesla revenue/auto" prose — pairwise cosines ~0.63–0.71, the same band as the *other* Tesla chunks MMR picked). At λ=0.7 the MMR scores across golden-aspect and ordinary-Tesla chunks are **near-ties**, so the pick is essentially arbitrary w.r.t. the revenue lines. MMR maximizes *embedding spread*, and embedding spread is **uncorrelated with the categories the question asks for**. No λ fixes it: lower λ picks *more* off-topic-but-diverse chunks; higher λ collapses to plain dense.

**Full eval (`eval --mmr`), vs dense baseline (golden v2):**

| | dense | MMR (λ=0.7) |
|---|---|---|
| **enumeration recall@5** | 0.12 | **0.12** (no movement — Q7 0.25, Q24 0.00) |
| overall recall@5 | 0.59 | 0.61 |
| overall recall@10 | 0.69 | **0.65** ↓ |
| semantic | 1.00 | **0.75** ↓ |
| exact-term | 0.92 | 1.00 ↑ (incidental — Q9 CUDA's 0005 surfaced) |
| lexical | 0.30 | 0.40 ↑ (incidental; hybrid owns lexical properly) |

The collateral confirms the mechanism from the other side: **semantic 1.00 → 0.75** and **recall@10 0.69 → 0.65** — MMR's redundancy penalty pushes *genuinely relevant* near-duplicates out (Q2 lost 0036). The exact-term/lexical bumps are real but incidental, not enumeration, and overlap with what hybrid already does better.

## Decision: MMR is a measured dead-end for enumeration

It whiffs the category it was built for (0.12 → 0.12) and adds net collateral. Not shipped; kept behind `--mmr` as the documented negative.

**The lesson (the point of trying it):** enumeration aspects are a **semantic/structural grouping** — "these are Tesla's revenue lines" — not an **embedding-geometry grouping**. MMR only sees geometry, so it is *structurally blind* to the grouping the question wants, and can't recover a grouping it can't perceive. The spine fires again — the simple tool was tried first and measured *no* — and the failure *names the requirement* for the next tool: to enumerate, you must **read the content and name the aspects**.

## Retrieve-then-expand (grounded aspect decomposition) — THE WIN

The tool the diagnosis demands. Pipeline:

```
1. SEED    = base.retrieve(question, ~12, company)          # ground the LLM in real chunks
2. aspects = LLM_extract(question, SEED)                    # grounded; [] if not an enumeration
3. if len(aspects) < 2:  return base.retrieve(question, k, company)    # passthrough
4. per     = { a: base.retrieve(a, k, company) for a in aspects }
5. return round_robin_merge(per, k)                         # guaranteed slots per aspect
```

**Why it should beat blind Phase B:** Phase B asked the LLM to enumerate aspects from the bare question and it returned the question unchanged (it doesn't *know* the segments). Here the LLM reads the **actual retrieved chunks** — for Q24 the overview chunk lists the four markets, so it reads them off and re-queries each subsection (which a single global query never reaches). Grounding is the whole difference.

**Design:** the LLM call does double duty — dispatch *and* extraction (returns `[]` for single-topic → passthrough, so non-enumeration questions are provably untouched). Cached by question, forced tool-use. Reuses `round_robin_merge` (aspects instead of companies — Phase A's structure, LLM-discovered grounded keys). Model: Haiku (grounding does the work; the sweep already showed Opus doesn't beat it at decomposition).

**Result (`eval --expand`, golden v2):**

| | dense | expand (haiku) |
|---|---|---|
| **enumeration recall@5** | 0.12 | **0.50** |
| **enumeration hit@5** | 0.50 | **1.00** |
| enumeration recall@10 | 0.25 | 0.62 |
| overall recall@5 | 0.59 | 0.64 |
| semantic / exact-term / cross-company / lexical | 1.00 / 0.92 / 0.67 / 0.30 | **1.00 / 0.92 / 0.67 / 0.30** (all held) |

- **Q7 → 0.75** (recall@10 = 1.00 — 0012 energy surfaces at depth 10). Grounded aspects were spot-on (energy / used cars / leasing / services / software), curing Phase B's *blind* under-split.
- **Q24 → 0.25, hit@5 1.00** — from a total miss to a hit. Partial because the seed surfaced NVIDIA's *application verticals* (telecom/healthcare/financial) rather than a clean "4 reporting segments" overview, so the LLM named those + missed Professional Visualization. There's a real **labeling question** here ("what end markets does NVIDIA serve" arguably *includes* the verticals → our 4-segment golden may be too narrow) — flagged, not chased.
- **The dispatch held** on every aggregated category — non-enumeration questions returned `[]` and passed through, so semantic/exact-term/cross-company/lexical are byte-identical to dense.

**Caveats (honest):**
- **Over-fires on broad "list the risks" questions** (Q1/Q6) and *narrows* them (recall 1.00→0.80/0.33). They're representative-labeled so hit@5 held and the aggregate is untouched — but the dispatch isn't perfectly clean: "main risks" is *technically* an enumeration, and splitting it retrieves narrower than the broad question.
- **Cost** — unlike Phase A (free) and the hybrid gate (free), expand puts an **LLM call on every query** (cached for the eval; a live `ask` pays one Haiku call + latency per question). This is the tradeoff that governs whether it ships as a default or stays opt-in.

## The lesson — the spine, completed

Expand is the **first LLM tool in the entire advanced stage to beat its target** (reranking-bge, Opus-decomposer, blind Phase B, RRF, the standalone gate, MMR all lost). It won for exactly one reason: **grounding**. Blind Phase B asked the LLM to enumerate from the bare question and got it back unchanged; grounded expand read the actual chunks and named the aspects from evidence. So the spine isn't "LLMs never win" — it's **"the bigger tool earns its keep only when given what it actually needs, and only a trustworthy measurement tells you whether it did."** Here the same model that *failed* blind (Phase B) *won* grounded — the difference was the retrieval pass, not the model.

## Composition — SHIPPED, and an emergent win

`Expand(Decomposition(Hybrid(interleave, gated)))` — expand outermost (its seed + per-aspect retrievals flow through the full stack; it passes through on non-enumeration, preserving the lexical/cross-company wins). Result (`eval --hybrid --fusion interleave --hybrid-gate --decompose --expand`):

| metric | dense | Decomp(Hybrid) | **+ expand (full stack)** |
|---|---|---|---|
| overall recall@5 | 0.59 | 0.73 | **0.84** |
| overall recall@10 | 0.69 | 0.81 | **0.92** |
| **overall hit@5** | 0.78 | 0.91 | **1.00** |
| lexical | 0.30 | 0.70 | **0.90** |
| cross-company | 0.67 | 0.94 | 0.94 |
| enumeration | 0.12 | 0.12 | **0.50** |
| exact-term / semantic | 0.92 / 1.00 | 0.92 / 0.75 | 0.92 / 0.75 |

**hit@5 = 1.00 across all 23 questions** — the generator now gets a relevant chunk for *every* question.

**The emergent win — Q18 TSMC, fixed by expand × hybrid (neither could alone):**
- Hybrid alone failed Q18: the *full* question ("reliance on TSMC to manufacture chips") dilutes "TSMC" with content words, so BM25 ranked the answer (0038) only #8 — out of the window.
- Expand alone failed Q18: its aspect queries ran through *dense*, which is blind to "TSMC."
- Composed, expand emits **focused** aspect queries (cached: `"NVIDIA reliance on TSMC manufacturing"`, `"NVIDIA TSMC supply chain"`, `"NVIDIA foundry partner TSMC"`) where "TSMC" dominates, and routing those through hybrid's BM25 lane ranks 0038 **#1** → guaranteed-slot merge surfaces it. Q18 goes from dead-in-every-prior-config to **1.00**, lifting lexical 0.70 → **0.90**.

This is the **"measure in composition" lesson a third time** (after the gate): two patterns compose *multiplicatively*, fixing a residual that neither addressed standalone. A bench test of either would have missed it.

**Shipped** into `ask` as `Expand(Decomposition(Hybrid(interleave, gated)))` (generate.py). The per-query Haiku call is marginal next to the Opus generation `ask` already makes, and it buys hit@5 0.91→1.00, lexical 0.70→0.90, and the whole enumeration category. RRF, the standalone gate, MMR, and blind Phase B stay behind their flags as the documented negatives that make the spine legible.

**Residual / carried:** Q24 still 0.25 (the end-markets labeling question — is the 4-segment golden too narrow?); expand over-fires on broad "list the risks" questions (Q1/Q6, representative-labeled so hit@5-safe); semantic keeps hybrid's small Q2 dip.
