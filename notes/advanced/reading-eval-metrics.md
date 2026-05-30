# How to read the retrieval metrics (recall@k & MRR)

**Takeaway:** Recall@k answers *"did we fetch the good chunks?"* (completeness); MRR answers *"was a good chunk near the top?"* (ranking). They can disagree — and *where* they disagree is the whole point. This file explains the arithmetic in plain terms with worked examples from our actual baseline run (`eval/golden.jsonl`, recorded in `eval-notes.md`).

## The two ideas in plain English

- **Recall@k** — *of the chunks that genuinely answer the question (our answer key), what fraction did retrieval put in its top k?* A completeness score.
  `recall@k = (relevant chunks found in top-k) / (total relevant chunks)`
- **MRR (Mean Reciprocal Rank)** — *how high up was the FIRST good chunk?* Rank 1 → 1.0, rank 2 → 1/2, rank 3 → 1/3, … For one question this is just the "reciprocal rank"; **MRR** is the average of it across all questions. It rewards surfacing *a* relevant chunk early.

## The one mental model that makes it all click

There are **two lists**, and we are grading one against the other:

1. **The answer key** — the `relevant_ids` *we* chose during labeling. Human ground truth: the chunks that actually answer the question. This never changes.
2. **The retrieval ranking** — what the retriever returned, ordered by similarity. This is what we're *grading*.

A chunk can rank #1 in the retrieval list and *not* be in the answer key (retrieval thought it was similar; we judged it doesn't answer the question). That mismatch is exactly what the metrics measure. **Relevance is our judgment; the score is how well retrieval's ranking recovers it.**

A few reading rules that come up constantly:
- **recall@5 vs recall@10.** @5 is the window the generator actually sees. @10 is a diagnostic: if a missed chunk shows up by rank 10, it *was* retrieved, just ranked too low — **reranking can fix that**. If it's missing even at @10, retrieval couldn't find it at all — **only better retrieval (hybrid / decomposition) can fix that**.
- **High MRR + low recall = danger.** MRR only looks at the *first* hit. A question can have a perfect MRR (1.0) while recall is 0.5 — see Q13. For "find the one best chunk" questions MRR is the right lens; for "list everything / compare" questions recall is the right lens.
- **hit@5** is the softest metric: did *at least one* relevant chunk make the top 5? It was 1.00 for every question in our baseline — the generator always had *some* grounding, even when completeness was poor.

---

## Worked example 1 — Q3 (semantic): partial recall, first hit at rank 2

**Question:** "How does NVIDIA describe its competitive position?"
**Answer key (relevant_ids):** `0000`, `0005`, `0227` — three chunks.
**Retrieval returned (top 5):**

| Rank | Chunk | In answer key? |
|---|---|---|
| 1 | `0012` (HQ / incorporation boilerplate) | ✗ |
| 2 | `0227` (pioneered accelerated computing) | ✓ |
| 3 | `0011` (gamers choose NVIDIA) | ✗ |
| 4 | `0000` (company overview) | ✓ |
| 5 | `0004` (markets served) | ✗ |
| 6–10 | …none is `0005`… | ✗ |

**recall@5** = found 2 of the 3 key chunks (`0227`, `0000`) in the top 5 → **2/3 = 0.67**. Missed `0005`.
**recall@10** = still 2 of 3 — `0005` isn't in the top 10 either → **0.67 (unchanged)**.
**MRR** = the first key chunk is `0227` at rank 2 → **1/2 = 0.50**.

**What to understand:**
- `recall@5 = 0.67`: the generator got 2 of the 3 chunks it needed — decent, not complete.
- `recall@10 = 0.67` *(still!)*: widening the window didn't recover `0005`. So this is a *"retrieval can't find it"* problem, not a *"ranked slightly too low"* problem → **reranking won't help Q3**; it needs better retrieval (hybrid).
- `MRR = 0.50`: retrieval's very top pick (`0012`) wasn't a keeper, but #2 was — a relevant chunk surfaces quickly, just not at the very top.

---

## Worked examples 2–6 (filled in as we walk each one)

Each is chosen to teach one new wrinkle. Data is from the baseline run.

### Example 2 — Q4 (semantic): when recall@5 and recall@10 DISAGREE

**Question:** "What does Tesla say about employee retention and talent?"
**Answer key:** `0113`, `0115`, `0057`, `0114` — four chunks.
**Retrieval returned (top 6, so we can see past the window):**

| Rank | Chunk | In answer key? |
|---|---|---|
| 1 | `0054` (common EV platform — not about talent) | ✗ |
| 2 | `0114` (managing key risks incl. talent) | ✓ |
| 3 | `0057` (workplace / employee commitment) | ✓ |
| 4 | `0113` (loss of key employees) | ✓ |
| 5 | `0059` (compensation committee) | ✗ |
| 6 | `0115` (competing for talent vs richer rivals) | ✓ ← just outside top 5 |

**recall@5** = 3 of 4 in the top 5 (`0114, 0057, 0113`) → **3/4 = 0.75**. Missed `0115`.
**recall@10** = `0115` sits at rank 6, inside the top 10 → all 4 found → **4/4 = 1.00**.
**MRR** = first key chunk `0114` at rank 2 → **1/2 = 0.50**.

**What to understand — contrast with Q3:**

| | missed chunk | recall@5 | recall@10 | meaning |
|---|---|---|---|---|
| Q3 | `0005` | 0.67 | **0.67** (unchanged) | missed chunk is *nowhere* — retrieval can't find it |
| Q4 | `0115` | 0.75 | **1.00** (jumps) | missed chunk *was* retrieved, just at rank 6 |

The **0.75 → 1.00 jump is the key idea**: `0115` is already in the candidate pool, parked one spot below the window. If a reranker reordered the top 10 and lifted it into the top 5, recall@5 → 1.00. **This is the reranking-fixable case** — reranking reshuffles what's already retrieved; it can't fetch new chunks, so recall@10 is its ceiling. Q4 is one concrete instance of the overall 0.79 → 0.89 reranking headroom.

*Answer-key aside:* rank 5 (`0059`, compensation committee) is arguably talent-related but we didn't label it relevant. A different labeler might have, changing these exact numbers — the inherent subjectivity of a golden set (see `eval-notes.md`). The metrics are only as good as the labels.

### Example 3 — Q12 (exact-term): the hybrid case + low MRR

**Question:** "Does Apple pay a dividend?"
**Answer key:** `0115`, `0116` — two dividend-policy chunks.
**Retrieval returned (top 6):**

| Rank | Chunk | In answer key? |
|---|---|---|
| 1 | `0138` (deemed repatriation **tax**) | ✗ |
| 2 | `0004` (payment services / Apple Card) | ✗ |
| 3 | `0058` (patent infringement notices) | ✗ |
| 4 | `0116` (cash dividend expected to continue) | ✓ |
| 5 | `0019` (employee headcount) | ✗ |
| 6+ | …`0115` absent even at rank 10… | ✗ |

**recall@5** = only `0116` (rank 4) → **1/2 = 0.50**. **recall@10** = `0115` still absent → **0.50 (unchanged)**. **MRR** = first hit at rank 4 → **1/4 = 0.25**.

**What to understand — two lessons:**
1. **Missing even at @10 → not reranking's job** (like Q3). But *why* can't dense find it? The question hinges on the literal word **"dividend."** Dense matches *meaning*, so "does Apple pay a dividend" looked similar to other money/finance text — it ranked a repatriation-**tax** chunk #1 and an Apple Card payments chunk #2. A **keyword search (BM25)** would rank by the literal term "dividend" and surface `0115/0116` directly. **This is the hybrid-search fix** (dense for meaning + sparse for exact terms).
2. **Low MRR has its own meaning.** MRR = 0.25 (first hit at rank 4) says retrieval's *top picks were off-target* — it burned ranks 1–3 on plausible-but-wrong chunks. Compare to Q13 below: same recall (0.50), opposite MRR (1.00) — because MRR only cares about the *first* hit.

### Example 4 — Q13 (cross-company): perfect MRR, poor recall (the big lesson)

**Question:** "How do Tesla and NVIDIA describe their AI investments?" (no company filter)
**Answer key:** TSLA `0000`, `0197`, `0183` + NVDA `0025`, `0012`, `0004` — six chunks, three per company.
**Unfiltered retrieval returned (top 5):**

| Rank | Chunk | In answer key? |
|---|---|---|
| 1 | `NVDA-0025` | ✓ |
| 2 | `NVDA-0012` | ✓ |
| 3 | `NVDA-0004` | ✓ |
| 4 | `NVDA-0227` | ✗ |
| 5 | `NVDA-0000` | ✗ (note: key has *TSLA*-0000, a different chunk) |

Every returned chunk is **NVIDIA**. Zero Tesla chunks — not in top 5, not even in top 10.

**recall@5** = 3 of 6 (the three NVDA key chunks; all three TSLA key chunks missed) → **3/6 = 0.50**.
**recall@10** = still **0.50** — no Tesla chunk appears even at depth 10.
**MRR** = first key chunk `NVDA-0025` at rank 1 → **1/1 = 1.00**.

**What to understand — the headline lesson:**
- **MRR = 1.00 (perfect) while recall@5 = 0.50 (half missing).** MRR is happy because the *first* chunk is relevant; recall exposes that an entire company's worth of answer (all of Tesla) never came back. If you'd glanced at MRR alone ("1.00 — great!") you'd have shipped a half-blind retrieval. **For comparison / enumeration questions, trust recall, not MRR.**
- *Why* it happened: unfiltered dense search collapsed entirely onto NVIDIA, whose AI prose embeds harder against the query. This is Stage 5 Finding 2, now quantified.
- *Which fix:* Tesla chunks aren't in the pool even at @10, so **reranking can't help** (nothing to reorder) and it's not a literal-term issue so **hybrid won't help either**. The fix is **decomposition** — run the query once per company (filtered), then merge — which structurally guarantees both companies get slots.
- *Aside on ids:* `NVDA-0000` ranked #5 but the key's `0000` is `TSLA-0000` — a different chunk that happens to share the `0000` suffix. The eval scores on **full ids** (`TSLA-2026-01-29-0000` ≠ `NVDA-2026-02-25-0000`), so there's no collision. This is exactly why the full chunk id (with ticker + date) is the citation/scoring key, not the bare index.

> **Contrast Q12 vs Q13 — same recall (0.50), opposite MRR:** Q12 has low MRR (0.25, first good chunk buried at rank 4); Q13 has perfect MRR (1.00, first chunk is good but the rest is missing). Two completely different failures that a single metric would have blurred together. This is the entire reason we report recall *and* MRR.

### Example 5 — Q7 (semantic, but really enumeration): the worst case
**Question:** "How does Tesla generate revenue beyond vehicle sales?" · **Key:** `0020, 0012, 0224, 0021` (four revenue lines).
Ranking: `0020`(✓ used-vehicles r1), `0019`(✗), `0223`(✗), `0023`(✗), `0074`(✗), `0224`(✓ r6)… `0012`,`0021` absent even at r10.
**recall@5 = 1/4 = 0.25** (worst) · **recall@10 = 2/4 = 0.50** · **MRR = 1/1 = 1.00**.
**Teaches:** a multi-aspect question (used cars / energy / leasing / services) where dense collapses onto the single most-similar aspect and misses the rest — structurally the *same* failure as cross-company, hiding in the "semantic" bucket. MRR = 1.0 misleads yet again.

### Example 6 — Q16 (negative control): scoring "the answer doesn't exist"
**Question:** "What is the CEO's home address?" · **Key:** *(empty)* — no chunk answers this.
**recall / MRR = undefined** (you can't compute a fraction of zero relevant chunks) → **scored separately**, not averaged in. The meaningful signal is **top-1 sim = 0.5656** (noise floor), which says "the corpus has no good match." 
**Teaches:** not every question has an answer, and the harness must say so honestly rather than fabricate a score. The low similarity *is* the result.

---

## Quick reference

| Metric | Question it answers | Best for | Watch out |
|---|---|---|---|
| hit@k | Did we get *any* relevant chunk in top-k? | "is there grounding at all" | very soft — was 1.00 for every question |
| recall@k (fraction) | What *share* of relevant chunks made top-k? | completeness; comparison/enumeration | the metric to trust when answers have many parts |
| recall@5 vs @10 | Is a miss recoverable by reranking? | diagnosing *which fix* applies | @5 = real window; @10 = reranking headroom |
| MRR | How high was the *first* relevant chunk? | "find the one best chunk" | can be perfect while recall is poor (Q13, Q7) |
