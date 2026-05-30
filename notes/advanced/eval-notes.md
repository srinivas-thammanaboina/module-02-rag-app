# Eval notes — retrieval evaluation harness (advanced stage)

**Takeaway:** You cannot honestly claim an advanced-RAG pattern "helped" without a number to beat. This harness turns "retrieval feels better" into recall@k and MRR over a small hand-labeled golden set — the baseline every later pattern (reranking, hybrid, decomposition) is measured against. The metric is the easy part; the golden set is the real engineering.

> This is the first artifact of the **advanced stage**. Convention from here on (decided with the user): advanced patterns are added as **new capability files composed behind the existing interfaces** (e.g. a `RerankingRetriever` that wraps the base `Retriever`), never as `v2` copies of existing modules. The naive v1 pipeline stays pristine and runnable as the baseline this harness measures against. Advanced notes live under `notes/advanced/`.

## Why eyeballing isn't enough (why the naive approach fails)

Stage 5 gave us *eyeballing skills* — score distribution, section coherence, chunk overlap. Those are real, but they're vibes: they can tell you a single retrieval looks healthy, they cannot tell you whether **change X improved retrieval across a representative set of questions**. "I added reranking and it feels better" is not engineering. The moment you have two configs to compare (dense vs dense+rerank), you need a metric computed over a fixed question set, or you're guessing.

## The metrics, and why exactly these two

For each golden question we know the chunk id(s) that actually answer it (`relevant_ids`). Run the question through a retriever, get a ranked list of returned ids, and score:

- **Recall@k (hit rate)** — did *at least one* relevant id land in the top-k? (Or, for multi-answer questions, *what fraction* of `relevant_ids` did we retrieve.) This is what the **generator** needs: the answer present in the context window. It answers *"did we even retrieve the answer?"*
- **MRR (mean reciprocal rank)** — `1 / (rank of the first relevant id)`, averaged over questions. It answers *"how high did the right chunk rank?"* This is the metric that specifically moves when **reranking** works: reranking's whole job is lifting a relevant chunk from, say, rank 7 to rank 2. Recall@5 might not even see that (7 was outside the window); MRR over the full candidate list does.

Recall = *whether*; MRR = *how well-placed*. Together they're enough to evaluate every pattern we'll add. We report both per-question and aggregated.

## The golden set — the actual hard part

A golden set is a list of `question → relevant_ids`. For our 3 filings (~678 chunks) we hand-label ~15–20 questions. Three things make this the real lesson, not clerical work:

1. **Relevance is a set, not a single answer.** "What are Tesla's main risks?" has many valid chunks. So `relevant_ids` is a list, and recall is fractional / hit-based, not exact-match.
2. **The bootstrapping trap (selection bias).** If you find the "correct" chunk *only* by running the current retriever and labeling what it returns, your golden set can only ever contain chunks the current retriever already finds — so it is structurally **incapable of revealing what current retrieval misses**, which is exactly what we want later patterns to fix. Mitigation: seed candidates from retrieval **and** read the relevant filing sections directly (`data/clean/`, `data/chunks/`) to catch misses the retriever wouldn't surface.
3. **Designed to stress the patterns.** The questions are chosen to exercise specific future patterns, so each pattern's win (or non-win) becomes visible:
   - **Semantic / paraphrase** questions — the baseline dense retrieval should handle these.
   - **Exact-term** questions (a literal phrase, a specific dollar figure, an exact section name) — dense retrieval *should* fumble these; **hybrid (BM25)** should later rescue them.
   - **Cross-company** questions — pure top-k can't balance them (retrieval Finding 2); **decomposition / round-robin (Experiment 7)** should fix balance.

## Harness design + tradeoffs

- **`eval/golden.jsonl`** — one JSON object per line: `{ "question", "company" (optional), "relevant_ids": [...], "category", "why" }`. Human-readable, diff-able, append-only.
- **`app/eval.py`** + a **`python cli.py eval`** subcommand — runs each golden question through a retriever, computes recall@k + MRR, prints a per-question table and aggregate.
- **Reads ANY retriever** (the base `Retriever`, or a composed `RerankingRetriever`/`HybridRetriever`). This is the whole point: A/B becomes "run `eval` against config A, then config B, diff the numbers." The harness never hard-codes which retrieval strategy it's scoring.

**Tradeoffs / honest limits (kept deliberately minimal):**
- **Retrieval-only.** We score the *retriever* (recall/MRR), NOT answer faithfulness or LLM-as-judge. That's Module 05 and needs the generator plus a judge model. We measure retrieval because that's precisely what reranking/hybrid/decomposition change.
- **~17 questions → directional, not significant.** A 0.71 → 0.88 shift on 17 questions is suggestive, not a p-value. We will not over-claim; the harness is a decision aid, not proof.
- **Label subjectivity.** "Relevant" is a human judgment. Mitigated by a written criterion per question (the `why` field) and labeling collaboratively.

## Metric definitions & scoring decisions (decided before building)

These four choices determine what the numbers *mean* — locked here so the harness is interpretable, not just runnable.

1. **Recall is reported two ways, because they answer different questions.**
   - **Hit-rate@k** — did *at least one* relevant chunk land in the top-k? Binary per question. This is what the generator minimally needs (any grounding). The "did it work at all" headline.
   - **Fraction recall@k** — `|retrieved ∩ relevant| / |relevant|`. Graded. This is what exposes *partial* coverage — e.g. cross-company Q13 where dense returns 3 of 6 relevant chunks = 0.50. The diagnostic that will show decomposition's win.

2. **Retrieve a configurable depth; headline @5, keep an @10 diagnostic.** The generator's real window is k=5, so **recall@5 + MRR** is the headline. But reranking's whole job is lifting a relevant chunk from rank ~7 into the window — invisible if we only ever look at the top 5. So eval retrieves a configurable depth (default 5 = the real window for the baseline) and also computes **recall@10** as a diagnostic, so near-misses sitting just outside the window are visible. When we A/B a reranker that retrieves wide then narrows, eval scores the final returned list the same way — keeping configs comparable.

3. **MRR = reciprocal rank of the *first* relevant chunk in the returned list, averaged over questions.** `1/rank` of the first hit (rank 1 → 1.0, rank 3 → 0.33), `0` if no relevant chunk is retrieved within the depth. This is the metric that moves most when reranking works: a relevant chunk going from rank 7 → rank 2 shows up as MRR `0 → 0.5`.

4. **The negative control (Q16, empty `relevant_ids`) is scored separately, never averaged in.** Recall/MRR are mathematically undefined (0/0) when there are no relevant chunks. So Q16 is **excluded from the recall/MRR aggregates** and reported on its own as a *confidence-floor* check: its top-1 similarity (≈0.566, grey band) is the signal that the corpus has no good match. The harness must handle empty `relevant_ids` gracefully — skip the metric, report the top-1 sim — not crash or score it 0.

**Plus — per-category aggregates.** Results are broken out by **semantic / exact-term / cross-company** (control reported on its own), because the whole point is seeing *which category* each future pattern improves. A single global number would hide exactly the signal we built the stress-categories to surface.

## Golden set — CONFIRMED

Status: **labels confirmed by the user and written to `eval/golden.jsonl`** (17 questions, 52 relevant_ids, all verified to exist in the corpus). Candidates were surfaced by running retrieval for all 17 questions + grepping the raw chunks for the exact-term literals (the selection-bias guard — catches chunks dense retrieval misses).

ID prefixes (filing dates): TSLA = `TSLA-2026-01-29-####` · AAPL = `AAPL-2025-10-31-####` · NVDA = `NVDA-2026-02-25-####`. Suffixes below.

| # | Question | Company | Category | relevant_ids (suffix) | Dense today | Pattern expected to help |
|---|---|---|---|---|---|---|
| 1 | What are the main risks Tesla faces? | TSLA | semantic | 0084,0114,0077,0149,0106 | all top-5 ✓ | baseline |
| 2 | What does Apple say about supply chain concentration? | AAPL | semantic | 0036,0051 | 0036 only at #5 | reranking |
| 3 | How does NVIDIA describe its competitive position? | NVDA | semantic | 0000,0005,0227 | 0005 missed in top-6 | reranking |
| 4 | What does Tesla say about employee retention and talent? | TSLA | semantic | 0113,0115,0057,0114 | all in top-6 ✓ | baseline |
| 5 | What macroeconomic factors affect Apple's results? | AAPL | semantic | 0026,0025,0106 | top-3 ✓ | baseline |
| 6 | What are NVIDIA's risks around supply and manufacturing? | NVDA | semantic | 0080,0102,0188 | top-5 ✓ | baseline |
| 7 | How does Tesla generate revenue beyond vehicle sales? | TSLA | semantic | 0020,0012,0224,0021 | 0012 (energy) missed | reranking/hybrid |
| 8 | What is NVIDIA NIM? | NVDA | exact-term | 0025 | #1 ✓ | dense already wins |
| 9 | What does NVIDIA say about CUDA? | NVDA | exact-term | 0016,0000,0005 | #1/#2 ✓ | dense already wins |
| 10 | What is Tesla's Supercharger network? | TSLA | exact-term | 0021,0022 | #1/#2 ✓ | dense already wins |
| 11 | What does Tesla say about Robotaxi? | TSLA | exact-term | 0074,0004,0179,0073 | top ✓ (diffuse) | judgment-heavy |
| 12 | Does Apple pay a dividend? | AAPL | exact-term | 0115,0116 | 0115 missed, 0116 #4 | **hybrid (best case)** |
| 13 | How do Tesla and NVIDIA describe their AI investments? | — | cross-company | TSLA:0000,0197,0183 / NVDA:0025,0012,0004 | NVDA-only → recall ~0.5 | **decomposition** |
| 14 | Compare supply chain risk for Apple and Tesla. | — | cross-company | AAPL:0026,0051 / TSLA:0114,0084 | missed AAPL 0051 → ~0.75 | decomposition |
| 15 | How do Tesla and Apple describe regulatory/legal risk? | — | cross-company | TSLA:0161,0045 / AAPL:0090,0086 | missed TSLA 0045 → ~0.75 | decomposition |
| 16 | What is the CEO's home address? | TSLA | control (negative) | *(empty)* | weak top-1 (0.566) | negative control |
| 17 | What are NVIDIA's gaming segment products? | NVDA | semantic | 0019,0011 | #2/#3 ✓ | baseline |

**Key finding from labeling (honest expectation-setting):** this corpus barely shows the classic "dense misses the exact term" failure. NIM/CUDA/Supercharger/Robotaxi all rank #1–#2 under dense, because in a 10-K those terms sit in chunks whose surrounding prose is *also* semantically on-topic. The hybrid win needs an exact identifier in *semantically dissimilar* text (e.g. "Item 1A"), which healthy-company filings lack. **So expect hybrid to show only a modest win here — a real result, not a failure.** The bigger movements will be **reranking** (Q2/Q3/Q7 — right chunk exists but ranks low/missed) and **decomposition** (Q13–15 — structurally ~50–75% recall).

**Judgment calls flagged for user review:** Q11 (Robotaxi is diffuse — 19 chunks mention it; picked the descriptive ones, not passing mentions); Q3 `0005` and Q7 `0012` are chunks dense *misses* — included on purpose as the selection-bias guard, but confirm they're truly "relevant."

## Sanity-check experiment — BASELINE (naive dense retriever)

Run: `python cli.py eval` (retrieval depth 10; headline @5, diagnostic @10). 16 scored questions + 1 negative control.

```
overall        hit@5=1.00  recall@5=0.79  recall@10=0.89  MRR=0.86  (n=16)
semantic       hit@5=1.00  recall@5=0.83  recall@10=0.90  MRR=0.81  (n=8)
exact-term     hit@5=1.00  recall@5=0.78  recall@10=0.90  MRR=0.85  (n=5)
cross-company  hit@5=1.00  recall@5=0.67  recall@10=0.83  MRR=1.00  (n=3)
control  Q16 top-1 sim=0.5656 (noise floor — expected, no good match)
```

**This is the number every advanced pattern must beat: recall@5 = 0.79, MRR = 0.86.**

### Finding 1 — MRR is highest exactly where recall is worst

Cross-company: **MRR = 1.00 but recall@5 = 0.67.** MRR only asks "is the *first* relevant chunk early?"; for a comparison question the dominant company's chunk ranks #1 (MRR perfect) while half the answer is still missing (recall poor). Reporting MRR alone (0.86 overall, "looks great") would have *hidden* the cross-company weakness entirely. Lesson: **metric choice is per-question-type** — MRR for "find the one best chunk", recall for enumeration/comparison. This is the whole justification for reporting both.

### Finding 2 — the eval quantified reranking's ceiling

`recall@5 = 0.79 → recall@10 = 0.89`. Reranking only *reorders the retrieved pool*, so with a depth-10 pool the best recall@5 it could reach is recall@10 = **0.89**. That ceiling is a fact the eval handed us, not a guess. Two consequences:
- To exceed 0.89 you need better *retrieval* (hybrid / decomposition / wider pool), not reranking. When we build reranking we'll retrieve **wide** (e.g. 25) before narrowing, precisely to raise the ceiling — then re-measure recall@25 as the new bound.
- It partitions the misses by which pattern can fix them:

| Misses present in the 6–10 pool (recall@10 = 1.0) → **reranking** | Misses absent even at depth 10 → **hybrid / decomposition** |
|---|---|
| Q4 `0115`, Q9 `0005`, Q11 `0073`, Q14 `0051`, Q15 `0045` | Q3 `0005`, Q7 `0012/0021/0224`, Q12 `0115` |

### Finding 3 — Q7 is an enumeration failure hiding in "semantic"

Q7 "revenue *beyond* vehicle sales" scored **recall@5 = 0.25** (worst question) and recall@10 = 0.50. It's multi-aspect (used cars, energy, leasing, services); dense collapsed onto the single most similar aspect (`0020`, rank 1 → MRR 1.0) and missed the rest even at depth 10. Structurally identical to the cross-company failure — multiple sub-topics, dense takes the dominant one. Decomposition is not only for multi-*company* questions; it applies to any enumeration query.

### Predictions vs. reality

Held: hit@5 perfect everywhere, Q3/Q12 weak as expected, cross-company the worst category, control at the noise floor. Surprise: Q7 far worse than predicted (0.25), revealing the enumeration-failure class above.

### Finding 4 — the problem is RANKING, not retrieval (depth sweep)

Ran `eval --depth {10,25,50}`:

```
recall@5 = 0.79   recall@10 = 0.89   recall@25 = 0.92   recall@50 = 1.00
```

**Every labeled relevant chunk is retrieved within the top 50** (1.00 across all categories). Dense retrieval isn't failing to *find* the chunks — it's failing to *rank* them into the top 5. This **reframes the roadmap and corrects Finding 2's partition**: that partition (reranking owns Q4, hybrid owns Q12, decomposition owns Q13) was true only *for a depth-10 pool*. With a pool of ~50, Q12's `0115` and Q13's Tesla chunks are in the candidate set — so a cross-encoder reranking a wide pool can *reach* them. Reranking's scope is much larger than the depth-10 view implied.

Caveats (so we don't over-claim):
- `recall@50 = 1.0` is partly a **small-corpus artifact** (678 chunks, ~3 relevant/question). Production corpora won't be perfect at depth 50 — there hybrid/decomposition genuinely earn their keep.
- **Q13's recall@5 is capped at 0.83** (6 relevant chunks, 5 slots) — read its reranked score against 0.83, not 1.0.
- **In the pool ≠ promoted.** Whether the cross-encoder ranks a buried chunk into the top 5 against 45+ distractors is empirical — that's the next measurement.

**Decision:** reranking candidate pool **N = 50** (perfect ceiling; trivial cost here). Kept as a knob (`--candidates`) so we can A/B against 25.

## Future / what this unlocks

Once the baseline exists, each advanced pattern is a measured experiment, not a vibe:
- **Reranking** (cross-encoder on a wide candidate set) — expect MRR to jump most.
- **Hybrid (dense + BM25, RRF)** — expect the exact-term questions to improve, semantic ones to stay flat.
- **Decomposition / round-robin (Experiment 7)** — expect cross-company balance to improve.
- (Module 05) faithfulness / answer-level eval with an LLM judge — the rung above retrieval eval.

## How to think about eval, generally

The discipline `advanced-rag.md` plants the flag on: *don't trust a RAG improvement you haven't measured.* Pasting in a cross-encoder is easy; knowing whether it helped *your* corpus is the job. A small, honest, stress-designed golden set with two simple metrics is the difference between tuning a system and decorating one.
