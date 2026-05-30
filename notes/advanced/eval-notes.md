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

## Golden set (built collaboratively — see labeling session)

_To be filled in during the interactive labeling session. Each entry records the question, its category (semantic / exact-term / cross-company), the agreed `relevant_ids`, and the one-line `why` that justifies the label._

| # | Question | Company | Category | relevant_ids | Why |
|---|---|---|---|---|---|
| _(pending labeling session)_ | | | | | |

## Sanity-check experiment (filled after first run)

_To be filled after building `eval.py` and running the baseline. Records baseline recall@5 and MRR for the naive dense retriever — the number every later pattern must beat._

## Future / what this unlocks

Once the baseline exists, each advanced pattern is a measured experiment, not a vibe:
- **Reranking** (cross-encoder on a wide candidate set) — expect MRR to jump most.
- **Hybrid (dense + BM25, RRF)** — expect the exact-term questions to improve, semantic ones to stay flat.
- **Decomposition / round-robin (Experiment 7)** — expect cross-company balance to improve.
- (Module 05) faithfulness / answer-level eval with an LLM judge — the rung above retrieval eval.

## How to think about eval, generally

The discipline `advanced-rag.md` plants the flag on: *don't trust a RAG improvement you haven't measured.* Pasting in a cross-encoder is easy; knowing whether it helped *your* corpus is the job. A small, honest, stress-designed golden set with two simple metrics is the difference between tuning a system and decorating one.
