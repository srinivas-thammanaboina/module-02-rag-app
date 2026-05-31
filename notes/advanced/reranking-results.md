# Reranking experiment — results & diagnosis (the instructive failure)

> ⚠️ **READ THE FINAL SECTION FIRST — major parts of this file are SUPERSEDED.** These runs were scored against a golden set later found to be mislabeled (see `eval-audit.md`). Re-running on the repaired eval (bottom section, *"Re-run on the REPAIRED eval"*) overturns two headline conclusions here: (1) the Q12 "genuine model failure" was a *label error* — the reranker was right; (2) the bge "domain-misfit / killer insight (stronger model scores worse ⇒ cosine-biased eval)" is **dead** — bge is simply *malfunctioning* in our harness (it confidently mis-ranks even a clean synthetic pair), not validly diverging from cosine. The instinct in this file — *don't trust a surprising number, look at the data* — was right; the conclusions it reached were not, because the data (labels) was itself broken.

**Takeaway:** Reranking the naive dense results with a cross-encoder (`ms-marco-MiniLM-L-6-v2`, pool 50) **regressed** retrieval on our corpus: overall recall@5 dropped 0.79 → 0.63. Diagnosing *why* split the regression into two distinct causes — one a **genuine model failure**, one a **flaw in our evaluation itself**. This is the most useful result of the advanced stage so far: it's a concrete case of why production RAG "upgrades" backfire, and why eval design is as hard as the system.

Setup (pool size N=50) and the depth sweep that justified it: `reranking-pool-sweep.md`.

## The result — `eval --rerank` (cross-encoder, pool=50)

```
    Q  category        hit@5  rec@5  rec@10   MRR  question
    1  semantic            1   0.40    0.60  1.00  What are the main risks Tesla faces?  ← misses in top-5: 0077,0084,0149
    2  semantic            1   1.00    1.00  1.00  What does Apple say about supply chain concent
    3  semantic            1   0.33    0.67  0.50  How does NVIDIA describe its competitive posit  ← misses: 0000,0005
    4  semantic            1   0.50    1.00  1.00  What does Tesla say about employee retention a  ← misses: 0113,0115
    5  semantic            1   0.67    0.67  1.00  What macroeconomic factors affect Apple's resu  ← misses: 0025
    6  semantic            1   0.67    0.67  0.50  What are NVIDIA's risks around supply and manu  ← misses: 0102
    7  semantic            1   0.50    0.50  1.00  How does Tesla generate revenue beyond vehicle  ← misses: 0012,0021
    8  exact-term          1   1.00    1.00  1.00  What is NVIDIA NIM?
    9  exact-term          1   1.00    1.00  1.00  What does NVIDIA say about CUDA?
   10  exact-term          1   1.00    1.00  1.00  What is Tesla's Supercharger network?
   11  exact-term          1   0.50    1.00  1.00  What does Tesla say about Robotaxi?  ← misses: 0004,0179
   12  exact-term          0   0.00    0.50  0.14  Does Apple pay a dividend?  ← misses: 0115,0116
   13  cross-company       1   0.50    0.50  1.00  How do Tesla and NVIDIA describe their AI inve  ← misses: 0000,0183,0197
   14  cross-company       1   0.50    0.75  1.00  Compare supply chain risk for Apple and Tesla.  ← misses: 0084,0114
   15  cross-company       1   0.50    0.50  1.00  How do Tesla and Apple describe regulatory/leg  ← misses: 0086,0045
   16  control-negative     —      —       —     —  What is the CEO's home address?
   17  semantic            1   1.00    1.00  0.33  What are NVIDIA's gaming segment products?

  overall          hit@5=0.94  recall@5=0.63  recall@10=0.77  MRR=0.84   (n=16)
  cross-company    hit@5=1.00  recall@5=0.50  recall@10=0.58  MRR=1.00   (n=3)
  exact-term       hit@5=0.80  recall@5=0.70  recall@10=0.90  MRR=0.83   (n=5)
  semantic         hit@5=1.00  recall@5=0.63  recall@10=0.76  MRR=0.79   (n=8)
```

### Baseline vs reranked

| | baseline | reranked (minilm) | Δ |
|---|---|---|---|
| overall recall@5 | 0.79 | **0.63** | **−0.16** |
| overall MRR | 0.86 | 0.84 | −0.02 |
| overall hit@5 | 1.00 | 0.94 | −0.06 |
| semantic recall@5 | 0.83 | 0.63 | −0.20 |
| exact-term recall@5 | 0.78 | 0.70 | −0.08 |
| cross-company recall@5 | 0.67 | 0.50 | −0.17 |

Every category dropped. A few precise questions improved (Q9 CUDA 0.67→1.00); broad ones cratered (Q1 1.00→0.40); the dividend question collapsed (Q12 0.50→**0.00**, hit@5=0).

## Diagnosis — look at the actual chunks (Rule 4)

Pulled baseline vs reranked top-5 for one suspected-spurious case (Q1) and one suspected-genuine case (Q12):

```
############ Q1: What are the main risks Tesla faces?  (labeled: 0077,0084,0106,0114,0149)
  --- BASELINE dense top-5 ---   (all 5 are labeled → recall 1.00)
   1. ✓KEY 0084 cos=0.772 | future growth and success dependent upon demand for our [vehicles]
   2. ✓KEY 0114 cos=0.733 | If we are not successful in managing these risks...
   3. ✓KEY 0077 cos=0.720 | If we experience production delays or inaccurately forecast demand...
   4. ✓KEY 0149 cos=0.718 | loss of previously available tax credits and carbon offset...
   5. ✓KEY 0106 cos=0.711 | statements and actions of Tesla and its management...
  --- RERANKED top-5 ---   (only 2 labeled → recall 0.40)
   1. ✓KEY 0114 cos=0.733 rr=1.56  | managing these risks...
   2. ✓KEY 0106 cos=0.711 rr=-0.54 | statements and actions of management...
   3.      0113 cos=0.678 rr=-1.18 | The loss of the services of any of our key employees...
   4.      0126 cos=0.688 rr=-2.71 | unions have filed unfair labor practice charges against us...
   5.      0068 cos=0.654 rr=-2.75 | if our suppliers do not accurately forecast...

############ Q12: Does Apple pay a dividend?  (labeled: 0115,0116)
  --- BASELINE dense top-5 ---   (0116 at rank 4 → recall 0.50)
   1.      0138 cos=0.765 | deemed repatriation tax payable...
   2.      0004 cos=0.675 | payment services, including Apple Card...
   3.      0058 cos=0.667 | notified that it may be infringing...
   4. ✓KEY 0116 cos=0.662 | expectations that its cash dividend will continue...
   5.      0019 cos=0.653 | approximately 166,000 full-time employees...
  --- RERANKED top-5 ---   (0116 pushed OUT → recall 0.00)
   1.      0138 cos=0.765 rr=4.41  | deemed repatriation tax payable...   ← scored WAY up
   2.      0004 cos=0.675 rr=-4.96 | Apple Card...
   3.      0019 cos=0.653 rr=-5.35 | full-time employees...
   4.      0021 cos=0.645 rr=-6.97 | Annual Reports on Form 10-K...
   5.      0126 cos=0.619 rr=-7.06 | ...
```

**Q1 — mostly SPURIOUS.** The three "new" chunks the reranker added (`0113` loss of key employees, `0126` labor-union charges, `0068` supplier forecasting) are **all genuine Tesla risk chunks** — valid answers we simply didn't label. So recall "dropping" to 0.40 is largely our sparse labels failing to credit good chunks, not real damage. (Small real component: it bumped the headline demand-risk `0084` for a niche labor one.)

**Q12 — GENUINELY worse.** The cross-encoder scored the repatriation-**tax** chunk `0138` at **+4.41** (wildly confident) and shoved the real dividend chunk `0116` out of the top 5 entirely. It latched onto the surface word "dividend" appearing in a *tax* context and mis-judged the actual answer — the exact opposite of the literal-term win we hoped for. Real model failure.

## The two lessons

### Lesson 1 (eval engineering) — a limited-response golden set can't fairly grade a *different* retriever

This was the user's doubt at labeling time, now proven: *"how can the eval be right if we only marked a few matching chunks — what about other valid ones we didn't cover?"* Exactly. Our golden set was seeded from the **baseline** retriever and, for broad questions, labels only ~5 of perhaps ~30 valid chunks. So when a new retriever returns *different but valid* chunks, recall drops **spuriously**. The aggregate −0.16 **overstates** the real regression.

Why this matters beyond our toy: **this is a top reason production RAG evals mislead teams.** An offline metric says a change "hurt recall," people revert a genuinely-fine change — because the labels were incomplete, not the retriever. Mitigations in the real world: label *exhaustively* per query (expensive), use *pooled* judgments across multiple systems (the TREC method — judge the union of what several retrievers return), or use an **LLM judge** that scores relevance of *whatever* was returned rather than checking against a fixed key (Module 05). Our small fixed-key golden set is fine for *precise* questions and unreliable for *broad* ones.

### Lesson 2 (model fit) — "reranking is the highest-ROI upgrade" is corpus- and model-dependent

Even setting label-bias aside and looking only at *precise* questions (where our labels genuinely are the full answer): Q12 0.50→**0.00** and Q3 0.67→0.33 are real regressions; Q9 0.67→1.00 is a real win; Q8/Q10 flat. Net: `ms-marco-MiniLM-L-6-v2` is a **poor domain fit** — it was trained on short web queries → short web passages, not long 10-K prose. The advanced-RAG playbook's "add reranking first, it's the biggest win" did **not** hold here. Measure, don't assume.

## Why this is the real deal

Both failure modes — incomplete eval labels, and a reranker that's the wrong fit for the domain — are exactly the things that silently break real RAG systems in production. Catching them on a 3-filing toy, with the diagnosis written down, is the point of the whole exercise.

## Second model — `bge-reranker-base` (the audit that broke the eval open)

Swapped to `BAAI/bge-reranker-base` (`eval --rerank --reranker bge`) expecting a better domain fit. It was **dramatically worse**:

```
  overall          hit@5=0.44  recall@5=0.17  recall@10=0.36  MRR=0.20   (n=16)
  cross-company    hit@5=0.67  recall@5=0.14  recall@10=0.39  MRR=0.21   (n=3)
  exact-term       hit@5=0.60  recall@5=0.35  recall@10=0.47  MRR=0.33   (n=5)
  semantic         hit@5=0.25  recall@5=0.07  recall@10=0.28  MRR=0.11   (n=8)
```

| | baseline | minilm | bge |
|---|---|---|---|
| overall recall@5 | 0.79 | 0.63 | **0.17** |
| overall MRR | 0.86 | 0.84 | **0.20** |

`recall@5 = 0.17` from a SOTA reranker is **near-random** — a red flag for a usage bug, not a model verdict. We did NOT accept it; we verified.

### Verification chain (don't trust a surprising number)

1. **Model loads & works on clean text.** Isolated sanity pair: the dividend sentence scored `0.539`, irrelevant `0.000`. No "newly initialized" warning → head loaded fine.
2. **Not a batching artifact.** Chunk `0116` scored `0.0014` identically alone, in a 2-pair batch, and in the 50-pair batch. Store text == jsonl text. Deterministic.
3. **So the score is *real* — bge genuinely rates `0116` near-zero for "Does Apple pay a dividend?"** despite `0116` opening with *"expectations that its cash dividend will continue..."* — which scored `0.54` as a clean standalone sentence.

### Why the same text scores 0.54 standalone but 0.0014 in the chunk

The chunk reveals it (`data/chunks/AAPL.jsonl`):
- `0116` **starts mid-sentence** ("expectations that...") — a chunk-boundary cut — and its actual topic is **stock-price *risk*** ("If the Company fails to meet expectations... the price of the Company's stock may decline significantly"). Dividends are one item in a risk list.
- A cross-encoder reads the chunk *holistically* and correctly judges it as a volatility-risk chunk, not a dividend-policy answer. The bi-encoder ranked it #4 purely on the term "dividend."

**bge isn't wrong about `0116` — our label is dubious.**

## What the reranking detour actually exposed (it audited our eval)

1. **A poorly-grounded question.** The corpus has *no* clean "Apple pays a dividend" answer — every "dividend" mention is in a risk or tax frame (`0115`/`0116` risk, `0138` tax). Q12 is closer to a hidden negative control. Its label `{0115,0116}` was charitable term-matching.
2. **A chunking flaw.** `0116` starts mid-sentence and mixes topics — a chunk-quality problem that makes its "relevant" label shaky and confuses a holistic reranker.
3. **Label selection bias** (minilm round): broad questions credit only the cosine-retrieved chunks.

## The killer insight

Why did the **stronger** model (bge, 0.17) score *worse* than the **weaker** one (minilm, 0.63)? Because bge diverges from cosine ranking *more confidently* — and **our golden set is biased toward cosine-retrieved chunks** (seeded from the bi-encoder). So the more a reranker improves on cosine, the more it disagrees with our labels, and the lower it scores. **A stronger reranker scoring lower is evidence the *eval* is biased, not that the model is bad.** This is precisely the trap that makes production teams revert good changes.

## Conclusion → fix the eval before judging patterns

We **cannot trust this eval to adjudicate retrieval patterns yet**. Chasing minilm vs bge vs hybrid on a cosine-biased, partially-mislabeled golden set measures the eval's flaws, not the patterns. Reranking is **parked, not concluded** — we can't fairly say whether it helps until the eval is sound.

**Next: an eval audit** (see SESSION-STATE):
1. Reclassify Q12 (no real answer in corpus) — control or drop.
2. Broad-question labels: expand to the full valid set, or judge precise-only, or move to LLM-as-judge.
3. Note/triage the `0116`-style mid-sentence chunk-boundary issue.
4. Then resume reranking/hybrid/decomposition on an eval we can trust. The strongest fix is an **LLM-as-judge** eval (score whatever is returned, no fixed key) — kills the cosine-seeding bias and bridges to Module 05.

---

# Re-run on the REPAIRED eval (the real verdict)

After the eval audit (`eval-audit.md`: Q12 re-labeled to `0138`, `recall_reliable` flag added, split-denominator aggregate), we re-ran both rerankers against the trustworthy baseline (`recall@5=0.79` n_rel=10, `MRR=0.91` n=16). This section is the authoritative one.

## minilm — predictions confirmed to two decimals; it's a WASH/TRADE, not a regression

**Before running, we predicted the re-run by re-scoring the OLD reranked output against the NEW labels** — because the reranker is deterministic, the reranked *retrieved ids* are identical to the prior session; only the labels + aggregation changed. The prediction held exactly:

| | baseline (repaired) | minilm predicted | minilm **actual** |
|---|---|---|---|
| overall recall@5 (n_rel=10) | 0.79 | ~0.80 | **0.80** |
| semantic (n_rel=3) | 0.75 | 0.83 | **0.83** |
| exact-term (n_rel=4) | 0.92 | 1.00 | **1.00** |
| cross-company (n_rel=3) | 0.67 | 0.50 | **0.50** |
| MRR (n=16) | 0.91 | ~0.89 | **0.90** |

**The methodological point, made concrete:** the cross-encoder's behavior never changed between the "−0.16 regression" session and now — *not one weight moved.* Only the labels did. The entire original "reranking regressed" headline was an artifact of the broken eval. This is the single cleanest demonstration in the project that **eval errors masquerade as model effects.**

**The real finding — minilm is a TRADE:**
- **Wins within-company:** Q7 enumeration `0.25→0.50` (surfaces more revenue streams), Q9 CUDA `0.67→1.00` (promotes the buried `0005`). Exact-term → 1.00.
- **Loses cross-company:** `0.67→0.50`. Mechanism is visible in the data: for Q14 "compare Apple and Tesla supply chain," baseline missed only `0051`; reranked now misses `0084,0114` — **the cross-encoder dropped both Tesla chunks and concentrated on Apple.** A cross-encoder maximizes *per-chunk* relevance, so on a comparison it piles up whichever company scores higher and buries the other. That's not reranking's job — **it's decomposition's.**
- **Opinionated:** reranked `recall@10` = 0.82 < baseline 0.90 — the cross-encoder lifts some relevant chunks into the top-5 while burying others below rank 10. Confidence, for better and worse.

Net: on this corpus minilm reranking does **not** net-improve recall (0.79→0.80, within noise on 10 questions) and slightly lowers MRR (0.91→0.90). The advanced-RAG playbook's "add reranking first, it's the biggest win" **does not hold here** — but for a subtler reason than the original file claimed: it's a genuine wash, not a regression.

## bge — a MALFUNCTION in our harness, not a domain-misfit (the old "killer insight" retracted)

bge re-run on the repaired eval was still catastrophic (`recall@5=0.19`, `MRR=0.20`, `hit@5=0.44`), but the *signature* flagged a bug, not a verdict: it **nailed** the clearest exact-term (Q8 NIM, 1.00) yet **zeroed** easy questions dense+minilm ace (Q2 supply-chain, Q9 CUDA, Q17 gaming all 0.00), and its `recall@10 ≫ recall@5` everywhere (it parks relevant chunks at ranks 6–10 while filling 1–5 with distractors). Domain-misfit degrades *uniformly*; this was confident inversion.

Two read-only diagnostics (`eval/debug_rerank.py`, `eval/debug_bge_isolation.py`) closed it:

**Pool dump (gaming question).** bge pinned ~10 *irrelevant* chunks at sigmoid ≈ 0.9997–0.9998 and ranked the two real gaming chunks (`0019` "GeForce RTX 50… gaming"; `0011` "Gamers choose NVIDIA GPUs") at **#33 and #38** of 50. minilm, same pool, ranked them #3/#4.

**Controlled isolation (raw logits, sigmoid off):**

| pair (query = "NVIDIA's gaming segment products?") | minilm logit | **bge logit** |
|---|---|---|
| synthetic OBVIOUS gaming ("GeForce RTX… gamers buy them") | +5.52 | **−1.62** |
| real 0019 (gaming) | +1.79 | −3.94 |
| real 0011 (gaming) | +1.17 | −3.27 |
| real 0041 (competition: AMD/Huawei/Intel — irrelevant) | −10.63 | **+8.61** |
| synthetic OBVIOUS irrelevant (tax) | −11.31 | −10.20 |

Diagnosis by elimination — bge's failure is **none** of the easy explanations:
- **Not saturation noise** — raw logits span −10 to +8.6 (real, confident judgments; the sigmoid merely *hides* this by crushing everything ≥+8 to ~1.0).
- **Not a batch effect** — isolated logit == batch logit, exactly.
- **Not a flipped sign** — NIM ranks #1; synthetic pair is correctly ordered (gaming > tax). Orientation is right.
- **Not a preview/label trap** — `0041`'s *full* text was read: it's the competitor-list chunk, genuinely irrelevant to "gaming products."
- **It simply fails** — bge rates a clean, unambiguous gaming sentence as *not relevant* (−1.62) while rating the competition chunk +8.6. minilm (identical code path) is correct throughout.

**Verdict:** bge-reranker-base produces **untrustworthy scores in this harness**. Therefore:
- The old `recall@5 = 0.17` is **void** — broken measurement, not a model verdict.
- **The "killer insight" is dead.** "A stronger reranker scoring lower ⇒ our golden set is cosine-biased" rested on (1) the Q12 label error and (2) bge's broken scores. bge was not "a better model diverging from cosine" — it was mis-ranking. The cosine-seeding-bias narrative collapses with it.

**Open (deferred, not needed for the verdict):** the *root cause* of bge's misbehavior — weak checkpoint vs a `CrossEncoder`/bge usage subtlety vs a tokenization quirk — is unpinned. The measurement is untrustworthy regardless; pinning it is optional future work.

## The corrected lessons

1. **Eval errors masquerade as model effects.** The original "−0.16 regression" and "0.17 catastrophe" were *both* eval artifacts (a label error + a broken model harness), surviving an entire diagnosis session because no one (a) re-read the mislabeled chunk's full text or (b) tested the reranker on a controlled pair. Verify the *measurement* before theorizing about the *system*.
2. **"Don't trust a surprising number" cuts both ways.** The original file applied it to defend a *low* number (bge 0.17 "we verified it"). But the verification stopped at "the model loads and scores one clean pair" — it never tested whether the model ranks *our* inputs sanely. A surprising number demands a controlled A/B (here: minilm vs bge on the same synthetic pair), not just a smoke test.
3. **Reranking is corpus-dependent, and on THIS corpus it's a wash.** minilm trades within-company gains for cross-company losses; bge is unusable. The trustworthy eval points the roadmap at **decomposition/round-robin** (cross-company Q13–15 + the Q7 enumeration class) as the real next lever — measured, not assumed.
