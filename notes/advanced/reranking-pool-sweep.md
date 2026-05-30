# Reranking pool-size experiment (the depth sweep)

**Takeaway:** Reranking can only reorder the chunks it was handed, so the candidate-pool size **N** sets a hard ceiling on what reranking can achieve: `recall@N`. Rather than guess N=25 (the textbook default), we *measured* the ceiling by sweeping the retrieval depth. The result reframed the whole problem — on this corpus retrieval is a **ranking** problem, not a **finding** problem — and told us to use **N=50**.

## The question

When we wrap the base retriever in a cross-encoder reranker, how many candidates should we fetch before re-scoring? `advanced-rag.md` suggests ~25.

## Why we did NOT just start with 25

A reranker re-scores and re-sorts the candidate pool — it **cannot fetch a chunk that wasn't retrieved**. So whatever pool size N we pick, the best recall@5 the reranker could ever reach is `recall@N` (if a relevant chunk isn't in the top N, no reordering brings it back). Picking N=25 blind would have two problems:

1. It silently **caps** our achievable recall at `recall@25`, without us knowing whether relevant chunks sit *beyond* rank 25.
2. We'd never learn the more important thing: **is our problem that retrieval can't *find* the right chunks, or that it can't *rank* them?** Those need completely different fixes (better retrieval vs. better ranking).

So: measure before optimizing. Generalized the eval's diagnostic column into `recall@depth` (added a `--depth` flag) and swept the depth to read the ceiling curve directly.

## The experiment

`python cli.py eval --depth {10, 25, 50}` against the 16-question golden set (naive dense retriever; control Q16 excluded from aggregates).

## Results

### Ceiling curve (aggregate recall over the retrieved pool)

| pool depth | overall | semantic | exact-term | cross-company |
|---|---|---|---|---|
| @5 (the window, for reference) | 0.79 | 0.83 | 0.78 | 0.67 |
| @10 | 0.89 | 0.90 | 0.90 | 0.83 |
| @25 | 0.92 | 0.93 | 0.90 | 0.94 |
| **@50** | **1.00** | **1.00** | **1.00** | **1.00** |

### Full run — `eval --depth 25` (pasted)

```
    Q  category        hit@5  rec@5  rec@25   MRR  question
    1  semantic            1   1.00    1.00  1.00  What are the main risks Tesla faces?
    2  semantic            1   1.00    1.00  1.00  What does Apple say about supply chain concent
    3  semantic            1   0.67    0.67  0.50  How does NVIDIA describe its competitive posit  ← misses in top-5: 0005
    4  semantic            1   0.75    1.00  0.50  What does Tesla say about employee retention a  ← misses in top-5: 0115
    5  semantic            1   1.00    1.00  1.00  What macroeconomic factors affect Apple's resu
    6  semantic            1   1.00    1.00  1.00  What are NVIDIA's risks around supply and manu
    7  semantic            1   0.25    0.75  1.00  How does Tesla generate revenue beyond vehicle  ← misses in top-5: 0012,0021,0224
    8  exact-term          1   1.00    1.00  1.00  What is NVIDIA NIM?
    9  exact-term          1   0.67    1.00  1.00  What does NVIDIA say about CUDA?  ← misses in top-5: 0005
   10  exact-term          1   1.00    1.00  1.00  What is Tesla's Supercharger network?
   11  exact-term          1   0.75    1.00  1.00  What does Tesla say about Robotaxi?  ← misses in top-5: 0073
   12  exact-term          1   0.50    0.50  0.25  Does Apple pay a dividend?  ← misses in top-5: 0115
   13  cross-company       1   0.50    0.83  1.00  How do Tesla and NVIDIA describe their AI inve  ← misses in top-5: 0000,0183,0197
   14  cross-company       1   0.75    1.00  1.00  Compare supply chain risk for Apple and Tesla.  ← misses in top-5: 0051
   15  cross-company       1   0.75    1.00  1.00  How do Tesla and Apple describe regulatory/leg  ← misses in top-5: 0045
   16  control-negative     —      —       —     —  What is the CEO's home address?
   17  semantic            1   1.00    1.00  0.50  What are NVIDIA's gaming segment products?

  overall          hit@5=1.00  recall@5=0.79  recall@25=0.92  MRR=0.86   (n=16)
  cross-company    hit@5=1.00  recall@5=0.67  recall@25=0.94  MRR=1.00   (n=3)
  exact-term       hit@5=1.00  recall@5=0.78  recall@25=0.90  MRR=0.85   (n=5)
  semantic         hit@5=1.00  recall@5=0.83  recall@25=0.93  MRR=0.81   (n=8)
  Q16 top-1 sim=0.5656  (noise floor — expected)
```

### Full run — `eval --depth 50` (pasted)

```
    Q  category        hit@5  rec@5  rec@50   MRR  question
    1  semantic            1   1.00    1.00  1.00  What are the main risks Tesla faces?
    2  semantic            1   1.00    1.00  1.00  What does Apple say about supply chain concent
    3  semantic            1   0.67    1.00  0.50  How does NVIDIA describe its competitive posit  ← misses in top-5: 0005
    4  semantic            1   0.75    1.00  0.50  What does Tesla say about employee retention a  ← misses in top-5: 0115
    5  semantic            1   1.00    1.00  1.00  What macroeconomic factors affect Apple's resu
    6  semantic            1   1.00    1.00  1.00  What are NVIDIA's risks around supply and manu
    7  semantic            1   0.25    1.00  1.00  How does Tesla generate revenue beyond vehicle  ← misses in top-5: 0012,0021,0224
    8  exact-term          1   1.00    1.00  1.00  What is NVIDIA NIM?
    9  exact-term          1   0.67    1.00  1.00  What does NVIDIA say about CUDA?  ← misses in top-5: 0005
   10  exact-term          1   1.00    1.00  1.00  What is Tesla's Supercharger network?
   11  exact-term          1   0.75    1.00  1.00  What does Tesla say about Robotaxi?  ← misses in top-5: 0073
   12  exact-term          1   0.50    1.00  0.25  Does Apple pay a dividend?  ← misses in top-5: 0115
   13  cross-company       1   0.50    1.00  1.00  How do Tesla and NVIDIA describe their AI inve  ← misses in top-5: 0000,0183,0197
   14  cross-company       1   0.75    1.00  1.00  Compare supply chain risk for Apple and Tesla.  ← misses in top-5: 0051
   15  cross-company       1   0.75    1.00  1.00  How do Tesla and Apple describe regulatory/leg  ← misses in top-5: 0045
   16  control-negative     —      —       —     —  What is the CEO's home address?
   17  semantic            1   1.00    1.00  0.50  What are NVIDIA's gaming segment products?

  overall          hit@5=1.00  recall@5=0.79  recall@50=1.00  MRR=0.86   (n=16)
  cross-company    hit@5=1.00  recall@5=0.67  recall@50=1.00  MRR=1.00   (n=3)
  exact-term       hit@5=1.00  recall@5=0.78  recall@50=1.00  MRR=0.85   (n=5)
  semantic         hit@5=1.00  recall@5=0.83  recall@50=1.00  MRR=0.81   (n=8)
  Q16 top-1 sim=0.5656  (noise floor — expected)
```

## My notes on these results

**1. The headline: this is a ranking problem, not a retrieval problem.** `recall@50 = 1.00` across every category — all 52 labeled relevant chunks are retrieved within the top 50. Dense retrieval *can* find everything; it just *ranks* it badly (recall@5 = 0.79). That single fact reframes the roadmap: if the chunks are all reachable, then a reranker with a wide-enough pool is the dominant lever.

**2. Where the buried chunks surface (tracking the misses across depths):**
- `0005` (Q3, Q9): not in top 25, appears by 50 — very deep.
- `0115` (Q12, dividend): not in top 25, appears by 50 — the literal-term chunk is buried deep, as predicted for an exact-term/dense mismatch.
- Q13 Tesla chunks: partially surface by 25 (recall 0.50 → 0.83), fully by 50.
- Q7 enumeration chunks: 0.25 → 0.75 (depth 25) → 1.00 (depth 50).

**3. This corrected an earlier claim.** With only the depth-10 numbers we'd partitioned the misses as "reranking owns Q4, hybrid owns Q12, decomposition owns Q13 — reranking can't reach those." The sweep shows that was true only *for a depth-10 pool*. At depth 50 those chunks are in the candidate set, so a cross-encoder reranker can at least *attempt* them. Reranking's scope is much larger than the depth-10 view implied.

**4. Why N = 50 (not 25):**
- `recall@25 = 0.92` but `recall@50 = 1.00`. Choosing 25 would leave `0005` and `0115` permanently out of reach for the reranker (they live beyond rank 25). 50 gives the cross-encoder a perfect ceiling — every relevant chunk available.
- Cost is trivial here: a cross-encoder over 50 chunks × 16 questions is nothing on a 678-chunk corpus.
- **Tradeoff acknowledged:** a bigger pool means more distractors the cross-encoder must rank *below* the relevant chunk. 50 = perfect ceiling but harder precision job; 25 = easier job but capped at 0.92. We chose 50 and kept `--candidates` as a knob to A/B 25 vs 50 empirically.

**5. Honest caveats (don't over-generalize):**
- `recall@50 = 1.0` is partly a **small-corpus artifact** — 678 chunks, ~3 relevant per question. A production corpus (millions of chunks) will *not* be perfect even at depth 50/100; there, hybrid and decomposition genuinely earn their keep. On this project, reranking has unusually large headroom.
- **Q13 is capped at recall@5 = 0.83** regardless of reranking — 6 relevant chunks competing for 5 slots. Read its reranked score against 0.83, not 1.0.
- **In the pool ≠ promoted.** A chunk being within the top 50 only means the reranker *can* reach it; whether the cross-encoder actually scores it into the top 5 (against 45+ distractors) is the empirical question the `--rerank` A/B answers next.

## What this experiment unlocked

The reranking build uses `candidate_pool = 50` (`app/rerank.py`). The next step — `eval --rerank` — measures how much of the 0.79 → 1.00 headroom the cross-encoder actually captures, and crucially whether it rescues the cases (Q12 dividend, Q13 cross-company) we'd previously assumed needed hybrid/decomposition.
