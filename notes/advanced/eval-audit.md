# Eval audit / repair — making the fixed-key golden set trustworthy again

**Takeaway:** The reranking detour didn't just regress — it acted as an adversarial audit and proved our golden set wasn't trustworthy. This is the repair pass. We chose **fixed-key first** (fix the labels/metric of the existing deterministic eval), with LLM-as-judge deferred as a separate later move. The headline lesson: **when you finally pull the full text of a chunk you'd only ever seen as a one-line preview, the prior diagnosis can invert.** It did.

## The cost ladder (why we fix in this order)

An eval audit restores *ground-truth integrity*. The fixes sit on a cost ladder, and the rungs are not equal:

- **Cheapest / safe:** fix the *question* or the *label* (one line in `golden.jsonl`).
- **Cheap / safe:** change *which metric we trust* for a question class (a reporting decision).
- **Expensive / destructive:** re-chunk the corpus — renumbers every chunk id → invalidates all 52 labels → needs a full re-embed (a "never without asking" op).

Bias: **fix at the label/question level first; treat re-chunking as a last resort that must be justified.** Everything the reranking detour exposed turned out to be fixable at the top of the ladder.

---

## Finding A (the big one) — Q12's label was simply WRONG; the corpus *does* have a clean answer

Prior conclusion (in `eval-notes.md`, `reranking-results.md`, SESSION-STATE): *"Q12 'Does Apple pay a dividend?' has no real answer chunk — every dividend mention is risk/tax-framed; its label was charitable term-matching → make it a negative control."*

**Refuted by the actual chunk text.** We pulled the full text of `0115`, `0116`, `0138` (AAPL.jsonl):

- `0115` / `0116` — section **"General Risks → the price of the Company's stock is subject to volatility."** They mention the cash dividend only as *backdrop to a stock-volatility risk* ("the price of its stock should reflect expectations that its cash dividend will continue…"). Not a dividend answer.
- `0138` — opens with a leftover repatriation-**tax** sentence (a chunk-boundary artifact — this is why its one-line preview reads "deemed repatriation tax payable…" and got it filed as "the tax chunk"), but its **substance is the "Capital Return Program" subsection of MD&A**:
  > "the Company's quarterly cash dividend was **$0.26 per share**. The Company intends to **increase its dividend on an annual basis**… **raised its quarterly dividend from $0.25 to $0.26**… paid **dividends and dividend equivalents of $15.4 billion**."

  That is the textbook, precise, factual answer. **0138 is the answer chunk.**

### The smoking gun in the baseline run

From `reranking-results.md`, baseline dense top-5 for Q12:

```
1.  0138  cos=0.765 | ...quarterly cash dividend was $0.26 per share...   ← THE REAL ANSWER, ranked #1
4. ✓KEY 0116 cos=0.662 | expectations that its cash dividend will continue...   ← what our label credited
```

**Dense retrieval put the correct chunk at rank 1.** Because the label credited `{0115,0116}` and treated `0138` as a distractor, the harness scored a *perfect* retrieval as recall 0.50 / MRR 0.25. And the "reranker regression" on Q12 (minilm/bge promoting `0138` to #1, scored a "genuine model failure" in the prior notes) was the **rerankers being correct** — our label penalized them for finding the real answer.

### The corrected lesson (sharper than the old one)

The previous session reached the right *verdict* — "the eval is the problem, not the model" — for the **wrong reason**. It blamed cosine-seeding bias and a missing corpus answer. The real mechanism for Q12 is plainer and more damning: **a hand-label error.** Someone classified `0138` from its leading preview line without reading the body. No amount of clever metric design fixes a wrong ground-truth label; only *looking at the data* (Rule 4) does. This is the single best argument in the whole project for "iterate against real data, not assumptions."

**Action taken:** Q12 `relevant_ids` → `["AAPL-2025-10-31-0138"]`, category stays `exact-term`, `recall_reliable: true`. The old "best hybrid-search case" framing is retired — dense already nails it at #1.

---

## Finding B — the "0116 mid-sentence cut" was never a chunking bug

`0116` starts mid-sentence ("expectations that its cash dividend…"), repeating the tail of `0115`. The prior notes flagged this as a chunk-quality flaw worth a chunker revisit.

**Quantified it (Rule 4):** **233 / 678 chunks (34.4%) start lowercase / mid-sentence.** That is the **overlap window working as designed** — overlapping chunks necessarily begin mid-sentence about a third of the time; no content is lost (the sentence's head lives in the previous chunk). It's a systemic, intentional property, not a 1-of-678 defect.

**Consequences:**
- **Re-chunking is off the table for this audit.** You can't "fix" a third of the corpus that's behaving correctly, and re-chunking would invalidate every label.
- AAPL `0115`/`0116` are used **only** by Q12. Re-labeling Q12 → `[0138]` **evicts both from the golden set entirely**, so the boundary issue stops touching the eval — exactly the "step 1 likely moots step 3" prediction from the whiteboard.

**Action:** no chunker change. Recorded as a known property. (If a future *generation*-quality issue traces to mid-sentence chunk starts, that's a chunker-v2 experiment on its own, measured separately — not smuggled into the eval audit.)

---

## Finding C — broad-question labels: quarantine the metric, don't fake the denominator (PENDING user confirm)

This one is **not** dissolved by data — it's a real structural limit. For an open question ("What are Tesla's *main* risks?") the relevant set is unbounded (~30 valid risk chunks); we labeled 5. **Fractional recall `|hit ∩ rel| / |rel|` has a fake denominator** — it measures agreement with our arbitrary 5-of-30 subset, not retrieval quality. This is why Q1 "dropped" to 0.40 under reranking: the reranker surfaced *other genuine* risk chunks we never credited.

**Decision: tag each question `recall_reliable` (true = `relevant_ids` is the complete/closed answer set, false = representative sample).** The harness then **averages fractional recall only over `recall_reliable: true` questions**; representative questions are judged on **hit@k + MRR** only (which are valid regardless of set completeness). This doesn't *cure* the incompleteness — it **quarantines** it: we stop reporting a number we can't defend. The real cure for broad questions is the deferred LLM-as-judge.

**Classifying principle (decided with user):** open-ended semantic questions ("main risks", "how does X describe…") are FALSE; bounded facts and *closeable* enumerations are TRUE. When borderline, default FALSE — don't trust a denominator you can't defend; let the future LLM-judge adjudicate the borderline ones (Q3/Q4/Q6).

**CONFIRMED classification (written to `eval/golden.jsonl` as `recall_reliable`):**

| # | Question | reliable | Reason |
|---|---|---|---|
| 1 | main risks Tesla | **FALSE** | ~30 valid risk chunks, labeled 5 — representative |
| 2 | supply chain concentration | TRUE | narrow sub-topic, 2 chunks ≈ complete |
| 3 | competitive position | **FALSE** | open-ended, many facets (judge to adjudicate) |
| 4 | employee retention/talent | **FALSE** | narrow but talent topics are many (judge to adjudicate) |
| 5 | macro factors | **FALSE** | open-ended |
| 6 | supply/manufacturing risks | **FALSE** | "risks around X" is open (judge to adjudicate) |
| 7 | revenue beyond vehicles | TRUE | **closed enumeration** (used cars / energy / leasing / services) |
| 8 | NVIDIA NIM | TRUE | exact factual, single chunk |
| 9 | CUDA | TRUE | exact-term, defining chunks |
| 10 | Supercharger | TRUE | exact-term, 2 chunks |
| 11 | Robotaxi | **FALSE** | diffuse — 19 mentions, picked 4 descriptive |
| 12 | dividend | TRUE | single factual answer (0138) |
| 13 | TSLA+NVDA AI investments | TRUE | cross-co. enumeration (denominator capped: 6 rel / 5 slots → recall@5 ceiling 0.83) |
| 14 | AAPL vs TSLA supply chain | TRUE | cross-co. enumeration |
| 15 | TSLA vs AAPL regulatory/legal | TRUE | cross-co. enumeration |
| 16 | CEO home address | — | control-negative, scored separately (no flag) |
| 17 | NVIDIA gaming products | TRUE | bounded, 2 chunks |

**Net: 10 reliable (fractional recall fair) · 6 representative (hit@k + MRR only) · 1 control.**

### `app/eval.py` change design (decided before coding)

Small + surgical — **no metric math changes** (`recall_at`/`mrr` untouched), only *which rows feed which average*:

1. **Carry the flag** — `evaluate()` reads `recall_reliable = g.get("recall_reliable", True)` and attaches it to each row. Control rows skip (no recall anyway).
2. **Split the denominator in `aggregate()`** — hit@5 and MRR averaged over **all scored** questions; fraction recall@5 / recall@pool averaged over **only `recall_reliable: true`** questions. Report both counts: `n` (hit/MRR) and `n_rel` (recall). A category with zero reliable questions shows recall `—`.
3. **Report honestly** — aggregate line e.g. `recall@5=0.86 (n_rel=10) · hit@5=1.00 MRR=0.88 (n=16)`. Per-question table parenthesizes the recall number for `reliable: false` rows — `(0.40)` — and replaces the `← misses…` annotation (misleading for representative labels) with `(representative labels — recall not aggregated)`. Hit/MRR print normally. The reranking A/B path flows through the same two functions, so it inherits the honest treatment for free.

**Consequence to internalize — THIS RESETS THE BASELINE.** The old `recall@5 = 0.79` averaged in 6 questions whose fractional recall was a fiction; the new recall@5 is over a *different, smaller, honest* set of 10, so **it is not comparable to 0.79**. From here, every advanced pattern is measured against the *new* number, on an eval we can defend.

---

## Repaired baseline — result of `python cli.py eval` on the audited golden set

```
overall        recall@5=0.79 recall@10=0.90 (n_rel=10)  ·  hit@5=1.00 MRR=0.91 (n=16)
cross-company  recall@5=0.67 recall@10=0.83 (n_rel=3)   ·  hit@5=1.00 MRR=1.00 (n=3)
exact-term     recall@5=0.92 recall@10=1.00 (n_rel=4)   ·  hit@5=1.00 MRR=1.00 (n=5)
semantic       recall@5=0.75 recall@10=0.83 (n_rel=3)   ·  hit@5=1.00 MRR=0.81 (n=8)
control  Q16 top-1 sim=0.5656 (noise floor — expected)
```

**This is the new trustworthy baseline every advanced pattern must beat:** `recall@5 = 0.79 (n_rel=10)`, `recall@10 ceiling = 0.90`, `MRR = 0.91 (n=16)`.

### Two numbers worth staring at

1. **recall@5: 0.79 → 0.79 (unchanged headline, changed meaning).** The old 0.79 averaged 16 questions, 6 with fake denominators (Q1/Q5/Q6 inflating at 1.00, Q7-type deflating). The new 0.79 is over only the 10 questions where fractional recall is meaningful. *Same digits, different population.* The point of the audit was never to move the number — it was to make the number defensible. (It's a coincidence they match.)

2. **MRR: 0.86 → 0.91 — 100% attributable to the Q12 fix.** Only Q12's labels changed, so only its MRR changed: first-relevant rank 4 → 1 ⇒ MRR 0.25 → 1.00. `0.86×16 = 13.76; +0.75; 14.51/16 = 0.907 ≈ 0.91`. **One wrong label moved aggregate MRR by ~0.05** — the exact size of effect teams ship/revert changes over. A single bad label was masquerading as a metric.

### What the repaired eval now isolates cleanly (real targets, no label-noise)

- **Q7 = 0.25** (revenue beyond vehicles) — enumeration failure, legitimately reliable + legitimately failing → **decomposition/round-robin**.
- **cross-company = 0.67** (n_rel=3) — weakest category, unchanged → **decomposition**.
- **Q9 `0005`** missed in top-5 but in the pool → **reranking** candidate.

**Honesty caveat:** per-category recall denominators are now tiny (semantic 3, exact-term 4, cross-company 3). Semantic recall (0.75) is basically Q7-dominated `(1.00+0.25+1.00)/3`. Per-category recall is directional only — never a stable rate.

## Status / remaining steps

1. ~~Q12 reclassify~~ → **DONE** (re-labeled to `0138`, Finding A).
2. ~~Chunking triage~~ → **DONE** (no change; Finding B).
3. ~~Broad labels / `recall_reliable`~~ → **DONE** (classification confirmed; `app/eval.py` split-denominator change shipped + unit-tested).
4. ~~Re-run baseline~~ → **DONE** (repaired baseline above).
5. ~~Re-run reranking against THIS baseline~~ → **DONE** (`reranking-results.md` → "Re-run on the REPAIRED eval"). Pool re-confirmed (`recall@50=1.00` holds). **minilm**: predictions confirmed to 2 decimals → a **wash/trade** (0.79→0.80; wins within-company, loses cross-company), NOT the old "regression." **bge**: proven a **harness malfunction** (controlled isolation test: rates a clean synthetic gaming sentence −1.62 while rating an irrelevant competition chunk +8.6) — old 0.17 + "killer insight" both **void**. The eval now points the roadmap at **decomposition**.
6. **NEXT — decomposition / round-robin (Experiment 7).** The trustworthy eval isolates the real targets: cross-company Q13–15 (0.67) and the Q7 enumeration class (0.25). This is the measured next lever.
7. **LATER** — LLM-as-judge (separate session) as the deeper fix for the 6 representative questions; optionally pin bge's root cause.
