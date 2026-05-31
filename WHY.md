# WHY — the design rationale behind module-02-rag-app

**Takeaway:** This document explains *why* the pipeline is shaped the way it is. The per-stage `notes/*.md` files go deep on one stage each (how it works, what we learned building it). This one is horizontal — the handful of design ideas that cut across stages and the reasoning that ties them together. If a point only matters to one stage, it lives in that stage's notes and is linked from here, not repeated.

It's written to be *reasoned from*, not memorized. Each principle ends with a couple of **self-test questions** — if you can answer them from scratch, you understand the choice; if you can only recognize the answer, you don't yet.

---

## The system in one paragraph

Given a question about a company's SEC 10-K, the system: cleans the filing into sections (**ingest**), splits each section into ~1000-character structure-aware pieces with stable ids (**chunk**), turns each piece into a 384-dim vector with a local embedding model (**embed**), stores text+vector+metadata in a persistent Chroma collection (**store**), finds the top-k pieces most similar to the question with an optional company filter (**retrieve**), and asks Claude to answer *only* from those pieces, citing each claim and refusing when the evidence isn't there (**generate**). Six stages, one module and one CLI subcommand each, an artifact handed from one to the next.

---

## The cross-cutting principles

These are the ideas that show up in more than one stage. They're the real "architecture" of the project — more than the file layout is.

### 1. Interfaces at every swap point

The two components most likely to be replaced in a real system — the embedding model and the vector store — sit behind abstract base classes (`Embedder`, `VectorStore`). Each has one concrete implementation today (`LocalSentenceTransformerEmbedder`, `ChromaVectorStore`), but the contract is what every downstream stage depends on, not the implementation.

The payoff is concrete: swapping to an OpenAI embedder or a Qdrant backend is *one new file plus a config flag*, not a refactor that ripples through `retrieve.py`, `store.py`, and the CLI. In any real RAG system you *will* A/B-test embedders and you *might* change stores; the seam is nearly free to build at the start and expensive to retrofit later. Interfaces are placed exactly where change is predictable — and nowhere else (see Principle 2).

*Check yourself:* Why put the seam at the embedder and the store, but **not** behind, say, the chunker? — If you swapped `bge-small` for an OpenAI embedder, which files change and which don't, and how does the `Embedder` contract guarantee that?

### 2. Mechanism stays visible

There are no clever abstractions hiding the math. The chunker is readable top to bottom; the retriever is ~10 lines of real logic. This is a deliberate cost — a "framework-shaped" version would be shorter — paid for a specific return: **if you can't read every step of what happens to your data, you can't debug it.** Most failed RAG systems in the wild are failed *retrieval* systems wrapped in elaborate abstractions that make the failure invisible. Keeping mechanism visible is what lets "the top result is wrong" decompose into "the embedding, the chunk, or the question — which one?"

This is in tension with Principle 1, and the tension is the point: interface where change is *predictable* (embedder, store); inline everything else so the data flow stays legible. Abstractions are a debt you take on only when you know what you're buying.

*Check yourself:* Principles 1 and 2 pull in opposite directions — when does an interface earn its keep and when is it just indirection? — Why is a 10-line retriever a feature rather than something to "clean up"?

### 3. Retrieval reports, the prompt acts (the two-layer design)

The confidence signal is the clearest example of separating *measurement* from *decision*. Stage 5 (retrieve) never refuses and never gates — it only **labels** the top-1 similarity with a confidence band. Stage 6 (generate) owns the action: hard-refuse below 0.52 (no API call), let the model judge the 0.52–0.58 grey band, answer above it. The measure and the decision live in different layers on purpose.

A single-layer `if score < X: return None` buried in the retriever would be brittle and almost impossible to reason about: it couples "how confident is retrieval?" to "what should we do about it?", and those change for different reasons. Splitting them means Stage 5 stays a pure mechanism you can inspect, and Stage 6's *policy* (hard gate vs. grey band vs. answer) can evolve without touching retrieval. The hybrid refusal gate — deterministic on garbage, model-judgment on borderline — is only expressible *because* the layers are split.

*Check yourself:* Why not have the retriever drop low-confidence chunks itself and save an API call? — What specifically becomes hard if "measure" and "decide" live in the same function?

### 4. Trust the rank, calibrate the score per-model

Embeddings turn meaning into geometry, and retrieval trusts that geometry to *rank* correctly — the top-k by cosine similarity. What it must **not** trust is the *absolute* score. Our model, `bge-small-en-v1.5`, was trained with a contrastive objective that separates positives from negatives but doesn't push unrelated pairs toward zero — so two random English sentences sit around cosine 0.5, and the model's useful range is ~0.45–0.90, not 0–1. The "0.7 means relevant" heuristic from a tutorial is true only for the model in that tutorial.

This is why every threshold in the system (the 0.52/0.58 refusal bands, the confidence labels) carries a comment saying it's calibrated to BGE specifically, and why the numbers live in `config`. Swap the embedder and the *rank* behavior still works, but every absolute threshold must be re-derived against the new model's noise floor — or the refusal gate silently misfires. Rank is portable; absolute score is not.

*Check yourself:* Top-k retrieval works fine on a model whose "unrelated" baseline is 0.5 — why? — If you swapped to OpenAI embeddings (unrelated ≈ 0.1) and kept the 0.52 refusal floor, what would break, and in which direction?

### 5. Honest about limitations

Real failure modes are documented as observations, not papered over, because a limitation you can name is one you can design around. Three that matter across stages:

- **Pure top-k can't answer cross-company comparisons** (retrieval Finding 2). "How do Tesla and NVIDIA describe their AI investments?" returns 5 NVIDIA chunks and 0 Tesla, because NVIDIA's AI prose embeds harder against the query. No prompt can fix this — the generator can only ground in what it's shown. The structural cure is per-company round-robin retrieval (Experiment 7) *upstream* of generation — **now built and default-on in `ask`**, so that question returns a balanced 2-Tesla/3-NVIDIA retrieval and a full comparative answer (advanced stage; `notes/advanced/decomposition-notes.md`).
- **Score compression** (Principle 4) means absolute thresholds are a calibration problem, never a guessable constant.
- **Refusal is three-state, not binary** (generation Finding C). Answer / partial / refuse — keyed to "do the chunks answer the part the user asked?", not "is there any related content?" A question can retrieve high-similarity chunks and still warrant a refusal (asking for Tesla's risks but filtered to Apple), and a topically-near retrieval with the specific fact absent (the CEO's address) must refuse rather than partial-answer.

The discipline underneath all three: **debug retrieval first, prompts last.** If retrieval is good, prompting is easy; if retrieval is poor, no prompt rescues it.

*Check yourself:* Why does removing the `--company` filter on a "compare X and Y" question produce a *single*-company answer instead of a balanced one? — Give one question that retrieves with high similarity but should still be refused, and say why.

### 6. Measure before you reach for the bigger tool

The advanced stage (post-Stage-7) added the thing this project deliberately lacked: a retrieval eval harness — recall@k + MRR over a hand-labeled golden set (`notes/advanced/eval-notes.md`). Its first job was a baseline; its real job turned out to be a bullshit detector. Four times, the intuitive upgrade — the more powerful, more expensive, more "obviously better" component — was tried, and four times the measurement said **no**:

- **reranking (cross-encoder) > dense** → no: a wash/trade on this corpus, not the playbook's "biggest win."
- **bge (SOTA reranker) > minilm** → no: bge was a *broken measurement* in our harness, not a better model — caught only by a controlled raw-logit isolation test, not by trusting the aggregate.
- **Opus > Haiku (as the query decomposer)** → no: +0.02 for ~10× the cost; it echoed the hard question unchanged, same as Haiku.
- **LLM query decomposition > a deterministic keyword split** → no: the LLM lost *even after* we handed it the same hard filter, because its *reworded* sub-queries ranked the right chunks worse than the original question did.

The two changes that actually moved retrieval were cheap: **repairing the eval's labels** (a single mislabeled golden answer was inflating a "regression" and masquerading as a ~0.05 metric shift) and **~30 deterministic lines** (per-company round-robin, Experiment 7). Twice over, "the model is bad" turned out to be "the eval is wrong" — a label error, then a harness bug. The discipline is the inverse of the usual instinct: **don't reach for the more powerful tool until a trustworthy measurement says the simpler one isn't enough — and when a result surprises you, suspect the measurement before the model.** Full arc in `notes/advanced/` (`eval-audit.md`, `reranking-results.md`, `decomposition-notes.md`).

*Check yourself:* We saw a SOTA reranker score near-random and a 10×-pricier decomposer add only +0.02 — in each case, what's the *first* thing to check, the model or the measurement, and why? — Phase A (keyword split + filter) beat an LLM decomposer that was handed the *same* filter; what was the LLM's reworded query costing us that the original question didn't?

---

## Decision log — the "why X, not Y" choices

| Decision | Chosen | Over | Why |
|---|---|---|---|
| Embedding model | local `bge-small-en-v1.5` | OpenAI / API embeddings | Runs on CPU in seconds for ~700 chunks, no API cost or key, MIT-licensed, benchmarks well. The interface (Principle 1) makes an API swap one file if we outgrow it. |
| Vector store | Chroma (persistent) | hand-rolled numpy / FAISS | Metadata filtering + upsert-by-id + persistence for free. Numpy would mean hand-maintaining an id→row index and a "stale embeddings mixed with fresh" bug class. Earns its keep mostly via **metadata filtering**. |
| Chunk size | ~1000 chars, structure-aware, 150 overlap | fixed-size splits | Respects paragraph/sentence boundaries so chunks are coherent semantic units; overlap preserves context across boundaries. Coherent chunks are a precondition for the embedding bet (Principle 4). |
| Retrieval filter | filter **before** similarity scoring | post-filter top-k | HNSW filters natively: scoring runs over the filtered subset only. Quality up *and* cost down — the cheapest accuracy lever in the pipeline, no tradeoff. |
| top-k | 5 | 3 / 10 | 3 lets one bad chunk poison the context; 10 makes the generator's context noisy. 5 keeps inspection manageable for a learning build. |
| Generation model | Claude Opus 4.8 | Opus 4.6 | Stronger on the exact Stage 6 stressors — citation-following, clean refusals, injection resistance. Note: 4.8 **deprecates the `temperature` parameter** (the API rejects it), so `generate.py` omits it. |

---

## Trusting quality without a full eval harness

This is a learning build, so quality is checked by *eyeballing skills* and *built-in invariants* rather than a metrics harness — but the gap is named, not hidden.

The eyeballing skills (from retrieval-notes): **score distribution** (a confident retrieval has a high top-1 and a falling tail; a flat tail near the noise floor means the corpus probably lacks the answer), **section coherence** (a focused question should pull mostly from one section), and **chunk overlap** (four adjacent chunks in the top-5 means over-retrieval). These sit between "no evaluation" and "a full eval harness."

The built-in invariants are stronger because they're mechanical: the **citation audit** extracts every `[id]` from the answer and checks it against the ids actually retrieved — an empty `unknown_citations` every run is the *evidence* the citation contract held, not an assumption. The **three-state refusal** (Principle 5) is the other invariant: the model is structurally pushed to ground, cite, or refuse, with the hard gate as a deterministic backstop below 0.52.

What a real eval harness would add, and why it's the right next rung: a golden set of ~20 questions with known-correct chunks, to measure retrieval recall@k and *calibrate the refusal threshold empirically* (find the lowest score among known-good retrievals, set the floor just below) instead of reading it off a band table; and an entailment check — "does chunk X actually support claim Y?" — which the citation audit deliberately does **not** do (it verifies the id exists, not that it backs the sentence).

*Check yourself:* The citation audit can pass while the answer is still wrong — how? — Why can't you set the refusal threshold correctly without a labeled question set?

---

## What's next / the production gap

The fixes are queued with reasons, so "what's missing" is a roadmap rather than a surprise. The advanced stage has now *measured* the first few (eval harness built; Principle 6); the rest stay queued.

**Measured (advanced stage — see `notes/advanced/`):**
- **Experiment 7 — round-robin retrieval** → **shipped (Phase A), recall@5 0.79 → 0.88.** Detect multiple named companies, run one filtered query each, reserve slots, merge. The structural cure for Finding 2, and the stage's biggest win. (An LLM-decomposition variant, Phase B/B+, *lost* to it — Principle 6.)
- **Cross-encoder re-rank on top-50** → **measured a wash** on this corpus (minilm 0.79→0.80; bge a harness bug). Not the playbook's promised win here.

**Still queued (hypothesis, not checkbox):**
- **Q7 / aspect-enumeration via retrieve-then-expand** — the one decomposition lever left unrealized: ground the split in a first retrieval pass instead of asking a blind LLM to enumerate.
- **Hybrid retrieval (dense + BM25)** — dense handles paraphrase, sparse handles exact identifiers ("Item 1A", specific dollar amounts). The golden set predicts only a *modest* win here (exact terms in these filings sit in semantically-similar prose).
- **MMR (diversity-aware selection)** to cut the "four adjacent chunks" over-retrieval problem.
- **LLM-as-judge eval** — score whatever is returned for relevance (no fixed key); the deeper fix for the broad "representative-label" questions the fixed-key harness can't grade fairly.
- **HyDE**; **a domain-tuned embedder** (`voyage-finance-2`) + **per-section context prefixes**; **promote `partial` to a first-class return signal** (generation Finding C follow-up).

Each of these is a deliberate experiment with a hypothesis, not a checkbox — which is the whole point of building the substrate first and measuring before adding machinery.

---

## How to read this alongside the rest

- For *what each stage does and the lessons from building it*: the per-stage `notes/*.md` (read in stage order).
- For *how to run it*: `README.md`.
- For *why it's shaped this way*: this file.

The notes plus the inline code comments plus this document are the design record. This one is the part you'd reach for when someone asks "why did you build it like that?" — and the self-test questions are how you find out whether you actually know.
