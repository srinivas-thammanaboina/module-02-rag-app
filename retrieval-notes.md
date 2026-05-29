# Retrieval notes — module-02-rag-app

**Takeaway:** Retrieval is the moment the system stops being a pile of stored chunks and starts answering questions. Mechanically it's three operations: embed the query, dot-product against every stored vector (or a filtered subset), return the top `k`. The conceptual simplicity matters because it makes failure modes legible — every wrong answer is one of *"the embedding was bad," "the chunks were bad,"* or *"the question is genuinely ambiguous."* The cheapest accuracy lever at this stage is **metadata filtering** — running the similarity search over only the rows that match a known constraint, instead of over the whole corpus.

## What retrieval mechanically is

```
question:  "what are Tesla's main risks?"
   │
   ▼ embedder.embed_query(question)
query_vec: float32[384]
   │
   ▼ collection.query(embeddings=[query_vec], n_results=k, where=...)
top-k rows ordered by cosine similarity (descending)
   │
   ▼ format for human / generator
{id, document, similarity, metadata} × k
```

The dot-product step is what every dense retrieval system does. Everything else — filters, re-rankers, fusion with sparse search — is a wrapper around it. Internalising this clean mental model is what lets you debug retrieval issues without flailing: if the top result is wrong, the issue is in *exactly one* of the four steps above.

## Filters happen BEFORE similarity scoring, not after

This is the single most important mental model for filtered retrieval, and the one most people misremember on first contact.

**Wrong picture (intuitive but wrong):**

```
1. Embed the question
2. Compute similarity vs ALL stored chunks
3. Sort by similarity
4. Drop rows that don't match the filter
5. Return the top k of what's left
```

**Right picture (what actually happens):**

```
1. Embed the question
2. Look up WHICH rows match the filter
3. Compute similarity vs ONLY those rows
4. Sort and return top k of those
```

The filter happens **first**. The similarity calculation never runs over the rest of the corpus. For our store, `collection.query(query_embeddings=[...], n_results=5, where={"ticker": "TSLA"})` runs the cosine search over Tesla's 251 chunks and returns the top 5 of those. It never touches Apple's or NVIDIA's rows. Chroma's HNSW index supports this natively; it's not a brute-force post-filter.

### Two consequences flow from getting this right

**Consequence 1: the top-k always exists and is always coherent.** Imagine the wrong picture for *"what are Tesla's risks?"* with `--company TSLA`. The unfiltered top-5 might be `[NVDA, NVDA, TSLA, AAPL, NVDA]` — most matched on the *risk* concept regardless of company. You drop the non-TSLA rows and you're left with **one** chunk. The "top-5" becomes top-1. In the right picture, the search runs over TSLA's 251 chunks; the *k* you asked for is the *k* you get, and every result is on-topic.

**Consequence 2: filtering is essentially free, often cheaper.** In the wrong picture you'd score 678 vectors, sort, then post-filter. That's the full cost of unfiltered retrieval plus a discard step. In the right picture you score 251 vectors (TSLA only). That's *cheaper* than unfiltered retrieval. Quality goes up; cost goes down. There is no tradeoff. This is what "cheapest accuracy lever" means concretely.

### The information-theoretic framing

A useful reframe:

- **Unfiltered retrieval** says: *"out of everything I know about, what's most similar to your question?"*
- **Filtered retrieval** says: *"out of the subset you've already told me matters, what's most similar?"*

The filter is **prior information** about which rows matter. When the user types `--company TSLA`, they're handing you a 100%-confidence prior. Discarding that information by searching the whole corpus and hoping the right rows naturally rise to the top is statistically wasteful.

## Why filtering is the cheapest accuracy lever

Consider:

> *"What does Tesla say about supply chain risk?"*

Same query, two retrieval strategies:

| Strategy | What top-5 might contain |
|---|---|
| **No filter** | Maybe 2 TSLA chunks (Item 1A risk factors), 1 AAPL chunk (also supply chain), 2 NVDA chunks (also supply chain). All are *valid matches for the question's text*; only 2 are matches for the *user's intent* of "Tesla specifically." |
| **`where={"ticker": "TSLA"}`** | 5 TSLA chunks. The same embedder, same query vector, same algorithm — but the search ran over 251 rows instead of 678, and every result is on-topic by construction. |

The unfiltered version isn't *wrong* — every chunk it returns is genuinely related to "supply chain risk." It's just answering a different question (*"who talks about supply chain risk?"*) than the user asked (*"what does Tesla say about supply chain risk?"*). The downstream generator then has to handle the mix: usually it picks the right one, sometimes it summarises across companies, occasionally it gets confused.

Filtering pre-solves the disambiguation. It's free at query time (HNSW handles it natively), free to implement (one keyword arg), and substantially improves the average answer. **This is the largest single-line accuracy gain available in any RAG pipeline.** Take it.

## When filtering hurts you

The trap is filtering when the user *didn't* mean a single company.

> *"How do Tesla and NVIDIA describe their AI investments?"*

Filtering by `ticker=TSLA` here loses NVIDIA's content entirely. The query is genuinely cross-company; the filter destroys the answer.

Two ways to handle this:

1. **Detect intent before filtering.** Parse the question for multiple companies before applying any filter. Outside Stage 5's scope; belongs in a query-rewriting layer.
2. **Default to no filter, opt-in to filtering.** Our CLI does this: `--company TSLA` is optional. The user (or eventually an LLM-driven router) decides. Wrong choice for now means a mixed-quality answer, not a missing answer.

A `--compare` flag in the CLI runs both strategies side-by-side on the same question — useful when iterating on prompts and when teaching the concept.

## The three signals that justify filtering

In any RAG system that supports filtering, the decision of *when* to apply a filter comes down to parsing the user's question for the right signals:

| Signal in the question | Example | Apply filter? |
|---|---|---|
| User explicitly names **one** company | *"Tesla's risk factors"* | Yes — `ticker=TSLA` |
| User names **multiple** companies | *"Compare Tesla and Apple"* | No — need both |
| User names **none** | *"Which company depends most on China?"* | No — system must consider all |

In a fully realised system, a small classifier — or even a one-shot LLM call — would parse the question and route accordingly. We're not building that classifier in this project. We're building the substrate it would sit on top of. The CLI's `--company` flag is the human's manual stand-in for the router we're not writing yet.

Filters in retrieval are a form of **constrained search**. They work the same way constraints work everywhere in computer science: they shrink the search space without changing the search algorithm. The art is in knowing what constraints to apply when, not in the constraints themselves.

## Pressure-test scenarios and decisions

Before writing the code we walked through four failure modes that a naive top-k design exhibits. Summary of what each surfaces and what we chose to do:

| Scenario | Failure mode | Decision |
|---|---|---|
| 1. Filter ↔ question text disagreement (e.g. `--company AAPL` on a Tesla question) | Filter silently wins; user gets Apple chunks for a Tesla question with no signal that intent and infrastructure disagreed. | **Mitigate now.** Detect ticker / company name in the question text; print a warning when the detected company differs from `--company`. The filter still wins — that's the contract — but the contradiction is no longer silent. |
| 2. Corpus genuinely doesn't contain the answer | Top-k still returns 5 results at the BGE noise floor (~0.50–0.55). Generator can't tell that the chunks don't answer the question. | **Mitigate now.** Surface the top-1 similarity prominently in the CLI output and label its confidence band per `embedding-notes.md`. Stage 6's prompt will act on this signal; Stage 5's job is to make it visible. |
| 3. Adjacent chunks dominate the top-k (because of 150-char overlap) | Top-5 may be five consecutive chunks from the same passage — same content told five times instead of five different things. | **Defer.** Need to see it on real queries before committing to MMR. Queued as Experiment 2 below. |
| 4. Question implies enumeration (*"Which companies discuss…"*) | Retrieval returns ranking, not enumeration. Structurally wrong tool for the question. | **Acknowledge, do not mitigate.** Belongs in a query-planning layer outside Stage 5's scope. The user-facing answer for now is "ask one company at a time." |

### Scenario 1 detail — company mismatch detector

A small helper `detect_companies_in_question(question) -> set[str]` scans for the three known tickers and their lowercase company names (`tesla`, `apple`, `nvidia`). If `--company X` is set and the detected set is non-empty and does not contain `X`, the CLI prints:

```
WARNING: question mentions {detected}, but --company=X was set.
The filter will win — retrieval will return X content.
```

False positives (mentions of a company not actually meant as a filter target) trigger spurious warnings; we accept that because false negatives (missing a real contradiction) would re-introduce the original silent-failure problem. Conservative bias: warn loudly, let the human decide.

### Scenario 2 detail — confidence labelling on top-1

The CLI prints `top-1 sim : X.XXXX (label)` where the label comes from the bands in `embedding-notes.md`:

| Top-1 cosine (BGE-small) | Label |
|---|---|
| ≥ 0.75 | very high (direct paraphrase) |
| 0.65 – 0.75 | high (clearly relevant) |
| 0.58 – 0.65 | moderate (likely relevant) |
| 0.52 – 0.58 | low (near BGE noise floor — corpus may not contain a good match) |
| < 0.52 | very low (likely no good match) |

Stage 5 only labels; it does not gate. Filtering out low-confidence results is Stage 6's call — the prompt may want to behave differently (refuse, hedge, ask for clarification) based on this signal. Single-layer "if score < X, return nothing" is brittle; the two-layer "retrieval reports, prompt acts" is robust.

## What we're building in `app/retrieve.py`

A thin orchestration layer. The hard work is already in the vector store; this module just wires it together with formatting.

```python
class Retriever:
    def __init__(self, store: VectorStore): ...
    def retrieve(self, question: str, k: int = 5, company: str | None = None) -> list[dict]:
        where = {"ticker": company.upper()} if company else None
        return self.store.query(question, k=k, where=where)
```

That's the whole class. ~10 lines of substance.

CLI surface:

```bash
python cli.py retrieve --question "..."                       # no filter
python cli.py retrieve --question "..." --company TSLA        # filtered
python cli.py retrieve --question "..." --company TSLA --k 8  # custom top-k
python cli.py retrieve --question "..." --compare             # both, side by side
```

The CLI's job for Stage 5 is to **show the user what was retrieved with full context**: similarity score, ticker, section, document text. The output should make it obvious whether retrieval did its job, without needing to read the full chunks.

## Design decisions baked into the code (so you remember why later)

**Filter syntax is Chroma's, hidden behind the wrapper.** The retriever takes `company: str | None`; the wrapper translates to `where={"ticker": ...}`. If we ever swap stores, this translation lives in one place. The retriever never sees raw Chroma `where` dicts.

**Top-k default = 5, configurable via `--k`.** Five is the "show me a handful" default. Three is too few (one bad chunk poisons the context); ten is too many (the generator's context window gets noisy). For learning, 5 makes inspection manageable; for production you'd calibrate.

**Score is reported alongside every result.** Even though absolute scores are model-specific (per `embedding-notes.md`), seeing them lets you spot two important patterns: (a) top scores all in the 0.70+ band → confident retrieval, (b) top scores in the 0.55–0.60 band → the corpus may not actually contain the answer. We'll surface this distinction in the demo runs.

**The `--compare` flag is teaching-only.** Production retrieval picks one strategy; the comparison view exists to make the filtering lesson concrete. It will likely not survive into a future "production-shaped" version of the project.

**The company-mismatch warning lives in the CLI layer, not the `Retriever`.** The `Retriever` class itself is pure mechanics — it does not parse the question. The warning belongs at the user-facing edge because it's a UX concern, not a retrieval concern. If we ever build a programmatic API (Stage 6+), it can opt into the same `detect_companies_in_question()` helper without coupling.

**Confidence labels are CLI-only too.** The retriever returns raw similarity scores; the labels are presentation. Stage 6 will read raw scores; the labels exist to make the noise-floor problem visible to a human eyeballing the output.

## Sanity-check experiment (run during this stage)

Three questions chosen to exercise different parts of the retrieval surface:

| # | Question | Expected behaviour |
|---|---|---|
| 1 | *"What are the main risks Tesla faces?"* | With `--company TSLA`: all 5 chunks from TSLA Item 1A. Score ≥ 0.65. |
| 2 | *"What does Apple say about supply chain concentration?"* | With `--company AAPL`: chunks span Item 1A + Item 7 (the topic crosses sections). Should still rank coherently. |
| 3 | *"How do Tesla and NVIDIA describe their AI investments?"* | No filter. Top-5 should include both TSLA and NVDA chunks. With `--company TSLA` it loses NVDA — demonstrating the failure mode. |

For each question we'll record: top-k similarities (the distribution shape), which sections appeared, whether the retrieved chunks would actually let a human answer the question (the "human grader" smoke test).

**Status:** implemented and run. Five questions used to exercise the design surface — three from the original plan plus two added to test the pressure-test mitigations directly.

### Question 1 — *"What are the main risks Tesla faces?"* with `--company TSLA`

```
Top-5 sims: 0.7722 / 0.7329 / 0.7198 / 0.7178 / 0.7114
Sections  : 5 × Item 1A. Risk Factors  (all)
Top-1 label: very high (direct paraphrase)
Elapsed   : 147 ms
```

Human grader: solid. Chunks cover EV demand and adoption risk, employee retention, production forecasting, tax-credit dependency, and third-party commentary risk. A reasonable spread of distinct risk topics within Item 1A.

### Question 2 — *"What does Apple say about supply chain concentration?"* with `--company AAPL --compare`

```
WITH FILTER (AAPL):
  Top-5 sims: 0.6901 / 0.6870 / 0.6790 / 0.6721 / 0.6718
  Sections  : 4 × Item 1A + 1 × Item 7 (MD&A — tariffs/operations)
  Top-1 label: high (clearly relevant)

WITHOUT FILTER:
  Top-5 sims: identical
  Tickers   : AAPL × 5
  Result    : IDENTICAL to filtered — same chunks, same scores, same order.
```

The topic crosses sections as predicted (Item 1A risk factors + Item 7 tariff discussion both ranked). **The notable finding is the filter producing no effect** — see Finding 1 below.

### Question 3 — *"How do Tesla and NVIDIA describe their AI investments?"* with `--company TSLA --compare`

```
WITH FILTER (TSLA):
  Top-5 sims: 0.6831 / 0.6791 / 0.6645 / 0.6570 / 0.6495
  Sections  : 2 × Item 1 + 3 × Item 7
  Tickers   : TSLA × 5  (NVDA content lost, as designed)
  Top-1 label: high (clearly relevant)

WITHOUT FILTER:
  Top-5 sims: 0.7625 / 0.7480 / 0.7443 / 0.7394 / 0.7392
  Sections  : 4 × Item 1 + 1 × Item 7
  Tickers   : NVDA × 5  (zero TSLA chunks in top-5)
  Top-1 label: very high (direct paraphrase)
```

**This is the sharpest finding of the run** — see Finding 2 below. Removing the filter does NOT produce a balanced cross-company view; it produces a single-company view dominated by whichever company's content embeds harder against the query.

### Question 4 — *"What are Tesla's risk factors?"* with `--company AAPL`  (warning trigger)

```
Top-5 sims: 0.6812 / 0.6433 / 0.6412 / 0.6299 / 0.6289
Sections  : 5 × AAPL Item 1A. Risk Factors  (all)
Top-1 label: high (clearly relevant)

Warning printed (verified on re-run):
  WARNING: question mentions ['TSLA'], but --company=AAPL was set.
  The filter will win — retrieval will return AAPL content.
  If you want different content, change or drop --company.
```

The contract is intact: filter wins over question text content. Even though the question explicitly mentions Tesla, the filter forces retrieval onto Apple's chunks, and the embedder finds Apple risk-factor content that is semantically close to the query's "risk factors" framing. The warning makes the contradiction visible to the user *before* the chunks scroll past — exactly the design intent of Pressure-test Mitigation 1.

### Question 5 — *"What is the CEO's home address?"* with `--company TSLA`  (confidence trigger)

```
Top-5 sims: 0.5656 / 0.5307 / 0.5296 / 0.5207 / 0.5070
Sections  : 4 × Item 1A + 1 × Item 1
Top-1 label: low (near BGE noise floor — corpus may not contain a good match)
```

Mitigation 2 works as designed. Top-1 sits at 0.566 — clearly in the noise band. Returned chunks are *about* Musk and his role as CEO (semantically near "CEO"), but none contain an address (nor would any 10-K). The label correctly signals "this is the best the corpus could do, and it's not great." Stage 6's prompt will read this signal and refuse cleanly.

## Reading top-k results critically — patterns to watch

Once results are in front of you, three things tell you whether retrieval worked:

1. **Score distribution.** If top-1 is 0.85 and top-5 is 0.62, retrieval is confident — the answer is probably in top-1 or top-2. If top-1 is 0.62 and top-5 is 0.58, the corpus may not contain a good match; retrieval pulled "the least bad" rows.
2. **Section coherence.** For a focused question, the top-5 should mostly come from the same section (e.g. all from Item 1A for a risk question). Spread across sections → either a diffuse question or noisy embeddings.
3. **Textual overlap between chunks.** Because we use 150-char overlap during chunking, two adjacent chunks share ~150 chars of text. If top-5 has two adjacent chunks from the same section, you're likely covering one important point well; if it has four adjacent chunks, you're over-retrieving (a re-ranking step would help).

These are eyeballing skills — they sit between "no evaluation" and "full eval harness." For a learning project they're sufficient; for a production system you'd quantify them.

## Hypotheses for future experiments (queued)

- **Experiment 2 — MMR (Maximal Marginal Relevance).** Replace pure top-k with a diversity-aware selector: pick the highest-scoring chunk, then iteratively pick chunks that are simultaneously *relevant to the query* and *dissimilar from already-picked chunks*. Cuts the "four adjacent chunks" problem. Standard RAG enhancement.
- **Experiment 3 — Cross-encoder re-rank on top-50.** Retrieve 50 candidates by cosine, then re-rank with `bge-reranker-base` which scores (query, document) pairs jointly. Slower per query, often a meaningful precision boost on technical text.
- **Experiment 4 — Hybrid retrieval (dense + BM25).** Add a sparse keyword index alongside Chroma; combine scores via reciprocal rank fusion. Dense handles paraphrase; sparse handles exact identifiers ("Item 1A," "GAAP," specific dollar amounts).
- **Experiment 5 — Sub-section filtering.** If the question matches a section keyword ("risk" → Item 1A, "management" → Item 7), prefer that section via metadata. Cheap heuristic; sometimes meaningfully helpful, sometimes a footgun.
- **Experiment 6 — HyDE (Hypothetical Document Embeddings).** Have an LLM draft a hypothetical answer to the query, then embed *that* and retrieve against it. Closes the distribution gap between short queries and long documents. Cheap to test if results disappoint.
- **Experiment 7 — Cross-company round-robin retrieval (motivated by Finding 2).** When the question mentions multiple companies, run a separate filtered retrieval per company and merge with reserved slots — e.g. for top-5 with two named companies, take top-3 from company A and top-2 from company B. Trade-off: each company's top-`k/N` may be weaker individually than the dominant company's top-5, but cross-company balance is guaranteed. Structural fix for the comparison-question failure mode that pure top-k cannot solve.

## Lessons to carry forward

### Finding 1 — Embedder bias can swallow filter behavior

When the question text already names the company (*"What does **Apple** say about supply chain concentration?"*), the embedding vector pulls so strongly toward that company's content that the unfiltered search converges to the same result as the filtered search. Question 2 returned the **identical** top-5 with and without `--company AAPL`.

This is not a failure. The filter is still correct to apply (cheap insurance, costs nothing). But it does mean that *"my filter doesn't seem to matter on this question"* is sometimes the right reaction — and is not evidence that the filter is broken or unhelpful in general. The filter adds visible value precisely when the query is **generic** ("what are the supply chain risks?" with no company named) — because that's when the unfiltered search would naturally mix companies.

The corollary worth remembering: **embedder bias is a kind of implicit filter**. The query embedding already biases the search toward whichever company's vocabulary it shares the most surface with. Explicit metadata filtering is a guarantee on top of that bias; both layers stack.

### Finding 2 — Pure top-k cannot answer cross-company comparison questions

Question 3 (*"How do Tesla and NVIDIA describe their AI investments?"*) without a filter returned **5 NVDA chunks and 0 TSLA chunks**. The naive intuition was that no filter → balanced retrieval across both companies. The actual behavior: no filter → whichever company's content embeds hardest against the query wins all the slots.

NVIDIA explicitly markets itself as an AI infrastructure company. Its prose around "AI investments" is denser, more directly matched to the query's vocabulary, and (in this corpus) more abundant. The embedder's similarity scoring picked it up; Tesla's also-AI-focused but differently-framed content didn't crack top-5.

**Structural implication:** for any question of the form *"compare X and Y"* or *"how do X, Y, Z differ in…"*, pure top-k cannot give a balanced answer. The only correct strategy is **per-company retrieval merged manually** — run the query once per company (each with its own ticker filter), take top-`k/N` from each, concatenate. That logic belongs in a query-planning layer above the retriever; Stage 5's pure-top-k API cannot solve it.

Queued as **Experiment 7 — Cross-company round-robin retrieval** below.

This finding upgrades Scenario 4 from the pressure test from "acknowledge limitation" to "demonstrated, repeatable failure mode that the current CLI cannot fix."

### Finding 3 — Confidence labelling works as designed

Question 5 (*"What is the CEO's home address?"*) produced top-1 similarity 0.566 — squarely in BGE-small's noise floor. The label *"low (near BGE noise floor — corpus may not contain a good match)"* fired correctly. The returned chunks were semantically near "CEO" (mentions of Musk's role) but contained no address — exactly the failure mode the label exists to surface.

Stage 6 can now read this signal directly. The prompt will be able to say *"the provided context does not contain an answer to the question"* when top-1 is below 0.58, instead of being forced to either hallucinate or hedge ambiguously.

### Finding 4 — Filter wins over question text content

Question 4 (*"What are Tesla's risk factors?"* with `--company AAPL`) returned 5 Apple Item 1A chunks at similarity 0.63–0.68. The query embedded toward Apple's risk-factor vocabulary — *not* toward Tesla's content — because that's what the filter allows the embedder to search over.

The contract is exactly as designed: `--company` is gospel; question text is just the embedding seed. The user's natural mistake (typing the wrong filter, or pasting the wrong question) produces *plausible-looking wrong content*, not zero content. The company-mismatch warning is the visible signal that something is off — without it, the CLI would silently return Apple content for a Tesla question, which is the worst kind of bug.

### Finding 5 — Confidence band thresholds must be re-derived per embedder

The thresholds we use (0.75 / 0.65 / 0.58 / 0.52) are calibrated specifically for `BAAI/bge-small-en-v1.5` on English prose, against the noise-floor data captured in `embedding-notes.md`. Swapping the embedder (Experiment 1 of `embedding-notes.md` future experiments — `bge-base`, `voyage-finance-2`, OpenAI) requires recalibrating these bands against the new model's actual score distribution. The labels are correct *for this embedder only*. Carrying them across model changes would silently mislead.

## How to think about retrieval, generally

Retrieval looks simple — top-k by similarity — and the implementation is simple, but the **art** of retrieval is in (a) chunk design, (b) embedder choice, (c) what you filter on, and (d) what you do after the top-k comes back. Stages 2, 3, 4 already locked in the first three for this project. Stage 5 is where you start *seeing* whether those choices were good. If retrieval is great, downstream prompting is easy. If retrieval is poor, no prompt engineering can rescue it — the generator can only work with what it's been shown.

The discipline to internalise: **debug retrieval first, prompts last.** Most failed RAG systems in the wild are failed retrieval systems with elaborate prompts compensating. We're building retrieval well so we don't need that compensation.
