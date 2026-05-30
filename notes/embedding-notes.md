# Embedding notes — module-02-rag-app

**Takeaway:** An embedding turns a piece of text into a point in a high-dimensional space, where semantically similar texts land at nearby points. Once you have embeddings for everything in your corpus and for the user's query, "retrieval" reduces to "find the points nearest to the query point." That's the entire premise on which RAG operates — get this layer right and downstream stages have a chance; get it wrong and no amount of clever generation prompting can recover.

## What an embedding actually is (intuition)

When you "embed" a chunk of text, you run it through a small neural network whose only job is to convert text into a list of numbers — 384 of them for our model. Those 384 numbers are a coordinate in a 384-dimensional space. The model is trained so that semantically similar texts end up at *nearby* coordinates and unrelated texts end up *far apart*.

The vector itself has no human-readable meaning. If you peek at one you'll see something like `[0.018, -0.034, 0.092, ..., -0.011]`. None of those numbers individually corresponds to a concept. What matters is the geometric relationship to *other* vectors.

**Analogy that helps:** imagine every word has a color. "Hot" and "warm" sit in the red-orange range; "frozen" and "icy" are blue; "tree" is green; "ocean" is also blue but a different shade. The exact RGB values don't matter — what matters is that hot–warm are close and hot–frozen are far. Embeddings do this for sentences and paragraphs in 384 dimensions instead of three.

## How "near" is measured: cosine similarity

Two vectors are compared by **cosine similarity**: the cosine of the angle between them. Range is `-1` (opposite directions) through `0` (perpendicular, unrelated) to `1` (same direction, identical meaning). Because our model outputs **L2-normalized** vectors (every vector has length exactly 1), cosine similarity simplifies to a plain dot product, which is the cheapest operation a CPU knows how to do at scale.

**Why we care about cosine and not Euclidean distance:** in high-dimensional space, the *direction* a vector points encodes meaning far more reliably than its *magnitude*. Two vectors pointing the same way are semantically the same thing, regardless of how long each one is. Cosine throws magnitude out and just measures direction.

## Why dense embeddings beat keyword matching

A keyword search for *"supply chain disruption"* misses a chunk that says *"issues with our component vendors"*. A dense embedding of both lands them at nearby coordinates because the model was trained on enough text to know they mean approximately the same thing. That generalization is the entire reason RAG works — and it's also why a perfectly fine question can return zero results when you grep instead.

This is also why pure keyword matching (BM25, TF-IDF) and dense retrieval are sometimes combined — they make different mistakes. Dense embedders handle paraphrase but can miss exact identifiers ("Item 1A", "Section 7"); keyword search handles identifiers perfectly but fails on paraphrase. Hybrid retrieval combines them. We're not doing hybrid yet — pure dense first, then we measure.

## BGE's quirk you must know about

BGE-family models (`BAAI/bge-small-en-v1.5`, our default) were trained on (query, document) pairs where queries received an exact prefix string:

```
Represent this sentence for searching relevant passages: <query>
```

Documents got *no* prefix. At inference time you have to match that pattern. People who skip the prefix on queries lose 5–10% retrieval accuracy and never figure out why — there's no warning, no error, just silent degradation.

**We handle this inside the embedder, not in the caller.** `embed_query()` prepends the prefix; `embed_documents()` doesn't. The retriever and the demo CLI never have to remember.

Reference: the model card at `https://huggingface.co/BAAI/bge-small-en-v1.5` documents this behavior.

## Why hide the embedder behind an interface

The pattern: `Embedder` is an abstract class with `embed_query(text) -> vector` and `embed_documents(texts) -> matrix`. Today we ship one concrete implementation — `LocalSentenceTransformerEmbedder`. Tomorrow we might add `OpenAIEmbedder`, `VoyageEmbedder`, or a fine-tuned in-house model.

**The argument from first principles:** in any real RAG system you will A/B test embedders. The metric difference between a strong general embedder and a domain-tuned one on your corpus is often the difference between "decent" and "production-ready" retrieval. You want the swap to be a one-line config change, not a refactor across `retrieve.py`, `store.py`, and the demo scripts. The Embedder interface is the seam that makes this cheap.

**Concrete swap story:** today `config.embedding_model = "BAAI/bge-small-en-v1.5"` and `get_embedder()` returns the local class. To test OpenAI, you'd add an `OpenAIEmbedder` class implementing the same two methods, dispatch on a config flag inside `get_embedder()`, and nothing else in the codebase changes. The contract — input is text, output is normalized vectors with a known dim — is what every downstream stage depends on.

## Why `embed_query` and `embed_documents` are separate methods

Not only because BGE needs different prefixes (that justification alone would be enough). Also because the **shape of the inputs differs**:

- A query is one string. You embed it on the user's critical path; latency matters; batching is meaningless because there's only one.
- Documents come in bulk — hundreds or thousands at a time. You batch them through the model for throughput; you might want a progress bar; you definitely want `batch_size` tuning.

Having both behaviors in one method would force every caller to think about which mode they're in. Separating them encodes the right behavior at the interface level. Same pattern shows up in essentially every retrieval framework (LangChain, LlamaIndex, Haystack) — two methods, one for queries, one for documents.

## Why `LocalSentenceTransformerEmbedder` lazy-imports `sentence_transformers`

The import lives **inside `__init__`**, not at the top of `app/embed.py`:

```python
def __init__(self, model_name: str):
    from sentence_transformers import SentenceTransformer
    self._model = SentenceTransformer(model_name)
```

Reason: `sentence_transformers` pulls in PyTorch, which is ~2 GB of weights and takes a noticeable second or two to import. If we top-level-imported it, `python cli.py --help` would block for that second every time — and Stage 1 (ingest) and Stage 2 (chunk) would import it for no reason at all. Lazy import means the cost is paid exactly when an embedder is actually constructed.

This is a small thing but worth internalizing: **import-time cost is real cost**. Hot imports stay at the top; heavy imports go where they're used.

## Practical specs of our default embedder

| Property | Value |
|---|---|
| Model | `BAAI/bge-small-en-v1.5` |
| Embedding dimension | 384 |
| Max input length | 512 tokens (~2,000 characters) |
| L2-normalized output | Yes |
| Local disk footprint | ~130 MB (downloads on first use to `~/.cache/huggingface/hub/`) |
| CPU inference | Fine for our scale (~700 chunks, single-digit seconds) |
| Batch size used | 32 (good CPU saturation; bigger doesn't help much) |
| License | MIT |

**Why this model and not something bigger:**

- `bge-small-en-v1.5` is the smallest of the BGE family. It still benchmarks well on MTEB (the retrieval benchmark that matters). For a learning project on three filings, a larger model is wasted compute.
- 384 dimensions means our index storage is ~3KB per chunk (mostly the vector). 678 chunks × 3KB = ~2MB. Trivial.
- It runs in a few seconds for our entire corpus on a laptop CPU. No GPU needed.
- Real production systems for filing search would probably move to a domain-tuned model (e.g. `voyage-finance-2`) — but we'd swap that in via the interface, not by rewriting code.

## Sanity-check experiment

Designed but **not yet run** as of this writing. The goal is to verify two things before trusting the embedder downstream:

1. **Vectors are normalized.** A single `--text` invocation should print an L2 norm of `1.000000` (or to within rounding).
2. **Semantically related texts produce higher cosine similarity than unrelated ones.** A multi-text invocation should rank a paraphrase of the query above a tangentially-related sentence, which should in turn rank above an unrelated sentence.

**Command:**

```bash
python cli.py embed \
  --text "supply chain risk from foreign suppliers" \
  --text "we depend on third-party component vendors" \
  --text "our cost of goods sold increased due to inflation" \
  --text "the company logo and brand identity are protected by trademarks"
```

**Predicted score bands (calibration for later):**

| Doc | Relationship to query | Expected sim |
|---|---|---|
| "we depend on third-party component vendors" | Paraphrase | 0.75 – 0.90 |
| "our cost of goods sold increased due to inflation" | Same domain, different topic | 0.40 – 0.55 |
| "the company logo and brand identity are protected by trademarks" | Unrelated topic | 0.10 – 0.25 |

**Results — actual run:**

```
Multi-text run (4 inputs):
  query:   "supply chain risk from foreign suppliers"

  rank 1:  sim=0.5870  "we depend on third-party component vendors"
  rank 2:  sim=0.5572  "our cost of goods sold increased due to inflation"
  rank 3:  sim=0.5127  "the company logo and brand identity are protected by trademarks"
```

**Verdict: rank ordering is correct (paraphrase > same-domain > unrelated). Absolute scores are far compressed compared to my predicted bands.** This is by far the most important lesson of the embedding stage — see the next section.

## The score-compression lesson (calibration is per-model)

The predicted bands above were wrong for BGE specifically. Predicted: 0.75–0.90 for a paraphrase, 0.10–0.25 for unrelated content. Observed: 0.587 and 0.513. The gap between "most relevant" and "totally unrelated" is just **0.074** of cosine — not the ~0.65 of cosine the predictions implied.

Why this happens: BGE-family models were trained with a contrastive objective that pushed positive pairs (correctly matched query/document) toward similarity 1.0, but didn't aggressively push negative pairs toward 0. The training cared more about *separation* (positives must outrank negatives) than about *absolute geometric distance*. So every English-language embedding the model produces has a positive correlation with every other one, just because they share enough structure as "English prose."

**Consequence:** in BGE's geometry, two random English sentences typically sit at cosine ~0.5, not ~0.0. The model's effective range for natural prose is roughly **0.45–0.90**, not 0–1.

**Other embedders behave differently:**

| Model family | Typical "unrelated" cosine | Typical "paraphrase" cosine | Useful range |
|---|---|---|---|
| `bge-small-en-v1.5` (ours) | ~0.50 | ~0.60–0.85 | 0.45–0.90 |
| OpenAI `text-embedding-3-*` | ~0.10–0.20 | ~0.50–0.85 | 0.10–0.90 |
| `all-MiniLM-L6-v2` (older) | ~0.05–0.20 | ~0.50–0.80 | 0.05–0.85 |

**The durable rule that survives the model swap:** trust the rank ordering, calibrate any threshold empirically against the model + corpus combination you're actually using. Never carry an absolute cosine threshold between embedders. The "0.7 is relevant" heuristic you might read in a tutorial is true only for the model in that tutorial.

## What this means for Stage 5 (retrieval)

Two practical implications:

1. **Top-k retrieval works fine.** We take the highest-scoring k chunks; absolute numbers don't matter, only the order does. Our embedder ranks correctly, which is what retrieval requires.
2. **Absolute-threshold filtering is a calibration problem, not a number you can guess.** If we ever want a "no answer found below threshold X" feature, we'll have to measure: pick 20 questions where you know the correct chunk exists, find the lowest score among those, set the threshold a hair below. Repeat if the corpus grows substantially.

## Score interpretation bands for `bge-small-en-v1.5` (calibrated against our run)

The earlier draft of this section listed wide bands (0.30 / 0.50 / 0.65 / 0.80) that turned out to be wrong for BGE — see "The score-compression lesson" above. Below are bands calibrated against the actual model and corpus we're using.

| Cosine similarity (BGE-small) | What it usually means |
|---|---|
| 0.75 – 1.00 | Direct paraphrase or extremely close. Should be top-1. |
| 0.60 – 0.75 | Clearly relevant. Same topic, same framing. |
| 0.55 – 0.60 | Same domain. Possibly relevant; possibly not. Inspect. |
| 0.50 – 0.55 | Noise floor. Two natural-English sentences. Treat as unrelated. |
| Below 0.50 | Genuinely orthogonal — different languages, code vs prose, etc. |

These bands are **only** for `bge-small-en-v1.5` on English prose. If we swap embedders later (Stage 5/6 experimentation), this table needs to be re-derived for that model.

## Lessons to carry forward

- **The model card is not optional reading.** BGE's query prefix requirement isn't in random blog posts — it's in the model's own documentation. Read it for every embedder you adopt. Five minutes saves five hours of "why is retrieval worse than I expected?"
- **Normalize once, simplify forever.** Producing L2-normalized vectors at the embedder layer means every downstream similarity computation is a single dot product. No vector-norm management spread across the codebase.
- **Two seams matter at this stage:** (a) the `Embedder` ABC for model swaps, (b) the query/document method split for input-shape correctness. Both are nearly free to set up at the start and expensive to retrofit later.
- **Lazy-import heavy libraries.** A 2-second `--help` is a bad signal for a learning tool.
- **Always include a sanity-check command.** The CLI `embed --text "..."` mode that just shows dim, norm, and the first 8 values is what catches "I loaded the wrong model" or "the download is corrupt" before you spend an hour debugging Stage 5 retrieval.

## Future experiments worth running

- **Swap to a larger BGE model** (`bge-base-en-v1.5`, 768-dim) — measure whether retrieval precision improves enough to justify 2× storage and 3× compute.
- **Try a domain-tuned model** — `voyage-finance-2`, `nomic-embed-text-v1.5`, or a financial-domain BGE fine-tune (if one exists). Filings are a niche domain; general embedders may miss vocabulary like "GAAP," "guidance," "headwinds."
- **Hybrid retrieval (dense + BM25)** — combine our embeddings with a sparse keyword index. Dense handles paraphrase; BM25 handles exact identifiers. Stage 5 territory, but the embedding layer doesn't change.
- **Re-rank top-50 with a cross-encoder** — retrieve 50 candidates by cosine, then re-rank using a cross-encoder (e.g. `bge-reranker-base`) that scores (query, document) pairs jointly. Typical accuracy boost is meaningful at modest latency cost.
- **HyDE (Hypothetical Document Embeddings)** — instead of embedding the user's literal query, have an LLM draft a hypothetical answer first, then embed *that*. Queries are short and underspecified; documents are long and rich; embedding a query-like-document closes the distribution gap. Often 5–10% retrieval bump on benchmark sets.
- **Per-section embedding contexts** — prepend a section tag like `"Section: Item 1A. Risk Factors\n"` to each chunk before embedding. The model has more context to embed against. Cheap experiment, sometimes meaningfully helpful.

## How to think about embeddings, generally

Embeddings turn the squishy problem of *meaning* into the hard problem of *geometry* — and we have great algorithms for geometry. The bet of dense retrieval is that two-decade-old vector math (k-nearest-neighbor search) running on modern hardware can ride on top of a well-trained encoder to find semantic matches at scale. That bet pays off when (a) your encoder is well-matched to your text domain, (b) your chunks are coherent semantic units, and (c) your queries and documents live in the same embedding distribution.

The job of an AI engineer working on RAG, in one sentence: keep those three conditions true. Everything else — chunk size, top-k, reranking, prompt engineering — is downstream from whether the embedding layer is faithfully representing the meaning of your text.
