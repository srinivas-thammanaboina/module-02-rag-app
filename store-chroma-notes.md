# Vector store notes (Chroma) — module-02-rag-app

**Takeaway:** A vector store is the threshold that turns disconnected stages — chunks, embeddings, queries — into a system. It persists `(id, document, vector, metadata)` rows so that retrieval has a single addressable place to look. Its real value is not raw search speed (any modern laptop can dot-product 10,000 vectors in milliseconds) but **integrity, metadata filtering, idempotent re-runs, and an upgrade path that doesn't require code changes**. Pick it for those properties; ignore the speed marketing.

## What a vector store actually is

A vector store is a small database whose primary index isn't a `WHERE col = X` clause but a *nearest-neighbour-in-vector-space* lookup. You hand it a query vector; it hands back the `k` rows whose stored vectors are closest. Every row carries three things alongside the vector:

- the **document text** (returned verbatim — the store never tries to reconstruct text from vectors, because you can't)
- a **metadata dict** (used to *filter* candidates before similarity ranking)
- a stable **id** (used to identify a row for upsert / delete)

That four-tuple — `(id, document, vector, metadata)` — is the unit of storage. Everything Chroma exposes is built around treating that tuple as atomic.

## The numpy primer (since "we could just use numpy" is the foil)

**numpy** is the foundational numerical-computing library for Python. It provides one core data type — the *N-dimensional array* (`ndarray`) — and thousands of operations that work on whole arrays at once rather than one element at a time. When you write `a @ b` in Python and `a`, `b` are numpy arrays, the multiplication is implemented in highly-optimised C (often calling BLAS / LAPACK / SIMD instructions under the hood). A pure-Python loop over the same data would be 100×–1000× slower.

For our problem specifically:

```python
embeddings = np.zeros((678, 384), dtype=np.float32)   # one matrix row per chunk
query      = np.zeros(384,        dtype=np.float32)   # one vector per query
similarities = embeddings @ query                      # one dot product per row, in one call
```

That last line returns a length-678 array of cosine similarities (because our vectors are L2-normalized, dot product = cosine). It runs in a few hundred microseconds. The whole "compute similarity of every chunk to the query" step — the *math* at the heart of dense retrieval — is one line of numpy. **Every vector database in existence is doing this math under the hood**, plus wrappers for storage, filtering, indexing, and bookkeeping.

This matters because it sets up the honest version of the question: "we already have the math; why do we need a database around it?" The answer isn't *speed* — it's everything *but* speed.

## Why a vector DB, not just numpy — the four reasons

Listed in increasing order of importance for our project.

### 1. Row integrity

The numpy version of the pipeline needs three things kept in lock-step: `embeddings.npy`, `texts.json` (or `.jsonl`), and `metadata.json`. The mapping between them is positional — row `i` of the matrix corresponds to entry `i` of the text list and entry `i` of the metadata list. Insert one row anywhere, all three structures must be updated in the same order. Sort one accidentally, you've silently corrupted the mapping — chunk #143's text is now paired with chunk #144's vector.

A vector store makes the row atomic: `(id, document, vector, metadata)` travel together always, indexed by id rather than by position. No alignment bugs are possible because there's nothing to align. The id is the only handle; everything else moves with it.

### 2. Approximate nearest neighbour at scale

Brute-force `embeddings @ query` stays comfortable up to about 10,000 chunks (single-digit milliseconds on a laptop). At 100,000+ you start wanting algorithmic shortcuts:

- **HNSW** (Hierarchical Navigable Small World) — a graph index that finds approximate nearest neighbours in `O(log n)` time at the cost of <5% recall loss. Chroma's default index.
- **IVF** (Inverted File) — a coarse-grained clustering trick: partition vectors into buckets, search only the bucket(s) nearest the query.

We don't *need* either at 678 chunks. The point is that picking Chroma now means our code doesn't change when the corpus grows. The seam absorbs the scaling concern. Junior move: write numpy now, rewrite the whole storage layer when it doesn't scale. Senior move: choose the abstraction that already handles the scale you might reach.

### 3. Idempotent re-runs (upsert semantics)

We'll re-build this index dozens of times while iterating on chunking, embedding, prompts. Every rebuild should produce the same final state — no duplicates, no orphan rows from previous runs. With Chroma's `upsert(id=...)`, **same id overwrites**. Running `python cli.py build` ten times → identical state to running it once.

In numpy you'd hand-roll this: maintain an `id → row-index` dict, handle the case where a previously-existing chunk is no longer in the rebuild (shrink), handle inserts (grow), keep all three parallel structures in sync. This is the bug-farming version of the storage layer. Every project that tries it eventually hits the "stale embeddings from a previous run silently mixed in with fresh ones" class of bug.

### 4. Metadata filtering — the actual reason

This is the one to take into an interview. Consider:

> *"What does Tesla say about supply chain risk?"*

| Strategy | What top-5 returns |
|---|---|
| **No filter** | A mix of TSLA / AAPL / NVDA chunks. The generator has to figure out which company the user meant. |
| **`where={"ticker": "TSLA"}`** | Five TSLA chunks. The similarity search runs *only* over Tesla's 251 chunks. |

Same embeddings, same algorithm, same model — one extra constraint and the answer quality jumps. **This is the cheapest accuracy lever in RAG**, and the strongest reason to put a vector DB in the system. Stage 5 will demonstrate this with a side-by-side comparison: same query, with and without the filter, look at the chunks returned.

You *can* do this in numpy. The five-line version is:

```python
mask    = np.array([m["ticker"] == "TSLA" for m in metadata])
indices = np.where(mask)[0]
sims    = embeddings[indices] @ q
top_local  = sims.argsort()[-5:][::-1]
top_global = indices[top_local]
```

Five lines, every one a place to introduce a gather/scatter indexing bug. In Chroma:

```python
collection.query(query_embeddings=[q], n_results=5, where={"ticker": "TSLA"})
```

One line. No index arithmetic. This is the abstraction the vector DB is buying you.

## When numpy *is* the right answer (calibration, not religion)

Pick numpy when:

- the corpus is small and **static** (not iterating on chunking/embedding/data sources)
- you need **no metadata filtering**
- everything fits in memory and persistence is "save a `.npy` file once"
- it's a **one-shot experiment** ("does dense retrieval beat keyword on this dataset?")

Reaching for a vector DB reflexively for every prototype is a junior reflex. Refusing to use one when you're building a system that will iterate, filter, and persist is also a junior reflex. The right answer is to know which spectrum you're on. We're building a system — second camp.

## Chosen design — single collection, ticker as metadata

**Collection name:** `filings` (single Chroma collection)

**Why one collection and not one per ticker:**

The tempting alternative is `tsla_filings`, `aapl_filings`, `nvda_filings`. Three reasons against it:

1. **Cross-company queries become awkward.** "Compare how Tesla and Apple describe supply chain risk" needs to query two collections, merge, re-rank. With a single collection and no `where` filter, this is one query.
2. **Adding a new ticker becomes a code change** instead of a data change. With ticker-as-metadata, adding NFLX is exactly the same operation as adding the third Tesla 10-K — push more chunks to the same collection.
3. **The HNSW index is built per-collection.** Three small indexes don't search faster than one slightly-larger index — they search a little slower because of overhead per call.

Metadata filtering gives us all the precision of separate collections with none of the orchestration cost.

## What each row will contain

```
id        : "AAPL-2025-10-31-0117"               (deterministic, from chunking.py)
document  : "Item 3.    Legal Proceedings\nDigital ..."  (the chunk text, verbatim)
embedding : float32[384]                          (L2-normalized BGE vector)
metadata  : {
  "ticker"            : "AAPL",
  "company_name"      : "Apple Inc.",
  "section"           : "Item 3. Legal Proceedings",
  "filing_date"       : "2025-10-31",
  "cik"               : "0000320193",
  "accession_number"  : "0000320193-25-000079",
  "source_url"        : "https://www.sec.gov/...",
  "chunk_index"       : 0,
  "char_start"        : 0,
  "char_end"          : 1058,
}
```

Every field has a reason to be there:

- **`id`** — primary key. Drives upsert. Stable across runs.
- **`document`** — what gets returned to the generator. Search never reconstructs text from the vector; the vector is search-key, the document is what we actually hand the LLM.
- **`embedding`** — the search key. Stored as float32 to halve the storage cost of float64 with no measurable accuracy loss.
- **`metadata.ticker / company_name / section / filing_date`** — the filter handles. Any of these can become a `where` clause at retrieval time.
- **`metadata.cik / accession_number / source_url`** — citation handles. Stage 6 will surface `source_url` directly to the user so they can verify any answer.
- **`metadata.chunk_index / char_start / char_end`** — debugging handles. "Which chunk was this, where in the section was it from?" — never used in retrieval, indispensable for diagnosing bad retrieval.

## Design decisions baked into the code (so you remember why later)

**The `VectorStore` class will wrap Chroma — same seam pattern as `Embedder`.**

The rest of the codebase will not touch Chroma's API directly. If we ever swap to Qdrant, Weaviate, or pgvector, we change `app/store.py` and nothing else. The interface will look approximately like:

```python
class VectorStore(ABC):
    def upsert_chunks(self, chunks: list[dict], embedder: Embedder) -> None: ...
    def query(self, query_text: str, k: int, where: dict | None = None) -> list[dict]: ...
    def count(self) -> int: ...

class ChromaVectorStore(VectorStore):
    # concrete implementation
```

**The embedder is injected, not constructed inside the store.**

`upsert_chunks(chunks, embedder)` takes an `Embedder` as a parameter rather than instantiating one internally. Same reason as everywhere else: the store shouldn't care which embedder produced the vectors. This also makes it trivial to test the store with a fake embedder.

**`upsert`, never `add`.**

Chroma exposes both. `add` raises on duplicate id; `upsert` overwrites. We always use `upsert` so re-runs are idempotent — the operational property that makes iterating on the pipeline pleasant rather than painful.

**Query return shape is normalized to a list of dicts.**

Chroma's raw response is awkward — it returns parallel arrays under nested keys (`{"ids": [[...]], "documents": [[...]], "distances": [[...]], "metadatas": [[...]]}`). Useful internally; ugly to use. The wrapper converts it to `[{"id": ..., "document": ..., "similarity": ..., "metadata": ...}, ...]` before returning. Every caller gets a clean shape.

**Cosine similarity is reported, not raw distance.**

Chroma returns `distance` (lower = closer); we transform to `similarity = 1 - distance` for L2-normalized cosine. This keeps the language consistent with `embedding-notes.md` — higher score means more similar, always, everywhere in the codebase.

**Persistence path is configurable.**

`config.chroma_path` resolves to `data/chroma/` by default. Test code can point to a tmp directory. The path is the only configuration the store needs beyond a collection name.

## Sanity-check plan — initial build & integrity check

Before trusting the store downstream we'll verify three properties:

1. **All chunks made it in.** `count()` returns 678 (TSLA 251 + AAPL 149 + NVDA 278).
2. **The ticker filter works as designed.** `query(..., where={"ticker": "TSLA"})` retrieves only TSLA chunks; counts per ticker via filtered queries match the chunking output.
3. **Rows are well-formed.** A spot-check of one row shows the embedding (length 384, finite floats), document (non-empty string), and metadata dict (all expected keys present).

If any of those fail, the storage layer is unfit and we stop until it's fixed. Retrieval against a broken index produces results that look real but aren't, which is the worst category of bug.

**Status:** implemented and run. All three sanity-check properties pass.

**Results — first build:**

```
Total rows in collection: 678
By ticker:    TSLA 251 | AAPL 149 | NVDA 278     (exact match to chunking Experiment 2)
Build time:   6.7 seconds  (embedding + upsert combined)
Disk size:    8.3 MB at data/chroma/             (~12 KB/row: vector + document + metadata + HNSW overhead)

Sample row (from inspect):
  id           : AAPL-2025-10-31-0000
  document len : 911 chars
  metadata     : {
    chunk_index      : 0
    company_name     : "Apple Inc."
    accession_number : "0000320193-25-000079"
    ticker           : "AAPL"
    char_start       : 0
    char_end         : 911
    filing_date      : "2025-10-31"
    source_url       : "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-..."
    section          : "Item 1. Business"
    cik              : "0000320193"
  }
```

**Verdict on the three sanity-check properties:**

1. **Total row count matches the chunker's output.** 678 in, 678 out. No dedupe collapse from duplicate ids, no silent loss.
2. **Ticker filter works.** `where={"ticker": "TSLA"}` returns 251, `AAPL` returns 149, `NVDA` returns 278. These match the per-file chunk counts exactly, which means metadata is being indexed and the `where` filter is wired through to the underlying scan. This is the foundation that Stage 5's `--company` filter will rest on.
3. **Rows are well-formed.** Sample row has all 10 expected metadata fields present, document text intact (911 chars, starts with the section header as designed), id matches the deterministic chunker format.

**Observations from the run:**

- **6.7 seconds end-to-end.** Embedding 678 chunks dominates; the actual upsert into Chroma is sub-second. This means re-runs during iteration are cheap — under 10 seconds from `python cli.py build` to a fresh index. Fast feedback loops matter for learning projects.
- **8.3 MB on disk for the persisted collection.** Most of that is the HNSW graph index. The raw vectors alone are 678 × 384 × 4 bytes = ~1 MB; documents + metadata add another ~1 MB; HNSW adds the remainder.
- **No errors, no warnings from Chroma itself.** The HuggingFace warnings (`HF_TOKEN` and `get_sentence_embedding_dimension` rename) are both upstream — one is an info nag, the other is the carry-forward TODO in `SESSION-STATE.md`.

This is the baseline. Stage 5 will exercise the `query()` path against it.

## Hypotheses for future experiments

- **Experiment 2 — hybrid retrieval (dense + BM25 / keyword).** Add a sparse keyword index alongside Chroma. Dense handles paraphrase; sparse handles exact identifiers ("Item 1A", "Section 7", specific GAAP terms). Combine scores with reciprocal rank fusion. Often a 5–15% precision lift on technical corpora.
- **Experiment 3 — cross-encoder rerank on top-50.** Retrieve 50 candidates by cosine, then re-rank using a cross-encoder (e.g. `bge-reranker-base`) that scores `(query, document)` pairs jointly. Slower per query, but often a meaningful precision boost.
- **Experiment 4 — section-aware sub-collections.** Test whether routing queries to a section-specific subset ("if the question mentions 'risk', filter `section LIKE '1A%'`") improves accuracy. Sometimes useful, sometimes a footgun — depends on how well users name sections.
- **Experiment 5 — embedding cache across runs.** Persist `(chunk_id → vector)` so re-builds re-use embeddings for unchanged chunks. Trivial when chunk ids are deterministic (which ours are). Matters once embedding costs are non-trivial (paid APIs, larger corpus).
- **Experiment 6 — pluggable backend.** Add `QdrantVectorStore` or `PgVectorStore` behind the same `VectorStore` interface. Confirms the abstraction holds; sometimes reveals quirks worth knowing (Chroma's metadata filtering syntax differs from Qdrant's, etc.).

## Lessons to carry forward

- **Persistence is the upgrade, not speed.** Anything a vector DB does, numpy *could* do at our scale. The DB earns its keep by making `(id, document, vector, metadata)` an atomic row, by making upsert trivial, and by making metadata filtering one kwarg instead of five lines of indexing arithmetic.
- **Pick the abstraction that matches your problem, not the one that matches today's scale.** 678 chunks doesn't need HNSW. Picking a vector DB anyway means the day we have 60,000 chunks isn't a code change, just more data.
- **The metadata schema is part of the API.** Anyone querying the store has to know what fields exist. Document it explicitly (the "What each row will contain" section above) so the contract is visible.
- **Wrap third-party APIs at the seam.** Chroma's `query()` return shape is awkward. Hiding it behind a normalized `list[dict]` means the awkwardness lives in one file, not in every caller.

## How to think about vector storage, generally

A vector store is a database optimised for one specific kind of lookup — nearest neighbour in a learned space — alongside the same boring concerns every database has (persistence, integrity, atomic updates, filtering, indexes). The hard parts of building one are the *boring* parts, not the ANN algorithm. That's why the practical choice when building RAG systems is almost never "should we write our own?" but "which production-grade one fits our deployment story?" — Chroma for embedded / local, Qdrant or Weaviate for self-hosted servers, pgvector if you already run Postgres, Pinecone or Turbopuffer for managed.

For a learning project, Chroma is the right call because it's embedded (no separate server to run), persists to local disk (no infrastructure setup), and exposes a Python API that maps cleanly to the four-tuple model. We'll outgrow it eventually — every team does — and the wrapper makes that outgrowing a one-file change.
