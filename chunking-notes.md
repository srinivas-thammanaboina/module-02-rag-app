# Chunking notes — module-02-rag-app

**Takeaway:** Chunking is the most underrated lever in a RAG pipeline. The size of a chunk decides what the embedding model "sees", and the *boundaries* of a chunk decide whether what it sees is a coherent idea or a fragment of garbage. A bad chunker will silently sabotage everything downstream: embeddings, retrieval, and generation will all run successfully and all produce wrong answers.

## Why fixed-size chunking is wrong

The simplest possible chunker — `text[i:i+1000]` for `i` stepped by 1000 — is wrong on filings because its cut points are decided by arithmetic, not by meaning. I tested this on TSLA's Item 1A (Risk Factors) and inspected three consecutive chunks at the seams:

| Chunk | Opens with | Closes with |
|---|---|---|
| 2 | *"e are not successful..."* | *"...processing power limitati"* |
| 3 | *"ons and the substantial..."* | *"...trade and shipping disru"* |
| 4 | *"ptions, port congestions..."* | *"...loss of access to important t"* |

Every cut lands inside a word. "limitations" gets split as `limitati` + `ons`, "disruptions" gets split as `disru` + `ptions`. None of these chunks is a self-contained thought.

## Why this destroys retrieval (not just aesthetics)

When chunk 2 is embedded, the vector represents the semantic content of the literal text it holds — *"e are not successful in achieving these goals... processing power limitati"*. That's partially gibberish and partially an incomplete thought. The embedding model produces a vector for it, but the vector is **not aligned with what the chunk is supposed to be about**. So when a user asks *"what power-related risks does Tesla face?"*, the matching chunk's vector is closer to "fragments of half-formed sentences" than to "power risks". Retrieval becomes a coin flip exactly when the user's question sits near a cut point — which is most questions, because cuts are everywhere.

There's a second compounding problem at generation time. Even when retrieval finds a roughly-correct chunk, passing broken text to the LLM means the model either (a) hedges its answer, (b) fabricates the missing context to fill the gap, or (c) cites the chunk verbatim, exposing the broken text to the user in the final citation. None of these are acceptable.

## The three principles of good chunking

Any chunker worth using must respect:

1. **Structure.** A chunk should never cut a paragraph mid-sentence if a paragraph boundary is nearby. The right order of preference for split points is: paragraph (`\n\n`), then line (`\n`), then sentence (`. ` / `? ` / `! `), then word (` `). Hard byte-offset cuts are a last resort.
2. **Size as a budget, not a target.** Each chunk should be *under* a maximum (around 1000 characters / 250 tokens for a small embedder) but otherwise as large as the natural structure allows. A 320-character paragraph stays as one chunk; a 1500-character paragraph gets split at its best internal sentence boundary.
3. **Overlap between chunks.** Each chunk includes the last ~150 characters of the previous chunk as a prefix. This means context isn't lost when an important idea straddles a boundary — the next chunk still has the lead-in. Overlap is a small storage cost (~15% redundancy) that buys substantial robustness against unlucky cut placement.

## Chosen design — paragraph-pack-then-overflow-split

The algorithm we'll implement:

1. Split the section text into paragraphs on `\n\n`.
2. Greedily pack paragraphs into a chunk until adding the next paragraph would exceed `max_size`.
3. If a single paragraph is larger than `max_size` on its own, recursively split it: try sentence boundaries first, then line breaks, then word boundaries, then a hard cut as the last resort.
4. After emitting each chunk, the next chunk starts by re-including the last `overlap` characters of the previous one — snapped backwards to the nearest sentence boundary so the overlap reads cleanly.
5. Drop any final fragment below `min_size` — those are usually trailing leftovers from end-of-section text.

This is essentially what LangChain's `RecursiveCharacterTextSplitter` does. We're implementing it ourselves (~60 lines) instead of importing it, so every decision is visible and tunable.

## Metadata propagation

Every chunk that goes into the vector store will carry the following metadata:

- `ticker`, `company_name`, `section`, `filing_date`
- `cik`, `accession_number`, `source_url`
- `chunk_index` — position within the section (0, 1, 2, …)
- `char_start`, `char_end` — exact offset back into the cleaned section text, so we can always locate a chunk's source slice

Metadata travels with the chunk. The retriever doesn't care about it; the generator uses it to attribute answers; the human uses it to verify the citation.

## Experiment 1 — paragraph-pack baseline

This is the first configuration we'll test. After running, paste retrieval observations underneath so we can compare future variants against it.

**Parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `max_size` (chars) | 1000 | ~250 tokens for `bge-small-en-v1.5`. Small enough to be a focused idea, big enough to contain one complete thought. |
| `overlap` (chars) | 150 | ~1–2 sentences of lead-in context. Trades ~15% index size for robustness across boundaries. |
| `min_size` (chars) | 200 | Drops trailing end-of-section fragments that would otherwise pollute retrieval. |
| Split preference | paragraph → sentence → line → word → hard cut | Respect structure before resorting to arbitrary offsets. |
| Embedder | `BAAI/bge-small-en-v1.5` (local) | 384-dim vectors, 512-token context window. Cheap to run locally and well-regarded on retrieval benchmarks. |

**Method choices being made deliberately:**

- Verbatim overlap (copy the actual previous-chunk text) rather than a "lead-in summary" — simpler, lossless, and how every production RAG system I'm aware of does it. ~25% smaller indexes are possible with summarized lead-ins but the complexity isn't worth it for a learning project.
- Writing the splitter ourselves rather than importing `langchain.text_splitter.RecursiveCharacterTextSplitter` — same algorithm, but every line is something I can reason about and change.

**Status:** designed; implementation pending.

**Expected output magnitude:**

Roughly 400 chunks across all three companies (~338K characters of cleaned section text at ~1000 chars/chunk with 150 chars of overlap).

**Results placeholder — fill in after testing:**

```
Total chunks: ___
By company:   TSLA ___ | AAPL ___ | NVDA ___
By section:   Item 1 ___ | Item 1A ___ | Item 3 ___ | Item 7 ___ | Item 7A ___
Chunk length: min ___ | median ___ | p95 ___ | max ___
Overlap actually achieved: median ___ chars
```

**Retrieval observations to capture (after Stage 5):**

- Sample query → top-k chunks returned → are they relevant?
- Where do similarity scores cluster? (0.6-0.7 is "loosely related"; 0.8+ is "clearly relevant")
- Are there obvious mis-retrievals where the top chunk is from the wrong section or wrong company?
- Do chunks read as self-contained thoughts when shown to a human grader?

**Hypotheses for future experiments to consider if Experiment 1 disappoints:**

- *Experiment 2 — smaller chunks (500/75):* might tighten retrieval precision for narrow questions like "what was Apple's Item 3 about" — but trades off context for the generator.
- *Experiment 3 — larger chunks (2000/300):* might help broad synthesis questions ("summarize Tesla's risk profile") — but risks pulling in irrelevant text within the same chunk.
- *Experiment 4 — section-aware chunking:* chunk by sub-headers inside long sections (e.g. each named risk inside Item 1A as its own chunk). Higher quality but requires sub-header detection — another regex layer.
- *Experiment 5 — semantic chunking:* split where sentence-level embedding similarity drops below a threshold. Theoretically cleanest but expensive to compute and harder to debug.

## How to think about chunking, generally

Every chunking decision is a tradeoff between **specificity** and **context**. Smaller chunks retrieve more precisely (the vector for a 200-character chunk is laser-focused on one idea) but offer the generator less to work with. Larger chunks offer richer context to the generator but dilute the embedding (a 3000-character chunk's vector is an averaged smear of multiple ideas). There's no universally right size — only a size right for your data, your typical queries, and your generator. Treat 1000/150 as a starting point, measure what happens, then move.
