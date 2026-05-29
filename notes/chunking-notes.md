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

1. **Structure.** A chunk should never cut a paragraph mid-sentence if a paragraph boundary is nearby. The right order of preference for split points is: paragraph, then sentence (`. ` / `? ` / `! `), then word (` `). Hard byte-offset cuts are a last resort. (How "paragraph" is encoded in your data depends on how the text was extracted — see the data observation below; for our 10-Ks it turned out to be a single `\n`, not `\n\n`.)
2. **Size as a budget, not a target.** Each chunk should be *under* a maximum (around 1000 characters / 250 tokens for a small embedder) but otherwise as large as the natural structure allows. A 320-character paragraph stays as one chunk; a 1500-character paragraph gets split at its best internal sentence boundary.
3. **Overlap between chunks.** Each chunk includes the last ~150 characters of the previous chunk as a prefix. This means context isn't lost when an important idea straddles a boundary — the next chunk still has the lead-in. Overlap is a small storage cost (~15% redundancy) that buys substantial robustness against unlucky cut placement.

## Chosen design — paragraph-pack-then-overflow-split

The algorithm we'll implement:

1. Split the section text into paragraphs on whatever the source's paragraph separator turns out to be (for our cleaned 10-Ks: a single `\n` — see "Data observation" below).
2. Greedily pack paragraphs into a chunk until adding the next paragraph would exceed `max_size`.
3. If a single paragraph is larger than `max_size` on its own, recursively split it: try sentence boundaries first, then word boundaries, then a hard cut as the last resort.
4. After emitting each chunk, the next chunk starts by re-including the last `overlap` characters of the previous one — snapped *forward* to the nearest sentence boundary inside the tail so the overlap text reads cleanly from a sentence start.
5. Drop any final fragment below `min_size` — those are usually trailing leftovers from end-of-section text.

This is essentially what LangChain's `RecursiveCharacterTextSplitter` does. We're implementing it ourselves (~120 lines including comments) instead of importing it, so every decision is visible and tunable.

## Data observation: paragraph separators in our cleaned text

Before writing the chunker we inspected the actual cleaned text in `data/clean/TSLA.json` and found something the "split on `\n\n`" plan glossed over: **our text contains only single newlines, never double.** Item 1A's 83,740 characters contains 172 newlines, every single one a run-length of exactly 1.

The cause traces back to Stage 1: `FilingParser.to_text()` uses `soup.get_text("\n")`, which joins each block-level HTML element with a single `\n`. There's no source of `\n\n` after that. Every "paragraph break" in our cleaned text is one `\n`, and that's also how sub-headings like *"Risks Related to Our Ability to Grow Our Business"* end up on their own line — separated by single `\n` from the body around them.

The packing algorithm doesn't care which character is the paragraph separator; it just needs to know what it is. So `SEPARATORS = ["\n", ". ", " "]` for us, with `"\n"` as the primary "paragraph" boundary. If we ever change `to_text()` to use a different joiner (or process a corpus where paragraphs really are `\n\n`-separated), this list is the only place that needs to change.

**Lesson worth carrying forward:** assumptions about text structure should be verified against your actual data before they get baked into algorithms. A 5-minute `python3 -c "..."` inspection of one file would have caught this before the first chunker run.

## Implementation decisions baked into the code

These are choices that don't show up in the design narrative but matter when you re-read the code in three months:

**Chunk IDs are deterministic: `TICKER-FILINGDATE-NNNN`.** For example, `AAPL-2025-10-31-0117`. Same filing chunked twice yields the same IDs. This matters at the Chroma stage — re-running the pipeline updates existing rows rather than duplicating them. The `NNNN` is a zero-padded global index across the whole filing (not per-section), so chunk numbers sort document-order across sections.

**Output format is JSONL (one chunk per line), not a single JSON array.** Three reasons. (a) Streamable — you can iterate without loading the whole file. (b) `grep`-able and `head`/`tail`-able for ad-hoc inspection. (c) Append-friendly if we ever want to incrementally add filings. Cost: tooling that expects regular JSON needs adapters, but every modern data tool I work with prefers JSONL for line-oriented records.

**`char_start` and `char_end` offsets are best-effort, not authoritative.** Each chunk is searched back into the original section text using `text.find(chunk_text, start_hint)`. If overlap stitching mutates the chunk's head enough that direct search fails, we fall back to searching for the chunk's *last 80 characters* (which are always verbatim from the source). In the rare case both fail, we record `(-1, -1)`. These offsets exist for human debugging — "where in the document did this chunk come from?" — never for retrieval. The chunk text itself is the source of truth.

**Trailing-scrap drop, not internal-scrap drop.** Only the *last* chunk of a section is checked against `min_size` for removal. Internal chunks are kept regardless of length. With the absorption-guard fix from Experiment 2, internal sub-min chunks shouldn't occur at all — but if they ever do, that's signal of a deeper bug, not noise to filter out. Keep them visible.

**The `_pack` joiner is `"\n"`.** When two atoms get concatenated into the same chunk, they're joined with `\n` (matching the source's paragraph separator). This means a re-read of a chunk preserves the paragraph structure of the original section. Important for the generator stage — citations will read naturally.

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

**Status:** implemented and run. **Two real defects found in the output — see "Observations" below. The numbers in the "Results" block are the buggy baseline, kept here as the diagnostic record. The fix is described after, and re-runs go under "Experiment 2".**

**Results — first run (with bugs present):**

```
Total chunks: 679
By company:   TSLA 251 | AAPL 150 | NVDA 278
By section:   Item 1   152  (TSLA  61, AAPL  23, NVDA  68)
              Item 1A  369  (TSLA 117, AAPL  94, NVDA 158)
              Item 3     8  (AAPL only)
              Item 7   138  (TSLA  71, AAPL  21, NVDA  46)
              Item 7A   12  (TSLA   2, AAPL   4, NVDA   6)
Chunk length: TSLA — min 236  | median 900 | p95 1118 | max 1147
              AAPL — min  63  | median 911 | p95 1123 | max 1149
              NVDA — min 226  | median 892 | p95 1116 | max 1149
Chunks over budget (> max_size=1000): TSLA 67 (26.7%) | AAPL 46 (30.7%) | NVDA 85 (30.6%)
Chunks under floor (< min_size=200):  TSLA 0 | AAPL 1 | NVDA 0
```

### Observations from the run

**Observation 1 — roughly 30% of chunks exceed the 1000-char budget.**

Across all three companies between 26.7% and 30.7% of chunks come in larger than `max_size`, with p95 at ~1118 and worst case at ~1149. This isn't a soft tolerance — it's a bug in the packing logic. See "Bug 1" below.

**Observation 2 — AAPL produced a 63-character chunk in Item 3.**

The chunk is just the section header and a sub-heading, with no body text attached:

```
chunk_id: AAPL-2025-10-31-0117
section : Item 3. Legal Proceedings
length  : 63 chars
text    : 'Item 3.\xa0\xa0\xa0\xa0Legal Proceedings\nDigital Markets Act Investigations'
```

This is too short to be a meaningful semantic unit. Embedding it produces a vector aligned with "section-header text" generally, not with anything answerable. Same root cause as Observation 1 — see "Bug 1".

**Observation 3 — tail previews in the CLI start mid-word. Not a chunk problem.**

The CLI's `_print_sample_chunk` slices `text[-160:]` for the tail preview without snapping to a word boundary. That's why the TSLA #2 sample shows:

```
tail: 'erience differentiate us from other companies. | Segment Information | ...'
```

"erience" is the back half of "experience" — the slice landed inside the word. Same in TSLA mid (`'ions, as part of which such unions have...'` — back half of "negotiations"), and in AAPL mid (`'s circumstances, which, among other things...'` — back half of "Apple's" or similar). The chunk *content* is intact; you can verify because the tail always ends on a period. Cosmetic display issue only. Will snap to a word boundary on the next CLI update.

### Bug 1 — overflow-on-reseed (cause of Observations 1 & 2)

When the packer's running buffer overflows on the next atom, the code emits the buffer and reseeds the next chunk as `overlap_text + "\n" + atom`. Two flaws:

1. If the running buffer is itself tiny (say 63 chars from a section header plus a sub-heading), the code still emits it as a complete chunk. That's how the AAPL Item 3 fragment escaped.
2. The reseed includes overlap unconditionally. If the incoming atom is already close to `max_size` (which it can be, since the atomize step guarantees atoms ≤ max_size but allows them to reach max_size), then `overlap (150) + atom (≈950) = 1100 chars`. The new buffer starts *over budget*, and nothing in the loop ever brings it back down. This is why p95 length sits around 1118.

Both observations share this single root cause.

### Fix applied (going into Experiment 2)

Two guards in `_pack`:

1. **Absorption guard.** When the next atom would overflow but the running buffer is below `min_size`, the atom is absorbed anyway and the chunk is allowed to overshoot slightly. Better one mildly-oversized chunk than a sub-200 scrap plus another over-budget chunk.
2. **Budget-checked reseed.** After a legitimate emit, the overlap prefix is only included on the new chunk if `len(overlap) + len(atom) ≤ max_size`. Otherwise the new chunk starts at the atom itself with no overlap. This makes overlap best-effort: it's there in 95%+ of cases, but never at the cost of correctness.

These changes shift my earlier "we keep internal short chunks because they reflect real structure" claim. That rationalization was masking the bug — internal sub-min chunks in this run were artifacts of the broken packer, not real structure. After the fix, internal sub-min chunks should not appear.

### Retrieval observations to capture (after Stage 5, against Experiment 2 output)

- Sample query → top-k chunks returned → are they relevant?
- Where do similarity scores cluster? (0.6-0.7 is "loosely related"; 0.8+ is "clearly relevant")
- Are there obvious mis-retrievals where the top chunk is from the wrong section or wrong company?
- Do chunks read as self-contained thoughts when shown to a human grader?

## Experiment 2 — paragraph-pack baseline with packer fix

Same parameters as Experiment 1 (`max_size=1000`, `overlap=150`, `min_size=200`). The only change is the two guards in `_pack`.

**Expected effects of the fix vs Experiment 1:**

- Chunks over `max_size`: should drop from ~30% to near 0%, with at most a small absorption-guard tail.
- 63-char AAPL Item 3 fragment: gone. Item 3 chunk count drops from 8 → ~6.
- Total chunk count: slight reduction overall (~5–15%), since the buggy fragmentation produced extra splits.
- Median length: holds in the high 800s / low 900s.

**Results — second run (with packer fix):**

```
Total chunks: 678   (Experiment 1 was 679 — Δ -1, from AAPL Item 3 fragment merge)
By company:   TSLA 251 | AAPL 149 | NVDA 278
By section:   Item 1   152
              Item 1A  369
              Item 3     7   (AAPL only; was 8 in Experiment 1)
              Item 7   138
              Item 7A   12

Chunk length: TSLA — min 236  | median 887 | p95 989 | max 1000
              AAPL — min 202  | median 891 | p95 994 | max 1058
              NVDA — min 226  | median 882 | p95 994 | max 1000

Chunks over budget (> 1000 chars): TSLA  0 | AAPL  1 | NVDA  0   (was 67 / 46 / 85)
Chunks under floor (<  200 chars): TSLA  0 | AAPL  0 | NVDA  0   (was  0 /  1 /  0)
```

### What the fix delivered

- **Over-budget rate dropped from ~30% to 0.15% (1 / 678).** The single 1058-char outlier is the AAPL Item 3 case where the absorption guard intentionally fired — a section header plus sub-heading combined buffer was below `min_size`, so the next atom was absorbed even though it overshot the budget by 58 chars. That's the correct trade-off: one mildly oversized chunk is strictly better than a 63-char scrap plus a 1014-char overshoot, which is what Experiment 1 produced for the same content.
- **The AAPL Item 3 sub-200 fragment is gone.** Item 3's first chunk now contains the section header, the first sub-heading, and the first body paragraph as one ~1058-char unit. Section count for Item 3 dropped from 8 → 7 accordingly.
- **Median length held steady (882–891 chars across companies).** The fix didn't shrink chunks on average; it just stopped the small-fragment-plus-overshoot pathology.
- **Minimums are now meaningfully above the 200-char floor (lowest is 202 in AAPL).** No internal scrap chunks remain.

### What Experiment 2 didn't change

- **TSLA Item 7A still produces only 2 chunks for 1,625 chars of content.** That section is barely above the floor; not much we can do at this chunk-size without dipping below it.
- **The CLI tail-preview is still cropped mid-word.** Cosmetic display issue carried forward — still to fix when next touching `_print_sample_chunk`.
- **NVDA Item 7A is short relative to NVIDIA's actual exposure (6 chunks).** This reflects the source filing's brevity in Item 7A, not the chunker's behavior.

Experiment 2 is the baseline we'll use going forward into Stage 3 (embedding). Retrieval observations will be captured against this output.

## Hypotheses for future experiments (if Experiment 2 still underperforms in retrieval)

- *Experiment 3 — smaller chunks (500/75):* might tighten retrieval precision for narrow questions like "what was Apple's Item 3 about" — but trades off context for the generator.
- *Experiment 4 — larger chunks (2000/300):* might help broad synthesis questions ("summarize Tesla's risk profile") — but risks pulling in irrelevant text within the same chunk.
- *Experiment 5 — section-aware chunking:* chunk by sub-headers inside long sections (e.g. each named risk inside Item 1A as its own chunk). Higher quality but requires sub-header detection — another regex layer.
- *Experiment 6 — semantic chunking:* split where sentence-level embedding similarity drops below a threshold. Theoretically cleanest but expensive to compute and harder to debug.

## How to think about chunking, generally

Every chunking decision is a tradeoff between **specificity** and **context**. Smaller chunks retrieve more precisely (the vector for a 200-character chunk is laser-focused on one idea) but offer the generator less to work with. Larger chunks offer richer context to the generator but dilute the embedding (a 3000-character chunk's vector is an averaged smear of multiple ideas). There's no universally right size — only a size right for your data, your typical queries, and your generator. Treat 1000/150 as a starting point, measure what happens, then move.
