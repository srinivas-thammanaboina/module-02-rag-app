# Build: Citation-Grounded Q&A Copilot over SEC 10-K Filings

## What I'm building and why
A RAG pipeline that answers questions about companies' 10-K filings (e.g. "What are
Tesla's biggest stated risks this year?") where EVERY claim in the answer cites back
to the exact filing section it came from, with a clickable source. This is a learning
project — I need to understand and verify each stage, not just get a working black box.

## Hard requirements (non-negotiable)
1. **Citations are the whole point.** Every sentence in a generated answer must be
   traceable to a specific chunk, and each chunk must carry metadata: company name,
   ticker, CIK, filing type, filing date, section (e.g. "Item 1A. Risk Factors"),
   and the source EDGAR URL. The final answer must surface these so I can click and verify.
2. **Every stage prints its intermediate output.** When I chunk, I see the chunks.
   When I embed, I see vector dimensions + a sample similarity score. When I retrieve,
   I see which chunks came back AND their similarity/distance scores AND why. When I
   generate, I see the final assembled prompt before it's sent. Make each stage runnable
   on its own from the CLI.
3. **The embedding step lives behind a single interface** (one class/function) so I can
   swap the local model for an API later by changing one place. Default to a local
   sentence-transformers model.

## Stack
- **Embeddings:** local via `sentence-transformers`. Pick a sensible default model
  (something like `BAAI/bge-small-en-v1.5`) but isolate it behind an `Embedder`
  interface with an `embed(texts: list[str]) -> list[vector]` method. Document how to
  swap in an API embedder.
- **Vector DB:** Chroma, persisted to local disk.
- **Generation:** [tell it which model/SDK you use here — e.g. Anthropic, OpenAI].
- **Language:** Python. Use a virtual env. Pin dependencies in requirements.txt.

## Data ingestion — SEC EDGAR (read carefully, this is where it goes wrong)
Pull filings LIVE from EDGAR. The SEC API has specific gotchas:
- **A `User-Agent` header is mandatory** on every request, formatted like
  "Name email@example.com". Requests without it are blocked. Make it configurable.
- **Respect the rate limit** (max ~10 requests/sec). Add a small delay between requests.
- To find a company's filings: resolve ticker → CIK (the company_tickers.json mapping
  from SEC), then hit the submissions endpoint at
  `https://data.sec.gov/submissions/CIK{10-digit-zero-padded}.json` to list filings.
- Filter to the most recent 10-K. The actual document is HTML hosted on
  `www.sec.gov/Archives/...`.
- **10-K HTML is inline-XBRL and very messy** — strip tags, scripts, and XBRL markup
  down to clean text before chunking. Preserve enough structure to identify the major
  Items (Item 1A Risk Factors, Item 7 MD&A, etc.) because section is required citation
  metadata.
- Start with 2-3 companies I specify (I'll use Tesla, Apple, and one more). Make the
  ticker list configurable, not hardcoded into logic.

## Chunking
- Use recursive/structure-aware chunking — split on the filing's natural boundaries
  (Items, then paragraphs) before falling back to a fixed size with overlap.
- Each chunk must retain its section label and source metadata through the whole pipeline.
- Make chunk size and overlap configurable so I can experiment and SEE the effect.

## Retrieval & answer
- Embed the question, retrieve top-k similar chunks from Chroma, support metadata
  filtering (e.g. restrict to one company).
- Assemble the retrieved chunks into a prompt that instructs the model to answer ONLY
  from the provided chunks and to cite each claim by chunk/section. If the chunks don't
  contain the answer, say so rather than inventing.
- Print the retrieved chunks with scores, then the assembled prompt, then the answer
  with its citations.

## Project structure
Organize it cleanly into separate modules per stage (ingest, chunk, embed, store,
retrieve, generate) plus a CLI entry point. I want to run stages independently.

## Deliverables
- The working pipeline.
- A `WHY.md` (short) explaining the key design choices: why this chunking strategy,
  why these metadata fields, how the citation flow works end to end, and what the
  tradeoffs of the local embedding model are vs. an API.
- A README with setup + how to run each stage.

## What NOT to do
- Don't add hybrid search or re-ranking yet — that's a deliberate later step. Get clean
  dense retrieval + citations working first.
- Don't hide the internals behind helpful abstractions. I want to see the mechanism.
- Don't hardcode the User-Agent, tickers, model name, chunk size, or top-k — config them.
