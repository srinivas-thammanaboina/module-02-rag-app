# Ingest observations — how to inspect your data critically

**Takeaway:** Your first ingestion will almost always look "fine" in the summary print but be quietly broken in the artifacts. Build the habit of laying the numbers from multiple inputs side-by-side and asking *"does this size make sense for what this section is supposed to contain?"* before you trust any of it.

## What "looking" at the data means here

After running Stage 1 for TSLA, AAPL, and NVDA, I pulled the section sizes into one table:

| Section | TSLA | AAPL | NVDA |
|---|---|---|---|
| Item 1 (Business) | 45,455 | 16,053 | 40,702 |
| Item 1A (Risk Factors) | 83,740 | **2,054** | 114,916 |
| Item 3 (Legal) | missing | 5,401 | missing |
| Item 7 (MD&A) | 46,781 | **7,107** | **8,139** |
| Item 7A (Market Risk) | 1,625 | 3,023 | 977 |

The bolded cells are the signal. Apple's Risk Factors in 2 KB? NVIDIA's MD&A in 8 KB after a record AI year? Those numbers fail a basic sniff test. **The lesson is to develop that sniff test before you have it** — by knowing roughly what each section *should* be sized like for the kind of company you're working with.

## How I caught it

1. **Numbers first.** I dumped just `(section name, char count)` per company. A single ticker would have looked fine. Three side-by-side made the anomaly screamingly obvious.
2. **Then the openers.** For every short section I printed the first 200 characters. The opening words of a 10-K section body are formulaic — "Risk Factors\nYou should carefully consider the risks...". When you see *"Item 7 of the Company's Annual Report on Form 10-K for the fiscal year ended..."* — that's prose, not a header. It's the section referencing itself from inside another section.
3. **Then the diagnosis.** Once I saw the same back-reference pattern in AAPL Item 1A, AAPL Item 7, and NVDA Item 7, the cause was clear: our regex matches *any* "Item N" string, including ones embedded in sentences. The "longest block" heuristic then picks whichever fragment of broken text happens to be biggest — often the reference, not the body.

## The general inspection recipe

When ingesting any new corpus, before you chunk:

1. **Tabulate sizes by document × section.** Numbers reveal anomalies that prose summaries hide.
2. **Print the first ~200 chars of every section.** Real bodies have stereotyped openers; references and TOC entries don't.
3. **Print the last ~200 chars.** Tells you whether you're cutting cleanly or trailing into page numbers, "Table of Contents" markers, or footnotes.
4. **Compare against your expectation of the source.** For 10-Ks: Risk Factors should be tens of KB, often 100+ KB for large companies. Market Risk is usually short. MD&A is always substantial.
5. **Look for patterns across documents.** A single short section might be the company's quirk. Three short sections across three companies is your heuristic failing.

## The 10-K-specific quirks I now know to watch for

- **Every Item header appears at least twice** — once in the table of contents (short, with a page number after it), once at the actual body. The "keep the longest occurrence" trick handles the TOC case.
- **Companies frequently reference other Items in prose** — e.g. *"as discussed in Item 7 of our Annual Report..."*. A regex that only looks for the literal string "Item N" will treat these as headers and split the document at the wrong places.
- **Reliable header signal:** the literal Item number is *immediately followed by the section's title text*. "Item 1A. Risk Factors" — yes. "Item 1A of this Form" — no. Prepositions after the Item number (*in*, *of*, *under*) are a near-perfect tell that you're looking at a reference, not a header.
- **Item 3 (Legal Proceedings) is often very short or absent.** Many companies stuff legal disclosures into a Notes-to-Financial-Statements reference. Don't assume every Item exists.
- **Modern 10-Ks are inline-XBRL** (XML with HTML attributes). BeautifulSoup with `lxml` HTML parser handles it but emits a warning. Cosmetic — filter or switch to `features="xml"`.
- **Encoding artifacts to expect:** non-breaking spaces (`\xa0`), curly quotes (`"…"`), page numbers trailing at section ends, *"Table of Contents"* phrases bleeding in. Plan to normalize these at the chunking step.

## Mental model — bad data → garbage everywhere

Bad ingestion is the most expensive bug in a RAG pipeline because the failure mode is silent. Embeddings will compute fine, Chroma will store fine, retrieval will return *something*, generation will produce confident-looking answers — and all of it will be wrong because the underlying chunks weren't what you thought they were. Every cleaning step belongs at the layer where the problem originates. If the chunking stage is the first time you notice your sections are broken, you're already three stages too late.

## Reusable diagnostic snippet

For any future ingest, this is the minimum-viable inspection:

```python
import json
d = json.load(open("data/clean/<TICKER>.json"))
for s in d["sections"]:
    txt = s["text"]
    print(f"{s['section']:60} {len(txt):>7} chars")
    print(f"  START: {repr(txt[:200])}")
    print(f"  END:   {repr(txt[-200:])}")
    print()
```

Run it for every document. Look at the table of sizes. Read the openers. Trust nothing until both pass the sniff test.

## How the header regex evolved (three iterations)

Text-parsing heuristics get refined iteratively against failure modes you find in the data. The final regex for identifying a real Item header in a 10-K layered **three filters**, each added in response to seeing a specific class of false match:

**Iteration 1 — too permissive:**
```python
re.compile(r"\bItem\s+(\d{1,2}[A-Z]?)\b\.?", re.IGNORECASE)
```
Matches every occurrence of "Item N" anywhere. Result: back-references in MD&A prose (*"as discussed in Item 7..."*) become false boundaries, chopping the real body into small pieces. AAPL Item 1A came out at 2KB instead of 70KB.

**Iteration 2 — added uppercase-letter-after-period:**
```python
re.compile(r"\bItem\s+(\d{1,2}[A-Z]?)\b\.?\s+(?=[A-Z\[])", re.IGNORECASE)
```
The idea: real headers are followed by an uppercase title word ("Business", "Risk Factors"); back-references are followed by lowercase prepositions ("in", "of", "under"). Looked correct on paper. Result: AAPL Item 1A fixed, but AAPL/NVDA Item 7 still broken.

**Trap I hit:** `re.IGNORECASE` makes character classes case-insensitive too. `[A-Z]` under IGNORECASE matches lowercase letters as well — so the filter didn't actually filter. Dropped IGNORECASE and enumerated the two casings (`Item` / `ITEM`) explicitly.

**Iteration 3 — added line-start anchor:**
```python
re.compile(
    r"^\s*(?:Item|ITEM)\s+(\d{1,2}[A-Za-z]?)\b\.?\s+(?=[A-Z\[])",
    flags=re.MULTILINE,
)
```
Even iteration 2 missed back-references that quoted the full section name in prose — *`refer to "Item 1A. Risk Factors,"...`*. The lookahead saw an uppercase "R" and accepted. The signal that distinguishes these from real headers: real headers sit at the **start of a line**; quoted references are mid-sentence. Adding `^` with `re.MULTILINE` killed them. NVDA Item 7 came back from missing entirely to 34KB.

## The pattern, generalized

When parsing a noisy text corpus with regex:
1. **Start permissive.** Your first regex will be wrong but quick to write.
2. **Inspect the failures.** Run the regex against real data, look at what it falsely matches *and* what it falsely rejects. Don't reason about which cases exist — *enumerate them from the data*.
3. **Layer filters, don't rewrite.** Each iteration adds one more signal. Three filters that each remove one class of false positive are far easier to debug than one giant regex that tries to do everything.
4. **Watch for flag interactions.** Most regex bugs in iteration 2 weren't logic bugs — they were `re.IGNORECASE` + `[A-Z]` doing something I didn't expect. When in doubt, write the cases you mean.

## Why this is "good enough" for a learning project

The final regex is brittle for atypical filings (this is *not* general-purpose 10-K parsing — a smaller-cap company that formats headers unusually would fail). A production system would either consume SEC's structured XBRL data directly or maintain explicit per-Item title patterns and expand them as new filings appear. For three known tickers, three layered heuristics is sufficient, and the cost of being wrong is visible (a missing section in the output table) rather than silent.

## Final section sizes across all three companies

| Section | TSLA | AAPL | NVDA |
|---|---|---|---|
| Item 1 (Business) | 45,455 | 16,053 | 48,241 |
| Item 1A (Risk Factors) | 83,740 | 68,047 | 114,916 |
| Item 3 (Legal) | — | 5,401 | — |
| Item 7 (MD&A) | 55,454 | 18,020 | 34,154 |
| Item 7A (Market Risk) | 1,625 | 3,023 | 4,253 |

TSLA and NVDA file legal proceedings as a cross-reference to financial-statement notes; the body is one sentence and falls below our 500-char floor. This is real, not a bug.
