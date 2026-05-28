"""
Stage 1: Ingest a company's latest 10-K from SEC EDGAR.

Pipeline for one ticker:
    ticker -> CIK -> latest 10-K submission -> filing HTML
           -> stripped clean text with Item-section markers preserved
           -> saved as JSON for downstream stages

Output (data/clean/<TICKER>.json):
    {
      "ticker": "TSLA",
      "cik": "0001318605",
      "company_name": "Tesla, Inc.",
      "filing_type": "10-K",
      "filing_date": "2025-01-30",
      "accession_number": "...",
      "source_url": "https://www.sec.gov/Archives/...",
      "sections": [
          {"section": "Item 1A. Risk Factors", "text": "..."},
          {"section": "Item 7. MD&A",          "text": "..."},
          ...
      ]
    }

Three classes, each with one job:
  EDGARClient   — HTTP wrapper. Handles User-Agent + rate limit (the two
                  things SEC blocks you for getting wrong).
  FilingFetcher — Resolves ticker -> CIK -> latest 10-K, downloads HTML.
                  Uses a disk cache so we don't re-hammer EDGAR.
  FilingParser  — Cleans HTML (10-Ks are inline-XBRL, very noisy) and splits
                  into sections by Item number. This is where the
                  "every Item appears twice (TOC + body)" quirk is handled.
"""

from __future__ import annotations

import json
import re
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from app.config import config

# 10-K filings are XHTML/inline-XBRL — technically XML. BS4's HTML parser
# (lxml) handles them fine but emits a noisy warning every time. Silence it
# here so CLI output stays readable.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# ---------------------------------------------------------------------------
# EDGARClient — HTTP layer
# ---------------------------------------------------------------------------

class EDGARClient:
    """Minimal SEC EDGAR HTTP client.

    The two SEC-specific rules this class enforces:
      1. Every request carries a User-Agent header (mandatory; missing it
         returns 403). We pull it from config and refuse to send a placeholder.
      2. We sleep `rate_limit_delay` seconds between requests so we stay
         under SEC's ~10 req/s ceiling. A naive script that fires requests
         in a loop will get IP-blocked.
    """

    def __init__(self, user_agent: str, rate_limit_delay: float):
        self.user_agent = user_agent
        self.rate_limit_delay = rate_limit_delay
        self._last_request_at: float = 0.0

    def _throttle(self):
        # Compute how long ago the last request was; sleep the remainder.
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_at = time.monotonic()

    def get(self, url: str) -> requests.Response:
        self._throttle()
        # 'Accept-Encoding: gzip' helps with EDGAR's large JSON files.
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp

    def get_json(self, url: str) -> dict:
        return self.get(url).json()


# ---------------------------------------------------------------------------
# FilingFetcher — ticker -> CIK -> latest 10-K -> HTML
# ---------------------------------------------------------------------------

@dataclass
class FilingInfo:
    """Everything we need to identify and cite one filing."""
    ticker: str
    cik: str                  # 10-digit zero-padded, e.g. "0001318605"
    company_name: str
    filing_type: str          # "10-K"
    filing_date: str          # "YYYY-MM-DD"
    accession_number: str     # "0001628280-25-003063" (with dashes)
    primary_doc: str          # filename of the main HTML doc inside the filing
    source_url: str           # full clickable URL to the primary HTML doc


class FilingFetcher:
    """Resolves ticker to the latest 10-K filing and downloads it.

    Why caching: the 10-K HTML can be ~10MB and EDGAR rate-limits aggressively.
    We cache by accession_number under data/raw/ so re-runs are instant and
    we don't keep hammering SEC during development.
    """

    # SEC's master ticker->CIK mapping. ~13k entries; updates daily.
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    def __init__(self, client: EDGARClient, raw_dir: Path):
        self.client = client
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._ticker_map: dict[str, dict] | None = None

    def _load_ticker_map(self) -> dict[str, dict]:
        """Lazy-load and cache the ticker->CIK mapping for the session."""
        if self._ticker_map is None:
            data = self.client.get_json(self.TICKERS_URL)
            # The JSON is keyed by integer index, not ticker. Re-index by ticker.
            self._ticker_map = {entry["ticker"]: entry for entry in data.values()}
        return self._ticker_map

    def ticker_to_cik(self, ticker: str) -> tuple[str, str]:
        """Returns (10-digit padded CIK, company name)."""
        ticker = ticker.upper()
        mapping = self._load_ticker_map()
        if ticker not in mapping:
            raise ValueError(f"Ticker {ticker!r} not found in SEC ticker map.")
        entry = mapping[ticker]
        cik_padded = str(entry["cik_str"]).zfill(10)
        return cik_padded, entry["title"]

    def latest_10k(self, ticker: str) -> FilingInfo:
        """Find the most recent 10-K filing for this ticker.

        The submissions endpoint returns the company's recent filings as
        parallel arrays (accessionNumber[i], form[i], filingDate[i], ...).
        We find the first index where form == "10-K".
        """
        cik, company_name = self.ticker_to_cik(ticker)
        sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        sub = self.client.get_json(sub_url)
        recent = sub["filings"]["recent"]

        # Walk the parallel arrays in order; entries are newest-first.
        idx = next(
            (i for i, form in enumerate(recent["form"]) if form == "10-K"),
            None,
        )
        if idx is None:
            raise ValueError(f"No 10-K found in recent filings for {ticker}.")

        accession = recent["accessionNumber"][idx]        # e.g. "0001628280-25-003063"
        accession_nodash = accession.replace("-", "")     # used in URL path
        primary_doc = recent["primaryDocument"][idx]
        filing_date = recent["filingDate"][idx]

        # CIK in the Archives URL is the *unpadded* integer form.
        cik_int = int(cik)
        source_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
            f"{accession_nodash}/{primary_doc}"
        )

        return FilingInfo(
            ticker=ticker.upper(),
            cik=cik,
            company_name=company_name,
            filing_type="10-K",
            filing_date=filing_date,
            accession_number=accession,
            primary_doc=primary_doc,
            source_url=source_url,
        )

    def fetch_html(self, info: FilingInfo) -> str:
        """Download the filing HTML, caching to disk by accession number."""
        cache_path = self.raw_dir / f"{info.ticker}-{info.accession_number}.html"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="replace")

        resp = self.client.get(info.source_url)
        html = resp.text
        cache_path.write_text(html, encoding="utf-8")
        return html


# ---------------------------------------------------------------------------
# FilingParser — HTML -> clean text -> sections
# ---------------------------------------------------------------------------

class FilingParser:
    """Cleans 10-K HTML and segments it by Item (Item 1, 1A, 7, 7A, etc.).

    Two important real-world quirks this handles:

    1. **10-K HTML is inline-XBRL.** Every number is wrapped in <ix:nonfraction>
       tags with namespaced attributes. BeautifulSoup + `.get_text()` flattens
       all of that down to readable prose; we also drop <script>/<style>.

    2. **"Item N" appears in three contexts; only one is a real header.**
       A 10-K mentions "Item N" in: (a) the TOC, (b) the actual section body,
       (c) inline prose like "as discussed in Item 7 of our prior 10-K...".
       A loose `\bItem N\b` regex matches all three, and our split then
       breaks real section bodies whenever a back-reference appears mid-body.
       This was the bug that left AAPL Item 1A at 2KB and NVDA Item 7 at 8KB
       in the first version of this parser.

       The fix is a tighter header pattern: a real header is followed by an
       UPPERCASE letter or "[" — the start of a section title like "Business"
       or "[Reserved]". A back-reference is followed by lowercase prepositions
       ("in", "of", "under", "as"). The regex below requires the uppercase
       tell, which excludes every back-reference seen in practice.

       TOC entries still match (they ARE real headers, just listed). So we
       keep the "longest candidate per item" trick to drop them — the body
       is always far longer than a one-line TOC entry.
    """

    # Sections we care about for citation. Anything else is dropped to keep
    # the index focused. (Easy to expand later.)
    TARGET_SECTIONS = {
        "1": "Item 1. Business",
        "1A": "Item 1A. Risk Factors",
        "3": "Item 3. Legal Proceedings",
        "7": "Item 7. Management's Discussion and Analysis",
        "7A": "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
    }

    # A real Item header is identified by THREE stacked signals:
    #
    #   (a) Line-start (`^` with re.MULTILINE) — real headers sit at the
    #       start of a line after a paragraph break in the rendered text.
    #       Back-references in prose like "...refer to 'Item 1A. Risk
    #       Factors,' our Consolidated..." are mid-sentence; this anchor
    #       excludes them.
    #
    #   (b) Followed by an UPPERCASE letter or "[" — i.e. the start of a
    #       section title like "Business", "Risk Factors", or "[Reserved]".
    #       Back-references like "Item 7 in our Annual Report..." are
    #       followed by lowercase prepositions; this lookahead excludes them.
    #
    #   (c) Casing enumerated, not via re.IGNORECASE — that flag would make
    #       `[A-Z]` match lowercase too and re-admit every back-reference.
    #
    # The TOC matches with all three of these (it IS a real header listing).
    # That's fine — the "longest candidate per item" rule downstream drops
    # TOC entries because the actual body is vastly longer.
    ITEM_HEADER_RE = re.compile(
        r"^\s*(?:Item|ITEM)\s+(\d{1,2}[A-Za-z]?)\b\.?\s+(?=[A-Z\[])",
        flags=re.MULTILINE,
    )

    def to_text(self, html: str) -> str:
        """Strip HTML/XBRL down to plain text while preserving paragraph breaks."""
        soup = BeautifulSoup(html, "lxml")
        # Remove script/style blocks completely — they contain no readable content.
        for tag in soup(["script", "style"]):
            tag.decompose()
        # `separator="\n"` keeps block boundaries; `strip=True` trims whitespace
        # at each text node.
        text = soup.get_text(separator="\n", strip=True)
        # Collapse runs of blank lines (XBRL HTML produces a LOT of these).
        text = re.sub(r"\n{2,}", "\n\n", text)
        return text

    def split_into_sections(self, text: str) -> list[dict]:
        """Slice the cleaned text at every Item header occurrence, then for each
        target section keep the LONGEST candidate (= body, not TOC link).

        Returns a list of {"section": str, "text": str}.
        """
        # 1. Find positions of every Item header occurrence.
        matches = list(self.ITEM_HEADER_RE.finditer(text))
        if not matches:
            return []

        # 2. Build candidate (item_id, body_text) pairs from one header to the next.
        candidates: dict[str, list[str]] = {}
        for i, m in enumerate(matches):
            item_id = m.group(1).upper()
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            candidates.setdefault(item_id, []).append(body)

        # 3. For each target section, keep the longest candidate.
        sections: list[dict] = []
        for item_id, label in self.TARGET_SECTIONS.items():
            blocks = candidates.get(item_id, [])
            if not blocks:
                continue
            body = max(blocks, key=len)
            # Drop sections that are still tiny — those are TOC entries
            # with no real body (e.g. company doesn't file under that item).
            if len(body) < 500:
                continue
            sections.append({"section": label, "text": body})
        return sections


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def ingest_ticker(ticker: str) -> dict:
    """End-to-end Stage 1 for a single ticker. Returns the same dict that
    gets persisted to data/clean/<TICKER>.json."""

    user_agent = config.require_sec_user_agent()
    client = EDGARClient(user_agent=user_agent, rate_limit_delay=config.sec_rate_limit_delay)
    fetcher = FilingFetcher(client=client, raw_dir=config.raw_dir)
    parser = FilingParser()

    info = fetcher.latest_10k(ticker)
    html = fetcher.fetch_html(info)
    full_text = parser.to_text(html)
    sections = parser.split_into_sections(full_text)

    result = {**asdict(info), "sections": sections}

    # Persist clean output for downstream stages.
    config.clean_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.clean_dir / f"{info.ticker}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_cli(args) -> None:
    """CLI entry point: ingest one ticker and print a summary so we can
    visually verify each step worked."""
    print(f"\n[ingest] ticker = {args.ticker}\n")
    result = ingest_ticker(args.ticker)

    print(f"  company      : {result['company_name']}")
    print(f"  CIK          : {result['cik']}")
    print(f"  filing_type  : {result['filing_type']}")
    print(f"  filing_date  : {result['filing_date']}")
    print(f"  accession    : {result['accession_number']}")
    print(f"  source_url   : {result['source_url']}")
    print(f"  sections     : {len(result['sections'])} kept")
    for s in result["sections"]:
        preview = s["text"][:120].replace("\n", " ")
        print(f"    - {s['section']:55s} | {len(s['text']):>7,} chars | {preview!r}...")
    print(f"\n  saved -> data/clean/{result['ticker']}.json")
