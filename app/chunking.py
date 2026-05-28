"""
Stage 2: Structure-aware chunking.

Reads a parsed filing from `data/clean/<TICKER>.json`, splits each section into
chunks that respect natural boundaries (line, sentence, word), and writes the
result as JSONL to `data/chunks/<TICKER>.jsonl`. Each chunk carries the full
metadata trail (ticker, section, source_url, etc.) so downstream stages can
cite back to the source.

See `chunking-notes.md` for the design rationale and Experiment 1's parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.config import config


# ---------------------------------------------------------------------------
# The chunker itself
# ---------------------------------------------------------------------------

# Separator hierarchy for the recursive overflow-split step. Ordered most
# semantically meaningful first. NOTE: our cleaned 10-K text uses single
# newlines as paragraph/heading boundaries (BS4's get_text("\n") joins
# block-level elements with a single \n, not \n\n) — so "\n" is the
# paragraph separator for us. See chunking-notes.md for the data observation.
SEPARATORS: list[str] = ["\n", ". ", " "]


@dataclass
class RecursiveChunker:
    """
    Paragraph-pack-then-overflow-split chunker.

    Two phases per section:
      1. Atomize: break the text into pieces that are each <= max_size.
         Long paragraphs get recursively split on the separator hierarchy:
         line ('\\n') -> sentence ('. ') -> word (' ') -> hard cut.
      2. Pack: greedily concatenate atoms into chunks under max_size.
         Each new chunk is prefixed with the tail of the previous one
         (snapped to a sentence boundary) for cross-boundary context.

    Chunks shorter than min_size are dropped at the end — these are
    usually trailing scraps that pollute retrieval more than they help.
    """

    max_size: int
    overlap: int
    min_size: int

    # ---- public ----

    def split(self, text: str) -> list[str]:
        """Split a section's text into chunk strings (no metadata yet)."""
        if not text or not text.strip():
            return []
        atoms = self._atomize(text, sep_idx=0)
        return self._pack(atoms)

    # ---- atomize ----

    def _atomize(self, text: str, sep_idx: int) -> list[str]:
        """Recursively break `text` into pieces each <= max_size.

        Tries each separator in `SEPARATORS` in order. If a single piece
        is still oversized after trying separator N, recurses with N+1.
        Hard-cuts at max_size only when no separator works.
        """
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.max_size:
            return [text]
        if sep_idx >= len(SEPARATORS):
            # Last resort: arithmetic chunking. Should be rare in real text.
            return [text[i : i + self.max_size] for i in range(0, len(text), self.max_size)]

        sep = SEPARATORS[sep_idx]
        parts = text.split(sep)
        if len(parts) == 1:
            # This separator doesn't appear; try the next finer one.
            return self._atomize(text, sep_idx + 1)

        # Reassemble parts back under max_size. If an individual part is
        # itself oversized, recurse to break it down further.
        result: list[str] = []
        buf = ""
        for part in parts:
            if len(part) > self.max_size:
                if buf:
                    result.append(buf)
                    buf = ""
                result.extend(self._atomize(part, sep_idx + 1))
                continue
            candidate = (buf + sep + part) if buf else part
            if len(candidate) > self.max_size:
                if buf:
                    result.append(buf)
                buf = part
            else:
                buf = candidate
        if buf:
            result.append(buf)
        return result

    # ---- pack ----

    def _pack(self, atoms: Iterable[str]) -> list[str]:
        """Greedy-pack atoms into chunks. Each new chunk starts with overlap
        text copied from the tail of the previous chunk.

        Two corner-case guards prevent the "emit a tiny header then immediately
        overshoot the next chunk" failure mode:

        1. If buf is below min_size when an atom would overflow, we absorb
           the atom anyway and accept a single slightly-oversized chunk.
           Better than emitting a scrap and inheriting it as overlap.
        2. When reseeding after a real emit, overlap is best-effort: we drop
           it for that chunk if including it would push over max_size.
        """
        chunks: list[str] = []
        buf = ""
        for atom in atoms:
            if not atom:
                continue
            if not buf:
                buf = atom
                continue
            # Joiner is "\n" (matches the source's paragraph separator).
            candidate = buf + "\n" + atom
            if len(candidate) <= self.max_size:
                buf = candidate
                continue

            # Candidate overflows. Decide whether to emit or absorb.
            if len(buf) < self.min_size:
                # Don't emit a tiny chunk — fold the atom in even if oversized.
                buf = candidate
                continue

            # Normal case: emit buf, reseed with overlap+atom (budget-checked).
            chunks.append(buf)
            overlap_text = self._tail_overlap(buf)
            if overlap_text and len(overlap_text) + 1 + len(atom) <= self.max_size:
                buf = overlap_text + "\n" + atom
            else:
                buf = atom  # skip overlap; the atom itself is near the budget

        if buf:
            chunks.append(buf)

        # Drop the trailing chunk if it's a scrap below min_size. After the
        # absorption guard above, internal sub-min chunks shouldn't occur,
        # so we only ever need to check the tail.
        if chunks and len(chunks[-1]) < self.min_size:
            chunks.pop()
        return chunks

    # ---- overlap ----

    def _tail_overlap(self, text: str) -> str:
        """Return the last `overlap` chars of `text`, snapped forward to
        a sentence/line boundary so the overlap reads cleanly. If no
        boundary is found inside the tail, falls back to a word boundary,
        then to the raw tail."""
        if self.overlap <= 0 or not text:
            return ""
        if len(text) <= self.overlap:
            return text
        tail = text[-self.overlap :]
        # Prefer to start the overlap right after a sentence terminator.
        for marker in [". ", "? ", "! ", "\n"]:
            idx = tail.find(marker)
            if idx != -1:
                return tail[idx + len(marker) :]
        # Otherwise start at the first word boundary inside the tail.
        idx = tail.find(" ")
        if idx != -1:
            return tail[idx + 1 :]
        return tail


# ---------------------------------------------------------------------------
# Filing-level orchestration (per ticker)
# ---------------------------------------------------------------------------


def chunk_filing(filing: dict, chunker: RecursiveChunker) -> list[dict]:
    """Turn one parsed filing into a list of chunk dicts ready for embedding.

    Each chunk dict has:
      - `chunk_id`     deterministic id, e.g. "TSLA-2026-01-29-0042"
      - `text`         the chunk content (including any overlap prefix)
      - `metadata`     ticker / company / section / filing_date / cik /
                       accession / source_url / chunk_index / char_start /
                       char_end

    `char_start`/`char_end` are best-effort offsets into the section's
    cleaned text. With overlap prefixes the exact start can shift by a few
    chars; that's fine — these fields are for human debugging, not the
    retrieval signal.
    """
    base_meta = {
        "ticker": filing["ticker"],
        "company_name": filing["company_name"],
        "filing_date": filing["filing_date"],
        "cik": filing["cik"],
        "accession_number": filing["accession_number"],
        "source_url": filing["source_url"],
    }

    out: list[dict] = []
    global_idx = 0
    for section in filing["sections"]:
        section_name: str = section["section"]
        section_text: str = section["text"]
        section_chunks = chunker.split(section_text)

        cursor = 0  # search hint for char_start lookup
        for i, chunk_text in enumerate(section_chunks):
            char_start, char_end = _locate(section_text, chunk_text, cursor)
            if char_start >= 0:
                cursor = char_start + 1

            out.append(
                {
                    "chunk_id": f"{filing['ticker']}-{filing['filing_date']}-{global_idx:04d}",
                    "text": chunk_text,
                    "metadata": {
                        **base_meta,
                        "section": section_name,
                        "chunk_index": i,
                        "char_start": char_start,
                        "char_end": char_end,
                    },
                }
            )
            global_idx += 1
    return out


def _locate(section_text: str, chunk_text: str, start_hint: int) -> tuple[int, int]:
    """Find where `chunk_text` lives inside `section_text`.

    Direct substring search first (works for the first chunk and chunks
    whose overlap-stitching happens to leave the head intact). Falls back
    to searching for the chunk's last 80 chars, which are always verbatim
    from the source (only the head can be overlap-mutated). Returns
    (-1, -1) if even that fails, which should be very rare."""
    pos = section_text.find(chunk_text, start_hint)
    if pos == -1:
        pos = section_text.find(chunk_text)
    if pos != -1:
        return pos, pos + len(chunk_text)
    # Fallback: locate using the chunk's tail (unaffected by overlap prefix).
    tail = chunk_text[-80:]
    tail_pos = section_text.find(tail, start_hint)
    if tail_pos == -1:
        tail_pos = section_text.find(tail)
    if tail_pos != -1:
        end = tail_pos + len(tail)
        return max(0, end - len(chunk_text)), end
    return -1, -1


# ---------------------------------------------------------------------------
# CLI entry point — `python cli.py chunk --ticker TSLA`
# ---------------------------------------------------------------------------


def run_cli(args) -> None:
    """Read data/clean/<TICKER>.json, chunk, write data/chunks/<TICKER>.jsonl,
    print summary stats and 2 sample chunks for sanity checking."""
    ticker = args.ticker.upper()
    clean_path = config.clean_dir / f"{ticker}.json"
    if not clean_path.exists():
        raise SystemExit(
            f"No cleaned filing at {clean_path}. Run `python cli.py ingest --ticker {ticker}` first."
        )

    filing = json.loads(clean_path.read_text())

    chunker = RecursiveChunker(
        max_size=config.chunk_size,
        overlap=config.chunk_overlap,
        min_size=config.chunk_min_size,
    )
    chunks = chunk_filing(filing, chunker)

    # Write JSONL (one chunk per line — easy to grep, stream-process, diff).
    chunks_dir = config.root / "data" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    out_path = chunks_dir / f"{ticker}.jsonl"
    with out_path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    _print_summary(ticker, chunks, out_path)


def _print_summary(ticker: str, chunks: list[dict], out_path: Path) -> None:
    """Pretty-print counts, length distribution, and two sample chunks."""
    if not chunks:
        print(f"  [{ticker}] no chunks produced — check the input filing.")
        return

    # Counts per section.
    per_section: dict[str, int] = {}
    for c in chunks:
        per_section[c["metadata"]["section"]] = per_section.get(c["metadata"]["section"], 0) + 1

    # Length distribution.
    lengths = sorted(len(c["text"]) for c in chunks)
    n = len(lengths)
    median = lengths[n // 2]
    p95 = lengths[min(n - 1, int(n * 0.95))]

    print()
    print(f"  ticker            : {ticker}")
    print(f"  total chunks      : {n}")
    print(f"  per section       :")
    for sec, count in per_section.items():
        print(f"      {sec:60} {count:>4} chunks")
    print(f"  length (chars)    : min {lengths[0]} | median {median} | p95 {p95} | max {lengths[-1]}")
    print(f"  saved to          : {out_path.relative_to(config.root)}")

    # Two samples — one early, one mid-document — so the user can eyeball
    # what an actual chunk looks like end-to-end.
    print()
    print("  --- sample chunk #2 ---")
    _print_sample_chunk(chunks[min(1, n - 1)])
    print("  --- sample chunk near middle ---")
    _print_sample_chunk(chunks[n // 2])


def _print_sample_chunk(c: dict) -> None:
    meta = c["metadata"]
    text = c["text"]
    preview_head = text[:200].replace("\n", " | ")
    preview_tail = text[-160:].replace("\n", " | ")
    print(f"    id        : {c['chunk_id']}")
    print(f"    section   : {meta['section']}")
    print(f"    length    : {len(text)} chars")
    print(f"    offsets   : [{meta['char_start']}, {meta['char_end']}]")
    print(f"    head      : {preview_head!r}")
    print(f"    tail      : {preview_tail!r}")
    print()
