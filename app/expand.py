"""
Advanced stage: retrieve-then-expand (grounded aspect decomposition) — Experiment 9.

The enumeration failure (Q7/Q24): a question asks to enumerate several implicit
aspects that each live in a different chunk ("revenue beyond vehicle sales" ->
used cars / energy / leasing / services), and dense collapses onto the single
dominant one. MMR (deterministic diversity) failed it — embedding-geometry
spread is uncorrelated with the semantic aspects the question wants
(enumeration-notes.md). And blind LLM decomposition (Phase B) failed it too —
asked "what are Tesla's revenue streams?" with no context, the LLM returned the
question unchanged, because it doesn't KNOW the segments.

The fix is to GROUND the LLM: retrieve a seed pool first, show the LLM the actual
chunks, and let it name the aspects FROM the evidence — then re-query each. For
Q24 the overview chunk literally lists the markets, so the LLM reads them off and
re-queries each subsection (which a single global query never reaches). Grounding
is the whole difference from Phase B.

    1. SEED    = base.retrieve(question, SEED_POOL, company)   # ground the LLM
    2. aspects = LLM_extract(question, SEED)                   # [] if single-topic
    3. if < 2 aspects:  passthrough to base (provably no collateral)
    4. per-aspect retrieve + round_robin_merge                 # guaranteed slots

Composition: `ExpandRetriever` wraps the base `Retriever`, honors the same
`.retrieve()` contract, and reuses Phase A's `round_robin_merge`. Design +
predictions in notes/advanced/enumeration-notes.md.
"""

from __future__ import annotations

import json

from app.config import config
from app.decompose import round_robin_merge

# Aspect extraction is a cheap reading task -> Haiku by default; opus/sonnet are
# here to A/B extraction QUALITY (does a bigger model split better when grounded?).
EXPANDER_MODELS = {
    "haiku": config.decomposer_model,   # claude-haiku-4-5
    "sonnet": "claude-sonnet-4-6",
    "opus": config.anthropic_model,     # claude-opus-4-8
}

# How many chunks to ground the LLM's aspect extraction in. Wide enough to
# surface an overview/list chunk (Q24) and the distinct aspect chunks (Q7),
# small enough to keep the prompt cheap.
SEED_POOL = 12

# Per-chunk excerpt length in the grounding prompt — enough to read the aspect
# names off, not the whole chunk.
_SEED_CHARS = 320

# Cache aspect extractions (corpus is static) so the eval is reproducible and we
# don't re-bill. Nested by model, like the Phase B cache, so A/B keeps each
# model's splits side by side. Separate file from decomp_cache.json.
_CACHE_PATH = config.root / "data" / "expand_cache.json"

# Structured output: force the aspects via a tool call. An EMPTY list is a valid,
# meaningful answer (= "not an enumeration, pass through").
_EXPAND_TOOL = {
    "name": "submit_aspects",
    "description": (
        "Return one focused retrieval query per distinct aspect the question asks "
        "to enumerate, grounded in the excerpts. EMPTY list if the question is "
        "answerable from a single topic (not an enumeration)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "aspects": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "0-6 focused search queries, one per distinct aspect/category/"
                    "segment/item present in the EXCERPTS that the question asks to "
                    "enumerate. Use the actual names that appear in the excerpts. "
                    "EMPTY if the question is single-topic."
                ),
            }
        },
        "required": ["aspects"],
    },
}

_SYSTEM_PROMPT = """\
You are given a question about a company's SEC 10-K filing and a set of excerpts \
retrieved for it.

Decide whether answering the question requires ENUMERATING several DISTINCT \
aspects — categories, segments, products, revenue lines, or markets — that the \
question asks to list (cues: "beyond X", "what does EACH cover", "the various", \
"types of", "ways", "what ... serve") and that each live in a DIFFERENT part of \
the filing.

- If YES: return one focused retrieval query per aspect, GROUNDED IN THE \
EXCERPTS. Use the actual names that appear there (the specific segments/markets/\
revenue lines the filing names). Each query should name the company and that one \
aspect, phrased to retrieve that aspect's passage.
- If NO (single topic, or a single comparison already handled upstream): return \
an EMPTY list.

Do NOT invent aspects the excerpts don't support. Do NOT split a single-topic \
question. Always call submit_aspects."""


# ---------------------------------------------------------------------------
# Cache (nested {model: {question: aspects}})
# ---------------------------------------------------------------------------


def _read_all_caches() -> dict[str, dict[str, list[str]]]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _load_cache(model: str) -> dict[str, list[str]]:
    entry = _read_all_caches().get(model)
    return entry if isinstance(entry, dict) else {}


def _save_cache(model: str, entries: dict[str, list[str]]) -> None:
    all_caches = _read_all_caches()
    all_caches[model] = entries
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(all_caches, indent=2))


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


def _format_seed(seed: list[dict]) -> str:
    """Render seed chunks as compact id-labeled excerpts for the grounding turn."""
    blocks = []
    for r in seed:
        text = (r.get("document") or "").strip().replace("\n", " ")[:_SEED_CHARS]
        blocks.append(f"[{r['id']}] {text}")
    return "\n".join(blocks)


class ExpandRetriever:
    """Grounded aspect decomposition wrapped around a base retriever.

    Retrieve a seed pool, ask the LLM to name the enumeration's aspects FROM that
    evidence (cached), retrieve each aspect, round-robin merge. A single-topic
    question yields no aspects -> passthrough, identical to baseline. Honors the
    base `.retrieve()` contract, so it's a drop-in for eval/ask.
    """

    def __init__(self, base, model: str | None = None):
        self._base = base
        self._model = model or config.decomposer_model
        self._client = None  # lazy — the anthropic SDK import is heavy
        self._cache = _load_cache(self._model)

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=config.require_anthropic_key())
        return self._client

    def _call_llm(self, question: str, seed: list[dict]) -> list[str]:
        """One grounded call -> aspect queries via forced tool use. Falls back to
        [] (= passthrough) on any malformed/empty/error result."""
        try:
            user_turn = f"Question: {question}\n\nRetrieved excerpts:\n{_format_seed(seed)}"
            resp = self._get_client().messages.create(
                model=self._model,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                tools=[_EXPAND_TOOL],
                tool_choice={"type": "tool", "name": "submit_aspects"},
                messages=[{"role": "user", "content": user_turn}],
            )
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use" and block.name == "submit_aspects":
                    aspects = block.input.get("aspects", [])
                    cleaned = [a.strip() for a in aspects if isinstance(a, str) and a.strip()]
                    return cleaned[:6]
        except Exception:
            pass
        return []

    def expand(self, question: str, seed: list[dict]) -> list[str]:
        """Cached grounded aspects for a question (write-through cache)."""
        if question in self._cache:
            return self._cache[question]
        aspects = self._call_llm(question, seed)
        self._cache[question] = aspects
        _save_cache(self._model, self._cache)
        return aspects

    def retrieve(self, question: str, k: int = 5, company: str | None = None) -> list[dict]:
        # 1. Ground the LLM in a seed retrieval.
        seed = self._base.retrieve(question, k=SEED_POOL, company=company)
        # 2. Grounded aspect extraction (cached). 3. Dispatch.
        aspects = self.expand(question, seed)
        if len(aspects) < 2:
            return self._base.retrieve(question, k=k, company=company)  # passthrough
        # 4. Per-aspect retrieval + guaranteed-slots merge.
        per_aspect = {a: self._base.retrieve(a, k=k, company=company) for a in aspects}
        return round_robin_merge(per_aspect, k)
