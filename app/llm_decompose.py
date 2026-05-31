"""
Advanced stage: decomposition Phase B — LLM query decomposition (Experiment 7).

Phase A split questions on a KNOWN, detectable axis (company, by keyword match) —
deterministic, free, provably safe. Phase B handles questions whose split axis is
SEMANTIC and unknown ahead of time (aspects: "revenue beyond vehicle sales" ->
{used cars, energy, leasing, services}). There's no keyword list for "aspects",
so we ask an LLM to find the axis and split on it. It's query UNDERSTANDING, not
retrieval — and it's strictly more general than Phase A (it splits cross-company
questions per company AND aspect questions per aspect).

The trade vs Phase A: an LLM now sits in front of every retrieval, so we lose
determinism (-> cache), zero-cost (-> 1 call per uncached question), and the
safety guarantee (atomic questions SHOULD pass through unchanged, but that's now
something to measure, not a proof). Design + the confirmed decisions + the
predictions are in notes/advanced/decomposition-notes.md.

Composition: `LLMDecompositionRetriever` wraps the base `Retriever`, honors the
same `.retrieve()` contract, and reuses Phase A's `round_robin_merge`.
"""

from __future__ import annotations

import json

from app.config import config
from app.decompose import round_robin_merge
from app.retrieve import detect_companies_in_question

# Decomposer model shorthands (full model names also accepted). Splitting is a
# cheap task -> Haiku by default; opus/sonnet are here to A/B decomposition
# QUALITY against retrieval-mechanism effects (see decomposition-notes.md).
DECOMPOSER_MODELS = {
    "haiku": config.decomposer_model,   # claude-haiku-4-5
    "sonnet": "claude-sonnet-4-6",
    "opus": config.anthropic_model,     # claude-opus-4-8
}

# Cache decompositions so the eval is reproducible and we don't re-bill the same
# questions every run. Nested by model name, so A/B-ing models keeps every
# model's splits side by side instead of clobbering each other.
_CACHE_PATH = config.root / "data" / "decomp_cache.json"

# Structured output: force the model to return sub-queries via a tool call rather
# than parsing free text (robust failure surface).
_DECOMPOSE_TOOL = {
    "name": "submit_subqueries",
    "description": "Return the minimal set of focused retrieval sub-queries the question decomposes into.",
    "input_schema": {
        "type": "object",
        "properties": {
            "subqueries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "1-5 self-contained retrieval queries. Exactly ONE element "
                    "(the original question, lightly cleaned) if it is already atomic."
                ),
            }
        },
        "required": ["subqueries"],
    },
}

_SYSTEM_PROMPT = """\
You split a question about company SEC 10-K filings into the minimal set of \
focused sub-queries needed to retrieve every relevant passage.

Rules:
- If the question spans multiple companies, split per company.
- If it enumerates or implies multiple distinct aspects/topics, split one \
sub-query per aspect.
- If it is already atomic (one company, one topic), return it unchanged as a \
SINGLE sub-query. Do not invent aspects the question does not imply.
- Each sub-query must be self-contained and name its company. Return 1-5 \
sub-queries, ordered as the question presents them.

Always call submit_subqueries."""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _read_all_caches() -> dict[str, dict[str, list[str]]]:
    """Read the whole {model: {question: subqueries}} cache file (or {})."""
    if not _CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    if "__model__" in raw:
        # Legacy flat format ({"__model__": m, question: subs}) -> migrate to
        # the nested {m: {question: subs}} shape so the prior run is preserved.
        model = raw["__model__"]
        return {model: {q: s for q, s in raw.items() if q != "__model__"}}
    return raw


def _load_cache(model: str) -> dict[str, list[str]]:
    """This model's cached splits (empty if it has never been run)."""
    entry = _read_all_caches().get(model)
    return entry if isinstance(entry, dict) else {}


def _save_cache(model: str, entries: dict[str, list[str]]) -> None:
    """Write this model's splits, preserving every other model's cache."""
    all_caches = _read_all_caches()
    all_caches[model] = entries
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(all_caches, indent=2))


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class LLMDecompositionRetriever:
    """LLM query decomposition wrapped around a base retriever.

    Decompose the question into focused sub-queries (cached), retrieve each on
    its TEXT (no per-sub-query filter — see notes decision 4), then round-robin
    merge. Atomic questions (one sub-query) pass straight through to the base.
    Honors the base `.retrieve()` contract, so it's a drop-in for eval/ask.
    """

    def __init__(self, base, model: str | None = None, filter_subqueries: bool = False):
        self._base = base
        self._model = model or config.decomposer_model
        # Phase B+: hard-filter any sub-query that names exactly one company to
        # that ticker (recovers Phase A's partition guarantee); aspect sub-queries
        # with no company stay text-only. See notes/advanced/decomposition-notes.md.
        self._filter_subqueries = filter_subqueries
        self._client = None  # lazy — the anthropic SDK import is heavy
        self._cache = _load_cache(self._model)

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=config.require_anthropic_key())
        return self._client

    def _call_llm(self, question: str) -> list[str]:
        """One Haiku call -> sub-queries via forced tool use. Falls back to
        [question] on any malformed/empty result (= baseline behavior)."""
        try:
            resp = self._get_client().messages.create(
                model=self._model,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                tools=[_DECOMPOSE_TOOL],
                tool_choice={"type": "tool", "name": "submit_subqueries"},
                messages=[{"role": "user", "content": f"Question: {question}"}],
            )
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use" and block.name == "submit_subqueries":
                    subs = block.input.get("subqueries", [])
                    cleaned = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
                    if cleaned:
                        return cleaned[:5]
        except Exception:
            pass
        return [question]  # fallback == passthrough

    def decompose(self, question: str) -> list[str]:
        """Cached sub-queries for a question (write-through cache)."""
        if question in self._cache:
            return self._cache[question]
        subs = self._call_llm(question)
        self._cache[question] = subs
        _save_cache(self._model, self._cache)
        return subs

    def _subquery_company(self, sub: str, caller_company: str | None) -> str | None:
        """The ticker filter to use for one sub-query.

        The caller's --company always wins. Otherwise, in Phase B+ mode, a
        sub-query that names EXACTLY one company is hard-filtered to it; a
        sub-query naming zero (an aspect) or two+ companies stays unfiltered.
        """
        if caller_company is not None:
            return caller_company
        if self._filter_subqueries:
            detected = detect_companies_in_question(sub)
            if len(detected) == 1:
                return next(iter(detected))
        return None

    def retrieve(self, question: str, k: int = 5, company: str | None = None) -> list[dict]:
        subs = self.decompose(question)
        # Atomic -> passthrough, identical to baseline for that question.
        if len(subs) <= 1:
            return self._base.retrieve(question, k=k, company=company)
        # Multi-part: retrieve each sub-query (Phase B+ may hard-filter per company), merge.
        per_sub = {
            sub: self._base.retrieve(sub, k=k, company=self._subquery_company(sub, company))
            for sub in subs
        }
        return round_robin_merge(per_sub, k)
