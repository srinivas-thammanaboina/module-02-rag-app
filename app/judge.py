"""
Advanced stage: LLM-as-judge eval — let the LLM help FINISH the answer key.

The problem (eval-audit / llm-judge-notes): 7 "representative" golden questions
have no complete list of right-answer chunks ("What are Tesla's main risks?" — the
filing lists dozens; we hand-picked 5). Their recall denominator is a fiction, so
they sit quarantined and effectively unmeasured.

The fix here treats the LLM as a LABELER, not a grader: for a question + ONE chunk
it answers "does this chunk belong in the answer key — yes/no?". Pool every chunk
any retriever surfaced for a stuck question, label each, and the yes-pile becomes a
fuller, defensible answer key. Our trusted recall@k scoring then runs on top,
unchanged.

The catch — an LLM judge is itself an unproven measuring stick — so the FIRST thing
this module does is CHECK THE GRADER: run it on the 16 questions whose answer key we
already trust (hand-labeled, complete) and measure agreement. Pass-bar, pre-committed
in llm-judge-notes.md:
  - recovers known-right chunks   >= 90%   (says "yes" to chunks we know are correct)
  - wrongly accepts known-wrong   <= ~10-15% (says "yes" to chunks we know are not)
If it can't reproduce a key we trust, its labels on the unknowns are worthless — stop.
Same discipline that caught the bge reranker being a broken measurement.

Never overwrites eval/golden.jsonl. The LLM labels land in a SIDECAR file alongside
the hand labels, so human-vs-LLM stays a diff and the original key survives.
"""

from __future__ import annotations

import json

from app.config import config

# Judge model shorthands (a full model name is also accepted). Labeling is a cheap
# reading task -> Haiku first, per the spine (don't assume the big model wins —
# the 16-question check decides). Escalate to opus only if Haiku fails the bar.
JUDGE_MODELS = {
    "haiku": config.decomposer_model,   # claude-haiku-4-5
    "sonnet": "claude-sonnet-4-6",
    "opus": config.anthropic_model,     # claude-opus-4-8
}

# How many chunks to pull from each pooled config when building the candidate pile.
# Generous on purpose — the whole point is to surface chunks dense misses so the
# judge can rescue them into the key.
DEFAULT_POOL = 20

# Per reliable question, how many of the highest-ranked NON-key chunks to test as
# "known-wrong". These are the hardest negatives (a generic chunk the retriever
# ranked high) — the most meaningful false-accept test, and it caps cost.
DEFAULT_NEG_SAMPLE = 8

# Cache labels (corpus is static) so the check is reproducible and we don't re-bill.
# Nested {model: {question: {chunk_id: bool}}}, like the decomp/expand caches.
_CACHE_PATH = config.root / "data" / "judge_cache.json"

# Where the judge's labels for the representative questions are written. A SIDECAR,
# never golden.jsonl — the hand key stays the source of truth and inspectable.
SIDECAR_PATH = config.root / "eval" / "golden_judge_labels.json"

# Excerpt length handed to the judge — enough to decide relevance, not the whole chunk.
_JUDGE_CHARS = 1200

# Bump when the rubric/prompt changes. The cache is namespaced by model#vN, so a
# bump re-judges from scratch (old verdicts stay on disk to diff against). v2 added
# the company constraint + fragment-robustness line (see llm-judge-notes.md).
_PROMPT_VERSION = 2

# The judge sees a chunk's source company by NAME (questions say "Tesla", not "TSLA").
_TICKER_NAME = {"TSLA": "Tesla", "AAPL": "Apple", "NVDA": "NVIDIA"}

# Structured output: force a clean yes/no via a tool call rather than parsing prose.
_JUDGE_TOOL = {
    "name": "submit_relevance",
    "description": "Report whether the excerpt belongs in the answer key for the question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "relevant": {
                "type": "boolean",
                "description": "true only if the excerpt directly answers the question or a distinct part of it.",
            },
            "reason": {
                "type": "string",
                "description": "One short clause explaining the call (for human inspection).",
            },
        },
        "required": ["relevant"],
    },
}

_SYSTEM_PROMPT = """\
You label whether ONE excerpt from a SEC 10-K filing belongs in the ANSWER KEY for \
a question — i.e. whether a correct answer would actually draw on this excerpt.

Say relevant=true ONLY if the excerpt contains information that directly answers the \
question or a distinct part of it. Be strict:
- A passing mention of a keyword, or text merely on the same general topic, is NOT \
relevant.
- The excerpt must carry a fact a correct answer would actually use.
- If the question asks to enumerate or compare, the excerpt is relevant when it \
covers at least one of the asked aspects/sides.

COMPANY CONSTRAINT (check this FIRST): the excerpt's source company is stated in the \
user turn. If the question is about specific companies and this excerpt's company is \
NOT one of them, answer relevant=false no matter how well the topic matches — it is \
the wrong company's filing.

Excerpts are overlapping windows and may begin or end mid-sentence; judge them on the \
substantive content they DO contain, not on whether they read as a complete passage.

Otherwise say relevant=false. When genuinely unsure about topic relevance, lean false \
— a true label puts this chunk into the graded answer key, so the bar is "would a \
careful analyst cite it", not "is it loosely related".

Always call submit_relevance."""


# ---------------------------------------------------------------------------
# Chunk text (read straight from the source JSONL, not the store)
# ---------------------------------------------------------------------------


def load_chunk_texts() -> dict[str, str]:
    """Map every chunk id -> its text, read from data/chunks/*.jsonl.

    Read from the source files (the same input the index is built from) rather
    than the vector store, so we have text for ANY id — including known-right
    chunks that no retriever ever surfaces.
    """
    texts: dict[str, str] = {}
    chunks_dir = config.root / "data" / "chunks"
    for path in sorted(chunks_dir.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                texts[c["chunk_id"]] = c["text"]
    return texts


# ---------------------------------------------------------------------------
# Cache (nested {model: {question: {chunk_id: bool}}})
# ---------------------------------------------------------------------------


def _read_all_caches() -> dict:
    if not _CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _load_cache(model: str) -> dict:
    entry = _read_all_caches().get(model)
    return entry if isinstance(entry, dict) else {}


def _save_cache(model: str, entries: dict) -> None:
    all_caches = _read_all_caches()
    all_caches[model] = entries
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(all_caches, indent=2))


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


class RelevanceJudge:
    """Labels (question, chunk) -> relevant yes/no, cached and grounded.

    Narrow on purpose: it never ranks or scores a set, only judges one real chunk
    against one question, so the call is checkable. Reuses the lazy-client +
    write-through cache pattern from expand.py / llm_decompose.py.
    """

    def __init__(self, model: str | None = None):
        self._model = model or config.decomposer_model
        # Cache namespace = model + rubric version, so changing the prompt re-judges
        # from scratch instead of returning stale verdicts (old ones stay on disk).
        self._ns = f"{self._model}#v{_PROMPT_VERSION}"
        self._client = None  # lazy — the anthropic SDK import is heavy
        self._cache = _load_cache(self._ns)

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=config.require_anthropic_key())
        return self._client

    def _call_llm(self, question: str, company: str, text: str) -> bool:
        """One labeling call -> bool via forced tool use. Falls back to False
        (leave it out of the key) on any malformed/error result."""
        try:
            user_turn = (
                f"Question: {question}\n\n"
                f"This excerpt is from {company}'s 10-K filing.\n"
                f'Excerpt:\n"""\n{text[:_JUDGE_CHARS]}\n"""'
            )
            resp = self._get_client().messages.create(
                model=self._model,
                max_tokens=256,
                system=_SYSTEM_PROMPT,
                tools=[_JUDGE_TOOL],
                tool_choice={"type": "tool", "name": "submit_relevance"},
                messages=[{"role": "user", "content": user_turn}],
            )
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use" and block.name == "submit_relevance":
                    return bool(block.input.get("relevant", False))
        except Exception:
            pass
        return False

    def judge(self, question: str, chunk_id: str, text: str) -> bool:
        """Cached relevance label for one (question, chunk). Write-through.

        The chunk's source company is derived from the id prefix (TICKER-...) and
        handed to the judge so it can reject the wrong company's filing.
        """
        q_cache = self._cache.setdefault(question, {})
        if chunk_id in q_cache:
            return q_cache[chunk_id]
        ticker = chunk_id.split("-")[0]
        company = _TICKER_NAME.get(ticker, ticker)
        verdict = self._call_llm(question, company, text)
        q_cache[chunk_id] = verdict
        _save_cache(self._ns, self._cache)
        return verdict


# ---------------------------------------------------------------------------
# Candidate pool (the wide net the judge labels)
# ---------------------------------------------------------------------------


def _build_retrievers():
    """Return (dense_base, full_stack) sharing one vector store.

    Pooling both = a wide candidate net: dense covers semantic, the shipped stack
    adds BM25 (opaque tokens), per-company, and grounded-aspect chunks dense misses.
    Mirrors the exact stack `ask` ships (generate.py).
    """
    from app.retrieve import Retriever
    from app.store import get_vector_store
    from app.decompose import DecompositionRetriever
    from app.expand import ExpandRetriever
    from app.hybrid import HybridRetriever

    base = Retriever(get_vector_store())
    full = ExpandRetriever(
        DecompositionRetriever(
            HybridRetriever(base, fusion="interleave", gated=True)
        )
    )
    return base, full


def candidate_pool(question: str, company: str | None, retrievers, depth: int) -> list[str]:
    """Ordered, de-duplicated chunk ids any pooled config surfaced for the question.

    Dense first (so dense rank ~ pool rank), then ids only the full stack found.
    Order matters: the highest-ranked NON-key ids become the "hardest negatives".
    """
    base, full = retrievers
    ordered: list[str] = []
    seen: set[str] = set()
    for retriever in (base, full):
        for row in retriever.retrieve(question, k=depth, company=company):
            rid = row["id"]
            if rid not in seen:
                seen.add(rid)
                ordered.append(rid)
    return ordered


# ---------------------------------------------------------------------------
# Step 1 — check the grader (validate on the 16 trusted questions)
# ---------------------------------------------------------------------------


def validate(judge: RelevanceJudge, golden: list[dict], texts: dict[str, str],
             retrievers, depth: int, neg_sample: int) -> dict:
    """Measure judge-vs-human agreement on the COMPLETE-key (reliable) questions.

    For each reliable question we already trust the key, so:
      - known-right = the hand key            -> judge should say YES (recovery).
      - known-wrong = top non-key pooled ids  -> judge should say NO (false-accept).
    Returns aggregate rates + per-question rows + diagnostic miss/false-accept lists.
    """
    reliable = [g for g in golden if g.get("relevant_ids") and g.get("recall_reliable", True)]

    rows = []
    tot_right = tot_recovered = tot_wrong = tot_false_acc = 0
    for g in reliable:
        q = g["question"]
        company = g.get("company")
        key = list(g["relevant_ids"])

        # Recovery: does the judge say yes to chunks we KNOW are right?
        recovered = [cid for cid in key if cid in texts and judge.judge(q, cid, texts[cid])]

        # False-accepts: the highest-ranked pooled ids that are NOT in the key.
        pool = candidate_pool(q, company, retrievers, depth)
        negatives = [cid for cid in pool if cid not in set(key)][:neg_sample]
        false_acc = [cid for cid in negatives if cid in texts and judge.judge(q, cid, texts[cid])]

        tot_right += len(key)
        tot_recovered += len(recovered)
        tot_wrong += len(negatives)
        tot_false_acc += len(false_acc)
        rows.append({
            "id": g["id"],
            "category": g["category"],
            "n_right": len(key),
            "n_recovered": len(recovered),
            "missed": sorted(set(key) - set(recovered)),
            "n_neg": len(negatives),
            "false_accepts": false_acc,
        })

    recovery = tot_recovered / tot_right if tot_right else 0.0
    false_rate = tot_false_acc / tot_wrong if tot_wrong else 0.0
    return {
        "rows": rows,
        "recovery": recovery,
        "false_rate": false_rate,
        "tot_right": tot_right,
        "tot_recovered": tot_recovered,
        "tot_wrong": tot_wrong,
        "tot_false_acc": tot_false_acc,
        # The pre-committed bar (llm-judge-notes.md).
        "passed": recovery >= 0.90 and false_rate <= 0.15,
    }


# ---------------------------------------------------------------------------
# Step 2 — build the fuller key for the representative questions
# ---------------------------------------------------------------------------


def build_fuller_key(judge: RelevanceJudge, golden: list[dict], texts: dict[str, str],
                     retrievers, depth: int) -> dict:
    """Judge every pooled chunk for each representative question -> a fuller key.

    new key = hand key (kept; known-good) UNION the pooled chunks judged relevant.
    Returns a sidecar dict keyed by question id; the caller writes it to disk.
    """
    representative = [
        g for g in golden if g.get("relevant_ids") and not g.get("recall_reliable", True)
    ]

    sidecar = {"model": judge.model, "pool_depth": depth, "questions": {}}
    for g in representative:
        q = g["question"]
        company = g.get("company")
        hand = list(g["relevant_ids"])
        pool = candidate_pool(q, company, retrievers, depth)
        judged_yes = [cid for cid in pool if cid in texts and judge.judge(q, cid, texts[cid])]
        new_key = sorted(set(hand) | set(judged_yes))
        sidecar["questions"][str(g["id"])] = {
            "question": q,
            "hand_key": sorted(hand),
            "pool_size": len(pool),
            "judged_relevant": sorted(judged_yes),
            "new_key": new_key,
            # chunks the judge ADDED beyond the hand picks (the actual upgrade)
            "added": sorted(set(judged_yes) - set(hand)),
        }
    return sidecar


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _short(cid: str) -> str:
    """Trailing chunk number for compact display (TSLA-...-0084 -> 0084)."""
    return cid.split("-")[-1]


def _print_validation(stats: dict, model: str) -> None:
    print()
    print(f"  === STEP 1 — CHECK THE GRADER ({model}) ===")
    print("  Agreement vs the 16 trusted (complete-key) questions.")
    print()
    print(f"  {'Q':>3}  {'category':<14} {'recovered':>10}  {'false-acc':>10}  notes")
    print(f"  {'-'*3}  {'-'*14} {'-'*10}  {'-'*10}  {'-'*30}")
    for r in stats["rows"]:
        rec = f"{r['n_recovered']}/{r['n_right']}"
        fa = f"{len(r['false_accepts'])}/{r['n_neg']}"
        note = ""
        if r["missed"]:
            note = "missed: " + ",".join(_short(c) for c in r["missed"])
        if r["false_accepts"]:
            fa_note = "wrong-yes: " + ",".join(_short(c) for c in r["false_accepts"])
            note = f"{note}  {fa_note}" if note else fa_note
        print(f"  {r['id']:>3}  {r['category']:<14} {rec:>10}  {fa:>10}  {note}")

    print()
    print("  --- agreement vs the pre-committed bar ---")
    rec_ok = "OK" if stats["recovery"] >= 0.90 else "UNDER"
    fa_ok = "OK" if stats["false_rate"] <= 0.15 else "OVER"
    print(f"  recovery of known-right : {stats['recovery']:.2f}  "
          f"({stats['tot_recovered']}/{stats['tot_right']})   bar >= 0.90  [{rec_ok}]")
    print(f"  false-accept of known-wrong: {stats['false_rate']:.2f}  "
          f"({stats['tot_false_acc']}/{stats['tot_wrong']})   bar <= 0.15  [{fa_ok}]")
    print()
    verdict = "PASS — judge is trustworthy on this corpus" if stats["passed"] \
        else "FAIL — do NOT trust the judge's labels; revisit rubric/model"
    print(f"  VERDICT: {verdict}")
    print()


def _print_key_build(sidecar: dict) -> None:
    print()
    print(f"  === STEP 2 — FULLER KEY for the representative questions ({sidecar['model']}) ===")
    print()
    print(f"  {'Q':>3}  {'hand':>4}  {'pool':>4}  {'judged':>6}  {'new key':>7}  added")
    print(f"  {'-'*3}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*30}")
    for qid, d in sidecar["questions"].items():
        added = ",".join(_short(c) for c in d["added"]) or "(none)"
        print(f"  {qid:>3}  {len(d['hand_key']):>4}  {d['pool_size']:>4}  "
              f"{len(d['judged_relevant']):>6}  {len(d['new_key']):>7}  {added}")
    print()
    print(f"  wrote sidecar -> {SIDECAR_PATH.relative_to(config.root)}")
    print("  (golden.jsonl untouched — these are the judge's labels alongside the hand key)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_cli(args) -> None:
    """`python cli.py judge [--build-key] [--judge-model haiku] [--pool N] [--neg-sample N] [--force]`.

    Default: STEP 1 only — check the grader against the 16 trusted questions and
    print pass/fail vs the pre-committed bar. With --build-key, also run STEP 2
    (build the fuller key for the 7 representative questions) — but only if STEP 1
    passes, unless --force.
    """
    from app.eval import load_golden

    key = getattr(args, "judge_model", None) or "haiku"
    model = JUDGE_MODELS.get(key, key)  # shorthand or a full model name
    depth = getattr(args, "pool", None) or DEFAULT_POOL
    neg_sample = getattr(args, "neg_sample", None) or DEFAULT_NEG_SAMPLE

    golden = load_golden()
    texts = load_chunk_texts()
    judge = RelevanceJudge(model=model)
    retrievers = _build_retrievers()

    print()
    print(f"  judge model : {model}  (rubric v{_PROMPT_VERSION})")
    print(f"  pool depth  : {depth}   neg-sample/Q : {neg_sample}")
    print(f"  cache       : {_CACHE_PATH.relative_to(config.root)} (per-model; reruns are free)")

    # Step 1 — always.
    stats = validate(judge, golden, texts, retrievers, depth, neg_sample)
    _print_validation(stats, model)

    # Step 2 — opt-in, gated on the bar (unless forced).
    if getattr(args, "build_key", False):
        if not stats["passed"] and not getattr(args, "force", False):
            print("  --build-key skipped: judge FAILED the check. Re-run with --force to "
                  "build anyway (not recommended), or try --judge-model opus.")
            print()
            return
        sidecar = build_fuller_key(judge, golden, texts, retrievers, depth)
        SIDECAR_PATH.parent.mkdir(parents=True, exist_ok=True)
        SIDECAR_PATH.write_text(json.dumps(sidecar, indent=2))
        _print_key_build(sidecar)
