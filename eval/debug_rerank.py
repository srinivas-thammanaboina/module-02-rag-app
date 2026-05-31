"""Diagnostic: WHY does the bge reranker bury relevant chunks?

Read-only. For a few golden questions it prints each cross-encoder's reranked
top-10 (id, rerank score, cosine, preview) and reports where every labeled
relevant chunk actually landed. minilm (works) vs bge (broken) side by side, so
a usage/calibration bug shows up as bge putting distractors above the relevant
chunks that minilm ranks correctly.

Run: python eval/debug_rerank.py
"""

import os
import sys

# Run-from-root convenience: make `app` importable when invoked as a script
# (python eval/debug_rerank.py puts eval/ on the path, not the project root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieve import Retriever
from app.store import get_vector_store
from app.rerank import Reranker, RERANKER_MODELS

POOL = 50

# (question, company, relevant id suffixes). NIM is the one bge got RIGHT — kept
# as a control so we can see what "working" looks like next to the failures.
CASES = [
    ("What are NVIDIA's gaming segment products?", "NVDA", ["0019", "0011"]),
    ("What does NVIDIA say about CUDA?", "NVDA", ["0016", "0000", "0005"]),
    ("What is NVIDIA NIM?", "NVDA", ["0025"]),
]


def suffix(cid: str) -> str:
    return cid.split("-")[-1]


def main() -> None:
    base = Retriever(get_vector_store())
    rerankers = {k: Reranker(RERANKER_MODELS[k]) for k in ("minilm", "bge")}

    for question, company, rel_suffixes in CASES:
        rel = set(rel_suffixes)
        candidates = base.retrieve(question, k=POOL, company=company)
        print("=" * 90)
        print(f"Q: {question}   (relevant: {sorted(rel)}, pool={len(candidates)})")

        for key, reranker in rerankers.items():
            scores = reranker.score(question, [c["document"] for c in candidates])
            ranked = sorted(zip(candidates, scores), key=lambda p: p[1], reverse=True)
            positions = {suffix(c["id"]): (i, s) for i, (c, s) in enumerate(ranked, 1)}

            print(f"\n  --- {key} reranked top-10 ---")
            for i, (c, s) in enumerate(ranked[:10], 1):
                mark = "REL" if suffix(c["id"]) in rel else "   "
                preview = c["document"][:52].replace("\n", " ").strip()
                print(f"   {i:>2}. {mark} {suffix(c['id'])}  rr={s:+.4f}  cos={c['similarity']:.3f} | {preview}")

            placed = {
                r: f"rank {positions[r][0]} (rr={positions[r][1]:+.4f})" if r in positions else "NOT IN POOL"
                for r in sorted(rel)
            }
            print(f"      relevant placements: {placed}")
        print()


if __name__ == "__main__":
    main()
