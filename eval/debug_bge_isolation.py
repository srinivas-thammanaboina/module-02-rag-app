"""Decisive test: is bge's bad ranking a USAGE/calibration artifact or real?

The pool diagnostic showed bge saturating ~10 irrelevant chunks at sigmoid≈1.0
while the true gaming chunks score ≈0.02. This isolates the cause:

  - Prints RAW LOGITS (activation off) next to the sigmoid score, so we can see
    whether the sigmoid is just saturated (logits all large +) or the model
    genuinely ranks the gaming chunk below the distractor.
  - Scores a synthetic OBVIOUS gaming sentence vs an OBVIOUS irrelevant one — if
    bge can't separate those, the model/usage is broken, full stop.
  - Scores each pair ISOLATED and re-checks, to rule out a batch/context effect.

Read-only. Run: python eval/debug_bge_isolation.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

from app.config import config  # noqa: E402
from app.rerank import RERANKER_MODELS  # noqa: E402

QUESTION = "What are NVIDIA's gaming segment products?"


def load_nvda_chunks(suffixes: set[str]) -> dict[str, str]:
    path = config.root / "data" / "chunks" / "NVDA.jsonl"
    out = {}
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            sfx = c["chunk_id"].split("-")[-1]
            if sfx in suffixes:
                out[sfx] = c["text"]
    return out


def main() -> None:
    real = load_nvda_chunks({"0019", "0011", "0041"})

    cases = [
        ("synthetic OBVIOUS gaming",
         "NVIDIA GeForce RTX GPUs are our gaming products; gamers buy them to play video games."),
        ("synthetic OBVIOUS irrelevant",
         "The company's deemed repatriation tax payable was 8.8 billion dollars."),
        ("REAL 0019 (gaming, bge ranked #33)", real["0019"]),
        ("REAL 0011 (gaming, bge ranked #38)", real["0011"]),
        ("REAL 0041 (irrelevant, bge ranked #1 @0.9998)", real["0041"]),
    ]

    for key in ("minilm", "bge"):
        model = CrossEncoder(RERANKER_MODELS[key])
        pairs = [(QUESTION, doc) for _, doc in cases]
        # batch scores: raw logits (Identity) and the default activation (sigmoid for bge)
        raw_batch = model.predict(pairs, activation_fct=torch.nn.Identity())
        act_batch = model.predict(pairs)

        print("=" * 88)
        print(f"  {key}   (query: {QUESTION!r})")
        print(f"  {'label':<46} {'raw logit':>11} {'activated':>11} {'isolated raw':>13}")
        for (label, doc), rl, av in zip(cases, raw_batch, act_batch):
            iso = float(model.predict([(QUESTION, doc)], activation_fct=torch.nn.Identity())[0])
            print(f"  {label:<46} {float(rl):>11.4f} {float(av):>11.4f} {iso:>13.4f}")
        print()


if __name__ == "__main__":
    main()
