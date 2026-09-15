#!/usr/bin/env python
"""Check the evaluation set against the training corpus, sentence by sentence.

FLEURS ships train / validation / test splits. Our 984 h training corpus draws
on FLEURS *train*. The evaluation set here is test + validation, so the question
is whether any evaluated sentence also appears in training text.

It does. The FLEURS validation split reuses sentences that appear in the train
split, so 88 of its 150 distinct sentences are in our training data verbatim.
The test split is clean. Any table that mixes the two advantages the models
trained on that corpus (ours) over third-party checkpoints (hishab, Whisper,
wav2vec2), which is exactly backwards from what a fair comparison needs.

Run this before trusting any number in this repository.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_score import norm

CORPUS = Path("/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline/data2/train_corpus_ready.json")
PRED = Path("outputs_a5000/qwen3_adapter/predictions.json")


def main():
    if not CORPUS.exists():
        print(f"training corpus not present at {CORPUS}; cannot audit", file=sys.stderr)
        return 2

    by_source, all_text = {}, set()
    with CORPUS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = norm(r.get("text") or r.get("transcript") or "")
            all_text.add(t)
            by_source.setdefault(r.get("source", "?"), set()).add(t)

    print("training corpus, distinct normalised sentences per source")
    for k in sorted(by_source, key=lambda k: -len(by_source[k])):
        print(f"  {k:<18} {len(by_source[k]):>7}")
    print(f"  {'TOTAL distinct':<18} {len(all_text):>7}\n")

    ev = json.loads(PRED.read_text())
    rows = []
    for split in ("test", "validation"):
        refs = {norm(r["reference"]) for r in ev if r["split"] == split}
        n_rec = sum(1 for r in ev if r["split"] == split)
        ov = refs & all_text
        rows.append((split, n_rec, len(refs), len(ov)))
        print(f"{split:<11} {n_rec:>5} recordings  {len(refs):>4} distinct sentences  "
              f"{len(ov):>4} also in training  ({len(ov)/max(len(refs),1)*100:.1f}%)")

    contaminated = [r for r in rows if r[3] > 0]
    print()
    if contaminated:
        for split, n_rec, n_ref, n_ov in contaminated:
            print(f"VERDICT: the {split} split is CONTAMINATED for models trained on "
                  f"this corpus.\n         Exclude its {n_rec} recordings from any "
                  f"comparison against third-party checkpoints.")
        return 1
    print("VERDICT: no overlap found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
