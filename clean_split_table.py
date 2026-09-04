#!/usr/bin/env python
"""Rescore every model on the CONTAMINATION-FREE FLEURS test split only.

88 of the 150 distinct sentences in the FLEURS validation split appear verbatim
in our training corpus (via fleurs_train). The 402 validation recordings must
therefore be dropped from any comparison involving our own models: they inflate
ours and not the third-party checkpoints. The 920-utterance test split has zero
overlap with the training corpus and is the honest comparison surface.
"""
import json
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np
from bench_score import norm

MODELS = ["hishab_conformer_large", "qwen3_adapter", "ehzawad_fastconformer",
          "whisper_medium", "wav2vec2", "hishab_fastconformer"]
OURS = {"qwen3_adapter", "ehzawad_fastconformer"}


def counts(ref, hyp, char=False):
    f = jiwer.process_characters if char else jiwer.process_words
    m = f([norm(ref)], [norm(hyp)])
    unit = len(norm(ref)) if char else len(norm(ref).split())
    return m.substitutions + m.deletions + m.insertions, unit


def load(name, split="test"):
    d = json.loads(Path(f"outputs_a5000/{name}/predictions.json").read_text())
    return [r for r in d if r["split"] == split]


def boot_ci(e, n, keys, seed=20260903, reps=10000):
    g = defaultdict(list)
    for i, k in enumerate(keys):
        g[k].append(i)
    gi = [np.array(v) for v in g.values()]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        pick = np.concatenate([gi[j] for j in rng.integers(0, len(gi), len(gi))])
        out.append(e[pick].sum() / n[pick].sum())
    out = np.sort(out)
    return float(out[int(.025 * reps)]), float(out[int(.975 * reps)])


res, store = {}, {}
for m in MODELS:
    rows = load(m)
    ew, nw, ec, nc, keys = [], [], [], [], []
    for r in rows:
        a, b = counts(r["reference"], r["transcript"] or "")
        c, d = counts(r["reference"], r["transcript"] or "", char=True)
        ew.append(a); nw.append(b); ec.append(c); nc.append(d)
        keys.append(norm(r["reference"]))
    ew, nw, ec, nc = map(np.array, (ew, nw, ec, nc))
    store[m] = (ew, nw, ec, nc, keys)
    lo, hi = boot_ci(ew, nw, keys)
    res[m] = {"n": len(rows), "wer": round(float(ew.sum() / nw.sum()), 5),
              "wer_ci95": [round(lo, 5), round(hi, 5)],
              "cer": round(float(ec.sum() / nc.sum()), 5),
              "ours": m in OURS}

print(f"{'model':<26}{'WER%':>8}{'95% CI':>18}{'CER%':>8}")
for m, v in sorted(res.items(), key=lambda kv: kv[1]["wer"]):
    ci = f"[{v['wer_ci95'][0]*100:.2f}-{v['wer_ci95'][1]*100:.2f}]"
    print(f"{m:<26}{v['wer']*100:8.2f}{ci:>18}{v['cer']*100:8.2f}")

# paired: ours vs the best third-party model
A, B = "qwen3_adapter", "hishab_conformer_large"
ea, na, ca, nca, ka = store[A]
eb, nb, cb, ncb, kb = store[B]
assert ka == kb
g = defaultdict(list)
for i, k in enumerate(ka):
    g[k].append(i)
gi = [np.array(v) for v in g.values()]
rng = np.random.default_rng(20260903)
dw, dc = [], []
for _ in range(10000):
    pick = np.concatenate([gi[j] for j in rng.integers(0, len(gi), len(gi))])
    dw.append((ea[pick].sum() - eb[pick].sum()) / na[pick].sum())
    dc.append((ca[pick].sum() - cb[pick].sum()) / nca[pick].sum())
dw, dc = np.sort(dw), np.sort(dc)
paired = {
    "wer_delta_pp": round(float((ea.sum() - eb.sum()) / na.sum() * 100), 4),
    "wer_ci95_pp": [round(float(dw[250] * 100), 4), round(float(dw[9750] * 100), 4)],
    "cer_delta_pp": round(float((ca.sum() - cb.sum()) / nca.sum() * 100), 4),
    "cer_ci95_pp": [round(float(dc[250] * 100), 4), round(float(dc[9750] * 100), 4)]}
print("\npaired vs hishab_conformer_large (test-only):")
print(json.dumps(paired, indent=1))


# Speed on the same clean split, from the per-clip wall clock already recorded.
speeds = {}
for m in MODELS:
    rows = load(m)
    ms = [r["wall_clock_seconds"] * 1000 for r in rows
          if r.get("wall_clock_seconds") is not None]
    if ms:
        speeds[m] = {"ms_per_clip_mean": round(float(np.mean(ms)), 1),
                     "ms_per_clip_median": round(float(np.median(ms)), 1),
                     "ms_per_clip_p95": round(float(np.percentile(ms, 95)), 1),
                     "n": len(ms)}
print("\ntest-split speed (default runtime, from recorded per-clip wall clock)")
for m, v in speeds.items():
    print(f"  {m:<26}{v['ms_per_clip_mean']:9.1f} ms mean  "
          f"{v['ms_per_clip_median']:8.1f} median")

out = json.loads(Path("outputs_static/clean_test_table.json").read_text()) \
    if Path("outputs_static/clean_test_table.json").exists() else {}
out["speed_default_runtime"] = speeds
Path("outputs_a5000/clean_test_table.json").write_text(json.dumps(
    {"split": "FLEURS test only (920 utterances); zero sentence overlap with the "
              "984 h training corpus, verified by contamination_audit.py",
     "excluded": "FLEURS validation (402 utterances): 88 of its 150 distinct "
                 "sentences appear verbatim in the training corpus via fleurs_train, "
                 "which advantages the two models trained on it",
     "models": res, "paired_vs_best": paired,
     "speed_default_runtime": speeds}, indent=1))
print("\nwrote outputs_a5000/clean_test_table.json")
