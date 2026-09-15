#!/usr/bin/env python
"""Is the adapter's CER lead over hishab Conformer Large real?

Independent CIs that overlap do NOT settle this: the two models are scored on
the SAME utterances, so the comparison must be paired. Bootstrap the per-
utterance difference in character errors, clustered by distinct reference
sentence (the 1,322 recordings contain only 499 distinct sentences, so
resampling recordings independently understates the variance).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from bench_score import norm

A, B = "qwen3_adapter", "hishab_conformer_large"


def load(name):
    d = json.loads(Path(f"outputs/{name}/predictions.json").read_text())
    return {r["file"]: (r["reference"], r["transcript"] or "") for r in d}


def cer_counts(ref, hyp):
    import jiwer
    m = jiwer.process_characters([norm(ref)], [norm(hyp)])
    return m.substitutions + m.deletions + m.insertions, len(norm(ref))


a, b = load(A), load(B)
files = sorted(set(a) & set(b))
ea, eb, n, key = [], [], [], []
for f in files:
    ra, ha = a[f]; rb, hb = b[f]
    assert norm(ra) == norm(rb), f"reference mismatch on {f}"
    x, c = cer_counts(ra, ha); y, _ = cer_counts(rb, hb)
    ea.append(x); eb.append(y); n.append(c); key.append(norm(ra))
ea, eb, n = map(np.array, (ea, eb, n))

groups = defaultdict(list)
for i, k in enumerate(key):
    groups[k].append(i)
gidx = [np.array(v) for v in groups.values()]

cer_a, cer_b = ea.sum() / n.sum(), eb.sum() / n.sum()
rng = np.random.default_rng(20260903)
deltas = []
for _ in range(10000):
    pick = np.concatenate([gidx[j] for j in rng.integers(0, len(gidx), len(gidx))])
    deltas.append((ea[pick].sum() - eb[pick].sum()) / n[pick].sum())
deltas = np.sort(deltas)

rep = {"a": A, "b": B, "n_utterances": len(files), "n_sentence_clusters": len(gidx),
       "cer_a": round(float(cer_a), 5), "cer_b": round(float(cer_b), 5),
       "delta_pp_a_minus_b": round(float((cer_a - cer_b) * 100), 4),
       "delta_ci95_pp": [round(float(deltas[250] * 100), 4),
                         round(float(deltas[9750] * 100), 4)],
       "p_a_better": round(float((deltas < 0).mean()), 4)}
rep["significant_at_95"] = bool(deltas[250] < 0 and deltas[9750] < 0)
Path("outputs_static").mkdir(exist_ok=True)
Path("outputs_static/paired_cer.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
