#!/usr/bin/env python
"""Paired WER: adapter vs hishab Conformer Large on the same 1,322 utterances.

The README claims the two are "statistically indistinguishable" on the basis
that their independent confidence intervals overlap. Overlapping independent
intervals are a weak and often misleading test -- two models scored on the SAME
utterances must be compared pairwise. This does that, clustered by the 499
distinct reference sentences.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from bench_score import norm

import sys
A, B = sys.argv[1], sys.argv[2]


def load(n):
    """Test split only: the validation split is contaminated for our models."""
    d = json.loads(Path(f"outputs/{n}/predictions.json").read_text())
    return {r["file"]: (r["reference"], r["transcript"] or "")
            for r in d if r["split"] == "test"}


def werc(ref, hyp):
    import jiwer
    m = jiwer.process_words([norm(ref)], [norm(hyp)])
    return m.substitutions + m.deletions + m.insertions, len(norm(ref).split())


a, b = load(A), load(B)
files = sorted(set(a) & set(b))
ea, eb, n, key = [], [], [], []
for f in files:
    ra, ha = a[f]; rb, hb = b[f]
    assert norm(ra) == norm(rb)
    x, c = werc(ra, ha); y, _ = werc(rb, hb)
    ea.append(x); eb.append(y); n.append(c); key.append(norm(ra))
ea, eb, n = map(np.array, (ea, eb, n))

groups = defaultdict(list)
for i, k in enumerate(key):
    groups[k].append(i)
gidx = [np.array(v) for v in groups.values()]

rng = np.random.default_rng(20260903)
d = []
for _ in range(10000):
    pick = np.concatenate([gidx[j] for j in rng.integers(0, len(gidx), len(gidx))])
    d.append((ea[pick].sum() - eb[pick].sum()) / n[pick].sum())
d = np.sort(d)

rep = {"a": A, "b": B, "n": len(files), "clusters": len(gidx),
       "wer_a": round(float(ea.sum() / n.sum()), 5),
       "wer_b": round(float(eb.sum() / n.sum()), 5),
       "delta_pp_a_minus_b": round(float((ea.sum() - eb.sum()) / n.sum() * 100), 4),
       "delta_ci95_pp": [round(float(d[250] * 100), 4), round(float(d[9750] * 100), 4)],
       "p_a_better": round(float((d < 0).mean()), 4)}
rep["b_significantly_better"] = bool(d[250] > 0)
Path(f"outputs_static/paired_wer_{A}_vs_{B}.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
