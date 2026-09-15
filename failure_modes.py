#!/usr/bin/env python
"""Failure-mode and latency-distribution report for the adapter.

A mean WER and a mean ms/clip hide the behaviours that actually decide whether
a model is usable: does it hallucinate, loop, return nothing, hit the token
cap, or drift into the wrong script? And a mean latency says nothing about the
tail a user waits on. This reports all of it from the saved predictions.
"""
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_score import norm

DEVANAGARI = re.compile(r"[ऀ-ॣ०-ॿ]")   # excludes danda U+0964/5
BENGALI = re.compile(r"[ঀ-৿]")
LATIN = re.compile(r"[A-Za-z]")


def longest_run(words):
    """Longest run of one repeated token -- the classic AR degeneration."""
    best = run = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        best = max(best, run)
    return best


def report(name, split="test"):
    rows = [r for r in json.loads(Path(f"outputs_a5000/{name}/predictions.json").read_text())
            if r["split"] == split]
    n = len(rows)
    empty = ratio_hi = ratio_lo = deva = latin = loops = failed = 0
    ratios, ms = [], []
    for r in rows:
        hyp = (r["transcript"] or "").strip()
        ref = r["reference"]
        if r.get("error") or not r.get("ok", True):
            failed += 1
        if not hyp:
            empty += 1
            continue
        h, rf = norm(hyp).split(), norm(ref).split()
        if rf:
            q = len(h) / len(rf)
            ratios.append(q)
            ratio_hi += q > 1.5           # likely hallucination / repetition
            ratio_lo += q < 0.5           # likely truncation / dropped speech
        if DEVANAGARI.search(hyp):
            deva += 1
        if LATIN.search(hyp):
            latin += 1
        if h and longest_run(h) >= 4:
            loops += 1
        if r.get("wall_clock_seconds") is not None:
            ms.append(r["wall_clock_seconds"] * 1000)

    print(f"\n=== {name} ({split}, n={n}) ===")
    print(f"  failed requests          {failed}")
    print(f"  empty hypotheses         {empty}")
    print(f"  >=4x repeated token      {loops}   (autoregressive looping)")
    print(f"  length ratio > 1.5       {ratio_hi}   (hallucination / insertion)")
    print(f"  length ratio < 0.5       {ratio_lo}   (truncation / dropped speech)")
    print(f"  contains Devanagari      {deva}   (script leakage; danda excluded)")
    print(f"  contains Latin           {latin}")
    if ratios:
        print(f"  hyp/ref length ratio     mean {np.mean(ratios):.3f}  "
              f"p05 {np.percentile(ratios,5):.3f}  p95 {np.percentile(ratios,95):.3f}")
    if ms:
        a = np.array(ms)
        print(f"  latency ms/clip          mean {a.mean():8.1f}  p50 {np.percentile(a,50):8.1f}"
              f"  p95 {np.percentile(a,95):8.1f}  p99 {np.percentile(a,99):8.1f}"
              f"  max {a.max():8.1f}")
    return {"n": n, "failed": failed, "empty": empty, "loops": loops,
            "hallucination_gt1_5x": ratio_hi, "truncation_lt0_5x": ratio_lo,
            "devanagari": deva, "latin": latin,
            "latency_ms": ({"mean": round(float(np.mean(ms)), 1),
                            "p50": round(float(np.percentile(ms, 50)), 1),
                            "p95": round(float(np.percentile(ms, 95)), 1),
                            "p99": round(float(np.percentile(ms, 99)), 1),
                            "max": round(float(np.max(ms)), 1)} if ms else None)}


if __name__ == "__main__":
    out = {m: report(m) for m in ["qwen3_adapter", "whisper_medium",
                                  "hishab_conformer_large"]}
    Path("outputs_a5000/failure_modes.json").write_text(json.dumps(out, indent=1))
    print("\nwrote outputs_a5000/failure_modes.json")
