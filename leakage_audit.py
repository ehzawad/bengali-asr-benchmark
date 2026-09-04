#!/usr/bin/env python
"""Leakage audit beyond text matching: audio fingerprints and speaker reuse.

contamination_audit.py compares normalised transcripts. That catches reused
sentences, but not the same recording appearing under a different transcript,
and it says nothing about speakers. This adds:

  1. exact audio fingerprints (SHA-256 over decoded 16 kHz mono float32 PCM)
     of every FLEURS test clip against every duration-compatible training clip
  2. near-duplicate screening on (duration, RMS), which survives re-encoding
  3. what can and cannot be said about speaker reuse

Only duration-compatible training clips are decoded: an identical recording
must have an identical duration, so this is exact for the exact-match question
while decoding 26% of the corpus instead of all 113 GB of it.
"""
import bisect
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

PIPELINE_ROOT = Path("/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline")
CORPUS = PIPELINE_ROOT / "data2/final_arm_b0_ready.json"
EVAL_DIR = Path("/mnt/sdb/arafat/ehz/llm/bench_assets/eval_fleurs_bn")
PRED = Path("outputs_a5000/qwen3_adapter/predictions.json")
TOL = 0.02


def fingerprint(path):
    try:
        a, sr = sf.read(str(path), dtype="float32")
    except Exception:
        return None
    if a.ndim > 1:
        a = a.mean(axis=1)
    h = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:24]
    return h, round(len(a) / sr, 2), round(float(np.sqrt((a ** 2).mean())), 4), str(path)


def main():
    if not CORPUS.exists():
        print(f"training corpus absent at {CORPUS}", file=sys.stderr)
        return 2

    ev = [r for r in json.loads(PRED.read_text()) if r["split"] == "test"]
    print(f"fingerprinting {len(ev)} FLEURS test clips ...", flush=True)
    with ProcessPoolExecutor(8) as ex:
        evfp = [f for f in ex.map(fingerprint, [EVAL_DIR / r["file"] for r in ev]) if f]
    ev_hash = {h: Path(p).name for h, _, _, p in evfp}
    ev_sig = defaultdict(list)
    for h, d, rms, p in evfp:
        ev_sig[(d, rms)].append(Path(p).name)
    durs = sorted({d for _, d, _, _ in evfp})
    print(f"  {len(ev_hash)} hashed, {len(durs)} distinct durations", flush=True)

    train = []
    with CORPUS.open() as f:
        for line in f:
            if line.strip():
                train.append(json.loads(line))

    def near(d):
        i = bisect.bisect_left(durs, d - TOL)
        return i < len(durs) and durs[i] <= d + TOL

    cand = []
    for r in train:
        try:
            if near(round(float(r["duration"]), 2)):
                p = Path(r["audio_filepath"])
                cand.append(p if p.is_absolute() else PIPELINE_ROOT / p)
        except (KeyError, TypeError, ValueError):
            continue
    print(f"training rows {len(train):,}; duration-compatible {len(cand):,} "
          f"({len(cand)/len(train)*100:.1f}%) -- decoding those", flush=True)

    exact, near_dup, done, missing = [], [], 0, 0
    with ProcessPoolExecutor(8) as ex:
        for fp in ex.map(fingerprint, cand, chunksize=64):
            done += 1
            if fp is None:
                missing += 1
            else:
                h, d, rms, p = fp
                if h in ev_hash:
                    exact.append((ev_hash[h], p))
                elif (d, rms) in ev_sig:
                    near_dup.append((ev_sig[(d, rms)][0], p))
            if done % 10000 == 0:
                print(f"  {done:,}/{len(cand):,}  exact={len(exact)} "
                      f"near={len(near_dup)}", flush=True)

    print(f"\ndecoded {done:,} ({missing:,} unreadable)")
    if missing:
        print("  NOTE: unreadable clips are IndicVoices -- its audio was deleted")
        print("  after corpus construction, so that source is covered by the TEXT")
        print("  audit (zero overlap) but NOT at the audio level. State that limit.")
    print(f"EXACT audio matches with FLEURS test:     {len(exact)}")
    print(f"near-duplicate (duration+RMS) candidates: {len(near_dup)}")
    for a, b in exact[:5]:
        print(f"   EXACT {a}  <-  {b}")
    for a, b in near_dup[:5]:
        print(f"   NEAR  {a}  <-  {b}")

    spk = Counter(r.get("speaker") or r.get("speaker_id") or "?" for r in train)
    print(f"\ntraining rows carrying a speaker id: "
          f"{sum(v for k, v in spk.items() if k != '?'):,} / {len(train):,}")
    print("The FLEURS manifest used here carries no speaker ids, so speaker")
    print("overlap cannot be excluded by identifier. The exact-audio result")
    print("above is the strongest available evidence; report that limit rather")
    print("than claiming speaker disjointness.")

    Path("outputs_a5000/leakage_audit.json").write_text(json.dumps(
        {"eval_clips": len(ev_hash), "training_rows": len(train),
         "duration_compatible_decoded": done,
         "exact_audio_matches": len(exact),
         "near_duplicate_candidates": len(near_dup),
         "speaker_ids_available": False}, indent=1))
    print("\nVERDICT:", "no exact audio leakage found" if not exact
          else "EXACT AUDIO LEAKAGE -- do not publish these numbers")
    return 0 if not exact else 1


if __name__ == "__main__":
    sys.exit(main())
