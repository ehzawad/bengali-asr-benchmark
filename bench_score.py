#!/usr/bin/env python
"""Score every run in outputs/ and write one summary.

Accuracy is recomputed here from raw predictions with the SAME normaliser the
published benchmark used, so the headline WER and the intervals come from the
same hypotheses.

Two intervals are reported, deliberately:
  * per-utterance bootstrap  — matches the published report, kept for continuity
  * sentence-clustered bootstrap — resamples the 499 DISTINCT sentences rather
    than the 1,322 recordings. Several speakers read the same sentence, so the
    recordings are not independent and the per-utterance interval is optimistic.
Comparison claims must use the clustered interval on PAIRED differences.
"""
import json, re
from collections import defaultdict
from pathlib import Path

import jiwer
import unicodedata

import numpy as np

OUT = Path("outputs_a5000")
B = 10000
PUNCT = re.compile(r"[\"“”‘’'।,.!?;:()\[\]{}—–\-]")


def norm(t):
    """Scoring surface: punctuation to word boundaries, then Unicode NFC.

    NFC is not cosmetic here. Bengali writes several letters two ways -- য়
    as U+09DF or as U+09AF + U+09BC, and likewise ড় and ঢ় -- which are
    canonically equivalent but different byte sequences. The FLEURS references
    and the NeMo/Qwen models use the decomposed form; Whisper and wav2vec2 emit
    the precomposed one. Without normalising, every such character scores as a
    substitution and those two models are penalised for an encoding convention
    rather than for mistranscribing: Whisper medium measured 27.88% WER
    unnormalised against 16.23% normalised on the FLEURS test split.

    U+09DF is a Unicode composition exclusion, so NFC maps both spellings to
    the decomposed form; either NFC or NFD would work, NFC is the convention.
    """
    return unicodedata.normalize(
        "NFC", re.sub(r"\s+", " ", PUNCT.sub(" ", t)).strip())


def counts(preds):
    """Per-utterance word/char error and length counts, plus the sentence key
    each utterance belongs to (for clustered resampling)."""
    we, wn, ce, cn, keys, splits = [], [], [], [], [], []
    for p in preds:
        r, h = norm(p["raw_reference"]), norm(p["transcript"])
        o = jiwer.process_words([r], [h if h else "*"])
        we.append(o.substitutions + o.deletions + o.insertions)
        wn.append(len(r.split()))
        c = jiwer.process_characters([r], [h if h else "*"])
        ce.append(c.substitutions + c.deletions + c.insertions)
        cn.append(len(r))
        keys.append(r)
        splits.append(p.get("split", "?"))
    return (np.array(we), np.array(wn), np.array(ce), np.array(cn),
            np.array(keys), np.array(splits))


def boot(e, n, groups=None, seed=20260903):
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = np.arange(len(e))
        vals = [e[i].sum() / max(n[i].sum(), 1)
                for i in (rng.integers(0, len(idx), len(idx)) for _ in range(B))]
    else:
        buckets = defaultdict(list)
        for i, g in enumerate(groups):
            buckets[g].append(i)
        blocks = [np.array(v) for v in buckets.values()]
        k = len(blocks)
        vals = []
        for _ in range(B):
            pick = rng.integers(0, k, k)
            sel = np.concatenate([blocks[j] for j in pick])
            vals.append(e[sel].sum() / max(n[sel].sum(), 1))
    return [round(float(np.percentile(vals, 2.5)), 5),
            round(float(np.percentile(vals, 97.5)), 5)]


def main():
    rows = {}
    for d in sorted(OUT.iterdir()):
        f = d / "predictions.json"
        if not d.is_dir() or not f.exists() or d.name.startswith("_"):
            continue
        preds = json.loads(f.read_text())
        meta = json.loads((d / "run_meta.json").read_text())
        we, wn, ce, cn, keys, splits = counts(preds)
        rec = {
            "meta": meta, "n": len(preds),
            "wer": round(float(we.sum() / wn.sum()), 5),
            "cer": round(float(ce.sum() / cn.sum()), 5),
            "wer_ci_utterance": boot(we, wn),
            "wer_ci_sentence_clustered": boot(we, wn, groups=keys),
            "word_counts": {"errors": int(we.sum()), "ref_words": int(wn.sum())},
            "failures": meta.get("failures", 0),
            "empty_hyps": sum(1 for p in preds if not p["transcript"].strip()),
            "multi_piece_clips": sum(1 for p in preds if p.get("n_pieces", 1) > 1),
            "forced_cuts": sum(p.get("forced_cuts", 0) for p in preds),
        }
        for sp in ("test", "validation"):
            m = splits == sp
            if m.any():
                rec[f"wer_{sp}"] = round(float(we[m].sum() / wn[m].sum()), 5)
                rec[f"cer_{sp}"] = round(float(ce[m].sum() / cn[m].sum()), 5)
                rec[f"n_{sp}"] = int(m.sum())
        rows[d.name] = rec
        print(f"{d.name:24s} WER {rec['wer']*100:6.2f}%  "
              f"utt-CI [{rec['wer_ci_utterance'][0]*100:.2f}-{rec['wer_ci_utterance'][1]*100:.2f}]  "
              f"clustered [{rec['wer_ci_sentence_clustered'][0]*100:.2f}-"
              f"{rec['wer_ci_sentence_clustered'][1]*100:.2f}]  "
              f"CER {rec['cer']*100:.2f}%  {rec['meta']['ms_per_clip_mean']:.1f} ms/clip",
              flush=True)

    man = json.loads(Path("eval_manifest.json").read_text())
    summary = {
        "eval": {"dataset": man["dataset"], "config": man["config"],
                 "splits": man["splits"], "n": man["n"],
                 "distinct_references": len({r["reference"] for r in man["rows"]})},
        "protocol": {
            "gpu": "NVIDIA RTX A5000",
            "one_model_resident_at_a_time": True,
            "audio_path": "vendored asr_core.py, proven byte-identical across "
                          "both interpreters (SHA-256 over staged chunks)",
            "decoding": "greedy, batch 1, no external LM or rescoring",
            "speed_metric": "warmed in-process batch-1 inference time; includes "
                            "asr_core's temp-WAV staging, as in the original probe",
            "end_to_end_serving": None,
            "end_to_end_note": "The original harness that produced throughput, "
                               "queueing latency and failed-request counts was "
                               "never vendored and was unavailable; those columns "
                               "are absent rather than estimated.",
        },
        "checkpoints": json.loads((OUT / "checkpoint_revisions.json").read_text()),
        "models": rows,
    }
    Path("outputs_a5000/summary_a5000.json").write_text(json.dumps(summary, indent=1))
    print("\nwrote outputs_a5000/summary_a5000.json")


if __name__ == "__main__":
    main()
