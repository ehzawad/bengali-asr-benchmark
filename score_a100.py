#!/usr/bin/env python
"""Summarise the same-GPU latency runs (outputs_a100/<label>.json, written by time_models.py)
into outputs_a100/summary.json, and score Voxtral Mini's batch-1 timing transcripts against its
batched accuracy run (outputs_fleurs/voxtral_mini_bn) with the benchmark's own scorer.

All six models were timed on one Colab A100-SXM4-40GB on the 920 FLEURS test clips, batch 1,
after one discarded warm-up clip, each clip between CUDA synchronisations and including
asr_core's loading, chunking and PCM16 staging. These time the configured pipelines (NeMo,
transformers or vLLM, each with its own precision and per-call overhead), not the models in
isolation; each row is one pass.
"""
import json
import sys
from pathlib import Path

import jiwer
import numpy as np

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from bench_score import norm  # noqa: E402  (the published scorer, unchanged)

OUT = A / "outputs_a100"
ORDER = ["fastconformer_ctc", "hishab_conformer_large", "whisper_medium", "wav2vec2", "hishab_fastconformer",
         "voxtral_mini_bn"]
NOTES = [
    "One Colab A100-SXM4-40GB; FLEURS test (920 clips); batch 1; one warm-up clip discarded; per-clip wall time "
    "between torch.cuda.synchronize() calls, including asr_core load, chunking and PCM16 staging.",
    "The five published models ran through bench_run.py's backends at the pinned revisions (torch 2.11 cu128, "
    "NeMo 2.7.3; Whisper in FP16). Voxtral Mini ran through vLLM 0.31.0 (torch 2.13 cu130), one request at a "
    "time, with CUDA graphs. Frameworks, precision and per-call overheads differ.",
    "A low-priority, single-threaded scoring process shared the VM's CPUs while fastconformer_ctc, "
    "hishab_conformer_large, hishab_fastconformer and wav2vec2 were timed; its effect was not measured.",
    "One pass per model: differences of a few milliseconds are not an established ordering, and p95 is variation "
    "across clips, not uncertainty in the mean. Startup is excluded.",
    "identical_to_published_transcripts compares each model's A100 transcripts with its committed A5000 "
    "predictions; Voxtral Mini has no A5000 run, so its timing transcripts are compared with its batched "
    "accuracy run instead (a different engine configuration: eager, up to 256 pieces at once).",
]


def errors(pairs):
    we = wn = ce = cn = 0
    for ref, hyp in pairs:
        r, h = norm(ref), norm(hyp)
        o = jiwer.process_words([r], [h if h else "*"])
        c = jiwer.process_characters([r], [h if h else "*"])
        we += o.substitutions + o.deletions + o.insertions; wn += len(r.split())
        ce += c.substitutions + c.deletions + c.insertions; cn += len(r)
    return {"word_errors": we, "ref_words": wn, "wer": round(we / wn, 6), "char_errors": ce, "ref_chars": cn,
            "cer": round(ce / cn, 6)}


def main():
    models = {}
    for k in ORDER:
        f = OUT / f"{k}.json"
        if not f.exists():
            continue
        t = json.loads(f.read_text())
        models[k] = {key: t.get(key) for key in ("model", "revision", "kind", "n", "ms_per_clip_mean",
                                                   "ms_per_clip_median", "ms_per_clip_p95", "rtf",
                                                   "model_load_s", "identical_to_published_transcripts",
                                                   "gpu_after", "torch", "vllm")}
    summary = {"gpu": "NVIDIA A100-SXM4-40GB (Colab)", "date": "2026-10-13", "models": models, "notes": NOTES}
    mini = OUT / "voxtral_mini_bn.json"
    if mini.exists():
        timed = {c["file"]: c["transcript"] for c in json.loads(mini.read_text())["per_clip"]}
        acc = [p for p in json.loads((A / "outputs_fleurs/voxtral_mini_bn/predictions.json").read_text())
               if p["split"] == "test"]
        summary["voxtral_mini_bn_timing_vs_accuracy_run"] = {
            "identical_transcripts": f"{sum(timed[p['file']] == p['transcript'] for p in acc)}/{len(acc)}",
            "accuracy_run": errors([(p["raw_reference"], p["transcript"]) for p in acc]),
            "timing_run": errors([(p["raw_reference"], timed[p["file"]]) for p in acc])}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    for k, m in models.items():
        print(f"{k:24s} {m['ms_per_clip_mean']:9.2f} ms mean {m['ms_per_clip_median']:9.2f} median "
              f"{m['ms_per_clip_p95']:9.2f} p95  {1 / m['rtf']:6.0f}x real time  "
              f"identical {m['identical_to_published_transcripts'] or '-'}")
    if "voxtral_mini_bn_timing_vs_accuracy_run" in summary:
        print("voxtral_mini_bn timing vs accuracy run:", json.dumps(summary["voxtral_mini_bn_timing_vs_accuracy_run"]))
    print(f"wrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
