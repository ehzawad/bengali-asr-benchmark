#!/usr/bin/env python
"""Batch-1 latency of one ASR model on this GPU, with bench_run.py's own protocol:
the backends and transcribe_clip are imported from the benchmark (asr_core load ->
<=25 s chunks -> PCM16 staging -> backend.transcribe); a warm-up clip is decoded and
discarded; each clip is timed between torch.cuda.synchronize() calls. Every model is
timed on the same FLEURS test clips, one model per process.

Kinds: nemo | whisper | wav2vec2 (bench_run backends) and voxtral_vllm (vLLM, one
request at a time, greedy, language "bn", 500-token cap, CUDA graphs on). This produced
outputs_a100/*.json; score_a100.py summarises them.

  time_models.py --bench-dir . --manifest eval_manifest.json --audio-dir <FLEURS wav dir> \
                 --kind K --model M [--revision R] --label L --out outputs_a100/L.json [--published-label P]
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def gpu_state():
    q = "nvidia-smi --query-gpu=name,memory.used,utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader"
    try:
        return subprocess.check_output(q.split(), text=True).strip()
    except Exception:
        return "unavailable"


class VoxtralVLLM:
    """vLLM, one request per staged piece (batch 1)."""

    def __init__(self, model, **kw):
        import vllm
        from mistral_common.protocol.transcription.request import TranscriptionRequest
        try:
            from mistral_common.tokens.tokenizers.audio import Audio
        except ImportError:
            from mistral_common.audio import Audio
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, tokenizer_mode="mistral", config_format="mistral", load_format="mistral",
                       gpu_memory_utilization=0.85, max_model_len=2048, max_num_seqs=8,
                       limit_mm_per_prompt={"audio": 1}, seed=0)
        self.instruct = self.llm.get_tokenizer().instruct
        self.sp = SamplingParams(temperature=0.0, max_tokens=500)
        self.TR, self.Audio, self.model, self.version = TranscriptionRequest, Audio, model, vllm.__version__

    def transcribe(self, paths, batch_size=1, verbose=False):
        out = []
        for p in paths:
            wav = BR._read_staged(p)
            a = self.Audio(wav, 16000, format="wav")
            tok = self.instruct.encode_transcription(
                self.TR(model=self.model, audio=a.to_base64(a.format), language="bn"))
            prompt = {"prompt_token_ids": tok.tokens,
                      "multi_modal_data": {"audio": [(x.audio_array, 16000) for x in tok.audios]}}
            o = self.llm.generate([prompt], self.sp, use_tqdm=False)[0].outputs[0]
            out.append(SimpleNamespace(text=o.text.strip(), n_tokens=len(o.token_ids)))
        return out


def main():
    global BR
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-dir", default=None, help="prefix for relative 'file' entries")
    ap.add_argument("--kind", required=True, choices=["nemo", "whisper", "wav2vec2", "voxtral_vllm"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--published-label", default=None, help="outputs_fleurs/<label> to compare transcripts with")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    bench = Path(a.bench_dir).resolve()
    sys.path.insert(0, str(bench))
    import bench_run as BR  # the benchmark's backends and transcribe_clip, unchanged

    man = json.loads(Path(a.manifest).read_text())
    rows = [r for r in man["rows"] if r["split"] == "test"]
    rows = rows[: a.limit] if a.limit else rows
    for r in rows:
        if a.audio_dir and not Path(r["file"]).is_absolute():
            r["file"] = str(Path(a.audio_dir) / r["file"])

    t_load = time.perf_counter()
    if a.kind == "voxtral_vllm":
        be = VoxtralVLLM(a.model)
    else:
        be = BR.KINDS[a.kind](a.model, revision=a.revision)
    load_s = time.perf_counter() - t_load

    BR.transcribe_clip(be, Path(rows[0]["file"]))   # warm-up, discarded (as bench_run.py does)
    torch.cuda.synchronize()

    times, texts = [], []
    gpu_before = gpu_state()
    for r in rows:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        txt, _, _ = BR.transcribe_clip(be, Path(r["file"]))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        texts.append(txt)
    gpu_after = gpu_state()

    t = np.array(times)
    audio_s = float(sum(float(r["duration"]) for r in rows))
    res = {"label": a.label, "kind": a.kind, "model": a.model, "revision": a.revision, "n": len(rows),
           "ms_per_clip_mean": round(float(t.mean() * 1000), 2),
           "ms_per_clip_median": round(float(np.median(t) * 1000), 2),
           "ms_per_clip_p95": round(float(np.percentile(t, 95) * 1000), 2),
           "inference_wall_s": round(float(t.sum()), 1), "audio_seconds": round(audio_s, 1),
           "rtf": round(float(t.sum()) / audio_s, 5), "model_load_s": round(load_s, 1),
           "protocol": "bench_run.py: warm-up discarded; per clip cuda-synchronised wall time of transcribe_clip "
                       "(asr_core load, chunks, PCM16 staging, backend.transcribe); batch 1; greedy",
           "gpu_before": gpu_before, "gpu_after": gpu_after, "torch": torch.__version__,
           "python": platform.python_version(),
           "per_clip": [{"file": Path(r["file"]).name, "seconds": round(s, 6), "transcript": x}
                        for r, s, x in zip(rows, times, texts)]}
    if a.kind == "voxtral_vllm":
        res["vllm"] = be.version
    if a.published_label:
        pub = {Path(p["file"]).name: p["transcript"] for p in json.loads(
            (bench / "outputs_fleurs" / a.published_label / "predictions.json").read_text())}
        same = sum(1 for r, x in zip(rows, texts) if pub.get(Path(r["file"]).name) == x)
        res["identical_to_published_transcripts"] = f"{same}/{len(rows)}"
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[{a.label}] {res['ms_per_clip_mean']} ms/clip mean, {res['ms_per_clip_median']} median, "
          f"p95 {res['ms_per_clip_p95']}; RTF {res['rtf']}; "
          f"published-identical {res.get('identical_to_published_transcripts', 'n/a')}", flush=True)


if __name__ == "__main__":
    main()
