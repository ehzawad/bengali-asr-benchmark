#!/usr/bin/env python
"""Decode benchmark manifests with Voxtral Mini bn-BD under vLLM, writing predictions.json
and run_meta.json in bench_run_set.py's format. This produced the voxtral_mini_bn rows.

The audio path is the benchmark's, unchanged: asr_core.load_audio -> asr_core.chunks
(<= 25 s, pause-seeking) -> each piece staged as PCM16 16 kHz WAV exactly as asr_core.decode
stages it, then read back as float32 as bench_run reads it. Pieces of one clip are joined
with a single space after stripping, as asr_core.decode joins them.

The prompt is the one vLLM's own transcription endpoint builds for Voxtral
(VoxtralForConditionalGeneration.get_generation_prompt: mistral_common encode_transcription,
language "bn", audio passed as lossless WAV). Decoding is greedy with max_tokens 500.

What differs from bench_run_set.py: pieces are decoded in vLLM's continuous batches (up to
--max-num-seqs at once, eager mode), so there is no per-clip latency here; time_models.py
measures batch-1 latency separately. Per-file SHA-256 is verified against the manifest
first (a set with any mismatch is refused, exit 5), and an existing run_meta.json is never
overwritten.

  bench_run_voxtral_vllm.py --bench-dir . --model <Mistral-format model dir> --hf-revision <rev> \\
      --manifest eval_manifest.json --audio-dir <FLEURS wav dir> --out-root outputs_fleurs --label voxtral_mini_bn
  bench_run_voxtral_vllm.py --bench-dir . --model <dir> --hf-revision <rev> --jobs jobs.json
      (jobs.json: [{"manifest", "label", "out_root", "audio_dir"}, ...], one engine for all)
"""
import argparse
import gc
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def staged(piece):
    """asr_core.decode's staging (clip, PCM16 WAV) + bench_run._read_staged's read."""
    buf = io.BytesIO()
    sf.write(buf, np.clip(piece, -1.0, 1.0), TARGET_SR, subtype="PCM_16", format="WAV")
    buf.seek(0)
    wav, _ = sf.read(buf, dtype="float32")
    return wav


def gpu_state():
    q = "nvidia-smi --query-gpu=name,memory.used,utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader"
    try:
        return subprocess.check_output(q.split(), text=True).strip()
    except Exception:
        return "unavailable"


def prepare(job, asr_core, limit):
    """sha-verify and stage one manifest; returns None (skip) or "refused"."""
    man_path = Path(job["manifest"])
    man = json.loads(man_path.read_text())
    rows = man["rows"][:limit] if limit else man["rows"]
    out_dir = Path(job["out_root"]) / job["label"]
    if (out_dir / "run_meta.json").exists():
        print(f"[{job['label']}] {out_dir}/run_meta.json exists; skipping (never overwritten)", flush=True)
        return None

    def resolve(r):
        p = Path(r["file"])
        return p if p.is_absolute() else Path(job.get("audio_dir") or ".") / p

    field = job.get("sha_field", "sha256")
    bad = [str(resolve(r)) for r in rows if sha256_file(resolve(r)) != r[field]]
    if bad:
        print(f"[{job['label']}] REFUSING: {len(bad)} file(s) do not match the manifest {field}, e.g. {bad[0]}",
              flush=True)
        return "refused"
    t = time.perf_counter()
    items, info = [], []
    for i, r in enumerate(rows):
        audio, note = asr_core.load_audio(str(resolve(r)))
        pieces, forced = asr_core.chunks(audio)
        info.append({"n_pieces": len(pieces), "forced_cuts": forced, "note": note})
        for k, piece in enumerate(pieces):
            items.append((i, k, staged(piece)))
    prep_s = time.perf_counter() - t
    print(f"[{job['label']}] {len(rows)} clips -> {len(items)} pieces staged in {prep_s:.1f}s", flush=True)
    return {"man": man, "man_path": man_path, "rows": rows, "out_dir": out_dir, "items": items,
            "info": info, "prep_s": prep_s, "sha_field": field}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True, help="this repository (for asr_core)")
    ap.add_argument("--manifest", help="one job: manifest path")
    ap.add_argument("--audio-dir", default=None, help="one job: prefix for relative 'file' entries (FLEURS)")
    ap.add_argument("--out-root", help="one job: output root")
    ap.add_argument("--label", help="one job: run label")
    ap.add_argument("--jobs", help="JSON list of {manifest, label, out_root, audio_dir}; one engine for all")
    ap.add_argument("--model", required=True, help="Mistral-format model directory (consolidated.safetensors, "
                                                    "params.json, tekken.json)")
    ap.add_argument("--hf-revision", default=None, help="provenance: the HF snapshot of --model")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    jobs = (json.loads(Path(a.jobs).read_text()) if a.jobs else
            [{"manifest": a.manifest, "label": a.label, "out_root": a.out_root, "audio_dir": a.audio_dir}])
    sys.path.insert(0, str(Path(a.bench_dir).resolve()))
    import asr_core  # the benchmark's single audio path

    prepared = [(job, prepare(job, asr_core, a.limit)) for job in jobs]
    todo = [(job, p) for job, p in prepared if isinstance(p, dict)]
    refused = [job["label"] for job, p in prepared if p == "refused"]
    if not todo:
        print("nothing to decode", flush=True)
        return 5 if refused else 0

    import torch
    import vllm
    from mistral_common.protocol.transcription.request import TranscriptionRequest
    from mistral_common.tokens.tokenizers.audio import Audio
    from vllm import LLM, SamplingParams

    t_load = time.perf_counter()
    llm = LLM(model=a.model, tokenizer_mode="mistral", config_format="mistral", load_format="mistral",
              gpu_memory_utilization=a.gpu_memory_utilization, max_num_seqs=a.max_num_seqs,
              max_model_len=a.max_model_len, max_num_batched_tokens=a.max_num_batched_tokens,
              limit_mm_per_prompt={"audio": 1}, enforce_eager=True, seed=0)
    load_s = time.perf_counter() - t_load
    instruct = llm.get_tokenizer().instruct
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)

    def build_prompts(items):
        prompts = []
        for _, _, wav in items:
            audio = Audio(wav, TARGET_SR, format="wav")  # lossless, as get_generation_prompt does
            req = TranscriptionRequest(model=a.model, audio=audio.to_base64(audio.format), language="bn")
            tok = instruct.encode_transcription(req)
            prompts.append({"prompt_token_ids": tok.tokens,
                            "multi_modal_data": {"audio": [(x.audio_array, TARGET_SR) for x in tok.audios]}})
        return prompts

    for job, p in todo:
        label, items, rows, out_dir = job["label"], p["items"], p["rows"], p["out_dir"]
        prompts = build_prompts(items)  # per set, so host memory stays bounded
        out_dir.mkdir(parents=True, exist_ok=True)
        gpu_before = gpu_state()
        t_gen = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=True)
        gen_s = time.perf_counter() - t_gen
        gpu_after = gpu_state()

        texts, finish, ntok = {}, {}, {}
        for (i, k, _), o in zip(items, outs):
            c = o.outputs[0]
            texts[(i, k)], finish[(i, k)], ntok[(i, k)] = c.text.strip(), c.finish_reason, len(c.token_ids)

        preds = []
        for i, r in enumerate(rows):
            n = p["info"][i]["n_pieces"]
            parts = [texts[(i, k)] for k in range(n)]
            preds.append({"file": r["file"], "utt_id": r.get("utt_id"), "split": r.get("split"),
                          "stratum": r.get("stratum"), "reference": r["reference"],
                          "raw_reference": r["raw_reference"], "ok": True,
                          "transcript": " ".join(t for t in parts if t),
                          "wall_clock_seconds": None, "error": None,
                          "n_pieces": n, "forced_cuts": p["info"][i]["forced_cuts"],
                          "piece_texts": parts,
                          "gen_tokens": [ntok[(i, k)] for k in range(n)],
                          "finish_reasons": [finish[(i, k)] for k in range(n)]})

        audio_s = float(sum(float(r["duration"]) for r in rows))
        meta = {
            "label": label, "kind": "voxtral_vllm", "model": a.model, "hf_revision": a.hf_revision,
            "dataset": p["man"].get("dataset"), "manifest": str(p["man_path"].resolve()),
            "manifest_sha256": sha256_file(p["man_path"]), "status": "complete",
            "audio_sha_field_verified": p["sha_field"], "audio_rebuild": p["man"].get("audio_rebuild"),
            "n": len(preds), "n_manifest": len(rows), "failures": 0, "pieces": len(items),
            "audio_seconds": round(audio_s, 1),
            "ms_per_clip_mean": None, "ms_per_clip_median": None, "ms_per_clip_p95": None,
            "latency_note": "not measured here: pieces decoded in vLLM continuous batches; see time_models.py",
            "generate_wall_s": round(gen_s, 1), "audio_prep_s": round(p["prep_s"], 1),
            "engine_load_s": round(load_s, 1), "audio_s_per_wall_s": round(audio_s / gen_s, 2),
            "max_num_batched_tokens": a.max_num_batched_tokens,
            "gen_tokens_total": int(sum(ntok.values())),
            "finish_reasons": {fr: sum(1 for v in finish.values() if v == fr) for fr in set(finish.values())},
            "decoding": f"greedy (temperature 0), max_tokens {a.max_tokens}, no external LM or rescoring",
            "batching": f"vLLM continuous batching, max_num_seqs {a.max_num_seqs}",
            "engine": {"vllm": vllm.__version__, "gpu_memory_utilization": a.gpu_memory_utilization,
                       "max_model_len": a.max_model_len, "enforce_eager": True,
                       "tokenizer_mode": "mistral", "load_format": "mistral"},
            "gpu_before": gpu_before, "gpu_after": gpu_after,
            "harness": "bench_run_voxtral_vllm.py over asr_core (load_audio, chunks, PCM16 staging)",
            "torch": torch.__version__, "python": platform.python_version(),
        }
        (out_dir / "predictions.json").write_text(json.dumps(preds, ensure_ascii=False, indent=1))
        (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
        p["items"] = None
        del prompts, outs
        gc.collect()
        print(f"[{label}] complete: {len(preds)} clips, {len(items)} pieces, generate {gen_s:.0f}s, "
              f"finish={meta['finish_reasons']}", flush=True)

    if refused:
        print(f"refused (sha mismatch): {refused}", flush=True)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
