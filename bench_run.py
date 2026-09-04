#!/usr/bin/env python
"""Run one Bengali ASR model over the 1,322-utterance FLEURS eval set.

Every model reads the SAME canonical PCM16/16 kHz files listed in
eval_manifest.json, in the same order, through the same loader in this file —
so the audio path is shared by construction and verifiable by the per-file
SHA-256 in the manifest. Decoding is sequential at batch 1 with a discarded
warm-up, which is the interactive-latency workload and is applied identically
to CTC and autoregressive models alike.

A model that raises on a clip yields an EMPTY hypothesis that stays in the
denominator; failures are never dropped.

  bench_run.py --kind nemo --model /path/to.nemo --label ehzawad_fastconformer
"""
import argparse, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import asr_core            # THE shared audio path, vendored from the benchmark
from types import SimpleNamespace

ASSETS = Path(".")
MANIFEST = ASSETS / "eval_manifest.json"


def transcribe_clip(backend, path):
    """load -> downmix/nan-scrub/resample -> 25 s chunking -> staged PCM16 ->
    model.transcribe(paths). Identical for every model, by construction."""
    audio, _ = asr_core.load_audio(str(path))
    pieces, forced = asr_core.chunks(audio)
    return asr_core.decode(backend, pieces), len(pieces), forced


# --------------------------------------------------------------- backends
class NeMoBackend:
    def __init__(self, model, **kw):
        import nemo.collections.asr as nemo_asr
        self.m = (nemo_asr.models.ASRModel.restore_from(model)
                  if str(model).endswith(".nemo")
                  else nemo_asr.models.ASRModel.from_pretrained(model))
        self.m = self.m.to("cuda").eval()

    def transcribe(self, paths, batch_size=1, verbose=False):
        return self.m.transcribe(list(paths), batch_size=batch_size,
                                 verbose=verbose)


class WhisperBackend:
    def __init__(self, model, revision=None, **kw):
        from transformers import (GenerationConfig,
                                  WhisperForConditionalGeneration,
                                  WhisperProcessor)
        self.proc = WhisperProcessor.from_pretrained(model, revision=revision)
        self.m = WhisperForConditionalGeneration.from_pretrained(
            model, revision=revision, torch_dtype=torch.float16).to("cuda").eval()
        self.refreshed = False
        try:  # some fine-tunes ship a pre-multilingual generation config
            self.m.generate(torch.zeros(1, 80, 3000, dtype=torch.float16,
                                        device="cuda"),
                            language="bn", task="transcribe", max_new_tokens=1)
        except Exception:
            self.m.generation_config = GenerationConfig.from_pretrained(
                "openai/whisper-medium")
            self.refreshed = True

    def transcribe(self, paths, batch_size=1, verbose=False):
        out = []
        for p in paths:
            wav = _read_staged(p)
            f = self.proc(wav, sampling_rate=16000, return_tensors="pt"
                          ).input_features.to("cuda", torch.float16)
            with torch.inference_mode():
                ids = self.m.generate(f, language="bn", task="transcribe",
                                      num_beams=1, do_sample=False,
                                      max_new_tokens=256)
            out.append(SimpleNamespace(text=self.proc.batch_decode(
                ids, skip_special_tokens=True)[0].strip()))
        return out


class Wav2Vec2Backend:
    def __init__(self, model, revision=None, **kw):
        from transformers import AutoProcessor, AutoModelForCTC
        self.proc = AutoProcessor.from_pretrained(model, revision=revision)
        self.m = AutoModelForCTC.from_pretrained(
            model, revision=revision).to("cuda").eval()

    def transcribe(self, paths, batch_size=1, verbose=False):
        out = []
        for p in paths:
            wav = _read_staged(p)
            inp = self.proc(wav, sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                logits = self.m(inp.input_values.to("cuda")).logits
            out.append(SimpleNamespace(text=self.proc.batch_decode(
                logits.argmax(-1).cpu().numpy())[0].strip()))
        return out


class QwenAdapterBackend:
    """Qwen3-ASR + Bengali LoRA/projector adapter.

    Bengali is absent from the base model's supported-language list, so
    apply_transcription_request() REJECTS it; the assistant turn must be
    prefilled by hand with "language Bengali<asr_text>" and that prefill
    stripped from the decoded text. Greedy, no rescoring.
    """
    PREFILL = "language Bengali<asr_text>"

    def __init__(self, model, adapter=None, revision=None, **kw):
        from peft import PeftModel
        from transformers import (AutoProcessor,
                                  Qwen3ASRForConditionalGeneration)
        self.proc = AutoProcessor.from_pretrained(model, revision=revision)
        base = Qwen3ASRForConditionalGeneration.from_pretrained(
            model, revision=revision, dtype=torch.bfloat16)
        self.m = PeftModel.from_pretrained(base, adapter).to("cuda").eval()

    def transcribe(self, paths, batch_size=1, verbose=False):
        return [SimpleNamespace(text=self._one(_read_staged(p))) for p in paths]

    def _one(self, wav):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": wav}]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": self.PREFILL}]}]
        inputs = self.proc.apply_chat_template(
            [conv], tokenize=True, return_dict=True, continue_final_message=True)
        inputs = {k: (v.to("cuda", dtype=self.m.dtype)
                      if hasattr(v, "to") and v.is_floating_point()
                      else v.to("cuda") if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.m.generate(**inputs, max_new_tokens=512,
                                  do_sample=False, num_beams=1, use_cache=True)
        txt = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0]
        if "<asr_text>" in txt:
            txt = txt.split("<asr_text>")[-1]
        return txt.strip()


def _read_staged(path):
    """Read a chunk staged by asr_core.decode. Asserts the staged format so a
    future change cannot silently route a model through different audio."""
    info = sf.info(str(path))
    assert (info.samplerate == 16000 and info.channels == 1
            and info.subtype == "PCM_16"), f"unexpected staged format: {info}"
    wav, _ = sf.read(str(path), dtype="float32")
    return wav


KINDS = {"nemo": NeMoBackend, "whisper": WhisperBackend,
         "wav2vec2": Wav2Vec2Backend, "qwen": QwenAdapterBackend}


def gpu_state():
    q = ("nvidia-smi --query-gpu=name,memory.used,utilization.gpu,clocks.sm,"
         "temperature.gpu --format=csv,noheader")
    try:
        return subprocess.check_output(q.split(), text=True).strip()
    except Exception:
        return "unavailable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=list(KINDS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--revision")
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    man = json.loads(MANIFEST.read_text())
    rows = man["rows"][:a.limit] if a.limit else man["rows"]
    out_dir = ASSETS / "outputs" / a.label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{a.label}] loading {a.kind}: {a.model}", flush=True)
    be = KINDS[a.kind](a.model, adapter=a.adapter, revision=a.revision)

    audio_dir = ASSETS / "eval_fleurs_bn"
    transcribe_clip(be, audio_dir / rows[0]["file"])   # warm-up, discarded
    torch.cuda.synchronize()

    preds, times, failures = [], [], 0
    gpu_before = gpu_state()
    t_all = time.perf_counter()
    for i, r in enumerate(rows):
        p = audio_dir / r["file"]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            (txt, npieces, forced), err = transcribe_clip(be, p), None
        except Exception as e:                      # failure => empty hypothesis
            txt, npieces, forced = "", 0, 0
            err = f"{type(e).__name__}: {e}"
            failures += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        preds.append({"file": r["file"], "split": r["split"],
                      "reference": r["reference"],
                      "raw_reference": r["raw_reference"],
                      "ok": err is None, "transcript": txt,
                      "wall_clock_seconds": round(dt, 6), "error": err,
                      "n_pieces": npieces, "forced_cuts": forced})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}  {np.mean(times)*1000:.1f} ms/clip",
                  flush=True)
    total = time.perf_counter() - t_all

    audio_s = sum(r["duration"] for r in rows)
    t = np.array(times)
    meta = {
        "label": a.label, "kind": a.kind, "model": a.model,
        "adapter": a.adapter, "revision": a.revision,
        "n": len(rows), "failures": failures,
        "audio_seconds": round(audio_s, 1),
        "ms_per_clip_mean": round(float(t.mean() * 1000), 2),
        "ms_per_clip_median": round(float(np.median(t) * 1000), 2),
        "ms_per_clip_p95": round(float(np.percentile(t, 95) * 1000), 2),
        "inference_wall_s": round(float(t.sum()), 1),
        "loop_wall_s": round(total, 1),
        "audio_s_per_wall_s": round(audio_s / float(t.sum()), 1),
        "rtf": round(float(t.sum()) / audio_s, 5),
        "batch_size": 1, "decoding": "greedy, no external LM or rescoring",
        "gpu_before": gpu_before, "gpu_after": gpu_state(),
        "torch": torch.__version__, "python": platform.python_version(),
        "manifest_sha256": hashlib.sha256(
            MANIFEST.read_bytes()).hexdigest(),
    }
    if a.kind == "whisper":
        meta["generation_config_refreshed"] = bool(getattr(be, "refreshed", False))
    (out_dir / "predictions.json").write_text(
        json.dumps(preds, ensure_ascii=False, indent=1))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[{a.label}] {meta['ms_per_clip_mean']} ms/clip  "
          f"{meta['audio_s_per_wall_s']}x real time  failures={failures}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
