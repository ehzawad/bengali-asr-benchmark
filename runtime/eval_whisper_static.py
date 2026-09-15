#!/usr/bin/env python
"""Whisper medium on the full 1,322 with the SAME static-cache treatment.

The 8-clip probe put Whisper at 476 ms/clip against our 1,420, but our number
is being remeasured on all 1,322 clips and drifted upward (short-clip probe
bias plus asr_core WAV staging). Publishing a 1,322-clip row for us against an
8-clip row for Whisper would flatter us by construction, so Whisper gets the
identical full-set treatment through the identical audio path.
"""
import json, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asr_core
from bench_score import norm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MID, REV = "SayedShaun/bengali-whisper-medium", "5a602253bb554e7508cb1a723888c8505d2e5b2f"
OUT = Path("outputs_static"); OUT.mkdir(exist_ok=True)


class WhisperStatic:
    def __init__(self):
        self.proc = WhisperProcessor.from_pretrained(MID, revision=REV)
        self.m = WhisperForConditionalGeneration.from_pretrained(
            MID, revision=REV, dtype=torch.float16).to("cuda").eval()
        if getattr(self.m.generation_config, "no_timestamps_token_id", None) is None:
            ref = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-medium", dtype=torch.float16)
            self.m.generation_config = ref.generation_config
            del ref
        self.m.generation_config.cache_implementation = "static"

    def transcribe(self, paths, batch_size=1, verbose=False):
        return [SimpleNamespace(text=self._one(_staged(p))) for p in paths]

    def _one(self, wav):
        f = self.proc(wav, sampling_rate=16000,
                      return_tensors="pt").input_features.to("cuda", torch.float16)
        with torch.inference_mode():
            o = self.m.generate(f, language="bn", task="transcribe", do_sample=False)
        return self.proc.batch_decode(o, skip_special_tokens=True)[0].strip()


def _staged(path):
    info = sf.info(str(path))
    assert (info.samplerate == 16000 and info.channels == 1
            and info.subtype == "PCM_16"), f"unexpected staged format: {info}"
    return sf.read(str(path), dtype="float32")[0]


def main():
    base = json.loads(Path("outputs/whisper_medium/predictions.json").read_text())
    rows = [{"file": b["file"], "text": b["reference"],
             "shipped": b["transcript"] or ""} for b in base]
    be = WhisperStatic()

    a0, _ = asr_core.load_audio(f"eval_fleurs_bn/{rows[0]['file']}")
    p0, _ = asr_core.chunks(a0)
    asr_core.decode(be, p0)                       # warm

    hyps, ms = [], []
    for i, r in enumerate(rows):
        audio, _ = asr_core.load_audio(f"eval_fleurs_bn/{r['file']}")
        pieces, _ = asr_core.chunks(audio)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        txt = asr_core.decode(be, pieces)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        hyps.append(txt)
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(rows)}  mean {np.mean(ms):7.1f} ms/clip", flush=True)

    import jiwer
    def errs(ref, hyp):
        m = jiwer.process_words([norm(ref)], [norm(hyp)])
        return m.substitutions + m.deletions + m.insertions, len(norm(ref).split())

    e_new, e_old, nref, ident = [], [], [], 0
    for r, h in zip(rows, hyps):
        en, n = errs(r["text"], h); eo, _ = errs(r["text"], r["shipped"])
        e_new.append(en); e_old.append(eo); nref.append(n)
        ident += (norm(h) == norm(r["shipped"]))
    e_new, e_old, nref = map(np.array, (e_new, e_old, nref))

    rep = {"model": "whisper_medium", "runtime": "static KV cache, greedy batch 1",
           "n": len(rows),
           "wer_static": round(float(e_new.sum() / nref.sum()), 5),
           "wer_shipped": round(float(e_old.sum() / nref.sum()), 5),
           "identical_transcripts": f"{ident}/{len(rows)}",
           "ms_per_clip_mean": round(float(np.mean(ms)), 1),
           "ms_per_clip_p95": round(float(np.percentile(ms, 95)), 1)}
    Path(OUT / "whisper_static_full.json").write_text(json.dumps(rep, indent=1))
    Path(OUT / "whisper_predictions.json").write_text(json.dumps(
        [{"file": r["file"], "ref": r["text"], "hyp": h}
         for r, h in zip(rows, hyps)], ensure_ascii=False, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
