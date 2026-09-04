#!/usr/bin/env python
"""Give Whisper medium the SAME runtime treatment as the adapter.

If we compile our model and not the baseline, then claim we beat it, the claim
is manufactured. Whisper medium runs the same eager HF generate loop on the
same box (1804 ms/clip, ~45 ms/token), so it plausibly gains from the same
static-cache + CUDA-graph treatment. This measures that, so the final report
can show both models as-shipped AND both optimised.
"""
import json, time
from pathlib import Path

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MID, REV = "SayedShaun/bengali-whisper-medium", "5a602253bb554e7508cb1a723888c8505d2e5b2f"
NCLIP = 8
proc = WhisperProcessor.from_pretrained(MID, revision=REV)
rows = json.loads(Path("eval_manifest.json").read_text())["rows"][:NCLIP]
clips = [sf.read(f"eval_fleurs_bn/{r['file']}", dtype="float32")[0] for r in rows]


def load():
    m = WhisperForConditionalGeneration.from_pretrained(
        MID, revision=REV, dtype=torch.float16).to("cuda").eval()
    # Same generation-config repair the benchmark backend applies.
    if getattr(m.generation_config, "no_timestamps_token_id", None) is None:
        ref = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-medium", dtype=torch.float16)
        m.generation_config = ref.generation_config
        del ref
    return m


def feats(wav):
    f = proc(wav, sampling_rate=16000, return_tensors="pt").input_features
    return f.to("cuda", dtype=torch.float16)


def transcripts(m):
    out = []
    for w in clips:
        with torch.inference_mode():
            o = m.generate(feats(w), language="bn", task="transcribe", do_sample=False)
        out.append(proc.batch_decode(o, skip_special_tokens=True)[0].strip())
    return out


def bench(m, reps=2):
    tot_ms, tot_tok = 0.0, 0
    for w in clips:
        f = feats(w)
        with torch.inference_mode():
            m.generate(f, language="bn", task="transcribe", do_sample=False)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(reps):
                o = m.generate(f, language="bn", task="transcribe", do_sample=False)
            torch.cuda.synchronize()
        tot_ms += (time.perf_counter() - t0) / reps * 1000
        tot_tok += o.shape[1]
    return tot_ms / len(clips), tot_tok / len(clips)


res, ref = {}, None
for tag, static, mode in (("eager_baseline", False, None),
                          ("static_cache", True, None),
                          ("compile_reduce_overhead", True, "reduce-overhead")):
    try:
        m = load()
        if static:
            m.generation_config.cache_implementation = "static"
        if mode:
            m.forward = torch.compile(m.forward, mode=mode)
        t = transcripts(m)
        if ref is None:
            ref = t
        ms, tok = bench(m)
        res[tag] = {"ms_per_clip": round(ms, 1), "tok_per_clip": round(tok, 1),
                    "ms_per_token": round(ms / max(tok, 1), 2),
                    "exact_match_vs_eager": f"{sum(a==b for a,b in zip(t,ref))}/{len(ref)}"}
        print(f"whisper {tag:<26}{ms:9.1f} ms/clip{tok:8.1f} tok"
              f"{ms/max(tok,1):8.2f} ms/token  exact {res[tag]['exact_match_vs_eager']}",
              flush=True)
        del m; torch.cuda.empty_cache()
    except Exception as e:
        res[tag] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(f"whisper {tag:<26}FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
        torch.cuda.empty_cache()

Path("outputs_batch").mkdir(exist_ok=True)
Path("outputs_batch/whisper_runtime.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
