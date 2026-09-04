#!/usr/bin/env python
"""Localise the per-clip cost: audio encoder vs prefill vs per-token decode.

At ~78 generated tokens and ~9,000 ms/clip we see ~115 ms/token, roughly 20x
worse than weight-bandwidth arithmetic predicts for a 1.7B bf16 model on this
card. This splits the clip into its parts so the fix targets the real cost:

  t_encoder   : audio tower + projector alone
  t_prefill   : one forward over the full prompt (audio tokens + text prefill)
  t_1tok      : generate(max_new_tokens=1)  -> encoder + prefill + 1 decode step
  t_ntok      : generate(max_new_tokens=N)  -> slope gives true per-token cost

Per-token cost is then (t_ntok - t_1tok) / (N - 1), which excludes the one-off
encoder and prefill entirely.
"""
import json, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

P = "/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline"
BASE, REV = "Qwen/Qwen3-ASR-1.7B-hf", "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
ADAPTER = f"{P}/experiments/qwen_final/candidates/step_17280"
PREFILL = "language Bengali<asr_text>"
NCLIP = 6

proc = AutoProcessor.from_pretrained(BASE, revision=REV)
rows = json.loads(Path("eval_manifest.json").read_text())["rows"][:NCLIP + 1]
clips = [sf.read(f"eval_fleurs_bn/{r['file']}", dtype="float32")[0] for r in rows]

base = Qwen3ASRForConditionalGeneration.from_pretrained(
    BASE, revision=REV, dtype=torch.bfloat16).to("cuda").eval()
model = PeftModel.from_pretrained(base, ADAPTER).eval()
inner = model.base_model.model          # Qwen3ASRForConditionalGeneration


def prep(wav):
    conv = [{"role": "user", "content": [{"type": "audio", "audio": wav}]},
            {"role": "assistant", "content": [{"type": "text", "text": PREFILL}]}]
    d = proc.apply_chat_template([conv], tokenize=True, return_dict=True,
                                 continue_final_message=True)
    return {k: (v.to("cuda", dtype=torch.bfloat16)
                if hasattr(v, "to") and v.is_floating_point()
                else v.to("cuda") if hasattr(v, "to") else v)
            for k, v in d.items()}


def timeit(fn, reps=3):
    fn()                                   # warm
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


res = {"per_clip": []}
for i, wav in enumerate(clips[1:], 1):
    inp = prep(wav)
    n_in = inp["input_ids"].shape[1]

    with torch.inference_mode():
        t_enc = timeit(lambda: inner.model.audio_tower(
            inp["input_features"], inp.get("input_features_mask")))
        t_pre = timeit(lambda: inner(**{k: v for k, v in inp.items()},
                                     use_cache=True))
        t_1 = timeit(lambda: inner.generate(**inp, max_new_tokens=1,
                                            do_sample=False, use_cache=True), 2)
        out = inner.generate(**inp, max_new_tokens=512, do_sample=False,
                             use_cache=True)
        n_new = out.shape[1] - n_in
        t_n = timeit(lambda: inner.generate(**inp, max_new_tokens=512,
                                            do_sample=False, use_cache=True), 2)

    per_tok = (t_n - t_1) / max(n_new - 1, 1)
    res["per_clip"].append({
        "dur_s": round(len(wav) / 16000, 2), "prompt_tokens": int(n_in),
        "generated_tokens": int(n_new),
        "t_encoder_ms": round(t_enc, 1), "t_prefill_fwd_ms": round(t_pre, 1),
        "t_gen1_ms": round(t_1, 1), "t_genN_ms": round(t_n, 1),
        "ms_per_decode_token": round(per_tok, 2)})
    r = res["per_clip"][-1]
    print(f"clip{i} {r['dur_s']:5.1f}s  prompt {r['prompt_tokens']:4d}  "
          f"gen {r['generated_tokens']:4d} tok | enc {r['t_encoder_ms']:7.1f} "
          f"prefill {r['t_prefill_fwd_ms']:7.1f} gen1 {r['t_gen1_ms']:7.1f} "
          f"genN {r['t_genN_ms']:8.1f} => {r['ms_per_decode_token']:6.2f} ms/token",
          flush=True)

tok = np.mean([c["ms_per_decode_token"] for c in res["per_clip"]])
enc = np.mean([c["t_encoder_ms"] for c in res["per_clip"]])
gen = np.mean([c["t_genN_ms"] for c in res["per_clip"]])
ntk = np.mean([c["generated_tokens"] for c in res["per_clip"]])
res["summary"] = {"mean_ms_per_decode_token": round(float(tok), 2),
                  "mean_encoder_ms": round(float(enc), 1),
                  "mean_total_ms": round(float(gen), 1),
                  "mean_generated_tokens": round(float(ntk), 1),
                  "decode_share": round(float(tok * ntk / gen), 3),
                  "theoretical_ms_per_token_at_768GBs": round(4.1 / 0.768, 2)}
print("\n", json.dumps(res["summary"], indent=1))
Path("outputs_batch").mkdir(exist_ok=True)
Path("outputs_batch/decompose.json").write_text(json.dumps(res, indent=1))
