#!/usr/bin/env python
"""Find where the ~119 ms/token goes.

Compares, on identical clips: (a) the PEFT-wrapped adapter as benchmarked,
(b) the same weights after merge_and_unload(), which folds LoRA into the base
matrices and removes ~88 extra module calls per token, and (c) the bare base
model, as a floor. Also reports tokens generated so ms/token is real rather
than inferred.
"""
import json, time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

P = "/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline"
BASE, REV = "Qwen/Qwen3-ASR-1.7B-hf", "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
ADAPTER = f"{P}/experiments/qwen_final/candidates/step_17280"
PREFILL = "language Bengali<asr_text>"
N = 12

proc = AutoProcessor.from_pretrained(BASE, revision=REV)
rows = json.loads(Path("eval_manifest.json").read_text())["rows"][:N]
import soundfile as sf
clips = [sf.read(f"eval_fleurs_bn/{r['file']}", dtype="float32")[0] for r in rows]


def run(model, label):
    tot_t, tot_tok = 0.0, 0
    for i, wav in enumerate(clips):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": wav}]},
                {"role": "assistant", "content": [{"type": "text", "text": PREFILL}]}]
        inp = proc.apply_chat_template([conv], tokenize=True, return_dict=True,
                                       continue_final_message=True)
        inp = {k: (v.to("cuda", dtype=model.dtype)
                   if hasattr(v, "to") and v.is_floating_point()
                   else v.to("cuda") if hasattr(v, "to") else v)
               for k, v in inp.items()}
        n_in = inp["input_ids"].shape[1]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=512, do_sample=False,
                                 num_beams=1, use_cache=True)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        if i == 0:
            continue                                   # warm-up
        tot_t += dt; tot_tok += out.shape[1] - n_in
    n = len(clips) - 1
    print(f"{label:34s} {tot_t/n*1000:8.1f} ms/clip   "
          f"{tot_tok/n:6.1f} tok/clip   {tot_t/max(tot_tok,1)*1000:6.2f} ms/token",
          flush=True)
    return tot_t / n * 1000


base = Qwen3ASRForConditionalGeneration.from_pretrained(
    BASE, revision=REV, dtype=torch.bfloat16).to("cuda").eval()
peft = PeftModel.from_pretrained(base, ADAPTER).eval()
a = run(peft, "PEFT-wrapped (as benchmarked)")

merged = peft.merge_and_unload().eval()
b = run(merged, "merge_and_unload()")

print(f"\nmerge speedup: {a/b:.2f}x")
Path("outputs_batch").mkdir(exist_ok=True)
Path("outputs_batch/profile_decode.json").write_text(json.dumps(
    {"peft_ms_per_clip": round(a, 1), "merged_ms_per_clip": round(b, 1),
     "speedup": round(a / b, 3), "n_clips": len(clips) - 1}, indent=1))
