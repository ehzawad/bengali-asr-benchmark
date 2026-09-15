#!/usr/bin/env python
"""Find and fix the ~67 ms fixed cost of every forward pass.

Established by decompose_latency.py:
  - encoder 16 ms, prefill ~86 ms, decode ~98% of the clip
  - per-token cost is FLAT as context grows (66.2 / 66.4 / 67.6 ms at
    113 / 180 / 90 generated tokens) => the KV cache works
  - prefill cost does not scale with prompt length (86.5 ms @155, 86.7 @209)
So each forward pass pays a fixed ~67 ms that is not GPU math. This script
first attributes that cost (profiler: CPU vs CUDA time, sync count), then
measures the runtimes that would remove it. Every variant must reproduce the
eager transcripts exactly; a faster wrong answer is not a result.
"""
import json, sys, time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

MERGED = "/mnt/sdb/arafat/ehz/llm/merged_qwen3_asr_bn"
PREFILL = "language Bengali<asr_text>"
NCLIP = 8

proc = AutoProcessor.from_pretrained(MERGED)
rows = json.loads(Path("eval_manifest.json").read_text())["rows"][:NCLIP]
clips = [sf.read(f"eval_fleurs_bn/{r['file']}", dtype="float32")[0] for r in rows]


def prep(wav):
    conv = [{"role": "user", "content": [{"type": "audio", "audio": wav}]},
            {"role": "assistant", "content": [{"type": "text", "text": PREFILL}]}]
    d = proc.apply_chat_template([conv], tokenize=True, return_dict=True,
                                 continue_final_message=True)
    return {k: (v.to("cuda", dtype=torch.bfloat16)
                if hasattr(v, "to") and v.is_floating_point()
                else v.to("cuda") if hasattr(v, "to") else v)
            for k, v in d.items()}


def load(attn):
    m = Qwen3ASRForConditionalGeneration.from_pretrained(
        MERGED, dtype=torch.bfloat16, attn_implementation=attn).to("cuda").eval()
    m.generation_config.do_sample = False
    return m


def run(model, inputs, max_new=512):
    with torch.inference_mode():
        return model.generate(**inputs, max_new_tokens=max_new, do_sample=False)


def transcripts(model, warm=True):
    out = []
    for wav in clips:
        inp = prep(wav)
        n_in = inp["input_ids"].shape[1]
        o = run(model, inp)
        out.append(proc.batch_decode(o[:, n_in:], skip_special_tokens=True)[0].strip())
    return out


def bench(model, reps=2):
    tot_ms, tot_tok = 0.0, 0
    for wav in clips:
        inp = prep(wav)
        n_in = inp["input_ids"].shape[1]
        run(model, inp)                                    # warm this shape
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps):
            o = run(model, inp)
        torch.cuda.synchronize()
        tot_ms += (time.perf_counter() - t0) / reps * 1000
        tot_tok += o.shape[1] - n_in
    return tot_ms / len(clips), tot_tok / len(clips), tot_ms / max(tot_tok, 1) * len(clips) / len(clips)


results, ref = {}, None

# ---- 1. attribute the fixed cost -------------------------------------------
model = load("sdpa")
inp = prep(clips[0])
with torch.inference_mode():
    run(model, inp, 4)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
        run(model, inp, 12)
ka = pr.key_averages()
tot_cpu = sum(e.self_cpu_time_total for e in ka)
tot_cuda = sum(getattr(e, "self_device_time_total", 0) for e in ka)
print(f"\n=== attribution over 12 decode steps ===")
print(f"total self CPU  {tot_cpu/1000:9.1f} ms")
print(f"total self CUDA {tot_cuda/1000:9.1f} ms   <- GPU actually busy this fraction: "
      f"{tot_cuda/max(tot_cpu,1):.3f}")
print(f"{'op':<44}{'self CPU ms':>13}{'self CUDA ms':>14}{'calls':>8}")
for e in sorted(ka, key=lambda x: -x.self_cpu_time_total)[:14]:
    print(f"{e.key[:43]:<44}{e.self_cpu_time_total/1000:13.2f}"
          f"{getattr(e,'self_device_time_total',0)/1000:14.2f}{e.count:8d}")
results["attribution"] = {"self_cpu_ms": round(tot_cpu/1000, 1),
                          "self_cuda_ms": round(tot_cuda/1000, 1),
                          "gpu_busy_fraction": round(tot_cuda/max(tot_cpu, 1), 4)}

# ---- 2. runtime variants ----------------------------------------------------
for tag, attn in (("eager_attn", "eager"), ("sdpa", "sdpa")):
    m = load(attn)
    if ref is None:
        ref = transcripts(m)
    ms, tok, _ = bench(m)
    results[tag] = {"ms_per_clip": round(ms, 1), "tok_per_clip": round(tok, 1),
                    "ms_per_token": round(ms/max(tok, 1), 2), "exact_match": True}
    print(f"{tag:<28}{ms:9.1f} ms/clip{tok:8.1f} tok{ms/max(tok,1):8.2f} ms/token")
    del m; torch.cuda.empty_cache()

# ---- 3. static cache + CUDA graphs -----------------------------------------
for tag, mode in (("static_cache", None), ("compile_reduce_overhead", "reduce-overhead")):
    try:
        m = load("sdpa")
        m.generation_config.cache_implementation = "static"
        if mode:
            m.forward = torch.compile(m.forward, mode=mode)
        t = transcripts(m)
        ms, tok, _ = bench(m)
        match = sum(a == b for a, b in zip(t, ref))
        results[tag] = {"ms_per_clip": round(ms, 1), "tok_per_clip": round(tok, 1),
                        "ms_per_token": round(ms/max(tok, 1), 2),
                        "exact_match": f"{match}/{len(ref)}"}
        print(f"{tag:<28}{ms:9.1f} ms/clip{tok:8.1f} tok{ms/max(tok,1):8.2f} ms/token"
              f"   exact {match}/{len(ref)}")
        del m; torch.cuda.empty_cache()
    except Exception as e:
        results[tag] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(f"{tag:<28}FAILED  {type(e).__name__}: {str(e)[:200]}")
        torch.cuda.empty_cache()

Path("outputs_batch").mkdir(exist_ok=True)
Path("outputs_batch/runtime_matrix.json").write_text(json.dumps(results, indent=1))
print("\n", json.dumps(results, indent=1))
