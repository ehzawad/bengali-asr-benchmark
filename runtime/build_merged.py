#!/usr/bin/env python
"""Fold the LoRA + projector into base weights and save a standalone model.

merge_and_unload() removes the 88 LoRA wrapper modules by folding B@A*(alpha/r)
into each base weight, and swaps each modules_to_save entry for its trained
copy. Measured 1.94x decode speedup. Weights round to bf16 on the way in, so
the result must be re-scored, not assumed equivalent.
"""
import json, shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

P = "/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline"
BASE, REV = "Qwen/Qwen3-ASR-1.7B-hf", "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
ADAPTER = f"{P}/experiments/qwen_final/candidates/step_17280"
OUT = Path("/mnt/sdb/arafat/ehz/llm/merged_qwen3_asr_bn")

base = Qwen3ASRForConditionalGeneration.from_pretrained(
    BASE, revision=REV, dtype=torch.bfloat16)
peft = PeftModel.from_pretrained(base, ADAPTER)

# Record what the adapter actually trained, so the merge can be audited.
trained = {"lora": [], "modules_to_save": []}
for n, _ in peft.named_parameters():
    if "lora_" in n:
        trained["lora"].append(n.split(".lora_")[0].replace("base_model.model.", ""))
    elif "modules_to_save" in n:
        trained["modules_to_save"].append(
            n.split(".modules_to_save")[0].replace("base_model.model.", ""))
trained = {k: sorted(set(v)) for k, v in trained.items()}
print(f"LoRA targets: {len(trained['lora'])}  "
      f"modules_to_save: {len(trained['modules_to_save'])}")
for m in trained["modules_to_save"]:
    print("  saved-whole:", m)

# Snapshot a projector weight pre-merge: it rides modules_to_save, and a silent
# drop here is the failure mode that would keep WER plausible but wrong.
def proj_w(m, trained_copy):
    """Return the ACTIVE projector weight.

    Under PEFT the projector exists twice: `original_module` (frozen base) and
    `modules_to_save.default` (the trained copy that is actually used). Compare
    the trained copy against the post-merge weight; comparing the original
    would pass even if the training were silently dropped.
    """
    want = "modules_to_save.default" if trained_copy else None
    for n, prm in m.named_parameters():
        if "multi_modal_projector" not in n or not n.endswith("weight"):
            continue
        if trained_copy and want not in n:
            continue
        if not trained_copy and ("modules_to_save" in n or "original_module" in n):
            continue
        return n, prm.detach().float().clone()
    return None, None


pname, pre = proj_w(peft, True)
merged = peft.merge_and_unload()
mname, post = proj_w(merged, False)
assert pre is not None and post is not None, "projector weight not found"
delta = float((pre - post).abs().max())
print(f"projector {pname} -> {mname}  max|Δ| after merge = {delta:.3e}")
assert delta == 0.0, "projector changed during merge — modules_to_save lost"

if OUT.exists():
    shutil.rmtree(OUT)
merged.save_pretrained(OUT, safe_serialization=True)
AutoProcessor.from_pretrained(BASE, revision=REV).save_pretrained(OUT)
(OUT / "MERGE_PROVENANCE.json").write_text(json.dumps(
    {"base": BASE, "revision": REV, "adapter": ADAPTER,
     "n_lora_targets": len(trained["lora"]),
     "modules_to_save": trained["modules_to_save"],
     "projector_max_abs_delta": delta,
     "dtype": "bfloat16"}, indent=1))
print("wrote", OUT)
print("size", sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 2**30, "GiB")
