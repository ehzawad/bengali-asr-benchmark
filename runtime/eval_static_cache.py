#!/usr/bin/env python
"""Correctness gate for the merged + static-KV-cache runtime.

The static cache is 6.5x faster but changed 3 of 8 transcripts on the probe
set: a fixed-size padded cache alters bf16 reduction order, flipping argmax at
near-ties. Speed is worthless if accuracy moved, so this re-scores all 1,322
clips through the SAME vendored asr_core audio path as the published
benchmark, and reports a PAIRED comparison against the shipped predictions --
per-utterance error deltas, not just two independent WERs whose CIs overlap.
"""
import json, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asr_core
from bench_score import norm, boot            # published scoring surface
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

MERGED = "/mnt/sdb/arafat/ehz/llm/merged_qwen3_asr_bn"
PREFILL = "language Bengali<asr_text>"
OUT = Path("outputs_static")
OUT.mkdir(exist_ok=True)


class StaticCacheBackend:
    def __init__(self):
        self.proc = AutoProcessor.from_pretrained(MERGED)
        self.m = Qwen3ASRForConditionalGeneration.from_pretrained(
            MERGED, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda").eval()
        self.m.generation_config.cache_implementation = "static"

    def transcribe(self, paths, batch_size=1, verbose=False):
        return [SimpleNamespace(text=self._one(_staged(p))) for p in paths]

    def _one(self, wav):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": wav}]},
                {"role": "assistant", "content": [{"type": "text", "text": PREFILL}]}]
        inp = self.proc.apply_chat_template([conv], tokenize=True,
                                            return_dict=True,
                                            continue_final_message=True)
        inp = {k: (v.to("cuda", dtype=torch.bfloat16)
                   if hasattr(v, "to") and v.is_floating_point()
                   else v.to("cuda") if hasattr(v, "to") else v)
               for k, v in inp.items()}
        with torch.inference_mode():
            out = self.m.generate(**inp, max_new_tokens=512, do_sample=False,
                                  num_beams=1, use_cache=True)
        txt = self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0]
        if "<asr_text>" in txt:
            txt = txt.split("<asr_text>")[-1]
        return txt.strip()


def _staged(path):
    info = sf.info(str(path))
    assert (info.samplerate == 16000 and info.channels == 1
            and info.subtype == "PCM_16"), f"unexpected staged format: {info}"
    return sf.read(str(path), dtype="float32")[0]


def main():
    base = json.loads(Path("outputs/qwen3_adapter/predictions.json").read_text())
    rows = [{"file": b["file"], "text": b["reference"],
             "shipped": b["transcript"] or ""} for b in base]
    be = StaticCacheBackend()

    # Warm once so the first clip does not carry compile/alloc cost.
    a0, _ = asr_core.load_audio(f"eval_fleurs_bn/{rows[0]['file']}")
    p0, _ = asr_core.chunks(a0)
    asr_core.decode(be, p0)

    hyps, ms = [], []
    t_all = time.perf_counter()
    for i, r in enumerate(rows):
        audio, _ = asr_core.load_audio(f"eval_fleurs_bn/{r['file']}")
        pieces, _ = asr_core.chunks(audio)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        txt = asr_core.decode(be, pieces)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        hyps.append(txt)
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(rows)}  mean {np.mean(ms):7.1f} ms/clip", flush=True)
    wall = time.perf_counter() - t_all

    refs = [r["text"] for r in rows]
    Path(OUT / "predictions.json").write_text(json.dumps(
        [{"file": r["file"], "ref": r["text"], "hyp": h}
         for r, h in zip(rows, hyps)], ensure_ascii=False, indent=1))

    # Paired against the shipped adapter predictions.
    bh = {r["file"]: r["shipped"] for r in rows}

    import jiwer
    def errs(ref, hyp):
        m = jiwer.process_words([norm(ref)], [norm(hyp)])
        return m.substitutions + m.deletions + m.insertions, len(norm(ref).split())

    e_new, e_old, nref, ident = [], [], [], 0
    for r, h in zip(rows, hyps):
        en, n = errs(r["text"], h)
        eo, _ = errs(r["text"], bh.get(r["file"], ""))
        e_new.append(en); e_old.append(eo); nref.append(n)
        ident += (norm(h) == norm(bh.get(r["file"], "")))
    e_new, e_old, nref = map(np.array, (e_new, e_old, nref))

    wer_new, wer_old = e_new.sum() / nref.sum(), e_old.sum() / nref.sum()
    d = e_new - e_old
    rng = np.random.default_rng(20260903)
    idx = rng.integers(0, len(d), (4000, len(d)))
    delta_bs = np.sort((d[idx].sum(1)) / nref[idx].sum(1))

    rep = {"runtime": "merged + static KV cache, sdpa, greedy batch 1",
           "n": len(rows),
           "wer_static": round(float(wer_new), 5),
           "wer_shipped": round(float(wer_old), 5),
           "wer_delta_pp": round(float((wer_new - wer_old) * 100), 4),
           "delta_ci95_pp": [round(float(delta_bs[100] * 100), 4),
                             round(float(delta_bs[3900] * 100), 4)],
           "identical_transcripts": f"{ident}/{len(rows)}",
           "ms_per_clip_mean": round(float(np.mean(ms)), 1),
           "ms_per_clip_p95": round(float(np.percentile(ms, 95)), 1),
           "wall_s": round(wall, 1)}
    Path(OUT / "static_cache_gate.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
