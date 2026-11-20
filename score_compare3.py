#!/usr/bin/env python
"""Three models on one GPU: Rumble-2-Flash bn-BD, Voxtral Mini bn-BD and Qwen3-ASR 1.7B bn-BD, each decoded on FLEURS
and the ten subsets with the unchanged harness (run_all.sh's backends: greedy, batch 1, one model resident at a time) on
one A100-SXM4-40GB (the six Indian sets on a second card of the same model) and on one B200. The runs are under
outputs_compare3/<gpu>/outputs_fleurs/<label>/ and outputs_compare3/<gpu>/outputs_multiset/<set>__<label>/.

- Accuracy: the cleaned subsets (clean_subsets/<set>.json), scored with bench_score.norm and jiwer and a 95 % bootstrap
  over distinct references, as score_clean.py does.
- Latency: per-clip time over every clip of the set (FLEURS: the 920 test recordings), as in the main tables, from the
  per-clip wall clock in predictions.json (for the subsets this equals run_meta.json's ms_per_clip_*).
- cap_hits: scored clips with a piece stopped by the token cap (finish reason "length").
- Output tokens: tokens per FLEURS test transcript under each model's own tokenizer (Rumble and Qwen3-ASR: the
  checkpoint's Hugging Face tokenizer; Voxtral Mini: its Tekken tokenizer through mistral_common), transcript text
  only (no prompt, language tag or end token), with a least-squares fit of each clip's time against its token count.
  The tokenizers are read from the Hub at the revisions in outputs_fleurs/checkpoint_revisions.json (the ‡ models need
  access) or from --tokenizer label=<local dir>; --keep-tokens keeps the committed token block instead.

  python score_compare3.py [--tokenizer label=dir ...] [--keep-tokens]  -> outputs_compare3/summary.json, tables on stdout
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from bench_score import counts, boot   # noqa: E402  (the published scorer, unchanged)

R = A / "outputs_compare3"
GPUS = {"a100": "A100-SXM4-40GB", "b200": "B200"}
MODELS = {"rumble_bn": ("Rumble-2-Flash bn-BD ‡", "ehzawad/Rumble-2-Flash-bn-BD"),
          "voxtral_mini_bn": ("Voxtral Mini bn-BD ‡", "ehzawad/Voxtral-Mini-3B-2507-bn-BD"),
          "qwen3_asr_bn": ("Qwen3-ASR 1.7B bn-BD ‡", "ehzawad/Qwen3-ASR-1.7B-bn-BD")}
SETS = [("fleurs_test", "FLEURS test"), ("subakko_test", "SUBAK.KO test"), ("banspeech", "BanSpeech"),
        ("cv17_bn_test", "Common Voice 17 test"), ("ben10_valid", "Ben-10 validation (dialects)"),
        ("vaani_test", "Vaani test"), ("spring_r1_test", "SPRING-INX R1 eval"), ("spring_r2_test", "SPRING-INX R2 eval"),
        ("kathbath_test_known", "Kathbath (known spk)"), ("kathbath_test_unknown", "Kathbath (unknown spk)"),
        ("indicvoices_conv", "IndicVoices conversational")]
SEED = 20260903
REV = json.loads((A / "outputs_fleurs" / "checkpoint_revisions.json").read_text())


def load(gpu, model, set_):
    d = R / gpu / ("outputs_fleurs/" + model if set_ == "fleurs_test" else f"outputs_multiset/{set_}__{model}")
    meta, preds = json.loads((d / "run_meta.json").read_text()), json.loads((d / "predictions.json").read_text())
    assert meta["model"] == MODELS[model][1] and meta["revision"] == REV[MODELS[model][1]], d
    assert meta["batch_size"] == 1 and meta["failures"] == 0 and meta.get("status", "complete") == "complete", d
    assert meta["gpu_before"].startswith("NVIDIA " + GPUS[gpu] + ","), (d, meta["gpu_before"])
    if set_ == "fleurs_test":
        preds = [p for p in preds if p["split"] == "test"]
        key = lambda p: Path(p["file"]).name   # noqa: E731
    else:
        key = lambda p: p["utt_id"]            # noqa: E731
    return meta, preds, key


def ms_stats(preds):
    t = np.array([p["wall_clock_seconds"] * 1000 for p in preds])
    return {"ms_per_clip_mean": round(float(t.mean()), 2), "ms_per_clip_median": round(float(np.median(t)), 2),
            "ms_per_clip_p95": round(float(np.percentile(t, 95)), 2), "clips_timed": len(t)}


def tokenizers(args):
    """label -> (encode function, description)."""
    over = dict(x.split("=", 1) for x in args.tokenizer)
    out = {}
    for m, (_, hf) in MODELS.items():
        d = over.get(m)
        if d is None:
            from huggingface_hub import snapshot_download
            d = snapshot_download(hf, revision=REV[hf], allow_patterns=["tekken.json"] if m == "voxtral_mini_bn" else
                                  ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                                   "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"])
        d = Path(d)
        if m == "voxtral_mini_bn":
            from mistral_common.tokens.tokenizers.tekken import Tekkenizer
            tok, f = Tekkenizer.from_file(str(d / "tekken.json")), d / "tekken.json"
            enc = lambda s, tok=tok: tok.encode(s, bos=False, eos=False)   # noqa: E731
        else:
            from transformers import AutoTokenizer
            tok, f = AutoTokenizer.from_pretrained(str(d)), d / "tokenizer.json"
            enc = lambda s, tok=tok: tok(s, add_special_tokens=False).input_ids   # noqa: E731
        out[m] = (enc, f"{hf}@{REV[hf]} {f.name} (sha256 {hashlib.sha256(f.read_bytes()).hexdigest()}), "
                       f"{type(tok).__name__}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", action="append", default=[], help="label=<dir with that model's tokenizer files>")
    ap.add_argument("--keep-tokens", action="store_true", help="keep the committed token counts (no tokenizer access)")
    args = ap.parse_args()
    old = json.loads((R / "summary.json").read_text()) if (R / "summary.json").exists() else {}
    summary = {"note": __doc__.split("\n\n  python")[0],
               "models": {m: {"name": n, "checkpoint": hf, "revision": REV[hf]} for m, (n, hf) in MODELS.items()},
               "gpus": {}, "sets": {}, "runs": {}, "average": {}}
    kept = {}
    for s, _ in SETS:
        cl = json.loads((A / "clean_subsets" / f"{s}.json").read_text())
        kept[s] = cl["kept"]
        summary["sets"][s] = {"n_full": cl["n_full"], "n_kept": cl["n_kept"]}
    texts = {}
    for gpu in GPUS:
        summary["runs"][gpu], devices = {}, set()
        for s, _ in SETS:
            summary["runs"][gpu][s] = {}
            for m in MODELS:
                meta, preds, key = load(gpu, m, s)
                devices.add(meta["gpu_before"].split(",")[0])
                by = {key(p): p for p in preds}
                assert len(by) == len(preds) == summary["sets"][s]["n_full"], (gpu, m, s)
                sel = [by[k] for k in kept[s]]
                we, wn, ce, cn, keys, _ = counts(sel)
                r = {"n": len(sel), "wer": float(we.sum() / wn.sum()), "cer": float(ce.sum() / cn.sum()),
                     "wer_ci_reference_clustered": boot(we, wn, groups=keys, seed=SEED),
                     "word_errors": int(we.sum()), "ref_words": int(wn.sum()),
                     "cap_hits": sum("length" in p.get("finish_reasons", []) for p in sel)}
                r.update(ms_stats(preds))
                if s != "fleurs_test":   # the subsets' run_meta times are over the same clips
                    assert all(abs(r[k] - meta[k]) < 0.011 for k in ("ms_per_clip_mean", "ms_per_clip_median",
                                                                      "ms_per_clip_p95")), (gpu, m, s)
                summary["runs"][gpu][s][m] = r
                texts[(gpu, s, m)] = {k: by[k]["transcript"] for k in kept[s]}
                if s == "fleurs_test":
                    texts[(gpu, "fleurs_all", m)] = [(p["transcript"], p["wall_clock_seconds"] * 1000) for p in preds]
        assert len(devices) == 1, devices
        summary["gpus"][gpu] = {"device": devices.pop()}
        summary["average"][gpu] = {m: {
            "wer": float(np.mean([summary["runs"][gpu][s][m]["wer"] for s, _ in SETS])),
            "cer": float(np.mean([summary["runs"][gpu][s][m]["cer"] for s, _ in SETS])),
            "ms_per_clip_mean": float(np.mean([summary["runs"][gpu][s][m]["ms_per_clip_mean"] for s, _ in SETS])),
            "cap_hits": int(sum(summary["runs"][gpu][s][m]["cap_hits"] for s, _ in SETS))} for m in MODELS}
    # the same model on the two GPUs: how many scored transcripts are identical
    summary["same_transcript_b200_vs_a100"] = {
        m: {s: sum(texts[("a100", s, m)][k] == texts[("b200", s, m)][k] for k in kept[s]) for s, _ in SETS} for m in MODELS}

    try:
        if args.keep_tokens:
            raise RuntimeError("--keep-tokens")
        toks = tokenizers(args)
        summary["output_tokens_fleurs_test"] = {"tokenizers": {m: d for m, (_, d) in toks.items()}}
        for gpu in GPUS:
            summary["output_tokens_fleurs_test"][gpu] = {}
            for m in MODELS:
                tt = texts[(gpu, "fleurs_all", m)]
                n = np.array([len(toks[m][0](t)) for t, _ in tt], dtype=float)
                ms = np.array([x for _, x in tt])
                slope, icpt = np.polyfit(n, ms, 1)
                summary["output_tokens_fleurs_test"][gpu][m] = {
                    "clips": len(tt), "tokens_per_clip_mean": round(float(n.mean()), 2),
                    "tokens_per_clip_median": float(np.median(n)), "tokens_max": int(n.max()),
                    "fit_ms_per_token": round(float(slope), 3), "fit_intercept_ms": round(float(icpt), 1)}
    except Exception as e:   # no tokenizer access: keep the committed counts, or stop
        if not (args.keep_tokens and "output_tokens_fleurs_test" in old):
            raise
        print(f"tokens: kept from the committed summary.json ({e})", file=sys.stderr)
        summary["output_tokens_fleurs_test"] = old["output_tokens_fleurs_test"]
    (R / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")

    label = dict(SETS)
    tok = summary["output_tokens_fleurs_test"]
    for gpu in GPUS:
        print(f"\n{summary['gpus'][gpu]['device']}: WER on the cleaned subset · mean ms per clip\n")
        print("| Set | " + " | ".join(MODELS[m][0] for m in MODELS) + " |")
        print("|---|" + "---:|" * len(MODELS))
        for s, _ in SETS:
            print(f"| {label[s]} | " + " | ".join(f"{100 * summary['runs'][gpu][s][m]['wer']:.2f}% · "
                                                  f"{summary['runs'][gpu][s][m]['ms_per_clip_mean']:,.0f} ms"
                                                  for m in MODELS) + " |")
        av = summary["average"][gpu]
        print("| **Average, 11 sets** | " + " | ".join(f"**{100 * av[m]['wer']:.2f}% · {av[m]['ms_per_clip_mean']:,.0f} ms**"
                                                       for m in MODELS) + " |")
        print("| Clips at the token cap | " + " | ".join(str(av[m]["cap_hits"]) for m in MODELS) + " |")
        print("| Tokens per FLEURS test clip | " + " | ".join(f"{tok[gpu][m]['tokens_per_clip_mean']:.0f}" for m in MODELS)
              + " |")
        print("| ms per token (fit, FLEURS test) | " + " | ".join(f"{tok[gpu][m]['fit_ms_per_token']:.2f}" for m in MODELS)
              + " |")
    print()
    for m in MODELS:
        same = summary["same_transcript_b200_vs_a100"][m]
        n = sum(summary["sets"][s]["n_kept"] for s, _ in SETS)
        print(f"{m}: B200 transcript identical to the A100's on {sum(same.values()):,} of {n:,} scored clips; "
              f"average WER A100 {100 * summary['average']['a100'][m]['wer']:.2f}% B200 "
              f"{100 * summary['average']['b200'][m]['wer']:.2f}%; fit intercept A100 "
              f"{tok['a100'][m]['fit_intercept_ms']:.0f} ms B200 {tok['b200'][m]['fit_intercept_ms']:.0f} ms")
    print(f"\nwrote {R / 'summary.json'}")


if __name__ == "__main__":
    main()
