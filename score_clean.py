#!/usr/bin/env python
"""The leaderboard on the cleaned subsets of the eleven test sets.

Benchmark rule: a clip is scored only if it is at least 0.5 s long and its reference contains no digit (any
character for which Python's str.isdigit() is true, ASCII or Bengali). Clips shorter than 0.5 s hold too little
speech to score a word error rate on, and a reference that writes a number as digits cannot be matched by models
that write numbers as words. The rule looks only at the manifests (duration and reference), never at a model's
output, so every model is scored on the same clips. 10,291 of the 10,920 test clips remain.

The exact clip lists are in clean_subsets/<set>.json (kept and removed clips, by utt_id or, for FLEURS, by file
name). This script rebuilds them from the manifests and stops if a committed list disagrees; with --write it
(re)writes them. Every model's committed predictions on the full sets are then re-scored on the kept clips with
the benchmark's scorer (bench_score.norm and jiwer, as bench_score.py), with 95 % bootstraps over distinct
references (10,000 draws, seed 20260903).

  python score_clean.py [--write]     -> clean_subsets/*.json, outputs_clean/summary_clean.json, tables on stdout
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from bench_score import counts, boot   # noqa: E402  (the published scorer, unchanged)
from score_multiset import paired     # noqa: E402

SETS = [("fleurs_test", "FLEURS test"), ("vaani_test", "Vaani test"), ("spring_r1_test", "SPRING-INX R1 eval"),
        ("spring_r2_test", "SPRING-INX R2 eval"), ("kathbath_test_known", "Kathbath (known spk)"),
        ("kathbath_test_unknown", "Kathbath (unknown spk)"), ("indicvoices_conv", "IndicVoices conversational"),
        ("subakko_test", "SUBAK.KO test"), ("banspeech", "BanSpeech"), ("cv17_bn_test", "Common Voice 17 test"),
        ("ben10_valid", "Ben-10 validation (dialects)")]
MODELS = {
    "granite_speech_bn": "Granite Speech 5.0 2B bn-BD ‡",
    "whisper_turbo_bn": "Whisper turbo bn-BD ‡",
    "qwen3_asr_bn": "Qwen3-ASR 1.7B bn-BD ‡",
    "canary_1b_v2_bn": "Canary 1B v2 bn-BD ‡",
    "voxtral_mini_bn": "Voxtral Mini bn-BD ‡",
    "fastconformer_hybrid_bn": "FastConformer hybrid bn-BD ‡",
    "hishab_conformer_large": "Conformer Large (hishab)",
    "whisper_medium": "Whisper Medium",
    "fastconformer_ctc": "ehzawad FastConformer-CTC",
    "wav2vec2": "Wav2Vec2",
    "whistle_bn_bd": "Whistle bn-BD ‡ (55M, CPU engine)",
    "whistle_bn_bd_a100": "Whistle bn-BD ‡ (55M, A100, greedy)",
}
# the two Whistle rows are one checkpoint under two runtimes: each is ranked against the other models, not against
# the other; the remaining models' ranks are computed with the CPU row
WHISTLE = ["whistle_bn_bd", "whistle_bn_bd_a100"]
MIN_SECONDS = 0.5
SEED = 20260903
PAIRED = ["granite_speech_bn", "voxtral_mini_bn", "fastconformer_ctc"]   # compared pairwise with every other model
# provenance flags from the model cards (as in the README): N = card does not list the corpus, S = trained on another
# split of it; every other cell is U (the card does not say either way). FLEURS is reported without flags.
FLAGS = {"fastconformer_ctc": dict(subakko_test="S", banspeech="N", cv17_bn_test="S", ben10_valid="N", vaani_test="N",
                                   spring_r1_test="N", spring_r2_test="N", kathbath_test_known="S",
                                   kathbath_test_unknown="S", indicvoices_conv="S"),
         "hishab_conformer_large": dict(subakko_test="S", banspeech="N", cv17_bn_test="S", ben10_valid="N", vaani_test="N",
                                        spring_r1_test="N", spring_r2_test="N", kathbath_test_known="S",
                                        kathbath_test_unknown="S", indicvoices_conv="N")}


def rows_of(name):
    if name == "fleurs_test":
        m = json.loads((A / "eval_manifest.json").read_text())
        return [dict(r, key=r["file"]) for r in m["rows"] if r["split"] == "test"]
    m = json.loads((A / "multiset" / name / "eval_manifest.json").read_text())
    return [dict(r, key=r["utt_id"]) for r in m["rows"]]


def clean_list(name, rows):
    short = [r["key"] for r in rows if float(r["duration"]) < MIN_SECONDS]
    digit = [r["key"] for r in rows if any(c.isdigit() for c in r["raw_reference"])]
    drop = set(short) | set(digit)
    return {"set": name, "rule": f"keep a clip if duration >= {MIN_SECONDS} s and no character of raw_reference is a digit (str.isdigit)",
            "key": "file" if name == "fleurs_test" else "utt_id", "n_full": len(rows), "n_kept": len(rows) - len(drop),
            "removed_short": short, "removed_digit_reference": digit, "kept": [r["key"] for r in rows if r["key"] not in drop]}


def preds_of(model, name, rows):
    p = (A / "outputs_fleurs" / model / "predictions.json") if name == "fleurs_test" else \
        (A / "outputs_multiset" / f"{name}__{model}" / "predictions.json")
    preds = json.loads(p.read_text())
    if name == "fleurs_test":
        preds = [q for q in preds if q["split"] == "test"]
        by = {Path(q["file"]).name: q for q in preds}
    else:
        by = {q["utt_id"]: q for q in preds}
    assert len(by) == len(preds) == len(rows), (model, name, len(preds), len(rows))
    out = []
    for r in rows:
        q = by[r["key"]]
        assert q["raw_reference"] == r["raw_reference"], (model, name, r["key"])
        out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write clean_subsets/*.json instead of checking them")
    a = ap.parse_args()
    for m in [m for m in MODELS if not (A / "outputs_fleurs" / m / "predictions.json").exists()]:
        print(f"WARNING: no outputs for {m}; left out", file=sys.stderr)
        del MODELS[m]
    (A / "clean_subsets").mkdir(exist_ok=True)
    summary = {"rule": "clips shorter than 0.5 s and clips whose reference contains a digit are removed; lists in clean_subsets/",
               "bootstrap": {"draws": 10000, "seed": SEED, "unit": "distinct reference"}, "sets": {}, "models": {}}
    wer, store = {m: {} for m in MODELS}, {}
    for name, _ in SETS:
        rows = rows_of(name)
        cl = clean_list(name, rows)
        f = A / "clean_subsets" / f"{name}.json"
        if a.write:
            f.write_text(json.dumps(cl, ensure_ascii=False, indent=1) + "\n")
        else:
            assert json.loads(f.read_text()) == cl, f"{f} disagrees with the rule applied to the manifest"
        keep = set(cl["kept"])
        idx = [i for i, r in enumerate(rows) if r["key"] in keep]
        summary["sets"][name] = {"n_full": cl["n_full"], "n_kept": cl["n_kept"], "removed_short": len(cl["removed_short"]),
                                 "removed_digit_reference": len(cl["removed_digit_reference"]),
                                 "removed": cl["n_full"] - cl["n_kept"]}
        for m in MODELS:
            full = counts(preds_of(m, name, rows))
            we, wn, ce, cn, keys = (x[idx] for x in full[:5])
            store[(name, m)] = (we, wn, ce, cn, keys)
            r = {"n": len(idx), "wer": float(we.sum() / wn.sum()), "cer": float(ce.sum() / cn.sum()),
                 "wer_ci_reference_clustered": boot(we, wn, groups=keys, seed=SEED),
                 "word_errors": int(we.sum()), "ref_words": int(wn.sum()),
                 "wer_full_set": float(full[0].sum() / full[1].sum())}
            summary["models"].setdefault(m, {})[name] = r
            wer[m][name] = r["wer"]
        print(f"scored {name}", file=sys.stderr, flush=True)
    names = [n for n, _ in SETS]
    for m in MODELS:
        others = [x for x in MODELS if x not in WHISTLE or x == m] if m in WHISTLE else [x for x in MODELS if x != "whistle_bn_bd_a100"]
        ranks = [1 + sum(wer[o][n] < wer[m][n] for o in others if o != m) for n in names]
        s = summary["models"][m]
        s["macro"] = {"wer": float(np.mean([wer[m][n] for n in names])),
                      "cer": float(np.mean([s[n]["cer"] for n in names])),
                      "average_rank": float(np.mean(ranks)), "ranks": dict(zip(names, ranks)),
                      "best_on": [n for n in names if all(wer[m][n] <= wer[o][n] for o in others)]}
    # PAIRED model minus every other model on the same clean clips (WER/CER points, 95 % bootstrap over distinct
    # references, the method of score_multiset.paired)
    summary["paired"] = {}
    for m in [x for x in PAIRED if x in MODELS]:
        summary["paired"][m] = {n: {o: paired(store[(n, m)], store[(n, o)]) for o in MODELS if o != m} for n in names}
    (A / "outputs_clean").mkdir(exist_ok=True)
    (A / "outputs_clean" / "summary_clean.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")

    order = sorted(MODELS, key=lambda m: summary["models"][m]["macro"]["average_rank"])
    label = dict(SETS)
    print("| Model | Average rank (1 is best) | Sets with the lowest WER | Average WER | Average CER |")
    print("|---|---:|---|---:|---:|")
    for m in order:
        s = summary["models"][m]["macro"]
        print(f"| {MODELS[m]} | {s['average_rank']:.1f} | {', '.join(label[n] for n in s['best_on']) or '—'} | "
              f"{100 * s['wer']:.1f}% | {100 * s['cer']:.1f}% |")
    print()
    print("| Set | Full | Removed (< 0.5 s) | Removed (digits) | Scored |")
    print("|---|---:|---:|---:|---:|")
    for n in names:
        s = summary["sets"][n]
        print(f"| {label[n]} | {s['n_full']:,} | {s['removed_short']} | {s['removed_digit_reference']} | {s['n_kept']:,} |")
    tot = {k: sum(summary["sets"][n][k] for n in names) for k in ("n_full", "removed_short", "removed_digit_reference", "n_kept")}
    print(f"| All | {tot['n_full']:,} | {tot['removed_short']} | {tot['removed_digit_reference']} | {tot['n_kept']:,} |")
    for group in (names[7:], names[1:7], names[:1]):
        print()
        print("| Model | " + " | ".join(label[n] for n in group) + " |")
        print("|---|" + "---:|" * len(group))
        for m in order:
            cells = []
            for n in group:
                r = summary["models"][m][n]
                best = all(r["wer"] <= summary["models"][o][n]["wer"] for o in MODELS if o not in WHISTLE or o == m)
                lo, hi = r["wer_ci_reference_clustered"]
                v = f"{100 * r['wer']:.2f}%"
                flag = "" if n == "fleurs_test" else f" <sub>{FLAGS.get(m, {}).get(n, 'U')}</sub>"
                cells.append(f"{'**' + v + '**' if best else v} [{100 * lo:.1f}–{100 * hi:.1f}]{flag}")
            print(f"| {MODELS[m]} | " + " | ".join(cells) + " |")
    for m, per in summary["paired"].items():
        print(f"\n{MODELS[m]} minus the model with the lowest WER among the others (paired, WER points, 95 % CI):\n")
        print(f"| Set | {MODELS[m]} | Best other model | Difference |")
        print("|---|---:|---|---|")
        for n in names:
            o = min((x for x in MODELS if x != m and x not in WHISTLE), key=lambda x: wer[x][n])
            d = per[n][o]
            lo, hi = d["wer_ci95_pp"]
            verdict = "lower" if hi < 0 else "higher" if lo > 0 else "inconclusive"
            print(f"| {label[n]} | {100 * wer[m][n]:.2f}% | {MODELS[o]} {100 * wer[o][n]:.2f}% | "
                  f"{d['wer_delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}], {verdict} |")


if __name__ == "__main__":
    main()
