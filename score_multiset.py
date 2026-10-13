#!/usr/bin/env python
"""Score the multi-dataset comparison : every outputs_multiset/<set>__<model>
run, plus fastconformer_ctc's stored hypotheses on Vaani/SPRING from its earlier
full-split evaluation (re-scored, not re-decoded), with the benchmark's own scorer (bench_score.norm: punctuation to
word boundaries, NFC) and two 95% bootstraps (utterance; distinct-reference clusters).
SPRING sets also get a "no Latin-script reference characters" diagnostic for every
model. Voxtral Mini bn-BD (voxtral_mini_bn, decoded with vLLM on an A100 by
bench_run_voxtral_vllm.py) is also compared pairwise with each published model on each set:
WER/CER differences with a 95% bootstrap over distinct-reference clusters (10,000 draws,
seed 20260903), the method of clean_split_table.py. Writes
outputs_multiset/summary_multiset.json and prints the table.
"""
import json, os, re, sys
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_score import norm, boot   # noqa: E402  (the published scorer, unchanged)

A = Path(__file__).resolve().parent
OUT = A / "outputs_multiset"
SETS = ["vaani_test", "spring_r1_test", "spring_r2_test", "kathbath_test_known", "kathbath_test_unknown", "indicvoices_conv"]
MODELS = ["fastconformer_ctc", "hishab_conformer_large",
          "whisper_medium", "wav2vec2", "hishab_fastconformer", "voxtral_mini_bn"]
PUBLISHED = MODELS[:5]
NEW = "voxtral_mini_bn"
LATIN = re.compile(r"[A-Za-z]")


def counts(pairs):
    we, wn, ce, cn, keys = [], [], [], [], []
    for ref, hyp in pairs:
        r, h = norm(ref), norm(hyp)
        o = jiwer.process_words([r], [h if h else "*"])
        we.append(o.substitutions + o.deletions + o.insertions); wn.append(len(r.split()))
        c = jiwer.process_characters([r], [h if h else "*"])
        ce.append(c.substitutions + c.deletions + c.insertions); cn.append(len(r)); keys.append(r)
    return np.array(we), np.array(wn), np.array(ce), np.array(cn), np.array(keys)


def score(pairs):
    we, wn, ce, cn, keys = counts(pairs)
    return {"n": len(pairs), "wer": round(float(we.sum() / wn.sum()), 5), "cer": round(float(ce.sum() / cn.sum()), 5),
            "wer_ci_utterance": boot(we, wn), "wer_ci_reference_clustered": boot(we, wn, groups=keys),
            "ref_words": int(wn.sum()), "errors": int(we.sum())}


def paired(sa, sb):
    """a minus b, on the same utterances; the method of clean_split_table.paired_delta."""
    ea, na, ca, nca, ka = sa
    eb, nb, cb, ncb, kb = sb
    assert list(ka) == list(kb), "paired runs are not aligned on the same references"
    g = defaultdict(list)
    for i, k in enumerate(ka):
        g[k].append(i)
    gi = [np.array(v) for v in g.values()]
    rng = np.random.default_rng(20260903)
    dw, dc = [], []
    for _ in range(10000):
        pick = np.concatenate([gi[j] for j in rng.integers(0, len(gi), len(gi))])
        dw.append((ea[pick].sum() - eb[pick].sum()) / na[pick].sum())
        dc.append((ca[pick].sum() - cb[pick].sum()) / nca[pick].sum())
    dw, dc = np.sort(dw), np.sort(dc)
    return {"wer_delta_pp": round(float((ea.sum() - eb.sum()) / na.sum() * 100), 4),
            "wer_ci95_pp": [round(float(dw[250] * 100), 4), round(float(dw[9750] * 100), 4)],
            "cer_delta_pp": round(float((ca.sum() - cb.sum()) / nca.sum() * 100), 4),
            "cer_ci95_pp": [round(float(dc[250] * 100), 4), round(float(dc[9750] * 100), 4)]}


def load_run(set_, model):
    d = OUT / f"{set_}__{model}"
    if not (d / "run_meta.json").exists():
        return None
    meta = json.loads((d / "run_meta.json").read_text()); preds = json.loads((d / "predictions.json").read_text())
    if meta.get("status") != "complete":
        return {"status": meta.get("status"), "meta": meta, "preds": preds}
    return {"status": "complete", "meta": meta, "preds": preds}


def main():
    table, summary = [], {"sets": {}, "models": {}, "paired": {}, "note": __doc__}
    stores = {}
    for set_ in SETS:
        mf = A / "multiset" / set_ / "eval_manifest.json"
        if not mf.exists():
            continue
        man = json.loads(mf.read_text())
        summary["sets"][set_] = {k: man.get(k) for k in ("dataset", "source", "n", "hours", "ref_words", "seed", "sampling", "strata", "source_rows", "source_manifest_sha256", "speakers")}
        for model in MODELS:
            run = load_run(set_, model)
            if run is None:
                continue
            preds = run["preds"]
            rec = {"status": run["status"], "ms_per_clip_mean": run["meta"].get("ms_per_clip_mean"),
                   "failures": run["meta"].get("failures", 0), "missing_in_store": run["meta"].get("missing_in_store", 0),
                   "empty_hyps": sum(1 for p in preds if not p["transcript"].strip())}
            pairs = [(p["raw_reference"], p["transcript"]) for p in preds]
            rec.update(score(pairs))
            if any("finish_reasons" in p for p in preds):  # vLLM runs: pieces stopped by the 500-token cap
                capped = [i for i, p in enumerate(preds) if "length" in p.get("finish_reasons", [])]
                rec["clips_at_token_cap"] = len(capped)
                if capped:
                    rec["without_capped_clips"] = score([x for i, x in enumerate(pairs) if i not in set(capped)])
            if run["status"] == "complete":
                stores[(set_, model)] = counts(pairs)
            if set_.startswith("spring"):
                nl = [(p["raw_reference"], p["transcript"]) for p in preds if not LATIN.search(p["raw_reference"])]
                wl = [(p["raw_reference"], p["transcript"]) for p in preds if LATIN.search(p["raw_reference"])]
                rec["no_latin_ref"] = score(nl); rec["latin_ref"] = score(wl)
                rec["hyps_with_latin"] = sum(1 for p in preds if LATIN.search(p["transcript"]))
            summary["models"].setdefault(model, {})[set_] = rec
            table.append((set_, model, rec))
        if (set_, NEW) in stores:
            summary["paired"][set_] = {f"{NEW}_vs_{b}": paired(stores[(set_, NEW)], stores[(set_, b)])
                                       for b in PUBLISHED if (set_, b) in stores}
    (OUT / "summary_multiset.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    cur = None
    for set_, model, r in table:
        if set_ != cur:
            cur = set_; print(f"\n=== {set_}  (n={summary['sets'][set_]['n']}, {summary['sets'][set_]['hours']} h) ===")
        extra = f"  no-Latin {r['no_latin_ref']['wer']*100:6.2f}% (n={r['no_latin_ref']['n']})  Latin {r['latin_ref']['wer']*100:6.2f}%" if "no_latin_ref" in r else ""
        ms = f"{r['ms_per_clip_mean']:8.1f}" if r["ms_per_clip_mean"] else "     N/A"
        print(f"{model:24s} WER {r['wer']*100:6.2f}% [{r['wer_ci_reference_clustered'][0]*100:.2f}-{r['wer_ci_reference_clustered'][1]*100:.2f}]  CER {r['cer']*100:5.2f}%  {ms} ms  empty={r['empty_hyps']}{extra}  {'' if r['status']=='complete' else '['+str(r['status'])[:40]+']'}")
    for set_, d in summary["paired"].items():
        print(f"\n{NEW} minus each published model on {set_} (WER points, 95% clustered bootstrap):")
        for k, v in d.items():
            print(f"  {k.split('_vs_')[1]:24s} {v['wer_delta_pp']:+6.2f} [{v['wer_ci95_pp'][0]:+.2f}, {v['wer_ci95_pp'][1]:+.2f}]")
    print(f"\nwrote {OUT/'summary_multiset.json'}")


if __name__ == "__main__":
    main()
