#!/usr/bin/env python
"""Score the multi-dataset comparison : every outputs_multiset/<set>__<model>
run, plus fastconformer_ctc's stored hypotheses on Vaani/SPRING from its earlier
full-split evaluation (re-scored, not re-decoded), with the benchmark's own scorer (bench_score.norm: punctuation to
word boundaries, NFC) and two 95% bootstraps (utterance; distinct-reference clusters).
SPRING sets also get a "no Latin-script reference characters" diagnostic for every
model. Writes outputs_multiset/summary_multiset.json and prints the table.
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
MODELS = ["fastconformer_ctc", "qwen3_adapter", "hishab_conformer_large",
          "whisper_medium", "wav2vec2", "hishab_fastconformer"]
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


def load_run(set_, model):
    d = OUT / f"{set_}__{model}"
    if not (d / "run_meta.json").exists():
        return None
    meta = json.loads((d / "run_meta.json").read_text()); preds = json.loads((d / "predictions.json").read_text())
    if meta.get("status") != "complete":
        return {"status": meta.get("status"), "meta": meta, "preds": preds}
    return {"status": "complete", "meta": meta, "preds": preds}


def main():
    table, summary = [], {"sets": {}, "models": {}, "note": __doc__}
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
            rec.update(score([(p["raw_reference"], p["transcript"]) for p in preds]))
            if set_.startswith("spring"):
                nl = [(p["raw_reference"], p["transcript"]) for p in preds if not LATIN.search(p["raw_reference"])]
                wl = [(p["raw_reference"], p["transcript"]) for p in preds if LATIN.search(p["raw_reference"])]
                rec["no_latin_ref"] = score(nl); rec["latin_ref"] = score(wl)
                rec["hyps_with_latin"] = sum(1 for p in preds if LATIN.search(p["transcript"]))
            summary["models"].setdefault(model, {})[set_] = rec
            table.append((set_, model, rec))
    (OUT / "summary_multiset.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    cur = None
    for set_, model, r in table:
        if set_ != cur:
            cur = set_; print(f"\n=== {set_}  (n={summary['sets'][set_]['n']}, {summary['sets'][set_]['hours']} h) ===")
        extra = f"  no-Latin {r['no_latin_ref']['wer']*100:6.2f}% (n={r['no_latin_ref']['n']})  Latin {r['latin_ref']['wer']*100:6.2f}%" if "no_latin_ref" in r else ""
        ms = f"{r['ms_per_clip_mean']:8.1f}" if r["ms_per_clip_mean"] else "     N/A"
        print(f"{model:24s} WER {r['wer']*100:6.2f}% [{r['wer_ci_reference_clustered'][0]*100:.2f}-{r['wer_ci_reference_clustered'][1]*100:.2f}]  CER {r['cer']*100:5.2f}%  {ms} ms  empty={r['empty_hyps']}{extra}  {'' if r['status']=='complete' else '['+str(r['status'])[:40]+']'}")
    print(f"\nwrote {OUT/'summary_multiset.json'}")


if __name__ == "__main__":
    main()
