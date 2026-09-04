"""Collect every benchmark run into outputs/summary.json.

The report renders from this file rather than from numbers typed into the
plotting script, so a figure in the PDF cannot drift from the run that produced
it. Accuracy is recomputed here from each run's raw predictions.json using the
harness's own normaliser, so the intervals and the headline WER come from the
same hypotheses.
"""

import json
import re
import unicodedata
from pathlib import Path

import jiwer
import numpy as np

OUT = Path("outputs")

# Same normaliser as the harness (scripts/benchmark.py: normalize_for_wer).
PUNCT = re.compile(r"[\"“”‘’'।,.!?;:()\[\]{}—–\-]")


def norm(text: str) -> str:
    """Punctuation to word boundaries, then Unicode NFC.

    NFC matters here: Bengali writes য় either as U+09DF or as U+09AF + U+09BC
    (likewise ড় and ঢ়). These are canonically equivalent but differ byte-wise.
    The references use the decomposed spelling; Whisper and wav2vec2 emit the
    precomposed one. Without normalising, every such character counts as a
    substitution, which inflated Whisper medium from 16.22% to 27.87% WER and
    wav2vec2 from 20.71% to 31.58% in the original run of this report.
    """
    return unicodedata.normalize(
        "NFC", re.sub(r"\s+", " ", PUNCT.sub(" ", text)).strip())


# The four compared models, all served locally through one process on one GPU.
RUNS = [
    ("hishab_conformer_large", "Conformer Large",
     "hishab/titu_stt_bn_conformer_large", "121.5M", "RTX 5080", False),
    ("ehzawad_fastconformer", "ehzawad FastConformer",
     "ehzawad/stt_bn_fastconformer", "115.6M", "RTX 5080", True),
    ("wav2vec2_local", "Wav2Vec2",
     "SayedShaun/bangla-wave2vec2-unigram", "315.5M", "RTX 5080", False),
    ("hishab_fastconformer", "hishab FastConformer",
     "hishab/titu_stt_bn_fastconformer", "115.6M", "RTX 5080", False),
    ("whisper_medium", "Whisper Medium",
     "SayedShaun/bengali-whisper-medium", "763.9M", "RTX 5080", False),
]

# Kept separate, never in the comparison: the same wav2vec2 weights reached over
# the network as the pre-existing Triton service, for the before/after on page 3.
ASIDES = [
    ("wav2vec2_remote", "Wav2Vec2 (remote service)",
     "same weights, ONNX under Triton", "315.5M", "RTX 3070 (remote)", False),
]

NUM = r"([0-9]+\.?[0-9]*)"


def parse_report(path: Path) -> dict:
    text = path.read_text()

    def grab(pattern, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    return {
        "throughput": grab(rf"Throughput\s+{NUM}"),
        "wall_clock": grab(rf"Batch wall clock\s+{NUM}"),
        "lat_mean": grab(rf"mean {NUM}s"),
        "lat_p50": grab(rf"p50\s+{NUM}s"),
        "lat_p95": grab(rf"p95\s+{NUM}s"),
        "lat_p99": grab(rf"p99 {NUM}s"),
        "failed": grab(r"Utterances\s+\d+\s+\((\d+) failed\)", int),
        "n": grab(r"Utterances\s+(\d+)", int),
    }


def score(records: list[dict], rng) -> dict:
    refs = [norm(r["reference"]) for r in records]
    hyps = [norm(r["transcript"]) for r in records]
    words = jiwer.process_words(refs, hyps)
    chars = jiwer.process_characters(refs, hyps)

    # Utterance-level bootstrap: resample utterances, recompute corpus WER.
    per = [jiwer.process_words([a], [b]) for a, b in zip(refs, hyps)]
    err = np.array([p.substitutions + p.insertions + p.deletions for p in per], float)
    tot = np.array([p.substitutions + p.deletions + p.hits for p in per], float)
    idx = rng.integers(0, len(records), (10000, len(records)))
    boot = err[idx].sum(1) / tot[idx].sum(1)

    return {
        "wer": words.wer, "cer": chars.cer,
        "wer_lo": float(np.percentile(boot, 2.5)),
        "wer_hi": float(np.percentile(boot, 97.5)),
        "word": [words.substitutions, words.insertions, words.deletions, words.hits],
        "char": [chars.substitutions, chars.insertions, chars.deletions, chars.hits],
    }


def main() -> None:
    rng = np.random.default_rng(0)
    summary = {"models": [], "speed_probe": {}}

    for key, name, checkpoint, params, gpu, is_new in RUNS:
        d = OUT / key
        if not (d / "predictions.json").exists():
            print(f"skip {key}: no predictions.json")
            continue
        records = json.loads((d / "predictions.json").read_text())
        entry = {
            "key": key, "name": name, "checkpoint": checkpoint,
            "params": params, "gpu": gpu, "is_new": is_new,
            "local": gpu.startswith("RTX 5080"),
        }
        entry.update(parse_report(d / "report.txt"))
        entry.update(score(records, rng))

        # Split-level scores: the merged set is the comparison basis, but
        # test-only is what published FLEURS numbers are quoted on.
        for split in ("test", "validation"):
            sub = [r for r in records if f"_{split}_" in r["file"]]
            if sub:
                s = score(sub, rng)
                entry[f"wer_{split}"] = s["wer"]
                entry[f"cer_{split}"] = s["cer"]
        summary["models"].append(entry)
        print(f"{name:24s} WER {entry['wer']:.4f} "
              f"[{entry['wer_lo']:.4f},{entry['wer_hi']:.4f}]  "
              f"{entry['throughput']:.2f} req/s")

    summary["asides"] = []
    for key, name, checkpoint, params, gpu, is_new in ASIDES:
        d = OUT / key
        if not (d / "predictions.json").exists():
            continue
        records = json.loads((d / "predictions.json").read_text())
        entry = {"key": key, "name": name, "checkpoint": checkpoint,
                 "params": params, "gpu": gpu, "is_new": is_new, "local": False}
        entry.update(parse_report(d / "report.txt"))
        entry.update(score(records, rng))
        summary["asides"].append(entry)
        print(f"{name:24s} WER {entry['wer']:.4f}  "
              f"{entry['throughput']:.2f} req/s   (aside, not compared)")

    probe = OUT / "speed_probe.json"
    if probe.exists():
        summary["speed_probe"] = json.loads(probe.read_text())

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
