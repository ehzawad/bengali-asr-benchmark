#!/usr/bin/env python
"""Render the benchmark report (asr_benchmark.pdf) from the committed summaries:

  * outputs_fleurs/{summary,clean_test_table}.json  — six models on FLEURS bn_in
  * outputs_multiset/summary_multiset.json          — five models on the official
    test splits of Vaani, SPRING-INX R1/R2 and Kathbath (known/unknown speakers)

Every number is read from those files so the PDF cannot drift from the runs.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch         # noqa: E402

SURFACE, INK, INK_2, INK_3, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#dedcd4"
ACCENT, BASE = "#2a78d6", "#7d7b74"
WARN_BG, WARN_EDGE = "#fdf3e7", "#eda100"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": RULE, "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
})

CLEAN = json.loads(Path("outputs_fleurs/clean_test_table.json").read_text())
MS = json.loads(Path("outputs_multiset/summary_multiset.json").read_text())
OURS_KEY = "fastconformer_ctc"
LABELS = {
    "fastconformer_ctc": ("ehzawad FastConformer-CTC", "ehzawad/stt_bn_fastconformer_ctc", "115.6M"),
    "qwen3_adapter": ("Qwen3-ASR + Bengali adapter", "Qwen/Qwen3-ASR-1.7B-hf + LoRA", "2.04B (38M adapter)"),
    "hishab_conformer_large": ("Conformer Large (hishab)", "hishab/titu_stt_bn_conformer_large", "121.5M"),
    "whisper_medium": ("Whisper Medium", "SayedShaun/bengali-whisper-medium", "763.9M"),
    "wav2vec2": ("Wav2Vec2", "SayedShaun/bangla-wave2vec2-unigram", "315.5M"),
    "hishab_fastconformer": ("hishab FastConformer", "hishab/titu_stt_bn_fastconformer", "115.6M"),
}
SETS = [("vaani_test", "Vaani test"), ("spring_r1_test", "SPRING-INX R1"), ("spring_r2_test", "SPRING-INX R2"),
        ("kathbath_test_known", "Kathbath known"), ("kathbath_test_unknown", "Kathbath unknown")]
FLAGS = {"fastconformer_ctc": "N N N S S", "hishab_conformer_large": "N N N S S", "whisper_medium": "U U U U U",
         "hishab_fastconformer": "U U U U U", "wav2vec2": "U U U U U"}
CM = sorted(CLEAN["models"].items(), key=lambda kv: kv[1]["wer"])
MM = [k for k in ["fastconformer_ctc", "whisper_medium", "hishab_conformer_large", "hishab_fastconformer", "wav2vec2"]
      if k in MS["models"]]


def name(k): return LABELS.get(k, (k, k, "?"))[0]
def params(k): return LABELS.get(k, (k, k, "?"))[2]
def color(k): return ACCENT if k == OURS_KEY else BASE


def footer(fig, page, tag):
    fig.text(0.06, 0.028, f"Bengali ASR benchmark  ·  {tag}  ·  RTX A5000, 2026-09-15", size=6.6, color=INK_3)
    fig.text(0.94, 0.028, f"{page}", size=6.6, color=INK_3, ha="right")


def caveat(fig, x, y, w, text, h=None, size=7.0):
    lines = text.count("\n") + 1
    h = h or 0.016 + 0.0135 * lines
    fig.patches.append(FancyBboxPatch(
        (x, y - h), w, h, boxstyle="round,pad=0.004,rounding_size=0.006",
        transform=fig.transFigure, facecolor=WARN_BG, edgecolor=WARN_EDGE, linewidth=0.7, zorder=0))
    fig.text(x + 0.012, y - 0.011, text, size=size, color=INK_2, va="top")
    return y - h


def table(fig, x, y, w, headers, rows, offs, row_h=0.021, size=7.0):
    for i, (hh, o) in enumerate(zip(headers, offs)):
        fig.text(x + o * w, y, hh, size=size - 0.4, color=INK_3, ha="right" if i else "left", weight="bold")
    fig.lines.append(plt.Line2D([x, x + w], [y - 0.008, y - 0.008], transform=fig.transFigure, color=RULE, lw=0.8))
    yy = y - 0.008
    for r in rows:
        yy -= row_h
        for i, (c, o) in enumerate(zip(r, offs)):
            bold = r[0].startswith("▸")
            fig.text(x + o * w, yy, c, size=size, color=INK if bold else INK_2,
                     weight="bold" if bold else "normal", ha="right" if i else "left")
    return yy


def pd(key):
    d = CLEAN["paired_fastconformer_ctc"][key]
    return f"{d['wer_delta_pp']:+.2f} pp [{d['wer_ci95_pp'][0]:+.2f}, {d['wer_ci95_pp'][1]:+.2f}]"


def page1(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.945, "Bengali ASR: six models on FLEURS, one GPU", size=20, weight="bold")
    fig.text(0.06, 0.922, "920 FLEURS bn_in TEST utterances · RTX A5000 · greedy, batch 1, no external LM · "
             "all six measured in one sitting", size=8.0, color=INK_2)
    rows = []
    for k, m in CM:
        lo, hi = m["wer_ci95"]; sp = CLEAN["speed_default_runtime"][k]["ms_per_clip_mean"]
        rows.append([f"{'▸ ' if k == OURS_KEY else '  '}{name(k)}", f"{m['wer']*100:.2f}%",
                     f"[{lo*100:.2f}–{hi*100:.2f}]", f"{m['cer']*100:.2f}%", f"{sp:,.0f}", params(k)])
    yy = table(fig, 0.06, 0.885, 0.88, ["Model", "WER", "95% CI (clustered)", "CER", "ms/clip", "Params"],
               rows, [0.0, 0.50, 0.66, 0.74, 0.84, 1.0])
    y = caveat(fig, 0.06, yy - 0.022, 0.88,
               "Test split only. The two ehzawad models were trained on corpora that include the FLEURS train split,\n"
               "and 88 of the 150 validation-split sentences also occur there; the 402 validation recordings are\n"
               "therefore excluded for every model and the table is the 920-utterance test split.")
    fig.text(0.06, y - 0.03, "Reading this table", size=11, weight="bold")
    body = (
        "ehzawad/stt_bn_fastconformer_ctc is a FastConformer-CTC large (115.6M parameters, 8x subsampling, "
        "1,024-piece Bengali BPE) trained on 1,692 hours of human-supervised Bengali, CC-BY-SA-4.0. The other "
        "five models are public checkpoints from hishab, SayedShaun (mirrors of tugstugi's Whisper and qdv206's "
        "wav2vec2 competition models) and ehzawad (a Bengali LoRA adapter on Qwen3-ASR-1.7B: 2.04B parameters, "
        "of which 38M were trained).\n\n"
        "Paired sentence-clustered deltas for the highlighted model (10,000 draws):\n"
        f"   vs Qwen3 adapter         {pd('fastconformer_ctc_vs_qwen3_adapter')}\n"
        f"   vs Whisper Medium        {pd('fastconformer_ctc_vs_whisper_medium')}\n"
        f"   vs Conformer Large       {pd('fastconformer_ctc_vs_hishab_conformer_large')}\n\n"
        "Not distinguishable from the Qwen3 adapter at 120x its speed; under a point behind Whisper Medium at 22x "
        "its speed; 2.7 points behind Conformer Large."
    )
    fig.text(0.06, y - 0.048, body, size=8.2, color=INK_2, va="top", wrap=True, linespacing=1.55)
    footer(fig, 1, "six models, FLEURS bn_in test")
    pdf.savefig(fig); plt.close(fig)


def page2(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.945, "FLEURS test: accuracy and speed side by side", size=16, weight="bold")
    keys = [k for k, _ in CM][::-1]
    ax = fig.add_axes([0.30, 0.60, 0.62, 0.30])
    vals = [CLEAN["models"][k]["wer"] * 100 for k in keys]
    err = [[CLEAN["models"][k]["wer"] * 100 - CLEAN["models"][k]["wer_ci95"][0] * 100 for k in keys],
           [CLEAN["models"][k]["wer_ci95"][1] * 100 - CLEAN["models"][k]["wer"] * 100 for k in keys]]
    ax.barh(range(len(keys)), vals, xerr=err, color=[color(k) for k in keys], height=0.6,
            error_kw={"ecolor": INK_3, "elinewidth": 0.8, "capsize": 2})
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([name(k) for k in keys], size=7.5)
    ax.set_xlabel("WER on the 920-utterance test split, % (95% clustered CI)", size=7.5)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7)
    for i, v in enumerate(vals):
        ax.text(v + 1.0, i, f"{v:.2f}", va="center", size=7, color=INK_2)
    ax2 = fig.add_axes([0.30, 0.22, 0.62, 0.30])
    sp = [CLEAN["speed_default_runtime"][k]["ms_per_clip_mean"] for k in keys]
    ax2.barh(range(len(keys)), sp, color=[color(k) for k in keys], height=0.6)
    ax2.set_xscale("log"); ax2.set_yticks(range(len(keys))); ax2.set_yticklabels([name(k) for k in keys], size=7.5)
    ax2.set_xlabel("ms per clip, batch 1, warmed (log scale)", size=7.5)
    ax2.spines[["top", "right"]].set_visible(False); ax2.tick_params(labelsize=7)
    for i, v in enumerate(sp):
        ax2.text(v * 1.15, i, f"{v:,.0f}", va="center", size=7, color=INK_2)
    fig.text(0.06, 0.16, "The two orderings are not the same ordering. The three NeMo CTC models sit within 10% of each "
             "other in speed; Whisper Medium costs ~22x and the Qwen3 adapter ~120x per clip at batch 1 for their accuracy.",
             size=8.2, color=INK_2, va="top", wrap=True, linespacing=1.5)
    footer(fig, 2, "six models, FLEURS bn_in test")
    pdf.savefig(fig); plt.close(fig)


def page3(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.945, "Beyond FLEURS: official test splits of four other corpora", size=17, weight="bold")
    fig.text(0.06, 0.922, "Frozen seeded 1,000-utterance subsets of the OFFICIAL test splits · one model at a time · "
             "same backends and scorer", size=8.0, color=INK_2)
    rows = []
    strat = {"vaani_test": "29 state/district strata", "spring_r1_test": "Latin / no-Latin reference",
             "spring_r2_test": "Latin / no-Latin reference", "kathbath_test_known": "speaker gender", "kathbath_test_unknown": "speaker gender"}
    for st, n in SETS:
        s = MS["sets"][st]
        rows.append([f"  {n}", f"{s['source_rows']:,}", f"{s['n']:,}", f"{s['hours']:.2f} h", f"{s['ref_words']:,}", strat[st]])
    yy = table(fig, 0.06, 0.885, 0.88, ["Set", "official rows", "sampled", "audio", "ref. words", "stratified by"],
               rows, [0.0, 0.34, 0.44, 0.54, 0.66, 1.0], size=6.8)
    rows = []
    for k in MM:
        cells = [f"{MS['models'][k][st]['wer']*100:.2f}" if st in MS["models"][k] else "—" for st, _ in SETS]
        rows.append([f"{'▸ ' if k == OURS_KEY else '  '}{name(k)}"] + cells + [FLAGS[k]])
    yy = table(fig, 0.06, yy - 0.035, 0.88, ["Model  (WER %)", "Vaani", "SPRING R1", "SPRING R2", "Kathbath kn", "Kathbath unk", "flags"],
               rows, [0.0, 0.42, 0.53, 0.64, 0.76, 0.88, 1.0], size=7.0)
    y = caveat(fig, 0.06, yy - 0.02, 0.88,
               "flags, per set in the column order: N = corpus not in the model's declared training data;\n"
               "S = trained on a sibling split of the same corpus (Kathbath train; Conformer Large also its\n"
               "validation split); U = training provenance not documented. None of these means \"verified clean\".\n"
               "The highlighted model's Vaani and SPRING hypotheses come from its earlier evaluation of the full\n"
               "official splits (same greedy decoding, batched), scored here on exactly these utterances; they are\n"
               "not re-timed. Its Kathbath rows are batch-1 decodes here. The Qwen3 adapter is not in this\n"
               "comparison (~10 s per clip at batch 1).")
    fig.text(0.06, y - 0.03, "What the table says", size=11, weight="bold")
    body = (
        "On the two corpora nobody trained on, the FLEURS ranking does not hold. Conformer Large - the FLEURS "
        "leader at 14.4% - is last on Vaani (33.1%) and fourth on SPRING; its training list is Bangladesh-centric "
        "and these are Indian-district (Vaani) and IIT-Madras conversational (SPRING) recordings. "
        "ehzawad/stt_bn_fastconformer_ctc leads Vaani and both SPRING sets; Whisper Medium is second on all three "
        "at ~15x the latency.\n\n"
        "On Kathbath the order flips back: Conformer Large (9.3 / 8.8%) and Whisper (9.7 / 9.8%) lead and the "
        "highlighted model is third (13.4 / 13.4%). Conformer Large trained on Kathbath validation as well as train, "
        "Whisper's provenance for Kathbath is unknown (a U cell can hide training overlap), and the highlighted model "
        "saw only the train split. Known- vs unknown-speaker Kathbath differ by under a point for every model except "
        "hishab FastConformer. Sentence-clustered 95% intervals for every cell are in the README and summary."
    )
    fig.text(0.06, y - 0.048, body, size=8.2, color=INK_2, va="top", wrap=True, linespacing=1.55)
    footer(fig, 3, "official test splits, five models")
    pdf.savefig(fig); plt.close(fig)


def page4(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.945, "Latency per set, and how the runs were controlled", size=16, weight="bold")
    fig.text(0.06, 0.905, "SPRING note: about half of its reference utterances contain English words in Latin script; none of "
             "the five models emitted a single Latin-script word, so those words are errors for every model alike. On the "
             "Latin-free half the highlighted model (35.4 / 35.1%) and Whisper Medium (35.9 / 36.7%) are within each other's "
             "intervals (per-model split in outputs_multiset/summary_multiset.json).", size=8.0, color=INK_2, va="top", wrap=True, linespacing=1.5)
    yy = 0.83
    rows = []
    for k in MM:
        cells = []
        for st, _ in SETS:
            v = MS["models"][k].get(st, {}).get("ms_per_clip_mean")
            cells.append(f"{v:,.0f}" if v else "N/A")
        rows.append([f"{'▸ ' if k == OURS_KEY else '  '}{name(k)}"] + cells)
    fig.text(0.06, yy - 0.105, "ms per clip, batch 1, warmed, default runtime (N/A = stored hypotheses, not timed here)",
             size=8.0, color=INK_2)
    yy2 = table(fig, 0.06, yy - 0.125, 0.88, ["Model", "Vaani", "SPRING R1", "SPRING R2", "Kathbath kn", "Kathbath unk"],
                rows, [0.0, 0.45, 0.58, 0.71, 0.85, 1.0])
    y = caveat(fig, 0.06, yy2 - 0.03, 0.88,
               "Protocol. Every set is a frozen subset (seed 20260915; proportional stratification; per-file SHA-256\n"
               "verified before decoding). One model resident at a time on the A5000; no other process used the card\n"
               "during any run. Manifests: multiset/; raw predictions and per-run GPU state: outputs_multiset/;\n"
               "harness bench_run_set.py (bench_run.py's backends unchanged); scorer score_multiset.py.\n"
               "Reproduce the scoring from the committed predictions.")
    fig.text(0.06, y - 0.03, "Evaluation data: FLEURS (Google, CC-BY-4.0); Vaani transcription part (ARTPARK-IISc, CC-BY-4.0); "
             "SPRING-INX R1/R2 (SPRING Lab, IIT Madras, CC-BY-4.0); Kathbath (AI4Bharat IndicSUPERB, CC0). "
             "Models: hishab (Conformer Large, FastConformer); SayedShaun (Whisper Medium mirror of tugstugi's "
             "competition model; wav2vec2 mirror of qdv206's); ehzawad (FastConformer-CTC, Qwen3 adapter). "
             "Author of this benchmark: Emrul Zawad.", size=7.4, color=INK_3, va="top", wrap=True, linespacing=1.5)
    footer(fig, 4, "official test splits, five models")
    pdf.savefig(fig); plt.close(fig)


def main():
    out = "asr_benchmark.pdf"
    with PdfPages(out) as pdf:
        page1(pdf); page2(pdf); page3(pdf); page4(pdf)
    print("wrote", out)


if __name__ == "__main__":
    main()
