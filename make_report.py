#!/usr/bin/env python
"""Render the benchmark report (asr_benchmark.pdf) from the committed summaries:

  * outputs_fleurs/clean_test_table.json    — five models on the FLEURS bn_in test split
  * outputs_multiset/summary_multiset.json  — the same five on the official test splits of
    Vaani, SPRING-INX R1/R2, Kathbath (known/unknown speakers) and IndicVoices conversational
All five ran on one NVIDIA A100-SXM4-40GB (Colab), greedy at batch 1, one model at a time.
Voxtral Mini bn-BD, a private fine-tune, went through the same harness, decoded by vLLM.

Every number is read from those files so the PDF cannot drift from the runs, and the
orderings the prose states are asserted against them.
"""
import json
import math
import statistics
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
OURS_KEY, MINI_KEY, MINI_COLOR = "fastconformer_ctc", "voxtral_mini_bn", "#c0571f"
LABELS = {
    "fastconformer_ctc": ("ehzawad FastConformer-CTC", "ehzawad/stt_bn_fastconformer_ctc", "115.6M"),
    "hishab_conformer_large": ("Conformer Large (hishab)", "hishab/titu_stt_bn_conformer_large", "121.5M"),
    "whisper_medium": ("Whisper Medium", "SayedShaun/bengali-whisper-medium", "763.9M"),
    "wav2vec2": ("Wav2Vec2", "SayedShaun/bangla-wave2vec2-unigram", "315.5M"),
    "voxtral_mini_bn": ("Voxtral Mini bn-BD ‡", "ehzawad/Voxtral-Mini-3B-2507-bn-BD (private)", "4,676M"),
}
SETS = [("vaani_test", "Vaani test"), ("spring_r1_test", "SPRING-INX R1"), ("spring_r2_test", "SPRING-INX R2"),
        ("kathbath_test_known", "Kathbath known"), ("kathbath_test_unknown", "Kathbath unknown"),
        ("indicvoices_conv", "IndicVoices conv.")]
IV = "indicvoices_conv"
FLAGS = {"fastconformer_ctc": "N N N S S S", "hishab_conformer_large": "N N N S S N", "whisper_medium": "U U U U U U",
         "wav2vec2": "U U U U U U", "voxtral_mini_bn": "U U U U U U"}
CM = sorted(CLEAN["models"].items(), key=lambda kv: kv[1]["wer"])
MM = [k for k in ["fastconformer_ctc", "whisper_medium", "hishab_conformer_large", "wav2vec2",
                  "voxtral_mini_bn"] if k in MS["models"]]
PUBLIC = [k for k in MM if k != MINI_KEY]
NEMO = ["fastconformer_ctc", "hishab_conformer_large"]
SPEED = CLEAN["speed_default_runtime"]  # batch-1 latency per FLEURS test clip, on the A100
GPU_TAG = "A100-SXM4-40GB, 2026-10-13"
ORDINAL = ["first", "second", "third", "fourth", "fifth"]


def name(k): return LABELS.get(k, (k, k, "?"))[0]
def params(k): return LABELS.get(k, (k, k, "?"))[2]
def color(k): return ACCENT if k == OURS_KEY else (MINI_COLOR if k == MINI_KEY else BASE)


def wer(k, st=None):
    """WER in percent: the FLEURS test split when st is None, else that official-split subset."""
    return 100 * (CLEAN["models"][k]["wer"] if st is None else MS["models"][k][st]["wer"])


def ms(k, st=None):
    return SPEED[k]["ms_per_clip_mean"] if st is None else MS["models"][k][st]["ms_per_clip_mean"]


def place(k, st=None, among=None):
    """k's rank by WER on st among the given models (default: all five): "first" ... "last"."""
    order = sorted(among or MM, key=lambda m: wer(m, st))
    i = order.index(k)
    return "last" if i == len(order) - 1 else ORDINAL[i]


def slower(k, sets):
    """Range of k's per-clip latency as a multiple of each NeMo CTC model's, over the given sets."""
    r = [ms(k, st) / ms(n, st) for st in sets for n in NEMO]
    return min(r), max(r)


def nolatin(k, st):
    r = MS["models"][k][st]["no_latin_ref"]
    return 100 * r["wer"], [100 * x for x in r["wer_ci_reference_clustered"]]


def footer(fig, page, tag, where=GPU_TAG):
    fig.text(0.06, 0.028, f"Bengali ASR benchmark  ·  {tag}  ·  {where}", size=6.6, color=INK_3)
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


def pd(key, block="paired_fastconformer_ctc"):
    d = CLEAN[block][key]
    return f"{d['wer_delta_pp']:+.2f} pp [{d['wer_ci95_pp'][0]:+.2f}, {d['wer_ci95_pp'][1]:+.2f}]"


def verdict(d):
    lo, hi = d["wer_ci95_pp"]
    return "lower" if hi < 0 else ("higher" if lo > 0 else "inconclusive")


def page1(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.945, "Bengali ASR: five models on FLEURS", size=20, weight="bold")
    fig.text(0.06, 0.922, "920 FLEURS bn_in TEST utterances · one A100-SXM4-40GB · greedy, batch 1, no external LM · "
             "‡ decoded by vLLM", size=8.0, color=INK_2)
    rows = []
    for k, m in CM:
        lo, hi = m["wer_ci95"]; sp = SPEED.get(k, {}).get("ms_per_clip_mean")
        rows.append([f"{'▸ ' if k == OURS_KEY else '  '}{name(k)}", f"{m['wer']*100:.2f}%",
                     f"[{lo*100:.2f}–{hi*100:.2f}]", f"{m['cer']*100:.2f}%", f"{sp:,.0f}" if sp else "—", params(k)])
    yy = table(fig, 0.06, 0.885, 0.88, ["Model", "WER", "95% CI (clustered)", "CER", "ms/clip", "Params"],
               rows, [0.0, 0.50, 0.66, 0.74, 0.84, 1.0])
    y = caveat(fig, 0.06, yy - 0.022, 0.88,
               "Test split only. The ehzawad model trained on a corpus that includes the FLEURS train split, and 88 of\n"
               "the 150 validation-split sentences also occur there; the 402 validation recordings are therefore\n"
               "excluded for every model and the table is the 920-utterance test split.")
    fig.text(0.06, y - 0.03, "Reading this table", size=11, weight="bold")
    d_wh = CLEAN["paired_fastconformer_ctc"]["fastconformer_ctc_vs_whisper_medium"]["wer_delta_pp"]
    d_cl = CLEAN["paired_fastconformer_ctc"]["fastconformer_ctc_vs_hishab_conformer_large"]["wer_delta_pp"]
    assert d_wh > 0 and d_cl > 0
    body = (
        "ehzawad/stt_bn_fastconformer_ctc is a FastConformer-CTC large (115.6M parameters, 8x subsampling, "
        "1,024-piece Bengali BPE) trained on 1,692 hours of human-supervised Bengali, CC-BY-SA-4.0. The other "
        "three public models are checkpoints from hishab and SayedShaun (mirrors of tugstugi's Whisper and qdv206's "
        "wav2vec2 competition models).\n\n"
        "Paired sentence-clustered deltas for the highlighted model (10,000 draws):\n"
        f"   vs Whisper Medium        {pd('fastconformer_ctc_vs_whisper_medium')}\n"
        f"   vs Conformer Large       {pd('fastconformer_ctc_vs_hishab_conformer_large')}\n\n"
        f"{d_wh:.1f} points behind Whisper Medium at {ms('whisper_medium') / ms(OURS_KEY):.0f}x its speed; "
        f"{d_cl:.1f} points behind Conformer Large. The highlighted model had also been evaluated on FLEURS test "
        "during its own development, so read its FLEURS row as a like-for-like re-measurement and the other corpora "
        "as the fresh evidence."
    )
    if MINI_KEY in CLEAN["models"]:
        body += (
            "\n\n‡ Voxtral Mini bn-BD is a private fine-tune of Voxtral Mini 3B whose card documents no training data. "
            "It went through the same harness, asr_core path and scorer, decoded at batch 1 by vLLM 0.31 rather than "
            "by PyTorch directly. Against Conformer Large on this split: "
            f"{pd('voxtral_mini_bn_vs_hishab_conformer_large', 'paired_voxtral_mini_bn')}, "
            f"{verdict(CLEAN['paired_voxtral_mini_bn']['voxtral_mini_bn_vs_hishab_conformer_large'])}."
        )
    fig.text(0.06, y - 0.048, body, size=8.2, color=INK_2, va="top", wrap=True, linespacing=1.55)
    footer(fig, 1, "FLEURS bn_in test")
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
    for i, (v, top) in enumerate(zip(vals, err[1])):
        ax.text(v + top + 0.4, i, f"{v:.2f}", va="center", size=7, color=INK_2)
    ax2 = fig.add_axes([0.30, 0.22, 0.62, 0.30])
    keys = [k for k in keys if k in SPEED]
    sp = [SPEED[k]["ms_per_clip_mean"] for k in keys]
    ax2.barh(range(len(keys)), sp, color=[color(k) for k in keys], height=0.6)
    ax2.set_xscale("log"); ax2.set_yticks(range(len(keys))); ax2.set_yticklabels([name(k) for k in keys], size=7.5)
    ax2.set_xlabel("ms per clip, batch 1, warmed (log scale)", size=7.5)
    ax2.spines[["top", "right"]].set_visible(False); ax2.tick_params(labelsize=7)
    for i, v in enumerate(sp):
        ax2.text(v * 1.15, i, f"{v:,.0f}", va="center", size=7, color=INK_2)
    nemo = [ms(n) for n in NEMO]
    spread = math.ceil((max(nemo) / min(nemo) - 1) * 100)
    fig.text(0.06, 0.16, "The two orderings are not the same ordering. The two NeMo CTC models are within "
             f"{spread}% of each other in speed; Whisper Medium costs ~{ms('whisper_medium') / statistics.mean(nemo):.0f}x "
             f"their time per clip at batch 1 for its accuracy, and Voxtral Mini ~{ms(MINI_KEY) / statistics.mean(nemo):.0f}x "
             "(vLLM with CUDA graphs).",
             size=8.2, color=INK_2, va="top", wrap=True, linespacing=1.5)
    footer(fig, 2, "FLEURS bn_in test")
    pdf.savefig(fig); plt.close(fig)


def page3(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.945, "Beyond FLEURS: six more test sets", size=17, weight="bold")
    fig.text(0.06, 0.922, "Frozen seeded 1,000-utterance subsets of the OFFICIAL splits · one model at a time · "
             "same backends and scorer", size=8.0, color=INK_2)
    rows = []
    strat = {"vaani_test": "29 state/district strata", "spring_r1_test": "Latin / no-Latin reference",
             "spring_r2_test": "Latin / no-Latin reference", "kathbath_test_known": "speaker gender", "kathbath_test_unknown": "speaker gender",
             "indicvoices_conv": "scenario (conversation / extempore)"}
    for st, n in SETS:
        s = MS["sets"][st]
        rows.append([f"  {n}", f"{s['source_rows']:,}", f"{s['n']:,}", f"{s['hours']:.2f} h", f"{s['ref_words']:,}", strat[st]])
    yy = table(fig, 0.06, 0.885, 0.88, ["Set", "official rows", "sampled", "audio", "ref. words", "stratified by"],
               rows, [0.0, 0.34, 0.44, 0.54, 0.66, 1.0], size=6.8)
    rows = []
    for k in MM:
        cells = [f"{MS['models'][k][st]['wer']*100:.2f}" if st in MS["models"][k] else "—" for st, _ in SETS]
        rows.append([f"{'▸ ' if k == OURS_KEY else '  '}{name(k)}"] + cells + [FLAGS[k]])
    yy = table(fig, 0.06, yy - 0.035, 0.88, ["Model  (WER %)", "Vaani", "SPR R1", "SPR R2", "Kb known", "Kb unk", "IV conv", "flags"],
               rows, [0.0, 0.36, 0.46, 0.56, 0.66, 0.76, 0.86, 1.0], size=6.8)
    y = caveat(fig, 0.06, yy - 0.02, 0.88,
               "flags, per set in the column order: N = corpus not in the model's declared training data;\n"
               "S = trained on a sibling split of the same corpus (Kathbath train; Conformer Large also its\n"
               "validation split); U = training provenance not documented. None of these means \"verified clean\".\n"
               "Every cell is a batch-1 decode from this run. IndicVoices conversational is NOT a clean head-to-head\n"
               "for the highlighted model: it trained on the IndicVoices train split (same collection, prompts and\n"
               f"transcription conventions, different speakers), which is a large part of its {wer(OURS_KEY, IV):.1f}% there.")
    fig.text(0.06, y - 0.03, "What the table says", size=11, weight="bold")
    cl, wh = "hishab_conformer_large", "whisper_medium"
    unseen, kb = ["vaani_test", "spring_r1_test", "spring_r2_test"], ["kathbath_test_known", "kathbath_test_unknown"]
    assert place(cl, among=PUBLIC) == "first" and place(cl, "vaani_test") == "last"
    assert [place(cl, st) for st in unseen[1:]] == ["fourth", "fourth"]
    assert all(place(OURS_KEY, st, PUBLIC) == "first" and place(wh, st, PUBLIC) == "second" for st in unseen)
    assert all(wer(MINI_KEY, st) < wer(OURS_KEY, st) for st in unseen[:2])
    assert all(sorted(MM, key=lambda m: wer(m, st)) == [cl, wh, MINI_KEY, OURS_KEY, "wav2vec2"] for st in kb)
    assert all(abs(wer(k, kb[0]) - wer(k, kb[1])) < 1 for k in MM)
    lo, hi = slower(wh, unseen)
    others = [k for k in PUBLIC if k != OURS_KEY]
    empty = {k: MS["models"][k][IV]["empty_hyps"] for k in others}
    kbs = lambda k: f"{wer(k, kb[0]):.1f} / {wer(k, kb[1]):.1f}%"
    body = (
        "On the corpora nobody is documented as having trained on, the FLEURS ranking does not hold. Conformer Large, "
        f"the FLEURS leader at {wer(cl):.1f}%, is last on Vaani ({wer(cl, 'vaani_test'):.1f}%) and fourth on both SPRING "
        "sets; its model card lists read and broadcast corpora and neither Vaani nor SPRING-INX. "
        "ehzawad/stt_bn_fastconformer_ctc leads the four public models on Vaani and both SPRING sets; Whisper Medium "
        f"is second on all three at {lo:.0f}-{hi:.0f}x the latency of the NeMo CTC models. The private Voxtral Mini "
        "fine-tune (‡, U on every set) is lower still on Vaani and SPRING R1 (page 5).\n\n"
        f"On Kathbath the order flips back: Conformer Large ({kbs(cl)}) and Whisper ({kbs(wh)}) lead, Voxtral Mini "
        f"follows ({kbs(MINI_KEY)}) and the highlighted model is fourth ({kbs(OURS_KEY)}). Conformer Large trained on "
        "Kathbath validation as well as train, Whisper's provenance for Kathbath is unknown (a U cell can hide "
        "training overlap), and the highlighted model saw only the train split. Known- vs unknown-speaker Kathbath "
        "differ by under a point for every model. On IndicVoices conversational the three other public models read "
        f"{min(wer(k, IV) for k in others):.0f}-{max(wer(k, IV) for k in others):.0f}%; Conformer Large, the one N "
        f"cell, emitted {empty[cl]} empty hypotheses there and Wav2Vec2 {empty['wav2vec2']}. Sentence-clustered 95% "
        "intervals for every cell are in the README and summary."
    )
    fig.text(0.06, y - 0.048, body, size=8.2, color=INK_2, va="top", wrap=True, linespacing=1.55)
    footer(fig, 3, "six more test sets")
    pdf.savefig(fig); plt.close(fig)


def page4(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.945, "Latency per set, and how the runs were controlled", size=16, weight="bold")
    spring = ["spring_r1_test", "spring_r2_test"]
    assert not any(MS["models"][k][st]["hyps_with_latin"] for k in MM for st in spring)
    nl = {k: [nolatin(k, st) for st in spring] for k in (OURS_KEY, "whisper_medium", MINI_KEY)}
    for (a, ia), (b, ib) in zip(nl[OURS_KEY], nl["whisper_medium"]):
        assert ia[0] <= b <= ia[1] and ib[0] <= a <= ib[1]   # each inside the other's interval
    both = lambda k: f"{nl[k][0][0]:.1f} / {nl[k][1][0]:.1f}%"
    fig.text(0.06, 0.905, "SPRING note: about half of its reference utterances contain English words in Latin script; no "
             "model emitted a single Latin-script word, so those words are errors for every model alike. On the "
             f"Latin-free half the highlighted model ({both(OURS_KEY)}) and Whisper Medium ({both('whisper_medium')}) are "
             f"within each other's intervals, and Voxtral Mini reads {both(MINI_KEY)} (per-model split in "
             "outputs_multiset/summary_multiset.json).", size=8.0, color=INK_2, va="top", wrap=True, linespacing=1.5)
    yy = 0.93
    rows = []
    for k in MM:
        cells = []
        for st, _ in SETS:
            v = MS["models"][k].get(st, {}).get("ms_per_clip_mean")
            cells.append(f"{v:,.0f}" if v else "N/A")
        rows.append([f"{'▸ ' if k == OURS_KEY else '  '}{name(k)}"] + cells)
    fig.text(0.06, yy - 0.105, "ms per clip, batch 1, warmed (Voxtral Mini through vLLM with CUDA graphs)",
             size=8.0, color=INK_2)
    yy2 = table(fig, 0.06, yy - 0.125, 0.88, ["Model", "Vaani", "SPR R1", "SPR R2", "Kb known", "Kb unk", "IV conv"],
                rows, [0.0, 0.40, 0.52, 0.64, 0.76, 0.88, 1.0])
    y = caveat(fig, 0.06, yy2 - 0.03, 0.88,
               "Protocol. Every set is a frozen subset (seed 20260915; proportional stratification; per-file SHA-256\n"
               "verified before decoding). One model resident at a time on the A100; no other process used the card\n"
               "during any run. Manifests: multiset/; raw predictions and per-run GPU state: outputs_multiset/;\n"
               "harness bench_run_set.py (bench_run.py's backends unchanged; Voxtral Mini through the voxtral_vllm\n"
               "backend); scorer score_multiset.py. Reproduce the scoring from the committed predictions.")
    fig.text(0.06, y - 0.03, "Evaluation data: FLEURS (Google, CC-BY-4.0); Vaani transcription part (ARTPARK-IISc, CC-BY-4.0); "
             "SPRING-INX R1/R2 (SPRING Lab, IIT Madras, CC-BY-4.0); Kathbath (AI4Bharat IndicSUPERB, CC0); IndicVoices "
             "(AI4Bharat, CC-BY-4.0). Models: hishab (Conformer Large); SayedShaun (Whisper Medium mirror of "
             "tugstugi's competition model; wav2vec2 mirror of qdv206's); ehzawad (FastConformer-CTC; Voxtral Mini bn-BD, "
             "private, fine-tuned from mistralai/Voxtral-Mini-3B-2507, Apache-2.0). "
             "Author of this benchmark: Emrul Zawad.", size=7.4, color=INK_3, va="top", wrap=True, linespacing=1.5)
    footer(fig, 4, "six more test sets")
    pdf.savefig(fig); plt.close(fig)


def page5(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.945, "Voxtral Mini bn-BD against the public models", size=16, weight="bold")
    fig.text(0.06, 0.922, "Paired difference on the same recordings, WER points, 95% reference-clustered bootstrap "
             "(10,000 draws, seed 20260903)", size=7.6, color=INK_2)
    rows = []
    best_f = min((k for k in CLEAN["models"] if k != MINI_KEY), key=lambda k: CLEAN["models"][k]["wer"])
    d = CLEAN["paired_voxtral_mini_bn"][f"voxtral_mini_bn_vs_{best_f}"]
    rows.append(["  FLEURS test", name(best_f), f"{d['wer_delta_pp']:+.2f}",
                 f"[{d['wer_ci95_pp'][0]:+.2f}, {d['wer_ci95_pp'][1]:+.2f}]", verdict(d)])
    for st, n in SETS:
        pub = {k: v[st]["wer"] for k, v in MS["models"].items() if k != MINI_KEY and st in v}
        b = min(pub, key=pub.get)
        d = MS["paired"][st][f"voxtral_mini_bn_vs_{b}"]
        rows.append([f"  {n}", name(b), f"{d['wer_delta_pp']:+.2f}",
                     f"[{d['wer_ci95_pp'][0]:+.2f}, {d['wer_ci95_pp'][1]:+.2f}]", verdict(d)])
    yy = table(fig, 0.06, 0.885, 0.88, ["Set", "best public model", "WER points", "95% CI", "paired interval"],
               rows, [0.0, 0.52, 0.66, 0.84, 1.0], size=6.9)
    r2 = MS["models"][MINI_KEY]["spring_r2_test"]
    loops = ""
    if r2.get("without_capped_clips"):
        w = r2["without_capped_clips"]
        loops = (f"On SPRING-INX R2, {r2['clips_at_token_cap']} recordings ran to the 500-token cap repeating a word; they "
                 f"contribute {r2['errors'] - w['errors']:,} of its {r2['errors']:,} word errors and stay in its "
                 f"{r2['wer']*100:.2f}% (without them {w['wer']*100:.2f}% on {w['n']:,} recordings, a conditional "
                 "diagnostic, not a corrected score). ")
    body = (
        "Voxtral Mini bn-BD (4.7B parameters) is a private fine-tune of mistralai/Voxtral-Mini-3B-2507; its card documents "
        "no training data, so every cell is U. It ran through the benchmark's harness like the public models (manifests, "
        "SHA-256 checks, asr_core path, batch 1, scorer) on the same A100, decoded by vLLM 0.31 with CUDA graphs: one "
        "piece per request, greedy, at most 500 new tokens per piece. On FLEURS test it takes "
        f"{ms(MINI_KEY):,.0f} ms per clip, {ms(MINI_KEY) / ms(OURS_KEY):.0f}x the highlighted model. Like Conformer Large "
        "and Whisper Medium it writes numbers as Bengali words, which the scorer counts as errors against references "
        "that use digits. " + loops +
        "Intervals are pointwise, not adjusted for multiple comparisons; an inconclusive result is not evidence of "
        "equivalence."
    )
    fig.text(0.06, yy - 0.03, body, size=8.0, color=INK_2, va="top", wrap=True, linespacing=1.55)
    footer(fig, 5, "Voxtral Mini bn-BD")
    pdf.savefig(fig); plt.close(fig)


def main():
    out = "asr_benchmark.pdf"
    with PdfPages(out) as pdf:
        page1(pdf); page2(pdf); page3(pdf); page4(pdf)
        if MINI_KEY in MS["models"]:
            page5(pdf)
    print("wrote", out)


if __name__ == "__main__":
    main()
