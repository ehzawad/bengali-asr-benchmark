#!/usr/bin/env python
"""Render the benchmark report (asr_benchmark.pdf): a three-page summary centred on Voxtral Mini
bn-BD, read from the committed summaries:

  * outputs_fleurs/clean_test_table.json    — five models on the FLEURS bn_in test split
  * outputs_multiset/summary_multiset.json  — the same five on six official-split subsets
  * eval_manifest.json                      — FLEURS clip durations
All five ran on one NVIDIA A100-SXM4-40GB (Colab), greedy at batch 1, one model at a time.

Every number is read from those files so the PDF cannot drift from the runs, and the claims the
text makes are asserted against them. The README carries the full tables and intervals.
"""
import json
import statistics
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

PAGE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NAVY, NAVY_SUB = "#16213e", "#c9d1e3"
MINI = "voxtral_mini_bn"
MINI_TINT, BEST_TINT, ROW_ALT = "#fdebe2", "#fff1c7", "#f3f2ee"
# One fixed colour per model, Voxtral Mini first; this order passes the categorical palette checks
# (adjacent colour-blind and normal-vision separation on the light page).
MODELS = [
    (MINI, "Voxtral Mini bn-BD", "#eb6834"),
    ("fastconformer_ctc", "ehzawad FastConformer-CTC", "#2a78d6"),
    ("hishab_conformer_large", "Conformer Large (hishab)", "#1baf7a"),
    ("whisper_medium", "Whisper Medium", "#eda100"),
    ("wav2vec2", "Wav2Vec2", "#e87ba4"),
]
NAME = {k: n for k, n, _ in MODELS}
COLOR = {k: c for k, _, c in MODELS}
PUBLIC = [k for k, _, _ in MODELS if k != MINI]
SETS = [("fleurs_test", "FLEURS", "read speech"),
        ("vaani_test", "Vaani", "spontaneous speech"),
        ("spring_r1_test", "SPRING-INX R1", "conversations"),
        ("spring_r2_test", "SPRING-INX R2", "conversations"),
        ("kathbath_test_known", "Kathbath", "read, known speakers"),
        ("kathbath_test_unknown", "Kathbath", "read, new speakers"),
        ("indicvoices_conv", "IndicVoices", "conversations")]
VERDICT = {"lower": ("✓  Better", "#0ca30c"), "higher": ("✗  Worse", "#d03b3b"),
           "inconclusive": ("≈  Too close to call", "#7c7b76")}
PT = 1 / (11.69 * 72)   # one point as a fraction of the page height

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": PAGE, "axes.facecolor": PAGE,
                     "savefig.facecolor": PAGE, "text.color": INK, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": INK_2})

CLEAN = json.loads(Path("outputs_fleurs/clean_test_table.json").read_text())
MS = json.loads(Path("outputs_multiset/summary_multiset.json").read_text())
FLEURS_TEST_S = statistics.mean(r["duration"] for r in json.loads(Path("eval_manifest.json").read_text())["rows"]
                                if r["split"] == "test")


def wer(k, s):
    return 100 * (CLEAN["models"][k]["wer"] if s == "fleurs_test" else MS["models"][k][s]["wer"])


def paired(s, b):
    return (CLEAN["paired_voxtral_mini_bn"] if s == "fleurs_test" else MS["paired"][s])[f"{MINI}_vs_{b}"]


def best_public(s):
    return min(PUBLIC, key=lambda k: wer(k, s))


def verdict(d):
    lo, hi = d["wer_ci95_pp"]
    return "lower" if hi < 0 else ("higher" if lo > 0 else "inconclusive")


def rank(k, s):
    return sorted(NAME, key=lambda m: wer(m, s)).index(k) + 1


def ms(k):
    return CLEAN["speed_default_runtime"][k]["ms_per_clip_mean"]


def signed(x, d=1):
    return f"{x:+.{d}f}".replace("-", "−")


def text_block(fig, x, y, text, width, size=9.5, color=INK_2, weight="normal", spacing=1.5):
    """Wrapped text from the top-left corner; returns the y below it."""
    lines = textwrap.wrap(text, width)
    fig.text(x, y, "\n".join(lines), size=size, color=color, weight=weight, va="top", linespacing=spacing)
    return y - len(lines) * size * spacing * PT


def bullets(fig, x, y, items, width, size=9.2, gap=0.006):
    for item in items:
        fig.text(x, y, "•", size=size, color=COLOR[MINI], va="top", weight="bold")
        y = text_block(fig, x + 0.02, y, item, width, size=size) - gap
    return y


def heading(fig, x, y, text, size=13):
    fig.patches.append(Rectangle((x, y - 0.0125), 0.008, 0.016, transform=fig.transFigure,
                                 facecolor=COLOR[MINI], edgecolor="none"))
    fig.text(x + 0.018, y, text, size=size, weight="bold", va="top")


def box(fig, x, y, w, h, color, radius=0.008, zorder=0):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                      transform=fig.transFigure, facecolor=color, edgecolor="none", zorder=zorder))


def header(fig, title, subtitle):
    fig.patches.append(Rectangle((0, 0.89), 1, 0.11, transform=fig.transFigure, facecolor=NAVY, edgecolor="none"))
    fig.patches.append(Rectangle((0, 0.884), 1, 0.006, transform=fig.transFigure, facecolor=COLOR[MINI],
                                 edgecolor="none"))
    fig.text(0.06, 0.957, title, size=24, weight="bold", color="white", va="center")
    fig.text(0.06, 0.915, subtitle, size=10, color=NAVY_SUB, va="center")


def footer(fig, page):
    fig.text(0.06, 0.03, "bengali-asr-benchmark · one Colab A100-SXM4-40GB · 13 Oct 2026 · full tables and "
             "confidence intervals in README.md", size=7, color=MUTED)
    fig.text(0.94, 0.03, str(page), size=7, color=MUTED, ha="right")


def legend(fig, y):
    """One swatch and name per model, measured so entries never overlap; wraps to a second row."""
    renderer, x = fig.canvas.get_renderer(), 0.06
    for k, n, c in MODELS:
        t = fig.text(x + 0.024, y, n, size=8.8, va="center", weight="bold" if k == MINI else "normal")
        w = t.get_window_extent(renderer).width / fig.bbox.width
        if x + 0.024 + w > 0.95:
            x, y = 0.06, y - 0.022
            t.set_position((x + 0.024, y))
        fig.patches.append(FancyBboxPatch((x, y - 0.0055), 0.016, 0.011, boxstyle="round,pad=0,rounding_size=0.002",
                                          transform=fig.transFigure, facecolor=c, edgecolor="none"))
        x += 0.024 + w + 0.032
    return y


# ---------------------------------------------------------------------------------------------
def page1(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "Voxtral Mini bn-BD", "Bengali speech-to-text: how a private fine-tune compares with four "
           "public models on seven test sets")

    v = {s: verdict(paired(s, best_public(s))) for s, _, _ in SETS}
    assert rank(MINI, "vaani_test") == 1 and rank(MINI, "spring_r1_test") == 1
    assert v["vaani_test"] == v["spring_r1_test"] == "lower" and v["fleurs_test"] == "inconclusive"
    assert all(verdict(paired(s, b)) == "higher" for s in ("kathbath_test_known", "kathbath_test_unknown")
               for b in ("hishab_conformer_large", "whisper_medium"))
    y = text_block(fig, 0.06, 0.862,
                   "Voxtral Mini is the most accurate of the five models on spontaneous and conversational speech "
                   "(Vaani, SPRING-INX R1), level with the best on FLEURS, and behind Conformer Large and Whisper "
                   "Medium on read Kathbath speech.", 92, size=10.5, color=INK)

    best_f = best_public("fleurs_test")
    whisper_x = ms("whisper_medium") / ms(MINI)
    tiles = [
        (f"{wer(MINI, 'vaani_test'):.1f}%", "word error rate on Vaani", "lowest of all five models"),
        (f"{wer(MINI, 'spring_r1_test'):.1f}%", "word error rate on SPRING-INX R1", "lowest of all five models"),
        (f"{wer(MINI, 'fleurs_test'):.1f}%", "word error rate on FLEURS",
         f"best public model: {wer(best_f, 'fleurs_test'):.1f}%, too close to call"),
        (f"{ms(MINI) / 1000:.2f} s", f"per {FLEURS_TEST_S:.0f}-second clip on an A100",
         f"{whisper_x:.0f}× faster than Whisper Medium"),
    ]
    top = y - 0.022
    for i, (big, what, note) in enumerate(tiles):
        x, ty = 0.06 + (i % 2) * 0.45, top - (i // 2) * 0.1
        box(fig, x, ty - 0.088, 0.43, 0.088, MINI_TINT)
        fig.patches.append(Rectangle((x, ty - 0.088), 0.007, 0.088, transform=fig.transFigure,
                                     facecolor=COLOR[MINI], edgecolor="none"))
        fig.text(x + 0.03, ty - 0.012, big, size=25, weight="bold", va="top")
        fig.text(x + 0.03, ty - 0.055, what, size=9.2, color=INK, va="top")
        fig.text(x + 0.03, ty - 0.071, note, size=8.4, color=INK_2, va="top")

    y = top - 0.215
    heading(fig, 0.06, y, "Against the best public model on each test set")
    y -= 0.04
    cols = {"set": 0.06, "mini": 0.385, "bestw": 0.475, "best": 0.49, "gap": 0.785, "pill": 0.8}
    for key, label, ha in (("set", "Test set", "left"), ("mini", "Voxtral Mini", "right"),
                           ("best", "Best public model", "left"), ("gap", "Gap", "right"),
                           ("pill", "Verdict", "left")):
        x = 0.418 if key == "best" else cols[key]
        fig.text(x, y, label, size=8.2, color=MUTED, weight="bold", ha=ha, va="center")
    y -= 0.012
    row_h = 0.043
    for i, (s, label, kind) in enumerate(SETS):
        b = best_public(s)
        d = paired(s, b)
        cy = y - row_h * i - row_h / 2
        if i % 2 == 0:
            box(fig, 0.05, cy - row_h / 2, 0.92, row_h, ROW_ALT, radius=0.004)
        fig.text(cols["set"], cy + 0.006, label, size=9.6, weight="bold", va="center")
        fig.text(cols["set"], cy - 0.009, kind, size=7.6, color=MUTED, va="center")
        fig.text(cols["mini"], cy, f"{wer(MINI, s):.1f}%", size=10.5, weight="bold", ha="right", va="center")
        fig.text(cols["best"], cy, NAME[b], size=9, color=INK_2, va="center")
        fig.text(cols["bestw"], cy, f"{wer(b, s):.1f}%", size=10.5, color=INK_2, ha="right", va="center")
        fig.text(cols["gap"], cy, signed(d["wer_delta_pp"]), size=9.6, ha="right", va="center", weight="bold")
        word, c = VERDICT[verdict(d)]
        box(fig, cols["pill"], cy - 0.0105, 0.158, 0.021, c, radius=0.0105, zorder=1)
        fig.text(cols["pill"] + 0.079, cy, word, size=8.4, color="white", weight="bold", ha="center", va="center",
                 zorder=2)
    y -= row_h * len(SETS) + 0.016
    text_block(fig, 0.06, y,
               "Word error rate (WER): the share of words transcribed wrongly, lower is better. Gap: Voxtral Mini "
               "minus the best public model, in percentage points. Better or Worse means the gap holds up in a "
               "paired statistical test on the same recordings (95% bootstrap); Too close to call means it does not.",
               118, size=7.6, color=MUTED, spacing=1.45)
    footer(fig, 1)
    pdf.savefig(fig); plt.close(fig)


def page2(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "Every model, every test set", "Word error rate, lower is better · Voxtral Mini in orange")
    top = legend(fig, 0.857) - 0.02

    ax = fig.add_axes([0.27, 0.445, 0.67, top - 0.445])
    n, group = len(MODELS), len(MODELS) + 1.4
    centers = []
    for i, (s, label, kind) in enumerate(SETS):
        base = -i * group
        centers.append(base - (n - 1) / 2)
        for j, (k, _, c) in enumerate(MODELS):
            yy = base - j
            ax.barh(yy, wer(k, s), height=0.82, color=c, alpha=1 if k == MINI else 0.5,
                    edgecolor=PAGE, linewidth=1.2)
            if k == MINI:
                ax.text(wer(k, s) + 0.6, yy, f"{wer(k, s):.1f}%", va="center", size=8.2, weight="bold", color=INK)
    ax.set_yticks(centers)
    ax.set_yticklabels([f"{label}\n{kind}" for _, label, kind in SETS], size=8.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 50)
    ax.set_xticks(range(0, 51, 10))
    ax.set_xticklabels([f"{t}%" for t in range(0, 51, 10)], size=8)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_ylim(-(len(SETS) - 1) * group - n + 0.3, 0.7)

    y = 0.4
    heading(fig, 0.06, y, "The numbers")
    y -= 0.042
    xs = [0.385 + 0.0825 * i for i in range(len(SETS))]
    heads = ["FLEURS", "Vaani", "SPRING\nR1", "SPRING\nR2", "Kathbath\nknown", "Kathbath\nnew", "Indic\nVoices"]
    for x, h in zip(xs, heads):
        fig.text(x, y, h, size=7.6, color=MUTED, weight="bold", ha="right", va="center", linespacing=1.3)
    y -= 0.022
    best = {s: min(NAME, key=lambda k: wer(k, s)) for s, _, _ in SETS}
    for i, (k, n_, c) in enumerate(MODELS):
        cy = y - 0.03 * i - 0.015
        if k == MINI:
            box(fig, 0.05, cy - 0.015, 0.91, 0.03, MINI_TINT, radius=0.004)
        fig.patches.append(Rectangle((0.06, cy - 0.005), 0.012, 0.01, transform=fig.transFigure, facecolor=c,
                                     edgecolor="none"))
        fig.text(0.08, cy, n_, size=8.8, weight="bold" if k == MINI else "normal", va="center")
        for x, (s, _, _) in zip(xs, SETS):
            if best[s] == k:
                box(fig, x - 0.062, cy - 0.011, 0.066, 0.022, BEST_TINT, radius=0.004, zorder=1)
            fig.text(x, cy, f"{wer(k, s):.1f}", size=9, ha="right", va="center", zorder=2,
                     weight="bold" if (k == MINI or best[s] == k) else "normal")
    y -= 0.03 * len(MODELS) + 0.02
    box(fig, 0.06, y - 0.009, 0.03, 0.013, BEST_TINT, radius=0.003)
    fig.text(0.1, y - 0.0025, "lowest word error rate on that set", size=7.8, color=MUTED, va="center")
    y -= 0.03
    assert best["indicvoices_conv"] == "fastconformer_ctc"
    bullets(fig, 0.06, y, [
        "Kathbath and IndicVoices are not clean head-to-heads: ehzawad FastConformer-CTC trained on other splits "
        "of both (hence its lead on IndicVoices), and Conformer Large on Kathbath.",
        "Voxtral Mini's training data is not documented, so overlap with any of these sets cannot be ruled out.",
    ], 112, size=8.4)
    footer(fig, 2)
    pdf.savefig(fig); plt.close(fig)


def page3(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "Speed, method and caveats", f"Average time to transcribe one FLEURS test clip "
           f"({FLEURS_TEST_S:.1f} s of audio), one clip at a time on one A100")

    ax = fig.add_axes([0.30, 0.64, 0.6, 0.21])
    keys = [k for k, _, _ in MODELS]
    for j, k in enumerate(keys):
        ax.barh(-j, ms(k), height=0.62, color=COLOR[k], alpha=1 if k == MINI else 0.5)
        ax.text(ms(k) + 25, -j, f"{ms(k):,.0f} ms", va="center", size=8.8,
                weight="bold" if k == MINI else "normal", color=INK)
    ax.set_yticks([-j for j in range(len(keys))])
    ax.set_yticklabels([NAME[k] for k in keys], size=9)
    for t in ax.get_yticklabels():
        if t.get_text() == NAME[MINI]:
            t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(ms(k) for k in keys) * 1.18)
    ax.set_xlabel("milliseconds per clip (shorter is faster)", size=8.2, color=MUTED)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)

    nemo = statistics.mean(ms(k) for k in ("fastconformer_ctc", "hishab_conformer_large"))
    y = text_block(fig, 0.06, 0.595,
                   f"Voxtral Mini takes {ms(MINI) / 1000:.2f} s for a {FLEURS_TEST_S:.0f}-second clip, about "
                   f"{FLEURS_TEST_S * 1000 / ms(MINI):.0f}× faster than real time. That is "
                   f"{ms('whisper_medium') / ms(MINI):.0f}× faster than Whisper Medium and {ms(MINI) / nemo:.0f}× "
                   "slower than the two NeMo CTC models (FastConformer-CTC and Conformer Large). The times cover each "
                   "model's whole pipeline, audio loading included, on vLLM for Voxtral Mini and PyTorch for the "
                   "others, so they compare setups, not just the models.", 100, size=9.4, color=INK)

    y -= 0.03
    heading(fig, 0.06, y, "How it was measured")
    y = bullets(fig, 0.06, y - 0.035, [
        "One Colab A100 (40 GB) on 13 Oct 2026, one model on the GPU at a time.",
        "The same audio for every model: 16 kHz, cut at pauses into pieces of at most 25 s.",
        "Plain decoding: greedy, one clip at a time, no external language model.",
        f"FLEURS: the {CLEAN['models'][MINI]['n']} test recordings. Other sets: "
        f"{MS['sets']['vaani_test']['n']:,} recordings each, sampled from the official splits.",
        "One scorer for everyone: punctuation removed and Bengali text Unicode-normalised before counting errors.",
    ], 108)

    r2 = MS["models"][MINI]["spring_r2_test"]
    capped = r2["errors"] - r2["without_capped_clips"]["errors"]
    assert not any(MS["models"][k][s]["hyps_with_latin"] for k in NAME for s in ("spring_r1_test", "spring_r2_test"))
    y -= 0.02
    heading(fig, 0.06, y, "Keep in mind")
    bullets(fig, 0.06, y - 0.035, [
        "These are Indian Bengali test sets; none is a dedicated test of Bangladeshi speech.",
        "About half of SPRING-INX's reference sentences contain English words written in Latin letters, which no "
        "model produces, so SPRING error rates are inflated for every model alike.",
        f"On SPRING-INX R2, Voxtral Mini got stuck repeating one word on {r2['clips_at_token_cap']} recordings; those "
        f"alone account for {capped:,} of its {r2['errors']:,} word errors there, and they stay in its score.",
        "Like Conformer Large and Whisper Medium, Voxtral Mini writes numbers as Bengali words, which counts as an "
        "error wherever the reference uses digits.",
    ], 108)
    footer(fig, 3)
    pdf.savefig(fig); plt.close(fig)


def main():
    out = "asr_benchmark.pdf"
    with PdfPages(out) as pdf:
        page1(pdf); page2(pdf); page3(pdf)
    print("wrote", out)


if __name__ == "__main__":
    main()
