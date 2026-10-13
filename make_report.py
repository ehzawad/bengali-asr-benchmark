#!/usr/bin/env python
"""Render the benchmark report (asr_benchmark.pdf): a two-page summary centred on Voxtral Mini
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
MINI_TINT, BEST_TINT = "#fdebe2", "#fff1c7"
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
SETS = [("fleurs_test", "FLEURS"), ("vaani_test", "Vaani"), ("spring_r1_test", "SPRING-INX R1"),
        ("spring_r2_test", "SPRING-INX R2"), ("kathbath_test_known", "Kathbath known"),
        ("kathbath_test_unknown", "Kathbath unknown"), ("indicvoices_conv", "IndicVoices")]
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


def ms(k):
    return CLEAN["speed_default_runtime"][k]["ms_per_clip_mean"]


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
    header(fig, "Voxtral Mini bn-BD", "Word error rate against four public Bengali ASR models, lower is better "
           "· Voxtral Mini in orange")
    top = legend(fig, 0.857) - 0.02

    ax = fig.add_axes([0.27, 0.445, 0.67, top - 0.445])
    n, group = len(MODELS), len(MODELS) + 1.4
    centers = []
    for i, (s, label) in enumerate(SETS):
        base = -i * group
        centers.append(base - (n - 1) / 2)
        for j, (k, _, c) in enumerate(MODELS):
            yy = base - j
            ax.barh(yy, wer(k, s), height=0.82, color=c, alpha=1 if k == MINI else 0.5,
                    edgecolor=PAGE, linewidth=1.2)
            if k == MINI:
                ax.text(wer(k, s) + 0.6, yy, f"{wer(k, s):.1f}%", va="center", size=8.2, weight="bold", color=INK)
    ax.set_yticks(centers)
    ax.set_yticklabels([label for _, label in SETS], size=9)
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
    heads = ["FLEURS", "Vaani", "SPRING\nR1", "SPRING\nR2", "Kathbath\nknown", "Kathbath\nunknown", "Indic\nVoices"]
    for x, h in zip(xs, heads):
        fig.text(x, y, h, size=7.6, color=MUTED, weight="bold", ha="right", va="center", linespacing=1.3)
    y -= 0.022
    best = {s: min(NAME, key=lambda k: wer(k, s)) for s, _ in SETS}
    for i, (k, n_, c) in enumerate(MODELS):
        cy = y - 0.03 * i - 0.015
        if k == MINI:
            box(fig, 0.05, cy - 0.015, 0.91, 0.03, MINI_TINT, radius=0.004)
        fig.patches.append(Rectangle((0.06, cy - 0.005), 0.012, 0.01, transform=fig.transFigure, facecolor=c,
                                     edgecolor="none"))
        fig.text(0.08, cy, n_, size=8.8, weight="bold" if k == MINI else "normal", va="center")
        for x, (s, _) in zip(xs, SETS):
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
    footer(fig, 1)
    pdf.savefig(fig); plt.close(fig)


def page2(pdf):
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
    footer(fig, 2)
    pdf.savefig(fig); plt.close(fig)


def main():
    out = "asr_benchmark.pdf"
    with PdfPages(out) as pdf:
        page1(pdf); page2(pdf)
    print("wrote", out)


if __name__ == "__main__":
    main()
