#!/usr/bin/env python
"""Render the benchmark report (asr_benchmark.pdf): a summary centred on Voxtral Mini bn-BD (pages 1-3, full
sets) and the three-model comparison on one GPU (page 4, cleaned subsets), read from the committed summaries:

  * outputs_fleurs/clean_test_table.json    — every model on the FLEURS bn_in test split
  * outputs_multiset/summary_multiset.json  — the same models on the ten subsets
  * outputs_compare3/summary.json           — Rumble-2-Flash, Voxtral Mini and Qwen3-ASR bn-BD on one A100 and one B200
  * eval_manifest.json                      — FLEURS clip durations
Pages 1-3: one NVIDIA A100-SXM4-40GB (Colab), greedy at batch 1, one model at a time; Whistle bn-BD on an A100 80 GB
and a CPU, Rumble-2-Flash bn-BD on another A100-SXM4-40GB. Granite Speech bn-BD is in the README only.

Every number is read from those files so the PDF cannot drift from the runs, and the claims the
text makes are asserted against them. The README carries the full tables and intervals.
"""
import json
import math
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
# One fixed colour per model (Voxtral Mini first), in an order that passes the categorical palette checks
# (adjacent colour-blind and normal-vision separation on the light page); the ninth is a neutral grey.
ALL_MODELS = [
    (MINI, "Voxtral Mini bn-BD", "#eb6834", "private"),
    ("fastconformer_hybrid_bn", "FastConformer hybrid bn-BD", "#1baf7a", "private"),
    ("canary_1b_v2_bn", "Canary 1B v2 bn-BD", "#eda100", "private"),
    ("qwen3_asr_bn", "Qwen3-ASR 1.7B bn-BD", "#e87ba4", "private"),
    ("whisper_turbo_bn", "Whisper turbo bn-BD", "#008300", "private"),
    ("whistle_bn_bd_a100", "Whistle bn-BD (55M, A100)", "#9a5b13", "private"),
    ("whistle_bn_bd", "Whistle bn-BD (55M, CPU)", "#9a5b13", "private"),
    ("rumble_bn", "Rumble-2-Flash bn-BD", "#0a7fa0", "private"),   # passes the checks against its neighbours here and on page 4
    ("fastconformer_ctc", "ehzawad FastConformer-CTC", "#6250d6", "public"),
    ("hishab_conformer_large", "Conformer Large (hishab)", "#e34948", "public"),
    ("whisper_medium", "Whisper Medium", "#2a78d6", "public"),
    ("wav2vec2", "Wav2Vec2", "#8a8880", "public"),
]
NAME = {k: n for k, n, _, _ in ALL_MODELS}
COLOR = {k: c for k, _, c, _ in ALL_MODELS}
GROUP = {k: g for k, _, _, g in ALL_MODELS}
# (key, chart label, two-line table header)
BD_SETS = [("subakko_test", "SUBAK.KO", "SUBAK.KO"), ("banspeech", "BanSpeech", "Ban-\nSpeech"),
           ("cv17_bn_test", "Common Voice 17", "Common\nVoice 17"), ("ben10_valid", "Ben-10 dialects", "Ben-10\ndialects")]
IN_SETS = [("fleurs_test", "FLEURS", "FLEURS"), ("vaani_test", "Vaani", "Vaani"),
           ("spring_r1_test", "SPRING-INX R1", "SPRING\nR1"), ("spring_r2_test", "SPRING-INX R2", "SPRING\nR2"),
           ("kathbath_test_known", "Kathbath known", "Kathbath\nknown"),
           ("kathbath_test_unknown", "Kathbath unknown", "Kathbath\nunknown"), ("indicvoices_conv", "IndicVoices", "Indic\nVoices")]
PT = 1 / (11.69 * 72)   # one point as a fraction of the page height

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": PAGE, "axes.facecolor": PAGE,
                     "savefig.facecolor": PAGE, "text.color": INK, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": INK_2})

CLEAN = json.loads(Path("outputs_fleurs/clean_test_table.json").read_text())
MS = json.loads(Path("outputs_multiset/summary_multiset.json").read_text())
MODELS = [(k, n, c) for k, n, c, _ in ALL_MODELS
          if k in CLEAN["models"] and all(s in MS["models"].get(k, {}) for s in MS["sets"])]
KEYS = [k for k, _, _ in MODELS]
PUBLIC = [k for k in KEYS if GROUP[k] == "public"]
FLEURS_TEST_S = statistics.mean(r["duration"] for r in json.loads(Path("eval_manifest.json").read_text())["rows"]
                                if r["split"] == "test")


VLLM = {MINI, "whisper_medium", "whisper_turbo_bn", "qwen3_asr_bn", "rumble_bn"}   # the autoregressive decoders
CPU = {"whistle_bn_bd"}   # run on CPU with its own engine, so its time is not compared with the A100 rows
WHISTLE = {"whistle_bn_bd", "whistle_bn_bd_a100"}   # the same checkpoint, A100 (JAX, greedy) and CPU (needle engine)
HATCH = "///"
GPU_WHISTLE = (json.loads(Path("outputs_fleurs/whistle_bn_bd_a100/run_meta.json").read_text())["device"]
               .replace("GPU: NVIDIA ", "an ").replace("-SXM4-80GB", " (80 GB)") if Path("outputs_fleurs/whistle_bn_bd_a100/run_meta.json").exists() else "")


def andjoin(xs):
    xs = list(xs)
    return "none" if not xs else xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


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


def footer(fig, page, text=None):
    fig.text(0.06, 0.03, text or "bengali-asr-benchmark · Colab A100-SXM4-40GB, 13 Oct 2026 · Whistle bn-BD: A100 80 GB "
             "and CPU, 21-22 Oct · full tables in README.md", size=7, color=MUTED)
    if text is None and "rumble_bn" in KEYS:
        fig.text(0.06, 0.017, "Rumble-2-Flash bn-BD: another A100-SXM4-40GB, 20 Nov 2026", size=7, color=MUTED)
    fig.text(0.94, 0.03, str(page), size=7, color=MUTED, ha="right")


def legend(fig, y):
    """Swatches and names, the private fine-tunes on one row and the public models on the next; entries are
    measured so they never overlap and wrap if a row is full."""
    renderer = fig.canvas.get_renderer()
    for group, title in (("private", "ehzawad bn-BD fine-tunes (private)"), ("public", "Public models")):
        fig.text(0.06, y, title, size=7.6, color=MUTED, weight="bold", va="center")
        x = 0.06
        y -= 0.018
        for k, n, c in [m for m in MODELS if GROUP[m[0]] == group]:
            t = fig.text(x + 0.024, y, n, size=8.6, va="center", weight="bold" if k == MINI else "normal")
            w = t.get_window_extent(renderer).width / fig.bbox.width
            if x + 0.024 + w > 0.95:
                x, y = 0.06, y - 0.018
                t.set_position((x + 0.024, y))
            fig.patches.append(FancyBboxPatch((x, y - 0.0055), 0.016, 0.011, boxstyle="round,pad=0,rounding_size=0.002",
                                              transform=fig.transFigure, facecolor=c,
                                              edgecolor=PAGE if k in CPU else "none", hatch=HATCH if k in CPU else None,
                                              linewidth=0))
            x += 0.024 + w + 0.03
        y -= 0.022
    return y + 0.022


# ---------------------------------------------------------------------------------------------
def wer_page(pdf, page, title, subtitle, sets, notes):
    """Bars for every model on every set (Voxtral Mini labelled), then the numbers, then notes."""
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, title, subtitle)
    top = legend(fig, 0.857) - 0.02
    height = (0.06 if len(MODELS) <= 6 else 0.041 if len(MODELS) <= 11 else 0.036) * len(sets) - 0.01
    ax = fig.add_axes([0.27, top - height, 0.67, height])
    n, group = len(MODELS), len(MODELS) + 1.4
    xmax = 10 * math.ceil(max(wer(k, s) for k in KEYS for s, _, _ in sets) / 10 + 0.05)
    centers = []
    for i, (s, label, _) in enumerate(sets):
        base = -i * group
        centers.append(base - (n - 1) / 2)
        for j, (k, _, c) in enumerate(MODELS):
            yy = base - j
            ax.barh(yy, wer(k, s), height=0.82, color=c, alpha=1 if k == MINI else 0.5,
                    edgecolor=PAGE, linewidth=1.2, hatch=HATCH if k in CPU else None)
            if k == MINI:
                ax.text(wer(k, s) + xmax * 0.012, yy, f"{wer(k, s):.1f}%", va="center", size=8.2, weight="bold",
                        color=INK)
    ax.set_yticks(centers)
    ax.set_yticklabels([label for _, label, _ in sets], size=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax)
    ticks = range(0, xmax + 1, 10 if xmax <= 60 else 20)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f"{t}%" for t in ticks], size=8)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_ylim(-(len(sets) - 1) * group - n + 0.3, 0.7)

    y = top - height - 0.045
    heading(fig, 0.06, y, "The numbers")
    y -= 0.042
    xs = [0.385 + 0.0825 * i for i in range(len(sets))]
    for x, (_, _, head) in zip(xs, sets):
        fig.text(x, y, head, size=7.6, color=MUTED, weight="bold", ha="right", va="center", linespacing=1.3)
    y -= 0.022
    best = {s: min((m for m, _, _ in MODELS), key=lambda k: wer(k, s)) for s, _, _ in sets}
    ROW = 0.03 if len(MODELS) <= 6 else 0.021
    for i, (k, n_, c) in enumerate(MODELS):
        cy = y - ROW * i - ROW / 2
        if k == MINI:
            box(fig, 0.05, cy - ROW / 2, 0.36 + 0.0825 * len(sets), ROW, MINI_TINT, radius=0.004)
        fig.patches.append(Rectangle((0.06, cy - 0.005), 0.012, 0.01, transform=fig.transFigure, facecolor=c,
                                     edgecolor=PAGE if k in CPU else "none", hatch=HATCH if k in CPU else None,
                                     linewidth=0))
        fig.text(0.08, cy, n_, size=8.8, weight="bold" if k == MINI else "normal", va="center")
        for x, (s, _, _) in zip(xs, sets):
            if best[s] == k:
                box(fig, x - 0.062, cy - ROW * 0.37, 0.066, ROW * 0.74, BEST_TINT, radius=0.004, zorder=1)
            fig.text(x, cy, f"{wer(k, s):.1f}", size=9, ha="right", va="center", zorder=2,
                     weight="bold" if (k == MINI or best[s] == k) else "normal")
    y -= ROW * len(MODELS) + 0.02
    box(fig, 0.06, y - 0.009, 0.03, 0.013, BEST_TINT, radius=0.003)
    fig.text(0.1, y - 0.0025, "lowest word error rate on that set", size=7.8, color=MUTED, va="center")
    bullets(fig, 0.06, y - 0.03, notes, 112, size=8.4)
    footer(fig, page)
    pdf.savefig(fig); plt.close(fig)
    return best


def BD_NOTES(best):
    """Notes under the Bangladeshi page, read from the summaries."""
    led = [label for s, label, _ in BD_SETS if best[s] == MINI]
    sub_ = MS["models"][MINI]["subakko_test"]
    assert all(GROUP[best[s]] in ("private", "public") for s, _, _ in BD_SETS)
    return [
        f"Voxtral Mini has the lowest word error rate on {andjoin(led)}." if led else
        "Voxtral Mini does not have the lowest word error rate on any of these sets.",
        f"{sub_['digit_refs']} of SUBAK.KO's {sub_['n']:,} references write numbers as digits, which models that write "
        f"numbers as words cannot match; on the other recordings Voxtral Mini reads {100 * sub_['no_digit_ref']['wer']:.1f}%.",
        "SUBAK.KO and Common Voice are not clean head-to-heads: ehzawad FastConformer-CTC and Conformer Large trained on "
        "both corpora, and FastConformer-CTC's Common Voice data may include these very clips.",
        "Ben-10 is regional dialect speech written in dialect spelling; its test split is private, so this is its "
        "validation split. Common Voice records no speaker region. The private models' training data is undocumented.",
    ]


def page1(pdf):
    best = {s: min(KEYS, key=lambda k: wer(k, s)) for s, _, _ in BD_SETS}
    wer_page(pdf, 1, "Voxtral Mini bn-BD", "Bangladeshi and crowd-sourced test sets · word error rate, lower is "
             "better · Voxtral Mini in orange", BD_SETS, BD_NOTES(best))


def page2(pdf):
    best = {s: min(KEYS, key=lambda k: wer(k, s)) for s, _, _ in IN_SETS}
    wer_page(pdf, 2, "Indian Bengali test sets", "Word error rate, lower is better · Voxtral Mini in orange", IN_SETS, [
        "Kathbath and IndicVoices are not clean head-to-heads: ehzawad FastConformer-CTC trained on other splits "
        "of both, and Conformer Large on Kathbath.",
        "The private models' training data is undocumented, so overlap with any of these sets cannot be ruled out.",
    ])


def page3(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "Speed, method and caveats", f"Time per FLEURS test clip "
           f"({FLEURS_TEST_S:.1f} s of audio), one at a time, on one A100 (Whistle bn-BD also on CPU)")

    ax = fig.add_axes([0.30, 0.665, 0.6, 0.19])
    keys = [k for k, _, _ in MODELS]
    for j, k in enumerate(keys):
        ax.barh(-j, ms(k), height=0.62, color=COLOR[k], alpha=1 if k == MINI else 0.5,
                hatch=HATCH if k in CPU else None, edgecolor=PAGE if k in CPU else None)
        ax.text(ms(k) + 25, -j, f"{ms(k):,.0f} ms" + (" on CPU" if k in CPU else ""), va="center", size=8.8,
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

    faster = [NAME[k] for k in KEYS if k not in CPU and ms(k) < ms(MINI)]
    slower = [NAME[k] for k in KEYS if k not in CPU and ms(k) > ms(MINI)]
    vllm = [NAME[k] for k in KEYS if k in VLLM]
    y = text_block(fig, 0.06, 0.62,
                   f"Voxtral Mini takes {ms(MINI) / 1000:.2f} s for a {FLEURS_TEST_S:.0f}-second clip, about "
                   f"{FLEURS_TEST_S * 1000 / ms(MINI):.0f}× faster than real time. "
                   + (f"Faster: {andjoin(faster)}. " if faster else "It is the fastest model here. ")
                   + (f"Slower: {andjoin(slower)}. " if slower else "It is the slowest model here. ")
                   + "The times cover each model's whole pipeline, audio loading included: "
                   f"vLLM for {andjoin(vllm)}, NeMo or PyTorch for the rest, so they compare setups, not just "
                   "the models."
                   + (" Whistle bn-BD appears twice: on an A100 in JAX (greedy), comparable with the other bars, and "
                      "hatched on one CPU core in its needle engine, the on-device cost."
                      if CPU & set(KEYS) else ""), 100, size=9.4, color=INK)

    y -= 0.022
    heading(fig, 0.06, y, "How it was measured")
    y = bullets(fig, 0.06, y - 0.032, [
        f"One Colab A100 (40 GB) on 13 Oct 2026, one model on the GPU at a time; Whistle bn-BD on {GPU_WHISTLE} "
        "and a CPU on 21-22 Oct" + ("; Rumble-2-Flash bn-BD on another A100 (40 GB) on 20 Nov 2026, vLLM 0.31."
                                    if "rumble_bn" in KEYS else "."),
        "The same audio for every model: 16 kHz, cut at pauses into pieces of at most 25 s.",
        "Plain decoding: greedy, one clip at a time, no external language model (Whistle bn-BD on CPU: its engine's "
        "default decoding).",
        f"FLEURS: the {CLEAN['models'][MINI]['n']} test recordings. Other sets: "
        f"{MS['sets']['vaani_test']['n']:,} recordings each, a seeded sample of each set's official split.",
        "One scorer for everyone: punctuation removed and Bengali text Unicode-normalised before counting errors.",
    ], 108)

    r2 = MS["models"][MINI]["spring_r2_test"]
    capped = r2["errors"] - r2["without_capped_clips"]["errors"]
    SPRING = ("spring_r1_test", "spring_r2_test")
    assert not any(MS["models"][k][s]["hyps_with_latin"] for k in KEYS if k not in WHISTLE for s in SPRING)
    latin = {k: [MS["models"][k][s]["hyps_with_latin"] for s in SPRING] for k in KEYS if k in WHISTLE}
    y -= 0.012
    heading(fig, 0.06, y, "Keep in mind")
    bullets(fig, 0.06, y - 0.032, [
        "Seven of the eleven sets are Indian Bengali; SUBAK.KO, BanSpeech and Ben-10 are Bangladeshi.",
        "About half of SPRING-INX's reference sentences contain English words written in Latin letters, which no "
        "model reproduces, so SPRING error rates are inflated for every model alike."
        + (" Whistle bn-BD leaks English fragments into "
           + " and ".join(f"{r1}/{r2_} ({'CPU' if k in CPU else 'A100'})" for k, (r1, r2_) in latin.items())
           + " of 1,000 R1/R2 transcripts." if latin else ""),
        f"On SPRING-INX R2, Voxtral Mini got stuck repeating one word on {r2['clips_at_token_cap']} recordings; those "
        f"alone account for {capped:,} of its {r2['errors']:,} word errors there, and they stay in its score.",
        "Like Conformer Large and Whisper Medium, Voxtral Mini writes numbers as Bengali words, which counts as an "
        "error wherever the reference uses digits.",
    ] + ([f"Whisper's tokenizer spends about 13 tokens per Bengali word, so Whisper turbo's longest pieces were cut "
          f"at its 448-token limit: {MS['models']['whisper_turbo_bn']['ben10_valid']['clips_at_token_cap']:,} of "
          "Ben-10's 1,000 recordings."] if "whisper_turbo_bn" in KEYS else []), 112, size=8.8, gap=0.004)
    footer(fig, 3)
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Page 4: Rumble-2-Flash, Voxtral Mini and Qwen3-ASR bn-BD re-decoded together on one A100 and one B200
C3_PATH = Path("outputs_compare3/summary.json")
C3_ORDER = [MINI, "rumble_bn", "qwen3_asr_bn"]   # Rumble in the middle: each adjacent pair of bars passes the checks
C3_SETS = [("fleurs_test", "FLEURS"), ("subakko_test", "SUBAK.KO"), ("banspeech", "BanSpeech"),
           ("cv17_bn_test", "Common Voice 17"), ("ben10_valid", "Ben-10 dialects"), ("vaani_test", "Vaani"),
           ("spring_r1_test", "SPRING-INX R1"), ("spring_r2_test", "SPRING-INX R2"),
           ("kathbath_test_known", "Kathbath known"), ("kathbath_test_unknown", "Kathbath unknown"),
           ("indicvoices_conv", "IndicVoices")]


def c3_bars(ax, rows, xmax, ticks, yticklabels=True):
    """rows: [(label, {model: value}, bold label)] top first; three bars per row in C3_ORDER, with a 2 px gap between
    bars; the numbers are in the table below the charts."""
    n, group = len(C3_ORDER), len(C3_ORDER) + 1.3
    centers = []
    for i, (label, vals, _) in enumerate(rows):
        base = -i * group
        centers.append(base - (n - 1) / 2)
        for j, m in enumerate(C3_ORDER):
            ax.barh(base - j, vals[m], height=0.86, color=COLOR[m], edgecolor=PAGE, linewidth=1.0)
    ax.set_yticks(centers)
    ax.set_yticklabels([r[0] for r in rows] if yticklabels else [""] * len(rows), size=8.2)
    if yticklabels:
        for t, r in zip(ax.get_yticklabels(), rows):
            if r[2]:
                t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax)
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", labelsize=7.6)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_ylim(-(len(rows) - 1) * group - n + 0.4, 0.6)


def page4(pdf, c3):
    runs, avg, tok = c3["runs"], c3["average"], c3["output_tokens_fleurs_test"]
    gpus = {g: c3["gpus"][g]["device"].replace("NVIDIA ", "") for g in ("a100", "b200")}
    short = {MINI: "Voxtral Mini", "rumble_bn": "Rumble", "qwen3_asr_bn": "Qwen3-ASR"}
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "Three models, one GPU", f"Every set decoded by all three on one {gpus['a100']} and on one "
           f"{gpus['b200']} · cleaned subsets")
    x, y = 0.06, 0.862   # legend: one row, in the bar order
    renderer = fig.canvas.get_renderer()
    for m in C3_ORDER:
        fig.patches.append(FancyBboxPatch((x, y - 0.0055), 0.016, 0.011, boxstyle="round,pad=0,rounding_size=0.002",
                                          transform=fig.transFigure, facecolor=COLOR[m], edgecolor="none"))
        t = fig.text(x + 0.024, y, NAME[m], size=8.6, va="center")
        x += 0.024 + t.get_window_extent(renderer).width / fig.bbox.width + 0.035

    # word error rate, A100 runs
    w = lambda g, s, m: 100 * runs[g][s][m]["wer"]   # noqa: E731
    heading(fig, 0.06, 0.835, f"Word error rate, {gpus['a100']} runs (lower is better)", size=11.5)
    rows = [("Average, 11 sets", {m: 100 * avg["a100"][m]["wer"] for m in C3_ORDER}, True)] + \
           [(label, {m: w("a100", s, m) for m in C3_ORDER}, False) for s, label in C3_SETS]
    xmax = 10 * math.ceil(max(v for _, vals, _ in rows for v in vals.values()) / 10 + 0.05)
    ax = fig.add_axes([0.22, 0.60, 0.72, 0.21])
    c3_bars(ax, rows, xmax, list(range(0, xmax + 1, 10)))
    ax.set_xticklabels([f"{t}%" for t in range(0, xmax + 1, 10)])
    gap = max(abs(w("b200", s, m) - w("a100", s, m)) for s, _ in C3_SETS for m in C3_ORDER)
    worst = max(((s, m) for s, _ in C3_SETS for m in C3_ORDER), key=lambda k: abs(w("b200", *k) - w("a100", *k)))
    text_block(fig, 0.06, 0.572, f"The {gpus['b200']} runs read within {gap:.1f} points of these on every set (the "
               f"largest gap: {short[worst[1]]} on {dict(C3_SETS)[worst[0]]}).", 130, size=7.8)

    # time per clip, both GPUs on one scale
    heading(fig, 0.06, 0.54, "Time per clip, mean over each set (shorter is faster)", size=11.5)
    ms_ = lambda g, s, m: runs[g][s][m]["ms_per_clip_mean"]   # noqa: E731
    top = max(ms_(g, s, m) for g in ("a100", "b200") for s, _ in C3_SETS for m in C3_ORDER)
    step = 400 if top > 800 else 200 if top > 400 else 100
    xmax = step * math.ceil(top * 1.05 / step)
    for i, g in enumerate(("a100", "b200")):
        ax = fig.add_axes([0.22 + i * 0.385, 0.33, 0.33, 0.17])
        rows = [("Average, 11 sets", {m: avg[g][m]["ms_per_clip_mean"] for m in C3_ORDER}, True)] + \
               [(label, {m: ms_(g, s, m) for m in C3_ORDER}, False) for s, label in C3_SETS]
        c3_bars(ax, rows, xmax, list(range(0, xmax + 1, step)), yticklabels=(i == 0))
        ax.set_title(gpus[g], size=9, weight="bold", loc="left", color=INK)
        ax.set_xlabel("ms per clip", size=7.6, color=MUTED)

    # the averages and the token arithmetic
    y = 0.262
    heading(fig, 0.06, y, "Averages over the eleven sets", size=11.5)
    y -= 0.034
    cols = [("WER\nA100", lambda m: f"{100 * avg['a100'][m]['wer']:.2f}%"),
            ("WER\nB200", lambda m: f"{100 * avg['b200'][m]['wer']:.2f}%"),
            ("ms/clip\nA100", lambda m: f"{avg['a100'][m]['ms_per_clip_mean']:.0f}"),
            ("ms/clip\nB200", lambda m: f"{avg['b200'][m]['ms_per_clip_mean']:.0f}"),
            ("tokens/\nclip", lambda m: f"{tok['a100'][m]['tokens_per_clip_mean']:.0f}"),
            ("ms/token\nA100", lambda m: f"{tok['a100'][m]['fit_ms_per_token']:.2f}"),
            ("ms/token\nB200", lambda m: f"{tok['b200'][m]['fit_ms_per_token']:.2f}")]
    xs = [0.44 + 0.083 * i for i in range(len(cols))]
    for x_, (head, _) in zip(xs, cols):
        fig.text(x_, y, head, size=7.4, color=MUTED, weight="bold", ha="right", va="center", linespacing=1.25)
    y -= 0.026
    for m in C3_ORDER:
        fig.patches.append(Rectangle((0.06, y - 0.005), 0.012, 0.01, transform=fig.transFigure, facecolor=COLOR[m],
                                     edgecolor="none"))
        fig.text(0.08, y, NAME[m], size=8.6, va="center")
        for x_, (_, f) in zip(xs, cols):
            fig.text(x_, y, f(m), size=8.6, ha="right", va="center")
        y -= 0.02
    r, v = "rumble_bn", MINI
    fast = lambda g: min(C3_ORDER, key=lambda m: avg[g][m]["ms_per_clip_mean"])   # noqa: E731
    assert fast("a100") == r and fast("b200") == v, "the text below assumes this speed order"
    spread = {g: max(avg[g][m]["wer"] for m in C3_ORDER) - min(avg[g][m]["wer"] for m in C3_ORDER) for g in avg}
    scored = sum(c3["sets"][s]["n_kept"] for s in c3["sets"])
    same = [sum(d.values()) / scored for d in c3["same_transcript_b200_vs_a100"].values()]
    bullets(fig, 0.06, y - 0.004, [
        f"Accuracy: the three averages span {100 * spread['a100']:.2f} points on the A100 and "
        f"{100 * spread['b200']:.2f} on the B200; each model's transcripts on the two GPUs are identical on "
        f"{min(same):.0%}-{max(same):.0%} of the scored clips.",
        f"Speed: on the A100 Rumble is {avg['a100'][v]['ms_per_clip_mean'] / avg['a100'][r]['ms_per_clip_mean']:.1f}x "
        f"faster than Voxtral Mini; on the B200 Voxtral Mini is faster. Voxtral Mini writes a FLEURS test transcript in "
        f"{tok['a100'][r]['tokens_per_clip_mean'] / tok['a100'][v]['tokens_per_clip_mean']:.1f}x fewer tokens (each "
        f"model's own tokenizer), but each of its tokens costs "
        f"{tok['a100'][v]['fit_ms_per_token'] / tok['a100'][r]['fit_ms_per_token']:.1f}x Rumble's on the A100 and "
        f"{tok['b200'][v]['fit_ms_per_token'] / tok['b200'][r]['fit_ms_per_token']:.1f}x on the B200 (ms/token: "
        "least-squares fit of each FLEURS test clip's time against its output tokens).",
    ], 118, size=8.0, gap=0.002)
    footer(fig, 4, f"bengali-asr-benchmark · three-model comparison, 20 Nov 2026 · {gpus['a100']} (two cards of "
           f"that model) and {gpus['b200']}, vLLM 0.31")
    pdf.savefig(fig); plt.close(fig)


def main():
    out = "asr_benchmark.pdf"
    with PdfPages(out) as pdf:
        page1(pdf); page2(pdf); page3(pdf)
        if C3_PATH.exists():
            page4(pdf, json.loads(C3_PATH.read_text()))
    print("wrote", out)


if __name__ == "__main__":
    main()
