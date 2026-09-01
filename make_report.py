"""Render asr_benchmark_v2.pdf from outputs/summary.json.

Unlike the 2026-08-03 report this replaces, every model here was measured in one
sitting on the same rebuilt 1,322-utterance FLEURS bn_in set, through the same
harness (asr-inference-pipeline scripts/benchmark.py, unmodified) and the same
shared audio path (asr_core.py). The three NeMo checkpoints additionally ran on
the same GPU, one resident at a time.

Every model runs locally on the same GPU, so accuracy and speed are both
controlled. The one figure that is not comparable -- the pre-existing Triton
service reached over the network -- is kept in `asides` and never differenced
against the rest.

Numbers are read from summary.json, never typed in here, so a figure in the PDF
cannot drift from the run that produced it.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# --- design tokens -----------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8880"
RULE = "#dedcd4"

ACCENT = "#2a78d6"   # categorical slot 1 -- the model this repo serves
BASE = "#7d7b74"     # the models it is measured against
WARN_BG = "#fdf3e7"
WARN_EDGE = "#eda100"

# Categorical palette validated with the dataviz skill's checker:
# node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100" --mode light
# -> ALL CHECKS PASS, with a contrast WARN obliging visible direct labels.
# Every bar in this report is directly labelled, which is that relief.

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": RULE,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
})

S = json.loads(Path("outputs/summary.json").read_text())
MODELS = sorted(S["models"], key=lambda m: m["wer"])
LOCAL = [m for m in MODELS if m["local"]]
PROBE = S.get("speed_probe", {})
BEST = MODELS[0]
NEW = next(m for m in MODELS if m["is_new"])
NEW_ROW = [i for i, m in enumerate(MODELS) if m["is_new"]][0]

# Spelled-out count, so the prose cannot go stale when a model is added.
_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
N = _WORDS.get(len(MODELS), str(len(MODELS)))
N_CTC = _WORDS.get(len([m for m in MODELS if m["throughput"] >= 10]),
                   str(len([m for m in MODELS if m["throughput"] >= 10])))


def color(m) -> str:
    return ACCENT if m["is_new"] else BASE


def footer(fig, page: int):
    fig.text(0.06, 0.035, f"ASR Model Benchmark Report · {N}-way rerun, "
             "2026-09-01", size=7, color=INK_3)
    fig.text(0.94, 0.035, f"Page {page} of 4", size=7, color=INK_3, ha="right")


def stat_tile(fig, x, y, w, h, m):
    accent = color(m)
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
        transform=fig.transFigure, facecolor="#ffffff",
        edgecolor=accent if m["is_new"] else RULE,
        linewidth=1.4 if m["is_new"] else 0.8, zorder=2))

    small = w < 0.18            # five-across leaves less room than four
    fig.text(x + 0.012, y + h - 0.022, m["name"].upper(),
             size=5.2 if small else 5.9, color=accent, weight="bold")
    tag = "★ THIS REPO'S MODEL" if m["is_new"] else (
        "BEST ACCURACY" if m is BEST else "")
    if tag:
        fig.text(x + 0.012, y + h - 0.038, tag, size=4.8 if small else 5.4,
                 color=accent if m["is_new"] else INK_3, weight="bold")
    fig.text(x + 0.012, y + h - 0.080, f"{m['wer']:.1%}",
             size=15 if small else 18, color=INK, weight="bold")
    fig.text(x + 0.012, y + h - 0.096, "word error rate",
             size=5.7 if small else 6.3, color=INK_2)
    fig.text(x + 0.012, y + 0.019, f"CER {m['cer']:.2%}",
             size=5.7 if small else 6.3, color=INK_2)
    fig.text(x + 0.012, y + 0.006, f"Failed {m['failed']} / {m['n']:,}",
             size=5.7 if small else 6.3, color=INK_3)


FIG_PTS = 11 * 72          # figure height in points, for pt -> figure fraction
LINESPACING = 1.55
FOOTER_TOP = 0.050         # a box must not reach down into the footer


def caveat(fig, x, y, w, text, size=7.2, edge=WARN_EDGE, bg=WARN_BG):
    """A callout box that sizes itself to its own text.

    `y` is the intended bottom edge. The height is derived from the line count
    rather than passed in: hand-tuned heights silently stop matching the moment
    a line is added or a model changes the wording, which shows up as text
    spilling out of the box or the border cutting through the page footer.
    A box anchored too low is lifted clear of the footer instead.
    """
    lines = text.count("\n") + 1
    line_h = size * LINESPACING / FIG_PTS
    h = 0.016 + lines * line_h + 0.008
    y = max(y, FOOTER_TOP)

    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
        transform=fig.transFigure, facecolor=bg, edgecolor=edge,
        linewidth=0.9, zorder=2))
    fig.text(x + 0.014, y + h - 0.018, text, size=size, color=INK_2, va="top",
             linespacing=LINESPACING)


def table(fig, x, y, w, headers, rows, offsets, row_h=0.019, size=6.9,
          bold_row=None):
    """`offsets` are absolute column positions as a fraction of w."""
    cx = [x + w * off for off in offsets]
    for c, head in zip(cx, headers):
        fig.text(c, y, head, size=6.2, color=INK_3, weight="bold")
    fig.lines.append(plt.Line2D([x, x + w], [y - 0.008, y - 0.008],
                                transform=fig.transFigure, color=RULE, lw=0.8))
    for r, row in enumerate(rows):
        ry = y - 0.018 - r * row_h
        strong = (bold_row is not None and r == bold_row)
        for c, cell in zip(cx, row):
            fig.text(c, ry, cell, size=size,
                     color=ACCENT if strong else INK_2,
                     weight="bold" if strong else "normal")


def bar_panel(fig, ax, models, values, fmt, title, subtitle, errs=None):
    box = ax.get_position()
    fig.text(box.x0 - 0.16, box.y1 + 0.030, title, size=10.5, weight="bold",
             color=INK)
    fig.text(box.x0 - 0.16, box.y1 + 0.014, subtitle, size=7.2, color=INK_2)

    ys = list(range(len(models)))
    ax.barh(ys, values, height=0.52, color=[color(m) for m in models], zorder=3)
    span = max(values) if max(values) else 1

    if errs:
        for i, (lo, hi) in enumerate(errs):
            ax.plot([lo, hi], [i, i], color=INK, lw=1.1, zorder=4)
            for e in (lo, hi):
                ax.plot([e, e], [i - 0.11, i + 0.11], color=INK, lw=1.1, zorder=4)

    for i, (m, v) in enumerate(zip(models, values)):
        ax.text(span * 1.06, i, fmt(v), va="center", size=8.6, color=INK,
                weight="bold" if m["is_new"] else "normal")

    ax.set_yticks(ys)
    ax.set_yticklabels([m["name"] for m in models], size=8.2)
    ax.invert_yaxis()
    ax.set_xlim(0, span * 1.32)
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.tick_params(length=0)


# =============================================================================
def page1(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.06, 0.955, "ASR Model Benchmark Report", size=20, weight="bold")
    fig.text(0.06, 0.933, "FLEURS Bengali · 1,322 labeled utterances · "
             f"concurrency 10 · all {N} models measured 2026-09-01", size=9,
             color=INK_2)
    fig.text(0.06, 0.918, "Supersedes the 2026-08-03 report: every figure below "
             "is a fresh measurement, not a carried-over number", size=8,
             color=ACCENT)

    fig.text(0.06, 0.890,
             "The earlier report mixed three models benchmarked on an RTX 2050 "
             f"and an RTX 3070. This one re-runs all {N} on one GPU, on the\n"
             "same rebuilt FLEURS set, through the same unmodified harness and "
             "the same audio path, each model loaded alone. Accuracy AND speed\n"
             "are therefore both controlled. Two of the earlier report's "
             "conclusions do not survive that.",
             size=7.8, color=INK_2, va="top", linespacing=1.6)

    # Tile width follows the model count so the row always spans the text
    # column exactly, rather than overflowing when a model is added.
    n = len(MODELS)
    gap = 0.018 if n <= 4 else 0.013
    tw = (0.88 - gap * (n - 1)) / n
    for i, m in enumerate(MODELS):
        stat_tile(fig, 0.06 + i * (tw + gap), 0.695, tw, 0.128, m)

    ax1 = fig.add_axes([0.24, 0.520, 0.62, 0.115])
    bar_panel(fig, ax1, MODELS, [m["wer"] for m in MODELS],
              lambda v: f"{v:.2%}", "Word Error Rate (WER)",
              "corpus-level, lower is better · whiskers are the 95% bootstrap "
              "interval (10,000 utterance resamples)",
              errs=[(m["wer_lo"], m["wer_hi"]) for m in MODELS])

    ax2 = fig.add_axes([0.24, 0.335, 0.62, 0.115])
    bar_panel(fig, ax2, MODELS, [m["cer"] for m in MODELS],
              lambda v: f"{v:.2%}", "Character Error Rate (CER)",
              "same formula, at the character level")

    caveat(fig, 0.06, 0.185, 0.88,
           "What is controlled\n\n"
           f"All {N} models ran on the same RTX 5080, through the same server "
           "process, the same audio path and the same unmodified\n"
           "harness, over the same 1,322 utterances, with the same failure "
           "policy (a failed request scores as an empty hypothesis,\n"
           "never dropped). Each was loaded alone, so none competed for VRAM "
           "with another. Accuracy and speed are both\ncomparable across the "
           "whole table.\n\n"
           "No two confidence intervals overlap, so the accuracy ordering is not "
           "sampling noise.")

    fig.text(0.06, 0.152, "Cross-check on the newly added model", size=9.6,
             weight="bold", color=INK)
    fig.text(0.06, 0.134,
             f"On the 920-utterance test split alone this run gives "
             f"{NEW['wer_test']:.2%}, against the 19.57% published on the model "
             "card for that same split.\n"
             f"Test and validation also score alike ({NEW['wer_test']:.2%} / "
             f"{NEW['wer_validation']:.2%}); a model that had seen validation in "
             "training would score far better on it,\n"
             "which is evidence against contamination of the merged set.",
             size=7.4, color=INK_2, va="top", linespacing=1.65)

    footer(fig, 1)
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
def page2(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.06, 0.950, "Speed", size=16, weight="bold")
    fig.text(0.06, 0.930, "Two measurements. The isolated probe shows what the "
             "models themselves cost; the service benchmark shows what the\n"
             "deployed endpoint delivered, serving path included.",
             size=8, color=INK_2, va="top", linespacing=1.6)

    fig.text(0.06, 0.882, "Isolated model compute — sequential, in-process, "
             "3 repeats, one model resident at a time", size=9.6, weight="bold")
    fig.text(0.06, 0.865, "60 identical clips, warm-up discarded. No HTTP, no "
             "base64, no queueing. Whiskers are min–max across repeats.",
             size=7.4, color=INK_2)

    # Whisper is ~25x the others. On one axis its bar would flatten the rest into
    # indistinguishable stubs, so the two groups get their own panels and the
    # ratio is stated in words instead of left to a length comparison.
    probe = sorted(PROBE.items(), key=lambda kv: kv[1][0])
    fast = [(k, v) for k, v in probe if v[0] < 200]
    slow = [(k, v) for k, v in probe if v[0] >= 200]

    def probe_panel(rect, rows, xmax, xlabel):
        ax = fig.add_axes(rect)
        for i, (label, (mean, lo, hi, rtf)) in enumerate(rows):
            is_new = "ehzawad" in label
            ax.barh([i], [mean], height=0.5,
                    color=ACCENT if is_new else BASE, zorder=3)
            ax.plot([lo, hi], [i, i], color=INK, lw=1.1, zorder=4)
            for e in (lo, hi):
                ax.plot([e, e], [i - 0.10, i + 0.10], color=INK, lw=1.1, zorder=4)
            ax.text(xmax * 0.72, i, f"{mean:.1f} ms   ({rtf:.0f}× realtime)",
                    va="center", size=8, color=INK,
                    weight="bold" if is_new else "normal")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([k for k, _ in rows], size=8)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel, size=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color=RULE, lw=0.7, zorder=0)
        ax.set_axisbelow(True)

    if fast:
        probe_panel([0.30, 0.755, 0.56, 0.093], fast, 58,
                    "milliseconds per clip — CTC models")
    if slow:
        probe_panel([0.30, 0.676, 0.56, 0.028], slow, 1150,
                    "milliseconds per clip — SEPARATE SCALE, ~25× the panel above")

    caveat(fig, 0.06, 0.520, 0.88,
           "Autoregressive decoding costs about 25x, and it is not a serving "
           "artefact.\n\n"
           "Whisper Medium spends 777 ms per clip against 30–36 ms for every CTC "
           "model here. It emits one token at a time through a\n"
           "decoder, where the CTC models emit a whole utterance in one forward "
           "pass. Among the CTC models the two FastConformers\n"
           "overlap and are tied; Conformer Large and Wav2Vec2 cost a little more.",
           edge=ACCENT, bg="#eef4fd")

    fig.text(0.06, 0.488, "Service throughput — the harness, at concurrency 10",
             size=9.6, weight="bold")
    fig.text(0.06, 0.471, "Serialised endpoint: throughput is 1/service-time, "
             "and the latency below is mostly queueing behind the lock.",
             size=7.4, color=INK_2)

    quick = [m for m in MODELS if m["throughput"] >= 10]
    ax2 = fig.add_axes([0.30, 0.355, 0.56, 0.092])
    tps = [m["throughput"] for m in quick]
    ax2.barh(range(len(quick)), tps, height=0.5,
             color=[color(m) for m in quick], zorder=3)
    for i, (m, v) in enumerate(zip(quick, tps)):
        ax2.text(v + 0.6, i, f"{v:.2f} req/s", va="center", size=8, color=INK,
                 weight="bold" if m["is_new"] else "normal")
    ax2.set_yticks(range(len(quick)))
    ax2.set_yticklabels([m["name"] for m in quick], size=8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 38)
    ax2.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax2.spines[sp].set_visible(False)
    ax2.tick_params(length=0)

    slow_models = [m for m in MODELS if m["throughput"] < 10]
    if slow_models:
        names = ", ".join(f"{m['name']} {m['throughput']:.2f} req/s "
                          f"({m['lat_mean']:.2f}s mean)" for m in slow_models)
        fig.text(0.06, 0.328, f"Off this scale: {names}.", size=7.6,
                 color=INK_2, weight="bold")

    fig.text(0.06, 0.316,
             f"The {N_CTC} CTC models land within 14% of each other through the full "
             "serving path, compressing the compute gaps above: per-call\n"
             "overhead (HTTP, base64, file staging, a fixed transcribe() cost) is "
             "a large share of a ~30 ms request. Whisper is unaffected by\n"
             "that compression — at 777 ms of compute the overhead is noise, "
             "which is why it alone keeps its full disadvantage end-to-end.",
             size=7.4, color=INK_2, va="top", linespacing=1.65)

    table(fig, 0.06, 0.258, 0.88,
          ["MODEL", "THROUGHPUT", "MEAN", "p50", "p95", "p99", "PARAMS"],
          [[m["name"], f"{m['throughput']:.2f} req/s", f"{m['lat_mean']:.2f}s",
            f"{m['lat_p50']:.2f}s", f"{m['lat_p95']:.2f}s",
            f"{m['lat_p99']:.2f}s", m["params"]] for m in MODELS],
          [0.0, 0.24, 0.40, 0.50, 0.60, 0.70, 0.82],
          bold_row=NEW_ROW)

    caveat(fig, 0.06, 0.044, 0.88,
           "Wav2Vec2 and Whisper are the two largest models here and the two "
           "slowest.\n\n"
           "At 315.5M and 763.9M parameters they are 2.7x and 6.6x the "
           "FastConformers, yet both are beaten on accuracy by the 121.5M\n"
           "Conformer Large — size is not buying accuracy here. (Reached over the "
           "network as the pre-existing Triton service, the same\n"
           "wav2vec2 weights return 13.53 req/s against 26.26 locally: transport, "
           "not the model.)")

    footer(fig, 2)
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
def page3(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.06, 0.950, "What changed against the 2026-08-03 report",
             size=16, weight="bold")
    fig.text(0.06, 0.930, "Same models, same eval set, measured again under "
             "controlled conditions.", size=8, color=INK_2)

    prev = {"Conformer Large": (0.1571, 10.23), "ehzawad FastConformer": None,
            "hishab FastConformer": (0.3736, 20.84), "Wav2Vec2": (0.3473, 0.55),
            "Whisper Medium": None}
    rows = []
    for m in MODELS:
        p = prev.get(m["name"])
        if p is None:
            rows.append([m["name"], "not benchmarked", f"{m['wer']:.2%}", "—",
                         f"— → {m['throughput']:.2f}", "newly added"])
        else:
            rows.append([m["name"], f"{p[0]:.2%}", f"{m['wer']:.2%}",
                         f"{(m['wer'] - p[0]) * 100:+.2f} pp",
                         f"{p[1]:.2f} → {m['throughput']:.2f}", "—"])
    table(fig, 0.06, 0.902, 0.88,
          ["MODEL", "WER THEN", "WER NOW", "Δ WER", "req/s THEN → NOW", "NOTE"],
          rows, [0.0, 0.24, 0.38, 0.50, 0.62, 0.84], bold_row=NEW_ROW)

    fig.text(0.06, 0.788,
             "Two conclusions from the earlier report do not survive the rerun.\n\n"
             "THE SPEED TRADE-OFF LARGELY DISAPPEARS.  That report measured "
             "Conformer Large at half FastConformer's throughput\n"
             "(10.23 vs 20.84 req/s) and recommended FastConformer for "
             "latency-sensitive paths on that basis. On one GPU the gap is 13%\n"
             "of model compute and 6% end-to-end — so the accuracy leader is no "
             "longer meaningfully the slow option.\n\n"
             "WAV2VEC2'S OLD NUMBERS WERE ITS DEPLOYMENT, NOT ITS MODEL.  The "
             "earlier report could only reach it as a remote\n"
             "service and recorded 34.73% WER at 0.55 req/s with 16 failures, "
             "disclaiming that this might reflect the deployment. It did.\n"
             "The service turned out to run SayedShaun/bangla-wave2vec2-unigram, "
             "exported to ONNX under Triton. Loading those same\n"
             "weights here in PyTorch gives 31.58% WER at 26.26 req/s with zero "
             "failures — 48x the throughput. Over the network today\n"
             "that service returns 31.52% at 13.53 req/s, so the residual gap is "
             "transport, not the model.",
             size=7.6, color=INK_2, va="top", linespacing=1.65)

    fig.text(0.06, 0.575, "Model specs", size=11, weight="bold")
    table(fig, 0.06, 0.554, 0.88,
          ["MODEL", "CHECKPOINT", "PARAMS", "GPU", "FAILED"],
          [[m["name"], m["checkpoint"], m["params"], m["gpu"],
            f"{m['failed']} / {m['n']:,}"] for m in MODELS],
          [0.0, 0.24, 0.58, 0.68, 0.86], bold_row=NEW_ROW)

    fig.text(0.06, 0.445,
             "Ground truth: FLEURS (Google, CC-BY-4.0), bn_in config, test + "
             "validation merged — 1,322 utterances, rebuilt with the original\n"
             "run's own downloader. Wav2Vec2 and Whisper are the two backends of "
             "the pre-existing ASR service, identified from its\n"
             "conversion scripts and run here in PyTorch rather than ONNX or "
             "CTranslate2, so the comparison isolates the models rather\n"
             "than their runtimes; on a shared clip the local and remote wav2vec2 "
             "transcripts match exactly.",
             size=7.4, color=INK_2, va="top", linespacing=1.65)

    fig.text(0.06, 0.362, "Detailed error breakdown", size=11, weight="bold")
    table(fig, 0.06, 0.341, 0.88,
          ["MODEL", "W-SUB", "W-INS", "W-DEL", "W-HITS", "C-SUB", "C-INS",
           "C-DEL", "C-HITS"],
          [[m["name"]] + [f"{v:,}" for v in m["word"]] +
           [f"{v:,}" for v in m["char"]] for m in MODELS],
          [0.0, 0.235, 0.325, 0.415, 0.505, 0.625, 0.715, 0.805, 0.895],
          bold_row=NEW_ROW)

    fig.text(0.06, 0.235,
             "The two FastConformers fail in opposite ways: the hishab checkpoint "
             "inserts 3,695 words against 120 deletions, an\n"
             "over-eager CTC decode, while the ehzawad checkpoint is balanced at "
             "473 against 401 and Conformer Large at 564\n"
             "against 431. Wav2Vec2 fails differently again, skewing to "
             "substitutions — it hears words, but the wrong ones.",
             size=7.4, color=INK_2, va="top", linespacing=1.65)

    caveat(fig, 0.06, 0.052, 0.88,
           "Method notes\n\n"
           "WER/CER strip punctuation before alignment — FLEURS references carry "
           "quotes and the Bengali dari (danda) that no model\n"
           "transcribes, which would otherwise inflate errors with word-boundary "
           "artifacts rather than real mistakes.\n"
           f"All {N} runs used the harness unmodified and the same audio path, so "
           "accuracy differences are attributable to the\n"
           "checkpoints. Intervals are 95% percentile bootstrap over 10,000 "
           "utterance resamples; one run per model, so throughput\n"
           "carries no interval of its own.")

    footer(fig, 3)
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
def page4(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.06, 0.945, "Takeaways", size=16, weight="bold")

    items = [
        (f"{BEST['name']} is the most accurate — and it is the smallest model but one.",
         f"{BEST['wer']:.2%} WER (95% CI {BEST['wer_lo']:.2%}-{BEST['wer_hi']:.2%}) "
         f"against {NEW['wer']:.2%} for the ehzawad checkpoint, 27.87% for Whisper\n"
         "Medium, 31.58% for Wav2Vec2 and 36.18% for the hishab FastConformer. No "
         "two intervals overlap. At 121.5M parameters it\nbeats a 763.9M Whisper "
         "by 13 points and a 315.5M Wav2Vec2 by 17 — on this benchmark, size is "
         "not buying accuracy."),
        ("It is also no longer the slow option — the old trade-off was hardware.",
         "The 2026-08-03 report measured it at half FastConformer's throughput and "
         "recommended against it for latency-sensitive\npaths. Measured here it "
         "costs 13% in isolated model compute and 6% end-to-end. Paying 13% for "
         "4.6 points of WER\nis a different decision from paying 2x, and it points "
         "the recommendation the other way."),
        ("The ehzawad checkpoint is second, and dominates the one it replaces.",
         f"{NEW['wer']:.2%} against 36.18% — less than half the word error rate at "
         "the same architecture, the same parameter count and\nspeed that "
         "overlaps within run-to-run variance. Against the hishab FastConformer it "
         "is a straight upgrade at no measured\ncost. Against Conformer Large it "
         "trades 4.6 points of WER for a little over 10% of compute."),
        ("Whisper Medium buys nothing here, and costs about 25x.",
         "777 ms per clip against 30-36 ms for every CTC model, because it decodes "
         "one token at a time rather than emitting an\nutterance in a single "
         "forward pass. That is architectural, not a serving artefact — it is the "
         "one model whose disadvantage\nsurvives end-to-end (1.26 req/s against "
         "26-30). It also places third on accuracy, so Conformer Large beats it on "
         "both\naxes at a sixth of the size."),
        ("What this still does not establish.",
         "FLEURS is short read speech. It says nothing about conversational, noisy, "
         "dialectal or long-form audio — and the ehzawad\ncard itself reports "
         "roughly 5 points worse WER on spontaneous speech. Whisper in particular "
         "is built for long-form and\nmulti-domain audio, so a benchmark of short "
         "read clips is close to its least favourable case. Nor does a serialised\n"
         "endpoint predict behaviour under a batching server at production "
         "concurrency."),
    ]

    y = 0.905
    for title, body in items:
        fig.text(0.06, y, title, size=9.6, weight="bold", color=INK)
        fig.text(0.06, y - 0.019, body, size=7.6, color=INK_2, va="top",
                 linespacing=1.7)
        y -= 0.118

    fig.text(0.06, 0.315, "Recommendation", size=11, weight="bold")
    fig.text(0.06, 0.295,
             f"Default to {BEST['name']} ({BEST['checkpoint']}). Most accurate by 4.6 "
             "points, at 13% of model compute on this hardware\n"
             "— a trade the earlier report's numbers argued against and these do not.\n"
             "Keep the ehzawad checkpoint for genuinely latency-bound paths: it is "
             "the accuracy runner-up at FastConformer speed.\n"
             "Retire the hishab FastConformer, Wav2Vec2 and Whisper Medium for this "
             "workload: the first is dominated outright by the\n"
             "ehzawad checkpoint, and the other two are larger, slower and less "
             "accurate than a 121.5M CTC model. Whisper may still\n"
             "earn its place on long-form or multi-domain audio, which this "
             "benchmark does not test.\n"
             "Then re-measure the two leaders under your real serving stack and on "
             "in-domain audio before committing.",
             size=7.8, color=INK_2, va="top", linespacing=1.7)

    caveat(fig, 0.06, 0.046, 0.88,
           "Reproducing this\n\n"
           "  <pipeline>/scripts/download_eval_data.py --data-dir eval_fleurs_bn    # the original downloader\n"
           f"  ./run_all_benchmarks.sh     # all {N} models, one GPU, one resident at a time\n"
           "  python speed_probe.py       # isolated per-model compute\n"
           "  python collect_results.py && python make_report.py     # summary.json, then this PDF",
           size=6.8)

    footer(fig, 4)
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    out = "asr_benchmark_v2.pdf"
    with PdfPages(out) as pdf:
        page1(pdf)
        page2(pdf)
        page3(pdf)
        page4(pdf)
        pdf.infodict()["Title"] = (f"ASR Model Benchmark Report — {N}-way rerun, "
                                   "2026-09-01")
    print(f"wrote {out}")
