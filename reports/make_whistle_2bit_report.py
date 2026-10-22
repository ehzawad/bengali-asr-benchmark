#!/usr/bin/env python
"""Render reports/whistle_2bit_vs_4bit_cpu.pdf: Whistle bn-BD as a 2-bit and as a 4-bit .cact archive, both run on
CPU in Cactus Compute's own inference engine (cactus-needle 3.2.1, libneedle).

Numbers are read from reports/whistle_2bit_vs_4bit.json; its 4-bit rows are asserted against the committed
benchmark summaries (whistle_bn_bd in outputs_fleurs/clean_test_table.json and
outputs_multiset/summary_multiset.json), so the PDF cannot drift from the published 4-bit runs.
Run from the repository root:  python reports/make_whistle_2bit_report.py
"""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

PAGE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NAVY, NAVY_SUB, ACCENT = "#16213e", "#c9d1e3", "#eb6834"
C4, C2 = "#2a78d6", "#eb6834"            # 4-bit blue, 2-bit orange (validated as an adjacent pair)
C4_LIGHT, C2_LIGHT = "#a9c8ee", "#f6c2a8"  # decode segment tints of the same hues
TINT, ZEBRA = "#fdebe2", "#f3f2ee"
PT = 1 / (11.69 * 72)

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": PAGE, "axes.facecolor": PAGE,
                     "savefig.facecolor": PAGE, "text.color": INK, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": INK_2, "hatch.color": PAGE, "hatch.linewidth": 1.0})

D = json.loads(Path("reports/whistle_2bit_vs_4bit.json").read_text())
SETS, MEAN, VAL = D["test"]["sets"], D["test"]["mean"], D["validation"]
SPEED, PROBE, SIZE = D["speed"]["rows"], D["mixed_layout_probe"]["rows"], D["sizes"]

# The 4-bit rows are the published benchmark rows.
_clean = json.loads(Path("outputs_fleurs/clean_test_table.json").read_text())["models"]["whistle_bn_bd"]
_ms = json.loads(Path("outputs_multiset/summary_multiset.json").read_text())["models"]["whistle_bn_bd"]
for r in SETS:
    ref = _clean if r["set"] == "fleurs_test" else _ms[r["set"]]
    assert abs(100 * ref["wer"] - r["w4"]["wer"]) < 0.006 and abs(100 * ref["cer"] - r["w4"]["cer"]) < 0.006, r["set"]
    assert ref["n"] == r["n"]
assert len(SETS) == 11


def sp(cpu, threads, bits):
    return next(r for r in SPEED if r["cpu"] == cpu and r["threads"] == threads and r["bits"] == bits)


CPUS = [("zen5", "AMD Zen 5", "AMD Zen 5 server CPU (AVX-512)"),
        ("cascadelake", "Intel Cascade Lake", "Intel Cascade Lake server CPU (AVX-512, no VBMI)")]
CLIPS = SPEED[0]["clips"]
AUDIO_S = SPEED[0]["ms_per_clip"] / SPEED[0]["rtf"] / 1000
assert all(r["clips"] == CLIPS for r in SPEED)
slow1 = sp("zen5", 1, 2)["ms_per_clip"] / sp("zen5", 1, 4)["ms_per_clip"]
smaller = 1 - SIZE["w2"]["bytes"] / SIZE["w4"]["bytes"]
better = [r["label"] for r in SETS if r["delta"]["wer"] < 0]
worst = max(SETS, key=lambda r: r["delta"]["wer"])
ttft_ratio = [sp(c, t, 2)["ttft_ms_per_clip"] / sp(c, t, 4)["ttft_ms_per_clip"] for c, _, _ in CPUS for t in (1, 4)]
tok_ratio = [sp(c, t, 2)["decode_ms_per_token"] / sp(c, t, 4)["decode_ms_per_token"] for c, _, _ in CPUS for t in (1, 4)]
assert min(ttft_ratio) > 3 and all(0.85 < x < 1.05 for x in tok_ratio)


def sgn(x, nd=2):
    return f"{x:+.{nd}f}".replace("-", "−")


def text_block(fig, x, y, text, width, size=9.5, color=INK_2, weight="normal", spacing=1.5):
    lines = textwrap.wrap(text, width)
    fig.text(x, y, "\n".join(lines), size=size, color=color, weight=weight, va="top", linespacing=spacing)
    return y - len(lines) * size * spacing * PT


def bullets(fig, x, y, items, width, size=9.0, gap=0.006):
    for item in items:
        fig.text(x, y, "•", size=size, color=ACCENT, va="top", weight="bold")
        y = text_block(fig, x + 0.02, y, item, width, size=size) - gap
    return y


def heading(fig, x, y, text, size=13):
    fig.patches.append(Rectangle((x, y - 0.0125), 0.008, 0.016, transform=fig.transFigure,
                                 facecolor=ACCENT, edgecolor="none"))
    fig.text(x + 0.018, y, text, size=size, weight="bold", va="top")


def box(fig, x, y, w, h, color, radius=0.006, zorder=0):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                      transform=fig.transFigure, facecolor=color, edgecolor="none", zorder=zorder))


def swatch(fig, x, y, color, hatch=None):
    fig.patches.append(FancyBboxPatch((x, y - 0.0055), 0.016, 0.011, boxstyle="round,pad=0,rounding_size=0.002",
                                      transform=fig.transFigure, facecolor=color, edgecolor="none", hatch=hatch))


def header(fig, title, subtitle):
    fig.patches.append(Rectangle((0, 0.89), 1, 0.11, transform=fig.transFigure, facecolor=NAVY, edgecolor="none"))
    fig.patches.append(Rectangle((0, 0.884), 1, 0.006, transform=fig.transFigure, facecolor=ACCENT, edgecolor="none"))
    fig.text(0.06, 0.957, title, size=22, weight="bold", color="white", va="center")
    fig.text(0.06, 0.915, subtitle, size=9.6, color=NAVY_SUB, va="center")


def footer(fig, page):
    fig.text(0.06, 0.03, "bengali-asr-benchmark \u00b7 Whistle bn-BD 2-bit vs 4-bit, Cactus CPU inference \u00b7 "
             "22 Oct 2026 \u00b7 numbers in reports/whistle_2bit_vs_4bit.json",
             size=7, color=MUTED)
    fig.text(0.94, 0.03, str(page), size=7, color=MUTED, ha="right")


def tile(fig, x, y, w, h, big, small, label):
    box(fig, x, y, w, h, "#f1f0eb")
    fig.text(x + 0.015, y + h - 0.016, label, size=7.6, color=MUTED, weight="bold", va="top")
    fig.text(x + 0.015, y + h * 0.43, big, size=17, weight="bold", va="center")
    fig.text(x + 0.015, y + 0.014, small, size=7.8, color=INK_2, va="bottom")


# ---------------------------------------------------------------------------------------------------------------
def page1(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "Whistle bn-BD: 2-bit vs 4-bit",
           "Cactus CPU inference (cactus-needle 3.2.1, libneedle) · eleven test sets · "
           "lower is better for every number")

    w, gap, top = 0.2, 0.0133, 0.858
    tiles = [
        (f"{SIZE['w2']['mb']:.2f} MB", f"4-bit {SIZE['w4']['mb']:.1f} MB, {100 * smaller:.0f}% smaller",
         "ARCHIVE SIZE"),
        (f"{MEAN['w2']['wer']:.2f}%", f"4-bit {MEAN['w4']['wer']:.2f}%, {sgn(MEAN['delta']['wer'])} pt",
         "MEAN TEST WER"),
        (f"{VAL['w2']['wer']:.2f}%", f"4-bit {VAL['w4']['wer']:.2f}%, {sgn(VAL['w2']['wer'] - VAL['w4']['wer'])} pt",
         "FLEURS VALIDATION WER"),
        (f"{sp('zen5', 1, 2)['ms_per_clip']:,.0f} ms", f"4-bit {sp('zen5', 1, 4)['ms_per_clip']:,.0f} ms, "
         f"{slow1:.1f}× slower", "CPU PER CLIP, ZEN 5, 1 THR"),
    ]
    for i, (big, small, label) in enumerate(tiles):
        tile(fig, 0.06 + i * (w + gap), top - 0.085, w, 0.085, big, small, label)
    fig.text(0.06, top - 0.097, "Large figures: the 2-bit archive; the 4-bit archive below each.", size=7.6,
             color=MUTED, va="top")

    y = top - 0.125
    heading(fig, 0.06, y, "Summary")
    z5_4 = sp("zen5", 4, 2)["ms_per_clip"] / sp("zen5", 4, 4)["ms_per_clip"]
    cl_1 = sp("cascadelake", 1, 2)["ms_per_clip"] / sp("cascadelake", 1, 4)["ms_per_clip"]
    cl_4 = sp("cascadelake", 4, 2)["ms_per_clip"] / sp("cascadelake", 4, 4)["ms_per_clip"]
    y = text_block(fig, 0.06, y - 0.032,
                   f"The 2-bit archive is {SIZE['w2']['mb']:.2f} MB against {SIZE['w4']['mb']:.1f} MB for the published "
                   f"4-bit one ({100 * smaller:.0f}% smaller). Accuracy drops a little: the mean word error rate over "
                   f"the eleven test sets goes from {MEAN['w4']['wer']:.2f}% to {MEAN['w2']['wer']:.2f}% "
                   f"({sgn(MEAN['delta']['wer'])} points), and the 2-bit model is worse on ten sets, by at most "
                   f"{worst['delta']['wer']:.2f} points ({worst['label']}), and better on {', '.join(better)}. "
                   f"On x86 CPUs it is much slower: {slow1:.1f}× on one Zen 5 thread, {z5_4:.1f}× on four, and "
                   f"{cl_1:.1f}× / {cl_4:.1f}× on Cascade Lake. All of the slowdown is in the audio encoder and "
                   "prefill (time to first token, "
                   f"{min(ttft_ratio):.1f}\u2013{max(ttft_ratio):.1f}× longer); decoding costs about the same per token "
                   "for both archives. ARM CPUs, where the engine's 2-bit kernels may behave differently, were not "
                   "tested. The 2-bit archive suits a tight storage or download budget; the 4-bit archive "
                   "remains the faster and slightly more accurate choice on x86.",
                   104, size=9.3, color=INK)

    y -= 0.022
    heading(fig, 0.06, y, "Table 1 · Word and character error rate per test set (%)")
    y -= 0.04
    cols = [("Test set", 0.075, "left"), ("clips", 0.315, "right"),
            ("4-bit", 0.425, "right"), ("2-bit", 0.51, "right"), ("Δ", 0.595, "right"),
            ("4-bit", 0.725, "right"), ("2-bit", 0.81, "right"), ("Δ", 0.895, "right")]
    fig.text(0.51, y + 0.017, "WER", size=8, color=MUTED, weight="bold", ha="center")
    fig.text(0.81, y + 0.017, "CER", size=8, color=MUTED, weight="bold", ha="center")
    fig.add_artist(plt.Line2D([0.36, 0.6], [y + 0.0105] * 2, transform=fig.transFigure, color=GRID, lw=0.8))
    fig.add_artist(plt.Line2D([0.66, 0.9], [y + 0.0105] * 2, transform=fig.transFigure, color=GRID, lw=0.8))
    for name, x, ha in cols:
        fig.text(x, y, name, size=8, color=MUTED, weight="bold", ha=ha, va="center")
    y -= 0.008
    fig.add_artist(plt.Line2D([0.06, 0.91], [y] * 2, transform=fig.transFigure, color=GRID, lw=0.8))
    ROW = 0.0215
    for i, r in enumerate(SETS + [None]):
        cy = y - ROW * i - ROW / 2 - 0.002
        if r is None:
            box(fig, 0.06, cy - ROW / 2, 0.85, ROW, TINT, radius=0.004)
            vals = ["Mean of the eleven sets", "", MEAN["w4"]["wer"], MEAN["w2"]["wer"], MEAN["delta"]["wer"],
                    MEAN["w4"]["cer"], MEAN["w2"]["cer"], MEAN["delta"]["cer"]]
            wt = "bold"
        else:
            if i % 2:
                box(fig, 0.06, cy - ROW / 2, 0.85, ROW, ZEBRA, radius=0.003)
            vals = [r["label"], f"{r['n']:,}", r["w4"]["wer"], r["w2"]["wer"], r["delta"]["wer"],
                    r["w4"]["cer"], r["w2"]["cer"], r["delta"]["cer"]]
            wt = "normal"
        for j, ((_, x, ha), v) in enumerate(zip(cols, vals)):
            s = v if isinstance(v, str) else (sgn(v) if j in (4, 7) else f"{v:.2f}")
            fig.text(x, cy, s, size=8.8, ha=ha, va="center", weight=wt,
                     color=INK if j not in (4, 7) else INK_2)
    y -= ROW * (len(SETS) + 1) + 0.022
    y = text_block(fig, 0.06, y,
                   "Δ = 2-bit minus 4-bit, in percentage points; positive means the 2-bit archive makes more "
                   "errors. The 4-bit column is the benchmark's published Whistle bn-BD CPU row. FLEURS has 920 test "
                   "recordings; the other sets are the benchmark's seeded 1,000-recording samples of each official "
                   "split (Ben-10: validation split). The mean is unweighted over the sets. "
                   f"Checkpoint selection used FLEURS validation ({VAL['n']} clips): 4-bit "
                   f"{VAL['w4']['wer']:.2f} / {VAL['w4']['cer']:.2f}, 2-bit {VAL['w2']['wer']:.2f} / "
                   f"{VAL['w2']['cer']:.2f} WER / CER.", 118, size=7.8, color=MUTED, spacing=1.45)
    y -= 0.02
    heading(fig, 0.06, y, "Keep in mind")
    bullets(fig, 0.06, y - 0.032, [
        "Speed was measured on x86 server CPUs only. Phones and other ARM devices, the engine's main target, were "
        "not tested, and the 2-bit/4-bit speed ratio there may differ.",
        "The per-set differences are small next to the spread between sets; no confidence intervals are given for "
        "the difference.",
    ], 112, size=8.4, gap=0.004)
    footer(fig, 1)
    pdf.savefig(fig); plt.close(fig)


def page2(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "Where the 2-bit archive costs",
           "Accuracy change per test set, and CPU time per clip split into prefill and decoding")

    # Chart 1: WER delta per set.
    y = 0.85
    heading(fig, 0.06, y, "Change in word error rate, 2-bit minus 4-bit (points)")
    fig.text(0.06, y - 0.03, "Orange bars: 2-bit worse; blue bar: 2-bit better. Dashed line: mean change.",
             size=8, color=MUTED, va="top")
    ax = fig.add_axes([0.25, 0.53, 0.66, 0.255])
    order = sorted(SETS, key=lambda r: r["delta"]["wer"], reverse=True)
    for j, r in enumerate(order):
        d = r["delta"]["wer"]
        ax.barh(-j, d, height=0.66, color=C2 if d > 0 else C4, edgecolor=PAGE, linewidth=1.0)
        ax.text(d + (0.03 if d >= 0 else -0.03), -j, sgn(d), va="center", ha="left" if d >= 0 else "right",
                size=8.2, color=INK, bbox=dict(facecolor=PAGE, edgecolor="none", pad=0.6))
    ax.axvline(0, color=INK_2, lw=0.9)
    ax.axvline(MEAN["delta"]["wer"], color=INK_2, lw=0.9, ls=(0, (4, 3)))
    ax.text(MEAN["delta"]["wer"], 0.95, f" mean {sgn(MEAN['delta']['wer'])}", size=7.8, color=INK_2,
            va="bottom", ha="left")
    ax.set_yticks([-j for j in range(len(order))])
    ax.set_yticklabels([r["label"] for r in order], size=8.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.6, 1.4)
    ax.set_ylim(-len(order) + 0.4, 1.5)
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
    ax.set_xticklabels(["−0.5", "0", "+0.5", "+1.0"], size=8)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)

    # Chart 2: ms per clip, grouped by CPU and threads, stacked prefill + decode.
    y = 0.475
    heading(fig, 0.06, y, f"CPU time per clip (ms), {CLIPS} FLEURS validation clips")
    lx, ly = 0.06, y - 0.036
    for col, hatch, lab in [(C4, None, "4-bit prefill"), (C4_LIGHT, None, "4-bit decoding"),
                            (C2, None, "2-bit prefill"), (C2_LIGHT, None, "2-bit decoding")]:
        swatch(fig, lx, ly, col, hatch)
        t = fig.text(lx + 0.022, ly, lab, size=8.2, va="center", color=INK_2)
        lx += 0.022 + t.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width + 0.03
    fig.text(0.06, ly - 0.02, "Prefill = time to first token (audio encoder and prompt). Label = total; "
             "each bar is one engine process.", size=8, color=MUTED, va="center")
    ax = fig.add_axes([0.1, 0.13, 0.81, 0.265])
    groups = [(c, t) for c, _, _ in CPUS for t in (1, 4)]
    bw = 0.36
    for g, (c, t) in enumerate(groups):
        for k, (bits, cs, cl) in enumerate([(4, C4, C4_LIGHT), (2, C2, C2_LIGHT)]):
            r = sp(c, t, bits)
            x = g + (k - 0.5) * (bw + 0.03)
            ax.bar(x, r["ttft_ms_per_clip"], width=bw, color=cs, edgecolor=PAGE, linewidth=1.0)
            ax.bar(x, r["decode_ms_per_clip"], bottom=r["ttft_ms_per_clip"], width=bw, color=cl,
                   edgecolor=PAGE, linewidth=1.0)
            ax.text(x, r["ms_per_clip"] + 80, f"{r['ms_per_clip']:,.0f}", ha="center", va="bottom", size=8.2,
                    color=INK, weight="bold" if bits == 2 else "normal")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{name}\n{t} thread{'s' if t > 1 else ''}" for c, name, _ in CPUS for t in (1, 4)],
                       size=8.6, color=INK_2)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 8000)
    ax.set_yticks(range(0, 8001, 2000))
    ax.set_yticklabels([f"{v:,}" for v in range(0, 8001, 2000)], size=8)
    ax.set_ylabel("milliseconds per clip", size=8.2, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    footer(fig, 2)
    pdf.savefig(fig); plt.close(fig)


def page3(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    header(fig, "Speed table and method", "Same machine, same clips and same engine for both archives on each CPU")

    y = 0.85
    heading(fig, 0.06, y, "Table 2 · CPU speed, one engine process")
    y -= 0.042
    cols = [("CPU", 0.075, "left"), ("threads", 0.4, "right"), ("archive", 0.485, "right"),
            ("ms/clip", 0.575, "right"), ("prefill ms", 0.67, "right"), ("decode ms", 0.765, "right"),
            ("ms/token", 0.85, "right"), ("RTF", 0.925, "right")]
    for name, x, ha in cols:
        fig.text(x, y, name, size=8, color=MUTED, weight="bold", ha=ha, va="center")
    y -= 0.01
    fig.add_artist(plt.Line2D([0.06, 0.935], [y] * 2, transform=fig.transFigure, color=GRID, lw=0.8))
    ROW, i = 0.019, 0
    for c, name, _ in CPUS:
        for t in (1, 4):
            for bits in (4, 2):
                r = sp(c, t, bits)
                cy = y - ROW * i - ROW / 2 - 0.002
                if bits == 2:
                    box(fig, 0.06, cy - ROW / 2, 0.875, ROW, TINT, radius=0.003)
                vals = [name if (t, bits) == (1, 4) else "", str(t) if bits == 4 else "", f"{bits}-bit",
                        f"{r['ms_per_clip']:,.0f}", f"{r['ttft_ms_per_clip']:,.0f}", f"{r['decode_ms_per_clip']:,.0f}",
                        f"{r['decode_ms_per_token']:.2f}", f"{r['rtf']:.3f}"]
                for (_, x, ha), v in zip(cols, vals):
                    fig.text(x, cy, v, size=8.8, ha=ha, va="center")
                i += 1
        if c != CPUS[-1][0]:
            ly = y - ROW * i - 0.002
            fig.add_artist(plt.Line2D([0.06, 0.935], [ly] * 2, transform=fig.transFigure, color=GRID, lw=0.6))
    y -= ROW * i + 0.02
    y = text_block(fig, 0.06, y,
                   f"Means over {CLIPS} FLEURS bn_in validation clips (about {AUDIO_S:.1f} s of audio each), both archives "
                   "run one after the other on the same machine. Prefill = the engine's reported time to first "
                   "token (audio encoder and prompt); decode = the rest of the clip's time; ms/token = decode "
                   "time over the estimated output tokens (about 54 per clip for both). RTF = processing time over "
                   f"audio duration. CPUs: {CPUS[0][2]}; {CPUS[1][2]}. "
                   "One pass per row; in a two-pass check on Zen 5 (the probe below) passes differed by under 1%.",
                   118, size=7.8, color=MUTED, spacing=1.45)

    y -= 0.022
    heading(fig, 0.06, y, "How it was measured")
    y = bullets(fig, 0.06, y - 0.032, [
        "Both archives are Cactus Compute .cact files of the same Whistle bn-BD model, run by Cactus's own CPU "
        "inference engine (cactus-needle 3.2.1, libneedle), not the PyTorch or GPU paths used elsewhere in the "
        "benchmark.",
        f"4-bit: the published {SIZE['w4']['mb']:.1f} MB archive, every quantised tensor at 4 bits. 2-bit: "
        f"{SIZE['w2']['bytes']:,} bytes; the embedding and mhc_phi tensors stay at 4 bits and the other "
        f"{SIZE['w2']['cq_tensor_widths']['CQ2']} quantised tensors are 2-bit, the same layout as Cactus's own English "
        "2-bit Whistle archive. The 2-bit model was fine-tuned for 2-bit weights and selected on FLEURS validation.",
        "Same engine settings for both: the engine's default decoding (its beam search), language bn, no external "
        "language model, no decoding options changed.",
        "Same audio and scorer as the benchmark: 16 kHz mono, cut at pauses into pieces of at most 25 s by the "
        "benchmark's audio path, piece texts joined with a space; punctuation removed and Bengali text "
        "Unicode-normalised before counting errors.",
    ], 112, size=8.4, gap=0.004)

    y -= 0.012
    heading(fig, 0.06, y, "Mixed-layout probe (not fine-tuned)")
    y = text_block(fig, 0.06, y - 0.032,
                   "To locate the slowdown, the 4-bit model was also exported with only one half at 2 bits and timed "
                   "on one Zen 5 thread (41 FLEURS validation clips, mean of two passes). These archives were not "
                   "fine-tuned, so only their speed and size are meaningful.", 114, size=8.4)
    y -= 0.012
    cols = [("layout", 0.075, "left"), ("size MB", 0.56, "right"), ("prefill ms", 0.7, "right"),
            ("ms / clip", 0.84, "right")]
    for name, x, ha in cols:
        fig.text(x, y, name, size=8, color=MUTED, weight="bold", ha=ha, va="center")
    y -= 0.008
    fig.add_artist(plt.Line2D([0.06, 0.86], [y] * 2, transform=fig.transFigure, color=GRID, lw=0.8))
    for i, r in enumerate(PROBE):
        cy = y - ROW * i - ROW / 2 - 0.002
        if i % 2:
            box(fig, 0.06, cy - ROW / 2, 0.8, ROW, ZEBRA, radius=0.003)
        for (_, x, ha), v in zip(cols, [r["layout"], f"{r['mb']:.1f}", f"{r['ttft_ms_per_clip']:,.0f}",
                                        f"{r['ms_per_clip']:,.0f}"]):
            fig.text(x, cy, v, size=8.8, ha=ha, va="center")
    y -= ROW * len(PROBE) + 0.02
    e42 = next(r for r in PROBE if r["layout"].startswith("encoder 4-bit"))
    e24 = next(r for r in PROBE if r["layout"].startswith("encoder 2-bit"))
    y = text_block(fig, 0.06, y,
                   f"A 2-bit encoder is what makes prefill slow ({e24['ttft_ms_per_clip']:,.0f} ms against "
                   f"{e42['ttft_ms_per_clip']:,.0f} ms with a 4-bit encoder). Keeping the encoder at 4 bits and the "
                   f"decoder at 2 bits gives a {e42['mb']:.1f} MB archive at {e42['ms_per_clip']:,.0f} ms per clip; "
                   "its accuracy after fine-tuning was not measured.", 114, size=8.4, color=INK_2)
    footer(fig, 3)
    pdf.savefig(fig); plt.close(fig)


def main():
    out = "reports/whistle_2bit_vs_4bit_cpu.pdf"
    with PdfPages(out, metadata={"Title": "Whistle bn-BD: 2-bit vs 4-bit (Cactus CPU inference)", "Author": "",
                                 "Creator": "", "Producer": ""}) as pdf:
        page1(pdf); page2(pdf); page3(pdf)
    print("wrote", out)


if __name__ == "__main__":
    main()
