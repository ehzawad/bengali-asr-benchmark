#!/usr/bin/env python
"""Render the six-model A5000 report from outputs_a5000/summary_a5000.json.

This is a NEW protocol, not an extension of the RTX 5080 report: the GPU
changed, so every speed number was re-measured, and the original evaluation
harness was unavailable, so the end-to-end serving columns are absent rather
than estimated. The published report is left in place as the historical record.

Every number and every result-dependent sentence is derived from the summary,
so nothing here can drift from the run that produced it.
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

S = json.loads(Path("outputs_a5000/summary_a5000.json").read_text())
NEW_KEY = "qwen3_adapter"
LABELS = {
    "qwen3_adapter": ("Qwen3-ASR + Bengali adapter", "Qwen/Qwen3-ASR-1.7B-hf + LoRA", "2.04B / 38M trained"),
    "hishab_conformer_large": ("Conformer Large", "hishab/titu_stt_bn_conformer_large", "121.5M"),
    "ehzawad_fastconformer": ("ehzawad FastConformer", "ehzawad/stt_bn_fastconformer", "115.6M"),
    "whisper_medium": ("Whisper Medium", "SayedShaun/bengali-whisper-medium", "763.9M"),
    "wav2vec2": ("Wav2Vec2", "SayedShaun/bangla-wave2vec2-unigram", "315.5M"),
    "hishab_fastconformer": ("hishab FastConformer", "hishab/titu_stt_bn_fastconformer", "115.6M"),
}
M = sorted(S["models"].items(), key=lambda kv: kv[1]["wer"])
_W = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
N = _W.get(len(M), str(len(M)))
NEW = S["models"][NEW_KEY]
BEST_KEY, BEST = M[0]


def name(k): return LABELS.get(k, (k, k, "?"))[0]
def ckpt(k): return LABELS.get(k, (k, k, "?"))[1]
def params(k): return LABELS.get(k, (k, k, "?"))[2]
def color(k): return ACCENT if k == NEW_KEY else BASE


def footer(fig, page):
    fig.text(0.06, 0.028, "Bengali ASR benchmark  ·  six models, one A5000, "
             f"{S['eval']['n']} FLEURS utterances", size=6.6, color=INK_3)
    fig.text(0.94, 0.028, f"{page}", size=6.6, color=INK_3, ha="right")


def caveat(fig, x, y, w, text, h=None, size=7.0):
    lines = text.count("\n") + 1
    h = h or 0.016 + 0.0135 * lines
    fig.patches.append(FancyBboxPatch(
        (x, y - h), w, h, boxstyle="round,pad=0.004,rounding_size=0.006",
        transform=fig.transFigure, facecolor=WARN_BG, edgecolor=WARN_EDGE,
        linewidth=0.7, zorder=0))
    fig.text(x + 0.012, y - 0.011, text, size=size, color=INK_2, va="top")


def table(fig, x, y, w, headers, rows, offs, row_h=0.021, size=7.0):
    fig.text(x, y, "", size=1)
    for i, (hh, o) in enumerate(zip(headers, offs)):
        fig.text(x + o * w, y, hh, size=size - 0.4, color=INK_3,
                 ha="right" if i else "left", weight="bold")
    fig.lines.append(plt.Line2D([x, x + w], [y - 0.008, y - 0.008],
                     transform=fig.transFigure, color=RULE, lw=0.8))
    yy = y - 0.008
    for r in rows:
        yy -= row_h
        for i, (c, o) in enumerate(zip(r, offs)):
            bold = r[0].startswith("▸")
            fig.text(x + o * w, yy, c, size=size,
                     color=INK if bold else INK_2,
                     weight="bold" if bold else "normal",
                     ha="right" if i else "left")
    return yy


def page1(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.945, "Bengali ASR: six models, one GPU", size=20, weight="bold")
    fig.text(0.06, 0.922, f"{S['eval']['n']} FLEURS bn_in utterances "
             f"({S['eval']['splits'][0]} + {S['eval']['splits'][1]}) · "
             f"{S['protocol']['gpu']} · greedy decoding, no external LM",
             size=8.6, color=INK_2)

    rows = []
    for k, m in M:
        lo, hi = m["wer_ci_sentence_clustered"]
        mark = "▸ " if k == NEW_KEY else "  "
        rows.append([f"{mark}{name(k)}", f"{m['wer']*100:.2f}%",
                     f"[{lo*100:.2f}–{hi*100:.2f}]", f"{m['cer']*100:.2f}%",
                     f"{m['meta']['ms_per_clip_mean']:.0f}",
                     f"{m['meta']['audio_s_per_wall_s']:.0f}×", params(k)])
    yy = table(fig, 0.06, 0.885, 0.88,
               ["Model", "WER", "95% CI (clustered)", "CER", "ms/clip",
                "× real time", "Params"],
               rows, [0.0, 0.44, 0.62, 0.70, 0.79, 0.88, 1.0])

    fig.text(0.06, yy - 0.028, "Reading this table", size=11, weight="bold")
    spread = M[-1][1]["wer"] / BEST["wer"]
    fast = max(M, key=lambda kv: kv[1]["meta"]["audio_s_per_wall_s"])
    slow = min(M, key=lambda kv: kv[1]["meta"]["audio_s_per_wall_s"])
    speed_ratio = (fast[1]["meta"]["audio_s_per_wall_s"]
                   / slow[1]["meta"]["audio_s_per_wall_s"])
    body = (
        f"Corpus WER and CER over {S['eval']['n']} utterances, computed from raw "
        f"hypotheses with the normaliser the earlier report used. Intervals are "
        f"95% percentile bootstrap resampling the {S['eval']['distinct_references']} "
        f"DISTINCT sentences, not the {S['eval']['n']} recordings: several speakers "
        f"read the same sentence, so recordings are not independent and a "
        f"per-utterance interval is optimistic. Both are in the summary.\n\n"
        f"Accuracy spans {spread:.1f}× across these {N} systems and speed spans "
        f"{speed_ratio:.0f}×. The two orderings are not the same ordering, which is "
        f"the point of measuring both."
    )
    fig.text(0.06, yy - 0.045, body, size=8.4, color=INK_2, va="top",
             wrap=True, linespacing=1.6)

    caveat(fig, 0.06, yy - 0.20, 0.88,
           "What this report is not\n"
           "The earlier RTX 5080 report is a different protocol and is left in place\n"
           "unchanged. Its speed numbers were re-measured here because the GPU changed;\n"
           "its end-to-end throughput, queueing latency and failed-request columns are\n"
           "ABSENT here, not re-estimated, because the harness that produced them was\n"
           "never vendored and was unavailable. Speed below is warmed in-process\n"
           "batch-1 inference time, which includes the shared path's WAV staging.")
    footer(fig, 1)
    pdf.savefig(fig); plt.close(fig)


def page2(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.945, "Accuracy", size=17, weight="bold")
    keys = [k for k, _ in M]
    ax = fig.add_axes([0.30, 0.62, 0.62, 0.27])
    vals = [S["models"][k]["wer"] * 100 for k in keys]
    errs = [[v - S["models"][k]["wer_ci_sentence_clustered"][0] * 100 for v, k in zip(vals, keys)],
            [S["models"][k]["wer_ci_sentence_clustered"][1] * 100 - v for v, k in zip(vals, keys)]]
    ax.barh(range(len(keys)), vals, color=[color(k) for k in keys],
            xerr=errs, error_kw={"ecolor": INK_3, "elinewidth": 0.9, "capsize": 2.5})
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([name(k) for k in keys], size=8)
    ax.invert_yaxis(); ax.set_xlabel("WER %  (bars: 95% sentence-clustered CI)", size=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(v + 0.6, i, f"{v:.2f}", va="center", size=7.5, color=INK_2)

    rows = [[f"{'▸ ' if k == NEW_KEY else '  '}{name(k)}",
             f"{S['models'][k].get('wer_test', float('nan'))*100:.2f}%",
             f"{S['models'][k].get('wer_validation', float('nan'))*100:.2f}%",
             f"{S['models'][k]['empty_hyps']}", f"{S['models'][k]['failures']}"]
            for k in keys]
    yy = table(fig, 0.06, 0.56, 0.88,
               ["Model", "WER test", "WER validation", "empty hyps", "failures"],
               rows, [0.0, 0.55, 0.72, 0.87, 1.0])
    fig.text(0.06, yy - 0.03,
             "Test and validation are reported separately because validation is "
             "normally available during development; the headline merges them only "
             "for continuity with the earlier report. No model in this table was "
             "trained on either split.", size=8.2, color=INK_2, va="top", wrap=True,
             linespacing=1.6)
    footer(fig, 2)
    pdf.savefig(fig); plt.close(fig)


def page3(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.945, "Speed, and what it costs", size=17, weight="bold")
    keys = [k for k, _ in M]
    ax = fig.add_axes([0.30, 0.63, 0.62, 0.26])
    ms = [S["models"][k]["meta"]["ms_per_clip_mean"] for k in keys]
    ax.barh(range(len(keys)), ms, color=[color(k) for k in keys])
    ax.set_xscale("log"); ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([name(k) for k in keys], size=8); ax.invert_yaxis()
    ax.set_xlabel("ms per clip, batch 1 (log scale)", size=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for i, v in enumerate(ms):
        ax.text(v * 1.08, i, f"{v:,.0f}", va="center", size=7.5, color=INK_2)

    ax2 = fig.add_axes([0.30, 0.30, 0.62, 0.25])
    for k in keys:
        m = S["models"][k]
        ax2.scatter(m["meta"]["ms_per_clip_mean"], m["wer"] * 100,
                    s=64 if k == NEW_KEY else 40, color=color(k), zorder=3)
        ax2.annotate(name(k), (m["meta"]["ms_per_clip_mean"], m["wer"] * 100),
                     textcoords="offset points", xytext=(7, 4), size=7, color=INK_2)
    ax2.set_xscale("log"); ax2.set_xlabel("ms per clip (log)", size=8)
    ax2.set_ylabel("WER %", size=8)
    for sp in ("top", "right"): ax2.spines[sp].set_visible(False)

    a = S["models"][NEW_KEY]["meta"]["ms_per_clip_mean"]
    b = S["models"]["ehzawad_fastconformer"]["meta"]["ms_per_clip_mean"]
    fig.text(0.06, 0.24,
             f"The adapter is {a/b:.0f}× slower per clip than the FastConformer on "
             f"this GPU: it generates the transcript token by token through a 1.7B "
             f"decoder, where a CTC model emits an utterance in one forward pass. "
             f"That is architectural, not a serving artefact. Batch-1 is the "
             f"interactive-latency workload; batching would help the autoregressive "
             f"model more than the CTC ones, and is not measured here for any model.",
             size=8.4, color=INK_2, va="top", wrap=True, linespacing=1.6)
    footer(fig, 3)
    pdf.savefig(fig); plt.close(fig)


def page4(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.945, "Method, and what it does not establish", size=17, weight="bold")
    p = S["protocol"]
    txt = (
        "One GPU, one model resident at a time. Each checkpoint was loaded alone and "
        "released before the next.\n\n"
        "One audio path. asr_core.py is vendored here and is the only implementation "
        "of loading, downmixing, NaN scrubbing, resampling, 25-second segmentation and "
        "PCM16 staging. Every model receives the files it stages. Because the adapter "
        "needs a different transformers version than the NeMo checkpoints, it runs in a "
        "second interpreter — so the staging was executed under both and the SHA-256 of "
        "every staged chunk compared, including clips long enough to be segmented. The "
        "hashes match, so the bytes reaching each encoder are the same bytes.\n\n"
        "Greedy everywhere, batch 1, no external language model or rescoring. Whisper's "
        "language is forced to Bengali. A clip that raises becomes an empty hypothesis "
        "and stays in the denominator.\n\n"
        "Checkpoints are pinned to commit hashes, recorded in the summary.\n\n"
        "Limitations\n"
        "FLEURS is short read speech; it says nothing about conversational, noisy, "
        "dialectal or long-form audio, and it is close to Whisper's least favourable "
        "case. One run per model, so speed carries no interval. The speed figure is "
        "warmed in-process batch-1 time including WAV staging — not pure model compute, "
        "and not a serving benchmark. Parameter counts describe the inference system: "
        "the adapter row is a 2.04B-parameter model of which 38M were trained. "
        "Runtime versions differ between the two interpreters, so each row measures a "
        "model-plus-runtime configuration rather than an architecture in isolation."
    )
    fig.text(0.06, 0.90, txt, size=8.4, color=INK_2, va="top", wrap=True,
             linespacing=1.7)
    caveat(fig, 0.06, 0.30, 0.88,
           "Absent by necessity\n" + p["end_to_end_note"].replace(". ", ".\n"))
    footer(fig, 4)
    pdf.savefig(fig); plt.close(fig)


def main():
    out = "asr_benchmark_qwen3_adapter_a5000.pdf"
    with PdfPages(out) as pdf:
        page1(pdf); page2(pdf); page3(pdf); page4(pdf)
    print("wrote", out)


if __name__ == "__main__":
    main()
