# Bengali ASR benchmark

A controlled comparison of four public Bengali speech-to-text models and one private fine-tune on
seven evaluation sets: the FLEURS Bengali test split, and 1,000-utterance samples of the official
test splits of Vaani, SPRING-INX R1 and R2, Kathbath (known- and unknown-speaker) and the
IndicVoices conversational validation split. Every decode ran on one GPU (a Colab A100-SXM4-40GB),
one model at a time, greedy at batch 1, through the same audio path and the same scorer, so
accuracy and speed are comparable across the tables.

The private model, **Voxtral Mini bn-BD** (marked ‡), went through the same harness, but was decoded
by vLLM rather than by PyTorch directly.

Report: [`asr_benchmark.pdf`](asr_benchmark.pdf), rendered by `make_report.py` from the committed summaries.

## Which one to use

- **`ehzawad/stt_bn_fastconformer_ctc`**: the most accurate public model on the corpora none of
  these models is documented as having trained on (the best public model on Vaani and on both
  SPRING sets), with 17.1 % WER on FLEURS at 88 ms per clip.
- **`hishab/titu_stt_bn_conformer_large`**: best on FLEURS (14.4 %) and on Kathbath, i.e. on read
  speech, at the same latency. But it is **last on Vaani** (33.1 %): its FLEURS lead does not
  transfer to spontaneous speech.
- **Whisper Medium**: second of the four public models on all seven sets, at 11–22× the latency of
  the NeMo CTC models. The most consistent choice if speed does not matter.
- **`ehzawad/Voxtral-Mini-3B-2507-bn-BD`** ‡: private, so not a choice for most readers. It has the
  lowest WER of all five on Vaani (22.1 %) and SPRING R1 (36.9 %), is indistinguishable from Conformer
  Large on FLEURS (14.7 %) and is behind Conformer Large and Whisper Medium on Kathbath, at 3–5× the
  latency of the NeMo CTC models.

The one finding worth reading on for: **on the corpora none of these models is documented as having
trained on, the FLEURS ranking does not hold.**

## Official test splits beyond FLEURS

Vaani (ARTPARK-IISc, spontaneous image-description speech from Indian districts) and SPRING-INX
R1/R2 (IIT Madras, conversational; about half the utterances contain Latin-script words) are not
listed as training data by any of these models. No model card names them, and for three of the
five the card does not settle it either way. Kathbath (AI4Bharat IndicSUPERB, read speech) and
IndicVoices (AI4Bharat, conversational and extempore speech) are corpora some of the models
trained another split of.

Each set is a seeded, stratified 1,000-utterance sample of the official split (1.3–2.6 h of
audio per set); manifests with per-file SHA-256 are under [`multiset/`](multiset/), predictions
under [`outputs_multiset/`](outputs_multiset/).

**WER on the 1,000-utterance official-test subsets** (best per column in bold; 95 % intervals in brackets):

| Model | Vaani test | SPRING-INX R1 eval | SPRING-INX R2 eval | Kathbath (known spk) | Kathbath (unknown spk) | IndicVoices conversational |
|---|---:|---:|---:|---:|---:|---:|
| **ehzawad FastConformer-CTC** | 24.67% [23.5–25.9] <sub>N</sub> | 39.23% [38.0–40.5] <sub>N</sub> | **38.42%** [37.2–39.7] <sub>N</sub> | 13.38% [12.6–14.2] <sub>S</sub> | 13.43% [12.6–14.3] <sub>S</sub> | **12.61%** [11.6–13.7] <sub>S</sub> |
| Whisper Medium | 26.27% [25.0–27.5] <sub>U</sub> | 39.97% [38.8–41.2] <sub>U</sub> | 39.70% [38.5–40.9] <sub>U</sub> | 9.69% [9.0–10.4] <sub>U</sub> | 9.86% [9.2–10.6] <sub>U</sub> | 25.00% [23.7–26.4] <sub>U</sub> |
| Conformer Large (hishab) | 33.10% [31.8–34.5] <sub>N</sub> | 42.46% [41.2–43.8] <sub>N</sub> | 42.25% [40.9–43.6] <sub>N</sub> | **9.33%** [8.6–10.1] <sub>S</sub> | **8.79%** [8.1–9.5] <sub>S</sub> | 29.68% [28.3–31.2] <sub>N</sub> |
| Wav2Vec2 | 31.28% [29.9–32.7] <sub>U</sub> | 47.28% [46.1–48.6] <sub>U</sub> | 47.32% [46.1–48.5] <sub>U</sub> | 19.07% [18.1–20.0] <sub>U</sub> | 18.80% [17.9–19.7] <sub>U</sub> | 33.02% [31.6–34.5] <sub>U</sub> |
| Voxtral Mini bn-BD ‡ | **22.08%** [20.9–23.2] <sub>U</sub> | **36.87%** [35.7–38.2] <sub>U</sub> | 40.01% [36.6–44.3] <sub>U</sub> | 10.77% [10.0–11.5] <sub>U</sub> | 10.82% [10.1–11.6] <sub>U</sub> | 13.40% [12.5–14.3] <sub>U</sub> |

Provenance flags, from the model cards only: **N**, the card does not list this corpus;
**S**, the model trained on another split of this corpus; **U**, the card does not say either
way. None of the three means "verified clean".

‡ Voxtral Mini bn-BD (4.7B parameters) is a private fine-tune of `mistralai/Voxtral-Mini-3B-2507`
whose card documents no training data, hence **U** everywhere. Its rows come from the same harness,
manifests, `asr_core` path and scorer as the others, decoded at batch 1 by vLLM 0.31 (the
`voxtral_vllm` backend of `bench_run.py`).

**What the table says.**

- The FLEURS ranking does not transfer. Conformer Large, the FLEURS leader at 14.4 %, is last on
  Vaani (33.1 %) and fourth on both SPRING sets. [Its model card](https://huggingface.co/hishab/titu_stt_bn_conformer_large)
  lists read and broadcast corpora (OpenSLR, Bengali.AI, MadASR, Kathbath, Common Voice, FLEURS,
  SUBAK.KO, Shrutilipi, Vasha-Bichita, IndicTTS) and none of Vaani, SPRING-INX or IndicVoices.
- On the three sets nobody is documented as having trained on, `ehzawad/stt_bn_fastconformer_ctc`
  leads the four public models. Whisper Medium is second on each, at 11–19× the latency of the
  NeMo CTC models.
- **Voxtral Mini bn-BD ‡** is lower still on Vaani and SPRING R1. Against the ehzawad model, the
  best public one there, it is −2.59 points [−3.24, −1.95] and −2.35 [−2.96, −1.76] (paired, 95 %
  reference-clustered bootstrap, `outputs_multiset/summary_multiset.json`).
  - It is higher than Conformer Large on both Kathbath sets (+1.44 [+0.77, +2.12] and
    +2.03 [+1.37, +2.71]), and higher than Whisper Medium there too (+1.08 [+0.43, +1.74] and
    +0.96 [+0.31, +1.59]).
  - Against the ehzawad model it is inconclusive on SPRING R2 and on IndicVoices.
  - Three SPRING R2 recordings ran to its 500-token cap, repeating one word. They hold 663 of that
    set's 8,143 word errors and stay in its score. On the other 997 it reads 36.82 %, a
    conditional diagnostic.
  - Its training data is undocumented, so none of this rules out overlap.
- On Kathbath the order flips back. Conformer Large (9.3 / 8.8 %) and Whisper Medium (9.7 / 9.9 %)
  lead, Voxtral Mini follows (10.8 / 10.8 %) and the ehzawad model is fourth (13.4 / 13.4 %).
  Conformer Large and the ehzawad model are **S** there: Conformer Large's card lists Kathbath train
  and validation, and the ehzawad model trained on Kathbath train. The other three are **U**, and a
  **U** cell can hide training overlap. Known- and unknown-speaker results differ by under a point
  for every model.
- **IndicVoices conversational is not a clean head-to-head for the ehzawad model.** It trained on
  the IndicVoices *train* split: different speakers, but the same collection, prompts and
  transcription conventions. That is a large part of why it reads 12.6 % where the other public
  models read 25–33 %. Read their cells as "the others, on Indian conversational speech they never
  saw".
  - Conformer Large (29.7 %) is the one **N** cell there. It emitted 7 empty hypotheses on this set
    and Wav2Vec2 emitted 33, all counted as full errors.
  - Voxtral Mini reads 13.4 %, inconclusive against the ehzawad model (+0.79 [−0.04, +1.56]).
    Whether it saw this collection is unknown.
- SPRING's 37–47 % WER overstates the real error rate. About half its references contain English
  words in Latin script, and no model emitted a single Latin-script word, so that content is an
  error for all five alike. On the Latin-free half, `ehzawad/stt_bn_fastconformer_ctc`
  (35.4 / 35.1 %) and Whisper Medium (36.0 / 36.6 %) are inside each other's intervals. Voxtral
  Mini reads 32.2 / 37.7 %.

## FLEURS Bengali test (920 utterances)

The familiar benchmark, for comparison with published numbers.

| Model | Checkpoint | WER (95% CI) | CER | ms/clip | Params |
|---|---|---|---|---|---|
| Conformer Large | `hishab/titu_stt_bn_conformer_large` | **14.37%** [13.11–15.73] | 4.48% | 87 | 121.5M |
| Voxtral Mini bn-BD ‡ | `ehzawad/Voxtral-Mini-3B-2507-bn-BD` (private) | 14.73% [13.60–15.93] | 4.78% | 438 | 4,676M |
| Whisper Medium | `SayedShaun/bengali-whisper-medium` | 16.22% [15.04–17.45] | 5.09% | 1,901 | 763.9M |
| **ehzawad FastConformer-CTC** | `ehzawad/stt_bn_fastconformer_ctc` | 17.12% [15.95–18.34] | 5.12% | 88 | 115.6M |
| Wav2Vec2 | `SayedShaun/bangla-wave2vec2-unigram` | 20.66% [19.47–21.90] | 6.02% | 56 | 315.5M |

Greedy decoding, batch 1, no external language model. No clip failed and no hypothesis came back
empty for any model. `ms/clip` is the mean single-clip latency on the A100:

| Model | mean ms/clip | median | p95 | × real time |
|---|---:|---:|---:|---:|
| Wav2Vec2 | 56 | 54 | 86 | 238× |
| Conformer Large | 87 | 83 | 96 | 155× |
| ehzawad FastConformer-CTC | 88 | 84 | 98 | 153× |
| Voxtral Mini bn-BD ‡ | 438 | 420 | 680 | 31× |
| Whisper Medium | 1,901 | 1,789 | 3,018 | 7× |

These numbers time the configured pipelines, not the models in isolation:

- The public models ran through `bench_run.py`'s PyTorch backends (torch 2.11; Whisper in FP16).
- Voxtral Mini ran through vLLM 0.31 with CUDA graphs (torch 2.13).
- A few milliseconds apart is not an established ordering, and p95 is variation across clips, not
  uncertainty in the mean.

**Test split only.** The ehzawad model trained on a corpus that includes the FLEURS *train* split,
which overlaps the FLEURS *validation* split, so the 402 validation recordings are excluded for
every model. It had also been evaluated on FLEURS test during its own development, so read its
FLEURS row as a like-for-like re-measurement and the other corpora above as the fresh evidence.

Paired comparisons for `ehzawad/stt_bn_fastconformer_ctc` on this split: +0.90 points behind
Whisper Medium [+0.28, +1.51] at 22× its speed; +2.74 points behind Conformer Large [+2.01, +3.46].

Voxtral Mini bn-BD ‡ on this split:

- Against Conformer Large: +0.35 points [−0.36, +1.06], not distinguishable.
- Against Whisper Medium: −1.49 [−2.13, −0.86].
- Against the ehzawad model: −2.39 [−3.03, −1.74].
- Its CER is higher than Conformer Large's: 4.78 % vs 4.48 %, with an interval that excludes zero.
- Like Conformer Large and Whisper Medium, it writes numbers as Bengali words. The scorer counts
  these as errors against the 187 references that use digits.

## How the comparison was run

- **One GPU, one model resident at a time**, a Colab A100-SXM4-40GB, on 2026-10-13. The
  official-split runner refuses to start while another compute process holds the card. Every run
  records the card's state before and after. `run_all.sh` lists every decode behind the committed
  outputs.
- **One audio path.** `asr_core.py` loads, downmixes and resamples to 16 kHz, cuts the audio at
  pauses into pieces of at most 25 s and stages them as PCM16; every backend reads those staged
  files.
  - Voxtral Mini runs in its own Python environment (vLLM). Its piece counts and forced cuts match
    every other model's on all 7,322 clips.
- **One scorer.** `bench_score.norm` strips punctuation to word boundaries and applies Unicode NFC
  before WER/CER. NFC is not cosmetic: Bengali writes য়, ড় and ঢ় either precomposed or as
  base + nukta.
  - The FLEURS references and the NeMo models use the decomposed form; Whisper and wav2vec2 use the
    precomposed one.
  - Without NFC, every such character scores as a substitution: Whisper Medium reads 27.9 %
    instead of 16.2 %.
  - If your own measurement of these models disagrees with a published number by ten points or
    more, this is usually why.
- **Greedy decoding, batch 1, no external LM** for every model. This is the interactive-latency
  workload, applied identically to CTC, encoder-decoder and audio-LLM decoders. Batching would help
  Whisper and Voxtral Mini far more; it is not measured here for anyone.
- **Timing.** One warm-up clip is discarded. Each clip is then timed between CUDA synchronisations,
  with `asr_core` loading, chunking and staging included and model startup excluded.
  - Runs were niced.
  - Other activity on the VM during the runs was limited to brief checks over SSH: log tails, a few
    seconds of scoring, a Google Drive mount during Whisper Medium's FLEURS run and a workspace
    setup during Voxtral Mini's. Their effect was not measured.
- **Every clip counts.** No model raised on a clip. Empty hypotheses stay in the denominator as full
  errors: three on SPRING R2, and on IndicVoices conversational 7 from Conformer Large and 33 from
  Wav2Vec2.
- **Checkpoints pinned** by commit hash in `outputs_fleurs/checkpoint_revisions.json`.
- **Voxtral Mini bn-BD ‡** is decoded one piece per request. It uses the prompt that vLLM's own
  transcription endpoint builds for Voxtral (language `bn`), greedy, with at most 500 new tokens per
  piece.
  - Against Hugging Face transformers at batch 1 on six FLEURS clips, the text was identical on all
    six.
  - An earlier decode of the same model in continuous vLLM batches gave the same transcript on 6,445
    of 6,920 test clips, with WER within 0.12 points on every set.
- **Intervals** are 95 % bootstraps over distinct references, because several speakers may read
  the same sentence. That matters on FLEURS (349 distinct sentences over 920 recordings). In the
  1,000-utterance subsets the references are nearly all distinct: 999 or 1,000 per set, and 871 on
  IndicVoices conversational, whose short replies repeat. There the two kinds of interval barely
  differ.

## Limitations

- FLEURS and Kathbath are read speech; Vaani, SPRING and IndicVoices conversational are
  spontaneous. There is no telephony, no far-field audio, no noise condition and no long-form
  recording: no public, human-transcribed, licensed long-form Bengali test set exists.
- Speed is per-clip latency at batch 1 on one GPU model, from one pass per model and set.
  Throughput under load, batched decoding and CPU inference are not measured.
- Cell intervals on the 1,000-utterance subsets run from about ±0.7 points (Kathbath) to ±1.5
  points (Vaani, SPRING, IndicVoices). Voxtral Mini's SPRING R2 cell is wider (−3.4 / +4.3) because
  of its three capped recordings. Differences inside an interval are not differences.
- Training provenance is only as good as the model cards; a **U** flag means it could not be
  established either way.
- Voxtral Mini is not public. It runs on a different inference engine (vLLM) from the public
  models, so its latency compares configured pipelines rather than architectures alone.

## Reproducing this

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
python build_eval.py                              # FLEURS bn_in test + validation -> eval_manifest.json, eval_fleurs_bn/
# every model on FLEURS and the six subsets, one at a time -> outputs_fleurs/, outputs_multiset/
BENCH_GPU=<GPU UUID> BENCH_PY=venv/bin/python FLEURS_ASSETS=. MULTISET_ASSETS=<subset audio dir> ./run_all.sh
python bench_score.py && python clean_split_table.py
python score_multiset.py
python make_report.py                             # -> asr_benchmark.pdf
```

- `MULTISET_ASSETS/<set>/eval_manifest.local.json` lists the same rows as
  `multiset/<set>/eval_manifest.json`, with local paths. `bench_run_set.py` checks every file's
  SHA-256 before decoding.
- Voxtral Mini bn-BD is private. It also needs `BENCH_PY_VLLM=<python with vllm[audio]==0.31.0>`
  and `VOXTRAL_MINI=<local snapshot of the model>`; without them `run_all.sh` skips it.
- `run_all.sh` skips a run whose `run_meta.json` exists and never overwrites one.

Subset sampling: seed 20260915, proportional stratification (Vaani by state/district, SPRING by
Latin/no-Latin reference, Kathbath by speaker gender, IndicVoices by scenario). The Vaani,
SPRING-INX, Kathbath and IndicVoices audio is not redistributed here; `multiset/*/eval_manifest.json`
lists every selected utterance with its SHA-256 so a rebuild can be verified.

## The web demo

`app.py` serves `ehzawad/stt_bn_fastconformer_ctc` behind a Gradio UI: record from the
microphone or upload a file, get Bengali text back. It shares `asr_core.py` with the benchmark,
so the demo and the measured numbers use the same audio handling.

```bash
./run.sh --daemon    # background, logs to service.log
./run.sh --stop
```

- Serves over HTTPS when `certs/cert.pem` and `certs/key.pem` exist. Browsers block microphone
  access on plain HTTP except on `localhost`.
- There is no authentication. Anyone who can reach the port can use it.

## Files

```
asr_benchmark.pdf        the report
asr_core.py              the single audio path shared by demo and benchmark
backends.py              model loading for NeMo, wav2vec2 CTC and Whisper seq2seq
bench_run.py             one model over the FLEURS set, batch 1, timed; its backends include Voxtral under vLLM
bench_run_set.py         the same backends over any eval manifest (the official-split runs)
bench_score.py           predictions -> WER/CER with bootstrap intervals -> summary.json
clean_split_table.py     the FLEURS test-split table with paired deltas
score_multiset.py        the official-split tables, and Voxtral Mini's paired comparisons
run_all.sh               every run, one model at a time, GPU pinned by UUID
make_report.py           summaries -> asr_benchmark.pdf
bench_server.py          HTTP wrapper to run any of these checkpoints
app.py, run.sh           the interactive web demo
build_eval.py, prove_audio_path.py, eval_manifest.json   the FLEURS evaluation set
multiset/                the official-split subsets with per-file SHA-256
outputs_fleurs/          FLEURS predictions, run metadata, summaries
outputs_multiset/        official-split predictions, run metadata, summary
requirements.txt         pinned environment for the public models
```

Not included: model checkpoints, the built evaluation sets, and the virtualenvs.

## Attribution and licences

- Evaluation data:
  - [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google), CC-BY-4.0.
  - [Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part) transcription
    part (ARTPARK-IISc), CC-BY-4.0.
  - SPRING-INX R1/R2 (SPRING Lab, IIT Madras), CC-BY-4.0.
  - [Kathbath](https://github.com/AI4Bharat/IndicSUPERB) (AI4Bharat IndicSUPERB), CC0 per its
    release page.
  - [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) (AI4Bharat), CC-BY-4.0.
  - The committed manifests and predictions carry reference transcripts alongside each model's
    hypothesis.
- `hishab/titu_stt_bn_conformer_large`: Hishab.
- `SayedShaun/bengali-whisper-medium` (mirror of tugstugi's Bengali.AI competition model),
  `SayedShaun/bangla-wave2vec2-unigram` (mirror of qdv206's).
- `ehzawad/stt_bn_fastconformer_ctc`: FastConformer-CTC large trained on 1,692 hours of
  human-supervised Bengali with a 1,024-piece Bengali BPE vocabulary, CC-BY-SA-4.0. Base model
  `nvidia/stt_en_fastconformer_ctc_large`, NVIDIA, CC-BY-4.0.
- `ehzawad/Voxtral-Mini-3B-2507-bn-BD`: private fine-tune of `mistralai/Voxtral-Mini-3B-2507`
  (Mistral AI, Apache-2.0).

Model licences are those of their respective publishers. Check each model card before use.

Author: Emrul Zawad (ehzawad).
