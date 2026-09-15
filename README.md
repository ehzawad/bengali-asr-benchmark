# Bengali ASR benchmark

A controlled comparison of five public Bengali speech-to-text models on six official test sets:
the FLEURS Bengali test split, and 1,000-utterance samples of the official test splits of Vaani,
SPRING-INX R1 and R2, Kathbath (known- and unknown-speaker) and the IndicVoices conversational
validation split. Every decode was run on one GPU (RTX A5000), one model at a time, through the
same audio path and the same scorer — with one exception, marked † below — so accuracy and speed
are comparable across the tables.

Report: [`asr_benchmark.pdf`](asr_benchmark.pdf), rendered by `make_report.py` from the committed summaries.

## Which one to use

- **`ehzawad/stt_bn_fastconformer_ctc`** — the most accurate model on the corpora none of these
  models is documented as having trained on (best on Vaani and on both SPRING sets), 17.1 % WER
  on FLEURS, ~80 ms per clip.
- **`hishab/titu_stt_bn_conformer_large`** — best on FLEURS (14.4 %) and on Kathbath, i.e. on read
  speech, at the same latency. But it is **last on Vaani** (33.1 %): its FLEURS lead does not
  transfer to spontaneous speech.
- **Whisper Medium** — second or close to it nearly everywhere, at 13–22× the latency of the CTC
  models. The most consistent choice if speed does not matter.

The one finding worth reading on for: **on the corpora none of these models is documented as having
trained on, the FLEURS ranking does not hold.**

## Official test splits beyond FLEURS

Vaani (ARTPARK-IISc, spontaneous image-description speech from Indian districts) and SPRING-INX
R1/R2 (IIT Madras, conversational; about half the utterances contain Latin-script words) are not
listed as training data by any of these models — no model card names them, and for three of the
five the card does not settle it either way. Kathbath (AI4Bharat IndicSUPERB, read speech) and
IndicVoices (AI4Bharat, conversational and extempore speech) are corpora some of the models
trained another split of.

Each set is a seeded, stratified 1,000-utterance sample of the official split (1.3–2.6 h of
audio per set); manifests with per-file SHA-256 are under [`multiset/`](multiset/), predictions
under [`outputs_multiset/`](outputs_multiset/).

**WER on the 1,000-utterance official-test subsets** (best per column in bold; 95 % intervals in brackets):

| Model | Vaani test | SPRING-INX R1 eval | SPRING-INX R2 eval | Kathbath (known spk) | Kathbath (unknown spk) | IndicVoices conversational |
|---|---:|---:|---:|---:|---:|---:|
| **ehzawad FastConformer-CTC** | **24.66%** [23.5–25.9] <sub>N</sub>† | **39.24%** [38.0–40.5] <sub>N</sub>† | **38.41%** [37.2–39.7] <sub>N</sub>† | 13.38% [12.6–14.2] <sub>S</sub> | 13.43% [12.6–14.3] <sub>S</sub> | **12.61%** [11.6–13.7] <sub>S</sub> |
| Whisper Medium | 26.25% [25.0–27.5] <sub>U</sub> | 39.96% [38.8–41.2] <sub>U</sub> | 39.70% [38.5–40.9] <sub>U</sub> | 9.69% [9.0–10.4] <sub>U</sub> | 9.79% [9.1–10.5] <sub>U</sub> | 25.03% [23.7–26.5] <sub>U</sub> |
| Conformer Large (hishab) | 33.10% [31.8–34.5] <sub>N</sub> | 42.46% [41.2–43.8] <sub>N</sub> | 42.24% [40.9–43.6] <sub>N</sub> | **9.33%** [8.6–10.1] <sub>S</sub> | **8.79%** [8.1–9.5] <sub>S</sub> | 29.68% [28.3–31.2] <sub>N</sub> |
| hishab FastConformer | 28.71% [27.4–30.0] <sub>U</sub> | 46.47% [44.8–48.2] <sub>U</sub> | 45.08% [43.5–46.7] <sub>U</sub> | 25.81% [24.1–27.6] <sub>U</sub> | 29.87% [27.5–32.4] <sub>U</sub> | 30.14% [28.7–31.7] <sub>U</sub> |
| Wav2Vec2 | 31.26% [29.9–32.7] <sub>U</sub> | 47.28% [46.1–48.6] <sub>U</sub> | 47.33% [46.1–48.6] <sub>U</sub> | 19.08% [18.1–20.1] <sub>U</sub> | 18.80% [17.9–19.7] <sub>U</sub> | 33.01% [31.6–34.5] <sub>U</sub> |

Provenance flags, from the model cards only: **N** — the card does not list this corpus;
**S** — the model trained on another split of this corpus; **U** — the card does not say either
way. None of the three means "verified clean".

† These three cells re-use the model's existing decode of the full official splits (same greedy
decoding, batched) rather than decoding them again here; they are scored on exactly these
utterances, so accuracy is comparable, but no latency is reported for them. Every other cell is a
batch-1 decode from this run.

**What the table says.**

- The FLEURS ranking does not transfer. Conformer Large, the FLEURS leader at 14.4 %, is last on
  Vaani (33.1 %) and third on both SPRING sets. [Its model card](https://huggingface.co/hishab/titu_stt_bn_conformer_large)
  lists read and broadcast corpora (OpenSLR, Bengali.AI, MadASR, Kathbath, Common Voice, FLEURS,
  SUBAK.KO, Shrutilipi, Vasha-Bichita, IndicTTS) and none of Vaani, SPRING-INX or IndicVoices.
- On the three sets nobody is documented as having trained on, `ehzawad/stt_bn_fastconformer_ctc`
  leads all five, with Whisper Medium second on each at 13–20× the latency of the NeMo CTC models.
- On Kathbath the order flips back: Conformer Large (9.3 / 8.8 %) and Whisper Medium (9.7 / 9.8 %)
  lead and the ehzawad model is third (13.4 / 13.4 %). Two of those cells are **S** (Conformer
  Large's card lists Kathbath train and validation; the ehzawad model trained on Kathbath train)
  and three are **U**, and a **U** cell can hide training overlap. Known- and unknown-speaker
  results differ by under a point for every model except hishab FastConformer.
- **IndicVoices conversational is not a clean head-to-head for the ehzawad model.** It trained
  on the IndicVoices *train* split — different speakers, but the same collection, prompts and
  transcription conventions — which is a large part of why it reads 12.6 % where the others read
  25–33 %. Read that column as "the others, on Indian conversational speech they never saw".
  Conformer Large (29.7 %) is the one **N** cell there: it emitted 7 empty hypotheses on this set;
  hishab FastConformer 21 and Wav2Vec2 33 (all counted as full errors).
- SPRING's 39–47 % WER overstates the real error rate: about half its references contain English
  words in Latin script, and no model emitted a single Latin-script word, so that content is an
  error for all five alike. On the Latin-free half, `ehzawad/stt_bn_fastconformer_ctc`
  (35.4 / 35.1 %) and Whisper Medium (35.9 / 36.7 %) are inside each other's intervals.

## FLEURS Bengali test (920 utterances)

The familiar benchmark, for comparison with published numbers.

| Model | Checkpoint | WER (95% CI) | CER | ms/clip | Params |
|---|---|---|---|---|---|
| Conformer Large | `hishab/titu_stt_bn_conformer_large` | **14.37%** [13.11–15.73] | 4.48% | 77 | 121.5M |
| Whisper Medium | `SayedShaun/bengali-whisper-medium` | 16.23% [15.05–17.45] | 5.09% | 1,866 | 763.9M |
| **ehzawad FastConformer-CTC** | `ehzawad/stt_bn_fastconformer_ctc` | 17.12% [15.95–18.34] | 5.12% | 83 | 115.6M |
| Wav2Vec2 | `SayedShaun/bangla-wave2vec2-unigram` | 20.64% [19.46–21.89] | 6.02% | 64 | 315.5M |
| hishab FastConformer | `hishab/titu_stt_bn_fastconformer` | 36.22% [33.88–38.72] | 9.05% | 79 | 115.6M |

Greedy decoding, batch 1, no external language model; no clip failed and no hypothesis came
back empty for any model. `ms/clip` is single-clip latency on the A5000.

**Test split only.** The ehzawad model trained on a corpus that includes the FLEURS *train* split,
which overlaps the FLEURS *validation* split, so the 402 validation recordings are excluded for
every model. It had also been evaluated on FLEURS test during its own development, so read its
FLEURS row as a like-for-like re-measurement and the other corpora above as the fresh evidence.

Paired comparisons for `ehzawad/stt_bn_fastconformer_ctc` on this split: +0.89 points behind
Whisper Medium [+0.27, +1.50] at 22× its speed; +2.74 points behind Conformer Large [+2.01, +3.46].

## How the comparison was run

- **One GPU, one model resident at a time**, on the same RTX A5000. The official-split runner
  refuses to start while another compute process holds the card; every run records the card's
  state before and after.
- **One audio path.** `asr_core.py` loads, downmixes and resamples to 16 kHz and hands identical
  bytes to every backend — NeMo, wav2vec2 and Whisper all read the same audio.
- **One scorer.** `bench_score.norm` strips punctuation to word boundaries and applies Unicode NFC
  before WER/CER. NFC is not cosmetic: Bengali writes য়, ড় and ঢ় either precomposed or as
  base + nukta; the FLEURS references and the NeMo models use the decomposed form, Whisper and
  wav2vec2 the precomposed one, and without NFC every such character scores as a substitution
  (Whisper Medium reads 27.9 % instead of 16.2 %). If your own measurement of these models
  disagrees with a published number by ten points or more, this is usually why.
- **Greedy decoding, batch 1, no external LM** for every decode made here — the interactive-latency
  workload, applied identically to CTC and autoregressive decoders. Batching would help Whisper
  far more; it is not measured here for anyone.
- **Every clip counts.** No model raised on a clip. Empty hypotheses (a few on SPRING R2, more on
  IndicVoices conversational for three models) stay in the denominator as full errors.
- **Checkpoints pinned** by commit hash in `outputs_fleurs/checkpoint_revisions.json`.
- **Intervals** are 95 % bootstraps over distinct references, because several speakers may read
  the same sentence. That matters on FLEURS (349 distinct sentences over 920 recordings); in the
  1,000-utterance subsets the references are effectively all distinct.

## Limitations

- FLEURS and Kathbath are read speech; Vaani, SPRING and IndicVoices conversational are
  spontaneous. No telephony, no far-field, no noise conditions, and no long-form recordings — no
  public, human-transcribed, licensed long-form Bengali test set exists.
- Speed is per-clip latency at batch 1 on one GPU model; throughput under load, batched decoding
  and CPU inference are not measured.
- Cell intervals on the 1,000-utterance subsets run from about ±0.7 points (Kathbath) to ±1.4
  points (Vaani, SPRING, IndicVoices), with one cell at ±2.5 points. Differences inside an
  interval are not differences.
- Training provenance is only as good as the model cards; a **U** flag means it could not be
  established either way.

## Reproducing this

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
python build_eval.py                              # FLEURS bn_in test + validation -> eval_manifest.json
export FASTCONFORMER_NEMO=model/stt_bn_fastconformer_ctc.nemo   # from huggingface.co/ehzawad/stt_bn_fastconformer_ctc
BENCH_GPU=<A5000 uuid> ./run_all_fleurs.sh        # five models on FLEURS, one at a time -> outputs_fleurs/
python bench_score.py && python clean_split_table.py
BENCH_GPU=<A5000 uuid> ./run_multiset.sh          # Vaani + SPRING subsets  -> outputs_multiset/
BENCH_GPU=<A5000 uuid> ./run_multiset_kathbath.sh # Kathbath subsets
BENCH_GPU=<A5000 uuid> ./run_multiset_conv.sh     # IndicVoices conversational subset
python score_multiset.py
python make_report.py                             # -> asr_benchmark.pdf
```

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

- Serves over HTTPS when `certs/cert.pem` and `certs/key.pem` exist — browsers block microphone
  access on plain HTTP except on `localhost`.
- There is no authentication. Anyone who can reach the port can use it.

## Files

```
asr_benchmark.pdf        the report
asr_core.py              the single audio path shared by demo and benchmark
backends.py              model loading for NeMo, wav2vec2 CTC and Whisper seq2seq
bench_run.py             one model over the FLEURS set, batch 1, timed
bench_run_set.py         the same backends over any eval manifest (the official-split runs)
bench_score.py           predictions -> WER/CER with bootstrap intervals -> summary.json
clean_split_table.py     the FLEURS test-split table with paired deltas
score_multiset.py        the official-split tables
run_all_fleurs.sh        five models on FLEURS, one at a time, GPU pinned by UUID
run_multiset.sh, run_multiset_kathbath.sh, run_multiset_conv.sh   the official-split runs
make_report.py           summaries -> asr_benchmark.pdf
bench_server.py          HTTP wrapper to run any of these checkpoints
app.py, run.sh           the interactive web demo
build_eval.py, prove_audio_path.py, eval_manifest.json   the FLEURS evaluation set
multiset/                the official-split subsets with per-file SHA-256
outputs_fleurs/          FLEURS predictions, run metadata, summaries
outputs_multiset/        official-split predictions, run metadata, summary
requirements.txt         pinned
```

Not included: model checkpoints, the built evaluation sets, and the virtualenv.

## Attribution and licences

- Evaluation data: [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google), CC-BY-4.0;
  [Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part) transcription part
  (ARTPARK-IISc), CC-BY-4.0; SPRING-INX R1/R2 (SPRING Lab, IIT Madras), CC-BY-4.0;
  [Kathbath](https://github.com/AI4Bharat/IndicSUPERB) (AI4Bharat IndicSUPERB), CC0 per its
  release page; [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) (AI4Bharat),
  CC-BY-4.0. The committed manifests and predictions carry reference transcripts alongside each
  model's hypothesis.
- `hishab/titu_stt_bn_conformer_large`, `hishab/titu_stt_bn_fastconformer` — Hishab.
- `SayedShaun/bengali-whisper-medium` (mirror of tugstugi's Bengali.AI competition model),
  `SayedShaun/bangla-wave2vec2-unigram` (mirror of qdv206's).
- `ehzawad/stt_bn_fastconformer_ctc` — FastConformer-CTC large trained on 1,692 hours of
  human-supervised Bengali with a 1,024-piece Bengali BPE vocabulary, CC-BY-SA-4.0; base model
  `nvidia/stt_en_fastconformer_ctc_large`, NVIDIA, CC-BY-4.0.

Model licences are those of their respective publishers. Check each model card before use.

Author: Emrul Zawad (ehzawad).
