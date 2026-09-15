# Bengali ASR benchmark

A controlled comparison of public Bengali speech-to-text models: six models on the
FLEURS Bengali test split, and five of them on the official test splits of four
other Bengali corpora (Vaani, SPRING-INX R1 and R2, Kathbath). Every model was
measured in one sitting on one GPU (RTX A5000), loaded one at a time, through the
same evaluation harness, the same audio path and the same scorer, so accuracy and
speed are comparable across the whole table. Plus the web demo used to try one of
the models interactively.

Report: [`asr_benchmark.pdf`](asr_benchmark.pdf) (rendered by `make_report.py` from
the committed summaries).

## FLEURS Bengali test (920 utterances)

| Model | Checkpoint | WER (95% CI) | CER | ms/clip | Params |
|---|---|---|---|---|---|
| Conformer Large | `hishab/titu_stt_bn_conformer_large` | **14.37%** [13.11–15.73] | 4.48% | 77 | 121.5M |
| Whisper Medium | `SayedShaun/bengali-whisper-medium` | 16.23% [15.05–17.45] | 5.09% | 1,866 | 763.9M |
| Qwen3-ASR + Bengali adapter | `ehzawad/stt_bn_qwen3_asr` on `Qwen/Qwen3-ASR-1.7B-hf` | 16.54% [15.43–17.68] | 4.68% | 9,728 | 2.04B (38M adapter) |
| **ehzawad FastConformer-CTC** | `ehzawad/stt_bn_fastconformer_ctc` | 17.12% [15.95–18.34] | 5.12% | 83 | 115.6M |
| Wav2Vec2 | `SayedShaun/bangla-wave2vec2-unigram` | 20.64% [19.46–21.89] | 6.02% | 64 | 315.5M |
| hishab FastConformer | `hishab/titu_stt_bn_fastconformer` | 36.22% [33.88–38.72] | 9.05% | 79 | 115.6M |

Corpus WER/CER on the 920-utterance **test** split, greedy decoding, batch 1, no
external language model, zero failed clips and zero empty hypotheses for any model.
Intervals are 95% percentile bootstraps over 10,000 resamples of the 349 distinct
reference sentences (several speakers read the same sentence, so recordings are not
independent). `ms/clip` is warmed in-process batch-1 inference time on the A5000,
including the shared audio path's WAV staging. The Qwen3 row's parameter count reads
"2.04B (38M adapter)": the base model has 2.04 billion parameters and only the 38
million LoRA-adapter parameters were trained for Bengali.

**Test split only.** The two `ehzawad` models were trained on corpora that include
the FLEURS *train* split, and 88 of the 150 distinct sentences in the FLEURS
*validation* split also occur there. The 402 validation recordings are therefore
excluded for every model. The 1,322-utterance figures (test + validation) are in
`outputs_fleurs/summary.json` for completeness.

Paired sentence-clustered deltas for `ehzawad/stt_bn_fastconformer_ctc` (10,000
draws): vs the Qwen3 adapter +0.58 pp [−0.25, +1.40] — not distinguishable, at 120×
its speed; vs Whisper Medium +0.89 pp [+0.27, +1.50] at 22× its speed; vs Conformer
Large +2.74 pp [+2.01, +3.46]. For the adapter vs Conformer Large: +2.16 pp
[+1.14, +3.17] on words, level on characters.

Reproduce: `python bench_score.py` and `python clean_split_table.py` over the
committed predictions in `outputs_fleurs/`.

## Official test splits of four other Bengali corpora

FLEURS is one read-speech test set that several of these models trained near (its
train split). This section compares the models on the **official test splits** of
corpora that, per every model card, none of them trained on — Vaani (ARTPARK-IISc,
spontaneous image-description speech from Indian districts) and SPRING-INX R1 and R2
(IIT Madras, conversational, about half the utterances contain Latin-script words) —
plus **Kathbath** (AI4Bharat IndicSUPERB, read speech; known- and unknown-speaker
test splits), which the ehzawad FastConformer and Conformer Large trained *near*
(Kathbath train, and for Conformer Large its validation split too).

Every set is a **frozen, seeded 1,000-utterance subset** of the official split
(seed 20260915; proportional stratification — Vaani by state/district, SPRING by
Latin/no-Latin reference, Kathbath by speaker gender), decoded one model at a time
through the unchanged `bench_run.py` backends ([`bench_run_set.py`](bench_run_set.py))
and scored with [`bench_score.py`](bench_score.py)'s normaliser, sentence-clustered
95% intervals. Manifests with per-file SHA-256 are under [`multiset/`](multiset/);
raw predictions and per-run GPU state under [`outputs_multiset/`](outputs_multiset/).
The Qwen3 adapter is not in this comparison (~10 s per clip at batch 1).

| Set | official rows | sampled | audio | ref. words |
|---|---:|---:|---:|---:|
| Vaani test | 12,695 | 1,000 | 1.26 h | 10,082 |
| SPRING-INX R1 eval | 1,879 | 1,000 | 2.54 h | 19,706 |
| SPRING-INX R2 eval | 1,872 | 1,000 | 2.61 h | 20,353 |
| Kathbath test (known speakers) | 2,806 | 1,000 | 1.79 h | 10,801 |
| Kathbath test (unknown speakers) | 1,783 | 1,000 | 1.81 h | 10,887 |

**WER on the 1,000-utterance official-test subsets** (best per column in bold; flag:
**N** = corpus not in the model's declared training data, **S** = trained on a sibling
split of the same corpus, **U** = training provenance insufficient to say — none of
them means "verified clean"):

| Model | Vaani test | SPRING-INX R1 eval | SPRING-INX R2 eval | Kathbath test (known spk) | Kathbath test (unknown spk) |
|---|---:|---:|---:|---:|---:|
| **ehzawad FastConformer-CTC** | **24.66%** [23.5–25.9] <sub>N</sub> | **39.24%** [38.0–40.5] <sub>N</sub> | **38.41%** [37.2–39.7] <sub>N</sub> | 13.38% [12.6–14.2] <sub>S</sub> | 13.43% [12.6–14.3] <sub>S</sub> |
| Whisper Medium | 26.25% [25.0–27.5] <sub>U</sub> | 39.96% [38.8–41.2] <sub>U</sub> | 39.70% [38.5–40.9] <sub>U</sub> | 9.69% [9.0–10.4] <sub>U</sub> | 9.79% [9.1–10.5] <sub>U</sub> |
| Conformer Large (hishab) | 33.10% [31.8–34.5] <sub>N</sub> | 42.46% [41.2–43.8] <sub>N</sub> | 42.24% [40.9–43.6] <sub>N</sub> | **9.33%** [8.6–10.1] <sub>S</sub> | **8.79%** [8.1–9.5] <sub>S</sub> |
| hishab FastConformer | 28.71% [27.4–30.0] <sub>U</sub> | 46.47% [44.8–48.2] <sub>U</sub> | 45.08% [43.5–46.7] <sub>U</sub> | 25.81% [24.1–27.6] <sub>U</sub> | 29.87% [27.5–32.4] <sub>U</sub> |
| Wav2Vec2 | 31.26% [29.9–32.7] <sub>U</sub> | 47.28% [46.1–48.6] <sub>U</sub> | 47.33% [46.1–48.6] <sub>U</sub> | 19.08% [18.1–20.1] <sub>U</sub> | 18.80% [17.9–19.7] <sub>U</sub> |

The ehzawad FastConformer's Vaani and SPRING hypotheses come from its earlier
evaluation of the full official splits (same greedy decoding, batched), scored here
on exactly these utterances; they are not re-timed, which is why its ms/clip is N/A
there. Its Kathbath rows, and every other cell, are batch-1 decodes here.

What the table says: on the two corpora nobody trained on, the FLEURS ranking does not
hold. Conformer Large — the FLEURS leader at 14.4% — is **last on Vaani** (33.1%) and
fourth on SPRING; its training list is Bangladesh-centric and these are Indian-district
and IIT-Madras recordings. The ehzawad FastConformer leads Vaani and both SPRING sets;
Whisper Medium is second on all three at ~15× the latency. On Kathbath the order flips
back: Conformer Large (9.3 / 8.8%) and Whisper (9.7 / 9.8%) lead, the ehzawad
FastConformer is third (13.4 / 13.4%) — Conformer Large trained on Kathbath validation
as well as train, Whisper's provenance for Kathbath is unknown (a **U** cell can hide
training overlap), and the ehzawad FastConformer saw only the train split. Known- vs
unknown-speaker Kathbath differ by under a point for every model except hishab
FastConformer.

A note on SPRING: about half of its reference utterances contain English words in Latin script, and
none of the five models emitted a single Latin-script word, so those words are errors for every model
alike; on the Latin-free half the ehzawad FastConformer and Whisper Medium are within each other's
intervals (details in `outputs_multiset/summary_multiset.json`).

<details>
<summary>ms/clip per set (batch 1, warmed, default runtime)</summary>

| Model | Vaani | SPRING R1 | SPRING R2 | Kathbath known | Kathbath unknown |
|---|---:|---:|---:|---:|---:|
| ehzawad FastConformer-CTC | N/A | N/A | N/A | 72 | 70 |
| Whisper Medium | 919 | 1,454 | 1,494 | 1,014 | 1,041 |
| Conformer Large (hishab) | 74 | 75 | 76 | 70 | 71 |
| hishab FastConformer | 72 | 74 | 73 | 70 | 71 |
| Wav2Vec2 | 33 | 49 | 51 | 35 | 35 |

</details>

Reproduce: `python score_multiset.py` over the committed predictions.

## What makes the comparison fair

- **One GPU, one model resident at a time.** Every number here was produced on the
  same RTX A5000 with nothing else on the card; the runners refuse to start while a
  foreign process holds it.
- **One audio path.** `asr_core.py` loads, downmixes, resamples to 16 kHz, chunks at
  25 s and stages PCM16 WAV identically for every backend; NeMo, wav2vec2, Whisper
  and Qwen read the same bytes (`prove_audio_path.py` hashes every staged chunk under
  both interpreters).
- **One scorer.** `bench_score.norm` strips punctuation to word boundaries and applies
  Unicode NFC before WER/CER. NFC is not cosmetic: Bengali writes য়, ড় and ঢ় either
  precomposed or as base + nukta; the FLEURS references and the NeMo/Qwen models use
  the decomposed form, Whisper and wav2vec2 the precomposed one, and without NFC
  every such character scores as a substitution (Whisper Medium reads 27.9% instead
  of 16.2%).
- **Greedy decoding, batch 1, no external LM** for every model — the
  interactive-latency workload, applied identically to CTC and autoregressive
  decoders. Batching would help the autoregressive models far more; it is not
  measured here for any model.
- **Failures stay in the denominator.** A model that raises on a clip yields an empty
  hypothesis; none did.
- **Checkpoints are pinned** to commit hashes in `outputs_fleurs/checkpoint_revisions.json`.
- **Intervals cluster by sentence** (FLEURS, Kathbath) or by distinct reference
  (Vaani, SPRING), because several speakers read the same text.

## Limitations

- FLEURS and Kathbath are read speech; Vaani and SPRING are spontaneous. No
  telephony, no far-field, no noise conditions.
- Speed is per-clip latency at batch 1 on one GPU model; throughput under load,
  batched decoding and CPU inference are not measured.
- The 1,000-utterance subsets give roughly ±1.3 pp intervals per cell; differences
  inside an interval are not differences.
- Training provenance for the third-party checkpoints is only as good as their model
  cards; a **U** flag means it could not be established either way.

## Reproducing this

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
python build_eval.py                              # FLEURS bn_in test + validation -> eval_manifest.json
BENCH_GPU=<A5000 uuid> ./run_all_fleurs.sh        # six models on FLEURS, one at a time
python bench_score.py && python clean_split_table.py
BENCH_GPU=<A5000 uuid> ./run_multiset.sh          # Vaani + SPRING subsets
BENCH_GPU=<A5000 uuid> ./run_multiset_kathbath.sh # Kathbath subsets
python score_multiset.py
python make_report.py                             # -> asr_benchmark.pdf
```

The Vaani, SPRING-INX and Kathbath audio is not redistributed here; `multiset/*/eval_manifest.json`
lists every selected utterance with its SHA-256 so a rebuild can be verified.

## The web demo

`app.py` serves `ehzawad/stt_bn_fastconformer_ctc` behind a Gradio UI for
interactive use: record from the microphone or upload a file, get Bengali text
back. It shares `asr_core.py` with the benchmark, so the demo and the measured
numbers use the same audio handling.

```bash
./run.sh --daemon    # background, logs to service.log
./run.sh --stop
```

Notes:

- HTTPS is used whenever `certs/cert.pem` and `certs/key.pem` exist, and this is
  not cosmetic: browsers gate `getUserMedia()` behind a secure context, so over
  plain HTTP the microphone is blocked before the app is reached. `localhost` is
  exempt from that rule; a LAN address is not.
- The transcript shown is raw decoder output. The repository's scoring
  normaliser is displayed separately and labelled, never substituted, because
  using a scoring surface as the display surface hides what the model actually
  predicted.
- There is no authentication. Anyone who can reach the port can use it.

## Files

```
asr_benchmark.pdf        the report
asr_core.py              the single audio path shared by demo and benchmark
backends.py              model loading for NeMo, wav2vec2 CTC and Whisper seq2seq
bench_run.py             one model over the FLEURS set, batch 1, timed
bench_run_set.py         the same backends over any eval manifest (the official-split runs)
bench_score.py           predictions -> WER/CER with both bootstraps -> summary.json
clean_split_table.py     the FLEURS test-split table with paired deltas
score_multiset.py        the official-split tables, incl. the SPRING Latin diagnostic
run_all_fleurs.sh        six models on FLEURS, one at a time, GPU pinned by UUID
run_multiset.sh, run_multiset_kathbath.sh   the official-split runs
make_report.py           summaries -> asr_benchmark.pdf
bench_server.py          serves any checkpoint under the harness's contract
app.py, run.sh           the interactive web demo
build_eval.py, prove_audio_path.py, eval_manifest.json   the FLEURS evaluation set
multiset/                frozen official-split subsets with per-file SHA-256
outputs_fleurs/          FLEURS predictions, run metadata, summaries
outputs_multiset/        official-split predictions, run metadata, summary
runtime/                 the Qwen3 adapter runtime study (static KV cache, merged LoRA)
requirements.txt         pinned
```

Not included: model checkpoints, the built evaluation sets, and the virtualenv.

## Attribution and licences

- Evaluation data: [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google),
  CC-BY-4.0; Vaani transcription part (ARTPARK-IISc), CC-BY-4.0; SPRING-INX R1/R2
  (SPRING Lab, IIT Madras), CC-BY-4.0; Kathbath (AI4Bharat IndicSUPERB), CC0.
  The committed manifests and predictions carry reference transcripts alongside each
  model's hypothesis.
- `hishab/titu_stt_bn_conformer_large`, `hishab/titu_stt_bn_fastconformer` — Hishab.
- `SayedShaun/bengali-whisper-medium` (mirror of tugstugi's Bengali.AI competition
  model), `SayedShaun/bangla-wave2vec2-unigram` (mirror of qdv206's).
- `ehzawad/stt_bn_fastconformer_ctc` — FastConformer-CTC large trained on 1,692 hours
  of human-supervised Bengali, 1,024-piece Bengali BPE, CC-BY-SA-4.0; base model
  `nvidia/stt_en_fastconformer_ctc_large`, NVIDIA, CC-BY-4.0.
  `ehzawad/stt_bn_qwen3_asr` — Bengali LoRA adapter on `Qwen/Qwen3-ASR-1.7B-hf`.

Model licences are those of their respective publishers. Check each model card
before use.

Author: Emrul Zawad (ehzawad).
