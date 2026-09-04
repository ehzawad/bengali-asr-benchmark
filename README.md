# Bengali ASR benchmark

> ### Correction (2026-09-04): Whisper Medium and wav2vec2 were scored wrong
>
> The table below previously understated two models by roughly 11 WER points.
> Bengali writes several letters two ways — য় is either U+09DF or U+09AF +
> U+09BC, and likewise ড় and ঢ় — canonically equivalent Unicode, different
> byte sequences. The references and the NeMo models use the decomposed
> spelling; **Whisper and wav2vec2 emit the precomposed one** (3,413 and 3,407
> such characters). The scorer compared raw strings, so each counted as a
> substitution.
>
> | model | was | is |
> |---|---|---|
> | Whisper Medium | 27.87% | **16.22%** |
> | wav2vec2 | 31.58% | **20.71%** |
>
> The other three models are unaffected — they emit zero precomposed
> characters. `norm()` in `collect_results.py` now applies NFC, and the table,
> intervals and PDF are regenerated. Whisper Medium is the second-most-accurate
> model here, not the fourth.

A controlled comparison of five Bengali speech-to-text models on the FLEURS
Bengali evaluation set, plus the web demo used to try one of them interactively.

Every model was measured in one sitting on one GPU (RTX 5080), loaded one at a
time, over the same 1,322 utterances, through the same evaluation harness and the
same audio preprocessing path. Accuracy and speed are therefore both comparable
across the whole table, which was not true of the earlier benchmark this
replaces.

Full report: [`asr_benchmark.pdf`](asr_benchmark.pdf)

## Results

| Model | Checkpoint | WER (95% CI) | CER | ms/clip | Params |
|---|---|---|---|---|---|
| Conformer Large | `hishab/titu_stt_bn_conformer_large` | **14.92%** [14.18-15.66] | 4.55% | 34.3 | 121.5M |
| Whisper Medium | `SayedShaun/bengali-whisper-medium` | 16.22% [15.55-16.88] | 5.05% | 777.2 | 763.9M |
| ehzawad FastConformer | `ehzawad/stt_bn_fastconformer` | 19.48% [18.77-20.16] | 5.68% | 31.7 | 115.6M |
| Wav2Vec2 | `SayedShaun/bangla-wave2vec2-unigram` | 20.71% [20.01-21.40] | 5.95% | 36.4 | 315.5M |
| hishab FastConformer | `hishab/titu_stt_bn_fastconformer` | 36.18% [34.74-37.67] | 9.03% | 30.5 | 115.6M |

Corpus-level WER/CER over 1,322 utterances, greedy decoding, no language model,
zero failed requests for any model. Intervals are 95% percentile bootstrap over
10,000 utterance resamples. `ms/clip` is isolated model compute from
`speed_probe.py`, not end-to-end request latency.

One pair of confidence intervals overlaps — Conformer Large [14.18-15.66] and
Whisper Medium [15.55-16.88] — so those two are not separated by this evidence.
Every other pair is disjoint. (Before the Unicode correction above no pair
overlapped, because the bug pushed Whisper into fourth place.)

## Findings

**Size does not buy accuracy here.** Conformer Large, at 121.5M parameters,
beats a 763.9M Whisper Medium by 13 WER points and a 315.5M Wav2Vec2 by 17.

**Autoregressive decoding costs about 25x.** Whisper spends 777 ms per clip
against 30-36 ms for every CTC model, because it emits one token at a time
through a decoder where the CTC models emit a whole utterance in one forward
pass. This is architectural, not a serving artefact: it is the only model whose
disadvantage survives end-to-end (1.26 req/s against 26-30 req/s), because at
777 ms of compute the per-request overhead that compresses the CTC models'
differences is negligible.

**The accuracy leader is not the slow option.** An earlier benchmark measured
Conformer Large at half FastConformer's throughput and recommended against it
for latency-sensitive paths. Measured on one GPU the gap is 13% of model compute
and 6% end-to-end. Paying 13% for 4.6 points of WER is a different decision from
paying 2x.

**The two FastConformers are tied on speed and far apart on accuracy.** Same
architecture, same parameter count (115.6M), 30.5 ms against 31.7 ms with
overlapping ranges - but 36.18% against 19.48% WER. Their error profiles are
opposites: the hishab checkpoint inserts 3,695 words against 120 deletions, the
signature of an over-eager CTC decode, while the ehzawad checkpoint is balanced
at 473 insertions against 401 deletions.

**A model can be misrepresented by its deployment.** Reached over the network as
a pre-existing Triton service, the wav2vec2 weights previously recorded 34.73%
WER at 0.55 req/s with 16 failed requests. The same weights loaded locally give
20.71% at 26.26 req/s with zero failures - 48x the throughput. Over the network
today that service returns 20.65% at 13.53 req/s, so the residual gap is
transport rather than the model.

## What makes the comparison fair

These are the choices that keep the differences attributable to the models
rather than to the measurement:

- **One GPU, one model at a time.** Each checkpoint is loaded alone and the
  previous process is confirmed dead before the next starts, so no model
  competes for VRAM with another and a stale process cannot answer for the
  wrong checkpoint.
- **One audio path for every model.** `asr_core.py` holds the only
  implementation of decoding, downmixing, resampling and segmentation.
  `backends.py` wraps each model family in the same `transcribe()` signature so
  every one is driven through it. Two separate preprocessing paths would surface
  as a model-quality difference.
- **One harness, unmodified.** The same evaluation script scores every run, with
  the same punctuation-stripping normaliser and the same failure policy: a failed
  request is scored as an empty hypothesis, never dropped.
- **Greedy decoding throughout.** `num_beams=1`, no language model, for every
  model. Beam search for Whisper alone would hand it a search budget the CTC
  models never received.
- **Whisper's language is forced to Bengali** rather than auto-detected. A
  mis-detected language makes Whisper emit fluent text in the wrong language,
  which scores as a catastrophic WER while looking like a model result rather
  than a configuration mistake.
- **PyTorch for every model.** The wav2vec2 and Whisper checkpoints are also
  deployed elsewhere as ONNX and CTranslate2 exports. Running them in PyTorch
  keeps the comparison about models rather than runtimes. On a shared clip the
  local and remote wav2vec2 transcripts match exactly, which confirms the
  checkpoint is the same one.
- **Speed measured two ways.** `speed_probe.py` times sequential in-process
  inference with warm-up discarded, isolating model compute. The harness measures
  end-to-end throughput at concurrency 10. The two disagree, and the report says
  why rather than picking the flattering one.

## Limitations

- **FLEURS is short read speech.** It says nothing about conversational, noisy,
  dialectal or long-form audio. The ehzawad model card reports roughly 5 points
  worse WER on spontaneous speech. Whisper Medium is built for long-form,
  multi-domain audio, so a benchmark of short read clips is close to its least
  favourable case; this result should not be read as a general verdict on it.
- **Run-to-run variance in the speed probe is about 7%.** Across two runs the
  CTC timings moved by roughly 2 ms. Differences smaller than that are not
  rankings, which is why the probe reports min-max whiskers.
- **One benchmark run per model**, so throughput and latency carry no confidence
  interval. Only accuracy does.
- **Serialised endpoint, not a batching server.** Throughput here is
  1/service-time under a lock, and the reported latency is mostly queueing. It
  does not predict behaviour under a production serving stack with dynamic
  batching.
- **The evaluation harness is not vendored here** (see below), so this repository
  is not self-contained for reproduction.
- **Parameter count is not architecture.** The two FastConformers report the same
  count and architecture family, but their layer geometry, feature frontend,
  tokenizer and decoding configuration were not compared. The 16.7-point gap
  between them is a difference between checkpoints; this benchmark does not
  establish which ingredient caused it.

## Reproducing this

The evaluation harness (`scripts/benchmark.py` and
`scripts/download_eval_data.py` from a separate ASR inference pipeline) is not
included in this repository. It is referenced by path, not vendored, because it
is not ours to redistribute. Supply your own via the `HARNESS` variable; it must
accept `--api-url`, `--data-dir`, `--concurrency` and `--output-dir`, and speak
this request/response contract:

```
POST /asr
{"config": {"language": {"sourceLanguage": "bn"}},
 "audio": [{"audioContent": "<base64 PCM16 WAV>"}]}
-> {"output": [{"source": "<transcript>"}], "time_taken": <seconds>}
```

Then:

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt

# Build the eval set: FLEURS bn_in, test + validation merged -> 1,322 utterances
python $PIPELINE/scripts/download_eval_data.py --data-dir eval_fleurs_bn

# Place the checkpoints under model/, then benchmark them one at a time
HARNESS=$PIPELINE/scripts/benchmark.py ./run_all_benchmarks.sh

python speed_probe.py       # isolated per-model compute
python collect_results.py   # -> outputs/summary.json
python make_report.py       # -> asr_benchmark.pdf
```

`make_report.py` renders exclusively from `outputs/summary.json`, which is
computed from the raw per-utterance predictions. No figure in the PDF is typed
in by hand, so a number in the report cannot drift from the run that produced it.

## The web demo

`app.py` serves `ehzawad/stt_bn_fastconformer` behind a Gradio UI for
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
- The transcript shown is raw decoder output. The repository's frozen scoring
  normaliser is displayed separately and labelled, never substituted, because
  using a scoring surface as the display surface hides what the model actually
  predicted.
- There is no authentication. Anyone who can reach the port can use it.

## Files

```
asr_benchmark.pdf     the report
asr_core.py              the single audio path shared by demo and benchmark
backends.py              model loading for NeMo, wav2vec2 CTC and Whisper seq2seq
bench_server.py          serves any checkpoint under the harness's contract
run_all_benchmarks.sh    benchmarks every checkpoint, one at a time
speed_probe.py           isolated per-model compute, warm-up discarded
collect_results.py       raw predictions -> outputs/summary.json, with intervals
make_report.py           outputs/summary.json -> the PDF
app.py, run.sh           the interactive web demo
outputs/                 per-model report.txt and predictions.json, plus summary
requirements.txt         pinned, verified on an RTX 5080 (sm_120, cu128 wheels)
```

Not included: model checkpoints, the built evaluation set, and the virtualenv.

## Attribution and licences

- Evaluation data: [FLEURS](https://huggingface.co/datasets/google/fleurs)
  (Google), CC-BY-4.0. `outputs/*/predictions.json` contains FLEURS reference
  transcripts alongside each model's hypothesis.
- `hishab/titu_stt_bn_conformer_large`, `hishab/titu_stt_bn_fastconformer` -
  Hishab.
- `SayedShaun/bangla-wave2vec2-unigram`, `SayedShaun/bengali-whisper-medium`.
- `ehzawad/stt_bn_fastconformer` - FastConformer-CTC large fine-tuned on 984
  hours of human-supervised Bengali, 1,024-piece Bengali BPE. Base model
  `nvidia/stt_en_fastconformer_ctc_large`, NVIDIA, CC-BY-4.0.

Model licences are those of their respective publishers. Check each model card
before use.
