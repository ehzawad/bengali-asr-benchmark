# Bengali ASR benchmark

> ### Correction (2026-09-04): Whisper Medium and wav2vec2 were scored wrong
>
> Both reports below previously understated two models by roughly 11 WER points.
> Bengali writes several letters two ways — য় is either U+09DF or U+09AF +
> U+09BC, and likewise ড় and ঢ় — which are canonically equivalent Unicode but
> different byte sequences. The FLEURS references and the NeMo/Qwen models use
> the decomposed spelling; **Whisper and wav2vec2 emit the precomposed one**
> (3,412 and 3,407 such characters respectively). The scorer compared raw
> strings, so every one of those counted as a substitution.
>
> | model | was | is |
> |---|---|---|
> | Whisper Medium | 27.87% | **16.22%** |
> | wav2vec2 | 31.58% | **20.71%** |
>
> The other four models are unaffected — they emit zero precomposed characters.
> `norm()` now applies NFC in both scorers and every table, PDF and interval
> below has been regenerated. This changes the ranking: Whisper Medium is the
> second-most-accurate model here, not the fourth.

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
Every other pair is disjoint, and that part of the ordering is a real ranking
rather than an artefact of which utterances landed in the set. (Before the
Unicode correction above, no pair overlapped; the correction moved Whisper from
fourth place into a statistical tie for first.)

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

## Second report: adding an LLM-decoder model (RTX A5000)

[`asr_benchmark_qwen3_adapter_a5000.pdf`](asr_benchmark_qwen3_adapter_a5000.pdf)

A sixth model joins the comparison: a Bengali LoRA adapter on
`Qwen/Qwen3-ASR-1.7B-hf`, an LLM-decoder ASR model whose supported-language list
does not include Bengali. Because the GPU changed, **every** model was
re-measured; none of the RTX 5080 numbers are carried across.

> **Evaluate on the test split only.** The 1322-utterance set (test + validation)
> used for the reproduction check below is **contaminated** for the two models
> trained on our corpus: 88 of the validation split's 150 distinct sentences
> appear verbatim in the training data via `fleurs_train`. It flatters
> `qwen3_adapter` and `ehzawad_fastconformer` and not the third-party
> checkpoints. Reproduce with `contamination_audit.py`. The primary table is
> therefore the clean 920-utterance test split.

| Model | Checkpoint | WER (95% CI) | CER | ms/clip | Params |
|---|---|---|---|---|---|
| Conformer Large | `hishab/titu_stt_bn_conformer_large` | **14.37%** [13.11–15.73] | 4.48% | 78 | 121.5M |
| Whisper Medium | `SayedShaun/bengali-whisper-medium` | 16.23% [15.05–17.45] | 5.09% | 1,820 | 763.9M |
| **Qwen3-ASR + Bengali adapter** | `ehzawad/stt_bn_qwen3_asr` on `Qwen/Qwen3-ASR-1.7B-hf` | 16.54% [15.43–17.68] | 4.68% | 9,258 | 2.04B / 38M trained |
| ehzawad FastConformer | `ehzawad/stt_bn_fastconformer` | 19.67% [18.50–20.88] | 5.79% | 78 | 115.6M |
| Wav2Vec2 | `SayedShaun/bangla-wave2vec2-unigram` | 20.64% [19.46–21.89] | 6.02% | 64 | 315.5M |
| hishab FastConformer | `hishab/titu_stt_bn_fastconformer` | 36.22% [33.88–38.72] | 9.05% | 76 | 115.6M |

> **Why Conformer Large appears as both 14.37% and 14.93%.** They are the same
> run scored on different sets: **14.37%** is the clean 920-utterance test split
> (the table above), **14.93%** is test + validation (1,322, folded away below).
> Every model has two such numbers. Paired deltas are internally consistent with
> whichever set they were computed on — 16.03 − 1.104 = 14.93 on the 1,322 set,
> and 16.54 − 2.162 = 14.37 on the test split. Quote the test-split figures.

**920 FLEURS bn_in test utterances**, zero sentence overlap with the training
corpus, same shared audio path, greedy decoding, batch 1, no external language
model, zero failed clips for any model. Intervals resample the 349 distinct
sentences. Speeds are the default runtime; see [`runtime/`](runtime/) for the
optimised figures.

Contamination changes the margin, not the order, but it changes what can be
claimed: on the contaminated 1322-utterance set the adapter reads 16.03% against
Conformer Large's 14.93% (1.10 pp); on the clean test split it reads 16.54%
against 14.37% (**2.16 pp**, 95% CI [1.14, 3.17]). About half the apparent
closeness was contamination.

<details>
<summary>The 1322-utterance table (test + validation), retained for the reproduction check</summary>

| Model | WER (95% CI) | CER | ms/clip |
|---|---|---|---|
| Conformer Large | 14.93% [13.81–16.06] | 4.55% | 77 |
| Qwen3-ASR + Bengali adapter | 16.03% [15.10–16.99] | 4.47% | 9,263 |
| Whisper Medium | 16.24% [15.25–17.26] | 5.05% | 1,804 |
| ehzawad FastConformer | 19.50% [18.49–20.49] | 5.68% | 77 |
| Wav2Vec2 | 20.71% [19.71–21.75] | 5.95% | 64 |
| hishab FastConformer | 36.18% [34.10–38.29] | 9.03% | 76 |

</details>

**The five previously published models reproduce to within 0.02 WER points**
(19.48→19.50, 14.92→14.93, 36.18→36.18, 20.71→20.71, 16.22→16.24), which is the
evidence that the rebuilt evaluation set and the re-implemented runner measure
the same thing the original harness did.

Worth stating plainly: this check originally passed against the *uncorrected*
numbers too (27.87→27.89, 31.58→31.59). A reproduction check confirms that two
implementations agree, not that either is right — both scorers shared the same
missing Unicode normalisation, so both were wrong in the same way and the
agreement looked like validation. The Unicode bug was caught by reading a
per-word error display, not by this check.

### What the sixth model shows

**Third on words, best-but-one on characters.** On the clean test split, paired
over identical utterances and clustered by the 349 distinct reference sentences:

```
vs Conformer Large   WER +2.162 pp   95% CI [+1.140, +3.171]   excludes zero -- behind
                     CER +0.201 pp   95% CI [-0.374, +0.735]   contains zero -- level
vs Whisper Medium    WER +0.310 pp   95% CI [-0.559, +1.176]   contains zero -- TIED
```

So the adapter is genuinely behind Conformer Large on words, level with it on
characters, and **statistically tied with Whisper Medium** — which is also about
3x faster per clip. Before the Unicode correction the adapter appeared to beat
Whisper Medium by 11 WER points; it does not.
An earlier draft claimed the adapter had "the lowest CER in the table" on the
contaminated set (4.47% vs 4.55%). On clean data it is nominally *behind*
(4.68% vs 4.48%) and the paired interval still contains zero, so neither a CER
lead nor "lowest CER" is supportable. See
[`runtime/paired_cer.py`](runtime/paired_cer.py) and
[`clean_split_table.py`](clean_split_table.py).

**It is 121× slower per clip than the CTC models.**
9,263 ms against 77 ms. A CTC model emits an utterance in one
forward pass; this one generates it token by token through a 1.7B decoder, and
Qwen's vocabulary has no Bengali merges, so each Bengali character costs roughly
three byte-tokens. Batch 1 is the interactive-latency workload and is the least
favourable setting for autoregressive decoding — batching helps it far more than
it helps the CTC models, and is not measured here for any model.

Most of that 9,263 ms turned out to be avoidable, and none of it was the audio
encoder. See [the runtime study](runtime/) — the shipped configuration is now
**1,461 ms/clip**, a 6.34× speedup at unchanged WER. Whisper Medium, given the
identical treatment, is still 3.12× faster.

**Adapting an unsupported language works.** Zero-shot, the base model transcribes
Bengali speech into Devanagari at 77% WER. Training 38M parameters — a rank-32
LoRA on the decoder plus the audio projector, encoder frozen — on 984 hours takes
it to 16.03%.

### Differences from the first report

- **RTX A5000, exclusively**, rather than the shared card, so no co-tenant can
  perturb timings. Every model re-measured; nothing carried over.
- **End-to-end throughput, queueing latency and failed-request counts are
  absent.** The harness that produced them was never vendored and was
  unavailable. They are omitted, not estimated.
- **Intervals resample the 499 distinct sentences**, not the 1322
  recordings. Several speakers read the same sentence, so recordings are not
  independent and the per-utterance interval is optimistic. Both are in the
  summary; the clustered one is quoted above.
- **The three NeMo models are speed-tied here** (77–77 ms) where the
  RTX 5080 separated them by 12%. At this speed tier the shared path's WAV
  staging is a large share of the measurement, which is why the metric is called
  warmed in-process batch-1 inference time rather than model compute.
- **Checkpoints are pinned to commit hashes** in `outputs_a5000/checkpoint_revisions.json`.
- The two interpreters (NeMo needs one transformers version, the adapter another)
  were shown to stage **byte-identical audio**: `prove_audio_path.py` hashes every
  staged chunk under both, including clips long enough to be segmented.

### Reproducing the second report

```bash
python build_eval.py                 # rebuild the 1,322-utterance set
python prove_audio_path.py           # run under BOTH interpreters, diff the hashes
BENCH_GPU=<uuid> ./run_all.sh        # six models, one at a time
python bench_score.py                # both bootstraps -> outputs_a5000/summary_a5000.json
python make_report_a5000.py          # -> asr_benchmark_qwen3_adapter_a5000.pdf
```

Unlike the first report, this one is self-contained: the runner and scorer are
here, so no external harness is required.

### Runtime study

[`runtime/`](runtime/) investigates the adapter's 120× speed gap and closes most
of it. Briefly: 99.7% of a clip is the autoregressive loop, the audio tower is
0.2%, and the per-token cost was 12.4× above the memory-bandwidth floor because
the decode loop issued **1,602 kernel launches per generated token** and left the
GPU idle 58% of the time. Merging the LoRA modules into the base weights (1.94×)
and switching to a static KV cache (a further 4.1×) bring 9,263 ms/clip to
**1,461 ms/clip**, with WER moving +0.051 pp on a 95% CI of [-0.035, +0.141].

Whisper Medium receives the same treatment (1,804 → 468 ms) because a runtime
flag applied to one model and not another manufactures the result rather than
measuring it. It remains 3.12× faster than the adapter; the residual gap is
structural — 2.0B parameters emitting ~130 tokens per clip against 769M emitting
~90 — and is not addressable by configuration.

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
