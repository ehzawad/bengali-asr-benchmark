# Runtime study: why the LLM-decoder adapter was slow, and what fixed it

The six-model comparison ran every model in its default configuration at batch
one. Under that protocol the Qwen3-ASR Bengali adapter took **9,263 ms/clip**
against 77 ms for the CTC Conformers — a 120x gap that is far larger than the
parameter counts explain. This directory contains the investigation of that
gap and the runtime change that closed most of it.

The headline: the gap was **not** the audio encoder, **not** the KV cache being
disabled, and **not** GPU arithmetic. It was kernel-launch overhead, and a
one-line runtime flag removed most of it.

## 1. Where the time actually goes

`decompose_latency.py` splits a clip into audio encoding, prompt prefill, and
the autoregressive loop by measuring the *slope* between `max_new_tokens=1` and
full generation, which cancels the one-off costs.

```
decode_share             0.997      99.7% of a clip is the decode loop
mean_encoder_ms         16.4        the audio tower is 0.2% of the clip
mean_generated_tokens  129.8
mean_ms_per_decode_token 66.05
theoretical @768 GB/s    5.34       12.4x gap
```

Two independent multipliers, not one:

1. **129.8 tokens per clip.** Qwen's vocabulary has no Bengali merges, so
   Bengali is encoded by byte fallback at ~6.4 tokens/word, with ~40% of target
   tokens being bare UTF-8 continuation bytes.
2. **66 ms per token**, 12.4x above what streaming 4 GB of bf16 weights at the
   card's bandwidth would cost.

Per-token cost was **flat** as context grew (67.6 / 66.4 / 66.2 ms at 90 / 180 /
113 generated tokens). A broken KV cache would grow linearly with position, so
this rules out the most common explanation. Prefill cost likewise did not scale
with prompt length (86.5 ms at 155 tokens, 86.7 ms at 209). Both facts point at
a fixed per-forward-pass cost rather than arithmetic.

## 2. Attributing the fixed cost

`runtime_matrix.py` profiles 12 decode steps:

```
total self CPU     616.0 ms
total self CUDA    259.2 ms      GPU busy only 42% of the time

cudaLaunchKernel   138.99 ms CPU     0.00 ms CUDA   19,225 calls
aten::mm            70.60           78.96           2,365
aten::mul           51.29           10.91           4,442
aten::mean          22.86            4.91           1,356
aten::pow           22.27            2.11           1,356
aten::rsqrt         13.93            2.09           1,356
aten::cat           18.01            4.97           1,416
```

**19,225 kernel launches for 12 tokens — 1,602 launches per token**, or 57 per
decoder layer. The `pow`/`mean`/`rsqrt`/`mul` quartet at 1,356 calls each is
RMSNorm running as four separate kernels rather than one fused op (~113 norms
per step: 28 layers x input/post-attention, plus Qwen3's per-head q_norm and
k_norm). `aten::cat` at 1,416 calls is the dynamic KV cache reallocating K and V
for every layer on every step.

The GPU is idle 58% of every decode step, waiting for Python to issue work.

## 3. What fixed it

| variant | ms/clip | ms/token | transcripts vs eager |
|---|---|---|---|
| PEFT-wrapped (as benchmarked) | 9,263 | 72.29 | - |
| merged, eager attention | 6,677.6 | 51.17 | identical |
| merged, SDPA attention | 5,774.7 | 44.25 | identical |
| merged + static KV cache | 1,420.2 | 10.88 | 5/8 identical |
| merged + `torch.compile` reduce-overhead | failed | - | CUDA-graph error |

Those are 8-clip probe figures. Re-measured on all 1,322 clips through the same
vendored `asr_core` audio path as the published benchmark, the shipped
configuration is **1,461.0 ms/clip** mean (p95 2,357.3) -- a **6.34x** speedup
over the 9,263 ms/clip in the comparison table.

Two changes, both cheap:

- **`merge_and_unload()`** folds the 196 LoRA modules into the base weights:
  **1.94x**. The adapter had been serving with every LoRA module dispatched
  separately at inference time.
- **A static KV cache** preallocates a fixed-size buffer, removing the per-layer
  `aten::cat` and letting the launch bookkeeping collapse: another **4.1x**.

## 3a. The speedup does not cost accuracy

Changing the cache changes bf16 reduction order, so argmax can flip at near
ties. That is not automatically harmless and was not assumed to be: all 1,322
clips were re-decoded and scored against the shipped predictions with a
**paired** bootstrap over per-utterance error counts.

```
wer_shipped     16.030%
wer_static      16.081%
delta           +0.051 pp    95% CI [-0.035, +0.141]
identical       1222/1322 transcripts (92.4%)
```

The interval contains zero: 7.6% of transcripts changed, and those changes are
worth about five hundredths of a percentage point in either direction. The
runtime change is adopted on that basis, not on the probe's transcript count.

`torch.compile(mode="reduce-overhead")` failed for *both* autoregressive models
in transformers 5.16.1 with `accessing tensor output of CUDAGraphs that has
been overwritten by a subsequent run`, so neither model receives CUDA graphs
here and the comparison stays symmetric.

## 4. The comparison must be symmetric

A static cache is a runtime flag, not a property of a model. Applying it only to
the model under study and then declaring victory would manufacture the result.
Whisper medium is the other autoregressive model in the benchmark and runs the
same `transformers` generate loop, so it receives the identical treatment
(`whisper_runtime.py`, `eval_whisper_static.py`). The CTC Conformers and
wav2vec2 are unaffected by construction: they perform a single forward pass and
have no KV cache.

Both autoregressive models, all 1,322 clips, same audio path, same card:

| model | default runtime | + static KV cache | speedup | WER before -> after |
|---|---|---|---|---|
| Qwen3-ASR Bengali adapter | 9,263.1 ms | **1,461.0 ms** | 6.34x | 16.030% -> 16.081% |
| Whisper medium (Bengali FT) | 1,804.0 ms | **467.6 ms** | 3.86x | 16.235% -> 16.235% |

**Whisper medium remains 3.12x faster than the adapter under identical
treatment**, and after the Unicode-normalisation correction it is also level with
it on accuracy (16.23% against 16.54%, paired difference +0.310 pp, 95% CI
[-0.559, +1.176]). The adapter gains more from the change only because it started
further from the hardware floor. Optimising one model and not the other would
have reversed the apparent ranking -- at 1,461 ms against Whisper's *unoptimised*
1,804 ms the adapter appears to win, and that comparison would have been an
artefact of unequal effort.

The residual 3.12x is structural and is not addressable by runtime work: the
adapter is a 2.0B model emitting ~130 tokens per clip where Whisper medium is a
769M model emitting ~90. That is roughly 2x per token and 1.46x in token count.
Closing it would require a Bengali-merge vocabulary (measured offline at 6.42 ->
1.44 tokens/word for a +8k extension) and/or decoder-depth reduction, both of
which are retraining projects, not configuration changes.

## 4a. A claim that did not survive testing

The adapter's character error rate (4.475%) is nominally lower than the best
model in the comparison, hishab Conformer Large (4.555%). A **paired** bootstrap
over per-utterance character errors, clustered by the 499 distinct reference
sentences, does not support calling that a lead:

```
delta  -0.081 pp    95% CI [-0.590, +0.414]    P(adapter better) = 0.62
```

The interval contains zero. "Lowest CER of the six" is therefore not a claim
this benchmark can make; "statistically indistinguishable from the best model on
CER" is. Two independent overlapping CIs would not have settled this either way
-- the models are scored on identical utterances, so the test has to be paired
(`paired_cer.py`).

## 5. A discarded measurement

An earlier batching sweep was thrown away rather than published. `asr_core.decode()`
calls `model.transcribe(paths, batch_size=1)` — the batch size is fixed at one
inside the shared audio path — so handing it more paths does not batch the
model, it only amortises per-call setup. The adapter, whose backend loops over
paths, went 8,986 -> 8,919 ms/clip from "batch" 1 to 4 (1.01x, i.e. nothing),
while the NeMo CTC model went 79.2 -> 40.5 ms/clip (1.96x) purely from
amortising NeMo's per-call dataloader construction. Reporting those two numbers
side by side would have compared overhead amortisation against true batching.
See `BATCHING_NOTE.md`.

## Reproducing

```bash
python runtime/build_merged.py         # fold LoRA + projector into base weights
python runtime/decompose_latency.py    # encoder vs prefill vs per-token decode
python runtime/runtime_matrix.py       # profiler attribution + variant sweep
python runtime/whisper_runtime.py      # identical treatment for the baseline
python runtime/eval_static_cache.py    # full 1,322-clip accuracy gate
python runtime/eval_whisper_static.py  # full 1,322-clip baseline gate
```
