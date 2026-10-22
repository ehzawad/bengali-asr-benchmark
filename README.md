# Bengali ASR benchmark

A controlled comparison of ten Bengali speech-to-text models on eleven evaluation sets. Four models are
public checkpoints; six are the author's private bn-BD fine-tunes, marked ‡, one of them a
55M-parameter on-device model (Whistle bn-BD) measured twice, on an A100 and on CPU. The sets are:

- the FLEURS Bengali test split;
- 1,000-utterance samples of six Indian sets: Vaani, SPRING-INX R1 and R2, Kathbath (known- and
  unknown-speaker) and IndicVoices conversational;
- 1,000-utterance samples of four sets added on 14 Oct 2026: SUBAK.KO and BanSpeech (Bangladeshi standard
  Bangla), Common Voice 17 (crowd-sourced) and Ben-10 (Bangladeshi regional dialects).

Every decode ran on one Colab A100-SXM4-40GB, one model at a time, greedy at batch 1, through the same
audio path and the same scorer, so accuracy and speed are comparable across the tables. The exception is
Whistle bn-BD, run later on an A100-SXM4-80GB in JAX (greedy, batch 1) and on CPU in its own engine; see
[On-device model: Whistle bn-BD](#on-device-model-whistle-bn-bd). Every other autoregressive decoder
(Voxtral Mini, Whisper Medium, Whisper turbo, Qwen3-ASR) ran through vLLM; the CTC and RNN-T models, Canary
and Wav2Vec2 through NeMo or PyTorch.

The ‡ models are private, so readers cannot run them; they are here to place the author's own
fine-tunes in the public field.

Report: [`asr_benchmark.pdf`](asr_benchmark.pdf), a short summary centred on Voxtral Mini bn-BD, rendered
by `make_report.py` from the committed summaries. This README has the full tables and intervals.

## Which one to use

Public models:

- **`ehzawad/stt_bn_fastconformer_ctc`**: the best public model on Vaani and both SPRING sets, corpora no
  public model's card lists, with 17.1 % WER on FLEURS at 88 ms per clip.
- **`hishab/titu_stt_bn_conformer_large`**: best of the public models on FLEURS (14.4 %), Kathbath and
  Common Voice, i.e. read speech, at the same latency. Its card lists Kathbath and Common Voice as training
  data, and it is **last on Vaani** (33.1 %).
- **Whisper Medium**: second of the public models on most sets and the best public model on BanSpeech
  (23.6 %), at 304 ms per clip under vLLM.

The private bn-BD fine-tunes ‡ take five of the top six places by average rank over all eleven sets:

| Model | Average rank (1 is best) | Sets with the lowest WER | Average WER |
|---|---:|---|---:|
| Whisper turbo bn-BD ‡ | 3.2 | Vaani, SPRING R1, SPRING R2 | 22.8% |
| Qwen3-ASR 1.7B bn-BD ‡ | 3.3 | IndicVoices | 22.9% |
| Canary 1B v2 bn-BD ‡ | 3.5 | FLEURS | 24.1% |
| Voxtral Mini bn-BD ‡ | 4.2 | SUBAK.KO, BanSpeech | 23.8% |
| FastConformer hybrid bn-BD ‡ | 4.8 | Ben-10 | 23.4% |
| Conformer Large (hishab) | 5.1 | Kathbath ×2, Common Voice (trained on all three) | 26.5% |
| Whisper Medium | 5.9 | — | 27.0% |
| ehzawad FastConformer-CTC | 6.5 | — | 26.4% |
| Whistle bn-BD ‡ (55M, CPU engine) | 8.6 | — | 30.0% |
| Whistle bn-BD ‡ (55M, A100, greedy) | 9.3 | — | 36.5% |
| Wav2Vec2 | 9.8 | — | 34.0% |

Ranks and averages weight the eleven sets equally; Ben-10's 60–80 % error rates dominate the average WER
(121 % for greedy Whistle on the A100, see below). The two Whistle rows are one checkpoint under two
runtimes, so each is ranked against the nine other models, not against the other; the other ranks are
those with the CPU row (with the A100 row instead, Conformer Large moves to 5.0, Whisper Medium to 5.8 and
Wav2Vec2 to 9.4).
Several leads are inside their intervals (see [Voxtral Mini against the best other model](#voxtral-mini-against-the-best-other-model)).

- **Whisper turbo bn-BD** is the most accurate on Indian conversational speech, but its multilingual
  tokenizer spends about 13 tokens per Bengali word, so its longest pieces run into the 448-token decoder
  limit (36 FLEURS test clips, 403 of 1,000 Ben-10 clips), and it is the slowest model here (700 ms).
- **Voxtral Mini bn-BD** has the lowest error rate on the two Bangladeshi standard-Bangla sets, SUBAK.KO and
  BanSpeech, though not separably from Canary and Qwen3-ASR there.
- **FastConformer hybrid bn-BD** is best on the Ben-10 dialects and runs in 0.1 s per clip, 4–7× faster
  than the other private models.

## Bangladeshi and crowd-sourced test sets

| Set | Source | Speech | Sampled |
|---|---|---|---|
| SUBAK.KO test | SUST, [`SUST-CSE-Speech/SUBAK.KO`](https://huggingface.co/datasets/SUST-CSE-Speech/SUBAK.KO), CC-BY-4.0 | Bangladeshi standard Bangla, studio-read and broadcast | 1,000 of 6,533, by corpus folder |
| BanSpeech | SUST, [`SUST-CSE-Speech/banspeech`](https://huggingface.co/datasets/SUST-CSE-Speech/banspeech), CC-BY-NC-4.0 (card text: CC-BY-4.0) | 13 broadcast domains: news, parliament, talk shows, drama, lectures, children's voices and more | 1,000 of 8,085, by domain |
| Common Voice 17 test | Mozilla, via [`fixie-ai/common_voice_17_0`](https://huggingface.co/datasets/fixie-ai/common_voice_17_0), CC0 | crowd-sourced read sentences; speaker region blank for 98.5 % of clips | 1,000 of 9,327, by the gender field |
| Ben-10 validation | Bengali.AI, [`bengaliAI/Ben-10`](https://huggingface.co/datasets/bengaliAI/Ben-10), CC0 | Bangladeshi regional dialects in dialect spelling, 12 districts | 1,000 of 1,666, by district |

`build_multiset_bd.py` builds them from those repositories at pinned commits, converts each clip with
ffmpeg 6.1.1 to 16 kHz PCM16 and writes the manifests under [`multiset/`](multiset/) with per-file SHA-256.
Ben-10's test split is private, so its public validation split stands in.

**WER on the four sets** (best per column in bold; 95 % intervals in brackets; flags as below):

| Model | SUBAK.KO test | BanSpeech | Common Voice 17 test | Ben-10 validation (dialects) |
|---|---:|---:|---:|---:|
| Voxtral Mini bn-BD ‡ | **11.66%** [10.6–12.8] <sub>U</sub> | **20.72%** [19.4–22.0] <sub>U</sub> | 10.32% [9.5–11.2] <sub>U</sub> | 69.97% [65.5–75.1] <sub>U</sub> |
| FastConformer hybrid bn-BD ‡ | 14.87% [13.6–16.2] <sub>U</sub> | 23.47% [22.0–25.0] <sub>U</sub> | 10.51% [9.7–11.4] <sub>U</sub> | **60.91%** [59.7–62.1] <sub>U</sub> |
| Canary 1B v2 bn-BD ‡ | 12.02% [11.0–13.1] <sub>U</sub> | 22.59% [21.1–24.1] <sub>U</sub> | 9.51% [8.7–10.3] <sub>U</sub> | 76.42% [71.9–81.3] <sub>U</sub> |
| Qwen3-ASR 1.7B bn-BD ‡ | 13.01% [11.3–15.4] <sub>U</sub> | 20.76% [19.4–22.2] <sub>U</sub> | 9.88% [8.4–12.0] <sub>U</sub> | 63.39% [61.2–65.7] <sub>U</sub> |
| Whisper turbo bn-BD ‡ | 14.42% [12.8–16.4] <sub>U</sub> | 21.96% [20.4–23.6] <sub>U</sub> | 9.59% [8.8–10.4] <sub>U</sub> | 62.21% [60.8–63.6] <sub>U</sub> |
| ehzawad FastConformer-CTC | 14.33% [13.4–15.3] <sub>S</sub> | 27.84% [26.2–29.5] <sub>N</sub> | 12.14% [11.2–13.0] <sub>S</sub> | 76.92% [75.8–78.0] <sub>N</sub> |
| Conformer Large (hishab) | 13.64% [12.5–14.8] <sub>S</sub> | 27.07% [25.5–28.7] <sub>N</sub> | **7.35%** [6.7–8.0] <sub>S</sub> | 63.54% [62.3–64.8] <sub>N</sub> |
| Whisper Medium | 21.36% [20.3–22.5] <sub>U</sub> | 23.59% [22.1–25.0] <sub>U</sub> | 10.12% [9.3–11.0] <sub>U</sub> | 74.79% [73.6–75.9] <sub>U</sub> |
| Wav2Vec2 | 26.07% [25.0–27.1] <sub>U</sub> | 32.81% [31.1–34.5] <sub>U</sub> | 16.70% [15.7–17.7] <sub>U</sub> | 80.50% [79.5–81.5] <sub>U</sub> |
| Whistle bn-BD ‡ (55M, CPU engine) | 18.79% [17.7–20.0] <sub>U</sub> | 31.54% [29.7–33.5] <sub>U</sub> | 17.10% [16.0–18.2] <sub>U</sub> | 79.87% [78.3–81.4] <sub>U</sub> |
| Whistle bn-BD ‡ (55M, A100, greedy) | 23.90% [20.8–27.5] <sub>U</sub> | 34.49% [30.9–39.4] <sub>U</sub> | 20.53% [17.9–23.8] <sub>U</sub> | 120.86% [114.9–127.2] <sub>U</sub> |

**What the table says.**

- **Voxtral Mini bn-BD ‡** has the lowest error rate on SUBAK.KO and BanSpeech. Against the best public
  model it is clearly lower on both: −1.99 points [−2.74, −1.25] against Conformer Large on SUBAK.KO and
  −2.87 [−3.82, −1.94] against Whisper Medium on BanSpeech (paired, 95 % reference-clustered bootstrap).
  Against the other private models it is not separable: −0.36 [−0.81, +0.05] against Canary on SUBAK.KO,
  −0.04 [−0.94, +0.87] against Qwen3-ASR on BanSpeech.
- **Digits.** 288 of SUBAK.KO's 1,000 references and 72 of BanSpeech's write numbers as digits, which models
  that write numbers as words cannot match. On the SUBAK.KO recordings without digits: Voxtral Mini 5.42 %,
  Canary 5.57 %, Qwen3-ASR 7.16 %, Conformer Large 7.85 %, the hybrid FastConformer 8.35 %, Whisper turbo
  8.60 %, ehzawad FastConformer-CTC 13.67 % (`no_digit_ref` and `paired_no_digit_refs` in
  `outputs_multiset/summary_multiset.json`).
- **Common Voice** is not a clean head-to-head for the two public NeMo models: ehzawad FastConformer-CTC
  trained on Common Voice's "validated" clips, which can include these very test clips, and Conformer
  Large's card lists Common Voice. Among the models without that overlap, Canary (9.51 %), Whisper turbo
  (9.59 %) and Qwen3-ASR (9.88 %) lead.
- **Ben-10** is regional dialect speech written in dialect spelling, and every model misses most words.
  The hybrid FastConformer reads 60.9 %, Whisper turbo 62.2 % (403 of its 1,000 clips hit the 448-token
  limit), Qwen3-ASR 63.4 % and Conformer Large 63.5 %. Voxtral Mini (70.0 %) and Qwen3-ASR each ran to their
  500-token cap on 18 clips, mostly repeating a word; without them Voxtral Mini reads 61.2 %, a conditional
  diagnostic, since those clips stay in its score. Canary (76.4 %) produced 37 outputs more than 2.5× the
  reference's length. Greedy Whistle bn-BD on the A100 produced 104 such outputs, repeating a word or
  phrase up to its length cap, hence its 120.9 % (more insertions than reference words); the same
  checkpoint in its CPU engine produced 5 and reads 79.9 %.
- ehzawad FastConformer-CTC returned 24 empty hypotheses on Ben-10 and the hybrid FastConformer 23 on
  SUBAK.KO; they count as full errors.

## Official test splits beyond FLEURS

Vaani (ARTPARK-IISc, spontaneous image-description speech from Indian districts) and SPRING-INX R1/R2
(IIT Madras, conversational; about half the utterances contain Latin-script words) are not listed as
training data by any of the public models. Kathbath (AI4Bharat IndicSUPERB, read speech) and IndicVoices
(AI4Bharat, conversational and extempore speech) are corpora some of the models trained another split of.
Each set is a seeded, stratified 1,000-utterance sample of the official split (1.3–2.6 h of audio per
set); manifests with per-file SHA-256 are under [`multiset/`](multiset/), predictions under
[`outputs_multiset/`](outputs_multiset/).

**WER on the 1,000-utterance official-split subsets** (best per column in bold; 95 % intervals in brackets):

| Model | Vaani test | SPRING-INX R1 eval | SPRING-INX R2 eval | Kathbath (known spk) | Kathbath (unknown spk) | IndicVoices conversational |
|---|---:|---:|---:|---:|---:|---:|
| Voxtral Mini bn-BD ‡ | 22.08% [20.9–23.2] <sub>U</sub> | 36.87% [35.7–38.2] <sub>U</sub> | 40.01% [36.6–44.3] <sub>U</sub> | 10.77% [10.0–11.5] <sub>U</sub> | 10.82% [10.1–11.6] <sub>U</sub> | 13.40% [12.5–14.3] <sub>U</sub> |
| FastConformer hybrid bn-BD ‡ | 22.53% [21.4–23.7] <sub>U</sub> | 36.92% [35.7–38.2] <sub>U</sub> | 36.60% [35.4–37.9] <sub>U</sub> | 11.60% [10.8–12.4] <sub>U</sub> | 11.41% [10.6–12.2] <sub>U</sub> | 12.70% [11.9–13.6] <sub>U</sub> |
| Canary 1B v2 bn-BD ‡ | 21.92% [20.8–23.1] <sub>U</sub> | 36.79% [35.6–38.0] <sub>U</sub> | 36.75% [35.2–38.7] <sub>U</sub> | 10.96% [10.2–11.7] <sub>U</sub> | 11.00% [10.2–11.8] <sub>U</sub> | 12.75% [11.9–13.7] <sub>U</sub> |
| Qwen3-ASR 1.7B bn-BD ‡ | 22.68% [20.9–25.2] <sub>U</sub> | 37.62% [36.2–39.2] <sub>U</sub> | 36.89% [35.3–38.8] <sub>U</sub> | 10.26% [9.5–11.0] <sub>U</sub> | 10.37% [9.6–11.1] <sub>U</sub> | **11.99%** [11.1–12.9] <sub>U</sub> |
| Whisper turbo bn-BD ‡ | **21.39%** [20.3–22.5] <sub>U</sub> | **36.30%** [35.1–37.6] <sub>U</sub> | **36.13%** [34.8–37.5] <sub>U</sub> | 10.43% [9.7–11.1] <sub>U</sub> | 10.42% [9.7–11.2] <sub>U</sub> | 13.01% [12.0–14.0] <sub>U</sub> |
| ehzawad FastConformer-CTC | 24.67% [23.5–25.9] <sub>N</sub> | 39.23% [38.0–40.5] <sub>N</sub> | 38.42% [37.2–39.7] <sub>N</sub> | 13.38% [12.6–14.2] <sub>S</sub> | 13.43% [12.6–14.3] <sub>S</sub> | 12.61% [11.6–13.7] <sub>S</sub> |
| Conformer Large (hishab) | 33.10% [31.8–34.5] <sub>N</sub> | 42.46% [41.2–43.8] <sub>N</sub> | 42.25% [40.9–43.6] <sub>N</sub> | **9.33%** [8.6–10.1] <sub>S</sub> | **8.79%** [8.1–9.5] <sub>S</sub> | 29.68% [28.3–31.2] <sub>N</sub> |
| Whisper Medium | 26.35% [25.1–27.6] <sub>U</sub> | 40.04% [38.8–41.3] <sub>U</sub> | 39.71% [38.5–40.9] <sub>U</sub> | 9.70% [9.0–10.4] <sub>U</sub> | 9.91% [9.2–10.6] <sub>U</sub> | 24.96% [23.6–26.4] <sub>U</sub> |
| Wav2Vec2 | 31.28% [29.9–32.7] <sub>U</sub> | 47.28% [46.1–48.6] <sub>U</sub> | 47.32% [46.1–48.5] <sub>U</sub> | 19.07% [18.1–20.0] <sub>U</sub> | 18.80% [17.9–19.7] <sub>U</sub> | 33.02% [31.6–34.5] <sub>U</sub> |
| Whistle bn-BD ‡ (55M, CPU engine) | 27.43% [26.1–28.7] <sub>U</sub> | 43.38% [42.1–44.8] <sub>U</sub> | 42.19% [40.9–43.5] <sub>U</sub> | 15.61% [14.7–16.5] <sub>U</sub> | 15.82% [14.9–16.8] <sub>U</sub> | 19.62% [18.3–21.0] <sub>U</sub> |
| Whistle bn-BD ‡ (55M, A100, greedy) | 28.74% [27.3–30.3] <sub>U</sub> | 48.08% [45.5–51.1] <sub>U</sub> | 47.51% [44.6–50.7] <sub>U</sub> | 17.57% [16.0–19.5] <sub>U</sub> | 17.36% [15.9–19.3] <sub>U</sub> | 21.26% [19.8–22.9] <sub>U</sub> |

Provenance flags, from the model cards only: **N**, the card does not list this corpus; **S**, the model
trained on another split of this corpus; **U**, the card does not say either way. The ‡ models' training
data is undocumented, hence **U** everywhere. None of the three means "verified clean".

**What the table says.**

- The FLEURS ranking does not transfer. Conformer Large, the best public model on FLEURS (14.4 %), is the
  last public model on Vaani (33.1 %) and third of four on both SPRING sets. [Its model card](https://huggingface.co/hishab/titu_stt_bn_conformer_large)
  lists read and broadcast corpora (OpenSLR, Bengali.AI, MadASR, Kathbath, Common Voice, FLEURS,
  SUBAK.KO, Shrutilipi, Vasha-Bichita, IndicTTS) and none of Vaani, SPRING-INX or IndicVoices.
- Among the public models, `ehzawad/stt_bn_fastconformer_ctc` leads on Vaani and both SPRING sets, with
  Whisper Medium second on each.
- The private fine-tunes are lower still on those three sets, Whisper turbo lowest of all (21.4 / 36.3 /
  36.1 %), Voxtral Mini at 22.1 / 36.9 / 40.0 %. Voxtral Mini's SPRING R2 score includes three recordings
  that ran to its 500-token cap repeating one word: they hold 663 of its 8,143 word errors there (36.8 %
  without them).
- On Kathbath the order flips back: Conformer Large (9.3 / 8.8 %) and Whisper Medium (9.7 / 9.9 %) lead.
  Conformer Large's card lists Kathbath train and validation and the ehzawad CTC model trained on Kathbath
  train; a **U** cell can hide training overlap. Known- and unknown-speaker results differ by under a point
  for every model.
- **IndicVoices conversational is not a clean head-to-head for the ehzawad CTC model**, which trained on
  its train split. Qwen3-ASR reads 12.0 %, the other private models up to 13.4 % (Whistle bn-BD 19.6 % on
  CPU); Conformer Large (29.7 %, the one **N** cell) emitted 7 empty hypotheses there and Wav2Vec2 33.
- SPRING's 36–47 % WER overstates the real error rate: about half its references contain English words
  in Latin script, and no model reproduced them: no other model emitted a single Latin-script word, and
  Whistle bn-BD only leaks English fragments from its English base model (26 of 1,000 R1 and 18 of 1,000 R2
  transcripts on CPU, 18 and 15 on the A100, such as "aren" or "minutes"), which count as errors. On the
  Latin-free half the other private models read 31.5–37.7 % and `ehzawad/stt_bn_fastconformer_ctc`
  35.4 / 35.1 %.

## FLEURS Bengali test (920 utterances)

The familiar benchmark, for comparison with published numbers.

| Model | Checkpoint | Decoded with | WER (95% CI) | CER | ms/clip | Params |
|---|---|---|---|---|---:|---:|
| Canary 1B v2 bn-BD ‡ | `ehzawad/canary-1b-v2-bn-BD` | NeMo | **14.35%** [13.23–15.54] | 4.66% | 485 | 957.3M |
| Conformer Large | `hishab/titu_stt_bn_conformer_large` | NeMo, CTC | 14.37% [13.11–15.73] | 4.48% | 87 | 121.5M |
| Qwen3-ASR 1.7B bn-BD ‡ | `ehzawad/Qwen3-ASR-1.7B-bn-BD` | vLLM | 14.53% [13.42–15.72] | 4.74% | 656 | 2,038M |
| Voxtral Mini bn-BD ‡ | `ehzawad/Voxtral-Mini-3B-2507-bn-BD` | vLLM | 14.73% [13.60–15.93] | 4.78% | 438 | 4,676M |
| Whisper turbo bn-BD ‡ | `ehzawad/whisper-large-v3-turbo-bn-BD` | vLLM | 15.08% [13.88–16.40] | 5.28% | 700 | 808.9M |
| FastConformer hybrid bn-BD ‡ | `ehzawad/fastconformer-hybrid-rnnt-ctc-large-bn-BD` | NeMo, RNN-T | 15.38% [14.26–16.54] | 4.91% | 102 | 114.7M |
| Whisper Medium | `SayedShaun/bengali-whisper-medium` | vLLM | 16.24% [15.07–17.46] | 5.09% | 304 | 763.9M |
| **ehzawad FastConformer-CTC** | `ehzawad/stt_bn_fastconformer_ctc` | NeMo, CTC | 17.12% [15.95–18.34] | 5.12% | 88 | 115.6M |
| Whistle bn-BD ‡ (55M, CPU engine) | `ehzawad/whistle-bn-BD` | needle engine, CPU | 19.17% [17.93–20.46] | 6.27% | 578 (CPU) | 57.2M |
| Wav2Vec2 | `SayedShaun/bangla-wave2vec2-unigram` | PyTorch, CTC | 20.66% [19.47–21.90] | 6.02% | 56 | 315.5M |
| Whistle bn-BD ‡ (55M, A100, greedy) | `ehzawad/whistle-bn-BD` | JAX, A100-80GB | 20.75% [19.26–22.48] | 7.79% | 139 | 57.2M |

Greedy decoding, batch 1, no external language model (Whistle bn-BD on CPU: its engine's own decoding).
No clip failed and no hypothesis came back empty for any model. `ms/clip` is the mean single-clip latency
on the A100 (the Colab A100-SXM4-40GB; Whistle bn-BD on an A100-SXM4-80GB, plus its CPU row):

| Model | mean ms/clip | median | p95 |
|---|---:|---:|---:|
| Wav2Vec2 | 56 | 54 | 86 |
| Conformer Large (hishab) | 87 | 83 | 96 |
| ehzawad FastConformer-CTC | 88 | 84 | 98 |
| FastConformer hybrid bn-BD ‡ | 102 | 98 | 119 |
| Whistle bn-BD ‡ (55M), A100-80GB, JAX | 139 | 131 | 235 |
| Whisper Medium | 304 | 288 | 461 |
| Voxtral Mini bn-BD ‡ | 438 | 420 | 680 |
| Canary 1B v2 bn-BD ‡ | 485 | 463 | 740 |
| Qwen3-ASR 1.7B bn-BD ‡ | 656 | 621 | 1,028 |
| Whisper turbo bn-BD ‡ | 700 | 666 | 1,108 |
| Whistle bn-BD ‡ (55M), CPU engine, one core | 578 | 530 | 1,012 |

These numbers time the configured pipelines, not the models in isolation:

- vLLM 0.31 (torch 2.13, CUDA graphs) for Voxtral Mini, Whisper Medium, Whisper turbo and Qwen3-ASR; NeMo
  2.7.3 or PyTorch (torch 2.11) for the rest. Whistle bn-BD on the A100: JAX 0.12 with XLA, see
  [On-device model: Whistle bn-BD](#on-device-model-whistle-bn-bd).
- The CTC and RNN-T models emit a whole piece in one pass; the others generate text token by token, so
  their time grows with the length of what was said.
- A few milliseconds apart is not an established ordering, and p95 is variation across clips, not
  uncertainty in the mean.

**Test split only.** The ehzawad CTC model trained on a corpus that includes the FLEURS *train* split,
which overlaps the FLEURS *validation* split, so the 402 validation recordings are excluded for every
model. It had also been evaluated on FLEURS test during its own development, so read its FLEURS row as a
like-for-like re-measurement and the other corpora above as the fresh evidence.

Paired comparisons for `ehzawad/stt_bn_fastconformer_ctc` on this split: +0.88 points behind Whisper
Medium [+0.27, +1.48] at 3.5× its speed; +2.74 points behind Conformer Large [+2.01, +3.46].

Voxtral Mini bn-BD ‡ on this split is not separable from Canary (+0.38 [−0.19, +0.95]), Conformer Large
(+0.35 [−0.36, +1.06]), Qwen3-ASR (+0.20 [−0.35, +0.74]) or Whisper turbo (−0.35 [−1.04, +0.32]), and
lower than the hybrid FastConformer (−0.65 [−1.21, −0.08]), Whisper Medium (−1.51 [−2.15, −0.88]), the
ehzawad CTC model (−2.39 [−3.03, −1.74]) and Wav2Vec2. Like Conformer Large and Whisper Medium, it writes
numbers as Bengali words; the scorer counts these as errors against the 187 references that use digits.

## Voxtral Mini against the best other model

Voxtral Mini bn-BD minus the model with the lowest WER on each set, on the same recordings (WER points,
95 % reference-clustered bootstrap, 10,000 draws; pointwise, not adjusted for multiple comparisons):

| Set | Voxtral Mini | Best other model | Voxtral Mini minus it, WER points (95% CI) |
|---|---:|---|---|
| FLEURS test | 14.73% | Canary 1B v2 bn-BD ‡ 14.35% | +0.38 [−0.19, +0.95], inconclusive |
| Vaani test | 22.08% | Whisper turbo bn-BD ‡ 21.39% | +0.69 [+0.10, +1.30], higher |
| SPRING-INX R1 eval | 36.87% | Whisper turbo bn-BD ‡ 36.30% | +0.57 [+0.05, +1.09], higher |
| SPRING-INX R2 eval | 40.01% | Whisper turbo bn-BD ‡ 36.13% | +3.88 [+0.80, +7.96], higher |
| Kathbath (known spk) | 10.77% | Conformer Large (hishab) 9.33% | +1.44 [+0.77, +2.12], higher |
| Kathbath (unknown spk) | 10.82% | Conformer Large (hishab) 8.79% | +2.03 [+1.37, +2.71], higher |
| IndicVoices conversational | 13.40% | Qwen3-ASR 1.7B bn-BD ‡ 11.99% | +1.41 [+0.81, +2.02], higher |
| SUBAK.KO test | 11.66% | Canary 1B v2 bn-BD ‡ 12.02% | −0.36 [−0.81, +0.05], inconclusive |
| BanSpeech | 20.72% | Qwen3-ASR 1.7B bn-BD ‡ 20.76% | −0.04 [−0.94, +0.87], inconclusive |
| Common Voice 17 test | 10.32% | Conformer Large (hishab) 7.35% | +2.97 [+2.24, +3.70], higher |
| Ben-10 validation (dialects) | 69.97% | FastConformer hybrid bn-BD ‡ 60.91% | +9.07 [+4.88, +13.96], higher |

Against each of the eight other models on every set, see `paired` in
`outputs_multiset/summary_multiset.json` and `paired_voxtral_mini_bn` in
`outputs_fleurs/clean_test_table.json`.

## On-device model: Whistle bn-BD

`ehzawad/whistle-bn-BD` ‡ is a full fine-tune of Cactus Compute's 55M-parameter Whistle (57.2M with its added
Bengali vocabulary), deployed as a 4-bit `.cact` archive
(30.5 MB) for the cactus-needle engine. The same checkpoint was measured under two runtimes, on 21-22 Oct
2026, and both rows are kept:

- **A100, JAX, greedy** (`whistle_bn_bd_a100`), comparable with the other A100 rows. The needle engine is
  CPU-only and vLLM does not support Whistle's architecture, so the GPU runtime is needle's own JAX model
  (JAX 0.12, XLA) on one A100-SXM4-80GB (the 40 GB cards were queued for hours), alone on the card. It uses
  the deployed numerics (4-bit weights, 8-bit activations and KV, bfloat16 compute) and the benchmark's
  protocol: greedy, language tag `bn` forced, one clip at a time, the `asr_core.py` pieces staged as PCM16,
  a warm-up clip discarded, each clip timed from audio loading to final text with the audio on node-local
  disk. Every jitted shape is compiled before timing, the encoder input is padded to the next 2 s instead of
  30 s, and the greedy loop runs on the device (one host sync per 64-token decoder bucket); the decoder
  recomputes its prefix at each step, as needle's Whistle has no KV-cache decoding path. A length cap of 12
  tokens per second of audio stops runaway outputs.
- **CPU, needle engine** (`whistle_bn_bd`), the on-device cost: the `.cact` file in the official needle
  engine on one core of an AMD EPYC 9575F, with the engine's own decoding (beam search), language `bn`. Its
  speed is not comparable with the A100 rows.
- **Why the accuracy differs.** Greedy decoding is worse than the engine's beam search for this small model:
  20.75 % against 19.17 % WER on FLEURS and 1.3–5.3 points higher on the other sets, except Ben-10
  (120.9 % against 79.9 %). Much of the gap is runaway repetition: outputs more than 2.5× the reference's
  length number 1 (FLEURS), 8 (SUBAK.KO), 13 and 15 (SPRING R1/R2) and 104 (Ben-10) for greedy on the A100,
  against 0, 2, 2, 6 and 5 for the engine. Without those clips greedy trails the engine by about a point
  (FLEURS 20.26 % against 19.17 %, SUBAK.KO 19.86 % against 18.76 %), a conditional diagnostic; the clips
  stay in both scores. As a check of the GPU runner, a batched greedy decode of the same checkpoint (batch
  32, 30 s padding) gives the identical transcript on 751 of 920 FLEURS clips and 20.27 % WER by the
  bn-whistle scorer, against 20.75 % for the per-clip runner; the other transcripts differ through
  bfloat16 numerics of the different shapes (batch 32 and 30 s against batch 1 and 2 s buckets).
- **Same audio path and scorer** for both: the vendored `asr_core.py` pieces (identical piece counts and
  forced cuts on every clip, checked on import), texts joined with a space, scored by the same
  `bench_score.norm` and jiwer.

## How the comparison was run

- **One GPU, one model resident at a time**, a Colab A100-SXM4-40GB, on 13 Oct 2026 (UTC). The
  official-split runner refuses to start while another compute process holds the card. Every run records
  the card's state before and after. `run_all.sh` lists every decode behind the committed outputs.
  Whistle bn-BD is the exception (an A100-SXM4-80GB and a CPU, 21-22 Oct 2026); its runs
  come from the private bn-whistle repository and were imported with the same checks.
- **One audio path.** `asr_core.py` loads, downmixes and resamples to 16 kHz, cuts the audio at pauses into
  pieces of at most 25 s and stages them as PCM16; every backend reads those staged files. The nine models'
  runs record identical piece counts and forced cuts on all 11,322 clips.
- **One scorer.** `bench_score.norm` strips punctuation to word boundaries and applies Unicode NFC before
  WER/CER. NFC is not cosmetic: Bengali writes য়, ড় and ঢ় either precomposed or as base + nukta.
  - The FLEURS references and the NeMo models use the decomposed form; Whisper and wav2vec2 use the
    precomposed one.
  - Without NFC, every such character scores as a substitution: Whisper Medium reads 27.9 % instead of
    16.2 %.
  - If your own measurement of these models disagrees with a published number by ten points or more, this
    is usually why.
- **Greedy decoding, batch 1, no external LM** for every model (Whistle bn-BD on CPU: its engine's own
  decoding). This is the interactive-latency workload,
  applied identically to every decoder type. Batching would help the autoregressive models far more; it is
  not measured here for anyone.
- **The autoregressive decoders run through vLLM 0.31**, one piece per request, with the prompt vLLM's own
  transcription endpoint builds for each architecture:
  - Voxtral Mini: language `bn`, at most 500 new tokens. Against Hugging Face transformers at batch 1 on six
    FLEURS clips, the text was identical on all six.
  - Whisper Medium and Whisper turbo: decoder prompt `<|startoftranscript|><|bn|><|transcribe|><|notimestamps|>`,
    up to Whisper's 448 decoder positions.
  - Qwen3-ASR: no forced language, as its card prescribes; the model writes its language tag and vLLM's
    clean-up strips it. At most 500 new tokens.
  - Whisper Medium was first decoded with transformers; under vLLM its FLEURS WER moved from 16.22 % to
    16.24 %, with 882 of 920 transcripts identical.
- **Canary** runs in NeMo as Bengali-to-Bengali transcription without the punctuation flag (`pnc="no"`),
  beam size 1; the **hybrid FastConformer** with its default RNN-T decoder (greedy).
- **Timing.** One warm-up clip is discarded. Each clip is then timed between CUDA synchronisations, with
  `asr_core` loading, chunking and staging included and model startup excluded. Runs were niced. Other
  activity on the VM during the runs was limited to brief checks over SSH (log tails, a few seconds of
  scoring) and a 10-minute mirror of finished outputs to Google Drive; its effect was not measured.
- **Every clip counts.** No model raised on a clip. Empty hypotheses and runaway outputs stay in the
  denominator as full errors.
- **Checkpoints pinned** by commit hash in `outputs_fleurs/checkpoint_revisions.json`.
- **Intervals** are 95 % bootstraps over distinct references, because several speakers may read the same
  sentence. That matters on FLEURS (349 distinct sentences over 920 recordings). In the 1,000-utterance
  subsets the references are nearly all distinct, except IndicVoices conversational (871), whose short
  replies repeat. There the two kinds of interval barely differ.

## Limitations

- FLEURS, Kathbath and SUBAK.KO's studio part are read speech; Vaani, SPRING, IndicVoices conversational
  and BanSpeech are largely spontaneous. There is no telephony and no long-form recording.
- Seven of the eleven sets are Indian Bengali. SUBAK.KO, BanSpeech and Ben-10 are Bangladeshi; Common
  Voice records no speaker region. Ben-10 is its validation split, as its test split is private.
- Speed is per-clip latency at batch 1 on one GPU model (Whistle bn-BD: an A100 80 GB card, the same
  chip with more memory), from one pass per model and set. Throughput
  under load, batched decoding and CPU inference (except Whistle bn-BD's CPU row) are not measured.
- Cell intervals on the 1,000-utterance subsets run from about ±0.7 points (Kathbath) to ±1.5 points
  (Vaani, SPRING, IndicVoices). A few cells are wider where a handful of runaway outputs carry many errors
  (Voxtral Mini's SPRING R2 and Ben-10, Canary's Ben-10, Qwen3-ASR's SUBAK.KO, greedy Whistle's SUBAK.KO,
  BanSpeech, Common Voice and SPRING). Differences inside an
  interval are not differences.
- Training provenance is only as good as the model cards; a **U** flag means it could not be established
  either way. The ‡ models are not public and their training data is undocumented.

## Reproducing this

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
python build_eval.py                              # FLEURS bn_in test + validation -> eval_manifest.json, eval_fleurs_bn/
python build_multiset_bd.py --out <subset audio dir>   # SUBAK.KO, BanSpeech, Common Voice 17, Ben-10
# every model on FLEURS and the ten subsets, one at a time -> outputs_fleurs/, outputs_multiset/
BENCH_GPU=<GPU UUID> BENCH_PY=venv/bin/python BENCH_PY_VLLM=<python with vllm[audio]==0.31.0> \
  FLEURS_ASSETS=. MULTISET_ASSETS=<subset audio dir> ./run_all.sh
python bench_score.py && python clean_split_table.py
python score_multiset.py
python make_report.py                             # -> asr_benchmark.pdf
```

- `MULTISET_ASSETS/<set>/eval_manifest.local.json` lists the same rows as `multiset/<set>/eval_manifest.json`,
  with local paths. `bench_run_set.py` checks every file's SHA-256 before decoding.
- `MODELS="..."` limits `run_all.sh` to some models. The ‡ models are private: they need a Hugging Face
  token with access, and Voxtral Mini also `VOXTRAL_MINI=<local snapshot of the model>`.
- `run_all.sh` skips a run whose `run_meta.json` exists and never overwrites one.

Subset sampling: seed 20260915, proportional stratification (Vaani by state/district, SPRING by
Latin/no-Latin reference, Kathbath by speaker gender, IndicVoices by scenario, SUBAK.KO by corpus folder,
BanSpeech by domain, Common Voice by the gender field, Ben-10 by district). The subset audio is not
redistributed here; `multiset/*/eval_manifest.json` lists every selected utterance with its SHA-256 so a
rebuild can be verified.

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
bench_run.py             one model over the FLEURS set, batch 1, timed; backends for NeMo, Canary, wav2vec2,
                         Whisper (transformers and vLLM), Voxtral and Qwen3-ASR (vLLM)
bench_run_set.py         the same backends over any eval manifest (the subset runs)
build_multiset_bd.py     the SUBAK.KO, BanSpeech, Common Voice 17 and Ben-10 subsets
bench_score.py           predictions -> WER/CER with bootstrap intervals -> summary.json
clean_split_table.py     the FLEURS test-split table with paired deltas
score_multiset.py        the subset tables, diagnostics, and Voxtral Mini's paired comparisons
run_all.sh               every run, one model at a time, GPU pinned by UUID
make_report.py           summaries -> asr_benchmark.pdf
bench_server.py          HTTP wrapper to run any of these checkpoints
app.py, run.sh           the interactive web demo
build_eval.py, prove_audio_path.py, eval_manifest.json   the FLEURS evaluation set
multiset/                the subsets with per-file SHA-256
outputs_fleurs/          FLEURS predictions, run metadata, summaries
outputs_multiset/        subset predictions, run metadata, summary
requirements.txt         pinned environment for the NeMo and PyTorch models
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
  - [SUBAK.KO](https://huggingface.co/datasets/SUST-CSE-Speech/SUBAK.KO) (Dept. of CSE, SUST), CC-BY-4.0.
  - [BanSpeech](https://huggingface.co/datasets/SUST-CSE-Speech/banspeech) (Dept. of CSE, SUST),
    CC-BY-NC-4.0 per its card metadata (CC-BY-4.0 in its card text).
  - [Common Voice](https://commonvoice.mozilla.org) 17.0 (Mozilla), CC0, read from the
    `fixie-ai/common_voice_17_0` mirror.
  - [Ben-10](https://huggingface.co/datasets/bengaliAI/Ben-10) (Bengali.AI), CC0.
  - The committed manifests and predictions carry reference transcripts alongside each model's
    hypothesis.
- `hishab/titu_stt_bn_conformer_large`: Hishab.
- `SayedShaun/bengali-whisper-medium` (mirror of tugstugi's Bengali.AI competition model),
  `SayedShaun/bangla-wave2vec2-unigram` (mirror of qdv206's).
- `ehzawad/stt_bn_fastconformer_ctc`: FastConformer-CTC large trained on 1,692 hours of
  human-supervised Bengali with a 1,024-piece Bengali BPE vocabulary, CC-BY-SA-4.0. Base model
  `nvidia/stt_en_fastconformer_ctc_large`, NVIDIA, CC-BY-4.0.
- The private ‡ fine-tunes, under the ehzawad Non-Commercial Model License except where noted:
  - `ehzawad/Voxtral-Mini-3B-2507-bn-BD`, from `mistralai/Voxtral-Mini-3B-2507` (Mistral AI, Apache-2.0).
  - `ehzawad/canary-1b-v2-bn-BD`, from `nvidia/canary-1b-v2` (NVIDIA, CC-BY-4.0).
  - `ehzawad/Qwen3-ASR-1.7B-bn-BD`, from `Qwen/Qwen3-ASR-1.7B` (Qwen team, Alibaba Cloud, Apache-2.0).
  - `ehzawad/whisper-large-v3-turbo-bn-BD`, from `openai/whisper-large-v3-turbo` (OpenAI, MIT).
  - `ehzawad/fastconformer-hybrid-rnnt-ctc-large-bn-BD`, from `nvidia/stt_en_fastconformer_hybrid_large_pc`
    (NVIDIA, CC-BY-4.0).
  - `ehzawad/whistle-bn-BD`, from `Cactus-Compute/whistle` (Cactus Compute, Apache-2.0).

Model licences are those of their respective publishers. Check each model card before use.

Author: Emrul Zawad (ehzawad).
