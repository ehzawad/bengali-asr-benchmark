# Bengali ASR benchmark

A controlled comparison of twelve Bengali speech-to-text models on eleven evaluation sets. Four models are
public checkpoints; eight are the author's private bn-BD fine-tunes, marked ‡, one of them a
55M-parameter on-device model (Whistle bn-BD) measured twice, on an A100 and on CPU. The sets are:

- the FLEURS Bengali test split;
- 1,000-utterance samples of six Indian sets: Vaani, SPRING-INX R1 and R2, Kathbath (known- and
  unknown-speaker) and IndicVoices conversational;
- 1,000-utterance samples of four sets added on 14 Oct 2026: SUBAK.KO and BanSpeech (Bangladeshi standard
  Bangla), Common Voice 17 (crowd-sourced) and Ben-10 (Bangladeshi regional dialects).

**Every score in this README is on the cleaned subsets of these sets**: a clip is scored only if it is at
least 0.5 s long and its reference contains no digit, which keeps 10,291 of the 10,920 test clips. The rule,
the reason for it and the exact clip lists are under [Cleaned subsets](#cleaned-subsets).

Every decode ran on one Colab A100-SXM4-40GB, one model at a time, greedy at batch 1, through the same
audio path and the same scorer, so accuracy and speed are comparable across the tables. The exceptions are
Whistle bn-BD, run later on an A100-SXM4-80GB in JAX (greedy, batch 1) and on CPU in its own engine (see
[On-device model: Whistle bn-BD](#on-device-model-whistle-bn-bd)), Granite Speech 5.0 2B bn-BD, added
on 23 Oct 2026 and decoded with the same harness on another A100-SXM4-40GB, and Rumble-2-Flash bn-BD, added
on 20 Nov 2026 in the same way on another A100-SXM4-40GB. Every other autoregressive decoder (Voxtral Mini,
Whisper Medium, Whisper turbo, Qwen3-ASR, Granite Speech, Rumble-2-Flash) ran through vLLM; the CTC and
RNN-T models, Canary and Wav2Vec2 through NeMo or PyTorch. Rumble-2-Flash, Voxtral Mini and Qwen3-ASR were
also decoded together on one A100 and on one B200 (see
[Three-model comparison on one GPU](#three-model-comparison-on-one-gpu)).

The ‡ models are private or gated, so not every reader can run them; they are here to place the author's
own fine-tunes in the public field.

Report: [`asr_benchmark.pdf`](asr_benchmark.pdf), a short summary centred on Voxtral Mini bn-BD on the full
sets (Rumble-2-Flash included, Granite Speech not), plus one page for the three-model comparison, rendered
by `make_report.py` from the committed summaries. Its first three pages predate the cleaned subsets. This
README has the current tables and intervals.

## Cleaned subsets

**Rule.** A clip is removed if it is shorter than 0.5 s (the manifest's duration) or if any character of
its reference is a digit (Python `str.isdigit()`: ASCII 0–9 and Bengali ০–৯). Nothing else changes: the
references that stay, every model's hypotheses and the scorer are those of the full sets.

**Why.**

- A clip under half a second holds a word or two, so it says little about recognition and a single error
  swings its score. All 82 such clips are IndicVoices conversational replies.
- A reference that writes a number in digits cannot be matched by a model that writes numbers as Bengali
  words, which is what most of these models do, so those clips measured number formatting rather than
  recognition: 187 FLEURS test, 288 SUBAK.KO and 72 BanSpeech references.

**Same clips for every model.** The rule reads only the manifests (duration and reference), never a model's
output.

| Set | Full | Removed (< 0.5 s) | Removed (digits) | Scored |
|---|---:|---:|---:|---:|
| FLEURS test | 920 | 0 | 187 | 733 |
| Vaani test | 1,000 | 0 | 0 | 1,000 |
| SPRING-INX R1 eval | 1,000 | 0 | 0 | 1,000 |
| SPRING-INX R2 eval | 1,000 | 0 | 0 | 1,000 |
| Kathbath (known spk) | 1,000 | 0 | 0 | 1,000 |
| Kathbath (unknown spk) | 1,000 | 0 | 0 | 1,000 |
| IndicVoices conversational | 1,000 | 82 | 0 | 918 |
| SUBAK.KO test | 1,000 | 0 | 288 | 712 |
| BanSpeech | 1,000 | 0 | 72 | 928 |
| Common Voice 17 test | 1,000 | 0 | 0 | 1,000 |
| Ben-10 validation (dialects) | 1,000 | 0 | 0 | 1,000 |
| All | 10,920 | 82 | 547 | 10,291 |

- **Clip lists:** [`clean_subsets/`](clean_subsets/)`<set>.json` lists the kept clips and the removed ones
  under each reason, by `utt_id` (FLEURS: by file name). `score_clean.py` rebuilds the lists from the
  manifests, stops if a committed list disagrees, and re-scores every model's committed predictions on the
  kept clips with `bench_score.norm` and jiwer, with 95 % bootstraps over distinct references, into
  [`outputs_clean/summary_clean.json`](outputs_clean/summary_clean.json).
- **Effect:** cleaning lowers each model's average WER over the eleven sets by 0.4–1.0 points and leaves
  the order nearly unchanged.
- **Full sets:** their scores stay in `outputs_fleurs/summary.json`, `outputs_fleurs/clean_test_table.json`
  and `outputs_multiset/summary_multiset.json`.

## Which one to use

Public models:

- **`ehzawad/stt_bn_fastconformer_ctc`**: the best public model on Vaani and both SPRING sets, corpora no
  public model's card lists, with 14.6 % WER on FLEURS at 88 ms per clip.
- **`hishab/titu_stt_bn_conformer_large`**: the lowest WER of all models on FLEURS (10.8 %), both Kathbath
  sets and Common Voice, i.e. read speech, at the same latency. Its card lists Kathbath, Common Voice and
  FLEURS as training data, and it is **last on Vaani** (33.1 %).
- **Whisper Medium**: second of the public models on most sets and the best public model on BanSpeech
  (21.9 %), at 304 ms per clip under vLLM.

All models by average rank over the eleven cleaned sets:

| Model | Average rank (1 is best) | Sets with the lowest WER | Average WER | Average CER |
|---|---:|---|---:|---:|
| Whisper turbo bn-BD ‡ | 3.4 | Vaani, SPRING R1, SPRING R2 | 21.9% | 9.9% |
| Qwen3-ASR 1.7B bn-BD ‡ | 3.5 | IndicVoices | 21.9% | 9.8% |
| Canary 1B v2 bn-BD ‡ | 3.9 | — | 23.1% | 10.4% |
| Voxtral Mini bn-BD ‡ | 4.6 | SUBAK.KO, BanSpeech | 22.8% | 10.2% |
| FastConformer hybrid bn-BD ‡ | 5.2 | Ben-10 | 22.4% | 9.8% |
| Conformer Large (hishab) | 6.1 | FLEURS, Kathbath ×2, Common Voice (trained on all four) | 25.5% | 10.7% |
| Rumble-2-Flash bn-BD ‡ | 6.2 | — | 22.9% | 9.9% |
| Granite Speech 5.0 2B bn-BD ‡ | 6.9 | — | 23.6% | 10.6% |
| Whisper Medium | 7.2 | — | 26.1% | 12.5% |
| ehzawad FastConformer-CTC | 8.6 | — | 25.9% | 11.6% |
| Whistle bn-BD ‡ (55M, CPU engine) | 10.5 | — | 29.2% | 14.2% |
| Whistle bn-BD ‡ (55M, A100, greedy) | 11.2 | — | 35.6% | 20.5% |
| Wav2Vec2 | 11.8 | — | 33.2% | 13.1% |

Ranks and averages weight the eleven sets equally; Ben-10's 60–80 % error rates dominate the average WER
(121 % for greedy Whistle on the A100, see below). The two Whistle rows are one checkpoint under two
runtimes, so each is ranked against the eleven other models, not against the other; the other ranks are
those with the CPU row.
Several leads are inside their intervals (see [Paired comparisons](#paired-comparisons)).

- **Whisper turbo bn-BD** is the most accurate on Indian conversational speech, but its multilingual
  tokenizer spends about 13 tokens per Bengali word, so its longest pieces run into the 448-token decoder
  limit (36 of the full 920 FLEURS test clips, 403 of 1,000 Ben-10 clips), and it is among the slowest
  models here (700 ms).
- **Voxtral Mini bn-BD** has the lowest error rate on the two Bangladeshi standard-Bangla sets, SUBAK.KO and
  BanSpeech, though not separably from Canary on SUBAK.KO or Qwen3-ASR on BanSpeech.
- **FastConformer hybrid bn-BD** is best on the Ben-10 dialects and runs in 0.1 s per clip, 2–9× faster
  than the other private models except Whistle bn-BD (139 ms on the A100).
- **Rumble-2-Flash bn-BD**, the newest and at 508.6M parameters the smallest of the models decoded through
  vLLM, is seventh by average rank (22.9 % average WER, the fifth lowest). It leads on no set: the best
  other model is lower by 0.9 points on BanSpeech (inside the interval) up to 5.1 points on Ben-10, outside
  the interval on every other set. It is the fastest of the vLLM decoders on the A100 (211 ms per FLEURS
  clip); on a B200 Voxtral Mini is faster (see
  [Three-model comparison on one GPU](#three-model-comparison-on-one-gpu)).
- **Granite Speech 5.0 2B bn-BD** is eighth by average rank (23.6 % average WER). It leads on
  no set: the best other model is lower by 0.8 points on SUBAK.KO (its closest set, 6.2 %) up to 8.5 points on
  Ben-10, each outside its interval, and it is the slowest model here (961 ms per FLEURS clip).

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

**WER on the four sets, cleaned subsets** (712, 928, 1,000 and 1,000 clips; best per column in bold; 95 %
intervals in brackets; flags as below):

| Model | SUBAK.KO test | BanSpeech | Common Voice 17 test | Ben-10 validation (dialects) |
|---|---:|---:|---:|---:|
| Whisper turbo bn-BD ‡ | 8.60% [7.0–11.0] <sub>U</sub> | 20.08% [18.6–21.7] <sub>U</sub> | 9.59% [8.8–10.4] <sub>U</sub> | 62.21% [60.8–63.6] <sub>U</sub> |
| Qwen3-ASR 1.7B bn-BD ‡ | 7.16% [5.4–10.2] <sub>U</sub> | 19.37% [18.0–20.8] <sub>U</sub> | 9.88% [8.4–12.0] <sub>U</sub> | 63.39% [61.2–65.7] <sub>U</sub> |
| Canary 1B v2 bn-BD ‡ | 5.57% [4.9–6.3] <sub>U</sub> | 21.19% [19.7–22.8] <sub>U</sub> | 9.51% [8.7–10.3] <sub>U</sub> | 76.42% [71.9–81.3] <sub>U</sub> |
| Voxtral Mini bn-BD ‡ | **5.42%** [4.8–6.1] <sub>U</sub> | **19.24%** [18.0–20.5] <sub>U</sub> | 10.32% [9.5–11.2] <sub>U</sub> | 69.97% [65.5–75.1] <sub>U</sub> |
| FastConformer hybrid bn-BD ‡ | 8.35% [7.4–9.4] <sub>U</sub> | 22.03% [20.6–23.6] <sub>U</sub> | 10.51% [9.7–11.4] <sub>U</sub> | **60.91%** [59.7–62.1] <sub>U</sub> |
| Conformer Large (hishab) | 7.85% [6.9–8.8] <sub>S</sub> | 25.44% [23.9–27.1] <sub>N</sub> | **7.35%** [6.7–8.0] <sub>S</sub> | 63.54% [62.3–64.8] <sub>N</sub> |
| Rumble-2-Flash bn-BD ‡ | 7.83% [7.1–8.6] <sub>U</sub> | 20.11% [18.7–21.6] <sub>U</sub> | 11.18% [10.3–12.1] <sub>U</sub> | 65.96% [64.0–67.9] <sub>U</sub> |
| Granite Speech 5.0 2B bn-BD ‡ | 6.23% [5.3–7.3] <sub>U</sub> | 22.91% [21.4–24.5] <sub>U</sub> | 10.99% [10.1–11.9] <sub>U</sub> | 69.39% [67.5–71.4] <sub>U</sub> |
| Whisper Medium | 16.48% [15.6–17.4] <sub>U</sub> | 21.93% [20.5–23.4] <sub>U</sub> | 10.12% [9.3–11.0] <sub>U</sub> | 74.79% [73.6–75.9] <sub>U</sub> |
| ehzawad FastConformer-CTC | 13.67% [12.5–14.8] <sub>S</sub> | 26.66% [25.1–28.3] <sub>N</sub> | 12.14% [11.2–13.0] <sub>S</sub> | 76.92% [75.8–78.0] <sub>N</sub> |
| Whistle bn-BD ‡ (55M, CPU engine) | 13.36% [12.3–14.5] <sub>U</sub> | 30.28% [28.4–32.3] <sub>U</sub> | 17.10% [16.0–18.2] <sub>U</sub> | 79.87% [78.3–81.4] <sub>U</sub> |
| Whistle bn-BD ‡ (55M, A100, greedy) | 18.84% [15.1–23.3] <sub>U</sub> | 32.69% [29.1–38.0] <sub>U</sub> | 20.53% [17.9–23.8] <sub>U</sub> | 120.86% [114.9–127.2] <sub>U</sub> |
| Wav2Vec2 | 21.55% [20.6–22.5] <sub>U</sub> | 31.33% [29.6–33.0] <sub>U</sub> | 16.70% [15.7–17.7] <sub>U</sub> | 80.50% [79.5–81.5] <sub>U</sub> |

**What the table says.**

- **Voxtral Mini bn-BD ‡** has the lowest error rate on SUBAK.KO and BanSpeech. Against the best public
  model it is clearly lower on both: −2.43 points [−3.24, −1.66] against Conformer Large on SUBAK.KO and
  −2.69 [−3.67, −1.70] against Whisper Medium on BanSpeech (paired, 95 % reference-clustered bootstrap).
  Against the other private models it is not separable: −0.14 [−0.53, +0.24] against Canary on SUBAK.KO,
  −0.13 [−1.10, +0.83] against Qwen3-ASR on BanSpeech.
- **Granite Speech bn-BD ‡** is third on SUBAK.KO (6.23 %, +0.81 [+0.19, +1.58] behind Voxtral Mini) and
  eighth on BanSpeech (22.91 %).
- **Rumble-2-Flash bn-BD ‡** is fifth on SUBAK.KO (7.83 %, +2.40 [+1.95, +2.87] behind Voxtral Mini) and
  fourth on BanSpeech (20.11 %), not separable from Voxtral Mini there (+0.87 [−0.08, +1.83]).
- **Digits.** The 288 SUBAK.KO and 72 BanSpeech references that write numbers as digits are outside the
  cleaned subsets (see [Cleaned subsets](#cleaned-subsets)). With them SUBAK.KO reads 4.5–6.5 points
  higher for the other public and private models, but only 0.7 points higher for the ehzawad CTC model
  (full-set scores in `outputs_multiset/summary_multiset.json`).
- **Common Voice** is not a clean head-to-head for the two public NeMo models: ehzawad FastConformer-CTC
  trained on Common Voice's "validated" clips, which can include these very test clips, and Conformer
  Large's card lists Common Voice. Among the models without that overlap, Canary (9.51 %), Whisper turbo
  (9.59 %) and Qwen3-ASR (9.88 %) lead.
- **Ben-10** is regional dialect speech written in dialect spelling, and every model misses most words.
  The hybrid FastConformer reads 60.9 %, Whisper turbo 62.2 % (403 of its 1,000 clips hit the 448-token
  limit), Qwen3-ASR 63.4 % and Conformer Large 63.5 %. Voxtral Mini (70.0 %) and Qwen3-ASR each ran to their
  500-token cap on 18 clips, mostly repeating a word; without them Voxtral Mini reads 61.2 %. Granite Speech
  (69.4 %) ran to its length cap on 42 clips; without them it reads 65.9 %. Rumble-2-Flash (66.0 %) ran to
  its 500-token cap on 15 clips; without them it reads 64.1 %. These are conditional diagnostics, since
  those clips stay in the scores. Canary (76.4 %) produced 37 outputs more than 2.5× the reference's length. Greedy Whistle bn-BD on the A100 produced 104 such outputs, repeating a word or
  phrase up to its length cap, hence its 120.9 % (more insertions than reference words); the same
  checkpoint in its CPU engine produced 5 and reads 79.9 %.
- ehzawad FastConformer-CTC returned 24 empty hypotheses on Ben-10, the hybrid FastConformer 5 on the
  cleaned SUBAK.KO and 6 on the cleaned BanSpeech, and Rumble-2-Flash 2 on the cleaned SUBAK.KO; they count
  as full errors.

## Official test splits beyond FLEURS

Vaani (ARTPARK-IISc, spontaneous image-description speech from Indian districts) and SPRING-INX R1/R2
(IIT Madras, conversational; about half the utterances contain Latin-script words) are not listed as
training data by any of the public models. Kathbath (AI4Bharat IndicSUPERB, read speech) and IndicVoices
(AI4Bharat, conversational and extempore speech) are corpora some of the models trained another split of.
Each set is a seeded, stratified 1,000-utterance sample of the official split (1.3–2.6 h of audio per
set); manifests with per-file SHA-256 are under [`multiset/`](multiset/), predictions under
[`outputs_multiset/`](outputs_multiset/). The cleaning rule removes clips only from IndicVoices
conversational (82 replies under 0.5 s); the other five sets are scored whole.

**WER on the official-split subsets, cleaned** (best per column in bold; 95 % intervals in brackets):

| Model | Vaani test | SPRING-INX R1 eval | SPRING-INX R2 eval | Kathbath (known spk) | Kathbath (unknown spk) | IndicVoices conversational |
|---|---:|---:|---:|---:|---:|---:|
| Whisper turbo bn-BD ‡ | **21.38%** [20.3–22.5] <sub>U</sub> | **36.30%** [35.1–37.6] <sub>U</sub> | **36.13%** [34.8–37.5] <sub>U</sub> | 10.43% [9.7–11.1] <sub>U</sub> | 10.42% [9.7–11.2] <sub>U</sub> | 12.99% [12.0–14.0] <sub>U</sub> |
| Qwen3-ASR 1.7B bn-BD ‡ | 22.68% [20.9–25.2] <sub>U</sub> | 37.62% [36.2–39.2] <sub>U</sub> | 36.89% [35.3–38.8] <sub>U</sub> | 10.26% [9.5–11.0] <sub>U</sub> | 10.37% [9.6–11.1] <sub>U</sub> | **11.91%** [11.0–12.8] <sub>U</sub> |
| Canary 1B v2 bn-BD ‡ | 21.92% [20.8–23.1] <sub>U</sub> | 36.79% [35.6–38.0] <sub>U</sub> | 36.75% [35.2–38.7] <sub>U</sub> | 10.96% [10.2–11.7] <sub>U</sub> | 11.00% [10.2–11.8] <sub>U</sub> | 12.65% [11.8–13.5] <sub>U</sub> |
| Voxtral Mini bn-BD ‡ | 22.08% [20.9–23.2] <sub>U</sub> | 36.87% [35.7–38.2] <sub>U</sub> | 40.01% [36.6–44.3] <sub>U</sub> | 10.77% [10.0–11.5] <sub>U</sub> | 10.82% [10.1–11.6] <sub>U</sub> | 13.34% [12.5–14.3] <sub>U</sub> |
| FastConformer hybrid bn-BD ‡ | 22.53% [21.4–23.7] <sub>U</sub> | 36.92% [35.7–38.2] <sub>U</sub> | 36.60% [35.4–37.9] <sub>U</sub> | 11.60% [10.8–12.4] <sub>U</sub> | 11.41% [10.6–12.2] <sub>U</sub> | 12.63% [11.8–13.5] <sub>U</sub> |
| Conformer Large (hishab) | 33.10% [31.8–34.5] <sub>N</sub> | 42.46% [41.2–43.8] <sub>N</sub> | 42.25% [40.9–43.6] <sub>N</sub> | **9.33%** [8.6–10.1] <sub>S</sub> | **8.79%** [8.1–9.5] <sub>S</sub> | 29.30% [27.9–30.8] <sub>N</sub> |
| Rumble-2-Flash bn-BD ‡ | 22.79% [21.6–24.0] <sub>U</sub> | 37.45% [36.2–38.7] <sub>U</sub> | 37.05% [35.8–38.4] <sub>U</sub> | 11.67% [10.9–12.5] <sub>U</sub> | 11.33% [10.6–12.1] <sub>U</sub> | 13.23% [12.4–14.1] <sub>U</sub> |
| Granite Speech 5.0 2B bn-BD ‡ | 22.52% [21.4–23.7] <sub>U</sub> | 39.15% [37.8–40.5] <sub>U</sub> | 38.08% [36.8–39.5] <sub>U</sub> | 11.72% [10.9–12.5] <sub>U</sub> | 11.46% [10.7–12.3] <sub>U</sub> | 14.04% [13.1–15.0] <sub>U</sub> |
| Whisper Medium | 26.35% [25.1–27.6] <sub>U</sub> | 40.04% [38.8–41.3] <sub>U</sub> | 39.71% [38.5–40.9] <sub>U</sub> | 9.70% [9.0–10.4] <sub>U</sub> | 9.91% [9.2–10.6] <sub>U</sub> | 24.43% [23.2–25.7] <sub>U</sub> |
| ehzawad FastConformer-CTC | 24.67% [23.5–25.9] <sub>N</sub> | 39.23% [38.0–40.5] <sub>N</sub> | 38.42% [37.2–39.7] <sub>N</sub> | 13.38% [12.6–14.2] <sub>S</sub> | 13.43% [12.6–14.3] <sub>S</sub> | 12.26% [11.4–13.1] <sub>S</sub> |
| Whistle bn-BD ‡ (55M, CPU engine) | 27.43% [26.1–28.7] <sub>U</sub> | 43.38% [42.1–44.8] <sub>U</sub> | 42.19% [40.9–43.5] <sub>U</sub> | 15.61% [14.7–16.5] <sub>U</sub> | 15.82% [14.9–16.8] <sub>U</sub> | 19.32% [18.1–20.5] <sub>U</sub> |
| Whistle bn-BD ‡ (55M, A100, greedy) | 28.74% [27.3–30.3] <sub>U</sub> | 48.08% [45.5–51.1] <sub>U</sub> | 47.51% [44.6–50.7] <sub>U</sub> | 17.57% [16.0–19.5] <sub>U</sub> | 17.36% [15.9–19.3] <sub>U</sub> | 20.95% [19.5–22.5] <sub>U</sub> |
| Wav2Vec2 | 31.28% [29.9–32.7] <sub>U</sub> | 47.28% [46.1–48.6] <sub>U</sub> | 47.31% [46.1–48.5] <sub>U</sub> | 19.07% [18.1–20.0] <sub>U</sub> | 18.80% [17.9–19.7] <sub>U</sub> | 32.72% [31.3–34.2] <sub>U</sub> |

Provenance flags, from the model cards only: **N**, the card does not list this corpus; **S**, the model
trained on another split of this corpus; **U**, the card does not say either way. The ‡ models' training
data is undocumented, hence **U** everywhere. None of the three means "verified clean".

**What the table says.**

- The FLEURS ranking does not transfer. Conformer Large, the best model on FLEURS (10.8 %), is the
  last public model on Vaani (33.1 %) and third of four on both SPRING sets. [Its model card](https://huggingface.co/hishab/titu_stt_bn_conformer_large)
  lists read and broadcast corpora (OpenSLR, Bengali.AI, MadASR, Kathbath, Common Voice, FLEURS,
  SUBAK.KO, Shrutilipi, Vasha-Bichita, IndicTTS) and none of Vaani, SPRING-INX or IndicVoices.
- Among the public models, `ehzawad/stt_bn_fastconformer_ctc` leads on Vaani and both SPRING sets, with
  Whisper Medium second on each.
- The private fine-tunes are lower still on those three sets, Whisper turbo lowest of all (21.4 / 36.3 /
  36.1 %), Voxtral Mini at 22.1 / 36.9 / 40.0 %, Granite Speech at 22.5 / 39.2 / 38.1 % and Rumble-2-Flash
  at 22.8 / 37.5 / 37.0 %. Voxtral Mini's
  SPRING R2 score includes three recordings that ran to its 500-token cap repeating one word: they hold 663
  of its 8,143 word errors there (36.8 % without them).
- On Kathbath the order flips back: Conformer Large (9.3 / 8.8 %) and Whisper Medium (9.7 / 9.9 %) lead.
  Conformer Large's card lists Kathbath train and validation and the ehzawad CTC model trained on Kathbath
  train; a **U** cell can hide training overlap. Known- and unknown-speaker results differ by under a point
  for every model.
- **IndicVoices conversational is not a clean head-to-head for the ehzawad CTC model**, which trained on
  its train split. Qwen3-ASR reads 11.9 %, the other private models up to 14.0 % (Whistle bn-BD 19.3 % on
  CPU); Conformer Large (29.3 %, the one **N** cell) emitted 1 empty hypothesis on the cleaned subset and
  Wav2Vec2 11.
- SPRING's 36–47 % WER overstates the real error rate: about half its references contain English words
  in Latin script, and no model reproduced them: no other model emitted a single Latin-script word, and
  Whistle bn-BD only leaks English fragments from its English base model (26 of 1,000 R1 and 18 of 1,000 R2
  transcripts on CPU, 18 and 15 on the A100, such as "aren" or "minutes"), which count as errors. On the
  Latin-free half the other private models read 31.5–37.7 % and `ehzawad/stt_bn_fastconformer_ctc`
  35.4 / 35.1 %.

## FLEURS Bengali test (733 of 920 utterances scored)

The familiar benchmark, for comparison with published numbers. The cleaning rule removes the 187
recordings whose references write numbers as digits.

| Model | Checkpoint | Decoded with | WER (95% CI) | CER | ms/clip | Params |
|---|---|---|---|---|---:|---:|
| Conformer Large | `hishab/titu_stt_bn_conformer_large` | NeMo, CTC | **10.83%** [9.83–11.87] | 2.20% | 87 | 121.5M |
| Canary 1B v2 bn-BD ‡ | `ehzawad/canary-1b-v2-bn-BD` | NeMo | 11.58% [10.60–12.60] | 2.68% | 485 | 957.3M |
| Qwen3-ASR 1.7B bn-BD ‡ | `ehzawad/Qwen3-ASR-1.7B-bn-BD` | vLLM | 11.86% [10.88–12.87] | 2.78% | 656 | 2,038M |
| Voxtral Mini bn-BD ‡ | `ehzawad/Voxtral-Mini-3B-2507-bn-BD` | vLLM | 12.08% [11.05–13.11] | 2.83% | 438 | 4,676M |
| Whisper turbo bn-BD ‡ | `ehzawad/whisper-large-v3-turbo-bn-BD` | vLLM | 12.39% [11.25–13.62] | 3.19% | 700 | 808.9M |
| FastConformer hybrid bn-BD ‡ | `ehzawad/fastconformer-hybrid-rnnt-ctc-large-bn-BD` | NeMo, RNN-T | 12.83% [11.84–13.86] | 2.98% | 102 | 114.7M |
| Rumble-2-Flash bn-BD ‡ | `ehzawad/Rumble-2-Flash-bn-BD` | vLLM | 13.01% [12.00–14.04] | 3.03% | 211 | 508.6M |
| Granite Speech 5.0 2B bn-BD ‡ | `ehzawad/granite-speech-5.0-2b-bn-BD` | vLLM | 13.49% [12.46–14.55] | 3.24% | 961 | 2,313M |
| Whisper Medium | `SayedShaun/bengali-whisper-medium` | vLLM | 13.90% [12.77–15.04] | 3.17% | 304 | 763.9M |
| **ehzawad FastConformer-CTC** | `ehzawad/stt_bn_fastconformer_ctc` | NeMo, CTC | 14.63% [13.50–15.78] | 3.47% | 88 | 115.6M |
| Whistle bn-BD ‡ (55M, CPU engine) | `ehzawad/whistle-bn-BD` | needle engine, CPU | 16.36% [15.20–17.57] | 4.27% | 578 (CPU) | 57.2M |
| Whistle bn-BD ‡ (55M, A100, greedy) | `ehzawad/whistle-bn-BD` | JAX, A100-80GB | 18.08% [16.48–20.06] | 6.04% | 139 | 57.2M |
| Wav2Vec2 | `SayedShaun/bangla-wave2vec2-unigram` | PyTorch, CTC | 18.25% [17.08–19.43] | 4.08% | 56 | 315.5M |

Greedy decoding, batch 1, no external language model (Whistle bn-BD on CPU: its engine's own decoding).
No clip failed and no hypothesis came back empty for any model. `ms/clip` is the mean single-clip latency
over the 920 test recordings on the A100 (the Colab A100-SXM4-40GB; Granite Speech bn-BD and
Rumble-2-Flash bn-BD on other A100-SXM4-40GB cards; Whistle bn-BD on an A100-SXM4-80GB, plus its CPU row):

| Model | mean ms/clip | median | p95 |
|---|---:|---:|---:|
| Wav2Vec2 | 56 | 54 | 86 |
| Conformer Large (hishab) | 87 | 83 | 96 |
| ehzawad FastConformer-CTC | 88 | 84 | 98 |
| FastConformer hybrid bn-BD ‡ | 102 | 98 | 119 |
| Whistle bn-BD ‡ (55M), A100-80GB, JAX | 139 | 131 | 235 |
| Rumble-2-Flash bn-BD ‡ | 211 | 199 | 330 |
| Whisper Medium | 304 | 288 | 461 |
| Voxtral Mini bn-BD ‡ | 438 | 420 | 680 |
| Canary 1B v2 bn-BD ‡ | 485 | 463 | 740 |
| Qwen3-ASR 1.7B bn-BD ‡ | 656 | 621 | 1,028 |
| Whisper turbo bn-BD ‡ | 700 | 666 | 1,108 |
| Granite Speech 5.0 2B bn-BD ‡ | 961 | 911 | 1,485 |
| Whistle bn-BD ‡ (55M), CPU engine, one core | 578 | 530 | 1,012 |

These numbers time the configured pipelines, not the models in isolation:

- vLLM 0.31 (torch 2.13, CUDA graphs) for Voxtral Mini, Whisper Medium, Whisper turbo, Qwen3-ASR,
  Granite Speech and Rumble-2-Flash; NeMo 2.7.3 or PyTorch (torch 2.11) for the rest. Whistle bn-BD on the
  A100: JAX 0.12 with XLA, see [On-device model: Whistle bn-BD](#on-device-model-whistle-bn-bd).
- The CTC and RNN-T models emit a whole piece in one pass; the others generate text token by token, so
  their time grows with the length of what was said.
- A few milliseconds apart is not an established ordering, and p95 is variation across clips, not
  uncertainty in the mean.

**Test split only.** The ehzawad CTC model trained on a corpus that includes the FLEURS *train* split,
which overlaps the FLEURS *validation* split, so the 402 validation recordings are excluded for every
model. It had also been evaluated on FLEURS test during its own development, so read its FLEURS row as a
like-for-like re-measurement and the other corpora above as the fresh evidence.

Paired comparisons for `ehzawad/stt_bn_fastconformer_ctc` on this split: +0.74 points behind Whisper
Medium [+0.04, +1.42] at 3.5× its speed; +3.81 points behind Conformer Large [+3.03, +4.54].

Voxtral Mini bn-BD ‡ on this split is not separable from Canary (+0.50 [−0.15, +1.16]), Qwen3-ASR
(+0.22 [−0.39, +0.83]) or Whisper turbo (−0.31 [−1.13, +0.45]); it is higher than Conformer Large
(+1.25 [+0.48, +2.01]) and lower than the hybrid FastConformer (−0.75 [−1.41, −0.09]), Rumble-2-Flash
(−0.93 [−1.56, −0.29]), Granite Speech (−1.41 [−2.16, −0.68]), Whisper Medium (−1.82 [−2.57, −1.07]), the
ehzawad CTC model (−2.55 [−3.28, −1.82]) and Wav2Vec2.

Granite Speech bn-BD ‡ on this split is lower than the ehzawad CTC model (−1.14 [−1.82, −0.47]), not
separable from Whisper Medium (−0.41 [−1.14, +0.31]) or Rumble-2-Flash (+0.48 [−0.18, +1.15]), and higher
than the private fine-tunes other than Whistle and Rumble-2-Flash, from the hybrid FastConformer
(+0.66 [+0.01, +1.29]) to Canary (+1.91 [+1.29, +2.55]), and than Conformer Large (+2.66 [+1.92, +3.40]).

Rumble-2-Flash bn-BD ‡ on this split is not separable from Granite Speech (−0.48 [−1.15, +0.18]), the hybrid
FastConformer (+0.18 [−0.38, +0.73]) or Whisper turbo (+0.62 [−0.11, +1.30]); it is lower than Whisper
Medium (−0.88 [−1.54, −0.24]), the ehzawad CTC model (−1.62 [−2.26, −0.97]), Wav2Vec2 and both Whistle rows,
and higher than Voxtral Mini (+0.93 [+0.29, +1.56]), Qwen3-ASR (+1.16 [+0.56, +1.75]), Canary
(+1.43 [+0.83, +2.03]) and Conformer Large (+2.18 [+1.46, +2.87]).

## Paired comparisons

Each model minus the model with the lowest WER among the others on each cleaned set, on the same recordings
(WER points, 95 % reference-clustered bootstrap, 10,000 draws; pointwise, not adjusted for multiple
comparisons). Every pair is in `paired` in `outputs_clean/summary_clean.json`.

**Voxtral Mini bn-BD ‡:**

| Set | Voxtral Mini | Best other model | Voxtral Mini minus it, WER points (95% CI) |
|---|---:|---|---|
| FLEURS test | 12.08% | Conformer Large (hishab) 10.83% | +1.25 [+0.48, +2.01], higher |
| Vaani test | 22.08% | Whisper turbo bn-BD ‡ 21.38% | +0.69 [+0.10, +1.30], higher |
| SPRING-INX R1 eval | 36.87% | Whisper turbo bn-BD ‡ 36.30% | +0.57 [+0.05, +1.09], higher |
| SPRING-INX R2 eval | 40.01% | Whisper turbo bn-BD ‡ 36.13% | +3.88 [+0.80, +7.96], higher |
| Kathbath (known spk) | 10.77% | Conformer Large (hishab) 9.33% | +1.44 [+0.77, +2.12], higher |
| Kathbath (unknown spk) | 10.82% | Conformer Large (hishab) 8.79% | +2.03 [+1.37, +2.71], higher |
| IndicVoices conversational | 13.34% | Qwen3-ASR 1.7B bn-BD ‡ 11.91% | +1.43 [+0.85, +2.03], higher |
| SUBAK.KO test | 5.42% | Canary 1B v2 bn-BD ‡ 5.57% | −0.14 [−0.53, +0.24], inconclusive |
| BanSpeech | 19.24% | Qwen3-ASR 1.7B bn-BD ‡ 19.37% | −0.13 [−1.10, +0.83], inconclusive |
| Common Voice 17 test | 10.32% | Conformer Large (hishab) 7.35% | +2.97 [+2.24, +3.70], higher |
| Ben-10 validation (dialects) | 69.97% | FastConformer hybrid bn-BD ‡ 60.91% | +9.07 [+4.88, +13.96], higher |

**Granite Speech 5.0 2B bn-BD ‡:**

| Set | Granite Speech | Best other model | Granite Speech minus it, WER points (95% CI) |
|---|---:|---|---|
| FLEURS test | 13.49% | Conformer Large (hishab) 10.83% | +2.66 [+1.92, +3.40], higher |
| Vaani test | 22.52% | Whisper turbo bn-BD ‡ 21.38% | +1.13 [+0.52, +1.73], higher |
| SPRING-INX R1 eval | 39.15% | Whisper turbo bn-BD ‡ 36.30% | +2.84 [+2.24, +3.47], higher |
| SPRING-INX R2 eval | 38.08% | Whisper turbo bn-BD ‡ 36.13% | +1.95 [+1.34, +2.60], higher |
| Kathbath (known spk) | 11.72% | Conformer Large (hishab) 9.33% | +2.38 [+1.68, +3.09], higher |
| Kathbath (unknown spk) | 11.46% | Conformer Large (hishab) 8.79% | +2.68 [+1.99, +3.37], higher |
| IndicVoices conversational | 14.04% | Qwen3-ASR 1.7B bn-BD ‡ 11.91% | +2.13 [+1.54, +2.72], higher |
| SUBAK.KO test | 6.23% | Voxtral Mini bn-BD ‡ 5.42% | +0.81 [+0.19, +1.58], higher |
| BanSpeech | 22.91% | Voxtral Mini bn-BD ‡ 19.24% | +3.67 [+2.64, +4.73], higher |
| Common Voice 17 test | 10.99% | Conformer Large (hishab) 7.35% | +3.65 [+2.90, +4.40], higher |
| Ben-10 validation (dialects) | 69.39% | FastConformer hybrid bn-BD ‡ 60.91% | +8.48 [+7.12, +9.98], higher |

**Rumble-2-Flash bn-BD ‡:**

| Set | Rumble-2-Flash | Best other model | Rumble-2-Flash minus it, WER points (95% CI) |
|---|---:|---|---|
| FLEURS test | 13.01% | Conformer Large (hishab) 10.83% | +2.18 [+1.46, +2.87], higher |
| Vaani test | 22.79% | Whisper turbo bn-BD ‡ 21.38% | +1.41 [+0.82, +2.02], higher |
| SPRING-INX R1 eval | 37.45% | Whisper turbo bn-BD ‡ 36.30% | +1.15 [+0.62, +1.68], higher |
| SPRING-INX R2 eval | 37.05% | Whisper turbo bn-BD ‡ 36.13% | +0.91 [+0.37, +1.46], higher |
| Kathbath (known spk) | 11.67% | Conformer Large (hishab) 9.33% | +2.34 [+1.67, +3.00], higher |
| Kathbath (unknown spk) | 11.33% | Conformer Large (hishab) 8.79% | +2.54 [+1.89, +3.19], higher |
| IndicVoices conversational | 13.23% | Qwen3-ASR 1.7B bn-BD ‡ 11.91% | +1.33 [+0.79, +1.87], higher |
| SUBAK.KO test | 7.83% | Voxtral Mini bn-BD ‡ 5.42% | +2.40 [+1.95, +2.87], higher |
| BanSpeech | 20.11% | Voxtral Mini bn-BD ‡ 19.24% | +0.87 [−0.08, +1.83], inconclusive |
| Common Voice 17 test | 11.18% | Conformer Large (hishab) 7.35% | +3.83 [+3.10, +4.55], higher |
| Ben-10 validation (dialects) | 65.96% | FastConformer hybrid bn-BD ‡ 60.91% | +5.05 [+3.69, +6.49], higher |

## Three-model comparison on one GPU

Rumble-2-Flash bn-BD ‡, Voxtral Mini bn-BD ‡ and Qwen3-ASR 1.7B bn-BD ‡ were decoded together on 20 Nov
2026, set by set, with the same harness and checkpoints (vLLM 0.31, torch 2.13; greedy, batch 1, one model
resident at a time): once on an A100-SXM4-40GB (FLEURS and the four Bangladeshi sets on one card, the six
Indian sets on a second card of the same model) and once on one B200. The A100 runs of Rumble-2-Flash are
the ones in the tables above. The Voxtral Mini re-run on the A100 gives its published WER on every set
(average 22.81 %). The Qwen3-ASR re-run averages 21.89 % against the published 21.94 %; single sets move
where a few clips run to the token cap (SUBAK.KO 6.18 % here, 7.16 % above; Ben-10 64.03 % against 63.39 %).

Each cell is the WER on the cleaned subset and the mean time per clip over the set's clips (FLEURS: the 920
test recordings). Tokens per clip: mean tokens in a FLEURS test transcript under the model's own tokenizer.
ms per token: the slope of a least-squares fit of each FLEURS test clip's time against its output tokens.

**A100-SXM4-40GB:**

| Set | Rumble-2-Flash bn-BD ‡ | Voxtral Mini bn-BD ‡ | Qwen3-ASR 1.7B bn-BD ‡ |
|---|---:|---:|---:|
| FLEURS test | 13.01% · 211 ms | 12.08% · 409 ms | 11.85% · 624 ms |
| SUBAK.KO test | 7.83% · 224 ms | 5.42% · 463 ms | 6.18% · 670 ms |
| BanSpeech | 20.11% · 84 ms | 19.24% · 179 ms | 19.27% · 239 ms |
| Common Voice 17 test | 11.18% · 108 ms | 10.32% · 211 ms | 9.97% · 315 ms |
| Ben-10 validation (dialects) | 65.96% · 348 ms | 69.97% · 901 ms | 64.03% · 1,077 ms |
| Vaani test | 22.79% · 106 ms | 22.08% · 216 ms | 22.61% · 304 ms |
| SPRING-INX R1 eval | 37.45% · 176 ms | 36.87% · 383 ms | 37.55% · 533 ms |
| SPRING-INX R2 eval | 37.05% · 184 ms | 40.01% · 405 ms | 36.84% · 552 ms |
| Kathbath (known spk) | 11.67% · 123 ms | 10.77% · 264 ms | 10.25% · 365 ms |
| Kathbath (unknown spk) | 11.33% · 126 ms | 10.82% · 264 ms | 10.39% · 366 ms |
| IndicVoices conversational | 13.23% · 126 ms | 13.34% · 268 ms | 11.89% · 370 ms |
| **Average, 11 sets** | **22.87% · 165 ms** | **22.81% · 360 ms** | **21.89% · 492 ms** |
| Clips at the token cap | 15 | 21 | 25 |
| Tokens per FLEURS test clip | 135 | 51 | 135 |
| ms per token (fit, FLEURS test) | 1.42 | 7.46 | 4.33 |

**B200:**

| Set | Rumble-2-Flash bn-BD ‡ | Voxtral Mini bn-BD ‡ | Qwen3-ASR 1.7B bn-BD ‡ |
|---|---:|---:|---:|
| FLEURS test | 13.02% · 179 ms | 12.09% · 140 ms | 11.78% · 245 ms |
| SUBAK.KO test | 8.08% · 187 ms | 5.46% · 156 ms | 6.66% · 262 ms |
| BanSpeech | 20.05% · 70 ms | 19.28% · 66 ms | 19.27% · 98 ms |
| Common Voice 17 test | 11.21% · 89 ms | 10.26% · 77 ms | 9.94% · 127 ms |
| Ben-10 validation (dialects) | 66.59% · 293 ms | 68.49% · 289 ms | 64.04% · 412 ms |
| Vaani test | 22.75% · 87 ms | 22.14% · 78 ms | 22.61% · 123 ms |
| SPRING-INX R1 eval | 37.46% · 150 ms | 36.91% · 130 ms | 37.60% · 209 ms |
| SPRING-INX R2 eval | 37.08% · 156 ms | 40.13% · 136 ms | 36.86% · 216 ms |
| Kathbath (known spk) | 11.57% · 103 ms | 10.79% · 93 ms | 10.31% · 146 ms |
| Kathbath (unknown spk) | 11.34% · 105 ms | 10.87% · 93 ms | 10.34% · 147 ms |
| IndicVoices conversational | 13.26% · 105 ms | 13.36% · 94 ms | 11.92% · 147 ms |
| **Average, 11 sets** | **22.95% · 139 ms** | **22.71% · 123 ms** | **21.94% · 194 ms** |
| Clips at the token cap | 17 | 19 | 26 |
| Tokens per FLEURS test clip | 135 | 51 | 135 |
| ms per token (fit, FLEURS test) | 1.22 | 2.39 | 1.64 |

**What the tables say.**

- **Average accuracy is about equal.** The three eleven-set averages span 0.98 points on the A100 and 1.01
  on the B200, and each model's average moves by 0.05–0.10 points between the two cards (88–91 % of the
  scored transcripts are identical). Single sets differ: on the A100 Voxtral Mini is lowest on four (SUBAK.KO,
  BanSpeech, Vaani, SPRING R1), Qwen3-ASR on the other seven, and Rumble-2-Flash is 0.2–2.4 points above
  the lowest on each set, below Voxtral Mini on Ben-10, SPRING R2 and IndicVoices.
- **The speed order reverses.** On the A100 Rumble-2-Flash is fastest on every set, 2.2× faster than
  Voxtral Mini and 3.0× faster than Qwen3-ASR on average; on the B200 Voxtral Mini is fastest on every set
  (123 ms against 139 and 194 ms).
- **Why.** Voxtral Mini's Tekken tokenizer writes a FLEURS test transcript in 51 tokens, the Qwen-derived
  tokenizers of Rumble-2-Flash and Qwen3-ASR in 135, so Voxtral Mini generates 2.7× fewer tokens. On the
  A100 each output token costs it 5.3× as much time as one of Rumble-2-Flash's (the fit slopes), which
  outweighs that; on the B200 only 2.0×, so the shorter output wins.

Runs: `outputs_compare3/a100/` and `outputs_compare3/b200/`, in the layout of `outputs_fleurs/` and
`outputs_multiset/`. `score_compare3.py` rebuilds
[`outputs_compare3/summary.json`](outputs_compare3/summary.json) from them: per set and model n, WER with a
95 % interval, CER, mean/median/p95 ms per clip and cap hits, the averages, the token counts and fits, and
how many transcripts match across the two GPUs.

## On-device model: Whistle bn-BD

`ehzawad/whistle-bn-BD` ‡ is a full fine-tune of Cactus Compute's 55M-parameter Whistle (57.2M with its added
Bengali vocabulary), deployed as a 4-bit `.cact` archive
(30.5 MB) for the cactus-needle engine. The same checkpoint was measured under two runtimes, on 21-22 Oct
2026, and both rows are kept (the numbers in this section are on the full sets, as first reported):

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
  come from the private bn-whistle repository and were imported with the same checks. Granite Speech bn-BD
  was added on 23 Oct 2026 through `run_all.sh` and the same harness, alone on another A100-SXM4-40GB, in the
  environment of the other vLLM decoders (vLLM 0.31, torch 2.13). Rumble-2-Flash bn-BD was added on 20 Nov
  2026 the same way, through `run_all.sh`'s `rumble_bn` entry and the same harness, on another
  A100-SXM4-40GB (FLEURS and the four Bangladeshi sets on one card, the six Indian sets on a second card of
  the same model), vLLM 0.31 and torch 2.13. Its weights were read from a local folder whose
  `model.safetensors` is byte-identical to the pinned revision (SHA-256 recorded in its `run_meta.json`).
- **One audio path.** `asr_core.py` loads, downmixes and resamples to 16 kHz, cuts the audio at pauses into
  pieces of at most 25 s and stages them as PCM16; every backend reads those staged files. The nine models'
  runs record identical piece counts and forced cuts on all 11,322 clips, and so do Granite Speech's.
  Granite Speech read local copies of the audio, checked against the manifests' SHA-256; 26 Kathbath
  recordings (18 known-speaker, 8 unknown-speaker) could not be rebuilt bit-exactly from the source archive
  (same duration, different PCM samples), so it decoded those local conversions; its two Kathbath
  `run_meta.json` files list them. Rumble-2-Flash also read local copies; this time every one of its 11,322
  files matched the manifests' SHA-256 (the six Indian sets were rebuilt bit-exactly from their sources),
  and its runs record the same piece counts and forced cuts.
- **One scorer.** `bench_score.norm` strips punctuation to word boundaries and applies Unicode NFC before
  WER/CER. NFC is not cosmetic: Bengali writes য়, ড় and ঢ় either precomposed or as base + nukta.
  - The FLEURS references and the NeMo models use the decomposed form; Whisper and wav2vec2 use the
    precomposed one.
  - Without NFC, every such character scores as a substitution: on the full FLEURS test split Whisper
    Medium reads 27.9 % instead of 16.2 %.
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
  - Granite Speech bn-BD: the checkpoint's chat template with an instruction to transcribe Bengali, the
    prompt the fine-tune expects (vLLM's endpoint prompt for this architecture is English-only); at most 25
    new tokens per second of audio plus 16 per piece, a cap chosen on the FLEURS validation split to bound
    greedy repetition loops (references there run at up to 20 tokens per second). On those 402 recordings
    this runner and Hugging Face transformers both read 15.78 % WER.
  - Rumble-2-Flash bn-BD: the prompt the fine-tune was trained and evaluated with, the controls
    `<nodigits><nofiller><verbatim>` in the system turn and the forced prefix `language Bengali<asr_text>`
    after the assistant header (vLLM's endpoint prompt for this architecture leaves the language unset and
    has no controls); at most 500 new tokens. Against Hugging Face transformers, the fine-tune's own decode:
    on the 804 FLEURS test clips that match its evaluation set one-to-one (by reference text and duration),
    732 transcripts are identical and WER is 14.38 % with this runner against 14.33 %; on the 342 matched
    validation clips, 309 are identical, and WER is 15.13 % against 15.11 % without the one clip where greedy
    vLLM decoding ran to the 500-token cap (17.31 % against 15.10 % with it).
- **Canary** runs in NeMo as Bengali-to-Bengali transcription without the punctuation flag (`pnc="no"`),
  beam size 1; the **hybrid FastConformer** with its default RNN-T decoder (greedy).
- **Timing.** One warm-up clip is discarded. Each clip is then timed between CUDA synchronisations, with
  `asr_core` loading, chunking and staging included and model startup excluded. Runs were niced. Other
  activity on the VM during the runs was limited to brief checks over SSH (log tails, a few seconds of
  scoring) and a 10-minute mirror of finished outputs to Google Drive; its effect was not measured.
- **Every scored clip counts.** The cleaned-subset rule removes clips by their duration and reference only,
  never by a model's output, and the same clips are scored for every model. No model raised on a clip. Empty
  hypotheses and runaway outputs stay in the denominator as full errors.
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
  chip with more memory; the three-model comparison adds one B200), from one pass per model and set.
  Throughput under load, batched decoding and CPU inference (except Whistle bn-BD's CPU row) are not
  measured.
- Cell intervals on the 1,000-utterance subsets run from about ±0.7 points (Kathbath) to ±1.5 points
  (Vaani, SPRING, IndicVoices). A few cells are wider where a handful of runaway outputs carry many errors
  (Voxtral Mini's SPRING R2 and Ben-10, Canary's Ben-10, Qwen3-ASR's SUBAK.KO, greedy Whistle's SUBAK.KO,
  BanSpeech, Common Voice and SPRING). Differences inside an
  interval are not differences.
- Training provenance is only as good as the model cards; a **U** flag means it could not be established
  either way. The ‡ models are not public (Rumble-2-Flash bn-BD: public, gated) and their training data is
  undocumented.

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
python score_clean.py                             # the cleaned subsets: every table in this README
python score_compare3.py                          # the three-model comparison -> outputs_compare3/summary.json
python make_report.py                             # -> asr_benchmark.pdf
```

- `MULTISET_ASSETS/<set>/eval_manifest.local.json` lists the same rows as `multiset/<set>/eval_manifest.json`,
  with local paths. `bench_run_set.py` checks every file's SHA-256 before decoding.
- `MODELS="..."` limits `run_all.sh` to some models. The ‡ models are private or gated: they need a Hugging
  Face token with access, and Voxtral Mini also `VOXTRAL_MINI=<local snapshot of the model>`.
- `score_compare3.py` counts output tokens with each model's tokenizer from the Hub (access needed) or from
  `--tokenizer <label>=<local dir>`; `--keep-tokens` keeps the committed counts.
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
asr_benchmark.pdf        the report (full sets, without Granite Speech; page 4: the three-model comparison)
asr_core.py              the single audio path shared by demo and benchmark
backends.py              model loading for NeMo, wav2vec2 CTC and Whisper seq2seq
bench_run.py             one model over the FLEURS set, batch 1, timed; backends for NeMo, Canary, wav2vec2,
                         Whisper (transformers and vLLM), Voxtral, Qwen3-ASR, Granite Speech and
                         Rumble-2-Flash (vLLM)
bench_run_set.py         the same backends over any eval manifest (the subset runs)
build_multiset_bd.py     the SUBAK.KO, BanSpeech, Common Voice 17 and Ben-10 subsets
bench_score.py           predictions -> WER/CER with bootstrap intervals -> summary.json
clean_split_table.py     the FLEURS test-split table with paired deltas (full split)
score_multiset.py        the subset tables, diagnostics, and Voxtral Mini's paired comparisons (full subsets)
score_clean.py           the cleaned-subset rule, its clip lists and every table in this README
clean_subsets/           the exact clips scored and removed on each of the eleven sets
outputs_clean/           summary_clean.json: cleaned-subset scores, intervals, ranks and paired comparisons
score_compare3.py        the three-model comparison -> outputs_compare3/summary.json
outputs_compare3/        Rumble-2-Flash, Voxtral Mini and Qwen3-ASR runs on one A100 (a100/) and one B200 (b200/)
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
  - `ehzawad/granite-speech-5.0-2b-bn-BD`, from `ibm-granite/granite-speech-5.0-2b` (IBM, Apache-2.0).
  - `ehzawad/Rumble-2-Flash-bn-BD` (public, gated; 508.6M parameters), from `Reson8/Rumble-2-Flash` (Reson8,
    Apache-2.0; Qwen3-ASR architecture), under the ehzawad Research-Only Non-Commercial Model License 1.0.

Model licences are those of their respective publishers. Check each model card before use.

Author: Emrul Zawad (ehzawad).
