#!/usr/bin/env bash
# Every model (four public, five private ehzawad bn-BD fine-tunes) on FLEURS (1,322 clips) and the ten
# 1,000-utterance subsets: greedy,
# batch 1, one model resident at a time, the harness unchanged (bench_run.py for FLEURS,
# bench_run_set.py for the subsets), checkpoints at the revisions in
# outputs_fleurs/checkpoint_revisions.json. This produced every run in outputs_fleurs/ and
# outputs_multiset/, on one Colab A100-SXM4-40GB. A completed run (run_meta.json present) is
# skipped, never overwritten.
#
#   BENCH_GPU=<GPU UUID> BENCH_PY=<python with the benchmark's requirements> \
#   BENCH_PY_VLLM=<python with vllm[audio]> VOXTRAL_MINI=<Mistral-format model dir> \
#   FLEURS_ASSETS=<dir> MULTISET_ASSETS=<dir> ./run_all.sh
#
# FLEURS_ASSETS holds eval_manifest.json and eval_fleurs_bn/ (python build_eval.py); the four Bangladeshi
# sets (subakko_test, banspeech, cv17_bn_test, ben10_valid) come from build_multiset_bd.py. The committed
# multiset/<set>/eval_manifest.json files carry the paths of the machine that built them;
# MULTISET_ASSETS/<set>/$LOCAL_MANIFEST must list the same rows with local paths, and
# bench_run_set.py re-checks every file's SHA-256 before decoding. Voxtral Mini bn-BD is private
# (ehzawad/Voxtral-Mini-3B-2507-bn-BD); without VOXTRAL_MINI it is skipped.
set -u
cd "$(dirname "$0")"
H=$(pwd)
GPU="${BENCH_GPU:?set BENCH_GPU to a GPU UUID}"
FLEURS_ASSETS=${FLEURS_ASSETS:?set FLEURS_ASSETS}
MULTISET_ASSETS=${MULTISET_ASSETS:?set MULTISET_ASSETS}
LOCAL_MANIFEST=${LOCAL_MANIFEST:-eval_manifest.local.json}
SETS=${SETS:-"vaani_test spring_r1_test spring_r2_test kathbath_test_known kathbath_test_unknown indicvoices_conv
             subakko_test banspeech cv17_bn_test ben10_valid"}
export CUDA_VISIBLE_DEVICES="$GPU"
LOG=run_all.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
rev() { python3 -c "import json; print(json.load(open('outputs_fleurs/checkpoint_revisions.json'))['$1'])"; }
nemo_path() { ${BENCH_PY:-python} -c "from huggingface_hub import hf_hub_download as d; print(d('$1', '$2', revision='$(rev "$1")'))"; }

fleurs() { local py=$1 label=$2 kind=$3 model=$4; shift 4
  [ -f "outputs_fleurs/$label/run_meta.json" ] && { say "skip fleurs $label (done)"; return; }
  say "=== fleurs $label"
  ( cd "$FLEURS_ASSETS" && BENCH_ASSETS=. BENCH_OUT_ROOT="$H/outputs_fleurs" nice -n 19 $py "$H/bench_run.py" \
      --kind "$kind" --model "$model" --label "$label" "$@" >> "$H/outputs_fleurs/logs_$label.txt" 2>&1 )
  say "fleurs $label rc=$? $(tail -1 "outputs_fleurs/logs_$label.txt" | head -c 160)"; }
multi() { local py=$1 set=$2 label=$3 kind=$4 model=$5; shift 5; local tag=${set}__${label}
  [ -f "outputs_multiset/$tag/run_meta.json" ] && { say "skip $tag (done)"; return; }
  say "=== $tag"
  nice -n 19 $py bench_run_set.py --kind "$kind" --model "$model" --label "$tag" \
      --manifest "$MULTISET_ASSETS/$set/$LOCAL_MANIFEST" --out-root outputs_multiset "$@" \
      >> "outputs_multiset/logs_$tag.txt" 2>&1
  say "$tag rc=$? $(tail -1 "outputs_multiset/logs_$tag.txt" | head -c 160)"; }

mkdir -p outputs_fleurs outputs_multiset
PY=${BENCH_PY:-python}
snapshot() { ${BENCH_PY:-python} -c "from huggingface_hub import snapshot_download as s; print(s('$1', revision='$(rev "$1")'))"; }
MODELS=${MODELS:-"fastconformer_ctc hishab_conformer_large wav2vec2 whisper_medium voxtral_mini_bn
                   fastconformer_hybrid_bn canary_1b_v2_bn whisper_turbo_bn qwen3_asr_bn"}
say "models: $(echo $MODELS); FLEURS and the subsets, one model at a time"
for label in $MODELS; do
  py=$PY
  case $label in
    fastconformer_ctc)      args=(nemo "$(nemo_path ehzawad/stt_bn_fastconformer_ctc stt_bn_fastconformer_ctc_v2.nemo)") ;;
    hishab_conformer_large) args=(nemo "$(nemo_path hishab/titu_stt_bn_conformer_large titu_stt_bn_conformer_large.nemo)") ;;
    wav2vec2)               args=(wav2vec2 SayedShaun/bangla-wave2vec2-unigram --revision "$(rev SayedShaun/bangla-wave2vec2-unigram)") ;;
    whisper_medium)         # every autoregressive decoder runs under vLLM (Whisper Medium, Whisper turbo, Qwen3-ASR, Voxtral Mini)
      py=${BENCH_PY_VLLM:?set BENCH_PY_VLLM for whisper_medium}
      export PATH="$(dirname "$py"):$PATH"
      args=(whisper_vllm "$(snapshot SayedShaun/bengali-whisper-medium)" --revision "$(rev SayedShaun/bengali-whisper-medium)") ;;
    voxtral_mini_bn)
      [ -n "${VOXTRAL_MINI:-}" ] || { say "skip voxtral_mini_bn (VOXTRAL_MINI not set)"; continue; }
      py=${BENCH_PY_VLLM:?set BENCH_PY_VLLM for voxtral_mini_bn}
      export PATH="$(dirname "$py"):$PATH"   # vLLM compiles CUDA graphs with ninja from that environment
      args=(voxtral_vllm "$VOXTRAL_MINI" --revision "$(rev ehzawad/Voxtral-Mini-3B-2507-bn-BD)") ;;
    # the other private bn-BD fine-tunes (ehzawad/*-bn-BD; a Hugging Face token with access is required)
    fastconformer_hybrid_bn)
      args=(nemo "$(nemo_path ehzawad/fastconformer-hybrid-rnnt-ctc-large-bn-BD fastconformer-hybrid-rnnt-ctc-large-bn-BD.nemo)") ;;
    canary_1b_v2_bn)        args=(canary "$(nemo_path ehzawad/canary-1b-v2-bn-BD canary-1b-v2-bn-BD.nemo)") ;;
    whisper_turbo_bn)
      py=${BENCH_PY_VLLM:?set BENCH_PY_VLLM for whisper_turbo_bn}
      export PATH="$(dirname "$py"):$PATH"
      args=(whisper_vllm "$(snapshot ehzawad/whisper-large-v3-turbo-bn-BD)" --revision "$(rev ehzawad/whisper-large-v3-turbo-bn-BD)") ;;
    qwen3_asr_bn)
      py=${BENCH_PY_VLLM:?set BENCH_PY_VLLM for qwen3_asr_bn}
      export PATH="$(dirname "$py"):$PATH"
      args=(qwen3_asr_vllm "$(snapshot ehzawad/Qwen3-ASR-1.7B-bn-BD)" --revision "$(rev ehzawad/Qwen3-ASR-1.7B-bn-BD)") ;;
    *) say "unknown model $label"; continue ;;
  esac
  fleurs "$py" $label "${args[@]}"
  for set in $SETS; do
    multi "$py" $set $label "${args[@]}"
  done
done
say "ALL-RUNS-DONE"
