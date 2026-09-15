#!/usr/bin/env bash
# Six models, one GPU, one model resident at a time, same eval set and loader.
# The GPU is pinned by UUID so a co-tenant on the other card cannot perturb
# timings, and each run is confirmed finished before the next begins.
set -u
cd "$(dirname "$0")"
P=/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline
GPU="${BENCH_GPU:?set BENCH_GPU to a GPU UUID}"
LOG=run_all.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }

run() { # label kind model [extra...]
  local label=$1 kind=$2 model=$3; shift 3
  if [ -f "outputs/$label/run_meta.json" ]; then say "skip $label (done)"; return 0; fi
  local py=$P/.venv/bin/python
  [ "$kind" = "qwen" ] && py=$P/.venv-qwen/bin/python
  say "=== $label ($kind) ==="
  CUDA_VISIBLE_DEVICES=$GPU $py bench_run.py --kind "$kind" --model "$model" \
      --label "$label" "$@" >> "logs_$label.txt" 2>&1
  local rc=$?
  say "$label rc=$rc $(tail -1 logs_$label.txt | head -c 120)"
  return $rc
}

R=$(cat checkpoint_revisions.json)
rev() { echo "$R" | .venv_rev_helper 2>/dev/null || python3 -c "
import json,sys; print(json.load(open('checkpoint_revisions.json'))['$1'])"; }

run ehzawad_fastconformer  nemo     "$P/experiments/fastconformer_ctc_bn_1kh/final.nemo"
run hishab_conformer_large nemo     "hishab/titu_stt_bn_conformer_large"
run hishab_fastconformer   nemo     "hishab/titu_stt_bn_fastconformer"
run whisper_medium         whisper  "SayedShaun/bengali-whisper-medium"   --revision "$(rev SayedShaun/bengali-whisper-medium)"
run wav2vec2               wav2vec2 "SayedShaun/bangla-wave2vec2-unigram" --revision "$(rev SayedShaun/bangla-wave2vec2-unigram)"
run qwen3_adapter          qwen     "Qwen/Qwen3-ASR-1.7B-hf" \
    --revision bcd2b5b7f32b480ab5790554cfa8347f246a14f3 \
    --adapter "$P/experiments/qwen_final/candidates/step_17280"
say "ALL-RUNS-DONE"
