#!/usr/bin/env bash
# Six models, one GPU (the A5000, by UUID), one model resident at a time, the
# same 1,322-clip eval set and loader as bench_run.py. Outputs go under
# outputs_fleurs/<label>. No pattern kills: bench_run.py is a direct child.
# The two local checkpoints are taken from FASTCONFORMER_NEMO and QWEN_ADAPTER
# (env), the two interpreters from BENCH_PY / BENCH_PY_QWEN.
set -u
cd "$(dirname "$0")"
P=${PIPELINE_ROOT:-/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline}
FASTCONFORMER_NEMO=${FASTCONFORMER_NEMO:-model/stt_bn_fastconformer_ctc.nemo}   # huggingface.co/ehzawad/stt_bn_fastconformer_ctc
QWEN_ADAPTER=${QWEN_ADAPTER:-$P/experiments/qwen_final/candidates/step_17280}      # huggingface.co/ehzawad/stt_bn_qwen3_asr
GPU="${BENCH_GPU:?set BENCH_GPU to a GPU UUID}"
SFX=
LOG=run_all_p4_repro.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }

run() { # label kind model [extra...]
  local label=$1$SFX kind=$2 model=$3; shift 3
  if [ -f "outputs_fleurs/$label/run_meta.json" ]; then say "skip $label (done)"; return 0; fi
  local py=${BENCH_PY:-$P/.venv/bin/python}
  [ "$kind" = "qwen" ] && py=${BENCH_PY_QWEN:-$P/.venv-qwen/bin/python}
  say "=== $label ($kind) ==="
  CUDA_VISIBLE_DEVICES=$GPU nice -n 19 $py bench_run.py --kind "$kind" --model "$model" \
      --label "$label" "$@" >> "logs_$label.txt" 2>&1
  local rc=$?
  say "$label rc=$rc $(tail -1 logs_$label.txt | head -c 120)"
  return $rc
}

rev() { python3 -c "
import json; print(json.load(open('checkpoint_revisions.json'))['$1'])"; }

say "six models on the FLEURS eval set, one at a time"
run fastconformer_ctc      nemo     "$FASTCONFORMER_NEMO"
run hishab_conformer_large nemo     "hishab/titu_stt_bn_conformer_large"
run hishab_fastconformer   nemo     "hishab/titu_stt_bn_fastconformer"
run whisper_medium         whisper  "SayedShaun/bengali-whisper-medium"   --revision "$(rev SayedShaun/bengali-whisper-medium)"
run wav2vec2               wav2vec2 "SayedShaun/bangla-wave2vec2-unigram" --revision "$(rev SayedShaun/bangla-wave2vec2-unigram)"
run qwen3_adapter          qwen     "Qwen/Qwen3-ASR-1.7B-hf" \
    --revision bcd2b5b7f32b480ab5790554cfa8347f246a14f3 \
    --adapter "$QWEN_ADAPTER"
say "ALL-RUNS-DONE"
