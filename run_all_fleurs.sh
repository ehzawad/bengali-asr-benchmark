#!/usr/bin/env bash
# Seven models, one GPU (the A5000, by UUID), one model resident at a time, the
# same 1,322-clip eval set and loader as run_all.sh. POST-RELEASE REPRODUCIBILITY
# RUN — declared prospectively, before any decode:
#   * stt_bn_fastconformer_ctc (step_116000, sha bea300ce…) was
#     selected and frozen on its own development panels before this script
#     existed. Nothing here selects, tunes or re-ranks anything.
#   * This is the THIRD time this model's lineage decodes FLEURS-bn test
#     (events one and two are registered in SPEC 8w.14 / 8w.18). FLEURS is
#     therefore not a held-out set for it in this run; these numbers are a
#     reproducibility check under one GPU, one loader and one scorer on one
#     date, never a clean generalisation estimate.
#   * Baselines are re-decoded so the comparison is made under identical
#     conditions; their published numbers stand.
# Outputs go under outputs/<label> so the published outputs/<label>
# trees are never touched. No pkill anywhere: bench_run.py is a direct child.
set -u
cd "$(dirname "$0")"
P=/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline
GPU="${BENCH_GPU:?set BENCH_GPU to a GPU UUID}"
SFX=
LOG=run_all_p4_repro.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }

run() { # label kind model [extra...]
  local label=$1$SFX kind=$2 model=$3; shift 3
  if [ -f "outputs/$label/run_meta.json" ]; then say "skip $label (done)"; return 0; fi
  local py=$P/.venv/bin/python
  [ "$kind" = "qwen" ] && py=$P/.venv-qwen/bin/python
  say "=== $label ($kind) ==="
  CUDA_VISIBLE_DEVICES=$GPU nice -n 19 $py bench_run.py --kind "$kind" --model "$model" \
      --label "$label" "$@" >> "logs_$label.txt" 2>&1
  local rc=$?
  say "$label rc=$rc $(tail -1 logs_$label.txt | head -c 120)"
  return $rc
}

rev() { python3 -c "
import json; print(json.load(open('checkpoint_revisions.json'))['$1'])"; }

say "PROSPECTIVE: post-release reproducibility run; fastconformer_ctc = third FLEURS exposure; no selection from these numbers"
run fastconformer_ctc nemo     "$P/experiments/p4/runs/N_O+macro_filtered_H120k/export/step_116000.nemo"
run hishab_conformer_large nemo     "hishab/titu_stt_bn_conformer_large"
run hishab_fastconformer   nemo     "hishab/titu_stt_bn_fastconformer"
run whisper_medium         whisper  "SayedShaun/bengali-whisper-medium"   --revision "$(rev SayedShaun/bengali-whisper-medium)"
run wav2vec2               wav2vec2 "SayedShaun/bangla-wave2vec2-unigram" --revision "$(rev SayedShaun/bangla-wave2vec2-unigram)"
run qwen3_adapter          qwen     "Qwen/Qwen3-ASR-1.7B-hf" \
    --revision bcd2b5b7f32b480ab5790554cfa8347f246a14f3 \
    --adapter "$P/experiments/qwen_final/candidates/step_17280"
say "ALL-RUNS-DONE"
