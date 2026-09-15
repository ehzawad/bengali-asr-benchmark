#!/usr/bin/env bash
# Kathbath official test (known / unknown speakers), frozen 1,000-utterance
# subsets, the models (Qwen dropped by the owner) including fastconformer_ctc (first decode of Kathbath test).
# Same guards as run_multiset.sh. Waits for run_multiset.sh (pid in $WAIT_PID)
# to exit before touching the card, so our own two runs never overlap.
set -u
cd "$(dirname "$0")"
P=${PIPELINE_ROOT:-/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline}
FASTCONFORMER_NEMO=${FASTCONFORMER_NEMO:-model/stt_bn_fastconformer_ctc.nemo}
QWEN_ADAPTER=${QWEN_ADAPTER:-$P/experiments/qwen_final/candidates/step_17280}
GPU="${BENCH_GPU:?set BENCH_GPU to a GPU UUID}"
export CUDA_VISIBLE_DEVICES="$GPU"
OUT=outputs_multiset
LOG=$OUT/run_multiset_kathbath.log
STOP=$OUT/STOP_kathbath
mkdir -p "$OUT"; rm -f "$STOP"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  say "run_multiset.sh (pid $WAIT_PID) has exited; starting"
fi
foreign() {
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', *' -v g="$GPU" '$1==g {print $2" "$3}' \
  | while read -r pid mib; do [ -e /proc/$pid ] || continue; [ "$(stat -c %u /proc/$pid)" = "$(id -u)" ] || echo "$pid $mib"; done
}
watch() { local label=$1; while true; do f=$(foreign); if [ -n "$f" ]; then echo "$(date -u +%FT%TZ) FOREIGN $f" >> "$OUT/cotenant_$label.log"; [ -e "$STOP" ] || { touch "$STOP"; say "co-tenant on the card during $label: $f -> STOP requested"; }; fi; sleep 15; done; }
run() { local set=$1 label=$2 kind=$3 model=$4; shift 4; local tag="${set}__${label}"
  if [ -f "$OUT/$tag/run_meta.json" ]; then say "skip $tag (done)"; return 0; fi
  local py=${BENCH_PY:-$P/.venv/bin/python}; [ "$kind" = "qwen" ] && py=${BENCH_PY_QWEN:-$P/.venv-qwen/bin/python}
  while [ -n "$(foreign)" ]; do say "waiting: foreign process on the card: $(foreign | tr '\n' ' ')"; sleep 60; done
  rm -f "$STOP"; say "=== $tag ($kind) ==="; watch "$tag" & local wpid=$!
  nice -n 19 $py bench_run_set.py --kind "$kind" --model "$model" --label "$tag" --manifest "multiset/$set/eval_manifest.json" --out-root "$OUT" --stop-file "$STOP" "$@" >> "$OUT/logs_$tag.txt" 2>&1
  local rc=$?; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
  say "$tag rc=$rc $(tail -1 "$OUT/logs_$tag.txt" | head -c 140)"; return $rc; }
rev() { python3 -c "
import json; print(json.load(open('checkpoint_revisions.json'))['$1'])"; }
say "official test split, frozen 1,000-utterance subset (seed 20260915); one model at a time"
for set in kathbath_test_known kathbath_test_unknown; do
  run $set fastconformer_ctc nemo "$FASTCONFORMER_NEMO"
  run $set hishab_conformer_large nemo     "hishab/titu_stt_bn_conformer_large"
  run $set hishab_fastconformer   nemo     "hishab/titu_stt_bn_fastconformer"
  run $set wav2vec2               wav2vec2 "SayedShaun/bangla-wave2vec2-unigram" --revision "$(rev SayedShaun/bangla-wave2vec2-unigram)"
  run $set whisper_medium         whisper  "SayedShaun/bengali-whisper-medium"   --revision "$(rev SayedShaun/bengali-whisper-medium)"
done
# Qwen adapter dropped by the owner (2026-09-15 17:20 UTC): too slow at batch 1.
say "ALL-RUNS-DONE"
