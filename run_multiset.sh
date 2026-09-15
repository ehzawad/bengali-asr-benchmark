#!/usr/bin/env bash
# Multi-dataset comparison on OFFICIAL Bengali test splits (2026-09-15).
# Sets: frozen seeded 1,000-utterance subsets of Vaani official test and
# SPRING-INX R1/R2 official eval (multiset/<set>/eval_manifest.json, per-file
# sha256), later Kathbath official test. One model resident at a time on the
# A5000 (by UUID). fastconformer_ctc is NOT decoded on Vaani/SPRING here: its
# one-shot hypotheses already exist and are re-scored  — decoding
# it again would be a repeat of a one-shot panel.
#
# Co-tenant control (owner's instruction): a run refuses to start while a
# foreign compute process holds the card (bench_run_set.py exit 4); while a
# run is in flight a watcher polls nvidia-smi every 15 s, appends every foreign
# process it sees to <out>/cotenant_<label>.log, and on the first sighting
# creates the STOP file — the harness finishes the clip in hand, writes what it
# has with status "stopped", and exits 3. Nothing is ever signalled. A stopped
# run is re-launched by re-running this script once the card is free (the
# harness refuses to overwrite a completed run, so only stopped labels re-run
# after their directory is moved aside by hand — never automatically).
set -u
cd "$(dirname "$0")"
P=/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline
GPU="${BENCH_GPU:?set BENCH_GPU to a GPU UUID}"
export CUDA_VISIBLE_DEVICES="$GPU"
OUT=outputs_multiset
LOG=$OUT/run_multiset.log
STOP=$OUT/STOP
mkdir -p "$OUT"; rm -f "$STOP"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

foreign() {  # pids of compute processes on OUR card not owned by us
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', *' -v g="$GPU" '$1==g {print $2" "$3}' \
  | while read -r pid mib; do
      [ -e /proc/$pid ] || continue
      [ "$(stat -c %u /proc/$pid)" = "$(id -u)" ] || echo "$pid $mib"
    done
}

watch() {  # background: log foreign processes; create STOP on first sighting
  local label=$1
  while true; do
    f=$(foreign)
    if [ -n "$f" ]; then
      echo "$(date -u +%FT%TZ) FOREIGN $f" >> "$OUT/cotenant_$label.log"
      [ -e "$STOP" ] || { touch "$STOP"; say "co-tenant on the card during $label: $f -> STOP requested"; }
    fi
    sleep 15
  done
}

run() { # set label kind model [extra...]
  local set=$1 label=$2 kind=$3 model=$4; shift 4
  local tag="${set}__${label}"
  if [ -f "$OUT/$tag/run_meta.json" ]; then say "skip $tag (done)"; return 0; fi
  local py=$P/.venv/bin/python
  [ "$kind" = "qwen" ] && py=$P/.venv-qwen/bin/python
  while [ -n "$(foreign)" ]; do say "waiting: foreign process on the card: $(foreign | tr '\n' ' ')"; sleep 60; done
  rm -f "$STOP"
  say "=== $tag ($kind) ==="
  watch "$tag" & local wpid=$!
  nice -n 19 $py bench_run_set.py --kind "$kind" --model "$model" --label "$tag" \
      --manifest "multiset/$set/eval_manifest.json" --out-root "$OUT" --stop-file "$STOP" "$@" \
      >> "$OUT/logs_$tag.txt" 2>&1
  local rc=$?
  kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null   # our own watcher subshell, by pid
  say "$tag rc=$rc $(tail -1 "$OUT/logs_$tag.txt" | head -c 140)"
  return $rc
}

rev() { python3 -c "
import json; print(json.load(open('checkpoint_revisions.json'))['$1'])"; }

say "PROSPECTIVE: comparison on official test splits, frozen subsets (seed 20260915); fastconformer_ctc re-scored from stored one-shot hypotheses, not re-decoded; one model at a time; foreign process on the card = cooperative stop"
for set in vaani_test spring_r1_test spring_r2_test; do
  run $set hishab_conformer_large nemo     "hishab/titu_stt_bn_conformer_large"
  run $set hishab_fastconformer   nemo     "hishab/titu_stt_bn_fastconformer"
  run $set wav2vec2               wav2vec2 "SayedShaun/bangla-wave2vec2-unigram" --revision "$(rev SayedShaun/bangla-wave2vec2-unigram)"
  run $set whisper_medium         whisper  "SayedShaun/bengali-whisper-medium"   --revision "$(rev SayedShaun/bengali-whisper-medium)"
done
for set in vaani_test spring_r1_test spring_r2_test; do
  run $set qwen3_adapter          qwen     "Qwen/Qwen3-ASR-1.7B-hf" \
      --revision bcd2b5b7f32b480ab5790554cfa8347f246a14f3 \
      --adapter "$P/experiments/qwen_final/candidates/step_17280"
done
say "ALL-RUNS-DONE"
