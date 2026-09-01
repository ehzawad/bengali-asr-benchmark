#!/usr/bin/env bash
# Benchmark every local checkpoint under identical conditions: same GPU,
# same server process, same shared audio path, same eval set, same concurrency,
# same scorer. Models are run one at a time so none competes for the GPU with
# another -- a concurrent run would measure contention, not the model.
set -euo pipefail
cd "$(dirname "$0")"

PY="$PWD/venv/bin/python"
# Path to the evaluation harness (scripts/benchmark.py from the ASR inference
# pipeline). It is not vendored here -- see README, "Reproducing this".
HARNESS="${HARNESS:?set HARNESS=/path/to/scripts/benchmark.py}"
DATA="$PWD/eval_fleurs_bn"
PORT=8018
CONC=10

MODELS=(
	"ehzawad_fastconformer:model/stt_bn_fastconformer.nemo"
	"hishab_fastconformer:model/titu_stt_bn_fastconformer.nemo"
	"hishab_conformer_large:model/titu_stt_bn_conformer_large.nemo"
	"whisper_medium:model/bengali_whisper_medium"
)

# The wav2vec2 behind the pre-existing ASR service. Its path is recorded in
# .w2v_path rather than hardcoded, since it lives in the shared HF cache.
if [[ -f .w2v_path ]]; then
	MODELS+=("wav2vec2_local:$(cat .w2v_path)")
fi

for entry in "${MODELS[@]}"; do
	name="${entry%%:*}"
	path="${entry#*:}"
	echo "=============================================================="
	echo "==> $name  ($path)"

	# Never leave a previous model resident: it would hold VRAM and, worse,
	# still answer on the port, silently benchmarking the wrong checkpoint.
	pkill -f "$PY bench_server.py" 2>/dev/null || true
	for _ in $(seq 1 30); do
		ss -ltn 2>/dev/null | grep -q ":$PORT" || break
		sleep 1
	done

	STT_MODEL="$path" setsid nohup "$PY" bench_server.py --port "$PORT" \
		> "bench_${name}.log" 2>&1 < /dev/null &
	disown

	ready=0
	for _ in $(seq 1 90); do
		if curl -s --max-time 3 "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; then
			ready=1; break
		fi
		sleep 3
	done
	if [[ $ready -ne 1 ]]; then
		echo "ERROR: $name did not come up; see bench_${name}.log" >&2
		tail -5 "bench_${name}.log" >&2
		continue
	fi

	# Confirm the port is serving the checkpoint we intended, not a leftover.
	serving=$(curl -s "http://127.0.0.1:$PORT/health")
	echo "    serving: $serving"
	case "$serving" in
		*"$path"*) ;;
		*) echo "ERROR: wrong model on :$PORT — $serving" >&2; continue ;;
	esac

	"$PY" "$HARNESS" \
		--api-url "http://127.0.0.1:$PORT/asr" \
		--data-dir "$DATA" \
		--concurrency "$CONC" \
		--output-dir "outputs/$name" 2>&1 | tail -3
done

pkill -f "$PY bench_server.py" 2>/dev/null || true
echo "=============================================================="
echo "done — results under outputs/"
