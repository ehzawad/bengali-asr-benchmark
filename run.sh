#!/usr/bin/env bash
# Bring up the Bengali STT web service. Idempotent: creates the venv and fetches
# the model only when they are missing, so re-running is just a restart.
#
#   ./run.sh              start in the foreground
#   ./run.sh --daemon     start detached, log to service.log
#   ./run.sh --stop       stop a running instance
#
# Fetching the model needs a Hugging Face token with read access to the private
# repo, supplied via the environment -- never committed here:
#   HF_TOKEN=hf_xxx ./run.sh
# Once model/stt_bn_fastconformer.nemo exists the token is not needed again.
set -euo pipefail

cd "$(dirname "$0")"

VENV=venv
# Absolute, so the pkill pattern below matches the running command line exactly
# and cannot collide with an unrelated app.py elsewhere on the box.
PYTHON="$PWD/$VENV/bin/python"
MODEL=model/stt_bn_fastconformer.nemo
HF_REPO=ehzawad/stt_bn_fastconformer
PORT=8017

stop_service() {
	# Match the venv interpreter specifically so this never kills an unrelated app.py.
	if pkill -f "$PYTHON app.py" 2>/dev/null; then
		# Wait for the port to actually clear. Returning while the old process
		# still holds it makes the next start fail with "address already in use",
		# and a readiness check would then pass against the *stale* listener.
		for _ in $(seq 1 30); do
			ss -ltn 2>/dev/null | grep -q ":$PORT" || { echo "stopped"; return 0; }
			sleep 1
		done
		pkill -9 -f "$PYTHON app.py" 2>/dev/null || true
		sleep 2
		echo "stopped (forced)"
	else
		echo "not running"
	fi
}

if [[ "${1:-}" == "--stop" ]]; then
	stop_service
	exit 0
fi

if [[ ! -x "$PYTHON" ]]; then
	echo "==> creating venv"
	python3 -m venv "$VENV"
	"$VENV/bin/pip" install --upgrade pip
fi

if ! "$PYTHON" -c "import nemo.collections.asr, gradio, librosa" 2>/dev/null; then
	echo "==> installing dependencies (several GB of CUDA wheels; takes a while)"
	"$VENV/bin/pip" install -r requirements.txt
fi

if [[ ! -f "$MODEL" ]]; then
	echo "==> downloading model from $HF_REPO"
	if [[ -z "${HF_TOKEN:-}" ]]; then
		echo "ERROR: $MODEL is missing and HF_TOKEN is not set." >&2
		echo "The model repo is private; re-run as: HF_TOKEN=hf_xxx $0" >&2
		exit 1
	fi
	mkdir -p model
	# Download to a temp name and rename only on success, so an interrupted
	# transfer cannot leave a truncated file that looks complete next run.
	curl -fL --retry 3 -H "Authorization: Bearer $HF_TOKEN" \
		-o "$MODEL.part" \
		"https://huggingface.co/$HF_REPO/resolve/main/stt_bn_fastconformer.nemo"
	mv "$MODEL.part" "$MODEL"
fi

if [[ "${1:-}" == "--daemon" ]]; then
	stop_service >/dev/null || true
	sleep 1
	setsid nohup "$PYTHON" app.py > service.log 2>&1 < /dev/null &
	pid=$!
	disown
	echo "==> starting in background (pid $pid); waiting for port $PORT"
	scheme=http
	[[ -f certs/cert.pem && -f certs/key.pem && "${STT_NO_SSL:-}" != "1" ]] && scheme=https
	for _ in $(seq 1 60); do
		# Check this pid is alive AND owns the port, so a stale listener from a
		# previous run can never be mistaken for a successful start.
		if ! kill -0 "$pid" 2>/dev/null; then
			echo "ERROR: process exited during startup; see service.log" >&2
			tail -5 service.log >&2
			exit 1
		fi
		if ss -ltnp 2>/dev/null | grep ":$PORT" | grep -q "pid=$pid"; then
			echo "==> up on $scheme://localhost:$PORT  (log: service.log)"
			exit 0
		fi
		sleep 2
	done
	echo "ERROR: did not bind port $PORT in time; see service.log" >&2
	exit 1
fi

exec "$PYTHON" app.py
