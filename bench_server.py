"""Serve any local ASR checkpoint under the contract scripts/benchmark.py
expects, so the original benchmark harness can be run against each model
unchanged rather than reimplemented.

A .nemo file loads through NeMo; any other path is treated as a Hugging Face
wav2vec2 directory (see backends.py).

    STT_MODEL=model/<name>.nemo   python bench_server.py [--port 8018]
    STT_MODEL=<hf-wav2vec2-dir>   python bench_server.py [--port 8018]

Contract:
    POST /asr
    {"config": {"language": {"sourceLanguage": "bn"}},
     "audio": [{"audioContent": "<base64 PCM16 WAV>"}]}
 -> {"output": [{"source": "<transcript>"}], "time_taken": <seconds>}

Every model is served through this one process and the shared asr_core audio
path, so preprocessing, chunking and decoding are identical across the models
being compared. Differences in the resulting WER are then attributable to the
checkpoints rather than to two different harnesses.

Inference is serialised behind a lock: one GPU, one model instance, and NeMo's
transcribe() is not documented as re-entrant. Throughput therefore measures
sequential model speed under a fixed client concurrency, not a batching stack --
but it measures it the same way for every model.
"""

import argparse
import base64
import io
import os
import threading
import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import asr_core
import backends

MODEL_PATH = os.environ.get("STT_MODEL", "model/stt_bn_fastconformer.nemo")
DEVICE = backends.DEVICE

print(f"[bench] loading {MODEL_PATH} on {DEVICE}", flush=True)
model = backends.load_model(MODEL_PATH)
print("[bench] ready", flush=True)

_lock = threading.Lock()
api = FastAPI(title="ASR benchmark endpoint")


class _Language(BaseModel):
    sourceLanguage: str = "bn"


class _Config(BaseModel):
    language: _Language = _Language()


class _Audio(BaseModel):
    audioContent: str


class AsrRequest(BaseModel):
    config: _Config = _Config()
    audio: list[_Audio]


@api.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH, "device": DEVICE}


@api.post("/asr")
def asr(req: AsrRequest) -> dict:
    started = time.perf_counter()
    texts: list[str] = []

    for item in req.audio:
        raw = base64.b64decode(item.audioContent)
        audio, _ = asr_core.load_audio(io.BytesIO(raw))
        pieces, _ = asr_core.chunks(audio)
        with _lock:
            texts.append(asr_core.decode(model, pieces))

    return {
        "output": [{"source": t} for t in texts],
        "time_taken": time.perf_counter() - started,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8018)
    args = ap.parse_args()
    uvicorn.run(api, host="127.0.0.1", port=args.port, log_level="warning")
