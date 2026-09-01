"""Isolate per-utterance model compute time, one model at a time.

The benchmark's throughput figure is 1/service_time under a serialised lock with
10 clients queued behind it, so its "latency" is mostly queueing and its
throughput folds in HTTP, base64 and file-staging overhead. This probe strips
all of that: same audio, same order, in-process, sequential, no server.

Each model is loaded alone and released before the next, so only one set of
weights is resident at a time. A warm-up pass is discarded because the first
call pays CUDA context and kernel autotuning costs that no later call repeats.
Several repeats are timed so the spread is visible rather than assumed away.
"""

import gc
import io
import json
import statistics
import sys
import tarfile
import time

import numpy as np
import torch

import asr_core
import backends
from pathlib import Path

W2V = Path(".w2v_path").read_text().strip() if Path(".w2v_path").exists() else None

MODELS = [
    ("ehzawad fastconformer", "model/stt_bn_fastconformer.nemo"),
    ("hishab fastconformer", "model/titu_stt_bn_fastconformer.nemo"),
    ("hishab conformer_large", "model/titu_stt_bn_conformer_large.nemo"),
    ("whisper medium", "model/bengali_whisper_medium"),
] + ([("wav2vec2 bangla", W2V)] if W2V else [])
N_CLIPS = 60
REPEATS = 3


def load_clips(n: int) -> list[np.ndarray]:
    """Same clips, same order, for every model."""
    clips = []
    with tarfile.open("eval_fleurs_bn/data.tar") as tar:
        for member in tar:
            if not member.isfile():
                continue
            raw = tar.extractfile(member).read()
            audio, _ = asr_core.load_audio(io.BytesIO(raw))
            clips.append(audio)
            if len(clips) >= n:
                break
    return clips


def main() -> None:
    clips = load_clips(N_CLIPS)
    total_audio = sum(len(c) for c in clips) / asr_core.TARGET_SR
    print(f"{len(clips)} clips, {total_audio:.1f}s of audio\n")

    results = {}
    for label, path in MODELS:
        model = backends.load_model(path)

        asr_core.decode(model, [clips[0]])  # warm-up, discarded

        runs = []
        for _ in range(REPEATS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for c in clips:
                asr_core.decode(model, [c])
            torch.cuda.synchronize()
            runs.append(time.perf_counter() - t0)

        per_clip = [r / len(clips) * 1000 for r in runs]
        rtf = total_audio / statistics.mean(runs)
        results[label] = (statistics.mean(per_clip), min(per_clip), max(per_clip), rtf)
        print(f"{label:24s} {statistics.mean(per_clip):6.1f} ms/clip "
              f"(min {min(per_clip):.1f}, max {max(per_clip):.1f})   "
              f"{rtf:6.1f}x realtime")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    json.dump(results, open("outputs/speed_probe.json", "w"), indent=2)
    print("\nwrote outputs/speed_probe.json")


if __name__ == "__main__":
    sys.exit(main())
