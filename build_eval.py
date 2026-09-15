"""Rebuild the 1,322-utterance FLEURS Bengali eval set (test + validation)
in the layout the published benchmark used: fleurs_bn_{split}_NNNN.wav at
16 kHz mono PCM16, plus a manifest carrying the raw reference, a SHA-256 per
clip, and the dataset revision — so a later run can prove it scored the same
audio and the same references.
"""
import hashlib, json, re
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

OUT = Path("./eval_fleurs_bn")
OUT.mkdir(parents=True, exist_ok=True)
PUNCT = re.compile(r"[\"“”‘’'।,.!?;:()\[\]{}—–\-]")


def norm(t):
    return re.sub(r"\s+", " ", PUNCT.sub(" ", t)).strip()


rows = []
for split, tag in (("test", "test"), ("validation", "validation")):
    ds = load_dataset("google/fleurs", "bn_in", split=split)
    # datasets 4.x decodes audio through torchcodec; read the container
    # bytes ourselves with soundfile instead of adding that dependency.
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"{split}: {len(ds)} rows", flush=True)
    for i, rec in enumerate(ds, start=1):
        text = (rec.get("transcription") or "").strip()
        audio = rec["audio"]
        if not text or audio is None:
            print(f"  SKIP empty {tag} {i}", flush=True)
            continue
        import io
        raw = audio.get("bytes")
        wav, sr = (sf.read(io.BytesIO(raw), dtype="float32") if raw
                   else sf.read(audio["path"], dtype="float32"))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        assert sr == 16000, sr
        name = f"fleurs_bn_{tag}_{i:04d}.wav"
        p = OUT / name
        sf.write(str(p), wav, 16000, subtype="PCM_16")
        rows.append({"file": name, "split": tag,
                     "raw_reference": text, "reference": norm(text),
                     "duration": round(len(wav) / 16000, 3),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
manifest = {"dataset": "google/fleurs", "config": "bn_in",
            "splits": ["test", "validation"], "n": len(rows),
            "normaliser": "PUNCT->space, collapse ws (benchmark collect_results.norm)",
            "rows": rows}
(OUT.parent / "eval_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=1))
from collections import Counter
print(Counter(r["split"] for r in rows), "total", len(rows))
print("distinct references:", len({r["reference"] for r in rows}))
print("wrote", OUT)
