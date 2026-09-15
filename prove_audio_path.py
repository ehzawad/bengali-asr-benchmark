#!/usr/bin/env python
"""Prove — not assert — that both interpreters stage byte-identical audio.

Runs the vendored asr_core (load_audio -> chunks -> PCM16 staging) over a
sample that deliberately includes the long clips which trigger 25 s chunking,
and prints a SHA-256 per staged chunk. Run under each venv and diff: equal
hashes mean the bytes reaching every model's encoder are the same bytes.
"""
import hashlib, json, sys, tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

import asr_core

ASSETS = Path(__file__).resolve().parent
man = json.loads((ASSETS / "eval_manifest.json").read_text())
rows = man["rows"]
longest = sorted(rows, key=lambda r: -r["duration"])[:5]      # force chunking
sample = longest + rows[:10] + rows[-5:]

out = []
for r in sample:
    audio, _ = asr_core.load_audio(str(ASSETS / "eval_fleurs_bn" / r["file"]))
    pieces, _forced = asr_core.chunks(audio)
    hs = []
    with tempfile.TemporaryDirectory() as td:
        for i, piece in enumerate(pieces):
            p = Path(td) / f"chunk_{i:03d}.wav"
            sf.write(str(p), np.clip(piece, -1.0, 1.0), asr_core.TARGET_SR,
                     subtype="PCM_16")
            hs.append(hashlib.sha256(p.read_bytes()).hexdigest()[:16])
    out.append({"file": r["file"], "dur": r["duration"],
                "n_pieces": len(pieces), "chunk_sha256": hs})

print(json.dumps({"python": sys.version.split()[0],
                  "numpy": np.__version__, "soundfile": sf.__version__,
                  "n_multipiece": sum(1 for o in out if o["n_pieces"] > 1),
                  "clips": out}, indent=1))
