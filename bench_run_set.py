#!/usr/bin/env python
"""bench_run.py generalised to any eval manifest — the 2026-09-15 multi-dataset
comparison on official Bengali test splits (Vaani test, SPRING-INX R1/R2 eval,
Kathbath test). The decode path is bench_run.py's, unchanged by construction:
the backends and transcribe_clip are IMPORTED from it, not copied.

Differences from bench_run.py, all of them plumbing:
  --manifest    eval manifest (rows carry absolute "file" paths + sha256)
  --out-root    where outputs/<label>/ goes (never the published outputs/)
  --stop-file   if this path appears, finish the current clip, write what was
                decoded so far with status "stopped", and exit 3 — the
                co-tenant watcher creates it when a foreign process lands on
                our card; nothing is ever signalled
Before the model is loaded the card is checked for foreign compute processes
and the run refuses to start if one is present (exit 4). Per-file sha256 is
verified against the manifest before decoding (exit 5 on mismatch), so every
model provably read the same bytes.
"""
import argparse, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_run as BR                      # backends + transcribe_clip, unchanged


def foreign_compute_apps():
    """(pid, user, MiB) of every compute process on the visible card(s) that is
    not ours. CUDA_VISIBLE_DEVICES holds our card's UUID."""
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"], text=True)
    except Exception:
        return []
    me = os.getuid(); hits = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3 or (uuid and parts[0] != uuid):
            continue
        pid = int(parts[1])
        try:
            owner = os.stat(f"/proc/{pid}").st_uid
        except FileNotFoundError:
            continue
        if owner != me:
            hits.append((pid, owner, int(parts[2])))
    return hits


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=list(BR.KINDS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--revision")
    ap.add_argument("--label", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    man = json.loads(Path(a.manifest).read_text())
    rows = man["rows"][:a.limit] if a.limit else man["rows"]
    out_dir = Path(a.out_root) / a.label
    if (out_dir / "run_meta.json").exists():
        print(f"[{a.label}] {out_dir}/run_meta.json exists; refusing to overwrite", flush=True)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    foreign = foreign_compute_apps()
    if foreign:
        print(f"[{a.label}] REFUSING to start: foreign compute process on our card: {foreign}", flush=True)
        return 4
    bad = [r["file"] for r in rows if sha256_file(r["file"]) != r["sha256"]]
    if bad:
        print(f"[{a.label}] REFUSING: {len(bad)} file(s) do not match the manifest sha256, e.g. {bad[0]}", flush=True)
        return 5

    print(f"[{a.label}] loading {a.kind}: {a.model}", flush=True)
    be = BR.KINDS[a.kind](a.model, adapter=a.adapter, revision=a.revision)

    BR.transcribe_clip(be, Path(rows[0]["file"]))   # warm-up, discarded
    torch.cuda.synchronize()

    preds, times, failures, status = [], [], 0, "complete"
    gpu_before = BR.gpu_state()
    t_all = time.perf_counter()
    for i, r in enumerate(rows):
        if a.stop_file and Path(a.stop_file).exists():
            status = f"stopped by {a.stop_file} after {i} clips"
            print(f"[{a.label}] {status}", flush=True)
            break
        p = Path(r["file"])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            (txt, npieces, forced), err = BR.transcribe_clip(be, p), None
        except Exception as e:                      # failure => empty hypothesis
            txt, npieces, forced = "", 0, 0
            err = f"{type(e).__name__}: {e}"
            failures += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        preds.append({"file": r["file"], "utt_id": r.get("utt_id"), "split": r["split"],
                      "stratum": r.get("stratum"),
                      "reference": r["reference"], "raw_reference": r["raw_reference"],
                      "ok": err is None, "transcript": txt,
                      "wall_clock_seconds": round(dt, 6), "error": err,
                      "n_pieces": npieces, "forced_cuts": forced})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}  {np.mean(times)*1000:.1f} ms/clip", flush=True)
    total = time.perf_counter() - t_all

    audio_s = sum(r["duration"] for r in rows[:len(preds)])
    t = np.array(times) if times else np.array([np.nan])
    meta = {
        "label": a.label, "kind": a.kind, "model": a.model,
        "adapter": a.adapter, "revision": a.revision,
        "dataset": man.get("dataset"), "manifest": str(Path(a.manifest).resolve()),
        "manifest_sha256": sha256_file(a.manifest),
        "status": status, "n": len(preds), "n_manifest": len(rows), "failures": failures,
        "audio_seconds": round(audio_s, 1),
        "ms_per_clip_mean": round(float(np.nanmean(t) * 1000), 2),
        "ms_per_clip_median": round(float(np.nanmedian(t) * 1000), 2),
        "ms_per_clip_p95": round(float(np.nanpercentile(t, 95) * 1000), 2),
        "inference_wall_s": round(float(np.nansum(t)), 1),
        "loop_wall_s": round(total, 1),
        "rtf": round(float(np.nansum(t)) / audio_s, 5) if audio_s else None,
        "batch_size": 1, "decoding": "greedy, no external LM or rescoring",
        "gpu_before": gpu_before, "gpu_after": BR.gpu_state(),
        "foreign_compute_apps_at_start": [], "harness": "bench_run_set.py over bench_run.py backends",
        "torch": torch.__version__, "python": platform.python_version(),
    }
    if a.kind == "whisper":
        meta["generation_config_refreshed"] = bool(getattr(be, "refreshed", False))
    (out_dir / "predictions.json").write_text(json.dumps(preds, ensure_ascii=False, indent=1))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[{a.label}] {status}: {meta['ms_per_clip_mean']} ms/clip  failures={failures}  n={len(preds)}", flush=True)
    return 0 if status == "complete" else 3


if __name__ == "__main__":
    sys.exit(main())
