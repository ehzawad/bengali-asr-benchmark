#!/usr/bin/env python
"""Build the Bangladeshi-Bangla evaluation subsets: seeded, stratified 1,000-utterance samples of four
public sets on the Hugging Face Hub, each pinned to a commit.

  subakko_test   SUST-CSE-Speech/SUBAK.KO, official test split (Bangladeshi standard Bangla)
  banspeech      SUST-CSE-Speech/banspeech, its 13 human-annotated domains (an evaluation-only benchmark)
  cv17_bn_test   fixie-ai/common_voice_17_0 (a mirror of Common Voice 17.0), bn test split
  ben10_valid    bengaliAI/Ben-10, validation split (Bangladeshi regional dialects; its test split is private)

Each selected clip is converted with ffmpeg to 16 kHz mono PCM16 WAV (bit-exact flags) and listed with its
SHA-256 in OUT/<set>/eval_manifest.json, in the format of the other multiset/ manifests. Sampling is the
benchmark's: proportional stratification, largest-remainder allocation, seeded shuffle within stratum,
rows sorted by utt_id. Rows whose normalised reference is empty are not eligible.

  python build_multiset_bd.py --out /content/bench_assets [--sets subakko_test banspeech ...]
"""
import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_score import norm  # noqa: E402  (the benchmark's scorer surface)

SEED, N = 20260915, 1000
SOURCES = {
    "subakko_test": ("SUST-CSE-Speech/SUBAK.KO", "78ec55bd4207a9268df6bf13ef0cd9303817f62a",
                     "SUST SUBAK.KO, official test split (Bangladeshi standard Bangla; CC-BY-4.0)",
                     "stratified by top-level corpus folder"),
    "banspeech": ("SUST-CSE-Speech/banspeech", "484dd7aeaa003dcad122d8e468f363d53215ec63",
                  "SUST BanSpeech, the 13 human-annotated domains (evaluation-only benchmark; "
                  "card: CC-BY-NC-4.0 / CC-BY-4.0)", "stratified by domain"),
    "cv17_bn_test": ("fixie-ai/common_voice_17_0", "34f78a43893414e7b6e271ba94c1d5e05f18b239",
                     "Mozilla Common Voice 17.0, bn test split, via the fixie-ai mirror (CC0)",
                     "stratified by the gender field"),
    "ben10_valid": ("bengaliAI/Ben-10", "ec1613b99066de1cd92dde0e9f054cbb2905d67f",
                    "Bengali.AI Ben-10, validation split (Bangladeshi regional dialects; CC0; the test split "
                    "is private)", "stratified by district"),
}
TEXT_COLUMNS = ("transcription", "transcript", "sentence", "text", "transcripts")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sample(rows):
    """Proportional stratified sample of N rows: largest-remainder allocation, seeded shuffle within stratum."""
    strata = defaultdict(list)
    for r in rows:
        strata[r["stratum"]].append(r)
    quota = {s: N * len(v) / len(rows) for s, v in strata.items()}
    alloc = {s: int(q) for s, q in quota.items()}
    for s in sorted(strata, key=lambda s: (alloc[s] - quota[s], s))[:N - sum(alloc.values())]:
        alloc[s] += 1
    rng = random.Random(SEED)
    picked = []
    for s in sorted(strata):
        members = sorted(strata[s], key=lambda r: r["utt_id"])
        rng.shuffle(members)
        picked += members[:alloc[s]]
    counts = {s: {"source": len(strata[s]), "sampled": alloc[s]} for s in sorted(strata)}
    return sorted(picked, key=lambda r: r["utt_id"]), counts


def text_column(names):
    return next(c for c in TEXT_COLUMNS if c in names)


def parquet_rows(repo, rev, files, describe):
    """Candidate rows from parquet shards; audio bytes are fetched later for the sampled rows only."""
    rows = []
    for fi, name in enumerate(files):
        local = hf_hub_download(repo, name, repo_type="dataset", revision=rev)
        cols = pq.ParquetFile(local).schema_arrow.names
        meta = [c for c in cols if c != "audio"]
        table = pq.read_table(local, columns=meta).to_pylist()
        tc = text_column(cols)
        for ri, rec in enumerate(table):
            utt_id, stratum = describe(name, rec)
            rows.append({"utt_id": utt_id, "stratum": stratum, "raw_reference": (rec[tc] or "").strip(),
                         "_src": (local, ri)})
    return rows


def parquet_audio(picked):
    by_file = defaultdict(list)
    for r in picked:
        by_file[r["_src"][0]].append(r)
    for local, members in by_file.items():
        audio = pq.read_table(local, columns=["audio"]).column("audio").to_pylist()
        for r in members:
            a = audio[r["_src"][1]]
            r["_bytes"], r["_ext"] = a["bytes"], Path(a.get("path") or "x.wav").suffix or ".wav"


def load(set_name):
    repo, rev, _, _ = SOURCES[set_name]
    files = list_repo_files(repo, repo_type="dataset", revision=rev)
    if set_name == "subakko_test":
        shards = sorted(f for f in files if re.fullmatch(r"data/test-\d+-of-\d+\.parquet", f))
        def describe(_, rec):
            parts = rec["file_path"].strip("/").split("/")
            return rec["file_path"].strip("/").rsplit(".", 1)[0].replace("/", "__"), parts[1]
        rows = parquet_rows(repo, rev, shards, describe)
    elif set_name == "banspeech":
        shards = sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))
        def describe(name, rec):
            domain = Path(name).name.split("-00000-")[0]
            path = rec.get("file_path") or rec.get("path") or ""
            return f"{domain}__{Path(path).stem}", domain
        rows = parquet_rows(repo, rev, shards, describe)
    elif set_name == "cv17_bn_test":
        shards = sorted(f for f in files if f.startswith("bn/test/") and f.endswith(".parquet"))
        def describe(_, rec):
            return Path(rec["path"]).stem, rec.get("gender") or "unspecified"
        rows = parquet_rows(repo, rev, shards, describe)
    elif set_name == "ben10_valid":
        import csv
        local = hf_hub_download(repo, "valid/valid.csv", repo_type="dataset", revision=rev)
        with open(local, encoding="utf-8-sig", newline="") as f:
            rows = [{"utt_id": re.sub(r"[^\w-]+", "_", Path(r["file_name"]).stem).strip("_"),
                     "stratum": r["district"], "raw_reference": (r["transcripts"] or "").strip(),
                     "_file": "valid/" + r["file_name"]} for r in csv.DictReader(f)]
        files = set(files)
        rows = [r for r in rows if r["_file"] in files]
    else:
        raise SystemExit(f"unknown set {set_name}")
    eligible = [r for r in rows if norm(r["raw_reference"])]
    ids = [r["utt_id"] for r in eligible]
    assert len(ids) == len(set(ids)), f"{set_name}: utterance ids are not unique"
    return rows, eligible


def build(set_name, out_root, ffmpeg_version):
    repo, rev, source, stratification = SOURCES[set_name]
    rows, eligible = load(set_name)
    picked, counts = sample(eligible)
    if set_name == "ben10_valid":
        for r in picked:
            local = hf_hub_download(repo, r["_file"], repo_type="dataset", revision=rev)
            r["_bytes"], r["_ext"] = Path(local).read_bytes(), Path(local).suffix
    else:
        parquet_audio(picked)
    out = Path(out_root) / set_name
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for r in picked:
            src = Path(tmp) / ("in" + r["_ext"])
            src.write_bytes(r["_bytes"])
            dst = out / "wavs" / (re.sub(r"[^\w.-]+", "_", r["utt_id"]) + ".wav")
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", "-map_metadata", "-1", "-fflags", "+bitexact",
                            "-flags:a", "+bitexact", str(dst)], check=True)
            info = sf.info(str(dst))
            assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
            manifest_rows.append({"file": str(dst.resolve()), "split": "test", "utt_id": r["utt_id"],
                                  "stratum": r["stratum"], "raw_reference": r["raw_reference"],
                                  "reference": r["raw_reference"], "duration": round(info.frames / 16000, 3),
                                  "sha256": sha256(dst)})
    manifest = {
        "dataset": set_name, "source": source, "source_repo": repo, "source_revision": rev,
        "source_manifest": None, "source_manifest_sha256": None, "source_rows": len(eligible),
        "source_rows_before_filter": len(rows), "n": len(manifest_rows), "seed": SEED,
        "sampling": f"proportional stratified ({stratification}), largest-remainder allocation, seeded shuffle "
                    "within stratum, sorted by utt_id; rows with an empty normalised reference not eligible",
        "strata": counts, "speakers": None, "conversion": f"ffmpeg ({ffmpeg_version}) -> 16 kHz mono PCM16, bit-exact",
        "hours": round(sum(r["duration"] for r in manifest_rows) / 3600, 3),
        "ref_words": sum(len(norm(r["raw_reference"]).split()) for r in manifest_rows),
        "rows": manifest_rows,
    }
    text = json.dumps(manifest, indent=1, ensure_ascii=False)
    (out / "eval_manifest.json").write_text(text)
    print(f"{set_name}: {len(rows)} rows, {len(eligible)} eligible, sampled {len(manifest_rows)} "
          f"({manifest['hours']} h, {manifest['ref_words']} words); strata {counts}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=list(SOURCES))
    a = ap.parse_args()
    version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    for s in a.sets:
        build(s, a.out, version.replace("ffmpeg version ", "").split(" Copyright")[0])


if __name__ == "__main__":
    main()
