"""Bengali speech-to-text web service.

Serves ehzawad/stt_bn_fastconformer (FastConformer-CTC large, 1024-piece Bengali
BPE) behind a Gradio UI on port 8017: record from the browser mic or upload a
file, get Bengali text back.

The model is a local .nemo checkpoint (model/stt_bn_fastconformer.nemo), so this
process never needs a Hugging Face token.

Audio is coerced to what the model was trained on -- 16 kHz mono -- rather than
fed in at the wrong rate, which a Conformer front-end would otherwise transcribe
as confident garbage.

LONG AUDIO. Training capped utterances at 30 s (max_duration in
model_config.yaml), so longer clips are segmented. Segmentation is only allowed
to cut at a frame quiet enough to plausibly be a pause: each side of a cut is
decoded independently, so a cut placed inside a word costs real accuracy (lost
boundary tokens, a fabricated word break, different BPE segmentation either
side). When no quiet frame exists in the search window the cut is forced anyway
and the UI says so, because silently returning a damaged transcript is worse
than saying which clips were hard to segment.

The displayed transcript is RAW decoder output. `normalize_bn_v1` -- the repo's
frozen scoring surface -- is shown separately and labelled as such: it belongs to
WER measurement, where it is applied symmetrically to reference and hypothesis,
and using it as the display surface would hide what the model actually predicted.
"""

import logging
import os
import sys
from pathlib import Path

import gradio as gr

import asr_core
import torch

REPO_ROOT = Path(__file__).parent
MODEL_PATH = REPO_ROOT / "model" / "stt_bn_fastconformer.nemo"
TARGET_SR = 16000

CHUNK_SECONDS = 25.0        # under the 30 s training cap
SEARCH_SECONDS = 2.0        # look this far *back* from a boundary for a pause
MIN_PIECE_SECONDS = 0.3     # train_ds min_duration is 0.1; keep clear of it
MAX_CLIP_SECONDS = 600.0    # abuse ceiling, not a model limit
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
QUEUE_MAX = 16
PORT = 8017

# Set when served behind a reverse proxy at a subpath (Caddy strips the prefix
# with handle_path, so Gradio sees "/" but must still emit "/stt/..." URLs to the
# browser). Empty means "reached directly on PORT", which is the LAN case.
ROOT_PATH = os.environ.get("STT_ROOT_PATH", "")

# HTTPS is not decoration here: browsers gate getUserMedia() behind a secure
# context, so over plain HTTP the mic is blocked before this app is reached.
# localhost is exempt from that rule, a LAN IP is not. Used automatically when
# the cert pair exists; set STT_NO_SSL=1 to force plain HTTP.
CERT_DIR = REPO_ROOT / "certs"
SSL_CERT = CERT_DIR / "cert.pem"
SSL_KEY = CERT_DIR / "key.pem"
USE_SSL = (
    SSL_CERT.exists() and SSL_KEY.exists() and os.environ.get("STT_NO_SSL") != "1"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stt_bn")

# The frozen scoring normalizer lives in the pipeline repo, which may not be
# checked out beside this file. Its absence downgrades a feature, not the service.
try:
    sys.path.insert(0, str(REPO_ROOT / "bengali-asr-pipeline"))
    from src.bn_normalize import normalize_bn_v1
except Exception:  # noqa: BLE001 - optional integration
    normalize_bn_v1 = None
    log.warning("bengali-asr-pipeline not found; scoring-surface output disabled")

log.info("loading %s ...", MODEL_PATH.name)

import nemo.collections.asr as nemo_asr  # noqa: E402  (slow import; after logging is up)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = nemo_asr.models.ASRModel.restore_from(str(MODEL_PATH), map_location=DEVICE)
model.eval()
log.info("model ready on %s", DEVICE)


# The audio path lives in asr_core so that this demo and the benchmark endpoint
# share one implementation -- if they drifted, a preprocessing difference would
# masquerade as a model-quality difference in the benchmark.
_resample = asr_core.resample
_downmix = asr_core.downmix
_load_audio = asr_core.load_audio
_chunks = asr_core.chunks


def _decode(pieces):
    return asr_core.decode(model, pieces)


def transcribe(audio_path):
    if not audio_path:
        return "", "", "Give me some audio first — record above, or upload a file."

    try:
        audio, downmix_note = _load_audio(audio_path)
    except Exception:
        log.exception("could not decode %s", audio_path)
        return "", "", "Could not decode that audio. Try a WAV, FLAC, MP3, or OGG file."

    duration = len(audio) / TARGET_SR
    if duration < 0.1:
        return "", "", "That clip is under 0.1 s — too short to transcribe."
    if duration > MAX_CLIP_SECONDS:
        return "", "", (
            f"That clip is {duration / 60:.1f} min. This demo caps input at "
            f"{MAX_CLIP_SECONDS / 60:.0f} min."
        )

    pieces, forced = _chunks(audio)
    try:
        text = _decode(pieces)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        log.exception("CUDA OOM on a %.1f s clip", duration)
        return "", "", "The GPU ran out of memory. Try a shorter clip."
    except Exception:
        log.exception("transcription failed on a %.1f s clip", duration)
        return "", "", "Transcription failed. The error is in the server log."

    notes = [f"{duration:.1f} s", f"{len(pieces)} segment{'s' if len(pieces) != 1 else ''}", DEVICE.upper()]
    if forced:
        notes.append(f"⚠ {forced} cut{'s' if forced != 1 else ''} forced mid-speech")
    if downmix_note:
        notes.append(f"⚠ {downmix_note}")
    if not text:
        notes.append("⚠ model returned an empty hypothesis")

    scored = normalize_bn_v1(text) if (normalize_bn_v1 and text) else ""
    return text, scored, " · ".join(notes)


with gr.Blocks(
    title="Bengali STT — FastConformer",
    # Uploaded speech is sensitive; do not let Gradio's cache accumulate it.
    delete_cache=(600, 600),
) as demo:
    gr.Markdown(
        "# বাংলা Speech-to-Text\n"
        "`ehzawad/stt_bn_fastconformer` — FastConformer-CTC large, greedy decode, "
        "no language model. Record or upload; audio is converted to 16 kHz mono "
        "automatically. Clips over 25 s are segmented at pauses before decoding."
    )

    with gr.Row():
        with gr.Column():
            mic = gr.Audio(
                sources=["microphone"], type="filepath", format="wav", label="Record"
            )
            mic_btn = gr.Button("Transcribe recording", variant="primary")
        with gr.Column():
            upload = gr.Audio(
                sources=["upload"], type="filepath", format="wav", label="Upload audio file"
            )
            up_btn = gr.Button("Transcribe file", variant="primary")

    out = gr.Textbox(label="Transcript (raw model output)", lines=6, buttons=["copy"])
    scored_out = gr.Textbox(
        label="Scoring surface (normalize_bn_v1) — used for WER, not the model's own output",
        lines=3,
        buttons=["copy"],
    )
    meta = gr.Markdown()

    # One GPU, one model instance: every request goes through a single lane.
    for btn, src in ((mic_btn, mic), (up_btn, upload)):
        btn.click(
            transcribe,
            inputs=src,
            outputs=[out, scored_out, meta],
            concurrency_id="gpu",
            concurrency_limit=1,
        )

if __name__ == "__main__":
    scheme = "https" if USE_SSL else "http"
    log.info("serving %s://0.0.0.0:%d%s", scheme, PORT, ROOT_PATH or "")

    demo.queue(max_size=QUEUE_MAX, default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        max_file_size=MAX_UPLOAD_BYTES,
        root_path=ROOT_PATH,
        ssl_certfile=str(SSL_CERT) if USE_SSL else None,
        ssl_keyfile=str(SSL_KEY) if USE_SSL else None,
        # The cert is self-signed, so Gradio's own startup call back to the
        # server would fail verification and abort the launch.
        ssl_verify=False,
        show_error=False,
        quiet=False,
    )
