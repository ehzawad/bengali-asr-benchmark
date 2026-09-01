"""Model loading for the benchmark, one entry point for both families.

`load_model` returns an object exposing NeMo's `transcribe(paths, batch_size,
verbose)` signature whatever the underlying family, so asr_core.decode drives
every model through byte-identical audio handling, chunking and staging. Leaving
each family its own preprocessing would put a confound in a comparison whose
purpose is to remove them.
"""

import torch

import asr_core

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Wav2Vec2CTC:
    """Wav2Vec2 wearing NeMo's transcribe() signature.

    Greedy CTC over argmax, matching the no-language-model decoding used for the
    NeMo checkpoints. The service this model was taken from runs an ONNX export
    under Triton; this is PyTorch, chosen so a comparison against the NeMo models
    isolates the model rather than the runtime.
    """

    def __init__(self, path: str):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained(path)
        self.model = Wav2Vec2ForCTC.from_pretrained(path).to(DEVICE).eval()

    def eval(self):
        return self

    def transcribe(self, paths, batch_size=1, verbose=False):
        from types import SimpleNamespace

        import soundfile as sf

        out = []
        for p in paths:
            audio, _ = sf.read(p, dtype="float32")
            inputs = self.processor(
                audio, sampling_rate=asr_core.TARGET_SR, return_tensors="pt"
            )
            with torch.inference_mode():
                logits = self.model(inputs.input_values.to(DEVICE)).logits
            ids = logits.argmax(dim=-1)
            out.append(SimpleNamespace(text=self.processor.batch_decode(ids)[0]))
        return out


class WhisperSeq2Seq:
    """Whisper wearing NeMo's transcribe() signature.

    Decoding is greedy (num_beams=1) to match the no-language-model, no-beam
    decoding used for every other model here. Anything wider would give Whisper
    a search budget the CTC models were not given.

    Language and task are forced to Bengali transcription rather than left to
    Whisper's own language detection. On short or noisy clips detection can pick
    the wrong language and Whisper will then emit fluent text in it -- a failure
    that scores as a catastrophic WER while looking like a model quality result
    rather than a configuration mistake. (The service's own conversion script
    warns about the same class of bug with the tokenizer.)
    """

    def __init__(self, path: str):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.processor = WhisperProcessor.from_pretrained(path)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(path).to(DEVICE).eval()
        )

    def eval(self):
        return self

    def transcribe(self, paths, batch_size=1, verbose=False):
        from types import SimpleNamespace

        import soundfile as sf

        out = []
        for p in paths:
            audio, _ = sf.read(p, dtype="float32")
            feats = self.processor(
                audio, sampling_rate=asr_core.TARGET_SR, return_tensors="pt"
            ).input_features.to(DEVICE)
            with torch.inference_mode():
                ids = self.model.generate(
                    feats,
                    language="bn",
                    task="transcribe",
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=440,
                )
            text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
            out.append(SimpleNamespace(text=text.strip()))
        return out


def load_model(path: str):
    """Dispatch on the checkpoint: .nemo -> NeMo, whisper in the path -> Whisper
    seq2seq, anything else -> a Hugging Face wav2vec2 CTC directory."""
    if path.endswith(".nemo"):
        import nemo.collections.asr as nemo_asr

        model = nemo_asr.models.ASRModel.restore_from(path, map_location=DEVICE)
    elif "whisper" in path.lower():
        model = WhisperSeq2Seq(path)
    else:
        model = Wav2Vec2CTC(path)
    model.eval()
    return model
