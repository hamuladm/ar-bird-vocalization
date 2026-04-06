import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from transformers import ConvNextForImageClassification

from config import (
    DEVICE,
    EVAL_SAMPLE_RATE,
    EVAL_MODEL_CHECKPOINT,
    EVAL_N_FFT,
    EVAL_HOP_LENGTH,
    EVAL_N_MELS,
    EVAL_SPEC_MEAN,
    EVAL_SPEC_STD,
    EVAL_BATCH_SIZE,
    EVAL_SPEC_SIZE
)
from utils.audio import load_segment

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


class SpectrogramTransform:
    def __init__(self, device: str = DEVICE):
        self.device = device

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=EVAL_N_FFT,
            hop_length=EVAL_HOP_LENGTH,
            power=2.0,
        ).to(device)

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=EVAL_N_MELS,
            sample_rate=EVAL_SAMPLE_RATE,
            n_stft=EVAL_N_FFT // 2 + 1,
        ).to(device)

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(device)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to(self.device)
        spec = self.spectrogram(waveform)
        mel_spec = self.mel_scale(spec)
        log_mel = self.amplitude_to_db(mel_spec)

        log_mel = log_mel.unsqueeze(1)

        log_mel = F.interpolate(
            log_mel,
            size=EVAL_SPEC_SIZE,
            mode="bilinear",
            align_corners=False,
        )

        log_mel = (log_mel - EVAL_SPEC_MEAN) / EVAL_SPEC_STD
        return log_mel


class EvalEmbedder:
    def __init__(self, checkpoint=EVAL_MODEL_CHECKPOINT, device=DEVICE):
        self.device = device
        self.model = (
            ConvNextForImageClassification.from_pretrained(
                checkpoint, ignore_mismatched_sizes=True
            )
            .to(device)
            .eval()
        )
        self.preprocessor = SpectrogramTransform(device=device)

    @torch.inference_mode()
    def extract(self, waveforms):
        spec = self.preprocessor(waveforms)
        backbone_out = self.model.convnext(spec)
        features = backbone_out.pooler_output
        logits = self.model.classifier(features)
        probs = F.softmax(logits, dim=-1)
        return {"probs": probs, "features": features}


def _collect_audio_paths(directory):
    return sorted(
        p for p in Path(directory).rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    )


def _load_and_resample(path):
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != EVAL_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, EVAL_SAMPLE_RATE)
    return waveform.squeeze(0)


def _collate_waveforms(waveforms):
    max_len = max(w.shape[0] for w in waveforms)
    batch = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        batch[i, : w.shape[0]] = w
    return batch


def _extract_batched(waveform_tensors, embedder, batch_size):
    all_probs = []
    all_features = []

    for start in range(0, len(waveform_tensors), batch_size):
        batch_wavs = waveform_tensors[start : start + batch_size]
        batch = _collate_waveforms(batch_wavs).to(embedder.device)
        result = embedder.extract(batch)
        all_probs.append(result["probs"].cpu().numpy())
        all_features.append(result["features"].cpu().numpy())

    return {
        "probs": np.concatenate(all_probs, axis=0),
        "features": np.concatenate(all_features, axis=0),
    }


def extract_embeddings_from_directory(directory, embedder, batch_size=EVAL_BATCH_SIZE):
    paths = _collect_audio_paths(directory)
    waveforms = [_load_and_resample(p) for p in paths]
    return _extract_batched(waveforms, embedder, batch_size)


def extract_embeddings_from_segments(segments, embedder, batch_size=EVAL_BATCH_SIZE):
    waveforms = []
    for seg in segments:
        seg["filepath"] = seg["filepath"].replace("/workspace/.hf_home/", "/home/dkham/.cache/huggingface/")
        audio_np = load_segment(
            seg["filepath"], seg["start"], seg["end"], EVAL_SAMPLE_RATE
        )
        waveforms.append(torch.from_numpy(audio_np).float())
    return _extract_batched(waveforms, embedder, batch_size)


def extract_embeddings_from_arrays(arrays, embedder, batch_size=EVAL_BATCH_SIZE):
    waveforms = [torch.from_numpy(a).float() for a in arrays]
    return _extract_batched(waveforms, embedder, batch_size)


def extract_embeddings_from_shards(directory, embedder, batch_size=EVAL_BATCH_SIZE):
    shard_paths = sorted(Path(directory).glob("*.npz"))
    waveforms = []
    for path in shard_paths:
        data = np.load(path)
        samples = data["samples"]
        lengths = data["lengths"]
        sr = int(data["sample_rate"])
        for i in range(len(lengths)):
            w = torch.from_numpy(samples[i, : lengths[i]]).float()
            if sr != EVAL_SAMPLE_RATE:
                w = torchaudio.functional.resample(w.unsqueeze(0), sr, EVAL_SAMPLE_RATE).squeeze(0)
            waveforms.append(w)
    return _extract_batched(waveforms, embedder, batch_size)
