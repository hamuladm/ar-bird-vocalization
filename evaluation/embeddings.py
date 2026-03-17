from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from transformers import ConvNextForImageClassification


from preprocessing.judge import SpectrogramTransform
from utils.audio_utils import load_segment
from utils.logging_utils import setup_logger
from config import DEVICE, SAMPLE_RATE, MODEL_CHECKPOINT

logger = setup_logger("eval_embeddings")

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


class EvalEmbedder:
    def __init__(
        self,
        checkpoint: str = MODEL_CHECKPOINT,
        device: str = DEVICE,
    ):

        self.device = device
        self.model = (
            ConvNextForImageClassification.from_pretrained(
                checkpoint,
                ignore_mismatched_sizes=True,
            )
            .to(device)
            .eval()
        )
        self.preprocessor = SpectrogramTransform(device=device)

    @torch.inference_mode()
    def extract(self, waveforms: torch.Tensor) -> dict[str, torch.Tensor]:
        spec = self.preprocessor(waveforms)
        backbone_out = self.model.convnext(spec)
        features = backbone_out.pooler_output
        logits = self.model.classifier(features)
        probs = F.softmax(logits, dim=-1)
        return {"probs": probs, "features": features}


def _collect_audio_paths(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    paths = sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    return paths


def _load_and_resample(path: Path) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.squeeze(0)


def _collate_waveforms(waveforms: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(w.shape[0] for w in waveforms)
    batch = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        batch[i, : w.shape[0]] = w
    return batch


def extract_embeddings_from_directory(
    directory: str | Path,
    embedder: EvalEmbedder,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    paths = _collect_audio_paths(directory)

    all_probs = []
    all_features = []

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        waveforms = [_load_and_resample(p) for p in batch_paths]
        batch = _collate_waveforms(waveforms).to(embedder.device)

        result = embedder.extract(batch)
        all_probs.append(result["probs"].cpu().numpy())
        all_features.append(result["features"].cpu().numpy())

        processed = min(start + batch_size, len(paths))
        logger.info(f"  Processed {processed}/{len(paths)} files")

    return {
        "probs": np.concatenate(all_probs, axis=0),
        "features": np.concatenate(all_features, axis=0),
    }


def _extract_batched(
    waveform_tensors: list[torch.Tensor],
    embedder: EvalEmbedder,
    batch_size: int,
    label: str,
) -> dict[str, np.ndarray]:
    all_probs = []
    all_features = []
    total = len(waveform_tensors)

    for start in range(0, total, batch_size):
        batch_wavs = waveform_tensors[start : start + batch_size]
        batch = _collate_waveforms(batch_wavs).to(embedder.device)

        result = embedder.extract(batch)
        all_probs.append(result["probs"].cpu().numpy())
        all_features.append(result["features"].cpu().numpy())

        processed = min(start + batch_size, total)
        logger.info(f"  [{label}] Processed {processed}/{total}")

    return {
        "probs": np.concatenate(all_probs, axis=0),
        "features": np.concatenate(all_features, axis=0),
    }


def extract_embeddings_from_segments(
    segments: list[dict],
    embedder: EvalEmbedder,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    waveforms = []
    for seg in segments:
        # TODO: Resolve this in a more sophisticated way
        seg["filepath"] = seg["filepath"].replace(
            "/workspace/.hf_home", "/home/dkham/.cache/huggingface"
        )
        audio_np = load_segment(seg["filepath"], seg["start"], seg["end"])
        waveforms.append(torch.from_numpy(audio_np).float())

    return _extract_batched(waveforms, embedder, batch_size, label="segments")


def extract_embeddings_from_arrays(
    arrays: Sequence[np.ndarray],
    embedder: EvalEmbedder,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    waveforms = [torch.from_numpy(a).float() for a in arrays]
    return _extract_batched(waveforms, embedder, batch_size, label="generated")
