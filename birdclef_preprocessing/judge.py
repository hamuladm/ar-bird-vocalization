import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Dict
from transformers import ConvNextForImageClassification

from config import (
    DEVICE,
    EVAL_SAMPLE_RATE,
    EVAL_CHUNK_SEC,
    EVAL_N_FFT,
    EVAL_HOP_LENGTH,
    EVAL_N_MELS,
    EVAL_SPEC_MEAN,
    EVAL_SPEC_STD,
    EVAL_SPEC_SIZE,
)


class SpectrogramTransform:
    def __init__(self, device: str = DEVICE):
        self.device = device

        self.spectrogram = T.Spectrogram(
            n_fft=EVAL_N_FFT,
            hop_length=EVAL_HOP_LENGTH,
            power=2.0,
        ).to(device)

        self.mel_scale = T.MelScale(
            n_mels=EVAL_N_MELS,
            sample_rate=EVAL_SAMPLE_RATE,
            n_stft=EVAL_N_FFT // 2 + 1,
        ).to(device)

        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80).to(device)

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


class BirdClassifier:
    def __init__(
        self,
        checkpoint: str = "DBD-research-group/ConvNeXT-Base-BirdSet-XCL",
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
        self._chunk_len = int(EVAL_CHUNK_SEC * EVAL_SAMPLE_RATE)

    def _chunk_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        n = waveform.shape[-1]
        if n <= self._chunk_len:
            return waveform.unsqueeze(0)
        chunks = []
        for start in range(0, n, self._chunk_len):
            seg = waveform[start : start + self._chunk_len]
            if seg.shape[-1] < self._chunk_len:
                seg = F.pad(seg, (0, self._chunk_len - seg.shape[-1]))
            chunks.append(seg)
        return torch.stack(chunks)

    @torch.inference_mode()
    def evaluate(
        self, audio_tensor: torch.Tensor, top_k: int = 2
    ) -> Dict[str, torch.Tensor]:
        batch_probs = []
        for i in range(audio_tensor.shape[0]):
            chunks = self._chunk_waveform(audio_tensor[i]).to(self.device)
            spec = self.preprocessor(chunks)
            outputs = self.model(spec)
            probs = F.softmax(outputs.logits, dim=-1)
            batch_probs.append(probs.mean(dim=0))
        probs = torch.stack(batch_probs)

        topk = torch.topk(probs, k=max(top_k, 2), dim=-1)
        entropy = -(probs * probs.clamp(min=1e-9).log()).sum(dim=-1)

        return {
            "topk_probs": topk.values,
            "topk_classes": topk.indices,
            "entropy": entropy,
            "top1_prob": topk.values[:, 0],
            "top2_prob": topk.values[:, 1],
            "top1_class": topk.indices[:, 0],
        }
