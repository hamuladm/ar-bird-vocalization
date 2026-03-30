import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Dict
from transformers import ConvNextForImageClassification

from config import (
    DEVICE,
    EVAL_SAMPLE_RATE,
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
        self.model = ConvNextForImageClassification.from_pretrained(
            checkpoint,
            ignore_mismatched_sizes=True,
        ).to(device).eval()
        self.preprocessor = SpectrogramTransform(device=device)

    @torch.inference_mode()
    def evaluate(self, audio_tensor: torch.Tensor, top_k: int = 2) -> Dict[str, torch.Tensor]:
        spec = self.preprocessor(audio_tensor)  # (B, 1, 128, 334)

        outputs = self.model(spec)
        probs = F.softmax(outputs.logits, dim=-1)

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
