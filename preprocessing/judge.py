import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Dict
from transformers import EfficientNetForImageClassification

from config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    SPEC_MEAN,
    SPEC_STD,
    MODEL_CHECKPOINT,
)

NUM_XCL_CLASSES = 9736


class SpectrogramTransform:
    def __init__(self, device: str = "cuda"):
        self.device = device

        self.spectrogram = T.Spectrogram(
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            power=2.0,
        ).to(device)

        self.mel_scale = T.MelScale(
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
            n_stft=N_FFT // 2 + 1,
        ).to(device)

        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80).to(device)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to(self.device)
        spec = self.spectrogram(waveform)           # (B, freq, time)
        mel_spec = self.mel_scale(spec)             # (B, n_mels, time)
        log_mel = self.amplitude_to_db(mel_spec)    # (B, n_mels, time)

        log_mel = log_mel.unsqueeze(1)

        log_mel = F.interpolate(
            log_mel,
            size=(256, 417),
            mode="bilinear",
            align_corners=False,
        )

        log_mel = (log_mel - SPEC_MEAN) / SPEC_STD
        return log_mel


class BirdClassifier:
    def __init__(
        self,
        checkpoint: str = MODEL_CHECKPOINT,
        device: str = "cuda",
    ):
        self.model = EfficientNetForImageClassification.from_pretrained(
            checkpoint,
            num_labels=NUM_XCL_CLASSES,
            num_channels=1,
            ignore_mismatched_sizes=True,
        ).to(device).eval()
        self.preprocessor = SpectrogramTransform(device=device)

    @torch.inference_mode()
    def evaluate(self, audio_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        spec = self.preprocessor(audio_tensor)  # (B, 1, 256, 417)

        outputs = self.model(spec)
        probs = F.softmax(outputs.logits, dim=-1)

        top2 = torch.topk(probs, k=2, dim=-1)
        return {
            "top1_prob": top2.values[:, 0],
            "top2_prob": top2.values[:, 1],
            "top1_class": top2.indices[:, 0],
        }
