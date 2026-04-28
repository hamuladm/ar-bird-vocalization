import logging
import math
import random
from collections import OrderedDict, defaultdict

import librosa
import numpy as np
import torch

from config import DEVICE
from ecogen.vqvae import VQVAE
from utils.audio import load_segment

logger = logging.getLogger(__name__)

_ECOGEN_SR = 16384
_ECOGEN_N_FFT = 1024
_ECOGEN_HOP_LEN = 512
_ECOGEN_CHUNK_SEC = 4
_ECOGEN_CHUNK_LEN = _ECOGEN_SR * _ECOGEN_CHUNK_SEC
_ECOGEN_TARGET_SEC = 10
_ECOGEN_TARGET_LEN = _ECOGEN_SR * _ECOGEN_TARGET_SEC


def _load_vqvae(model_path, device):
    model = VQVAE(in_channel=1)
    weights = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model" in weights:
        weights = weights["model"]
    if "state_dict" in weights:
        weights = weights["state_dict"]
    cleaned = OrderedDict()
    for key, value in weights.items():
        cleaned[key.replace("net.", "", 1) if key.startswith("net.") else key] = value
    model.load_state_dict(cleaned)
    return model.eval().to(device)


class EcogenGenerator:
    sample_rate = _ECOGEN_SR

    def __init__(
        self,
        model_path,
        segments,
        device=DEVICE,
        augmentation="noise",
        ratio=0.5,
        latent_stats_path=None,
    ):
        self.device = torch.device(device)
        self.augmentation = augmentation
        self.ratio = ratio

        self.model = _load_vqvae(model_path, self.device)

        self._class_segments: dict[str, list[dict]] = defaultdict(list)
        for seg in segments:
            self._class_segments[seg["ebird_code"]].append(seg)

        self._latent_stats: dict[str, dict] | None = None
        if latent_stats_path is not None:
            raw = torch.load(
                latent_stats_path, map_location=self.device, weights_only=False
            )
            self._latent_stats = {
                code: {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in d.items()
                }
                for code, d in raw.items()
            }
            logger.info(
                "loaded latent stats for %d classes from %s",
                len(self._latent_stats),
                latent_stats_path,
            )

        if self._latent_stats is not None:
            codes = sorted(self._latent_stats.keys())
        else:
            codes = sorted(self._class_segments.keys())
        self.ebird_to_id = {c: i for i, c in enumerate(codes)}
        self.id_to_ebird = {i: c for c, i in self.ebird_to_id.items()}

        for code, segs in self._class_segments.items():
            logger.info("ecogen source: %s — %d segments", code, len(segs))

    @staticmethod
    def _chunk_audio(audio_np):
        n_chunks = max(1, math.ceil(len(audio_np) / _ECOGEN_CHUNK_LEN))
        chunks = []
        for i in range(n_chunks):
            start = i * _ECOGEN_CHUNK_LEN
            chunk = audio_np[start : start + _ECOGEN_CHUNK_LEN]
            chunk = librosa.util.fix_length(data=chunk, size=_ECOGEN_CHUNK_LEN)
            chunks.append(chunk)
        return chunks

    def _audio_to_spectrogram(self, audio_np):
        audio_np = librosa.util.fix_length(data=audio_np, size=_ECOGEN_CHUNK_LEN)
        mel = librosa.feature.melspectrogram(
            y=audio_np,
            n_fft=_ECOGEN_N_FFT,
            hop_length=_ECOGEN_HOP_LEN,
        )
        mel_db = librosa.power_to_db(mel)
        if mel_db.shape[0] % 2 != 0:
            mel_db = mel_db[1:, :]
        if mel_db.shape[1] % 2 != 0:
            mel_db = mel_db[:, 1:]
        return torch.from_numpy(mel_db[np.newaxis, np.newaxis]).float()

    def _spectrogram_to_audio(self, spec_db_np):
        power = librosa.db_to_power(spec_db_np)
        return librosa.feature.inverse.mel_to_audio(
            power,
            sr=_ECOGEN_SR,
            n_fft=_ECOGEN_N_FFT,
            hop_length=_ECOGEN_HOP_LEN,
        ).astype(np.float32)

    def _load_audio(self, segment):
        segment["filepath"] = segment["filepath"].replace(
            "/workspace/.hf_home/", "/home/dkham/.cache/huggingface/"
        )
        return load_segment(
            segment["filepath"],
            segment["start"],
            segment["end"],
            target_sr=_ECOGEN_SR,
        )

    @torch.no_grad()
    def _encode(self, spec_tensor):
        spec_tensor = spec_tensor.to(self.device)
        q_t, q_b, _diff, _id_t, _id_b = self.model.encode(spec_tensor)
        return q_t, q_b

    @torch.no_grad()
    def _decode(self, q_t, q_b):
        recon = self.model.decode(q_t, q_b).detach()
        return recon.cpu().numpy()[0, 0]

    @staticmethod
    def _fit_to_target(audio_np):
        if len(audio_np) >= _ECOGEN_TARGET_LEN:
            return audio_np[:_ECOGEN_TARGET_LEN]
        return librosa.util.fix_length(data=audio_np, size=_ECOGEN_TARGET_LEN)

    def _augment_chunk_noise(self, chunk_np, ratio):
        spec = self._audio_to_spectrogram(chunk_np)
        q_t, q_b = self._encode(spec)
        q_t = ratio * torch.randn_like(q_t) + q_t
        q_b = ratio * torch.randn_like(q_b) + q_b
        return self._spectrogram_to_audio(self._decode(q_t, q_b))

    def _augment_chunk_interp(self, chunk_a_np, chunk_b_np, alpha, ratio):
        spec_a = self._audio_to_spectrogram(chunk_a_np)
        spec_b = self._audio_to_spectrogram(chunk_b_np)
        q_t_a, q_b_a = self._encode(spec_a)
        q_t_b, q_b_b = self._encode(spec_b)
        q_t = (q_t_b - q_t_a) * alpha + q_t_a
        q_b = (q_b_b - q_b_a) * alpha + q_b_a
        if ratio > 0:
            q_t = ratio * torch.randn_like(q_t) + q_t
            q_b = ratio * torch.randn_like(q_b) + q_b
        return self._spectrogram_to_audio(self._decode(q_t, q_b))

    def _generate_latent_chunk(self, stats, ratio):
        q_t = (
            torch.randn_like(stats["q_t_mean"]) * stats["q_t_std"] * ratio
            + stats["q_t_mean"]
        )
        q_b = (
            torch.randn_like(stats["q_b_mean"]) * stats["q_b_std"] * ratio
            + stats["q_b_mean"]
        )
        return self._spectrogram_to_audio(
            self._decode(q_t.unsqueeze(0), q_b.unsqueeze(0))
        )

    def generate(self, class_id, ratio=None, augmentation=None, **_kwargs):
        augmentation = augmentation or self.augmentation
        ratio = ratio if ratio is not None else self.ratio
        ebird_code = self.id_to_ebird[class_id]

        if augmentation == "latent_sampling":
            if self._latent_stats is None:
                raise RuntimeError(
                    "latent_sampling requires --latent-stats; "
                    "run scripts/extract_latent_stats.py first"
                )
            s = self._latent_stats[ebird_code]
            segs = self._class_segments.get(ebird_code, [])
            if segs:
                seg = random.choice(segs)
                audio = self._load_audio(seg)
                n_chunks = max(1, math.ceil(len(audio) / _ECOGEN_CHUNK_LEN))
            else:
                n_chunks = 3
            pieces = [self._generate_latent_chunk(s, ratio) for _ in range(n_chunks)]
            return self._fit_to_target(np.concatenate(pieces))

        segs = self._class_segments[ebird_code]

        if augmentation == "noise":
            seg = random.choice(segs)
            audio = self._load_audio(seg)
            chunks = self._chunk_audio(audio)
            pieces = [self._augment_chunk_noise(c, ratio) for c in chunks]
            return self._fit_to_target(np.concatenate(pieces))

        if augmentation == "interpolation":
            if len(segs) < 2:
                seg = segs[0]
                audio = self._load_audio(seg)
                chunks = self._chunk_audio(audio)
                pieces = [self._augment_chunk_noise(c, ratio) for c in chunks]
                return self._fit_to_target(np.concatenate(pieces))

            seg_a, seg_b = random.sample(segs, 2)
            audio_a = self._load_audio(seg_a)
            audio_b = self._load_audio(seg_b)
            chunks_a = self._chunk_audio(audio_a)
            chunks_b = self._chunk_audio(audio_b)
            n = max(len(chunks_a), len(chunks_b))
            alpha = random.random()
            pieces = []
            for i in range(n):
                ca = chunks_a[i % len(chunks_a)]
                cb = chunks_b[i % len(chunks_b)]
                pieces.append(self._augment_chunk_interp(ca, cb, alpha, ratio=0))
            return self._fit_to_target(np.concatenate(pieces))

        raise ValueError(f"Unknown augmentation: {augmentation!r}")

    def generate_batch(self, class_id, k, **kwargs):
        return [self.generate(class_id, **kwargs) for _ in range(k)]
