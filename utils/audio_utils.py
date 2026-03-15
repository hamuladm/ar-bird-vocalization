import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf
import librosa
from typing import Dict, List, Tuple
from collections import defaultdict

from config import SAMPLE_RATE, MAX_LENGTH


def load_segment(
    filepath: str,
    start: float,
    end: float,
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    info = sf.info(filepath)
    sr = info.samplerate

    start_frame = int(start * sr)
    end_frame = int(end * sr)
    audio, sr = sf.read(filepath, start=start_frame, stop=end_frame)

    if audio.ndim > 1:
        audio = librosa.to_mono(audio.T)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio


def load_audio_fixed_length(
    filepath: str,
    start: float,
    end: float,
    target_sr: int = SAMPLE_RATE,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    audio = load_segment(filepath, start, end, target_sr=target_sr)

    target_samples = target_sr * max_length
    if len(audio) < target_samples:
        pad_total = target_samples - len(audio)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode="constant")
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    return audio


def load_and_pad_batch(
    tasks: List[Tuple[str, float, float]],
    target_sr: int = SAMPLE_RATE,
    max_length: int = MAX_LENGTH,
) -> torch.Tensor:
    arrays = [load_audio_fixed_length(fp, s, e, target_sr=target_sr, max_length=max_length) for fp, s, e in tasks]
    return torch.stack([torch.from_numpy(a).float() for a in arrays])


class SegmentDataset(Dataset):
    def __init__(
        self,
        segments_info: List[Dict],
        target_sr: int = SAMPLE_RATE,
        max_length: int = MAX_LENGTH,
    ):
        self.segments_info = segments_info
        self.target_sr = target_sr
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.segments_info)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seg = self.segments_info[idx]
        audio = load_audio_fixed_length(
            seg["filepath"], seg["start"], seg["end"],
            target_sr=self.target_sr, max_length=self.max_length,
        )
        return torch.from_numpy(audio).float(), idx


def get_all_segments(item: dict) -> List[Tuple[float, float]]:
    detected_events = item.get("detected_events")
    if detected_events:
        return [(ev[0], ev[1]) for ev in detected_events]
    start = item.get("start_time", 0) or 0
    end = item.get("end_time") or MAX_LENGTH
    return [(start, end)]


def pack_segments(
    segments: List[Dict],
    target_sr: int = SAMPLE_RATE,
    target_length: float = MAX_LENGTH,
    seed: int = 42,
) -> List[Dict]:
    target_samples = int(target_sr * target_length)
    rng = np.random.default_rng(seed)

    by_class: Dict[str, List[Dict]] = defaultdict(list)
    for seg in segments:
        by_class[seg["ebird_code"]].append(seg)

    packed = []
    for ebird_code, class_segs in by_class.items():
        rng.shuffle(class_segs)

        current_parts: List[Tuple] = []
        current_len = 0

        for seg in class_segs:
            est_samples = int((seg["end"] - seg["start"]) * target_sr)
            skip = 0
            remaining = est_samples

            while remaining > 0:
                space = target_samples - current_len
                take = min(remaining, space)
                current_parts.append((seg["filepath"], seg["start"], seg["end"], skip, take))
                current_len += take
                skip += take
                remaining -= take

                if current_len >= target_samples:
                    packed.append({"ebird_code": ebird_code, "parts": current_parts})
                    current_parts = []
                    current_len = 0

        if current_parts:
            packed.append({"ebird_code": ebird_code, "parts": current_parts})

    rng.shuffle(packed)
    return packed


class PackedSegmentDataset(Dataset):
    """Map-style dataset for packed audio plans.

    Each item loads and stitches together one packing plan via
    load_packed_sample, returning (audio_tensor, label).
    """

    def __init__(
        self,
        packing_plans: List[Dict],
        ebird_to_id: Dict[str, int],
        target_sr: int = SAMPLE_RATE,
        target_length: float = MAX_LENGTH,
    ):
        self.plans = packing_plans
        self.ebird_to_id = ebird_to_id
        self.target_sr = target_sr
        self.target_length = target_length

    def __len__(self) -> int:
        return len(self.plans)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        plan = self.plans[idx]
        audio = load_packed_sample(
            plan, target_sr=self.target_sr, target_length=self.target_length,
        )
        label = self.ebird_to_id[plan["ebird_code"]]
        return torch.from_numpy(audio).float(), label


def load_packed_sample(
    plan: Dict,
    target_sr: int = SAMPLE_RATE,
    target_length: float = MAX_LENGTH,
) -> np.ndarray:
    target_samples = int(target_sr * target_length)
    buf = np.zeros(target_samples, dtype=np.float32)
    pos = 0

    for filepath, start, end, skip, take in plan["parts"]:
        audio = load_segment(filepath, start, end, target_sr=target_sr)
        available = len(audio) - skip
        actual_take = min(take, max(available, 0), target_samples - pos)
        if actual_take > 0:
            buf[pos : pos + actual_take] = audio[skip : skip + actual_take]
            pos += actual_take
        if pos >= target_samples:
            break

    return buf
