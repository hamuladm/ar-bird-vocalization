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


def bin_pack_segments(
    segments: List[Dict],
    target_length: float = MAX_LENGTH,
    gap_sec: float = 0.15,
    seed: int = 42,
) -> List[Dict]:
    """First-Fit Decreasing bin packing that keeps segments whole.

    Segments longer than *target_length* are random-cropped (crop window
    stored as ``crop_start`` / ``crop_end`` on the segment dict so the
    loader can use them).  Remaining segments are packed into bins of
    *target_length* seconds with *gap_sec* silence gaps between them.

    Returns a list of ``{"ebird_code": str, "segments": [seg_dict, ...]}``.
    """
    rng = np.random.default_rng(seed)

    by_class: Dict[str, List[Dict]] = defaultdict(list)
    for seg in segments:
        by_class[seg["ebird_code"]].append(seg)

    packed: List[Dict] = []

    for ebird_code, class_segs in by_class.items():
        rng.shuffle(class_segs)

        oversized: List[Dict] = []
        normal: List[Dict] = []

        for seg in class_segs:
            dur = seg["end"] - seg["start"]
            if dur > target_length:
                max_offset = dur - target_length
                offset = rng.uniform(0, max_offset)
                cropped = seg.copy()
                cropped["crop_start"] = seg["start"] + offset
                cropped["crop_end"] = seg["start"] + offset + target_length
                oversized.append(cropped)
            else:
                normal.append(seg)

        for seg in oversized:
            packed.append({"ebird_code": ebird_code, "segments": [seg]})

        normal.sort(key=lambda s: s["end"] - s["start"], reverse=True)

        # bins: list of (remaining_capacity, [segments])
        bins: List[Tuple[float, List[Dict]]] = []

        for seg in normal:
            dur = seg["end"] - seg["start"]
            placed = False
            for i, (remaining, bin_segs) in enumerate(bins):
                needed = dur + (gap_sec if bin_segs else 0.0)
                if needed <= remaining + 1e-6:
                    bins[i] = (remaining - needed, bin_segs + [seg])
                    placed = True
                    break
            if not placed:
                bins.append((target_length - dur, [seg]))

        for _, bin_segs in bins:
            packed.append({"ebird_code": ebird_code, "segments": bin_segs})

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
        gap_sec: float = 0.15,
        fade_sec: float = 0.02,
    ):
        self.plans = packing_plans
        self.ebird_to_id = ebird_to_id
        self.target_sr = target_sr
        self.target_length = target_length
        self.gap_sec = gap_sec
        self.fade_sec = fade_sec

    def __len__(self) -> int:
        return len(self.plans)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        plan = self.plans[idx]
        audio = load_packed_sample(
            plan,
            target_sr=self.target_sr,
            target_length=self.target_length,
            gap_sec=self.gap_sec,
            fade_sec=self.fade_sec,
        )
        label = self.ebird_to_id[plan["ebird_code"]]
        return torch.from_numpy(audio).float(), label


def load_packed_sample(
    plan: Dict,
    target_sr: int = SAMPLE_RATE,
    target_length: float = MAX_LENGTH,
    gap_sec: float = 0.15,
    fade_sec: float = 0.02,
) -> np.ndarray:
    """Load and stitch segments from a packing plan with fades and gaps.

    Applies linear fade-in / fade-out envelopes at each segment boundary
    and inserts *gap_sec* of silence between consecutive segments.
    Segments with ``crop_start`` / ``crop_end`` keys (from random-crop of
    oversized segments) are loaded using those bounds instead of the
    original ``start`` / ``end``.
    """
    target_samples = int(target_sr * target_length)
    gap_samples = int(target_sr * gap_sec)
    fade_samples = int(target_sr * fade_sec)
    buf = np.zeros(target_samples, dtype=np.float32)
    pos = 0

    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

    segments = plan["segments"]
    for i, seg in enumerate(segments):
        start = seg.get("crop_start", seg["start"])
        end = seg.get("crop_end", seg["end"])
        audio = load_segment(seg["filepath"], start, end, target_sr=target_sr)

        if len(audio) >= 2 * fade_samples:
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
        elif len(audio) >= fade_samples:
            audio[:fade_samples] *= fade_in

        take = min(len(audio), target_samples - pos)
        if take <= 0:
            break
        buf[pos: pos + take] = audio[:take]
        pos += take

        if i < len(segments) - 1:
            pos = min(pos + gap_samples, target_samples)

    return buf
