import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils.audio import load_segment, apply_fade
from config import (
    DEVICE,
    SAMPLE_RATE,
    CHUNK_LENGTH,
    FADE_SEC,
    SNAC_MODEL,
    CODEBOOK_SIZE,
    SNAC_INF_BATCH_SIZE,
    SNAC_INF_NUM_WORKERS,
    AG_PRETRAINED,
    AG_SAMPLE_RATE,
    AG_TARGET_LENGTH,
    AG_ENCODEC_BATCH_SIZE,
    AG_ENCODEC_NUM_WORKERS,
    SEGMENT_DIR,
    TOKEN_DIR,
    AG_TOKEN_DIR,
)


class SegmentDataset(Dataset):
    def __init__(self, segments, target_sr, target_length, fade_sec):
        self.segments = segments
        self.target_sr = target_sr
        self.target_length = target_length
        self.fade_samples = int(target_sr * fade_sec)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        audio = load_segment(seg["filepath"], seg["start"], seg["end"], self.target_sr)

        target_samples = int(self.target_sr * self.target_length)
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        elif len(audio) > target_samples:
            audio = audio[:target_samples]

        audio = apply_fade(audio, self.fade_samples)
        return torch.from_numpy(audio).float(), idx


def flatten_codes(codes, codebook_size=CODEBOOK_SIZE):
    c0, c1, c2, c3 = [c.cpu().numpy() for c in codes]
    batch_size = c0.shape[0]
    coarse_len = c0.shape[1]
    tokens_per_step = 15

    flat = np.zeros((batch_size, coarse_len * tokens_per_step), dtype=np.int64)
    for t in range(coarse_len):
        base = t * tokens_per_step
        flat[:, base] = c0[:, t] + 0 * codebook_size
        flat[:, base + 1] = c1[:, 2 * t] + 1 * codebook_size
        flat[:, base + 2] = c1[:, 2 * t + 1] + 1 * codebook_size
        flat[:, base + 3] = c2[:, 4 * t] + 2 * codebook_size
        flat[:, base + 4] = c2[:, 4 * t + 1] + 2 * codebook_size
        flat[:, base + 5] = c2[:, 4 * t + 2] + 2 * codebook_size
        flat[:, base + 6] = c2[:, 4 * t + 3] + 2 * codebook_size
        flat[:, base + 7] = c3[:, 8 * t] + 3 * codebook_size
        flat[:, base + 8] = c3[:, 8 * t + 1] + 3 * codebook_size
        flat[:, base + 9] = c3[:, 8 * t + 2] + 3 * codebook_size
        flat[:, base + 10] = c3[:, 8 * t + 3] + 3 * codebook_size
        flat[:, base + 11] = c3[:, 8 * t + 4] + 3 * codebook_size
        flat[:, base + 12] = c3[:, 8 * t + 5] + 3 * codebook_size
        flat[:, base + 13] = c3[:, 8 * t + 6] + 3 * codebook_size
        flat[:, base + 14] = c3[:, 8 * t + 7] + 3 * codebook_size

    return flat


def unflatten_codes(flat, codebook_size=CODEBOOK_SIZE):
    if flat.ndim == 2:
        flat = flat[0]

    tokens_per_step = 15
    coarse_len = len(flat) // tokens_per_step
    grouped = flat[: coarse_len * tokens_per_step].reshape(coarse_len, tokens_per_step)

    c0 = grouped[:, 0] - 0 * codebook_size
    c1 = grouped[:, 1:3] - 1 * codebook_size
    c2 = grouped[:, 3:7] - 2 * codebook_size
    c3 = grouped[:, 7:15] - 3 * codebook_size

    return [c0, c1.reshape(-1), c2.reshape(-1), c3.reshape(-1)]


def _build_label_array(segments, ebird_to_id):
    return np.array([ebird_to_id[s["ebird_code"]] for s in segments])


def _load_ebird_to_id(segment_dir):
    with open(Path(segment_dir) / "ebird_to_id.json") as f:
        return json.load(f)


def encode_snac(
    segment_json,
    output_dir,
    segment_dir=SEGMENT_DIR,
    device=DEVICE,
    batch_size=SNAC_INF_BATCH_SIZE,
    num_workers=SNAC_INF_NUM_WORKERS,
):
    from snac import SNAC

    with open(segment_json) as f:
        segments = json.load(f)

    ebird_to_id = _load_ebird_to_id(segment_dir)
    labels = _build_label_array(segments, ebird_to_id)

    dataset = SegmentDataset(segments, SAMPLE_RATE, CHUNK_LENGTH, FADE_SEC)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    model = SNAC.from_pretrained(SNAC_MODEL).eval().to(device)

    all_codes = []
    for audio_batch, _ in tqdm(loader, desc="SNAC encode"):
        waveforms = audio_batch.unsqueeze(1).to(device)
        with torch.inference_mode():
            codes = model.encode(waveforms)
        all_codes.append(flatten_codes(codes))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "tokens.npz", codes=np.concatenate(all_codes), labels=labels)
    print(f"Saved {len(labels)} SNAC samples -> {output_dir / 'tokens.npz'}")


def encode_encodec(
    segment_json,
    output_dir,
    segment_dir=SEGMENT_DIR,
    device=DEVICE,
    batch_size=AG_ENCODEC_BATCH_SIZE,
    num_workers=AG_ENCODEC_NUM_WORKERS,
):
    from audiocraft.models import AudioGen

    with open(segment_json) as f:
        segments = json.load(f)

    ebird_to_id = _load_ebird_to_id(segment_dir)
    labels = _build_label_array(segments, ebird_to_id)

    dataset = SegmentDataset(segments, AG_SAMPLE_RATE, AG_TARGET_LENGTH, FADE_SEC)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    audiogen = AudioGen.get_pretrained(AG_PRETRAINED, device=device)
    compression_model = audiogen.compression_model

    all_codes = []
    for audio_batch, _ in tqdm(loader, desc="EnCodec encode"):
        waveforms = audio_batch.unsqueeze(1).to(device)
        with torch.inference_mode():
            codes, scale = compression_model.encode(waveforms)
        all_codes.append(codes.cpu().numpy())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    codes_cat = np.concatenate(all_codes)
    np.savez(output_dir / "tokens.npz", codes=codes_cat, labels=labels)
    print(
        f"Saved {len(labels)} EnCodec samples (shape {codes_cat.shape}) -> {output_dir / 'tokens.npz'}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", choices=["snac", "encodec"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--segment-dir", type=str, default=str(SEGMENT_DIR))
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    segment_json = Path(args.segment_dir) / f"{args.split}_segments.json"

    if args.codec == "snac":
        out = args.output_dir or str(TOKEN_DIR / args.split)
        encode_snac(str(segment_json), out, segment_dir=args.segment_dir)
    else:
        out = args.output_dir or str(Path(AG_TOKEN_DIR) / args.split)
        encode_encodec(str(segment_json), out, segment_dir=args.segment_dir)
