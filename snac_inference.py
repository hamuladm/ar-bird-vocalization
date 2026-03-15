import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from snac import SNAC

from utils.logging_utils import setup_logger
from utils.audio_utils import pack_segments
from utils.mapping_utils import load_segments_and_mapping
from utils.audio_utils import PackedSegmentDataset
from config import (
    DEVICE,
    FILTERED_DIR,
    MAX_LENGTH,
    SEED,
    SNAC_MODEL,
    CODEBOOK_SIZE,
    SAMPLE_RATE,
    SNAC_INF_BATCH_SIZE,
    SNAC_INF_NUM_WORKERS,
    TOKEN_DIR,
)

logger = setup_logger("snac_inference")


def flatten_codes(codes: list[torch.Tensor]) -> np.ndarray:
    c0, c1, c2, c3 = [c.cpu().numpy() for c in codes]
    batch_size = c0.shape[0]
    coarse_len = c0.shape[1]

    tokens_per_step = 15 # 1 : 2 : 4 : 8
    flat = np.zeros((batch_size, coarse_len * tokens_per_step), dtype=np.int64)

    for t in range(coarse_len):
        base = t * tokens_per_step
        flat[:, base]      = c0[:, t]     + 0 * CODEBOOK_SIZE
        flat[:, base + 1]  = c1[:, 2*t]   + 1 * CODEBOOK_SIZE
        flat[:, base + 2]  = c1[:, 2*t+1] + 1 * CODEBOOK_SIZE
        flat[:, base + 3]  = c2[:, 4*t]   + 2 * CODEBOOK_SIZE
        flat[:, base + 4]  = c2[:, 4*t+1] + 2 * CODEBOOK_SIZE
        flat[:, base + 5]  = c2[:, 4*t+2] + 2 * CODEBOOK_SIZE
        flat[:, base + 6]  = c2[:, 4*t+3] + 2 * CODEBOOK_SIZE
        flat[:, base + 7]  = c3[:, 8*t]   + 3 * CODEBOOK_SIZE
        flat[:, base + 8]  = c3[:, 8*t+1] + 3 * CODEBOOK_SIZE
        flat[:, base + 9]  = c3[:, 8*t+2] + 3 * CODEBOOK_SIZE
        flat[:, base + 10] = c3[:, 8*t+3] + 3 * CODEBOOK_SIZE
        flat[:, base + 11] = c3[:, 8*t+4] + 3 * CODEBOOK_SIZE
        flat[:, base + 12] = c3[:, 8*t+5] + 3 * CODEBOOK_SIZE
        flat[:, base + 13] = c3[:, 8*t+6] + 3 * CODEBOOK_SIZE
        flat[:, base + 14] = c3[:, 8*t+7] + 3 * CODEBOOK_SIZE

    return flat


def unflatten_codes(flat: np.ndarray) -> list[np.ndarray]:
    if flat.ndim == 2:
        flat = flat[0]

    tokens_per_step = 15
    coarse_len = len(flat) // tokens_per_step
    grouped = flat[: coarse_len * tokens_per_step].reshape(coarse_len, tokens_per_step)

    c0 = grouped[:, 0]    - 0 * CODEBOOK_SIZE
    c1 = grouped[:, 1:3]  - 1 * CODEBOOK_SIZE
    c2 = grouped[:, 3:7]  - 2 * CODEBOOK_SIZE
    c3 = grouped[:, 7:15] - 3 * CODEBOOK_SIZE

    return [c0, c1.reshape(-1), c2.reshape(-1), c3.reshape(-1)]


def encode_batch(model: SNAC, audio_tensor: torch.Tensor, device: str) -> np.ndarray:
    waveforms = audio_tensor.unsqueeze(1).to(device)  # (B, 1, samples)
    with torch.inference_mode():
        codes = model.encode(waveforms)
    return flatten_codes(codes)


def encode_split(
    packing_plans: list,
    ebird_to_id: dict,
    output_dir: Path,
    model: SNAC,
    target_sr: int = SAMPLE_RATE,
    target_length: float = MAX_LENGTH,
    device: str = DEVICE,
    batch_size: int = SNAC_INF_BATCH_SIZE,
    num_workers: int = SNAC_INF_NUM_WORKERS,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PackedSegmentDataset(
        packing_plans, ebird_to_id,
        target_sr=target_sr, target_length=target_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    all_codes, all_labels = [], []

    for audio_batch, batch_labels in tqdm(loader, desc="Encode"):
        codes = encode_batch(model, audio_batch, device)
        all_codes.append(codes)
        all_labels.append(batch_labels.numpy())

    out_path = output_dir / "tokens.npz"
    np.savez(
        out_path,
        codes=np.concatenate(all_codes, axis=0),
        labels=np.concatenate(all_labels, axis=0),
    )
    logger.info(f"Saved {len(all_codes)} batches ({sum(len(c) for c in all_codes)} samples) to {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    filtered_dir = Path(FILTERED_DIR)
    segments, ebird_to_id = load_segments_and_mapping(filtered_dir, args.split, args.limit)

    logger.info(f"Planning packing for {len(segments)} segments into 5s chunks by class...")
    plans = pack_segments(segments, target_sr=SAMPLE_RATE, target_length=5, seed=SEED)
    logger.info(f"Planned {len(plans)} training samples (audio loaded lazily)")

    model = SNAC.from_pretrained(SNAC_MODEL).eval().to(DEVICE)

    output_dir = Path(TOKEN_DIR) / args.split
    encode_split(
        packing_plans=plans,
        ebird_to_id=ebird_to_id,
        output_dir=output_dir,
        model=model,
        target_sr=SAMPLE_RATE,
        target_length=5,
        device=DEVICE,
        batch_size=SNAC_INF_BATCH_SIZE,
        num_workers=SNAC_INF_NUM_WORKERS,
    )


if __name__ == "__main__":
    main()
