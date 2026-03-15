import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from audiocraft.models import AudioGen

from utils.logging_utils import setup_logger
from utils.audio_utils import bin_pack_segments, PackedSegmentDataset
from utils.mapping_utils import load_segments_and_mapping
from config import (
    DEVICE,
    SEED,
    RELAXED_FILTERED_DIR,
    AG_PRETRAINED,
    AG_RELAXED_TOKEN_DIR,
    AG_SAMPLE_RATE,
    AG_TARGET_LENGTH,
    AG_ENCODEC_BATCH_SIZE,
    AG_ENCODEC_NUM_WORKERS,
)

logger = setup_logger("encodec_inference")


def encode_batch(
    compression_model,
    audio_tensor: torch.Tensor,
    device: str,
) -> np.ndarray:
    waveforms = audio_tensor.unsqueeze(1).to(device)  # [B, 1, samples]
    with torch.inference_mode():
        codes, scale = compression_model.encode(waveforms)
    assert scale is None, "Expected discrete codec without scale"
    return codes.cpu().numpy()


def encode_split(
    packing_plans: list,
    ebird_to_id: dict,
    output_dir: Path,
    compression_model,
    target_sr: int = AG_SAMPLE_RATE,
    target_length: float = AG_TARGET_LENGTH,
    device: str = DEVICE,
    batch_size: int = AG_ENCODEC_BATCH_SIZE,
    num_workers: int = AG_ENCODEC_NUM_WORKERS,
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

    for audio_batch, batch_labels in tqdm(loader, desc="EnCodec Encode"):
        codes = encode_batch(compression_model, audio_batch, device)
        all_codes.append(codes)
        all_labels.append(batch_labels.numpy())

    codes_cat = np.concatenate(all_codes, axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)

    out_path = output_dir / "tokens.npz"
    np.savez(out_path, codes=codes_cat, labels=labels_cat)
    logger.info(
        f"Saved {len(codes_cat)} samples (codes shape {codes_cat.shape}) to {out_path}"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    filtered_dir = Path(RELAXED_FILTERED_DIR)
    segments, ebird_to_id = load_segments_and_mapping(
        filtered_dir, args.split, args.limit
    )

    logger.info(
        f"Packing {len(segments)} segments into {AG_TARGET_LENGTH}s chunks by class..."
    )
    plans = bin_pack_segments(
        segments, target_length=AG_TARGET_LENGTH, seed=SEED
    )
    logger.info(f"Planned {len(plans)} training samples")

    logger.info("Loading AudioGen compression model (EnCodec 16kHz)...")
    audiogen = AudioGen.get_pretrained(AG_PRETRAINED, device=DEVICE)
    compression_model = audiogen.compression_model

    output_dir = Path(AG_RELAXED_TOKEN_DIR) / args.split
    encode_split(
        packing_plans=plans,
        ebird_to_id=ebird_to_id,
        output_dir=output_dir,
        compression_model=compression_model,
        target_sr=AG_SAMPLE_RATE,
        target_length=AG_TARGET_LENGTH,
        device=DEVICE,
        batch_size=AG_ENCODEC_BATCH_SIZE,
        num_workers=AG_ENCODEC_NUM_WORKERS,
    )


if __name__ == "__main__":
    main()
