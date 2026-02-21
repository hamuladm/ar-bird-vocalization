import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from snac import SNAC

from utils import setup_logger, pack_segments, load_packed_sample, load_segments_and_mapping
from config import SNAC_MODEL, CODEBOOK_SIZE, SAMPLE_RATE

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
    target_length: float = 5,
    device: str = "cuda",
    batch_size: int = 32,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_codes, all_labels = [], []

    for start in tqdm(range(0, len(packing_plans), batch_size), desc="Encode"):
        batch_plans = packing_plans[start : start + batch_size]

        audio_arrays = [
            load_packed_sample(plan, target_sr=target_sr, target_length=target_length)
            for plan in batch_plans
        ]
        audio_tensor = torch.stack([torch.from_numpy(a).float() for a in audio_arrays])
        batch_labels = [ebird_to_id[plan["ebird_code"]] for plan in batch_plans]

        codes = encode_batch(model, audio_tensor, device)
        for i in range(len(batch_plans)):
            all_codes.append(codes[i])
            all_labels.append(batch_labels[i])

    out_path = output_dir / "tokens.npz"
    np.savez(out_path, codes=np.array(all_codes), labels=np.array(all_labels))
    logger.info(f"Saved {len(all_codes)} samples to {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered-dir", default="data/filtered", help="Dir with segment JSONs")
    parser.add_argument("--output-dir", default="data/snac_tokens", help="Output dir for encoded tokens")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train", help="Split to encode")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    filtered_dir = Path(args.filtered_dir)
    segments, ebird_to_id = load_segments_and_mapping(filtered_dir, args.split, args.limit)

    logger.info(f"Planning packing for {len(segments)} segments into 5s chunks by class...")
    plans = pack_segments(segments, target_sr=SAMPLE_RATE, target_length=5, seed=args.seed)
    logger.info(f"Planned {len(plans)} training samples (audio loaded lazily)")

    model = SNAC.from_pretrained(SNAC_MODEL).eval().to(args.device)

    output_dir = Path(args.output_dir) / args.split
    encode_split(
        packing_plans=plans,
        ebird_to_id=ebird_to_id,
        output_dir=output_dir,
        model=model,
        target_sr=SAMPLE_RATE,
        target_length=5,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
