import math

import numpy as np
import torch
import torchaudio
from pathlib import Path
from snac import SNAC

from config import (
    DEVICE,
    SAMPLE_RATE,
    SNAC_MODEL,
    CODEBOOK_SIZE,
    MAX_SEQ_LEN,
    SNAC_GEN_TEMPERATURE,
    SNAC_GEN_TOP_K,
)
from models.gpt2 import create_gpt2_model, generate_tokens, extract_snac_codes
from preprocessing.tokenize import unflatten_codes
from utils.checkpoint import load_checkpoint


def decode_to_audio(snac_codes, snac_model, device):
    code_arrays = unflatten_codes(snac_codes)

    ws = snac_model.attn_window_size or 1
    stride0 = snac_model.vq_strides[0]
    align = ws // math.gcd(stride0, ws)
    coarse_len = len(code_arrays[0])
    coarse_len = (coarse_len // align) * align

    code_arrays = [
        code_arrays[0][:coarse_len],
        code_arrays[1][: coarse_len * 2],
        code_arrays[2][: coarse_len * 4],
        code_arrays[3][: coarse_len * 8],
    ]

    code_arrays = [np.clip(c, 0, CODEBOOK_SIZE - 1) for c in code_arrays]
    codes_torch = [
        torch.from_numpy(c).long().unsqueeze(0).to(device) for c in code_arrays
    ]
    with torch.inference_mode():
        audio = snac_model.decode(codes_torch)
    return audio[0, 0].cpu().numpy()


@torch.no_grad()
def generate_audio_samples(
    model,
    snac_model,
    device,
    id_to_ebird,
    class_ids,
    sample_rate=SAMPLE_RATE,
    max_length=MAX_SEQ_LEN,
    temperature=1.0,
    top_k=50,
):
    was_training = model.training
    samples = []
    for class_id in class_ids:
        class_name = id_to_ebird.get(class_id, f"class_{class_id}")
        tokens = generate_tokens(
            model, device, class_id, max_length, temperature, top_k
        )
        snac_codes = extract_snac_codes(tokens)
        if len(snac_codes) < 15:
            continue
        audio = decode_to_audio(snac_codes, snac_model, device)
        samples.append((class_name, audio, sample_rate))
    if was_training:
        model.train()
    return samples


def load_generation_models(checkpoint_path, device, max_length=MAX_SEQ_LEN):
    ckpt = load_checkpoint(checkpoint_path, device=device)

    ebird_to_id = ckpt["ebird_to_id"]
    id_to_ebird = {i: c for c, i in ebird_to_id.items()}
    vocab_size = ckpt["model_state_dict"]["transformer.wte.weight"].shape[0]

    model = create_gpt2_model(vocab_size=vocab_size, n_positions=max_length)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    snac_model = SNAC.from_pretrained(SNAC_MODEL).eval().to(device)

    return {
        "model": model,
        "snac_model": snac_model,
        "ebird_to_id": ebird_to_id,
        "id_to_ebird": id_to_ebird,
        "n_classes": len(ebird_to_id),
        "epoch": ckpt.get("epoch"),
        "step": ckpt.get("global_step"),
        "val_loss": ckpt.get("val_loss"),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="generated")
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--list-classes", action="store_true")
    args = parser.parse_args()

    device = torch.device(DEVICE)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = load_generation_models(args.checkpoint, device)
    model = result["model"]
    snac_model = result["snac_model"]
    ebird_to_id = result["ebird_to_id"]
    id_to_ebird = result["id_to_ebird"]
    n_classes = result["n_classes"]
    step = result["step"]

    if args.list_classes:
        for cid in sorted(id_to_ebird.keys()):
            print(f"  {cid:4d}: {id_to_ebird[cid]}")
        return

    if args.class_name:
        if args.class_name not in ebird_to_id:
            print(f"Unknown class: {args.class_name}")
            return
        class_id = ebird_to_id[args.class_name]
    elif args.class_id is not None:
        class_id = args.class_id
    else:
        class_id = None

    for i in range(args.num_samples):
        sample_class = (
            class_id if class_id is not None else np.random.randint(0, n_classes)
        )
        class_name = id_to_ebird.get(sample_class, f"class_{sample_class}")

        tokens = generate_tokens(
            model,
            device,
            class_id=sample_class,
            max_length=MAX_SEQ_LEN,
            temperature=SNAC_GEN_TEMPERATURE,
            top_k=SNAC_GEN_TOP_K,
        )
        snac_codes = extract_snac_codes(tokens)

        if len(snac_codes) < 15:
            print(f"Too few tokens for {class_name}, skipping")
            continue

        audio = decode_to_audio(snac_codes, snac_model, device)
        filename = f"{class_name}_step{step}_t{SNAC_GEN_TEMPERATURE}_{i:03d}.wav"
        filepath = output_dir / filename
        torchaudio.save(
            str(filepath), torch.from_numpy(audio).unsqueeze(0), SAMPLE_RATE
        )
        print(f"Saved: {filepath} ({len(audio) / SAMPLE_RATE:.2f}s)")


if __name__ == "__main__":
    main()
