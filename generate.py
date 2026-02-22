import argparse
import numpy as np
import torch
import torchaudio
from pathlib import Path
from snac import SNAC

from utils.logging_utils import setup_logger
from utils.mapping_utils import load_ebird_mapping
from config import (
    SNAC_MODEL,
    SAMPLE_RATE,
    CODEBOOK_SIZE,
    MAX_SEQ_LEN,
)
from model import (
    create_gpt2_model,
    generate_tokens,
    extract_snac_codes,
)
from checkpoint import load_checkpoint
from snac_inference import unflatten_codes

logger = setup_logger("generate")


def decode_to_audio(snac_codes: np.ndarray, snac_model: SNAC, device: str) -> np.ndarray:
    code_arrays = unflatten_codes(snac_codes)
    code_arrays = [np.clip(c, 0, CODEBOOK_SIZE - 1) for c in code_arrays]
    codes_torch = [torch.from_numpy(c).long().unsqueeze(0).to(device) for c in code_arrays]
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
    """Generate one audio sample per class_id, returning (name, np_audio, sr) tuples."""
    was_training = model.training
    samples = []
    for class_id in class_ids:
        class_name = id_to_ebird.get(class_id, f"class_{class_id}")
        tokens = generate_tokens(model, device, class_id, max_length, temperature, top_k)
        snac_codes = extract_snac_codes(tokens)
        if len(snac_codes) < 15:
            continue
        audio = decode_to_audio(snac_codes, snac_model, device)
        samples.append((class_name, audio, sample_rate))
    if was_training:
        model.train()
    return samples


def load_generation_models(
    checkpoint_path: str,
    device: torch.device,
    filtered_dir: str,
    max_length: int,
    list_classes: bool = False,
) -> dict | None:
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device=device)

    if "ebird_to_id" in checkpoint:
        ebird_to_id = checkpoint["ebird_to_id"]
        id_to_ebird = {i: c for c, i in ebird_to_id.items()}
        logger.info(f"Loaded ebird_to_id from checkpoint ({len(ebird_to_id)} classes)")
    else:
        logger.warning("Checkpoint has no ebird_to_id, falling back to segment files")
        ebird_to_id, id_to_ebird = load_ebird_mapping(Path(filtered_dir))

    n_classes = len(ebird_to_id)
    vocab_size = checkpoint["model_state_dict"]["transformer.wte.weight"].shape[0]

    if list_classes:
        logger.info(f"Available bird classes ({n_classes} total):")
        for class_id in sorted(id_to_ebird.keys()):
            print(f"  {class_id:4d}: {id_to_ebird[class_id]}")
        return None

    model = create_gpt2_model(vocab_size=vocab_size, n_positions=max_length)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    epoch = checkpoint.get("epoch", "?")
    step = checkpoint.get("global_step", "?")
    val_loss = checkpoint.get("val_loss", "?")
    logger.info(f"Loaded model (epoch={epoch}, step={step}, val_loss={val_loss})")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    logger.info(f"Loading SNAC model: {SNAC_MODEL}")
    snac_model = SNAC.from_pretrained(SNAC_MODEL).eval().to(device)

    return {
        "model": model,
        "snac_model": snac_model,
        "ebird_to_id": ebird_to_id,
        "id_to_ebird": id_to_ebird,
        "n_classes": n_classes,
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="generated", help="Output directory for audio files")
    parser.add_argument("--filtered-dir", type=str, default="data/filtered",
                        help="Dir with segment JSONs (fallback if checkpoint lacks ebird_to_id)")
    parser.add_argument("--class-id", type=int, default=None, help="Bird class ID")
    parser.add_argument("--class-name", type=str, default=None, help="Bird species ebird code (e.g. gretit1)")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--max-length", type=int, default=MAX_SEQ_LEN, help="Max sequence length")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--list-classes", action="store_true", help="List all available bird classes and exit")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = load_generation_models(
        checkpoint_path=args.checkpoint,
        device=device,
        filtered_dir=args.filtered_dir,
        max_length=args.max_length,
        list_classes=args.list_classes,
    )
    if result is None:
        return

    model = result["model"]
    snac_model = result["snac_model"]
    ebird_to_id = result["ebird_to_id"]
    id_to_ebird = result["id_to_ebird"]
    n_classes = result["n_classes"]
    step = result["step"]

    if args.class_name:
        if args.class_name not in ebird_to_id:
            logger.error(f"Unknown class: {args.class_name}. Use --list-classes to see options.")
            return
        class_id = ebird_to_id[args.class_name]
    elif args.class_id is not None:
        class_id = args.class_id
    else:
        class_id = None

    for i in range(args.num_samples):
        sample_class = class_id if class_id is not None else np.random.randint(0, n_classes)
        class_name = id_to_ebird.get(sample_class, f"class_{sample_class}")

        logger.info(f"[{i+1}/{args.num_samples}] Generating for class {sample_class} ({class_name}) ...")

        tokens = generate_tokens(
            model, device,
            class_id=sample_class,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        snac_codes = extract_snac_codes(tokens)
        logger.info(f"  Generated {len(snac_codes)} SNAC tokens ({len(tokens)} total)")
    
        if len(snac_codes) < 15:
            logger.warning(f"  Too few tokens to decode, skipping")
            continue

        audio = decode_to_audio(snac_codes, snac_model, device)

        filename = f"{class_name}_step{step}_t{args.temperature}_{i:03d}.wav"
        filepath = output_dir / filename
        torchaudio.save(str(filepath), torch.from_numpy(audio).unsqueeze(0), SAMPLE_RATE)
        logger.info(f"  Saved: {filepath} ({len(audio)/SAMPLE_RATE:.2f}s)")

    logger.info("Done.")


if __name__ == "__main__":
    main()
