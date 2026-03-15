"""Generate bird vocalizations with a finetuned AudioGen model.

Usage examples::

    # Generate 3 samples for species "norcar" (Northern Cardinal)
    python -m finetune.generate \
        --checkpoint checkpoints/audiogen/stage3/best_model.pt \
        --class-name norcar --num-samples 3

    # List available species
    python -m finetune.generate \
        --checkpoint checkpoints/audiogen/stage3/best_model.pt \
        --list-classes
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio

from audiocraft.models import AudioGen
from audiocraft.modules.conditioners import ConditioningAttributes

from utils.logging_utils import setup_logger
from utils.mapping_utils import load_ebird_mapping
from config import (
    DEVICE,
    RELAXED_FILTERED_DIR,
    AG_GEN_DURATION,
    AG_GEN_TEMPERATURE,
    AG_GEN_TOP_K,
    AG_GEN_CFG_COEF,
)
from finetune.audiogen_model import (
    load_audiogen_for_finetuning,
    load_lm_checkpoint,
    make_species_conditions,
)

logger = setup_logger("audiogen_generate")


@torch.no_grad()
def generate_for_species(
    audiogen: AudioGen,
    species_id: int,
    duration: float = AG_GEN_DURATION,
    temperature: float = AG_GEN_TEMPERATURE,
    top_k: int = AG_GEN_TOP_K,
    cfg_coef: float = AG_GEN_CFG_COEF,
) -> torch.Tensor:
    audiogen.set_generation_params(
        duration=duration,
        temperature=temperature,
        top_k=top_k,
        cfg_coef=cfg_coef,
        use_sampling=True,
    )

    conditions = make_species_conditions([species_id])
    lm = audiogen.lm
    lm.eval()

    with audiogen.autocast:
        total_gen_len = int(duration * audiogen.frame_rate)
        gen_tokens = lm.generate(
            prompt=None,
            conditions=conditions,
            max_gen_len=total_gen_len,
            use_sampling=True,
            temp=temperature,
            top_k=top_k,
            cfg_coef=cfg_coef,
        )

    audio = audiogen.generate_audio(gen_tokens)
    return audio


def load_finetuned_model(
    checkpoint_path: str,
    filtered_dir: str,
    device: str = "cuda",
) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "ebird_to_id" in ckpt:
        ebird_to_id = ckpt["ebird_to_id"]
        id_to_ebird = {i: c for c, i in ebird_to_id.items()}
        n_species = ckpt["n_species"]
    else:
        ebird_to_id, id_to_ebird = load_ebird_mapping(Path(filtered_dir))
        n_species = len(ebird_to_id)

    audiogen, _ = load_audiogen_for_finetuning(n_species, device=device)
    audiogen.lm.load_state_dict(ckpt["lm_state_dict"])
    audiogen.lm.eval()

    logger.info(
        f"Loaded checkpoint: stage={ckpt.get('stage')}, "
        f"epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')}, "
        f"species={n_species}"
    )

    return {
        "audiogen": audiogen,
        "ebird_to_id": ebird_to_id,
        "id_to_ebird": id_to_ebird,
        "n_species": n_species,
        "stage": ckpt.get("stage"),
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
    }


@torch.no_grad()
def generate_audio_samples(
    audiogen: AudioGen,
    id_to_ebird: dict,
    class_ids: list[int],
    duration: float = AG_GEN_DURATION,
    temperature: float = AG_GEN_TEMPERATURE,
    top_k: int = AG_GEN_TOP_K,
    cfg_coef: float = AG_GEN_CFG_COEF,
) -> list[tuple[str, np.ndarray, int]]:
    samples = []
    for cid in class_ids:
        name = id_to_ebird.get(cid, f"class_{cid}")
        audio = generate_for_species(
            audiogen, cid,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            cfg_coef=cfg_coef,
        )
        audio_np = audio[0, 0].cpu().numpy()
        samples.append((name, audio_np, audiogen.sample_rate))
    return samples


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="generated/audiogen")
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--list-classes", action="store_true")
    args = parser.parse_args()

    device = torch.device(DEVICE)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = load_finetuned_model(
        args.checkpoint, RELAXED_FILTERED_DIR, device=str(device)
    )
    audiogen = result["audiogen"]
    ebird_to_id = result["ebird_to_id"]
    id_to_ebird = result["id_to_ebird"]
    n_species = result["n_species"]

    if args.list_classes:
        logger.info(f"Available bird classes ({n_species}):")
        for cid in sorted(id_to_ebird.keys()):
            print(f"  {cid:4d}: {id_to_ebird[cid]}")
        return

    if args.class_name:
        if args.class_name not in ebird_to_id:
            logger.error(
                f"Unknown class: {args.class_name}. Use --list-classes."
            )
            return
        class_id = ebird_to_id[args.class_name]
    elif args.class_id is not None:
        class_id = args.class_id
    else:
        class_id = None

    sr = audiogen.sample_rate
    stage = result["stage"]

    for i in range(args.num_samples):
        sample_class = (
            class_id if class_id is not None
            else np.random.randint(0, n_species)
        )
        class_name = id_to_ebird.get(sample_class, f"class_{sample_class}")

        logger.info(
            f"[{i + 1}/{args.num_samples}] Generating {AG_GEN_DURATION}s "
            f"for {class_name} (id={sample_class}) ..."
        )

        audio = generate_for_species(
            audiogen, sample_class,
            duration=AG_GEN_DURATION,
            temperature=AG_GEN_TEMPERATURE,
            top_k=AG_GEN_TOP_K,
            cfg_coef=AG_GEN_CFG_COEF,
        )
        audio_np = audio[0, 0].cpu()

        fname = (
            f"{class_name}_stage{stage}_t{AG_GEN_TEMPERATURE}_{i:03d}.wav"
        )
        filepath = output_dir / fname
        torchaudio.save(str(filepath), audio_np.unsqueeze(0), sr)
        logger.info(f"  Saved: {filepath} ({audio_np.shape[0] / sr:.2f}s)")

    logger.info("Done.")


if __name__ == "__main__":
    main()
