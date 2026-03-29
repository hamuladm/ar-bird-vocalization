import numpy as np
import torch
import torchaudio
from pathlib import Path


from config import (
    DEVICE,
    AG_GEN_DURATION,
    AG_GEN_TEMPERATURE,
    AG_GEN_TOP_K,
    AG_GEN_CFG_COEF,
    AG_STAGE2,
    AG_STAGE3,
)
from models.audiogen import load_audiogen, make_species_conditions
from models.lora import apply_lora


def _lora_rank_from_state_dict(sd):
    for key, tensor in sd.items():
        if key.endswith(".lora_A"):
            return int(tensor.shape[0])
    return None


def _lora_alpha_for_checkpoint(ckpt):
    if "lora_alpha" in ckpt:
        return float(ckpt["lora_alpha"])
    stage = ckpt.get("stage")
    if stage == 3:
        return float(AG_STAGE3.lora_alpha)
    return float(AG_STAGE2.lora_alpha)


@torch.no_grad()
def generate_for_species(
    audiogen,
    species_id,
    duration=AG_GEN_DURATION,
    temperature=AG_GEN_TEMPERATURE,
    top_k=AG_GEN_TOP_K,
    cfg_coef=AG_GEN_CFG_COEF,
):
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

    return audiogen.compression_model.decode(gen_tokens, None)


@torch.no_grad()
def generate_audio_samples(
    audiogen,
    id_to_ebird,
    class_ids,
    duration=AG_GEN_DURATION,
    temperature=AG_GEN_TEMPERATURE,
    top_k=AG_GEN_TOP_K,
    cfg_coef=AG_GEN_CFG_COEF,
):
    samples = []
    for cid in class_ids:
        name = id_to_ebird.get(cid, f"class_{cid}")
        audio = generate_for_species(
            audiogen, cid, duration, temperature, top_k, cfg_coef
        )
        audio_np = audio[0, 0].cpu().numpy()
        samples.append((name, audio_np, audiogen.sample_rate))
    return samples


def load_finetuned_model(
    checkpoint_path,
    device="cuda",
    *,
    lora_rank=None,
    lora_alpha=None,
):
    """Load full-finetune or LoRA checkpoints (LoRA is detected by ``*.lora_A`` keys)."""
    # Always unpickle on CPU: training checkpoints include optimizer + scheduler tensors
    # (~several × LM size). Mapping them to CUDA OOMs on 16GB before the model loads.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    lm_sd = ckpt["lm_state_dict"]
    ckpt.pop("optimizer_state_dict", None)
    ckpt.pop("scheduler_state_dict", None)

    ebird_to_id = ckpt["ebird_to_id"]
    id_to_ebird = {i: c for c, i in ebird_to_id.items()}
    n_species = ckpt["n_species"]

    audiogen, _ = load_audiogen(n_species, device=device)
    inferred_rank = _lora_rank_from_state_dict(lm_sd)
    if inferred_rank is not None:
        if lora_rank is not None:
            rank = lora_rank
        elif "lora_rank" in ckpt:
            rank = int(ckpt["lora_rank"])
        else:
            rank = inferred_rank
        alpha = (
            float(lora_alpha)
            if lora_alpha is not None
            else _lora_alpha_for_checkpoint(ckpt)
        )
        apply_lora(audiogen.lm, rank=rank, alpha=alpha)
    audiogen.lm.load_state_dict(lm_sd)
    audiogen.lm.eval()

    return {
        "audiogen": audiogen,
        "ebird_to_id": ebird_to_id,
        "id_to_ebird": id_to_ebird,
        "n_species": n_species,
        "stage": ckpt.get("stage"),
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "used_lora": inferred_rank is not None,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="generated/audiogen")
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--list-classes", action="store_true")
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="Override LoRA rank when loading a LoRA checkpoint (default: infer from weights)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="Override LoRA alpha when loading a LoRA checkpoint (default: ckpt or config by stage)",
    )
    args = parser.parse_args()

    device = torch.device(DEVICE)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = load_finetuned_model(
        args.checkpoint,
        device=str(device),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    audiogen = result["audiogen"]
    ebird_to_id = result["ebird_to_id"]
    id_to_ebird = result["id_to_ebird"]
    n_species = result["n_species"]

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

    sr = audiogen.sample_rate
    stage = result["stage"]

    for i in range(args.num_samples):
        sample_class = (
            class_id if class_id is not None else np.random.randint(0, n_species)
        )
        class_name = id_to_ebird.get(sample_class, f"class_{sample_class}")

        audio = generate_for_species(audiogen, sample_class)
        audio_np = audio[0, 0].cpu()

        fname = f"{class_name}_stage{stage}_t{AG_GEN_TEMPERATURE}_{i:03d}.wav"
        filepath = output_dir / fname
        torchaudio.save(str(filepath), audio_np.unsqueeze(0), sr)
        print(f"Saved: {filepath} ({audio_np.shape[0] / sr:.2f}s)")


if __name__ == "__main__":
    main()
