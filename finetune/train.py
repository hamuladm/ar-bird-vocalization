"""Three-stage finetuning pipeline for AudioGen-medium on bird vocalizations.

Stage 1 -- Species conditioning warmup:
    Freeze the transformer, codec embeddings, and output heads.
    Train only the new SpeciesConditioner embedding so the model learns
    to associate species IDs with audio generation.

Stage 2 -- General bird vocal structure:
    Unfreeze all 1.5B parameters. Train on relaxed-filtered data
    (diverse but noisier) so the model learns the broad structure of
    bird vocalizations.

Stage 3 -- Quality refinement:
    Continue from Stage 2 checkpoint.  Switch to strict-filtered
    high-quality data and a lower learning rate so the model learns
    the critical spectral and temporal features of each species.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from utils.logging_utils import setup_logger
from utils.mapping_utils import load_ebird_mapping
from config import (
    DEVICE,
    RELAXED_FILTERED_DIR,
    AG_RELAXED_TOKEN_DIR,
    AG_STRICT_TOKEN_DIR,
    AG_SAVE_DIR,
    AG_STAGE1,
    AG_STAGE2,
    AG_STAGE3,
    AG_NUM_WORKERS,
    AG_GRAD_ACCUM,
)

from finetune.audiogen_model import (
    load_audiogen_for_finetuning,
    freeze_for_stage1,
    unfreeze_all,
    save_lm_checkpoint,
    load_lm_checkpoint,
)
from finetune.audiogen_dataset import EnCodecTokenDataset, encodec_collate_fn

logger = setup_logger("finetune")


# ---------------------------------------------------------------------------
# Core training helpers
# ---------------------------------------------------------------------------

def compute_cross_entropy(lm, codes, conditions):
    """Run ``lm.compute_predictions`` and return masked cross-entropy loss."""
    output = lm.compute_predictions(codes, conditions)
    B, K, T, card = output.logits.shape

    logits_flat = output.logits.reshape(-1, card)
    codes_flat = codes.reshape(-1)
    mask_flat = output.mask.reshape(-1).float()

    ce = F.cross_entropy(logits_flat, codes_flat, reduction="none")
    loss = (ce * mask_flat).sum() / mask_flat.sum().clamp(min=1)
    return loss


def train_epoch(
    lm,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch,
    grad_accum_steps=1,
    use_amp=True,
):
    lm.train()
    total_loss = 0.0
    n_steps = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    optimizer.zero_grad()
    for step, batch in enumerate(pbar):
        codes = batch["codes"].to(device)
        conditions = batch["conditions"]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = compute_cross_entropy(lm, codes, conditions)
            scaled_loss = loss / grad_accum_steps

        scaled_loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(
                (p for p in lm.parameters() if p.requires_grad), 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        actual_loss = loss.item()
        total_loss += actual_loss
        n_steps += 1
        pbar.set_postfix(
            loss=f"{actual_loss:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

        if wandb.run:
            wandb.log({"train_loss": actual_loss, "lr": scheduler.get_last_lr()[0]})

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def validate_epoch(lm, dataloader, device, use_amp=True):
    lm.eval()
    total_loss = 0.0
    n_steps = 0

    for batch in tqdm(dataloader, desc="Validation"):
        codes = batch["codes"].to(device)
        conditions = batch["conditions"]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = compute_cross_entropy(lm, codes, conditions)

        total_loss += loss.item()
        n_steps += 1

    return total_loss / max(n_steps, 1)


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage(
    stage: int,
    lm,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    warmup_steps: int,
    save_dir: Path,
    n_species: int,
    ebird_to_id: dict,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    resume_checkpoint: str | None = None,
) -> Path:
    """Run one training stage and return the path to the best checkpoint."""
    logger.info(f"{'=' * 20}  Stage {stage}  {'=' * 20}")

    trainable_params = [p for p in lm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    total_opt_steps = (len(train_loader) // grad_accum_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_opt_steps
    )

    start_epoch = 1
    best_val_loss = float("inf")
    global_step = 0

    if resume_checkpoint:
        logger.info(f"Resuming stage from {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("val_loss") or float("inf")

    stage_dir = save_dir / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_epoch(
            lm, train_loader, optimizer, scheduler, device, epoch,
            grad_accum_steps=grad_accum_steps, use_amp=use_amp,
        )
        global_step += len(train_loader)

        val_loss = validate_epoch(lm, val_loader, device, use_amp=use_amp)
        logger.info(
            f"Stage {stage} | Epoch {epoch}: "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )

        if wandb.run:
            wandb.log({
                f"stage{stage}/train_loss_epoch": train_loss,
                f"stage{stage}/val_loss": val_loss,
                f"stage{stage}/epoch": epoch,
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_lm_checkpoint(
                stage_dir / "best_model.pt",
                lm, optimizer, scheduler, epoch, global_step,
                n_species, ebird_to_id, val_loss, stage,
            )
            logger.info(f"  -> new best model (val_loss={val_loss:.4f})")

        if epoch % 10 == 0:
            save_lm_checkpoint(
                stage_dir / f"checkpoint_epoch_{epoch}.pt",
                lm, optimizer, scheduler, epoch, global_step,
                n_species, ebird_to_id, val_loss, stage,
            )

    return stage_dir / "best_model.pt"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--load-from", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device(DEVICE)
    save_dir = Path(AG_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    ebird_to_id, id_to_ebird = load_ebird_mapping(Path(RELAXED_FILTERED_DIR))
    n_species = len(ebird_to_id)
    logger.info(f"Species count: {n_species}")

    audiogen, species_cond = load_audiogen_for_finetuning(
        n_species, device=str(device)
    )
    lm = audiogen.lm

    if args.load_from:
        logger.info(f"Loading weights from: {args.load_from}")
        load_lm_checkpoint(args.load_from, lm, device=device)

    if args.wandb:
        wandb.init(project="ar-bird-vocalization-audiogen")

    stage_configs = {1: AG_STAGE1, 2: AG_STAGE2, 3: AG_STAGE3}
    stages_to_run = [args.stage] if args.stage else [1, 2, 3]
    last_checkpoint: str | None = args.load_from

    for stage in stages_to_run:
        sc = stage_configs[stage]

        if stage == 1:
            token_dir = Path(AG_RELAXED_TOKEN_DIR)
            freeze_for_stage1(lm)
        elif stage == 2:
            token_dir = Path(AG_RELAXED_TOKEN_DIR)
            if last_checkpoint:
                load_lm_checkpoint(last_checkpoint, lm, device=device)
            unfreeze_all(lm)
        else:
            token_dir = Path(AG_STRICT_TOKEN_DIR)
            if last_checkpoint:
                load_lm_checkpoint(last_checkpoint, lm, device=device)
            unfreeze_all(lm)

        train_ds = EnCodecTokenDataset(token_dir / "train")
        val_ds = EnCodecTokenDataset(token_dir / "val")
        logger.info(
            f"Stage {stage} data: {len(train_ds)} train / {len(val_ds)} val "
            f"(codes shape per sample: {train_ds.codes.shape[1:]})"
        )

        train_loader = DataLoader(
            train_ds, batch_size=sc.batch_size, shuffle=True,
            collate_fn=encodec_collate_fn,
            num_workers=AG_NUM_WORKERS, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=sc.batch_size, shuffle=False,
            collate_fn=encodec_collate_fn,
            num_workers=AG_NUM_WORKERS, pin_memory=True,
        )

        resume = args.resume if args.stage == stage else None

        best_ckpt = run_stage(
            stage=stage,
            lm=lm,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=sc.epochs,
            lr=sc.learning_rate,
            warmup_steps=sc.warmup_steps,
            save_dir=save_dir,
            n_species=n_species,
            ebird_to_id=ebird_to_id,
            grad_accum_steps=AG_GRAD_ACCUM,
            use_amp=True,
            resume_checkpoint=resume,
        )
        last_checkpoint = str(best_ckpt)

    if args.wandb:
        wandb.finish()

    logger.info(f"All stages done.  Final checkpoint: {last_checkpoint}")


if __name__ == "__main__":
    main()
