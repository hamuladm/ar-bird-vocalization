import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb

from config import (
    DEVICE,
    SEGMENT_DIR,
    AG_TOKEN_DIR,
    PRETRAIN_SAVE_DIR,
    AG_STAGE1,
    AG_STAGE2,
    AG_STAGE3,
    AG_NUM_WORKERS,
    AG_GRAD_ACCUM,
    WANDB_PROJECT,
    WANDB_ENTITY,
)
from models.audiogen import (
    load_audiogen,
    freeze_for_stage1,
    unfreeze_all,
    save_lm_checkpoint,
    load_lm_checkpoint,
)
from audio_datasets.encodec_dataset import EnCodecTokenDataset, encodec_collate_fn


class AudioGenPretrainer:
    def __init__(self, stage=None, resume=None, load_from=None, use_wandb=False):
        self.device = torch.device(DEVICE)
        self.save_dir = Path(PRETRAIN_SAVE_DIR) / "audiogen"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        self.ebird_to_id = self._load_ebird_to_id()
        self.n_species = len(self.ebird_to_id)

        self.audiogen, self.species_cond = load_audiogen(
            self.n_species, device=str(self.device)
        )
        self.lm = self.audiogen.lm

        if load_from:
            load_lm_checkpoint(load_from, self.lm, device=self.device)

        self.stages_to_run = [stage] if stage else [1, 2, 3]
        self.resume = resume
        self.initial_load_from = load_from
        self.stage_configs = {1: AG_STAGE1, 2: AG_STAGE2, 3: AG_STAGE3}

    def _load_ebird_to_id(self):
        with open(SEGMENT_DIR / "ebird_to_id.json") as f:
            return json.load(f)

    def _compute_loss(self, codes, conditions):
        output = self.lm.compute_predictions(codes, conditions)
        B, K, T, card = output.logits.shape
        logits_flat = output.logits.reshape(-1, card)
        codes_flat = codes.reshape(-1)
        mask_flat = output.mask.reshape(-1).float()
        ce = F.cross_entropy(logits_flat, codes_flat, reduction="none")
        return (ce * mask_flat).sum() / mask_flat.sum().clamp(min=1)

    def _train_epoch(self, dataloader, optimizer, scheduler, epoch, grad_accum_steps=1):
        self.lm.train()
        total_loss = 0.0
        n_steps = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            codes = batch["codes"].to(self.device)
            conditions = batch["conditions"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = self._compute_loss(codes, conditions)
                scaled_loss = loss / grad_accum_steps

            scaled_loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(
                    (p for p in self.lm.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            n_steps += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )

            if wandb.run:
                wandb.log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

        return total_loss / max(n_steps, 1)

    @torch.no_grad()
    def _validate_epoch(self, dataloader):
        self.lm.eval()
        total_loss = 0.0
        n_steps = 0

        for batch in tqdm(dataloader, desc="Validation"):
            codes = batch["codes"].to(self.device)
            conditions = batch["conditions"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = self._compute_loss(codes, conditions)

            total_loss += loss.item()
            n_steps += 1

        return total_loss / max(n_steps, 1)

    def _run_stage(self, stage, last_checkpoint):
        sc = self.stage_configs[stage]
        token_dir = Path(AG_TOKEN_DIR)

        if stage == 1:
            freeze_for_stage1(self.lm)
        else:
            if last_checkpoint:
                load_lm_checkpoint(last_checkpoint, self.lm, device=self.device)
            unfreeze_all(self.lm)

        train_ds = EnCodecTokenDataset(token_dir / "train")
        val_ds = EnCodecTokenDataset(token_dir / "val")

        train_loader = DataLoader(
            train_ds,
            batch_size=sc.batch_size,
            shuffle=True,
            collate_fn=encodec_collate_fn,
            num_workers=AG_NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=sc.batch_size,
            shuffle=False,
            collate_fn=encodec_collate_fn,
            num_workers=AG_NUM_WORKERS,
            pin_memory=True,
        )

        trainable = [p for p in self.lm.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=sc.learning_rate, weight_decay=0.01)
        total_opt_steps = (len(train_loader) // AG_GRAD_ACCUM) * sc.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, sc.warmup_steps, total_opt_steps
        )

        start_epoch = 1
        best_val_loss = float("inf")
        global_step = 0

        resume = self.resume if self.stages_to_run[0] == stage and self.resume else None
        if resume:
            ckpt = torch.load(resume, map_location=self.device, weights_only=False)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt["global_step"]
            best_val_loss = ckpt.get("val_loss") or float("inf")

        stage_dir = self.save_dir / f"stage{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, sc.epochs + 1):
            train_loss = self._train_epoch(
                train_loader, optimizer, scheduler, epoch, AG_GRAD_ACCUM
            )
            global_step += len(train_loader)
            val_loss = self._validate_epoch(val_loader)
            print(
                f"Stage {stage} | Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

            if wandb.run:
                wandb.log(
                    {
                        f"stage{stage}/train_loss_epoch": train_loss,
                        f"stage{stage}/val_loss": val_loss,
                        f"stage{stage}/epoch": epoch,
                    }
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_lm_checkpoint(
                    stage_dir / "best_model.pt",
                    self.lm,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    self.n_species,
                    self.ebird_to_id,
                    val_loss,
                    stage,
                )

            if epoch % 10 == 0:
                save_lm_checkpoint(
                    stage_dir / f"checkpoint_epoch_{epoch}.pt",
                    self.lm,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    self.n_species,
                    self.ebird_to_id,
                    val_loss,
                    stage,
                )

        return str(stage_dir / "best_model.pt")

    def run(self):
        if self.use_wandb:
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                tags=["audiogen", "pretrain"],
            )

        last_checkpoint = self.initial_load_from
        for stage in self.stages_to_run:
            last_checkpoint = self._run_stage(stage, last_checkpoint)

        if self.use_wandb:
            wandb.finish()

        print(f"All stages done. Final checkpoint: {last_checkpoint}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--load-from", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    AudioGenPretrainer(
        stage=args.stage,
        resume=args.resume,
        load_from=args.load_from,
        use_wandb=args.wandb,
    ).run()
