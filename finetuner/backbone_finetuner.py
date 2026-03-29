import json
import math
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb

from snac import SNAC

from config import (
    DEVICE,
    SNAC_MODEL,
    SEGMENT_DIR,
    TOKEN_DIR,
    MAX_SEQ_LEN,
    PRETRAIN_NUM_WORKERS,
    WANDB_PROJECT,
    WANDB_ENTITY,
    BACKBONE,
)
from models.backbone import PAD_TOKEN, CLASS_TOKEN_OFFSET, create_model
from audio_datasets.snac_dataset import SNACTokenDataset, snac_collate_fn
from utils.checkpoint import save_checkpoint, load_checkpoint


class GPT2Finetuner:
    def __init__(
        self,
        epochs=50,
        batch_size=4,
        lr=5e-5,
        warmup_steps=500,
        grad_accum_steps=1,
        resume=None,
        load_from=None,
        use_wandb=False,
        sample_class_ids=None,
        num_sample_classes=3,
    ):
        self.device = torch.device(DEVICE)
        self.save_dir = Path("checkpoints/gpt2_finetune")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.epochs = epochs
        self.grad_accum_steps = max(1, int(grad_accum_steps))

        self.ebird_to_id = self._load_ebird_to_id()
        self.id_to_ebird = {i: c for c, i in self.ebird_to_id.items()}
        self.n_classes = len(self.ebird_to_id)
        self.vocab_size = CLASS_TOKEN_OFFSET + self.n_classes

        train_ds = SNACTokenDataset(TOKEN_DIR / "train", max_seq_len=MAX_SEQ_LEN)
        val_ds = SNACTokenDataset(TOKEN_DIR / "val", max_seq_len=MAX_SEQ_LEN)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=snac_collate_fn,
            num_workers=PRETRAIN_NUM_WORKERS,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=snac_collate_fn,
            num_workers=PRETRAIN_NUM_WORKERS,
            pin_memory=True,
        )

        self.backbone = BACKBONE
        self.model = create_model(
            backbone=self.backbone, vocab_size=self.vocab_size, n_positions=MAX_SEQ_LEN
        ).to(self.device)
        self.snac_model = SNAC.from_pretrained(SNAC_MODEL).eval().to(self.device)

        if load_from:
            ckpt = load_checkpoint(load_from, device=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01
        )
        steps_per_epoch = math.ceil(
            len(self.train_loader) / self.grad_accum_steps
        )
        total_steps = steps_per_epoch * epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        self.start_epoch = 1
        self.global_step = 0
        self.best_val_loss = float("inf")

        if resume:
            ckpt = load_checkpoint(resume, device=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.start_epoch = ckpt["epoch"] + 1
            self.global_step = ckpt["global_step"]
            self.best_val_loss = ckpt.get("val_loss") or float("inf")

        if sample_class_ids is not None:
            self.sample_class_ids = sample_class_ids
        else:
            rng = np.random.default_rng(42)
            self.sample_class_ids = rng.choice(
                self.n_classes,
                size=min(num_sample_classes, self.n_classes),
                replace=False,
            ).tolist()

    @staticmethod
    def _load_ebird_to_id():
        with open(SEGMENT_DIR / "ebird_to_id.json") as f:
            return json.load(f)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        self.optimizer.zero_grad(set_to_none=True)
        step = -1

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            targets = input_ids.clone()
            targets[targets == PAD_TOKEN] = -100

            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=targets
            )
            loss = outputs.loss
            scaled = loss / self.grad_accum_steps
            scaled.backward()

            total_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}"
            )

            if wandb.run:
                wandb.log(
                    {"train_loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}
                )

            if (step + 1) % self.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

        if (
            step >= 0
            and (step + 1) % self.grad_accum_steps != 0
        ):
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            targets = input_ids.clone()
            targets[targets == PAD_TOKEN] = -100

            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=targets
            )
            total_loss += outputs.loss.item()

        return total_loss / len(self.val_loader)

    def _save(self, path, epoch, val_loss):
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            epoch,
            self.global_step,
            self.vocab_size,
            self.n_classes,
            self.ebird_to_id,
            val_loss=val_loss,
            scheduler=self.scheduler,
            backbone=self.backbone,
        )

    def run(self):
        if self.use_wandb:
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                tags=[self.backbone, "finetune"],
            )

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch()
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

            if self.use_wandb:
                from generator.snac_generator import generate_audio_samples

                log_dict = {
                    "epoch": epoch,
                    "train_loss_epoch": train_loss,
                    "val_loss": val_loss,
                }
                samples = generate_audio_samples(
                    self.model,
                    self.snac_model,
                    self.device,
                    self.id_to_ebird,
                    class_ids=self.sample_class_ids,
                    max_length=MAX_SEQ_LEN,
                )
                for name, audio, sr in samples:
                    log_dict[f"audio/{name}"] = wandb.Audio(
                        audio, sample_rate=sr, caption=f"{name}_epoch{epoch}"
                    )
                wandb.log(log_dict)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save(self.save_dir / "best_model.pt", epoch, val_loss)

            if epoch % 10 == 0:
                self._save(
                    self.save_dir / f"checkpoint_epoch_{epoch}.pt", epoch, val_loss
                )

        if self.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--load-from", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Micro-batches per optimizer step (effective batch ≈ batch_size × this)",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--sample-classes", type=int, nargs="*", default=None)
    parser.add_argument("--num-sample-classes", type=int, default=3)
    args = parser.parse_args()

    GPT2Finetuner(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum,
        resume=args.resume,
        load_from=args.load_from,
        use_wandb=args.wandb,
        sample_class_ids=args.sample_classes,
        num_sample_classes=args.num_sample_classes,
    ).run()
