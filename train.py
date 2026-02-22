import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, get_cosine_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import wandb
import argparse

from snac import SNAC

from utils.logging_utils import setup_logger
from utils.mapping_utils import load_ebird_mapping
from config import (
    TOKEN_DIR,
    TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE,
    LEARNING_RATE,
    WARMUP_STEPS,
    MAX_SEQ_LEN,
    NUM_WORKERS,
    SAVE_DIR,
    WANDB_PROJECT,
    WANDB_ENTITY,
    SNAC_MODEL,
    SAMPLE_RATE,
)
from model import (
    SNAC_VOCAB_SIZE,
    PAD_TOKEN,
    CLASS_TOKEN_OFFSET,
    create_gpt2_model
)
from dataset import SNACTokenDataset, snac_collate_fn
from checkpoint import save_checkpoint, load_checkpoint
from generate import generate_audio_samples

logger = setup_logger("train")


def train_epoch(model: GPT2LMHeadModel,
                dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer,
                scheduler,
                device,
                epoch,
                save_dir=None,
                save_every_steps=200,
                global_step=0,
                vocab_size=None,
                n_classes=None,
                ebird_to_id=None,
                save_scheduler=None):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        targets = input_ids.clone()
        targets[targets == PAD_TOKEN] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets,
        )

        loss = outputs.loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}", step=global_step)

        if wandb.run:
            wandb.log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

        # if save_dir and global_step % save_every_steps == 0:
        #     save_checkpoint(
        #         save_dir / f"checkpoint_step_{global_step}.pt",
        #         model, optimizer, epoch, global_step,
        #         vocab_size, n_classes, ebird_to_id,
        #         scheduler=save_scheduler,
        #     )
        #     logger.info(f"Saved checkpoint at step {global_step}")

    return total_loss / len(dataloader), global_step


@torch.no_grad()
def validate_epoch(model: GPT2LMHeadModel, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Validation"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        targets = input_ids.clone()
        targets[targets == PAD_TOKEN] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets,
        )
        total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-dir", type=str, default=str(SAVE_DIR))
    parser.add_argument("--token-dir", type=str, default=str(TOKEN_DIR))
    parser.add_argument("--filtered-dir", type=str, default="data/filtered")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--sample-classes", type=int, nargs="*", default=None,
                        help="Class IDs for audio generation each epoch (default: 3 random)")
    parser.add_argument("--num-sample-classes", type=int, default=3,
                        help="Number of random classes to sample if --sample-classes not given")
    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    token_dir = Path(args.token_dir)

    ebird_to_id, id_to_ebird = load_ebird_mapping(Path(args.filtered_dir))
    n_classes = len(ebird_to_id)
    vocab_size = CLASS_TOKEN_OFFSET + n_classes
    logger.info(
        f"Classes: {n_classes}, Vocab: {vocab_size} "
        f"(SNAC: {SNAC_VOCAB_SIZE} + 3 special + {n_classes} class tokens)"
    )

    train_dataset = SNACTokenDataset(token_dir / "train", max_seq_len=args.max_seq_len)
    val_dataset = SNACTokenDataset(token_dir / "val", max_seq_len=args.max_seq_len)
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=snac_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=snac_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = create_gpt2_model(vocab_size=vocab_size, n_positions=args.max_seq_len).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    snac_model = SNAC.from_pretrained(SNAC_MODEL).eval().to(device)

    if args.sample_classes is not None:
        sample_class_ids = args.sample_classes
    else:
        rng = np.random.default_rng(42)
        sample_class_ids = rng.choice(n_classes, size=min(args.num_sample_classes, n_classes), replace=False).tolist()
    logger.info(f"Audio sample classes: {[id_to_ebird[c] for c in sample_class_ids]}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume, device=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            for _ in range(ckpt["global_step"]):
                scheduler.step()
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("val_loss") or float("inf")
        logger.info(f"Resumed: epoch={ckpt['epoch']}, step={global_step}, val_loss={best_val_loss}")

    if args.wandb:
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=vars(args))

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            save_dir=save_dir, save_every_steps=200, global_step=global_step,
            vocab_size=vocab_size, n_classes=n_classes, ebird_to_id=ebird_to_id,
            save_scheduler=scheduler,
        )
        val_loss = validate_epoch(model, val_loader, device)
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if args.wandb:
            log_dict = {"epoch": epoch, "train_loss_epoch": train_loss, "val_loss": val_loss}

            logger.info(f"Generating audio samples for epoch {epoch}...")
            samples = generate_audio_samples(
                model, snac_model, device, id_to_ebird,
                class_ids=sample_class_ids,
                max_length=args.max_seq_len,
            )
            for name, audio, sr in samples:
                log_dict[f"audio/{name}"] = wandb.Audio(audio, sample_rate=sr, caption=f"{name}_epoch{epoch}")

            wandb.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                save_dir / "best_model.pt",
                model, optimizer, epoch, global_step,
                vocab_size, n_classes, ebird_to_id, val_loss=val_loss,
                scheduler=scheduler,
            )
            logger.info(f"Saved best model (val_loss={val_loss:.4f})")

        if epoch % 10 == 0:
            save_checkpoint(
                save_dir / f"checkpoint_epoch_{epoch}.pt",
                model, optimizer, epoch, global_step,
                vocab_size, n_classes, ebird_to_id, val_loss=val_loss,
                scheduler=scheduler,
            )

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
