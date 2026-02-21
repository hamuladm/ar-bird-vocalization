import torch
from pathlib import Path


def save_checkpoint(path, model, optimizer, epoch, global_step, vocab_size, n_classes, ebird_to_id, val_loss=None):
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab_size": vocab_size,
        "n_classes": n_classes,
        "ebird_to_id": ebird_to_id,
        "val_loss": val_loss,
    }, path)


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)
