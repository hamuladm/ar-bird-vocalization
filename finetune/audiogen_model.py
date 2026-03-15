import typing as tp

import torch
from torch import nn

from audiocraft.models import AudioGen
from audiocraft.models.lm import LMModel
from audiocraft.modules.conditioners import (
    BaseConditioner,
    ConditioningAttributes,
    ConditionType,
)

from utils.logging_utils import setup_logger
from config import AG_PRETRAINED

logger = setup_logger("audiogen_model")


class SpeciesConditioner(BaseConditioner):
    def __init__(self, n_species: int, dim: int, output_dim: int):
        super().__init__(dim, output_dim)
        self.embed = nn.Embedding(n_species + 1, dim)
        self.n_species = n_species

    def tokenize(
        self, x: tp.List[tp.Optional[str]]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        device = self.embed.weight.device
        ids, mask = [], []
        for s in x:
            if s is not None:
                ids.append(int(s))
                mask.append(1)
            else:
                ids.append(self.n_species)
                mask.append(0)
        tokens = torch.tensor(ids, device=device).unsqueeze(1)  # [B, 1]
        mask_t = torch.tensor(mask, device=device).unsqueeze(1)  # [B, 1]
        return tokens, mask_t

    def forward(self, inputs: tp.Tuple[torch.Tensor, torch.Tensor]) -> ConditionType:
        tokens, mask = inputs
        embeds = self.embed(tokens)        # [B, 1, dim]
        embeds = self.output_proj(embeds)  # [B, 1, output_dim]
        embeds = embeds * mask.unsqueeze(-1)
        return embeds, mask


def load_audiogen_for_finetuning(
    n_species: int,
    device: str = "cuda",
    pretrained_name: str = AG_PRETRAINED,
) -> tp.Tuple[AudioGen, SpeciesConditioner]:
    logger.info(f"Loading pretrained AudioGen: {pretrained_name}")
    audiogen = AudioGen.get_pretrained(pretrained_name, device=device)
    lm = audiogen.lm

    output_dim = lm.dim
    species_cond = SpeciesConditioner(
        n_species, dim=output_dim, output_dim=output_dim
    )
    species_cond = species_cond.to(device)

    lm.condition_provider.conditioners = nn.ModuleDict(
        {"description": species_cond}
    )

    n_total = sum(p.numel() for p in lm.parameters())
    n_cond = sum(p.numel() for p in species_cond.parameters())
    logger.info(f"LM params: {n_total:,}  |  SpeciesConditioner params: {n_cond:,}")

    return audiogen, species_cond


def freeze_for_stage1(lm: LMModel) -> None:
    for param in lm.parameters():
        param.requires_grad = False
    for param in lm.condition_provider.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lm.parameters())
    logger.info(f"Stage 1 freeze: {trainable:,} trainable / {total:,} total")


def unfreeze_all(lm: LMModel) -> None:
    for param in lm.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in lm.parameters())
    logger.info(f"Full finetuning: {trainable:,} trainable params")


def make_species_conditions(species_ids: tp.List[int]) -> tp.List[ConditioningAttributes]:
    return [ConditioningAttributes(text={"description": str(sid)}) for sid in species_ids]


def save_lm_checkpoint(
    path,
    lm: LMModel,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    n_species: int,
    ebird_to_id: dict,
    val_loss: float = None,
    stage: int = None,
) -> None:
    torch.save(
        {
            "lm_state_dict": lm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "n_species": n_species,
            "ebird_to_id": ebird_to_id,
            "val_loss": val_loss,
            "stage": stage,
        },
        path,
    )


def load_lm_checkpoint(path, lm: LMModel, device="cpu") -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    lm.load_state_dict(ckpt["lm_state_dict"])
    return ckpt
