import torch
from torch import nn

from audiocraft.models import AudioGen
from audiocraft.modules.conditioners import BaseConditioner, ConditioningAttributes

from config import AG_PRETRAINED


class SpeciesConditioner(BaseConditioner):
    def __init__(self, n_species, dim, output_dim):
        super().__init__(dim, output_dim)
        self.embed = nn.Embedding(n_species + 1, dim)
        self.n_species = n_species

    def tokenize(self, x):
        device = self.embed.weight.device
        ids, mask = [], []
        for s in x:
            if s is not None:
                ids.append(int(s))
                mask.append(1)
            else:
                ids.append(self.n_species)
                mask.append(0)
        tokens = torch.tensor(ids, device=device).unsqueeze(1)
        mask_t = torch.tensor(mask, device=device).unsqueeze(1)
        return tokens, mask_t

    def forward(self, inputs):
        tokens, mask = inputs
        embeds = self.embed(tokens)
        embeds = self.output_proj(embeds)
        embeds = embeds * mask.unsqueeze(-1)
        return embeds, mask


def load_audiogen(n_species, device="cuda", pretrained_name=AG_PRETRAINED):
    audiogen = AudioGen.get_pretrained(pretrained_name, device=device)
    lm = audiogen.lm

    output_dim = lm.dim
    species_cond = SpeciesConditioner(n_species, dim=output_dim, output_dim=output_dim)
    species_cond = species_cond.to(device)

    lm.condition_provider.conditioners = nn.ModuleDict({"description": species_cond})

    return audiogen, species_cond


def freeze_for_stage1(lm):
    for param in lm.parameters():
        param.requires_grad = False
    for param in lm.condition_provider.parameters():
        param.requires_grad = True


def unfreeze_all(lm):
    for param in lm.parameters():
        param.requires_grad = True


def make_species_conditions(species_ids):
    return [
        ConditioningAttributes(text={"description": str(sid)}) for sid in species_ids
    ]


def save_lm_checkpoint(
    path,
    lm,
    optimizer,
    scheduler,
    epoch,
    global_step,
    n_species,
    ebird_to_id,
    val_loss=None,
    stage=None,
):
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


def load_lm_checkpoint(path, lm, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    lm.load_state_dict(ckpt["lm_state_dict"])
    return ckpt
