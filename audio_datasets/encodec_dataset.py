import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from models.audiogen import make_species_conditions


class EnCodecTokenDataset(Dataset):
    def __init__(self, split_dir, max_timesteps=None):
        self.split_dir = Path(split_dir)
        data = np.load(self.split_dir / "tokens.npz")
        self.codes = data["codes"]
        self.labels = data["labels"]
        self.max_timesteps = max_timesteps

        assert self.codes.ndim == 3, (
            f"Expected codes shape [N, K, T], got {self.codes.shape}"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        codes = self.codes[idx]
        label = int(self.labels[idx])

        if self.max_timesteps is not None and codes.shape[1] > self.max_timesteps:
            codes = codes[:, : self.max_timesteps]

        return {
            "codes": torch.from_numpy(codes.copy()).long(),
            "species_id": label,
        }


def make_encodec_collate_fn(special_token_id):
    def encodec_collate_fn(batch):
        codes_list = [item["codes"] for item in batch]
        species_ids = [item["species_id"] for item in batch]

        K = codes_list[0].shape[0]
        max_t = max(c.shape[1] for c in codes_list)

        padded = torch.full((len(batch), K, max_t), special_token_id, dtype=torch.long)
        for i, c in enumerate(codes_list):
            padded[i, :, : c.shape[1]] = c

        conditions = make_species_conditions(species_ids)

        return {
            "codes": padded,
            "conditions": conditions,
        }
    return encodec_collate_fn
