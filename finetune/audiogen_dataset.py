import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List

from audiocraft.modules.conditioners import ConditioningAttributes

from finetune.audiogen_model import make_species_conditions


class EnCodecTokenDataset(Dataset):
    """Dataset for pre-encoded EnCodec tokens stored in ``tokens.npz``.

    Each file contains:
        * ``codes``  -- int array of shape ``[N, K=4, T]``
        * ``labels`` -- int array of shape ``[N]`` (species class IDs)

    Returns dicts with ``codes`` (``LongTensor [K, T]``) and ``species_id``
    (``int``).
    """

    def __init__(self, split_dir: str, max_timesteps: int | None = None):
        self.split_dir = Path(split_dir)
        data = np.load(self.split_dir / "tokens.npz")
        self.codes = data["codes"]    # [N, K, T]
        self.labels = data["labels"]  # [N]
        self.max_timesteps = max_timesteps

        assert self.codes.ndim == 3, (
            f"Expected codes shape [N, K, T], got {self.codes.shape}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        codes = self.codes[idx]  # [K, T]
        label = int(self.labels[idx])

        if self.max_timesteps is not None and codes.shape[1] > self.max_timesteps:
            codes = codes[:, : self.max_timesteps]

        return {
            "codes": torch.from_numpy(codes.copy()).long(),
            "species_id": label,
        }


ENCODEC_SPECIAL_TOKEN = 1024  # == card, AudioGen's masking/padding value


def encodec_collate_fn(batch: List[Dict]) -> Dict:
    """Collate variable-length EnCodec token sequences.

    Pads along the time axis with ``ENCODEC_SPECIAL_TOKEN`` (the value
    AudioGen's ``LMModel`` uses for masked / unknown positions).

    Returns a dict with:
        * ``codes``      -- ``LongTensor [B, K, T_max]``
        * ``conditions`` -- ``List[ConditioningAttributes]`` (length B)
    """
    codes_list = [item["codes"] for item in batch]
    species_ids = [item["species_id"] for item in batch]

    K = codes_list[0].shape[0]
    max_t = max(c.shape[1] for c in codes_list)

    padded = torch.full(
        (len(batch), K, max_t), ENCODEC_SPECIAL_TOKEN, dtype=torch.long
    )
    for i, c in enumerate(codes_list):
        padded[i, :, : c.shape[1]] = c

    conditions = make_species_conditions(species_ids)

    return {
        "codes": padded,
        "conditions": conditions,
    }
