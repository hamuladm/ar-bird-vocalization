import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from config import MAX_SEQ_LEN
from models.gpt2 import CLASS_TOKEN_OFFSET, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


class SNACTokenDataset(Dataset):
    def __init__(self, split_dir, max_seq_len=MAX_SEQ_LEN):
        self.split_dir = Path(split_dir)
        data = np.load(self.split_dir / "tokens.npz")
        self.codes = data["codes"]
        self.labels = data["labels"]
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        flat_codes = self.codes[idx]

        if len(flat_codes) > self.max_seq_len:
            flat_codes = flat_codes[: self.max_seq_len]

        cls_token = CLASS_TOKEN_OFFSET + label
        tokens = np.concatenate([[cls_token, BOS_TOKEN], flat_codes, [EOS_TOKEN]])

        return {
            "input_ids": torch.from_numpy(tokens).long(),
            "label": label,
        }


def snac_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["label"] for item in batch]

    max_len = max(len(ids) for ids in input_ids)

    padded_ids = torch.full((len(batch), max_len), PAD_TOKEN, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, ids in enumerate(input_ids):
        padded_ids[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = 1

    return {
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor(labels, dtype=torch.long),
    }
