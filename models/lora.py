import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = original.weight
        self.bias = original.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        device = self.weight.device
        dtype = self.weight.dtype
        self.lora_A = nn.Parameter(
            torch.zeros(rank, self.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora

    def merge_and_unload(self) -> nn.Linear:
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        merged.weight.data.copy_(
            self.weight.data + (self.lora_B @ self.lora_A) * self.scaling
        )
        if self.bias is not None:
            merged.bias.data.copy_(self.bias.data)
        return merged


TARGET_SUFFIXES = ("out_proj", "linear1", "linear2")


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    target_suffixes: tuple[str, ...] = TARGET_SUFFIXES,
) -> int:
    parents: dict[str, nn.Module] = {name: mod for name, mod in model.named_modules()}
    replaced = 0
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(full_name.endswith(s) for s in target_suffixes):
            continue
        if "." in full_name:
            parent_name, attr_name = full_name.rsplit(".", 1)
            parent = parents[parent_name]
        else:
            parent = model
            attr_name = full_name
        setattr(parent, attr_name, LoRALinear(module, rank, alpha))
        replaced += 1
    return replaced


def freeze_for_lora(lm: nn.Module) -> None:
    for p in lm.parameters():
        p.requires_grad = False
    for name, p in lm.named_parameters():
        if "lora_" in name or "condition_provider" in name:
            p.requires_grad = True


def merge_lora(model: nn.Module) -> int:
    parents: dict[str, nn.Module] = {name: mod for name, mod in model.named_modules()}
    merged = 0
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        if "." in full_name:
            parent_name, attr_name = full_name.rsplit(".", 1)
            parent = parents[parent_name]
        else:
            parent = model
            attr_name = full_name
        setattr(parent, attr_name, module.merge_and_unload())
        merged += 1
    return merged


def lora_summary(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    return {"total": total, "trainable": trainable, "lora": lora_params}
