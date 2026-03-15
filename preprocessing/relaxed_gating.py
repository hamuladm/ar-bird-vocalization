from typing import Dict, List

import torch

from config import RelaxedGatingConfig
from preprocessing.code_translator import BirdTranslator
from preprocessing.taxonomy import TaxonomyMapper


class RelaxedGatingStrategy:
    def __init__(
        self,
        config: RelaxedGatingConfig = None,
        translator: BirdTranslator = None,
        taxonomy: TaxonomyMapper = None,
    ):
        self.config = config or RelaxedGatingConfig()
        self.translator = translator or BirdTranslator()
        self.taxonomy = taxonomy

    def passes_gates(
        self,
        metrics: Dict[str, torch.Tensor],
        idx: int,
        ground_truth_ebird: str = None,
    ) -> bool:
        topk_classes = metrics["topk_classes"][idx].tolist()
        entropy = metrics["entropy"][idx].item()
        predicted_xcl = int(metrics["top1_class"][idx].item())

        if not self.translator.check_topk_consistency(ground_truth_ebird, topk_classes):
            return False

        if entropy >= self.config.max_entropy:
            return False

        if not self.translator.check_family_consistency(ground_truth_ebird, predicted_xcl, self.taxonomy):
            return False

        return True

    def process_batch(
        self,
        metrics: Dict[str, torch.Tensor],
        ground_truth_labels: List = None,
    ) -> List[bool]:
        batch_size = metrics["top1_prob"].shape[0]
        return [self.passes_gates(metrics, idx=i, ground_truth_ebird=(ground_truth_labels[i] if ground_truth_labels else None)) for i in range(batch_size)]
