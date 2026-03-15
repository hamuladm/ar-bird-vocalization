from typing import Dict, List
import torch

from config import GatingConfig
from preprocessing.code_translator import BirdTranslator


class GatingStrategy:
    def __init__(self, config: GatingConfig = None, translator: BirdTranslator = None):
        self.config = config or GatingConfig()
        self.translator = translator or BirdTranslator()


    def passes_gates(
        self,
        metrics: Dict[str, torch.Tensor],
        idx: int = None,
        ground_truth_ebird: str = None,
    ) -> bool:
        top1 = metrics["top1_prob"][idx].item()
        top2 = metrics["top2_prob"][idx].item()
        predicted_xcl = int(metrics["top1_class"][idx].item())

        if top1 <= self.config.confidence_threshold:
            return False
        if top2 > 0 and top1 / top2 <= self.config.singularity_ratio:
            return False
        if not self.translator.check_consistency(ground_truth_ebird, predicted_xcl):
            return False
        return True


    def process_batch(
        self,
        metrics: Dict[str, torch.Tensor],
        ground_truth_labels: List = None,
    ) -> List[bool]:
        batch_size = metrics["top1_prob"].shape[0]
        return [self.passes_gates(metrics, idx=i, ground_truth_ebird=ground_truth_labels[i] if ground_truth_labels else None) for i in range(batch_size)]
