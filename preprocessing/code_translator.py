from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from datasets import load_dataset_builder

if TYPE_CHECKING:
    from preprocessing.taxonomy import TaxonomyMapper


class BirdTranslator:
    def __init__(self, load_metadata: bool = True):
        # xcl_idx -> ebird_code
        self.xcl_map: Dict[int, str] = {}
        # ebird_code -> xcm_idx
        self.xcm_map: Dict[str, int] = {}

        self._xcm_idx_to_ebird: Dict[int, str] = {}
        self._ebird_to_xcl_idx: Dict[str, int] = {}
        if load_metadata:
            self._load_mappings()

    def _load_mappings(self) -> None:
        builder_xcl = load_dataset_builder("DBD-research-group/BirdSet", "XCL", trust_remote_code=True)
        xcl_codes = builder_xcl.info.features["ebird_code"].names
        self.xcl_map = {idx: code for idx, code in enumerate(xcl_codes)}
        self._ebird_to_xcl_idx = {code: idx for idx, code in self.xcl_map.items()}

        builder_xcm = load_dataset_builder("DBD-research-group/BirdSet", "XCM", trust_remote_code=True)
        xcm_codes = builder_xcm.info.features["ebird_code"].names
        self.xcm_map = {code: idx for idx, code in enumerate(xcm_codes)}
        self._xcm_idx_to_ebird = {idx: code for code, idx in self.xcm_map.items()}

    def xcl2ebird(self, xcl_idx: int) -> Optional[str]:
        return self.xcl_map.get(xcl_idx)

    def check_consistency(self, ground_truth_ebird: str, predicted_xcl: int) -> bool:
        return self.xcl2ebird(predicted_xcl) == ground_truth_ebird

    def check_topk_consistency(self, ground_truth_ebird: str, predicted_xcl_indices: List[int]) -> bool:
        return any(self.xcl2ebird(idx) == ground_truth_ebird for idx in predicted_xcl_indices)

    def check_family_consistency(self, ground_truth_ebird: str, predicted_xcm: int, taxonomy: TaxonomyMapper) -> bool:
        predicted_ebird = self.xcl2ebird(predicted_xcm)
        if predicted_ebird is None:
            return False
        return taxonomy.same_family(ground_truth_ebird, predicted_ebird)
