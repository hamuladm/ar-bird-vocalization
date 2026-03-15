import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Union
from datasets import load_dataset
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.audio_utils import SegmentDataset, get_all_segments
from utils.logging_utils import setup_logger
from config import PipelineConfig, GatingConfig, RelaxedPipelineConfig, RelaxedGatingConfig
from preprocessing.judge import BirdClassifier
from preprocessing.gating import GatingStrategy
from preprocessing.relaxed_gating import RelaxedGatingStrategy
from preprocessing.code_translator import BirdTranslator
from preprocessing.taxonomy import TaxonomyMapper


logger = setup_logger("preprocessing_pipeline")


def classify_and_gate_segments(
    segments_info: List[Dict],
    judge: BirdClassifier,
    gating: Union[GatingStrategy, RelaxedGatingStrategy],
    batch_size: int,
    num_workers: int = 4,
    top_k: int = 2,
) -> Dict:
    passed_segments = []
    failed_segments = []
    passed_probs = []
    passed_entropies = []

    dataset = SegmentDataset(segments_info)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    for audio_batch, indices in tqdm(loader, desc="Evaluating (raw)"):
        metrics = judge.evaluate(audio_batch, top_k=top_k)
        batch_info = [segments_info[i] for i in indices]
        batch_labels = [s["ground_truth_ebird"] for s in batch_info]
        passed_mask = gating.process_batch(metrics, ground_truth_labels=batch_labels)

        for i, (seg_info, passed) in enumerate(zip(batch_info, passed_mask)):
            seg_copy = seg_info.copy()
            seg_copy["top1_prob"] = float(metrics["top1_prob"][i].item())
            seg_copy["top2_prob"] = float(metrics["top2_prob"][i].item())
            seg_copy["entropy"] = float(metrics["entropy"][i].item())
            seg_copy["predicted_class"] = int(metrics["top1_class"][i].item())
            if passed:
                passed_segments.append(seg_copy)
                passed_probs.append(seg_copy["top1_prob"])
                passed_entropies.append(seg_copy["entropy"])
            else:
                failed_segments.append(seg_copy)

    total = len(segments_info)
    num_passed = len(passed_segments)
    return {
        "passed_segments": passed_segments,
        "failed_segments": failed_segments,
        "total": total,
        "num_passed": num_passed,
        "retention_rate": num_passed / total if total > 0 else 0,
        "mean_confidence": float(np.mean(passed_probs)) if passed_probs else 0.0,
        "mean_entropy": float(np.mean(passed_entropies)) if passed_entropies else 0.0,
    }


def count_recordings_per_class(dataset) -> Dict[str, int]:
    ebird_names = dataset.features["ebird_code"].names
    counts: Dict[str, int] = {}
    for item in dataset:
        code = ebird_names[item["ebird_code"]]
        counts[code] = counts.get(code, 0) + 1
    return counts


def scan_dataset_segments(dataset, min_samples_per_class: int) -> List[Dict]:
    ebird_names = dataset.features["ebird_code"].names
    counts = count_recordings_per_class(dataset)
    allowed_codes = {c for c, n in counts.items() if n >= min_samples_per_class}
    logger.info(f"Classes with >={min_samples_per_class} samples: {len(allowed_codes)} / {len(counts)}")

    segments_info = []
    for item_idx, item in enumerate(tqdm(dataset, desc="Scanning dataset")):
        ebird_code = ebird_names[item["ebird_code"]]
        if ebird_code not in allowed_codes:
            continue
        segments = get_all_segments(item)
        for seg_idx, (start, end) in enumerate(segments):
            segments_info.append({
                "item_idx": item_idx,
                "segment_idx": seg_idx,
                "filepath": item["filepath"],
                "start": start,
                "end": end,
                "ebird_code": ebird_code,
                "ground_truth_ebird": ebird_code,
            })

    return segments_info


def save_filter_results(
    output_dir: Path,
    stats: Dict,
    passed_segments: List[Dict],
    failed_segments: List[Dict],
) -> None:
    with open(output_dir / "filter_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    with open(output_dir / "passed_segments.json", "w") as f:
        json.dump(passed_segments, f, indent=2)
    with open(output_dir / "failed_segments.json", "w") as f:
        json.dump(failed_segments, f, indent=2)


def filter_segments(config: PipelineConfig = None) -> Dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing pipeline...")
    judge = BirdClassifier(checkpoint=config.model_checkpoint, device=config.device)
    translator = BirdTranslator(load_metadata=True) if config.gating.require_label_match else None
    gating = GatingStrategy(config=config.gating, translator=translator)

    full_dataset = load_dataset("DBD-research-group/BirdSet", "XCM", split="train",trust_remote_code=True)

    segments_info = scan_dataset_segments(full_dataset, config.min_samples_per_class)

    results = classify_and_gate_segments(
        segments_info=segments_info,
        judge=judge,
        gating=gating,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    stats = {
        "total_segments": results["total"],
        "passed_segments": results["num_passed"],
        "failed_segments": results["total"] - results["num_passed"],
        "retention_rate": results["retention_rate"],
        "mean_confidence": results["mean_confidence"],
        "min_samples_per_class": config.min_samples_per_class,
        "config": {
            "confidence_threshold": config.gating.confidence_threshold,
            "singularity_ratio": config.gating.singularity_ratio,
            "require_label_match": config.gating.require_label_match,
        },
    }

    save_filter_results(output_dir, stats, results["passed_segments"], results["failed_segments"])

    return {
        "passed": results["passed_segments"],
        "failed": results["failed_segments"],
        "stats": stats,
    }


def filter_segments_relaxed(config: RelaxedPipelineConfig = None) -> Dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing relaxed pipeline...")
    judge = BirdClassifier(checkpoint=config.model_checkpoint, device=config.device)
    translator = BirdTranslator(load_metadata=True)
    taxonomy = TaxonomyMapper(cache_path=config.taxonomy_cache)
    gating = RelaxedGatingStrategy(config=config.gating, translator=translator, taxonomy=taxonomy)

    full_dataset = load_dataset("DBD-research-group/BirdSet", "XCM", split="train", trust_remote_code=True)
    segments_info = scan_dataset_segments(full_dataset, config.min_samples_per_class)

    results = classify_and_gate_segments(
        segments_info=segments_info,
        judge=judge,
        gating=gating,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        top_k=config.gating.top_k,
    )

    stats = {
        "total_segments": results["total"],
        "passed_segments": results["num_passed"],
        "failed_segments": results["total"] - results["num_passed"],
        "retention_rate": results["retention_rate"],
        "mean_confidence": results["mean_confidence"],
        "mean_entropy": results["mean_entropy"],
        "min_samples_per_class": config.min_samples_per_class,
        "config": {
            "top_k": config.gating.top_k,
            "max_entropy": config.gating.max_entropy,
        },
    }

    save_filter_results(output_dir, stats, results["passed_segments"], results["failed_segments"])

    return {
        "passed": results["passed_segments"],
        "failed": results["failed_segments"],
        "stats": stats,
    }


def save_splits(filtered_dir: Path, splits: Dict[str, List[Dict]]) -> None:
    for split_name, split_data in splits.items():
        split_path = filtered_dir / f"{split_name}_segments.json"
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=2)
        logger.info(f"Saved {split_name} split to {split_path}")


def create_filtered_splits(
    filtered_dir: str = "data/filtered",
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    filtered_dir = Path(filtered_dir)
    passed_path = filtered_dir / "passed_segments.json"
    
    if not passed_path.exists():
        raise FileNotFoundError(
            f"Filtered segments not found at {passed_path}. "
            "Run filter_segments() first."
        )
    
    with open(passed_path) as f:
        segments = json.load(f)
    
    rng = np.random.default_rng(seed)
    rng.shuffle(segments)
    
    n_total = len(segments)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test
    
    splits = {
        "train": segments[:n_train],
        "val": segments[n_train:n_train + n_val],
        "test": segments[n_train + n_val:],
    }
    
    logger.info(f"Created splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    save_splits(filtered_dir, splits)
    
    return splits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["strict", "relaxed"], default="strict")
    parser.add_argument("--create-splits", action="store_true")
    parser.add_argument("--splits-only", action="store_true")
    args = parser.parse_args()

    if args.splits_only:
        config = RelaxedPipelineConfig() if args.mode == "relaxed" else PipelineConfig()
        logger.info(f"Creating splits from existing filtered data at {config.output_dir}...")
        create_filtered_splits(
            filtered_dir=config.output_dir,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.seed,
        )
    elif args.mode == "relaxed":
        config = RelaxedPipelineConfig()
        logger.info(
            f"Running RELAXED pipeline: top_k={config.gating.top_k}, "
            f"max_entropy={config.gating.max_entropy}, "
            f"min_samples_per_class={config.min_samples_per_class}"
        )
        results = filter_segments_relaxed(config)

        if args.create_splits:
            logger.info("Creating train/val/test splits...")
            create_filtered_splits(
                filtered_dir=config.output_dir,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
                seed=config.seed,
            )
    else:
        config = PipelineConfig()
        logger.info(
            f"Running STRICT pipeline: confidence>{config.gating.confidence_threshold}, "
            f"singularity>{config.gating.singularity_ratio}, "
            f"label_match={config.gating.require_label_match}"
        )
        results = filter_segments(config)

        if args.create_splits:
            logger.info("Creating train/val/test splits...")
            create_filtered_splits(
                filtered_dir=config.output_dir,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
                seed=config.seed,
            )
