import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
from datasets import load_dataset

from utils import setup_logger, load_and_pad_batch, get_all_segments
from config import PipelineConfig, GatingConfig
from preprocessing import BirdClassifier, GatingStrategy, BirdTranslator


logger = setup_logger("preprocessing_pipeline")


def classify_and_gate_segments(
    segments_info: List[Dict],
    judge: BirdClassifier,
    gating: GatingStrategy,
    batch_size: int,
) -> Dict:
    passed_segments = []
    failed_segments = []
    passed_probs = []

    for batch_start in tqdm(range(0, len(segments_info), batch_size), desc="Evaluating (raw)"):

        batch_end = min(batch_start + batch_size, len(segments_info))
        batch_info = segments_info[batch_start : batch_end]
        tasks = [(s["filepath"], s["start"], s["end"]) for s in batch_info]

        audio_batch = load_and_pad_batch(tasks)
        metrics = judge.evaluate(audio_batch)
        batch_labels = [s["ground_truth_ebird"] for s in batch_info]
        passed_mask = gating.process_batch(metrics, ground_truth_labels=batch_labels)

        for i, (seg_info, passed) in enumerate(zip(batch_info, passed_mask)):
            seg_copy = seg_info.copy()
            seg_copy["top1_prob"] = float(metrics["top1_prob"][i].item())
            seg_copy["top2_prob"] = float(metrics["top2_prob"][i].item())
            seg_copy["predicted_class"] = int(metrics["top1_class"][i].item())
            if passed:
                passed_segments.append(seg_copy)
                passed_probs.append(seg_copy["top1_prob"])
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
        "mean_confidence": float(np.mean(passed_probs)) if passed_probs else 0.0
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
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="Gate 1: Minimum top-1 probability (default: 0.7)")
    parser.add_argument("--singularity-ratio", type=float, default=2.0,
                        help="Gate 2: Minimum top1/top2 ratio (default: 2.0)")
    parser.add_argument("--no-label-match", action="store_true",
                        help="Disable Gate 3 (label verification)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for inference (default: 64)")
    parser.add_argument("--device", default="cuda",
                        help="Device for inference (default: cuda)")
    parser.add_argument("--output-dir", default="data/filtered",
                        help="Output directory for filtered data")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of segments (for testing)")
    parser.add_argument("--create-splits", action="store_true",
                        help="Create train/val/test splits after filtering")
    parser.add_argument("--splits-only", action="store_true",
                        help="Only create splits from existing filtered data")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation set ratio (default: 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Test set ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()

    if args.splits_only:
        logger.info("Creating splits from existing filtered data...")
        create_filtered_splits(
            filtered_dir=args.output_dir,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        gating_config = GatingConfig(
            confidence_threshold=args.confidence_threshold,
            singularity_ratio=args.singularity_ratio,
            require_label_match=not args.no_label_match,
        )

        config = PipelineConfig(
            gating=gating_config,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        logger.info(f"Running pipeline with: confidence>{args.confidence_threshold}, "
                    f"singularity>{args.singularity_ratio}, "
                    f"label_match={not args.no_label_match}")

        results = filter_segments(config)

        if args.create_splits:
            logger.info("Creating train/val/test splits...")
            create_filtered_splits(
                filtered_dir=args.output_dir,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )
