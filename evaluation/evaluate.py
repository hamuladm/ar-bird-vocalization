import argparse
import json
from collections import Counter
from pathlib import Path
import random

import torch
from generate import load_generation_models, generate_audio_samples

from config import (
    DEVICE,
    FILTERED_DIR,
    MAX_SEQ_LEN,
    SNAC_GEN_TEMPERATURE,
    SNAC_GEN_TOP_K,
)
from utils.logging_utils import setup_logger
from evaluation.embeddings import (
    EvalEmbedder,
    extract_embeddings_from_directory,
    extract_embeddings_from_segments,
    extract_embeddings_from_arrays,
)
from evaluation.metrics import inception_score, compute_fad

logger = setup_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated bird vocalizations with IS and FAD."
    )

    seg_group = parser.add_argument_group("Segment + generation mode")
    seg_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )
    seg_group.add_argument(
        "--test-segments",
        type=str,
        default=None,
    )
    seg_group.add_argument(
        "--num-samples-per-class",
        type=int,
        default=10,
    )
    seg_group.add_argument(
        "--max-classes",
        type=int,
        default=1,
    )
    seg_group.add_argument(
        "--temperature",
        type=float,
        default=SNAC_GEN_TEMPERATURE,
    )
    seg_group.add_argument(
        "--top-k",
        type=int,
        default=SNAC_GEN_TOP_K,
    )

    dir_group = parser.add_argument_group("Directory mode (fallback)")
    dir_group.add_argument(
        "--generated-dir",
        type=str,
        default=None,
    )
    dir_group.add_argument(
        "--reference-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default="is,fad",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )
    return parser.parse_args()


def _run_segment_mode(args, metrics_to_compute: set) -> dict:
    compute_is_metric = "is" in metrics_to_compute
    compute_fad_metric = "fad" in metrics_to_compute
    device = torch.device(args.device)

    seg_path = Path(args.test_segments)
    with open(seg_path) as f:
        segments = json.load(f)

    class_counts = Counter(seg["ebird_code"] for seg in segments)
    logger.info(f"Found {len(class_counts)} unique classes in test set")

    gen_result = load_generation_models(
        checkpoint_path=args.checkpoint,
        device=device,
        filtered_dir=FILTERED_DIR,
        max_length=MAX_SEQ_LEN,
    )

    model = gen_result["model"]
    snac_model = gen_result["snac_model"]
    ebird_to_id = gen_result["ebird_to_id"]
    id_to_ebird = gen_result["id_to_ebird"]

    test_classes = sorted(class_counts.keys())
    known_classes = [c for c in test_classes if c in ebird_to_id]
    skipped = set(test_classes) - set(known_classes)
    if skipped:
        logger.warning(
            f"Skipping {len(skipped)} classes not in checkpoint: "
            f"{sorted(skipped)[:10]}{'...' if len(skipped) > 10 else ''}"
        )

    if args.max_classes is not None and args.max_classes < len(known_classes):
        random.seed(42)
        known_classes = sorted(random.sample(known_classes, args.max_classes))
        logger.info(f"Limited to {args.max_classes} classes")

    logger.info(
        f"Generating for {len(known_classes)} classes, "
        f"{args.num_samples_per_class} samples each"
    )

    class_ids = []
    for ebird_code in known_classes:
        cid = ebird_to_id[ebird_code]
        class_ids.extend([cid] * args.num_samples_per_class)

    gen_samples = generate_audio_samples(
        model=model,
        snac_model=snac_model,
        device=device,
        id_to_ebird=id_to_ebird,
        class_ids=class_ids,
        max_length=MAX_SEQ_LEN,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    gen_arrays = [audio for _, audio, _ in gen_samples]
    embedder = EvalEmbedder(device=args.device)
    gen_data = extract_embeddings_from_arrays(
        gen_arrays, embedder, batch_size=args.batch_size
    )

    eval_class_set = set(known_classes)
    ref_segments = [s for s in segments if s["ebird_code"] in eval_class_set]
    logger.info(f"Loading {len(ref_segments)} reference segments")

    ref_data = extract_embeddings_from_segments(
        ref_segments, embedder, batch_size=args.batch_size
    )

    results: dict = {}

    if compute_is_metric:
        is_value = inception_score(gen_data["probs"])
        gt_is_value = inception_score(ref_data["probs"])
        is_ratio = is_value / gt_is_value if gt_is_value > 0 else float("nan")
        results["inception_score"] = is_value
        results["gt_inception_score"] = gt_is_value
        results["is_ratio"] = is_ratio
        logger.info(f"Inception Score: {is_value:.4f}")
        logger.info(f"Ground Truth IS: {gt_is_value:.4f}")
        logger.info(f"IS ratio (gen/GT): {is_ratio:.4f}")

    if compute_fad_metric:
        fad_value = compute_fad(gen_data["features"], ref_data["features"])
        results["fad"] = fad_value
        logger.info(f"Frechet Audio Distance: {fad_value:.4f}")

    results["metadata"] = {
        "mode": "segment",
        "checkpoint": args.checkpoint,
        "test_segments": str(seg_path.resolve()),
        "num_classes": len(known_classes),
        "num_classes_skipped": len(skipped),
        "num_samples_per_class": args.num_samples_per_class,
        "num_generated": len(gen_samples),
        "num_reference": len(ref_segments),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "device": args.device,
    }

    return results


def _run_directory_mode(args, metrics_to_compute: set) -> dict:
    compute_is_metric = "is" in metrics_to_compute
    compute_fad_metric = "fad" in metrics_to_compute

    gen_dir = Path(args.generated_dir)
    ref_dir = Path(args.reference_dir)
    embedder = EvalEmbedder(device=args.device)

    logger.info(f"Extracting embeddings from generated audio: {gen_dir}")
    gen_data = extract_embeddings_from_directory(
        gen_dir, embedder, batch_size=args.batch_size
    )

    results: dict = {}

    if compute_is_metric:
        is_value = inception_score(gen_data["probs"])
        results["inception_score"] = is_value
        logger.info(f"Inception Score: {is_value:.4f}")

    if compute_fad_metric:
        logger.info(f"Extracting embeddings from reference audio: {ref_dir}")
        ref_data = extract_embeddings_from_directory(
            ref_dir, embedder, batch_size=args.batch_size
        )
        fad_value = compute_fad(gen_data["features"], ref_data["features"])
        results["fad"] = fad_value
        logger.info(f"Frechet Audio Distance: {fad_value:.4f}")

    results["metadata"] = {
        "mode": "directory",
        "generated_dir": str(gen_dir.resolve()),
        "reference_dir": str(ref_dir.resolve()) if compute_fad_metric else None,
        "num_generated": int(gen_data["probs"].shape[0]),
        "num_reference": int(ref_data["probs"].shape[0]) if compute_fad_metric else 0,
        "device": args.device,
    }

    return results


def main() -> None:
    args = parse_args()
    metrics_to_compute = {m.strip().lower() for m in args.metrics.split(",")}
    logger.info(f"Metrics: {', '.join(sorted(metrics_to_compute))}")
    logger.info(f"Device: {args.device}")

    segment_mode = args.checkpoint is not None and args.test_segments is not None
    directory_mode = args.generated_dir is not None

    if segment_mode:
        results = _run_segment_mode(args, metrics_to_compute)
        default_output = Path("eval_results.json")
    elif directory_mode:
        results = _run_directory_mode(args, metrics_to_compute)
        default_output = Path(args.generated_dir) / "eval_results.json"
    else:
        raise ValueError(
            "Provide either --checkpoint + --test-segments (segment mode) "
            "or --generated-dir (directory mode)."
        )

    output_path = Path(args.output) if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    if "inception_score" in results:
        print(f"  Inception Score (IS):         {results['inception_score']:.4f}")
    if "gt_inception_score" in results:
        print(f"  Ground Truth IS:              {results['gt_inception_score']:.4f}")
        print(f"  IS ratio (gen/GT):            {results['is_ratio']:.4f}")
    if "fad" in results:
        print(f"  Frechet Audio Distance (FAD): {results['fad']:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
