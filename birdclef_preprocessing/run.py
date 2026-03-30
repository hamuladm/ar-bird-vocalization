import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging

from config import (
    CHUNK_LENGTH,
    MIN_CHUNK_SEC,
    VAL_RATIO,
    TEST_RATIO,
    SEED,
    DEVICE,
    BC_DATA_DIR,
    BC_SEGMENT_DIR,
    BC_EBIRD_TO_ID_PATH,
    BC_MIN_SAMPLES_PER_CLASS,
    BC_GATING_MIN_TOP1_PROB,
    BC_GATING_MAX_ENTROPY,
    BC_GATING_BATCH_SIZE,
    EVAL_MODEL_CHECKPOINT,
)
from preprocessing.pipeline import split_segments, save_segments
from birdclef_preprocessing.metadata import (
    build_candidate_segments,
    build_segments,
    filter_min_samples_per_class,
)
from birdclef_preprocessing.gating import gate_segments
from judge import BirdClassifier

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "BirdCLEF segments: Aves ∩ backbone, optional ConvNeXT 3-stage gating, "
            "then min_samples per class."
        )
    )
    parser.add_argument("--data-dir", type=str, default=str(BC_DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(BC_SEGMENT_DIR))
    parser.add_argument(
        "--ebird-to-id",
        type=str,
        default=str(BC_EBIRD_TO_ID_PATH),
        help="Backbone ebird_to_id.json (species vocabulary)",
    )
    parser.add_argument("--chunk-sec", type=float, default=CHUNK_LENGTH)
    parser.add_argument("--min-chunk-sec", type=float, default=MIN_CHUNK_SEC)
    parser.add_argument("--min-samples", type=int, default=BC_MIN_SAMPLES_PER_CLASS)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--no-gating",
        action="store_true",
        help="Skip embedder gating; min_samples applies to all chunks (legacy behavior).",
    )
    parser.add_argument(
        "--min-top1-prob",
        type=float,
        default=BC_GATING_MIN_TOP1_PROB,
        help="Keep segment if top-1 probability is strictly greater than this",
    )
    parser.add_argument(
        "--max-entropy",
        type=float,
        default=BC_GATING_MAX_ENTROPY,
        help="Keep segment if entropy is strictly less than this",
    )
    parser.add_argument(
        "--gating-batch-size",
        type=int,
        default=BC_GATING_BATCH_SIZE,
        help="Batch size for ConvNeXT forward during gating",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=EVAL_MODEL_CHECKPOINT,
        help="Hugging Face id or local path for BirdSet ConvNeXT",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="Torch device (default: device from config.yaml)",
    )
    parser.add_argument(
        "--rewrite-hf-paths",
        action="store_true",
        help="Rewrite /workspace/.hf_home/ prefixes in segment paths (for relocated caches)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less logging (warnings only); tqdm bars are unchanged",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
        force=True,
    )

    logger.info(
        "BirdCLEF preprocessing: data_dir=%s output_dir=%s ebird_to_id=%s",
        args.data_dir,
        args.output_dir,
        args.ebird_to_id,
    )
    logger.info(
        "Splits: val_ratio=%.3f test_ratio=%.3f seed=%s min_samples=%s",
        args.val_ratio,
        args.test_ratio,
        args.seed,
        args.min_samples,
    )

    if args.no_gating:
        logger.info("Mode: no gating (candidates → min_samples → split)")
        segments = build_segments(
            args.data_dir,
            args.ebird_to_id,
            args.chunk_sec,
            args.min_chunk_sec,
            args.min_samples,
        )
    else:
        logger.info("Mode: 3-stage gating then min_samples")
        logger.info("Building candidate segments (species overlap + chunking)")
        candidates = build_candidate_segments(
            args.data_dir,
            args.ebird_to_id,
            args.chunk_sec,
            args.min_chunk_sec,
        )
        device = args.device or DEVICE
        logger.info("Loading classifier: %s on %s", args.checkpoint, device)
        classifier = BirdClassifier(checkpoint=args.checkpoint, device=device)
        segments = gate_segments(
            candidates,
            classifier,
            min_top1_prob=args.min_top1_prob,
            max_entropy=args.max_entropy,
            batch_size=max(1, args.gating_batch_size),
            rewrite_hf_paths=args.rewrite_hf_paths,
        )
        segments = filter_min_samples_per_class(segments, args.min_samples)

    splits = split_segments(segments, args.val_ratio, args.test_ratio, args.seed)
    save_segments(args.output_dir, splits, backbone_ebird_to_id_path=args.ebird_to_id)


if __name__ == "__main__":
    main()
