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
    BC_XCM_ENRICH_ENABLED,
    BC_XCM_PASSED_SEGMENTS_JSON,
    BC_XCM_FINETUNE_EBIRD_TO_ID_JSON,
    BC_XCM_MIN_TOP1_PROB,
    BC_XCM_PRETRAIN_SEGMENT_DIR,
    BC_XCM_QUOTA_MODE,
    BC_XCM_EXTRA_SEGMENTS_PER_CLASS,
    EVAL_MODEL_CHECKPOINT,
)
from preprocessing.pipeline import split_segments, save_segments
from birdclef_preprocessing.metadata import (
    build_candidate_segments,
    build_segments,
    filter_min_samples_per_class,
)
from birdclef_preprocessing.gating import gate_segments
from birdclef_preprocessing.xcm_enrich import enrich_with_xcm_from_jsons
from birdclef_preprocessing.judge import BirdClassifier

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "BirdCLEF segments: Aves ∩ backbone, optional ConvNeXT 3-stage gating, "
            "min_samples per class, optional enrichment from relaxed passed_segments.json, "
            "then train/val/test split."
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
    parser.add_argument(
        "--xcm-enrich",
        action="store_true",
        help="After BirdCLEF steps, add segments from passed_segments.json (birdclef.xcm_enrich)",
    )
    parser.add_argument(
        "--xcm-finetune-ebird-json",
        type=str,
        default=None,
        help="Finetune ebird_to_id.json (quota allowlist; default: birdclef.xcm_enrich / segment_dir)",
    )
    parser.add_argument(
        "--xcm-passed-json",
        type=str,
        default=None,
        help="JSON list of {filepath, start, end, ebird_code, top1_prob, …} (default: config)",
    )
    parser.add_argument(
        "--xcm-min-top1-prob",
        type=float,
        default=None,
        help="Skip rows with top1_prob <= this (default: birdclef.xcm_enrich.min_top1_prob or off)",
    )
    parser.add_argument(
        "--xcm-quota-mode",
        type=str,
        choices=("birdclef_train", "fixed_per_class"),
        default=None,
        help="XCM quota: birdclef_train vs fixed_per_class (default: config)",
    )
    parser.add_argument(
        "--xcm-extra-segments-per-class",
        type=int,
        default=None,
        help="With fixed_per_class: XCM segments per finetune class (default: config)",
    )
    parser.add_argument(
        "--xcm-gate",
        action="store_true",
        help="Run ConvNeXT gating on enriched segments only before merging",
    )
    parser.add_argument(
        "--pretrain-segment-dir",
        type=str,
        default=str(BC_XCM_PRETRAIN_SEGMENT_DIR)
        if BC_XCM_PRETRAIN_SEGMENT_DIR
        else None,
        help=(
            "Pretrain segment dir with val_segments.json / test_segments.json. "
            "Recordings in those holdout sets are excluded from XCM enrichment to "
            "prevent data leakage (default: config birdclef.xcm_enrich.pretrain_segment_dir)"
        ),
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

    for s in segments:
        s.setdefault("source", "birdclef")

    use_xcm = args.xcm_enrich or BC_XCM_ENRICH_ENABLED
    if use_xcm:
        finetune_json = Path(
            args.xcm_finetune_ebird_json or BC_XCM_FINETUNE_EBIRD_TO_ID_JSON
        )
        passed_json = Path(args.xcm_passed_json or BC_XCM_PASSED_SEGMENTS_JSON)
        min_tp = (
            args.xcm_min_top1_prob
            if args.xcm_min_top1_prob is not None
            else BC_XCM_MIN_TOP1_PROB
        )
        q_mode = args.xcm_quota_mode or BC_XCM_QUOTA_MODE
        x_extra = (
            args.xcm_extra_segments_per_class
            if args.xcm_extra_segments_per_class is not None
            else BC_XCM_EXTRA_SEGMENTS_PER_CLASS
        )
        xcm_segs, _finetune_map, enrich_stats = enrich_with_xcm_from_jsons(
            segments,
            finetune_ebird_to_id_json=finetune_json,
            passed_segments_json=passed_json,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            chunk_sec=args.chunk_sec,
            min_chunk_sec=args.min_chunk_sec,
            rewrite_hf_paths=args.rewrite_hf_paths,
            min_top1_prob=min_tp,
            quota_mode=q_mode,
            xcm_extra_segments_per_class=x_extra,
            pretrain_segment_dir=args.pretrain_segment_dir,
        )
        quota_sec = enrich_stats.get("initial_quota_sec", {})
        logger.info(
            "Enrichment from passed_segments: quota_mode=%s; total quota target %.0fs; "
            "finetune_classes=%d (from %s); passed_json=%s; min_top1_prob=%s; "
            "xcm_extra_segments_per_class=%s",
            q_mode,
            sum(quota_sec.values()),
            enrich_stats.get("finetune_n_classes", 0),
            finetune_json,
            passed_json,
            min_tp,
            enrich_stats.get("xcm_extra_segments_per_class"),
        )
        if args.xcm_gate and xcm_segs:
            dev = args.device or DEVICE
            logger.info("XCM 3-stage gating: %s on %s", args.checkpoint, dev)
            xcm_classifier = BirdClassifier(checkpoint=args.checkpoint, device=dev)
            xcm_segs = gate_segments(
                xcm_segs,
                xcm_classifier,
                min_top1_prob=args.min_top1_prob,
                max_entropy=args.max_entropy,
                batch_size=max(1, args.gating_batch_size),
                rewrite_hf_paths=args.rewrite_hf_paths,
            )
        segments = segments + xcm_segs
        logger.info("Total segments after XCM merge: %d", len(segments))

    splits = split_segments(segments, args.val_ratio, args.test_ratio, args.seed)
    save_segments(args.output_dir, splits, backbone_ebird_to_id_path=args.ebird_to_id)
    logger.info("Done.")


if __name__ == "__main__":
    main()
