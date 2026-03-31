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
    BC_XCM_HF_PATH,
    BC_XCM_HF_NAME,
    BC_XCM_AUDIO_ROOT,
    BC_XCM_METADATA_ONLY,
    BC_XCM_SHUFFLE_BUFFER_SIZE,
    BC_XCM_MAX_STREAM_PASSES,
    EVAL_MODEL_CHECKPOINT,
    EVAL_SAMPLE_RATE,
)
from preprocessing.pipeline import split_segments, save_segments
from birdclef_preprocessing.metadata import (
    build_candidate_segments,
    build_segments,
    filter_min_samples_per_class,
)
from birdclef_preprocessing.gating import gate_segments
from birdclef_preprocessing.xcm_enrich import (
    enrich_segments_with_xcm,
    subset_ebird_to_id_for_classes,
    train_quota_seconds_per_class,
)
from judge import BirdClassifier

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "BirdCLEF segments: Aves ∩ backbone, optional ConvNeXT 3-stage gating, "
            "min_samples per class, optional BirdSet XCM enrichment (event windows), "
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
        help="After BirdCLEF steps, add XCM segments (see birdclef.xcm_enrich in config)",
    )
    parser.add_argument(
        "--xcm-audio-root",
        type=str,
        default=None,
        help="Directory to join with XCM filepath when HF audio path is missing",
    )
    parser.add_argument(
        "--xcm-hf-path",
        type=str,
        default=None,
        help="Override Hugging Face dataset path for XCM",
    )
    parser.add_argument(
        "--xcm-hf-name",
        type=str,
        default=None,
        help="Override Hugging Face config / subset name (default: XCM)",
    )
    parser.add_argument(
        "--xcm-gate",
        action="store_true",
        help="Run ConvNeXT gating on XCM segments only before merging",
    )
    parser.add_argument(
        "--xcm-decode-audio",
        action="store_true",
        help="Load full XCM split with audio (huge download). Default is metadata-only streaming.",
    )
    parser.add_argument(
        "--xcm-shuffle-buffer",
        type=int,
        default=None,
        help="Iterable shuffle buffer for metadata-only mode (default: config)",
    )
    parser.add_argument(
        "--xcm-max-passes",
        type=int,
        default=None,
        help="Max streaming passes over XCM if caps not met (default: config)",
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
        hf_path = args.xcm_hf_path or BC_XCM_HF_PATH
        hf_name = args.xcm_hf_name or BC_XCM_HF_NAME
        audio_root = args.xcm_audio_root or BC_XCM_AUDIO_ROOT
        class_codes = {s["ebird_code"] for s in segments}
        ebird_subset = subset_ebird_to_id_for_classes(args.ebird_to_id, class_codes)
        quota_sec = train_quota_seconds_per_class(
            segments,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            chunk_sec=args.chunk_sec,
            ebird_to_id=ebird_subset,
        )
        logger.info(
            "XCM enrichment: quota = n_train×chunk (%.1fs); total target %.0fs; "
            "classes=%d; hf=%s/%s; audio_root=%s",
            args.chunk_sec,
            sum(quota_sec.values()),
            len(quota_sec),
            hf_path,
            hf_name,
            audio_root,
        )
        metadata_only = BC_XCM_METADATA_ONLY and not args.xcm_decode_audio
        shuffle_buf = (
            args.xcm_shuffle_buffer
            if args.xcm_shuffle_buffer is not None
            else BC_XCM_SHUFFLE_BUFFER_SIZE
        )
        max_passes = (
            args.xcm_max_passes
            if args.xcm_max_passes is not None
            else BC_XCM_MAX_STREAM_PASSES
        )
        xcm_segs, _ = enrich_segments_with_xcm(
            quota_seconds=quota_sec,
            hf_path=hf_path,
            hf_name=hf_name,
            seed=args.seed,
            chunk_sec=args.chunk_sec,
            min_chunk_sec=args.min_chunk_sec,
            audio_root=audio_root,
            sample_rate=EVAL_SAMPLE_RATE,
            metadata_only=metadata_only,
            shuffle_buffer_size=max(2, shuffle_buf),
            max_stream_passes=max(1, max_passes),
        )
        if args.xcm_gate and xcm_segs:
            dev = args.device or DEVICE
            logger.info("XCM 3-stage gating: %s on %s", args.checkpoint, dev)
            xcm_classifier = BirdClassifier(
                checkpoint=args.checkpoint, device=dev
            )
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
