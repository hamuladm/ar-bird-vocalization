import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from config import (
    CHUNK_LENGTH,
    MIN_CHUNK_SEC,
    VAL_RATIO,
    TEST_RATIO,
    SEED,
    BC_DATA_DIR,
    BC_SEGMENT_DIR,
    BC_EBIRD_TO_ID_PATH,
    BC_MIN_SAMPLES_PER_CLASS,
)
from preprocessing.pipeline import split_segments, save_segments
from birdclef_preprocessing.metadata import build_segments


def main():
    parser = argparse.ArgumentParser(
        description="BirdCLEF segments: backbone ∩ Aves species, then min_samples per class."
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
    args = parser.parse_args()

    segments = build_segments(
        args.data_dir,
        args.ebird_to_id,
        args.chunk_sec,
        args.min_chunk_sec,
        args.min_samples,
    )

    splits = split_segments(segments, args.val_ratio, args.test_ratio, args.seed)
    save_segments(args.output_dir, splits)


if __name__ == "__main__":
    main()
