import json
import numpy as np
from pathlib import Path
from collections import Counter
from datasets import load_dataset

from config import (
    CHUNK_LENGTH,
    MIN_CHUNK_SEC,
    MIN_SAMPLES_PER_CLASS,
    VAL_RATIO,
    TEST_RATIO,
    SEED,
    SEGMENT_DIR,
)


def count_per_class(dataset):
    ebird_names = dataset.features["ebird_code"].names
    counts = Counter()
    for item in dataset:
        counts[ebird_names[item["ebird_code"]]] += 1
    return counts


def chunk_recording(length, chunk_sec, min_chunk_sec):
    chunks = []
    start = 0.0
    while start + min_chunk_sec <= length:
        end = min(start + chunk_sec, length)
        if end - start < min_chunk_sec:
            break
        if end - start < chunk_sec:
            break
        chunks.append((start, end))
        start = end
    return chunks


def build_segments(dataset, min_samples_per_class, chunk_sec, min_chunk_sec):
    ebird_names = dataset.features["ebird_code"].names
    counts = count_per_class(dataset)
    allowed = {c for c, n in counts.items() if n >= min_samples_per_class}

    print(
        f"Classes with >= {min_samples_per_class} samples: {len(allowed)} / {len(counts)}"
    )

    segments = []
    for item in dataset:
        ebird_code = ebird_names[item["ebird_code"]]
        if ebird_code not in allowed:
            continue

        length = item.get("length")
        if length is None or length < min_chunk_sec:
            continue

        filepath = item["filepath"]
        for start, end in chunk_recording(float(length), chunk_sec, min_chunk_sec):
            segments.append(
                {
                    "filepath": filepath,
                    "start": start,
                    "end": end,
                    "ebird_code": ebird_code,
                }
            )

    return segments


def split_segments(segments, val_ratio, test_ratio, seed):
    rng = np.random.default_rng(seed)
    rng.shuffle(segments)

    n = len(segments)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": segments[: n - n_val - n_test],
        "val": segments[n - n_val - n_test : n - n_test],
        "test": segments[n - n_test :],
    }


def save_segments(output_dir, splits):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in splits.items():
        path = output_dir / f"{name}_segments.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {name}: {len(data)} segments -> {path}")

    ebird_codes = sorted(
        {s["ebird_code"] for split_data in splits.values() for s in split_data}
    )
    ebird_to_id = {code: i for i, code in enumerate(ebird_codes)}
    with open(output_dir / "ebird_to_id.json", "w") as f:
        json.dump(ebird_to_id, f, indent=2)
    print(f"Saved ebird_to_id: {len(ebird_to_id)} classes")


def preprocess_xcm(
    min_samples_per_class=MIN_SAMPLES_PER_CLASS,
    chunk_sec=CHUNK_LENGTH,
    min_chunk_sec=MIN_CHUNK_SEC,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
    seed=SEED,
    output_dir=SEGMENT_DIR,
):
    print("Loading XCM dataset...")
    dataset = load_dataset(
        "DBD-research-group/BirdSet", "XCM", split="train", trust_remote_code=True
    )
    print(f"Total recordings: {len(dataset)}")

    segments = build_segments(dataset, min_samples_per_class, chunk_sec, min_chunk_sec)
    print(f"Total {chunk_sec}s segments: {len(segments)}")

    splits = split_segments(segments, val_ratio, test_ratio, seed)
    save_segments(output_dir, splits)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_CLASS)
    parser.add_argument("--chunk-sec", type=float, default=CHUNK_LENGTH)
    parser.add_argument("--min-chunk-sec", type=float, default=MIN_CHUNK_SEC)
    parser.add_argument("--output-dir", type=str, default=str(SEGMENT_DIR))
    args = parser.parse_args()

    preprocess_xcm(
        min_samples_per_class=args.min_samples,
        chunk_sec=args.chunk_sec,
        min_chunk_sec=args.min_chunk_sec,
        output_dir=args.output_dir,
    )
