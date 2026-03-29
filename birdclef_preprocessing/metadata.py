import csv
import json
import soundfile as sf
from collections import Counter
from pathlib import Path

from preprocessing.pipeline import chunk_recording


def load_aves_labels(taxonomy_path):
    with open(taxonomy_path) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        return {
            row["primary_label"].strip()
            for row in reader
            if row["class_name"].strip() == "Aves"
        }


def load_backbone_species(ebird_to_id_path):
    """eBird codes the backbone was trained on (keys of ebird_to_id.json)."""
    with open(ebird_to_id_path) as f:
        data = json.load(f)
    return set(data.keys())


def load_train_csv(train_csv_path, valid_species):
    rows = []
    with open(train_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["primary_label"].strip() in valid_species:
                rows.append(row)
    return rows


def build_segments(
    data_dir,
    ebird_to_id_path,
    chunk_sec,
    min_chunk_sec,
    min_samples_per_class,
):
    data_dir = Path(data_dir)
    taxonomy_path = data_dir / "taxonomy.csv"
    train_csv_path = data_dir / "train.csv"
    audio_dir = data_dir / "train_audio"

    aves = load_aves_labels(taxonomy_path)
    backbone = load_backbone_species(ebird_to_id_path)
    valid_species = aves & backbone

    print(
        f"Species overlap (BirdCLEF Aves ∩ backbone): {len(valid_species)} "
        f"(aves={len(aves)}, backbone={len(backbone)})"
    )

    recordings = load_train_csv(train_csv_path, valid_species)
    print(f"Recordings after species filter: {len(recordings)}")

    segments = []
    skipped = 0
    for row in recordings:
        filepath = str(audio_dir / row["filename"])
        try:
            info = sf.info(filepath)
        except Exception:
            skipped += 1
            continue

        duration = info.duration
        if duration < min_chunk_sec:
            skipped += 1
            continue

        ebird_code = row["primary_label"].strip()
        for start, end in chunk_recording(duration, chunk_sec, min_chunk_sec):
            segments.append(
                {
                    "filepath": filepath,
                    "start": start,
                    "end": end,
                    "ebird_code": ebird_code,
                }
            )

    if skipped:
        print(f"Skipped {skipped} recordings (missing/too short)")
    print(f"Raw segments (before min_samples): {len(segments)}")

    counts = Counter(s["ebird_code"] for s in segments)
    allowed = {c for c, n in counts.items() if n >= min_samples_per_class}
    dropped_classes = len(counts) - len(allowed)
    filtered = [s for s in segments if s["ebird_code"] in allowed]
    print(
        f"After min_samples (>={min_samples_per_class}): {len(filtered)} segments, "
        f"{len(allowed)} classes (dropped {len(segments) - len(filtered)} segments, "
        f"{dropped_classes} classes below threshold)"
    )
    return filtered
