import csv
import json
import logging
import soundfile as sf
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

from preprocessing.pipeline import chunk_recording

logger = logging.getLogger(__name__)


def load_aves_labels(taxonomy_path):
    with open(taxonomy_path) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        return {
            row["primary_label"].strip()
            for row in reader
            if row["class_name"].strip() == "Aves"
        }


def load_backbone_species(ebird_to_id_path):
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


def build_candidate_segments(
    data_dir,
    ebird_to_id_path,
    chunk_sec,
    min_chunk_sec,
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
    logger.info("Recordings after species filter: %d", {len(recordings)})
    logger.info(
        "Chunking recordings: chunk_sec=%s min_chunk_sec=%s audio_dir=%s",
        chunk_sec,
        min_chunk_sec,
        audio_dir,
    )

    segments = []
    skipped = 0
    for row in tqdm(recordings, desc="Chunk recordings", unit="rec"):
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
    print(f"Candidate segments (before gating / min_samples): {len(segments)}")
    return segments


def filter_min_samples_per_class(segments, min_samples_per_class):
    counts = Counter(s["ebird_code"] for s in segments)
    allowed = {c for c, n in counts.items() if n >= min_samples_per_class}
    dropped_classes = len(counts) - len(allowed)
    filtered = [s for s in segments if s["ebird_code"] in allowed]
    below = sorted((c, n) for c, n in counts.items() if n < min_samples_per_class)
    if below:
        worst = below[:5]
        logger.info(
            "Classes below min_samples (showing up to 5 smallest): %s",
            ", ".join(f"{c}={n}" for c, n in worst),
        )
    print(
        f"After min_samples (>={min_samples_per_class}): {len(filtered)} segments, "
        f"{len(allowed)} classes (dropped {len(segments) - len(filtered)} segments, "
        f"{dropped_classes} classes below threshold)"
    )
    return filtered


def build_segments(
    data_dir,
    ebird_to_id_path,
    chunk_sec,
    min_chunk_sec,
    min_samples_per_class,
):
    candidates = build_candidate_segments(
        data_dir, ebird_to_id_path, chunk_sec, min_chunk_sec
    )
    return filter_min_samples_per_class(candidates, min_samples_per_class)
