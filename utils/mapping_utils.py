import json
from pathlib import Path


def load_ebird_mapping(filtered_dir):
    filtered_dir = Path(filtered_dir)
    all_codes = set()
    for name in ["train", "val", "test"]:
        path = filtered_dir / f"{name}_segments.json"
        if path.exists():
            with open(path) as f:
                for s in json.load(f):
                    all_codes.add(s["ebird_code"])
    if not all_codes and (filtered_dir / "passed_segments.json").exists():
        with open(filtered_dir / "passed_segments.json") as f:
            for s in json.load(f):
                all_codes.add(s["ebird_code"])
    if not all_codes:
        raise FileNotFoundError(f"No segment files in {filtered_dir}")

    ebird_to_id = {c: i for i, c in enumerate(sorted(all_codes))}
    id_to_ebird = {i: c for c, i in ebird_to_id.items()}
    return ebird_to_id, id_to_ebird


def load_segments_and_mapping(filtered_dir, split, limit=None):
    filtered_dir = Path(filtered_dir)
    ebird_to_id, id_to_ebird = load_ebird_mapping(filtered_dir)

    path = filtered_dir / f"{split}_segments.json"
    if path.exists():
        with open(path) as f:
            segments = json.load(f)
    elif (filtered_dir / "passed_segments.json").exists():
        with open(filtered_dir / "passed_segments.json") as f:
            segments = json.load(f)
    else:
        raise FileNotFoundError(f"No {split} segments in {filtered_dir}.")
    if limit:
        segments = segments[:limit]
    return segments, ebird_to_id
