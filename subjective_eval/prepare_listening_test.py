import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent

EBIRD_TO_ID_PATH = ROOT / "data" / "birdclef_segments" / "ebird_to_id.json"
ENRICHED_DIR = ROOT / "data" / "birdclef_enriched_segments"
OUTPUT_DIR = ROOT / "subjective_eval" / "mos_samples"

TARGET_SR = 32000
TARGET_LUFS = -16.0
REFERENCE_DURATION = 35.0

SYSTEMS = ["gt", "audiogen", "llama", "rf"]

MODELS_NPZ = {
    "audiogen": ROOT / "eval_generated_samples" / "audiogen_finetune_no_rerank",
    "llama": ROOT / "eval_generated_samples" / "llama_finetune_no_rerank",
    "rf": ROOT / "eval_generated_samples" / "rf_finetune",
}



def load_classes():
    with open(EBIRD_TO_ID_PATH) as f:
        return json.load(f)


def _load_all_segments():
    all_segs = []
    for name in ("train_segments.json", "val_segments.json", "test_segments.json"):
        p = ENRICHED_DIR / name
        if not p.exists():
            continue
        with open(p) as f:
            segs = json.load(f)
        all_segs.extend(segs)
    return all_segs


def _resolve_path(filepath):
    if filepath.startswith("/"):
        return filepath
    return str(ROOT / filepath)


def _load_and_slice(filepath, start_sec, end_sec, target_sr=TARGET_SR):
    wav, sr = torchaudio.load(_resolve_path(filepath))
    start = int(start_sec * sr)
    end = int(end_sec * sr)
    wav = wav[:1, start:end]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav, target_sr


def _normalize_lufs(wav, sr, target_lufs=TARGET_LUFS):
    audio = wav.squeeze(0).numpy()
    meter = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(audio)
    if not np.isfinite(current_lufs):
        return wav
    normalized = pyln.normalize.loudness(audio, current_lufs, target_lufs)
    normalized = np.clip(normalized, -1.0, 1.0)
    return torch.from_numpy(normalized).float().unsqueeze(0)


def _save_wav(wav, sr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = _normalize_lufs(wav, sr)
    torchaudio.save(str(path), wav, sr)



def load_generated_sample(model_dir, ebird_code, sample_idx=0):
    npz_path = model_dir / f"{ebird_code}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing shard: {npz_path}")
    data = np.load(npz_path)
    samples = data["samples"]
    lengths = data["lengths"]
    sr = int(data["sample_rate"])

    if sample_idx >= len(lengths):
        raise IndexError(
            f"{npz_path}: requested index {sample_idx} but only "
            f"{len(lengths)} samples available"
        )

    waveform = samples[sample_idx, : lengths[sample_idx]]
    wav = torch.from_numpy(waveform).float().unsqueeze(0)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav, TARGET_SR



def _find_best_anchor_recordings(classes, all_segs):
    by_recording = defaultdict(list)
    for s in all_segs:
        if s["ebird_code"] in classes:
            by_recording[(s["filepath"], s["ebird_code"])].append(s)

    best = {}
    for (fp, code), seg_list in by_recording.items():
        max_end = max(s["end"] for s in seg_list)
        min_start = min(s["start"] for s in seg_list)
        span = max_end - min_start
        prev = best.get(code)
        if prev is None or span > prev["span"]:
            best[code] = {
                "filepath": fp,
                "min_start": min_start,
                "max_end": max_end,
                "span": span,
            }
    return best


def extract_reference(rec, duration=REFERENCE_DURATION):
    start = rec["min_start"]
    end = min(start + duration, rec["max_end"])
    wav, sr = _load_and_slice(rec["filepath"], start, end)
    return wav, sr



def pick_gt_segments(classes, seed=42):
    with open(ENRICHED_DIR / "test_segments.json") as f:
        segments = json.load(f)
    by_class = defaultdict(list)
    for s in segments:
        if s["ebird_code"] in classes:
            by_class[s["ebird_code"]].append(s)
    rng = random.Random(seed)
    return {code: rng.choice(by_class[code]) for code in sorted(classes)}


def extract_gt_clip(seg):
    wav, sr = _load_and_slice(seg["filepath"], seg["start"], seg["end"])
    return wav, sr



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-idx", type=int, default=1)
    args = parser.parse_args()

    classes = load_classes()
    all_segs = _load_all_segments()
    print(f"Classes ({len(classes)}): {', '.join(sorted(classes))}")

    print("\n=== Reference recordings ===")
    anchor_recs = _find_best_anchor_recordings(classes, all_segs)

    gt_segs = pick_gt_segments(classes, seed=args.seed)

    print("\n=== Loading generated samples from npz shards ===")
    generated = {}
    for model, model_dir in MODELS_NPZ.items():
        for code in sorted(classes):
            wav, sr = load_generated_sample(model_dir, code, args.sample_idx)
            generated[(model, code)] = (wav, sr)
            dur = wav.shape[1] / sr
            print(f"  {model}/{code}: {dur:.1f}s (resampled to {sr} Hz)")

    print("\n=== Assembling MOS sample folders ===")
    rng = random.Random(args.seed)
    manifest = {}

    for code in sorted(classes):
        species_dir = args.output / code

        ref_wav, ref_sr = extract_reference(anchor_recs[code])
        _save_wav(ref_wav, ref_sr, species_dir / "reference.wav")
        ref_dur = ref_wav.shape[1] / ref_sr
        print(f"\n  {code}: reference {ref_dur:.1f}s")

        gt_wav, gt_sr = extract_gt_clip(gt_segs[code])

        system_wavs = {
            "gt": (gt_wav, gt_sr),
            "audiogen": generated[("audiogen", code)],
            "llama": generated[("llama", code)],
            "rf": generated[("rf", code)],
        }

        order = list(SYSTEMS)
        rng.shuffle(order)

        species_manifest = {}
        for slot_idx, system in enumerate(order, 1):
            slot_name = f"sample_{slot_idx}"
            wav, sr = system_wavs[system]
            _save_wav(wav, sr, species_dir / f"{slot_name}.wav")
            species_manifest[slot_name] = system
            dur = wav.shape[1] / sr
            print(f"    {slot_name} = {system} ({dur:.1f}s)")

        manifest[code] = species_manifest

    manifest_path = args.output / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")

    total_wavs = len(list(args.output.rglob("*.wav")))
    print("\n=== Summary ===")
    print(f"  Species: {len(classes)}")
    print(f"  Samples per species: {len(SYSTEMS)}")
    print(f"  Total trials: {len(classes) * len(SYSTEMS)}")
    print(f"  Total wav files: {total_wavs}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
