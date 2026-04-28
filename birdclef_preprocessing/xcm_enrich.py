import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import soundfile as sf
from tqdm.auto import tqdm

from preprocessing.pipeline import save_segments, split_segments


def _normalize_filepath(filepath: str, rewrite_hf_paths: bool) -> str:
    if rewrite_hf_paths:
        return filepath.replace(
            "/workspace/.hf_home/", "/home/dkham/.cache/huggingface/"
        )
    return filepath


logger = logging.getLogger(__name__)

XcmQuotaMode = Literal["birdclef_train", "fixed_per_class"]


def load_pretrain_holdout_filepaths(
    pretrain_segment_dir: str | Path,
    rewrite_hf_paths: bool = False,
) -> frozenset[str]:
    seg_dir = Path(pretrain_segment_dir)
    fps: set[str] = set()
    for name in ("val_segments.json", "test_segments.json"):
        p = seg_dir / name
        if not p.is_file():
            raise FileNotFoundError(
                f"Pretrain holdout file not found: {p}. "
                "Cannot guarantee no data leakage without it."
            )
        with open(p) as f:
            segments: list[dict] = json.load(f)
        for seg in segments:
            raw_fp = seg.get("filepath", "")
            if raw_fp:
                fps.add(_normalize_filepath(str(raw_fp), rewrite_hf_paths))
    logger.info(
        "Loaded %d unique pretrain holdout filepaths from %s",
        len(fps),
        seg_dir,
    )
    return frozenset(fps)


def load_finetune_ebird_to_id(path: str | Path) -> dict[str, int]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"finetune ebird_to_id JSON not found: {p}")
    with open(p) as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"expected non-empty JSON object in {p}")
    out: dict[str, int] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            raise TypeError(f"ebird codes must be str, got {type(k)!r} in {p}")
        out[k] = int(v)
    return out


def fixed_quota_seconds_per_class(
    finetune_map: dict[str, int],
    *,
    extra_segments_per_class: int,
    chunk_sec: float,
) -> dict[str, float]:
    if extra_segments_per_class < 0:
        raise ValueError("extra_segments_per_class must be >= 0")
    sec = float(extra_segments_per_class) * float(chunk_sec)
    return {c: sec for c in finetune_map}


def enrich_with_xcm_from_jsons(
    birdclef_segments: list[dict],
    *,
    finetune_ebird_to_id_json: str | Path,
    passed_segments_json: str | Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    chunk_sec: float,
    min_chunk_sec: float,
    rewrite_hf_paths: bool = False,
    min_top1_prob: float | None = None,
    quota_mode: XcmQuotaMode = "birdclef_train",
    xcm_extra_segments_per_class: int = 50,
    pretrain_segment_dir: str | Path | None = None,
) -> tuple[list[dict], dict[str, int], dict[str, Any]]:
    exclude_fps: frozenset[str] | None = None
    if pretrain_segment_dir is not None:
        exclude_fps = load_pretrain_holdout_filepaths(
            pretrain_segment_dir, rewrite_hf_paths=rewrite_hf_paths
        )

    finetune_map = load_finetune_ebird_to_id(finetune_ebird_to_id_json)
    if quota_mode == "birdclef_train":
        quota_sec = train_quota_seconds_per_class(
            birdclef_segments,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            chunk_sec=chunk_sec,
            ebird_to_id=finetune_map,
        )
    elif quota_mode == "fixed_per_class":
        quota_sec = fixed_quota_seconds_per_class(
            finetune_map,
            extra_segments_per_class=xcm_extra_segments_per_class,
            chunk_sec=chunk_sec,
        )
    else:
        raise ValueError(f"unknown quota_mode: {quota_mode!r}")

    xcm_segs, stats = enrich_segments_with_xcm(
        quota_seconds=quota_sec,
        passed_segments_json=passed_segments_json,
        seed=seed,
        chunk_sec=chunk_sec,
        min_chunk_sec=min_chunk_sec,
        rewrite_hf_paths=rewrite_hf_paths,
        min_top1_prob=min_top1_prob,
        exclude_filepaths=exclude_fps,
    )
    stats["finetune_ebird_to_id_path"] = str(Path(finetune_ebird_to_id_json).resolve())
    stats["finetune_n_classes"] = len(finetune_map)
    stats["quota_mode"] = quota_mode
    stats["xcm_extra_segments_per_class"] = (
        xcm_extra_segments_per_class if quota_mode == "fixed_per_class" else None
    )
    return xcm_segs, finetune_map, stats


def train_quota_seconds_per_class(
    bc_segments: list[dict],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    chunk_sec: float,
    ebird_to_id: dict[str, Any] | None = None,
) -> dict[str, float]:
    splits = split_segments(list(bc_segments), val_ratio, test_ratio, seed)
    train_counts = Counter(s["ebird_code"] for s in splits["train"])
    quota = {c: float(n) * float(chunk_sec) for c, n in train_counts.items()}
    if ebird_to_id is not None:
        allowed = frozenset(ebird_to_id.keys())
        quota = {c: v for c, v in quota.items() if c in allowed}
    return quota


def event_to_segment_window(
    ev_start: float,
    ev_end: float,
    file_len: float,
    chunk_sec: float,
    min_chunk_sec: float,
    rng: np.random.Generator,
) -> tuple[float, float] | None:
    file_len = float(file_len)
    if file_len < min_chunk_sec:
        return None

    ev_start = max(0.0, min(float(ev_start), file_len))
    ev_end = max(ev_start, min(float(ev_end), file_len))
    ev_dur = ev_end - ev_start

    if ev_dur <= 0:
        return None

    if file_len >= chunk_sec:
        if ev_dur >= chunk_sec:
            hi = ev_end - chunk_sec
            lo = ev_start
            if hi <= lo:
                t0 = lo
            else:
                t0 = float(rng.uniform(lo, hi))
            t1 = t0 + chunk_sec
        else:
            mid = 0.5 * (ev_start + ev_end)
            half = 0.5 * chunk_sec
            t0 = mid - half
            t0 = max(0.0, min(t0, file_len - chunk_sec))
            t1 = t0 + chunk_sec
    else:
        t0, t1 = 0.0, file_len

    dur = t1 - t0
    if dur < min_chunk_sec:
        need = min_chunk_sec - dur
        pad = 0.5 * need
        t0 -= pad
        t1 += pad
        if t0 < 0.0:
            t1 -= t0
            t0 = 0.0
        if t1 > file_len:
            t0 -= t1 - file_len
            t1 = file_len
        t0 = max(0.0, t0)
        t1 = min(file_len, t1)
        if t1 - t0 < min_chunk_sec:
            return None

    return (float(t0), float(t1))


def _duration_seconds(path: str, cache: dict[str, float]) -> float | None:
    if path in cache:
        return cache[path]
    try:
        info = sf.info(path)
    except OSError:
        return None
    cache[path] = float(info.duration)
    return cache[path]


def enrich_segments_with_xcm(
    *,
    quota_seconds: dict[str, float],
    passed_segments_json: str | Path,
    seed: int,
    chunk_sec: float,
    min_chunk_sec: float,
    rewrite_hf_paths: bool = False,
    min_top1_prob: float | None = None,
    exclude_filepaths: frozenset[str] | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    remaining_sec = {c: float(v) for c, v in quota_seconds.items() if v > 0.0}
    if not remaining_sec:
        return [], {}

    path = Path(passed_segments_json)
    if not path.is_file():
        raise FileNotFoundError(f"passed_segments_json not found: {path}")

    with open(path) as f:
        rows: list[dict] = json.load(f)

    rng = np.random.default_rng(seed)
    rng.shuffle(rows)

    stats: dict[str, Any] = {
        "rows_total": len(rows),
        "rows_scanned": 0,
        "segments_added": 0,
        "skipped_no_class": 0,
        "skipped_low_top1": 0,
        "skipped_no_path": 0,
        "skipped_pretrain_holdout": 0,
        "skipped_no_duration": 0,
        "skipped_bad_window": 0,
        "initial_quota_sec": dict(remaining_sec),
    }

    dur_cache: dict[str, float] = {}
    out: list[dict] = []

    def _all_done() -> bool:
        return all(v <= 1e-9 for v in remaining_sec.values())

    for row in tqdm(rows, desc="Relaxed / XCM enrich", unit="row"):
        if _all_done():
            break
        stats["rows_scanned"] += 1

        code = row.get("ebird_code")
        if not isinstance(code, str) or remaining_sec.get(code, 0.0) <= 0:
            stats["skipped_no_class"] += 1
            continue

        if min_top1_prob is not None:
            p = row.get("top1_prob")
            if p is not None and float(p) <= min_top1_prob:
                stats["skipped_low_top1"] += 1
                continue

        raw_fp = row.get("filepath")
        if not raw_fp:
            stats["skipped_no_path"] += 1
            continue
        fp = _normalize_filepath(str(raw_fp), rewrite_hf_paths)
        if not Path(fp).is_file():
            stats["skipped_no_path"] += 1
            continue

        if exclude_filepaths and fp in exclude_filepaths:
            stats["skipped_pretrain_holdout"] += 1
            continue

        file_len = _duration_seconds(fp, dur_cache)
        if file_len is None or file_len < min_chunk_sec:
            stats["skipped_no_duration"] += 1
            continue

        try:
            ev_start = float(row["start"])
            ev_end = float(row["end"])
        except (KeyError, TypeError, ValueError):
            stats["skipped_bad_window"] += 1
            continue

        window = event_to_segment_window(
            ev_start, ev_end, file_len, chunk_sec, min_chunk_sec, rng
        )
        if window is None:
            stats["skipped_bad_window"] += 1
            continue

        t0, t1 = window
        dur = t1 - t0
        if dur <= 0:
            stats["skipped_bad_window"] += 1
            continue

        out.append(
            {
                "filepath": fp,
                "start": t0,
                "end": t1,
                "ebird_code": code,
                "source": "relaxed",
            }
        )
        remaining_sec[code] = max(0.0, remaining_sec[code] - dur)
        stats["segments_added"] += 1

    stats["remaining_quota_sec"] = dict(remaining_sec)
    shortfall = {c: v for c, v in remaining_sec.items() if v > 1e-6}
    if shortfall:
        logger.warning(
            "Enrich: quota shortfall for %d classes (sample): %s",
            len(shortfall),
            str(list(shortfall.items())[:5]),
        )
    logger.info(
        "Enrich done: %d segments from %d rows scanned",
        stats["segments_added"],
        stats["rows_scanned"],
    )
    return out, stats


def subset_ebird_to_id_for_classes(
    backbone_ebird_to_id_path: str | Path,
    class_codes: set[str],
) -> dict[str, int]:
    path = Path(backbone_ebird_to_id_path)
    with open(path) as f:
        full_map = json.load(f)
    out = {}
    for code in class_codes:
        if code not in full_map:
            raise KeyError(f"ebird_code {code!r} missing from backbone {path}")
        out[code] = full_map[code]
    return out


def load_segments_list_json(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"segments JSON not found: {p}")
    with open(p) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected JSON array of segments in {p}")
    return data


def main() -> None:
    import argparse

    from config import (
        BC_EBIRD_TO_ID_PATH,
        BC_SEGMENT_DIR,
        BC_XCM_EXTRA_SEGMENTS_PER_CLASS,
        BC_XCM_FINETUNE_EBIRD_TO_ID_JSON,
        BC_XCM_MIN_TOP1_PROB,
        BC_XCM_PASSED_SEGMENTS_JSON,
        BC_XCM_PRETRAIN_SEGMENT_DIR,
        BC_XCM_QUOTA_MODE,
        CHUNK_LENGTH,
        MIN_CHUNK_SEC,
        SEED,
        TEST_RATIO,
        VAL_RATIO,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--birdclef-segments-json", type=str, required=True)
    parser.add_argument(
        "--finetune-ebird-to-id-json",
        type=str,
        default=str(BC_XCM_FINETUNE_EBIRD_TO_ID_JSON),
    )
    parser.add_argument(
        "--passed-segments-json",
        type=str,
        default=str(BC_XCM_PASSED_SEGMENTS_JSON),
    )
    parser.add_argument("--output-split-dir", type=str, default=str(BC_SEGMENT_DIR))
    parser.add_argument("--ebird-to-id", type=str, default=str(BC_EBIRD_TO_ID_PATH))
    parser.add_argument("--no-split-output", action="store_true")
    parser.add_argument("--output-merged-json", type=str, default=None)
    parser.add_argument("--output-xcm-only-json", type=str, default=None)
    parser.add_argument("--output-stats-json", type=str, default=None)
    parser.add_argument(
        "--xcm-quota-mode",
        type=str,
        choices=("birdclef_train", "fixed_per_class"),
        default=BC_XCM_QUOTA_MODE,
    )
    parser.add_argument(
        "--xcm-extra-segments-per-class",
        type=int,
        default=BC_XCM_EXTRA_SEGMENTS_PER_CLASS,
    )
    parser.add_argument("--chunk-sec", type=float, default=CHUNK_LENGTH)
    parser.add_argument("--min-chunk-sec", type=float, default=MIN_CHUNK_SEC)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rewrite-hf-paths", action="store_true")
    parser.add_argument("--min-top1-prob", type=float, default=None)
    parser.add_argument(
        "--pretrain-segment-dir",
        type=str,
        default=str(BC_XCM_PRETRAIN_SEGMENT_DIR)
        if BC_XCM_PRETRAIN_SEGMENT_DIR
        else None,
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
        force=True,
    )

    min_tp = BC_XCM_MIN_TOP1_PROB if args.min_top1_prob is None else args.min_top1_prob

    birdclef = load_segments_list_json(args.birdclef_segments_json)
    xcm_segs, _finetune_map, stats = enrich_with_xcm_from_jsons(
        birdclef,
        finetune_ebird_to_id_json=args.finetune_ebird_to_id_json,
        passed_segments_json=args.passed_segments_json,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        chunk_sec=args.chunk_sec,
        min_chunk_sec=args.min_chunk_sec,
        rewrite_hf_paths=args.rewrite_hf_paths,
        min_top1_prob=min_tp,
        quota_mode=args.xcm_quota_mode,
        xcm_extra_segments_per_class=args.xcm_extra_segments_per_class,
        pretrain_segment_dir=args.pretrain_segment_dir,
    )

    merged = list(birdclef) + list(xcm_segs)
    logger.info(
        "Merged pool: %d segments (%d birdclef + %d xcm)",
        len(merged),
        len(birdclef),
        len(xcm_segs),
    )

    if not args.no_split_output:
        splits = split_segments(merged, args.val_ratio, args.test_ratio, args.seed)
        split_dir = Path(args.output_split_dir)
        save_segments(
            split_dir,
            splits,
            backbone_ebird_to_id_path=args.ebird_to_id,
        )
        logger.info(
            "Splits -> %s (train=%d val=%d test=%d)",
            split_dir,
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )

    if args.output_merged_json:
        out_merged = Path(args.output_merged_json)
        out_merged.parent.mkdir(parents=True, exist_ok=True)
        with open(out_merged, "w") as f:
            json.dump(merged, f, indent=2)
        logger.info("Wrote merged JSON -> %s", out_merged)

    if args.output_xcm_only_json:
        p = Path(args.output_xcm_only_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(xcm_segs, f, indent=2)
        logger.info("Wrote %d xcm-only segments -> %s", len(xcm_segs), p)

    if args.output_stats_json:
        p = Path(args.output_stats_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Wrote enrich stats -> %s", p)


if __name__ == "__main__":
    main()
