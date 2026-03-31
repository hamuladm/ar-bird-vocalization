
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Audio, load_dataset
from tqdm.auto import tqdm

from preprocessing.pipeline import split_segments

logger = logging.getLogger(__name__)

XCM_METADATA_COLUMNS = (
    "ebird_code",
    "filepath",
    "length",
    "detected_events",
    "event_cluster",
)


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


def filter_bird_events(
    detected_events: list,
    event_cluster: list | None,
) -> list[tuple[float, float]]:
    if not detected_events:
        return []
    de = np.asarray(detected_events, dtype=np.float64)
    if de.ndim != 2 or de.shape[1] != 2:
        return []

    if event_cluster is None or len(event_cluster) != len(detected_events):
        return [(float(de[i, 0]), float(de[i, 1])) for i in range(len(de))]

    ec = np.asarray(event_cluster)
    if len(ec) == 1 and ec[0] == -1:
        return []

    if (not (len(ec) == 1 and ec[0] == -1)) or len(ec) > 1:
        mask = ec != -1
        de = de[mask]
    if len(de) < 1:
        return []

    return [(float(de[i, 0]), float(de[i, 1])) for i in range(len(de))]


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


def _resolve_audio_path(
    row: dict[str, Any],
    audio_root: str | None,
    *,
    verify_exists: bool = True,
) -> str | None:
    fp = row.get("filepath")
    if audio_root and fp:
        candidate = Path(audio_root) / fp
        if not verify_exists or candidate.is_file():
            return str(candidate)
    audio = row.get("audio")
    if isinstance(audio, dict):
        p = audio.get("path")
        if p and (not verify_exists or Path(p).is_file()):
            return str(p)
    return None


def _pack_segments_from_row(
    row: dict[str, Any],
    ebird_code: str,
    remaining_sec: dict[str, float],
    chunk_sec: float,
    min_chunk_sec: float,
    rng: np.random.Generator,
    audio_root: str | None,
) -> list[dict[str, Any]]:
    """Extract one or more training windows from bird events in one XCM row."""
    if remaining_sec.get(ebird_code, 0.0) <= 0:
        return []

    length = row.get("length")
    if length is None:
        return []
    file_len = float(length)

    events = filter_bird_events(
        row.get("detected_events") or [],
        row.get("event_cluster"),
    )
    if not events:
        return []

    path = _resolve_audio_path(row, audio_root)
    if not path:
        return []

    order = rng.permutation(len(events))
    out: list[dict[str, Any]] = []
    for j in order:
        if remaining_sec[ebird_code] <= 0:
            break
        ev_start, ev_end = events[int(j)]
        window = event_to_segment_window(
            ev_start, ev_end, file_len, chunk_sec, min_chunk_sec, rng
        )
        if window is None:
            continue
        t0, t1 = window
        dur = t1 - t0
        if dur <= 0:
            continue
        out.append(
            {
                "filepath": path,
                "start": t0,
                "end": t1,
                "ebird_code": ebird_code,
                "source": "xcm",
            }
        )
        remaining_sec[ebird_code] = max(0.0, remaining_sec[ebird_code] - dur)

    return out


def get_xcm_ebird_names(hf_path: str, hf_name: str) -> list[str]:
    tiny = load_dataset(
        hf_path,
        hf_name,
        split="train[:1]",
        trust_remote_code=True,
    )
    names = tiny.features["ebird_code"].names
    if names is None:
        raise ValueError("XCM dataset has no ebird_code.names")
    return list(names)


def load_xcm_train_full_audio(
    hf_path: str,
    hf_name: str,
    sample_rate: int = 32000,
):
    ds = load_dataset(hf_path, hf_name, split="train", trust_remote_code=True)
    ds = ds.cast_column(
        "audio",
        Audio(sampling_rate=sample_rate, decode=True),
    )
    logger.warning(
        "XCM: full dataset with audio decode — large download. Prefer metadata_only + audio_root."
    )
    return ds


def _xcm_metadata_stream(
    hf_path: str,
    hf_name: str,
    *,
    ebird_names: list[str],
    allowed_species: frozenset[str],
    shuffle_seed: int,
    shuffle_buffer_size: int,
):
    ds = load_dataset(
        hf_path,
        hf_name,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.select_columns(list(XCM_METADATA_COLUMNS))

    def _species_ok(ex: dict) -> bool:
        return ebird_names[int(ex["ebird_code"])] in allowed_species

    ds = ds.filter(_species_ok)
    ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)
    return ds


def enrich_segments_with_xcm(
    *,
    quota_seconds: dict[str, float],
    hf_path: str,
    hf_name: str,
    seed: int,
    chunk_sec: float,
    min_chunk_sec: float,
    audio_root: str | None,
    sample_rate: int = 32000,
    metadata_only: bool = True,
    shuffle_buffer_size: int = 50_000,
    max_stream_passes: int = 5,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Stream XCM and add segments until each class has filled its **quota_seconds**
    (typically n_train × chunk_sec from BirdCleF).

    Returns ``(xcm_segments, stats)``.
    """
    remaining_sec = {c: float(v) for c, v in quota_seconds.items() if v > 0.0}
    if not remaining_sec:
        return [], {}

    allowed_f = frozenset(remaining_sec.keys())
    stats: dict[str, Any] = {
        "rows_scanned": 0,
        "segments_added": 0,
        "skipped_no_path": 0,
        "skipped_no_event": 0,
        "skipped_bad_window": 0,
        "stream_passes": 0,
        "initial_quota_sec": dict(remaining_sec),
    }

    logger.info(
        "XCM enrich: %d classes, total quota %.0fs (metadata_only=%s, audio_root=%s)",
        len(remaining_sec),
        sum(remaining_sec.values()),
        metadata_only,
        audio_root,
    )
    if metadata_only and not audio_root:
        logger.warning(
            "XCM metadata_only without audio_root: paths may not resolve (use audio_root)."
        )

    ebird_names = get_xcm_ebird_names(hf_path, hf_name)
    rng = np.random.default_rng(seed)
    xcm_segments: list[dict] = []

    def _all_done() -> bool:
        return all(v <= 1e-9 for v in remaining_sec.values())

    if metadata_only:
        for pass_idx in range(max_stream_passes):
            if _all_done():
                break
            stats["stream_passes"] += 1
            stream = _xcm_metadata_stream(
                hf_path,
                hf_name,
                ebird_names=ebird_names,
                allowed_species=allowed_f,
                shuffle_seed=seed + pass_idx,
                shuffle_buffer_size=shuffle_buffer_size,
            )
            pbar = tqdm(desc=f"XCM enrich pass {pass_idx + 1}", unit="row")
            for row in stream:
                if _all_done():
                    break
                stats["rows_scanned"] += 1
                code = ebird_names[int(row["ebird_code"])]
                if remaining_sec.get(code, 0.0) <= 0:
                    continue

                if not filter_bird_events(
                    row.get("detected_events") or [],
                    row.get("event_cluster"),
                ):
                    stats["skipped_no_event"] += 1
                    continue

                if not _resolve_audio_path(row, audio_root):
                    stats["skipped_no_path"] += 1
                    continue

                new_segs = _pack_segments_from_row(
                    row,
                    code,
                    remaining_sec,
                    chunk_sec,
                    min_chunk_sec,
                    rng,
                    audio_root,
                )
                if not new_segs:
                    stats["skipped_bad_window"] += 1
                else:
                    xcm_segments.extend(new_segs)
                    stats["segments_added"] += len(new_segs)
                pbar.update(1)
            pbar.close()
    else:
        ds = load_xcm_train_full_audio(hf_path, hf_name, sample_rate=sample_rate)

        def _keep(ex: dict) -> bool:
            return ebird_names[int(ex["ebird_code"])] in allowed_f

        n_proc = min(4, max(1, os.cpu_count() or 1))
        filtered = ds.filter(_keep, num_proc=n_proc)
        shuffled = filtered.shuffle(seed=seed)
        logger.info("XCM rows after species filter: %d", len(shuffled))

        for i in tqdm(range(len(shuffled)), desc="XCM enrich", unit="row"):
            if _all_done():
                break
            row = shuffled[i]
            stats["rows_scanned"] += 1
            code = ebird_names[int(row["ebird_code"])]
            if remaining_sec.get(code, 0.0) <= 0:
                continue

            if not filter_bird_events(
                row.get("detected_events") or [],
                row.get("event_cluster"),
            ):
                stats["skipped_no_event"] += 1
                continue

            if not _resolve_audio_path(row, audio_root):
                stats["skipped_no_path"] += 1
                continue

            new_segs = _pack_segments_from_row(
                row,
                code,
                remaining_sec,
                chunk_sec,
                min_chunk_sec,
                rng,
                audio_root,
            )
            if not new_segs:
                stats["skipped_bad_window"] += 1
            else:
                xcm_segments.extend(new_segs)
                stats["segments_added"] += len(new_segs)

    stats["remaining_quota_sec"] = dict(remaining_sec)
    shortfall = {c: v for c, v in remaining_sec.items() if v > 1e-6}
    if shortfall:
        logger.warning(
            "XCM: could not fill full quota for %d classes (remaining seconds, sample): %s",
            len(shortfall),
            str(list(shortfall.items())[:5]),
        )
    logger.info(
        "XCM enrich done: %d segments, rows_scanned=%d, stream_passes=%d",
        len(xcm_segments),
        stats["rows_scanned"],
        stats["stream_passes"],
    )
    return xcm_segments, stats


def subset_ebird_to_id_for_classes(
    backbone_ebird_to_id_path: str | Path,
    class_codes: set[str],
) -> dict[str, int]:
    """Map ebird_code -> backbone id for classes that passed BirdCleF (for quotas / save)."""
    path = Path(backbone_ebird_to_id_path)
    with open(path) as f:
        full_map = json.load(f)
    out = {}
    for code in class_codes:
        if code not in full_map:
            raise KeyError(f"ebird_code {code!r} missing from backbone {path}")
        out[code] = full_map[code]
    return out
