from __future__ import annotations

import logging
import time

import torch
from tqdm.auto import tqdm

from config import EVAL_SAMPLE_RATE
from birdclef_preprocessing.judge import BirdClassifier
from utils.audio import load_segment

logger = logging.getLogger(__name__)


def _collate_waveforms(waveforms: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(w.shape[0] for w in waveforms)
    batch = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        batch[i, : w.shape[0]] = w
    return batch


def _normalize_filepath(filepath: str, rewrite_hf_paths: bool) -> str:
    if rewrite_hf_paths:
        return filepath.replace(
            "/workspace/.hf_home/", "/home/dkham/.cache/huggingface/"
        )
    return filepath


def _id2label_str(id2label: dict | None, class_idx: int) -> str:
    if not id2label:
        raise ValueError("ConvNeXT config has no id2label; cannot match ebird_code")
    if str(class_idx) in id2label:
        return str(id2label[str(class_idx)]).strip()
    if class_idx in id2label:
        return str(id2label[class_idx]).strip()
    return str(class_idx)


def gate_segments(
    segments: list[dict],
    classifier: BirdClassifier,
    *,
    min_top1_prob: float,
    max_entropy: float,
    batch_size: int,
    rewrite_hf_paths: bool = False,
) -> list[dict]:
    id2label = getattr(classifier.model.config, "id2label", None)
    device = next(classifier.model.parameters()).device
    n_in = len(segments)

    logger.info(
        "Gating %d segments (streaming): top1_prob > %.4f, entropy < %.4f, "
        "batch_size=%d, device=%s",
        n_in,
        min_top1_prob,
        max_entropy,
        batch_size,
        device,
    )
    if rewrite_hf_paths:
        logger.info(
            "Path rewrite enabled: /workspace/.hf_home/ -> ~/.cache/huggingface/"
        )

    n_load_fail = 0
    n_fail_prob = n_fail_label = n_fail_entropy = 0
    passed: list[dict] = []
    load_ok = 0
    n_batches = 0
    idx = 0
    t0 = time.perf_counter()

    with tqdm(total=n_in, desc="Gating (load+infer)", unit="seg") as pbar:
        while idx < n_in:
            batch_wavs: list[torch.Tensor] = []
            batch_meta: list[dict] = []

            while len(batch_wavs) < batch_size and idx < n_in:
                seg = segments[idx]
                idx += 1
                pbar.update(1)
                path = _normalize_filepath(seg["filepath"], rewrite_hf_paths)
                try:
                    audio = load_segment(
                        path, seg["start"], seg["end"], EVAL_SAMPLE_RATE
                    )
                except Exception:
                    n_load_fail += 1
                    continue
                batch_wavs.append(torch.from_numpy(audio).float())
                batch_meta.append(seg)

            if not batch_wavs:
                continue

            load_ok += len(batch_wavs)
            n_batches += 1
            batch = _collate_waveforms(batch_wavs).to(device)
            out = classifier.evaluate(batch)
            top1_prob = out["top1_prob"].float().cpu()
            entropy = out["entropy"].float().cpu()
            top1_class = out["top1_class"].long().cpu()
            del batch, out

            for i, seg in enumerate(batch_meta):
                p = float(top1_prob[i])
                e = float(entropy[i])
                pred_idx = int(top1_class[i])
                pred_label = _id2label_str(id2label, pred_idx)
                gt = seg["ebird_code"]

                if p <= min_top1_prob:
                    n_fail_prob += 1
                    continue
                if pred_label != gt:
                    n_fail_label += 1
                    continue
                if e >= max_entropy:
                    n_fail_entropy += 1
                    continue
                passed.append(seg)

    elapsed = time.perf_counter() - t0
    pass_rate_loaded = 100.0 * len(passed) / load_ok if load_ok else 0.0
    pass_rate_vs_in = 100.0 * len(passed) / n_in if n_in else 0.0

    logger.info(
        "Gating finished in %.1fs: %d batches, %d waveforms scored, %d load failures "
        "(~%.1f segments/s over all inputs)",
        elapsed,
        n_batches,
        load_ok,
        n_load_fail,
        n_in / elapsed if elapsed > 0 else 0.0,
    )
    logger.info(
        "Gating summary: in=%d, load_fail=%d, fail_top1_prob=%d, fail_label=%d, "
        "fail_entropy=%d, passed=%d (%.1f%% of loaded waveforms, %.1f%% of all inputs)",
        n_in,
        n_load_fail,
        n_fail_prob,
        n_fail_label,
        n_fail_entropy,
        len(passed),
        pass_rate_loaded,
        pass_rate_vs_in,
    )
    return passed
