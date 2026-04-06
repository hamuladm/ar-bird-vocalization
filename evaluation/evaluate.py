import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch

from config import DEVICE, SNAC_GEN_TEMPERATURE, SNAC_GEN_TOP_K
from generator import LlamaGenerator
from evaluation.embeddings import (
    EvalEmbedder,
    extract_embeddings_from_directory,
    extract_embeddings_from_segments,
    extract_embeddings_from_arrays,
)
from evaluation.metrics import inception_score, compute_fad


def parse_args():
    parser = argparse.ArgumentParser()

    seg_group = parser.add_argument_group("Segment + generation mode")
    seg_group.add_argument("--checkpoint", type=str, default=None)
    seg_group.add_argument("--test-segments", type=str, default=None)
    seg_group.add_argument("--num-samples-per-class", type=int, default=10)
    seg_group.add_argument("--max-classes", type=int, default=1)
    seg_group.add_argument("--temperature", type=float, default=SNAC_GEN_TEMPERATURE)
    seg_group.add_argument("--top-k", type=int, default=SNAC_GEN_TOP_K)

    dir_group = parser.add_argument_group("Directory mode")
    dir_group.add_argument("--generated-dir", type=str, default=None)
    dir_group.add_argument("--reference-dir", type=str, default=None)

    parser.add_argument("--metrics", type=str, default="is,fad")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def _run_segment_mode(args, metrics_to_compute):
    device = torch.device(args.device)

    with open(args.test_segments) as f:
        segments = json.load(f)

    class_counts = Counter(seg["ebird_code"] for seg in segments)
    gen = LlamaGenerator(args.checkpoint, device=str(device))

    known_classes = sorted(c for c in class_counts if c in gen.ebird_to_id)

    if args.max_classes is not None and args.max_classes < len(known_classes):
        random.seed(42)
        known_classes = sorted(random.sample(known_classes, args.max_classes))

    gen_arrays = []
    for ebird_code in known_classes:
        cid = gen.ebird_to_id[ebird_code]
        for _ in range(args.num_samples_per_class):
            audio = gen.generate(cid, temperature=args.temperature, top_k=args.top_k)
            if audio is not None:
                gen_arrays.append(audio)
    embedder = EvalEmbedder(device=args.device)
    gen_data = extract_embeddings_from_arrays(
        gen_arrays, embedder, batch_size=args.batch_size
    )

    eval_class_set = set(known_classes)
    ref_segments = [s for s in segments if s["ebird_code"] in eval_class_set]
    ref_data = extract_embeddings_from_segments(
        ref_segments, embedder, batch_size=args.batch_size
    )

    results = {}

    if "is" in metrics_to_compute:
        is_value = inception_score(gen_data["probs"])
        gt_is_value = inception_score(ref_data["probs"])
        results["inception_score"] = is_value
        results["gt_inception_score"] = gt_is_value
        results["is_ratio"] = (
            is_value / gt_is_value if gt_is_value > 0 else float("nan")
        )

    if "fad" in metrics_to_compute:
        results["fad"] = compute_fad(gen_data["features"], ref_data["features"])

    results["metadata"] = {
        "mode": "segment",
        "checkpoint": args.checkpoint,
        "num_classes": len(known_classes),
        "num_generated": len(gen_arrays),
        "num_reference": len(ref_segments),
    }
    return results


def _run_directory_mode(args, metrics_to_compute):
    gen_dir = Path(args.generated_dir)
    embedder = EvalEmbedder(device=args.device)
    gen_data = extract_embeddings_from_directory(
        gen_dir, embedder, batch_size=args.batch_size
    )

    results = {}

    if "is" in metrics_to_compute:
        results["inception_score"] = inception_score(gen_data["probs"])

    if "fad" in metrics_to_compute:
        ref_data = extract_embeddings_from_directory(
            args.reference_dir, embedder, batch_size=args.batch_size
        )
        results["fad"] = compute_fad(gen_data["features"], ref_data["features"])

    results["metadata"] = {
        "mode": "directory",
        "generated_dir": str(gen_dir.resolve()),
    }
    return results


def main():
    args = parse_args()
    metrics_to_compute = {m.strip().lower() for m in args.metrics.split(",")}

    segment_mode = args.checkpoint is not None and args.test_segments is not None
    directory_mode = args.generated_dir is not None

    if segment_mode:
        results = _run_segment_mode(args, metrics_to_compute)
        default_output = Path("eval_results.json")
    elif directory_mode:
        results = _run_directory_mode(args, metrics_to_compute)
        default_output = Path(args.generated_dir) / "eval_results.json"
    else:
        raise ValueError("Provide --checkpoint + --test-segments or --generated-dir")

    output_path = Path(args.output) if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    for key in ["inception_score", "gt_inception_score", "is_ratio", "fad"]:
        if key in results:
            print(f"  {key}: {results[key]:.4f}")


if __name__ == "__main__":
    main()
