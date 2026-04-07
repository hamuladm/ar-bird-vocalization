import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from config import (
    DEVICE,
    AG_GEN_DURATION,
    AG_GEN_TEMPERATURE,
    AG_GEN_TOP_K,
    AG_GEN_CFG_COEF,
    SNAC_GEN_TEMPERATURE,
    SNAC_GEN_TOP_K,
)
from reranker.reranker import Reranker

logger = logging.getLogger(__name__)


def _build_generator(args):
    if args.model_type == "audiogen":
        from generator.audiogen_generator import AudiogenGenerator
        return AudiogenGenerator(
            args.checkpoint,
            device=args.device,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            use_bf16=args.bf16,
        )
    from generator.llama_generator import LlamaGenerator
    return LlamaGenerator(args.checkpoint, device=args.device)


def _default_output_dir(model_type, stage):
    return Path("generated_samples") / f"{model_type}_{stage}"


def _gen_kwargs(args, model_type):
    if model_type == "audiogen":
        return dict(
            duration=args.duration,
            temperature=args.temperature,
            top_k=args.top_k,
            cfg_coef=args.cfg_coef,
        )
    return dict(
        temperature=args.temperature,
        top_k=args.top_k,
    )


def _save_class_shard(output_dir, ebird_code, waveforms, sample_rate):
    if not waveforms:
        return
    max_len = max(w.shape[0] for w in waveforms)
    lengths = np.array([w.shape[0] for w in waveforms], dtype=np.int64)
    samples = np.zeros((len(waveforms), max_len), dtype=np.float32)
    for i, w in enumerate(waveforms):
        samples[i, : w.shape[0]] = w
    np.savez(
        output_dir / f"{ebird_code}.npz",
        samples=samples,
        lengths=lengths,
        sample_rate=np.array(sample_rate),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True, choices=["audiogen", "llama"])
    parser.add_argument("--stage", type=str, required=True, choices=["pretrain", "finetune"])
    parser.add_argument("--test-segments", type=str, required=True)
    parser.add_argument("--n-per-class", type=int, default=10)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--cfg-coef", type=float, default=AG_GEN_CFG_COEF)
    parser.add_argument("--duration", type=float, default=AG_GEN_DURATION)
    parser.add_argument(
        "--discriminator", type=str, default="birdnet",
        choices=["birdnet", "convnext"],
    )
    args = parser.parse_args()

    if args.temperature is None:
        args.temperature = (
            AG_GEN_TEMPERATURE if args.model_type == "audiogen" else SNAC_GEN_TEMPERATURE
        )
    if args.top_k is None:
        args.top_k = (
            AG_GEN_TOP_K if args.model_type == "audiogen" else SNAC_GEN_TOP_K
        )

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    output_dir = Path(args.output) if args.output else _default_output_dir(args.model_type, args.stage)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.test_segments) as f:
        segments = json.load(f)
    test_ebird_codes = set(seg["ebird_code"] for seg in segments)

    generator = _build_generator(args)

    if args.discriminator == "birdnet":
        from evaluation.embeddings import BirdNetEmbedder
        embedder = BirdNetEmbedder(device=args.device)
    else:
        from evaluation.embeddings import EvalEmbedder
        embedder = EvalEmbedder(device=args.device)

    reranker = Reranker(generator, embedder, device=args.device)

    generator_codes = set(generator.ebird_to_id)
    if args.discriminator == "birdnet":
        disc_codes = set(embedder.ebird_to_idx)
    else:
        id2label = embedder.model.config.id2label
        disc_codes = {str(v).strip() for v in id2label.values()}

    classes = sorted(c for c in test_ebird_codes if c in generator_codes and c in disc_codes)
    logger.info(
        "generating %d samples/class for %d classes (k=%d, disc=%s), output=%s",
        args.n_per_class, len(classes), args.k, args.discriminator, output_dir,
    )

    gen_kwargs = _gen_kwargs(args, args.model_type)

    for ci, ebird_code in enumerate(classes):
        class_id = generator.ebird_to_id[ebird_code]
        waveforms = []
        for si in range(args.n_per_class):
            audio = reranker.generate(class_id, k=args.k, **gen_kwargs)
            if audio is None:
                logger.warning(
                    "class %s sample %d/%d: generation failed, skipping",
                    ebird_code, si + 1, args.n_per_class,
                )
                continue
            waveforms.append(audio)

        _save_class_shard(output_dir, ebird_code, waveforms, generator.sample_rate)
        logger.info(
            "[%d/%d] %s: saved %d/%d samples",
            ci + 1, len(classes), ebird_code, len(waveforms), args.n_per_class,
        )

    test_counts = Counter(seg["ebird_code"] for seg in segments if seg["ebird_code"] in set(classes))
    metadata = {
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "stage": args.stage,
        "discriminator": args.discriminator,
        "test_segments": args.test_segments,
        "n_per_class": args.n_per_class,
        "k": args.k,
        "num_classes": len(classes),
        "classes": classes,
        "sample_rate": generator.sample_rate,
        "gen_kwargs": gen_kwargs,
        "test_segments_per_class": dict(test_counts),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("done. metadata saved to %s/metadata.json", output_dir)


if __name__ == "__main__":
    main()
