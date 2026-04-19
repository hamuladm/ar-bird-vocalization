import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import (
    DEVICE,
    AG_GEN_DURATION,
    AG_GEN_TEMPERATURE,
    AG_GEN_TOP_K,
    AG_GEN_CFG_COEF,
    SNAC_GEN_TEMPERATURE,
    SNAC_GEN_TOP_K,
)

logger = logging.getLogger(__name__)


def _build_generator(args, ebird_codes=None):
    if args.model_type == "audiogen":
        from generator.audiogen_generator import AudiogenGenerator
        return AudiogenGenerator(
            args.checkpoint,
            device=args.device,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            use_bf16=args.bf16,
        )
    if args.model_type == "audiogen_text":
        from generator.audiogen_generator import TextAudiogenGenerator
        return TextAudiogenGenerator(
            ebird_codes,
            device=args.device,
            use_bf16=args.bf16,
            prompt_template=args.prompt_template,
        )
    from generator.llama_generator import LlamaGenerator
    return LlamaGenerator(args.checkpoint, device=args.device, use_bf16=args.bf16)


def _default_output_dir(model_type, stage):
    return Path("generated_samples") / f"{model_type}_{stage}"


def _gen_kwargs(args, model_type):
    if model_type in ("audiogen", "audiogen_text"):
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
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["audiogen", "audiogen_text", "llama"])
    parser.add_argument("--stage", type=str, required=True,
                        choices=["pretrain", "finetune", "baseline"])
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
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument(
        "--prompt-template", type=str, default="descriptive",
        help="Text prompt template for audiogen_text: 'scientific', 'common', "
             "'descriptive', or a custom format string with {sci_name}, {common_name}, {ebird_code}",
    )
    args = parser.parse_args()

    if args.model_type != "audiogen_text" and args.checkpoint is None:
        parser.error("--checkpoint is required for model types other than audiogen_text")

    if args.temperature is None:
        args.temperature = (
            AG_GEN_TEMPERATURE if args.model_type in ("audiogen", "audiogen_text")
            else SNAC_GEN_TEMPERATURE
        )
    if args.top_k is None:
        args.top_k = (
            AG_GEN_TOP_K if args.model_type in ("audiogen", "audiogen_text")
            else SNAC_GEN_TOP_K
        )

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    output_dir = Path(args.output) if args.output else _default_output_dir(args.model_type, args.stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output directory: %s", output_dir)

    logger.info("loading test segments from %s", args.test_segments)
    with open(args.test_segments) as f:
        segments = json.load(f)
    test_ebird_codes = set(seg["ebird_code"] for seg in segments)
    logger.info("found %d segments spanning %d unique classes", len(segments), len(test_ebird_codes))

    if args.model_type == "audiogen_text":
        logger.info("loading pretrained AudioGen with text conditioning (template=%s)", args.prompt_template)
    else:
        logger.info("loading %s generator from %s", args.model_type, args.checkpoint)
    t0 = time.time()
    generator = _build_generator(args, ebird_codes=test_ebird_codes)
    logger.info("generator loaded in %.1fs (%d known classes)", time.time() - t0, len(generator.ebird_to_id))

    use_reranker = not args.no_reranker
    reranker = None

    if use_reranker:
        from reranker.reranker import Reranker

        logger.info("loading %s discriminator/embedder", args.discriminator)
        t0 = time.time()
        if args.discriminator == "birdnet":
            from evaluation.embeddings import BirdNetEmbedder
            embedder = BirdNetEmbedder(device=args.device)
        else:
            from evaluation.embeddings import EvalEmbedder
            embedder = EvalEmbedder(device=args.device)
        logger.info("embedder loaded in %.1fs", time.time() - t0)

        reranker = Reranker(generator, embedder, device=args.device)

        generator_codes = set(generator.ebird_to_id)
        if args.discriminator == "birdnet":
            disc_codes = set(embedder.ebird_to_idx)
        else:
            id2label = embedder.model.config.id2label
            disc_codes = {str(v).strip() for v in id2label.values()}

        classes = sorted(c for c in test_ebird_codes if c in generator_codes and c in disc_codes)
        n_skipped = len(test_ebird_codes) - len(classes)
        if n_skipped:
            logger.warning(
                "skipping %d test classes not covered by generator+discriminator", n_skipped,
            )
    else:
        logger.info("reranker disabled, generating directly from model")
        generator_codes = set(generator.ebird_to_id)
        classes = sorted(c for c in test_ebird_codes if c in generator_codes)
        n_skipped = len(test_ebird_codes) - len(classes)
        if n_skipped:
            logger.warning("skipping %d test classes not in generator vocabulary", n_skipped)

    gen_kwargs = _gen_kwargs(args, args.model_type)
    total_samples = args.n_per_class * len(classes)
    logger.info(
        "generating %d samples/class for %d classes (%d total) | "
        "reranker=%s, k=%d, disc=%s | gen_kwargs=%s",
        args.n_per_class, len(classes), total_samples,
        use_reranker, args.k, args.discriminator, gen_kwargs,
    )

    total_generated = 0
    total_failed = 0
    run_start = time.time()

    class_bar = tqdm(classes, desc="classes", unit="cls", position=0)
    for ebird_code in class_bar:
        class_id = generator.ebird_to_id[ebird_code]
        waveforms = []
        class_failed = 0

        sample_bar = tqdm(
            range(args.n_per_class),
            desc=f"  {ebird_code}",
            unit="sample",
            position=1,
            leave=False,
        )
        for si in sample_bar:
            if use_reranker:
                audio = reranker.generate(class_id, k=args.k, **gen_kwargs)
            else:
                audio = generator.generate(class_id, **gen_kwargs)
            if audio is None:
                class_failed += 1
                sample_bar.set_postfix(ok=len(waveforms), fail=class_failed)
                continue
            waveforms.append(audio)
            sample_bar.set_postfix(ok=len(waveforms), fail=class_failed)

        total_generated += len(waveforms)
        total_failed += class_failed

        _save_class_shard(output_dir, ebird_code, waveforms, generator.sample_rate)
        class_bar.set_postfix(
            done=total_generated, failed=total_failed, current=ebird_code,
        )
        if class_failed:
            logger.warning(
                "%s: %d/%d samples failed", ebird_code, class_failed, args.n_per_class,
            )

    elapsed = time.time() - run_start
    logger.info(
        "generation complete in %.1fs | %d/%d samples generated, %d failed",
        elapsed, total_generated, total_samples, total_failed,
    )
    if total_generated:
        logger.info("avg %.2fs per sample", elapsed / total_generated)

    test_counts = Counter(seg["ebird_code"] for seg in segments if seg["ebird_code"] in set(classes))
    metadata = {
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "stage": args.stage,
        "reranker": use_reranker,
        "discriminator": args.discriminator if use_reranker else None,
        "test_segments": args.test_segments,
        "n_per_class": args.n_per_class,
        "k": args.k if use_reranker else None,
        "num_classes": len(classes),
        "classes": classes,
        "sample_rate": generator.sample_rate,
        "gen_kwargs": gen_kwargs,
        "test_segments_per_class": dict(test_counts),
        "total_generated": total_generated,
        "total_failed": total_failed,
        "elapsed_seconds": round(elapsed, 1),
    }
    if args.model_type == "audiogen_text":
        metadata["prompt_template"] = args.prompt_template
        metadata["prompts"] = {
            code: generator._prompts[code] for code in classes if code in generator._prompts
        }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("metadata saved to %s/metadata.json", output_dir)


if __name__ == "__main__":
    main()
