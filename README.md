# AR Bird Vocalization

Autoregressive generation of bird vocalizations using class-conditioned language models over discrete audio tokens.

> **Thesis:** [TODO: Add link to the thesis PDF]
>
> **Demo:** [https://hamuladm.github.io/ar-bird-vocalization/](https://hamuladm.github.io/ar-bird-vocalization/)
>
> **Pretrained weights:** [TODO: Add link to model checkpoints]

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Models](#models)
- [Compute Requirements](#compute-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Subjective Listening Test](#subjective-listening-test)
- [Dependencies](#dependencies)

---

## Overview

This project explores autoregressive generation of bird vocalizations conditioned on species identity. Three generation approaches are implemented and compared:

1. **LLaMA/GPT-2 + SNAC** -- A custom causal language model trained from scratch on discrete SNAC audio tokens (32 kHz, 4-level codebook) with class-conditioned next-token prediction.
2. **AudioGen + Species Conditioner** -- Meta's AudioGen-medium (EnCodec-based) fine-tuned with a learned species embedding that replaces the original text conditioner. Training uses a three-stage freeze schedule with optional LoRA.
3. **Ecogen VQ-VAE** -- A VQ-VAE operating on mel spectrograms, generating new samples via latent-space augmentation (noise injection, interpolation, and latent sampling).

The training data comes from the [BirdSet XCM](https://huggingface.co/datasets/DBD-research-group/BirdSet) dataset and optionally [BirdCLEF](https://www.kaggle.com/competitions/birdclef-2026) competition data, with quality gating via a ConvNeXt classifier.

Objective evaluation uses Inception Score (IS), Frechet Audio Distance (FAD), and classification accuracy via BirdNET, ConvNeXt, and EnCodec embedders. Subjective evaluation is conducted through MOS listening tests delivered via Telegram.

---

## Repository Structure

```
ar-bird-vocalization/
├── config.yaml                 # Central configuration (hyperparameters, paths, etc.)
├── config.py                   # Typed Python access to config.yaml
├── pyproject.toml              # Project metadata and dependencies
│
├── preprocessing/              # Data preparation
│   ├── pipeline.py             # XCM dataset download, filtering, chunking, train/val/test split
│   ├── tokenize.py             # SNAC and EnCodec tokenization of audio segments
│   └── upload_tokens.py        # Upload tokenized data to S3
│
├── birdclef_preprocessing/     # BirdCLEF-specific data pipeline
│   ├── run.py                  # Orchestrator: segments + gating + XCM enrichment
│   ├── metadata.py             # Build candidate segments from BirdCLEF audio/CSV
│   ├── gating.py               # ConvNeXt-based quality gating (top-1 prob + entropy)
│   ├── judge.py                # BirdClassifier wrapper for ConvNeXt
│   └── xcm_enrich.py           # Merge BirdCLEF segments with XCM data under quota rules
│
├── models/                     # Model definitions
│   ├── backbone.py             # GPT-2 / LLaMA factories with SNAC vocabulary
│   ├── audiogen.py             # AudioGen loading, SpeciesConditioner, stage freeze helpers
│   └── lora.py                 # Custom LoRA implementation (apply, freeze, merge, summary)
│
├── audio_datasets/             # PyTorch datasets
│   ├── snac_dataset.py         # SNACTokenDataset for backbone models
│   └── encodec_dataset.py      # EnCodecTokenDataset for AudioGen
│
├── pretrainer/                 # Pretraining on XCM data
│   ├── backbone_pretrainer.py  # SNAC token LM training (LLaMA/GPT-2)
│   └── audiogen_pretrainer.py  # Three-stage AudioGen training with LoRA
│
├── finetuner/                  # Fine-tuning on BirdCLEF data
│   ├── backbone_finetuner.py   # SNAC backbone fine-tuning
│   └── audiogen_finetuner.py   # Three-stage AudioGen fine-tuning (full LM)
│
├── generator/                  # Inference / generation
│   ├── llama_generator.py      # LLaMA/GPT-2 backbone generation + SNAC decode
│   ├── audiogen_generator.py   # AudioGen generation (species-conditioned or text-prompted)
│   └── ecogen_generator.py     # VQ-VAE mel-spectrogram generation
│
├── evaluation/                 # Objective evaluation
│   ├── generate_samples.py     # Batch generation of eval samples with optional reranking
│   ├── evaluate.py             # IS / FAD / accuracy computation
│   ├── embeddings.py           # BirdNET, ConvNeXt, EnCodec embedding extractors
│   └── metrics.py              # Metric implementations (IS, FAD, top-k accuracy)
│
├── reranker/
│   └── reranker.py             # Generate k candidates, score with classifier, pick best
│
├── ecogen/                     # VQ-VAE model
│   ├── vqvae.py                # Hierarchical VQ-VAE encoder/decoder
│   └── generate.py             # Standalone latent-space augmentation
│
├── subjective_eval/            # MOS listening tests
│   ├── prepare_listening_test.py
│   ├── build_survey.py
│   ├── analyze_responses.py
│   └── telegram_survey.py
│
├── scripts/                    # Utility scripts (figures, visualization)
├── utils/                      # Checkpoint I/O, audio loading helpers
├── vast_setup.sh               # Cloud GPU instance setup script
└── clean_cache.sh              # Remove __pycache__ and build artifacts
```

---

## Models

### 1. LLaMA/GPT-2 Backbone + SNAC

A causal language model trained from scratch on interleaved [SNAC](https://github.com/hubertsiuzdak/snac) tokens. SNAC encodes 32 kHz audio into 4 hierarchical codebook levels with 4096 entries each, interleaved into 15 tokens per time step.

| Parameter        | Default                                           |
| ---------------- | ------------------------------------------------- |
| Architecture     | LLaMA (configurable to GPT-2)                     |
| Hidden dimension | 768                                               |
| Layers           | 12                                                |
| Attention heads  | 12                                                |
| Context length   | 1664 tokens                                       |
| Vocabulary       | 16384 SNAC tokens + special tokens + class tokens |

The model is conditioned on species identity via a class token prepended before the BOS token. Generation uses autoregressive sampling with configurable temperature and top-k.

### 2. AudioGen + Species Conditioner

[AudioGen-medium](https://huggingface.co/facebook/audiogen-medium) (Meta's `audiocraft`) is a transformer language model operating on EnCodec tokens at 16 kHz. The original text conditioner is replaced with a learned `SpeciesConditioner` (species embedding + linear projector).

Training follows a three-stage schedule:

| Stage | What is trained                      | Default epochs | Batch size | LR   |
| ----- | ------------------------------------ | -------------- | ---------- | ---- |
| 1     | Species conditioner only (LM frozen) | 20             | 8          | 1e-3 |
| 2     | LoRA adapters + conditioner          | 50             | 4          | 2e-4 |
| 3     | LoRA adapters + conditioner          | 50             | 4          | 5e-5 |

LoRA (rank 16, alpha 32) is applied to `out_proj`, `linear1`, and `linear2` layers in the transformer. The fine-tuning path alternatively supports full-LM unfreezing without LoRA.

### 3. Ecogen VQ-VAE

A hierarchical VQ-VAE (Sonnet/VQ-VAE-2 style) operating on mel spectrograms. New samples are generated by augmenting latent codes via noise injection, interpolation between same-class latents, or sampling from per-class latent statistics.

### Evaluation Models

- **ConvNeXt** (`DBD-research-group/ConvNeXT-Base-BirdSet-XCL`) -- Log-mel classifier for gating, reranking, and accuracy evaluation.
- **BirdNET** (Cornell, Global 11K) -- Used as an alternative embedder for IS/FAD/accuracy.
- **EnCodec embedder** -- AudioGen's compression encoder pooled features for FAD computation.

---

## Compute Requirements

All training and inference scripts require a CUDA-capable GPU. Below are the minimum VRAM estimates:

| Task                                    | Minimum GPU VRAM | Notes                                                                       |
| --------------------------------------- | ---------------- | --------------------------------------------------------------------------- |
| **LLaMA backbone training** (bs=16)     | ~12 GB           | 12-layer, 768-dim model on SNAC tokens                                      |
| **LLaMA backbone inference**            | ~4 GB            | bf16 enabled by default                                                     |
| **AudioGen pretraining stage 1** (bs=8) | ~16 GB           | Only conditioner parameters are trained; LM is in eval mode                 |
| **AudioGen pretraining stages 2-3**     | ~48 GB           | LoRA keeps base weights frozen; gradient accumulation can reduce batch size |
| **AudioGen fine-tuning**                | ~48 GB           | All LM parameters unfrozen                                                  |
| **AudioGen inference**                  | ~8 GB            | bf16 optional; single-sample generation                                     |
| **Ecogen VQ-VAE inference**             | ~4 GB            | Mel-spectrogram operations; CPU also supported                              |
| **SNAC/EnCodec tokenization**           | ~6 GB            | Batch encoding of audio segments                                            |
| **Evaluation (ConvNeXt/BirdNET)**       | ~4 GB            | Embedding extraction + metric computation                                   |

**Recommended setup:** A single GPU with 24 GB VRAM (e.g. RTX 3090, RTX 4090, A5000, or similar). AudioGen full fine-tuning benefits from 40+ GB (A100/A6000). Gradient accumulation (`audiogen.grad_accum` in config) allows trading batch size for memory. bf16 is used by default where supported.

**CPU/RAM:** 32 GB system RAM recommended. Data loading uses 8 workers by default (`num_workers` in config).

---

## Installation

**Prerequisites:** Python >= 3.10, CUDA-capable GPU, [`uv`](https://docs.astral.sh/uv/) package manager.

```bash
git clone https://github.com/hamuladm/ar-bird-vocalization.git
cd ar-bird-vocalization
uv sync
```

This installs all dependencies from `pyproject.toml` into a managed virtual environment. All commands must be run via `uv run python` to use the correct environment.

### Cloud GPU Setup

For cloud instances (e.g. Vast.ai), a setup script is provided:

```bash
bash vast_setup.sh
```

This clones the repo, installs dependencies, downloads the XCM dataset, installs the AWS CLI, and syncs pretrained data from S3.

---

## Configuration

All hyperparameters, paths, and model settings are controlled through `config.yaml`. Key sections:

```yaml
device: "cuda"

audio:
  sample_rate: 32000 # SNAC path sample rate
  chunk_length: 10 # Segment duration in seconds
  fade_sec: 1.0 # Fade-in/out for chunks
  min_chunk_sec: 0.1 # Minimum valid chunk length

model:
  backbone: "llama" # "llama" or "gpt2"
  n_embd: 768
  n_layer: 12
  n_head: 12
  n_positions: 1664

snac:
  model: "hubertsiuzdak/snac_32khz"
  codebook_size: 4096
  n_levels: 4
  generation:
    temperature: 1.0
    top_k: 50

audiogen:
  pretrained: "facebook/audiogen-medium"
  sample_rate: 16000 # EnCodec path sample rate
  target_length: 10 # Target audio duration in seconds
  generation:
    duration: 10.0
    temperature: 1.0
    top_k: 250
    cfg_coef: 3.0 # Classifier-free guidance coefficient
```

Settings are accessed in Python via `config.py`, which exports typed constants (e.g. `SAMPLE_RATE`, `BACKBONE`, `AG_STAGE2`, etc.).

---

## Data Preparation

### Step 1: Segment Extraction (XCM)

Downloads the BirdSet XCM dataset, filters classes with sufficient samples, chunks recordings, and splits into train/val/test:

```bash
uv run python -m preprocessing.pipeline
```

| Flag              | Default         | Description                           |
| ----------------- | --------------- | ------------------------------------- |
| `--min-samples`   | 50              | Minimum segments per class to include |
| `--chunk-sec`     | 10              | Target chunk duration (seconds)       |
| `--min-chunk-sec` | 0.1             | Minimum valid chunk duration          |
| `--output-dir`    | `data/segments` | Output directory for segment JSONs    |

Produces `{train,val,test}_segments.json` and `ebird_to_id.json`.

### Step 2: BirdCLEF Preprocessing (optional)

For fine-tuning on BirdCLEF competition data with quality gating:

```bash
uv run python -m birdclef_preprocessing.run \
    --data-dir data/birdclef-2026 \
    --output-dir data/birdclef_segments
```

| Flag                             | Default                                        | Description                              |
| -------------------------------- | ---------------------------------------------- | ---------------------------------------- |
| `--no-gating`                    | false                                          | Skip ConvNeXt quality gating             |
| `--min-top1-prob`                | 0.7                                            | Minimum classifier confidence for gating |
| `--max-entropy`                  | 4.0                                            | Maximum prediction entropy for gating    |
| `--gating-batch-size`            | 32                                             | Batch size for ConvNeXt gating           |
| `--checkpoint`                   | `DBD-research-group/ConvNeXT-Base-BirdSet-XCL` | HuggingFace classifier checkpoint        |
| `--xcm-enrich`                   | false                                          | Merge with XCM segments under quota      |
| `--xcm-quota-mode`               | `fixed_per_class`                              | `fixed_per_class` or `birdclef_train`    |
| `--xcm-extra-segments-per-class` | 1000                                           | XCM segments allowed per class           |
| `--pretrain-segment-dir`         | `pretrain_data/segments`                       | Holdout set for leakage prevention       |

### Step 3: Tokenization

Encode audio segments into discrete tokens for training:

**SNAC tokens** (for LLaMA/GPT-2 backbone):

```bash
uv run python -m preprocessing.tokenize --codec snac --split train
uv run python -m preprocessing.tokenize --codec snac --split val
uv run python -m preprocessing.tokenize --codec snac --split test
```

**EnCodec tokens** (for AudioGen):

```bash
uv run python -m preprocessing.tokenize --codec encodec --split train
uv run python -m preprocessing.tokenize --codec encodec --split val
uv run python -m preprocessing.tokenize --codec encodec --split test
```

| Flag            | Default         | Description                                                |
| --------------- | --------------- | ---------------------------------------------------------- |
| `--codec`       | (required)      | `snac` or `encodec`                                        |
| `--split`       | `train`         | `train`, `val`, or `test`                                  |
| `--segment-dir` | `data/segments` | Input segment JSON directory                               |
| `--output-dir`  | auto            | SNAC: `data/snac_tokens/`, EnCodec: `data/encodec_tokens/` |

### Optional: Upload Tokens to S3

```bash
uv run python -m preprocessing.upload_tokens --codec snac --split train val test
```

---

## Training

### LLaMA/GPT-2 Backbone Pretraining

Train the SNAC token language model from scratch:

```bash
uv run python -m pretrainer.backbone_pretrainer --wandb
```

| Flag                   | Default | Description                                              |
| ---------------------- | ------- | -------------------------------------------------------- |
| `--resume`             | None    | Path to checkpoint to resume from                        |
| `--wandb`              | off     | Enable Weights & Biases logging                          |
| `--sample-classes`     | None    | Specific class IDs to train on (space-separated)         |
| `--num-sample-classes` | 3       | Number of classes to sample audio from during validation |

Hyperparameters from `config.yaml` (`pretrain` section): epochs=2, batch_size=16, lr=1e-4, warmup=1000 steps, AdamW + cosine schedule.

Checkpoints saved to `checkpoints/gpt2/`.

### LLaMA/GPT-2 Backbone Fine-tuning

Fine-tune a pretrained backbone on BirdCLEF data:

```bash
uv run python -m finetuner.backbone_finetuner \
    --load-from checkpoints/gpt2/best.pt \
    --epochs 50 \
    --batch-size 4 \
    --lr 5e-5 \
    --wandb
```

| Flag                   | Default | Description                                    |
| ---------------------- | ------- | ---------------------------------------------- |
| `--load-from`          | None    | Pretrained checkpoint to initialize from       |
| `--resume`             | None    | Resume fine-tuning from a fine-tune checkpoint |
| `--epochs`             | 50      | Number of fine-tuning epochs                   |
| `--batch-size`         | 4       | Batch size                                     |
| `--lr`                 | 5e-5    | Learning rate                                  |
| `--warmup-steps`       | 500     | Warmup steps                                   |
| `--grad-accum`         | 1       | Gradient accumulation steps                    |
| `--wandb`              | off     | Enable W&B logging                             |
| `--sample-classes`     | None    | Specific class IDs                             |
| `--num-sample-classes` | 3       | Classes to sample during validation            |

Checkpoints saved to `checkpoints/gpt2_finetune/`.

### AudioGen Pretraining

Three-stage training of AudioGen with species conditioning:

```bash
# Run all stages sequentially
uv run python -m pretrainer.audiogen_pretrainer --wandb

# Or run a specific stage
uv run python -m pretrainer.audiogen_pretrainer --stage 2 --wandb

# Resume from a checkpoint
uv run python -m pretrainer.audiogen_pretrainer --stage 2 \
    --resume checkpoints/audiogen/pretrain/stage2/latest.pt --wandb
```

| Flag          | Default | Description                                         |
| ------------- | ------- | --------------------------------------------------- |
| `--stage`     | None    | Run a specific stage (1, 2, or 3). Omit to run all. |
| `--resume`    | None    | Resume from a stage checkpoint                      |
| `--load-from` | None    | Initialize from a different checkpoint              |
| `--wandb`     | off     | Enable W&B logging                                  |

Stage hyperparameters are configured in `config.yaml` under `audiogen.stage1`, `stage2`, `stage3`. LoRA configuration (rank, alpha, enabled) is per-stage.

Checkpoints saved to `checkpoints/audiogen/pretrain/stage{1,2,3}/`.

### AudioGen Fine-tuning

Fine-tune AudioGen on BirdCLEF data (full LM unfreezing, no LoRA):

```bash
uv run python -m finetuner.audiogen_finetuner --stage 1 --wandb

uv run python -m finetuner.audiogen_finetuner --stage 2 \
    --load-from checkpoints/audiogen/pretrain/stage2/best.pt --wandb
```

| Flag                   | Default | Description                           |
| ---------------------- | ------- | ------------------------------------- |
| `--stage`              | None    | Stage (1, 2, or 3)                    |
| `--resume`             | None    | Resume from fine-tune checkpoint      |
| `--load-from`          | None    | Initialize from pretrained checkpoint |
| `--wandb`              | off     | Enable W&B logging                    |
| `--sample-classes`     | None    | Specific class IDs                    |
| `--num-sample-classes` | 3       | Classes for validation sampling       |

Checkpoints saved to `checkpoints/audiogen/finetune/stage{1,2,3}/`.

---

## Inference

### LLaMA/GPT-2 Generation

Generate bird vocalizations using the SNAC backbone:

```bash
# List available species
uv run python -m generator.llama_generator \
    --checkpoint checkpoints/gpt2/best.pt --list-classes

# Generate samples for a specific species
uv run python -m generator.llama_generator \
    --checkpoint checkpoints/gpt2/best.pt \
    --class-name "Parus major" \
    --num-samples 5 \
    --output generated/llama
```

| Flag             | Default     | Description                                        |
| ---------------- | ----------- | -------------------------------------------------- |
| `--checkpoint`   | (required)  | Path to model checkpoint                           |
| `--output`       | `generated` | Output directory for WAV files                     |
| `--class-id`     | None        | Species class ID (integer)                         |
| `--class-name`   | None        | Species name (eBird code or common name)           |
| `--num-samples`  | 1           | Number of samples to generate                      |
| `--list-classes` | off         | Print available classes and exit                   |
| `--no-bf16`      | off         | Disable bfloat16 (bf16 is used by default on CUDA) |

Generation parameters (temperature, top_k) are set in `config.yaml` under `snac.generation`.

### AudioGen Generation

Generate bird vocalizations using the fine-tuned AudioGen:

```bash
# Species-conditioned generation
uv run python -m generator.audiogen_generator \
    --checkpoint checkpoints/audiogen/pretrain/stage3/best.pt \
    --class-name "Turdus merula" \
    --num-samples 5 \
    --output generated/audiogen

# With LoRA overrides
uv run python -m generator.audiogen_generator \
    --checkpoint checkpoints/audiogen/pretrain/stage3/best.pt \
    --class-id 42 \
    --lora-rank 16 --lora-alpha 32 \
    --bf16
```

| Flag             | Default              | Description                                        |
| ---------------- | -------------------- | -------------------------------------------------- |
| `--checkpoint`   | (required)           | Path to model checkpoint                           |
| `--output`       | `generated/audiogen` | Output directory                                   |
| `--class-id`     | None                 | Species class ID                                   |
| `--class-name`   | None                 | Species name                                       |
| `--num-samples`  | 1                    | Number of samples to generate                      |
| `--list-classes` | off                  | Print available classes and exit                   |
| `--lora-rank`    | auto                 | LoRA rank override (auto-detected from checkpoint) |
| `--lora-alpha`   | auto                 | LoRA alpha override                                |
| `--bf16`         | off                  | Enable bfloat16 inference                          |

Generation parameters (duration, temperature, top_k, cfg_coef) are set in `config.yaml` under `audiogen.generation`.

### Text-Prompted AudioGen

The `AudiogenGenerator` also supports a text-prompt mode using the pretrained AudioGen text conditioner with descriptive/scientific/common species name templates.

### Ecogen VQ-VAE Generation

```bash
uv run python -m ecogen.generate \
    --data_paths "data/encodec_tokens/train/*.npz" \
    --out_folder generated/ecogen \
    --model_path checkpoints/ecogen/vqvae.pt \
    --augmentations noise,interpolation \
    --num_samples 5 \
    --device cuda
```

| Flag              | Default    | Description                               |
| ----------------- | ---------- | ----------------------------------------- |
| `--data_paths`    | (required) | Glob pattern for input token files        |
| `--out_folder`    | (required) | Output directory                          |
| `--model_path`    | (required) | VQ-VAE checkpoint path                    |
| `--augmentations` | `noise`    | Comma-separated: `noise`, `interpolation` |
| `--num_samples`   | 1          | Samples per input                         |
| `--device`        | `cpu`      | `cpu` or `cuda`                           |

---

## Evaluation

### Batch Sample Generation

Generate evaluation samples for all test species with optional reranking:

```bash
uv run python -m evaluation.generate_samples \
    --checkpoint checkpoints/audiogen/pretrain/stage3/best.pt \
    --model-type audiogen \
    --stage pretrain \
    --test-segments data/segments/test_segments.json \
    --n-per-class 10 \
    --k 8 \
    --discriminator convnext \
    --device cuda --bf16
```

| Flag                | Default      | Description                                                      |
| ------------------- | ------------ | ---------------------------------------------------------------- |
| `--checkpoint`      | (required\*) | Model checkpoint (\* not required for `audiogen_text`)           |
| `--model-type`      | (required)   | `audiogen`, `audiogen_text`, `llama`, or `ecogen`                |
| `--stage`           | (required)   | `pretrain`, `finetune`, or `baseline`                            |
| `--test-segments`   | (required)   | Path to test segments JSON                                       |
| `--n-per-class`     | 10           | Number of final samples per class                                |
| `--k`               | 8            | Candidates generated per sample (for reranking)                  |
| `--output`          | auto         | Output directory                                                 |
| `--device`          | `cuda`       | Device                                                           |
| `--bf16`            | off          | Enable bfloat16                                                  |
| `--lora-rank`       | None         | LoRA rank override                                               |
| `--lora-alpha`      | None         | LoRA alpha override                                              |
| `--temperature`     | from config  | Sampling temperature                                             |
| `--top-k`           | from config  | Top-k sampling                                                   |
| `--cfg-coef`        | from config  | Classifier-free guidance coefficient                             |
| `--duration`        | from config  | Audio duration in seconds                                        |
| `--discriminator`   | `birdnet`    | Reranker classifier: `birdnet` or `convnext`                     |
| `--no-reranker`     | off          | Disable reranking (use first candidate)                          |
| `--prompt-template` | None         | Text template for `audiogen_text` mode                           |
| `--augmentation`    | None         | Ecogen augmentation: `noise`, `interpolation`, `latent_sampling` |
| `--ratio`           | None         | Augmentation mixing ratio                                        |
| `--latent-stats`    | None         | Path to latent stats file (for `latent_sampling`)                |

### Objective Metrics

Compute IS, FAD, and classification accuracy:

```bash
# From pre-generated directories
uv run python -m evaluation.evaluate \
    --generated-dir eval_generated_samples/audiogen_pretrain/ \
    --test-segments data/segments/test_segments.json \
    --metrics is,fad,acc \
    --embedder convnext \
    --device cuda

# From a checkpoint (generates on-the-fly)
uv run python -m evaluation.evaluate \
    --checkpoint checkpoints/audiogen/pretrain/stage3/best.pt \
    --test-segments data/segments/test_segments.json \
    --metrics is,fad,acc \
    --embedder birdnet
```

| Flag                      | Default      | Description                                  |
| ------------------------- | ------------ | -------------------------------------------- |
| `--generated-dir`         | None         | Directory of generated WAV files             |
| `--reference-dir`         | None         | Directory of reference WAV files             |
| `--checkpoint`            | None         | Generate samples from checkpoint             |
| `--test-segments`         | None         | Test segments JSON                           |
| `--metrics`               | `is,fad,acc` | Comma-separated: `is`, `fad`, `acc`          |
| `--embedder`              | `birdnet`    | `birdnet`, `convnext`, or `encodec`          |
| `--batch-size`            | from config  | Batch size for embedding extraction          |
| `--restrict-classes`      | None         | JSON file with eBird codes for restricted IS |
| `--output`                | None         | Path to save JSON results                    |
| `--device`                | from config  | Device                                       |
| `--num-samples-per-class` | None         | Samples per class (checkpoint mode)          |
| `--max-classes`           | None         | Limit number of evaluated classes            |

---

## Subjective Listening Test

### Prepare Samples

Normalize and select samples for listening tests:

```bash
uv run python -m subjective_eval.prepare_listening_test \
    --output subjective_eval/mos_samples \
    --seed 42
```

### Build Survey

Generate a randomized survey order and HTML listening test page:

```bash
uv run python -m subjective_eval.build_survey \
    --samples-dir subjective_eval/mos_samples \
    --seed 42
```

### Telegram Survey

Deploy MOS polls via Telegram:

```bash
# Post polls
uv run python -m subjective_eval.telegram_survey \
    --token $TELEGRAM_BOT_TOKEN \
    --channel @your_channel \
    --csv subjective_eval/survey_order.csv \
    --delay 5

# Collect responses
uv run python -m subjective_eval.telegram_survey \
    --token $TELEGRAM_BOT_TOKEN \
    --collect
```

### Analyze Results

```bash
uv run python -m subjective_eval.analyze_responses \
    subjective_eval/mos_responses.json \
    --order subjective_eval/survey_order.csv
```

---

## Dependencies

Core dependencies (from `pyproject.toml`):

| Package               | Version   | Purpose                                        |
| --------------------- | --------- | ---------------------------------------------- |
| `torch`               | >= 2.9.1  | Deep learning framework                        |
| `torchaudio`          | >= 2.9.1  | Audio I/O and transforms                       |
| `transformers`        | >= 4.57.3 | GPT-2, LLaMA, ConvNeXt model architectures     |
| `audiocraft`          | >= 1.1.0  | Meta's AudioGen and EnCodec                    |
| `snac`                | >= 1.2.1  | SNAC audio codec (32 kHz, 4-level)             |
| `datasets`            | < 4.0.0   | HuggingFace Datasets (BirdSet XCM loading)     |
| `wandb`               | >= 0.16.0 | Experiment tracking                            |
| `scikit-learn`        | >= 1.7.2  | Data splitting utilities                       |
| `scipy`               | >= 1.11.0 | Signal processing, FAD computation             |
| `pandas`              | >= 2.0.0  | Metadata and CSV handling                      |
| `boto3`               | >= 1.35.0 | S3 manipulation for tokenized data             |
| `pyloudnorm`          | >= 0.2.0  | Audio loudness normalization (listening tests) |
| `python-telegram-bot` | >= 22.7   | Telegram MOS survey delivery                   |
| `lightning`           | >= 2.1.0  | (dependency; training loops are custom)        |
| `tqdm`                | >= 4.67.1 | Progress bars                                  |
| `ruff`                | >= 0.15.7 | Linter/formatter                               |

**Python:** >= 3.10 required.

All dependencies are managed via `uv` and locked in `uv.lock`. Install with `uv sync`.
