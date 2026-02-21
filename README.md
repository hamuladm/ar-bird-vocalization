# AR Bird Vocalization

Autoregressive generation of bird vocalizations using GPT-2 and SNAC audio tokens (for now).

## Overview

This project trains a GPT-2 language model on tokenized bird audio to generate novel bird vocalizations. Audio is encoded into discrete tokens using [SNAC](https://github.com/hubertsiuzdak/snac) (32kHz), and a class-conditioned GPT-2 model learns to predict token sequences autoregressively.

## Pipeline

1. **Preprocessing** (`preprocessing/`) — Download and filter bird audio segments using a BirdSet classifier, ensuring quality and class balance.
2. **SNAC Encoding** (`snac_inference.py`) — Encode filtered audio into flattened SNAC token sequences (4 codebook levels interleaved into 15 tokens per time step).
3. **Training** (`train.py`) — Train a class-conditioned GPT-2 model on the SNAC token sequences.
4. **Generation** (`generate.py`) — Sample new token sequences from the trained model and decode them back to audio via SNAC.

## Usage

1. Run preprocessing pipeline
```python
uv run python -m preprocessing.pipeline
```

2. Encode audio to SNAC tokens
```python
uv run python snac_inference.py --split train
```

3. Train the model
```python
uv run python train.py
```

4. Generate bird audio
```python
uv run python generate.py --checkpoint checkpoints/best_model.pt
```