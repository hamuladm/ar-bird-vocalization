import torch
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel

from config import (
    CODEBOOK_SIZE,
    SNAC_N_LEVELS,
    MAX_SEQ_LEN,
    N_EMBD,
    N_LAYER,
    N_HEAD,
    N_POSITIONS,
)

SNAC_VOCAB_SIZE = SNAC_N_LEVELS * CODEBOOK_SIZE
PAD_TOKEN = SNAC_VOCAB_SIZE
BOS_TOKEN = SNAC_VOCAB_SIZE + 1
EOS_TOKEN = SNAC_VOCAB_SIZE + 2
CLASS_TOKEN_OFFSET = SNAC_VOCAB_SIZE + 3


def create_gpt2_model(vocab_size, n_positions=N_POSITIONS, n_embd=N_EMBD, n_layer=N_LAYER, n_head=N_HEAD):
    gpt2_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=BOS_TOKEN,
        eos_token_id=EOS_TOKEN,
        pad_token_id=PAD_TOKEN,
    )
    return GPT2LMHeadModel(gpt2_config)


@torch.no_grad()
def generate_tokens(model, device, class_id, max_length=MAX_SEQ_LEN, temperature=1.0, top_k=50):
    model.eval()
    n_positions = model.config.n_positions

    cls_token = CLASS_TOKEN_OFFSET + class_id
    input_ids = torch.tensor([[cls_token, BOS_TOKEN]], device=device)

    for _ in range(max_length - 2):
        context = input_ids if input_ids.shape[1] <= n_positions else input_ids[:, -n_positions:]
        outputs = model(context)
        logits = outputs.logits[:, -1, :] / temperature

        if top_k > 0:
            topk_vals = torch.topk(logits, top_k).values
            logits[logits < topk_vals[:, -1:]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == EOS_TOKEN:
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids[0].cpu().numpy()


def extract_snac_codes(tokens):
    return tokens[(tokens >= 0) & (tokens < SNAC_VOCAB_SIZE)]
