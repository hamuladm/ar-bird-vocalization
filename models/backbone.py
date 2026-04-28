import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from config import (
    CODEBOOK_SIZE,
    SNAC_N_LEVELS,
    MAX_SEQ_LEN,
    N_EMBD,
    N_LAYER,
    N_HEAD,
    N_POSITIONS,
    BACKBONE,
    INTERMEDIATE_SIZE,
)

SNAC_VOCAB_SIZE = SNAC_N_LEVELS * CODEBOOK_SIZE
PAD_TOKEN = SNAC_VOCAB_SIZE
BOS_TOKEN = SNAC_VOCAB_SIZE + 1
EOS_TOKEN = SNAC_VOCAB_SIZE + 2
CLASS_TOKEN_OFFSET = SNAC_VOCAB_SIZE + 3


def _default_intermediate_size(n_embd):
    raw = int(2 * (4 * n_embd) / 3)
    return ((raw + 255) // 256) * 256


def create_gpt2_model(
    vocab_size, n_positions=N_POSITIONS, n_embd=N_EMBD, n_layer=N_LAYER, n_head=N_HEAD
):
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


def create_llama_model(
    vocab_size,
    n_positions=N_POSITIONS,
    n_embd=N_EMBD,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    intermediate_size=INTERMEDIATE_SIZE,
):
    if intermediate_size is None:
        intermediate_size = _default_intermediate_size(n_embd)

    llama_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=n_embd,
        intermediate_size=intermediate_size,
        num_hidden_layers=n_layer,
        num_attention_heads=n_head,
        num_key_value_heads=n_head,
        max_position_embeddings=n_positions,
        bos_token_id=BOS_TOKEN,
        eos_token_id=EOS_TOKEN,
        pad_token_id=PAD_TOKEN,
        tie_word_embeddings=True,
    )
    return LlamaForCausalLM(llama_config)


_BACKBONE_CREATORS = {
    "gpt2": create_gpt2_model,
    "llama": create_llama_model,
}


def create_model(backbone=BACKBONE, **kwargs):
    return _BACKBONE_CREATORS[backbone](**kwargs)


@torch.no_grad()
def generate_tokens(
    model, device, class_id, max_length=MAX_SEQ_LEN, temperature=1.0, top_k=50
):
    model.eval()
    n_positions = model.config.max_position_embeddings

    cls_token = CLASS_TOKEN_OFFSET + class_id
    input_ids = torch.tensor([[cls_token, BOS_TOKEN]], device=device, dtype=torch.long)

    past_key_values = None

    for _ in range(max_length - 2):
        if past_key_values is None:
            context = (
                input_ids
                if input_ids.shape[1] <= n_positions
                else input_ids[:, -n_positions:]
            )
            outputs = model(context, use_cache=True)
            past_key_values = outputs.past_key_values
        else:
            token_in = input_ids[:, -1:]
            outputs = model(
                token_in,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

        logits = outputs.logits[:, -1, :] / temperature

        if top_k > 0:
            topk_vals = torch.topk(logits, top_k).values
            logits = logits.clone()
            logits[logits < topk_vals[:, -1:]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == EOS_TOKEN:
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if input_ids.shape[1] > n_positions:
            past_key_values = None

    return input_ids[0].cpu().numpy()


def extract_snac_codes(tokens):
    return tokens[(tokens >= 0) & (tokens < SNAC_VOCAB_SIZE)]
