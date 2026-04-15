import logging
import math

import numpy as np
import torch
import torchaudio
from pathlib import Path
from snac import SNAC

from config import (
    DEVICE,
    SAMPLE_RATE,
    SNAC_MODEL,
    CODEBOOK_SIZE,
    MAX_SEQ_LEN,
    SNAC_GEN_TEMPERATURE,
    SNAC_GEN_TOP_K,
)
from models.backbone import (
    create_model,
    extract_snac_codes,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    CLASS_TOKEN_OFFSET,
)
from preprocessing.tokenize import unflatten_codes
from utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)


def _infer_vocab_size(state_dict, backbone):
    keys = {
        "gpt2": "transformer.wte.weight",
        "llama": "model.embed_tokens.weight",
    }
    key = keys.get(backbone)
    if key and key in state_dict:
        return state_dict[key].shape[0]
    for k, v in state_dict.items():
        if "embed" in k and "weight" in k and v.ndim == 2:
            return v.shape[0]
    raise RuntimeError("Cannot infer vocab_size from checkpoint weights")


def _decode_to_audio(snac_codes, snac_model, device):
    code_arrays = unflatten_codes(snac_codes)

    ws = snac_model.attn_window_size or 1
    stride0 = snac_model.vq_strides[0]
    align = ws // math.gcd(stride0, ws)
    coarse_len = len(code_arrays[0])
    coarse_len = (coarse_len // align) * align

    code_arrays = [
        code_arrays[0][:coarse_len],
        code_arrays[1][: coarse_len * 2],
        code_arrays[2][: coarse_len * 4],
        code_arrays[3][: coarse_len * 8],
    ]

    code_arrays = [np.clip(c, 0, CODEBOOK_SIZE - 1) for c in code_arrays]
    codes_torch = [
        torch.from_numpy(c).long().unsqueeze(0).to(device) for c in code_arrays
    ]
    with torch.inference_mode():
        audio = snac_model.decode(codes_torch)
    return audio[0, 0].cpu().numpy()


class LlamaGenerator:
    def __init__(self, checkpoint_path, device=DEVICE, use_bf16=True):
        self.device = torch.device(device)
        self.sample_rate = SAMPLE_RATE

        ckpt = load_checkpoint(checkpoint_path, device=self.device)

        self.ebird_to_id = ckpt["ebird_to_id"]
        self.id_to_ebird = {i: c for c, i in self.ebird_to_id.items()}
        self.n_classes = len(self.ebird_to_id)

        backbone = ckpt.get("backbone", "gpt2")
        vocab_size = _infer_vocab_size(ckpt["model_state_dict"], backbone)

        self.model = create_model(
            backbone=backbone, vocab_size=vocab_size, n_positions=MAX_SEQ_LEN
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device).eval()

        self.use_bf16 = use_bf16 and self.device.type == "cuda"
        if self.use_bf16:
            self.model = self.model.to(torch.bfloat16)
            logger.info("LlamaGenerator: model cast to bfloat16")

        self.snac_model = SNAC.from_pretrained(SNAC_MODEL).eval().to(self.device)

        self.epoch = ckpt.get("epoch")
        self.step = ckpt.get("global_step")
        self.val_loss = ckpt.get("val_loss")

    @classmethod
    def from_model(cls, model, snac_model, ebird_to_id, device=DEVICE):
        """Wrap an existing (possibly training) model. No bf16 cast."""
        obj = cls.__new__(cls)
        obj.device = torch.device(device)
        obj.sample_rate = SAMPLE_RATE
        obj.model = model
        obj.use_bf16 = False
        obj.snac_model = snac_model
        obj.ebird_to_id = ebird_to_id
        obj.id_to_ebird = {i: c for c, i in ebird_to_id.items()}
        obj.n_classes = len(ebird_to_id)
        obj.epoch = None
        obj.step = None
        obj.val_loss = None
        return obj

    def _build_prompt(self, class_id, batch_size=1):
        cls_token = CLASS_TOKEN_OFFSET + class_id
        prompt = torch.full(
            (batch_size, 2), BOS_TOKEN, device=self.device, dtype=torch.long,
        )
        prompt[:, 0] = cls_token
        return prompt

    @torch.no_grad()
    def generate(self, class_id, temperature=SNAC_GEN_TEMPERATURE, top_k=SNAC_GEN_TOP_K,
                 max_length=MAX_SEQ_LEN):
        prompt = self._build_prompt(class_id, batch_size=1)
        output = self.model.generate(
            prompt,
            max_new_tokens=max_length - 2,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=EOS_TOKEN,
            pad_token_id=PAD_TOKEN,
        )
        tokens = output[0].cpu().numpy()
        snac_codes = extract_snac_codes(tokens)
        if len(snac_codes) < 15:
            return None
        return _decode_to_audio(snac_codes, self.snac_model, self.device)

    @torch.no_grad()
    def generate_batch(self, class_id, k, temperature=SNAC_GEN_TEMPERATURE,
                       top_k=SNAC_GEN_TOP_K, max_length=MAX_SEQ_LEN):
        prompt = self._build_prompt(class_id, batch_size=k)
        outputs = self.model.generate(
            prompt,
            max_new_tokens=max_length - 2,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=EOS_TOKEN,
            pad_token_id=PAD_TOKEN,
        )
        waveforms = []
        for i in range(k):
            seq = outputs[i].cpu().numpy()
            eos_mask = seq == EOS_TOKEN
            if eos_mask.any():
                seq = seq[: eos_mask.argmax()]
            snac_codes = extract_snac_codes(seq)
            if len(snac_codes) < 15:
                continue
            waveforms.append(_decode_to_audio(snac_codes, self.snac_model, self.device))
        return waveforms


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="generated")
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--list-classes", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = LlamaGenerator(args.checkpoint, use_bf16=not args.no_bf16)

    if args.list_classes:
        for cid in sorted(gen.id_to_ebird.keys()):
            print(f"  {cid:4d}: {gen.id_to_ebird[cid]}")
        return

    if args.class_name:
        if args.class_name not in gen.ebird_to_id:
            print(f"Unknown class: {args.class_name}")
            return
        class_id = gen.ebird_to_id[args.class_name]
    elif args.class_id is not None:
        class_id = args.class_id
    else:
        class_id = None

    for i in range(args.num_samples):
        sample_class = (
            class_id if class_id is not None else np.random.randint(0, gen.n_classes)
        )
        class_name = gen.id_to_ebird.get(sample_class, f"class_{sample_class}")

        audio = gen.generate(sample_class)
        if audio is None:
            print(f"Too few tokens for {class_name}, skipping")
            continue

        filename = f"{class_name}_step{gen.step}_t{SNAC_GEN_TEMPERATURE}_{i:03d}.wav"
        filepath = output_dir / filename
        torchaudio.save(
            str(filepath), torch.from_numpy(audio).unsqueeze(0), gen.sample_rate
        )
        print(f"Saved: {filepath} ({len(audio) / gen.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
