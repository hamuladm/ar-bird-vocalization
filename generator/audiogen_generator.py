import csv
import numpy as np
import torch
import torchaudio
from pathlib import Path

from audiocraft.models import AudioGen
from audiocraft.utils.autocast import TorchAutocast

from config import (
    DEVICE,
    AG_PRETRAINED,
    AG_GEN_DURATION,
    AG_GEN_TEMPERATURE,
    AG_GEN_TOP_K,
    AG_GEN_CFG_COEF,
    AG_STAGE2,
    AG_STAGE3,
)
from models.audiogen import load_audiogen, make_species_conditions
from models.lora import apply_lora

_TAXONOMY_PATH = (
    Path(__file__).parent.parent / "data_relaxed" / "taxonomy" / "ebird_taxonomy.csv"
)


def _lora_rank_from_state_dict(sd):
    for key, tensor in sd.items():
        if key.endswith(".lora_A"):
            return int(tensor.shape[0])
    return None


def _lora_alpha_for_checkpoint(ckpt):
    if "lora_alpha" in ckpt:
        return float(ckpt["lora_alpha"])
    stage = ckpt.get("stage")
    if stage == 3:
        return float(AG_STAGE3.lora_alpha)
    return float(AG_STAGE2.lora_alpha)


class AudiogenGenerator:
    def __init__(
        self,
        checkpoint_path,
        device=DEVICE,
        lora_rank=None,
        lora_alpha=None,
        use_bf16=False,
    ):
        self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        lm_sd = ckpt["lm_state_dict"]
        ckpt.pop("optimizer_state_dict", None)
        ckpt.pop("scheduler_state_dict", None)

        self.ebird_to_id = ckpt["ebird_to_id"]
        self.id_to_ebird = {i: c for c, i in self.ebird_to_id.items()}
        self.n_species = ckpt["n_species"]

        self.audiogen, _ = load_audiogen(self.n_species, device=str(self.device))

        inferred_rank = _lora_rank_from_state_dict(lm_sd)
        if inferred_rank is not None:
            if lora_rank is not None:
                rank = lora_rank
            elif "lora_rank" in ckpt:
                rank = int(ckpt["lora_rank"])
            else:
                rank = inferred_rank
            alpha = (
                float(lora_alpha)
                if lora_alpha is not None
                else _lora_alpha_for_checkpoint(ckpt)
            )
            apply_lora(self.audiogen.lm, rank=rank, alpha=alpha)

        self.audiogen.lm.load_state_dict(lm_sd)
        self.audiogen.lm.eval()

        if use_bf16:
            if self.device.type != "cuda":
                raise ValueError("use_bf16=True requires CUDA.")
            self.audiogen.lm.to(torch.bfloat16)
            self.audiogen.autocast = TorchAutocast(
                enabled=True, device_type="cuda", dtype=torch.bfloat16
            )

        self.sample_rate = self.audiogen.sample_rate
        self.stage = ckpt.get("stage")
        self.epoch = ckpt.get("epoch")
        self.val_loss = ckpt.get("val_loss")
        self.used_lora = inferred_rank is not None

    @classmethod
    def from_model(cls, audiogen, ebird_to_id, device=DEVICE):
        obj = cls.__new__(cls)
        obj.device = torch.device(device)
        obj.audiogen = audiogen
        obj.ebird_to_id = ebird_to_id
        obj.id_to_ebird = {i: c for c, i in ebird_to_id.items()}
        obj.n_species = len(ebird_to_id)
        obj.sample_rate = audiogen.sample_rate
        obj.stage = None
        obj.epoch = None
        obj.val_loss = None
        obj.used_lora = False
        return obj

    @torch.no_grad()
    def generate(
        self,
        species_id,
        duration=AG_GEN_DURATION,
        temperature=AG_GEN_TEMPERATURE,
        top_k=AG_GEN_TOP_K,
        cfg_coef=AG_GEN_CFG_COEF,
    ):
        self.audiogen.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            cfg_coef=cfg_coef,
            use_sampling=True,
        )

        conditions = make_species_conditions([species_id])
        lm = self.audiogen.lm
        lm.eval()

        with self.audiogen.autocast:
            total_gen_len = int(duration * self.audiogen.frame_rate)
            gen_tokens = lm.generate(
                prompt=None,
                conditions=conditions,
                max_gen_len=total_gen_len,
                use_sampling=True,
                temp=temperature,
                top_k=top_k,
                cfg_coef=cfg_coef,
            )

        audio = self.audiogen.compression_model.decode(gen_tokens, None)
        return audio[0, 0].cpu().numpy()

    @torch.no_grad()
    def generate_batch(
        self,
        species_id,
        k,
        duration=AG_GEN_DURATION,
        temperature=AG_GEN_TEMPERATURE,
        top_k=AG_GEN_TOP_K,
        cfg_coef=AG_GEN_CFG_COEF,
    ):
        self.audiogen.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            cfg_coef=cfg_coef,
            use_sampling=True,
        )

        conditions = make_species_conditions([species_id] * k)
        lm = self.audiogen.lm
        lm.eval()

        with self.audiogen.autocast:
            total_gen_len = int(duration * self.audiogen.frame_rate)
            gen_tokens = lm.generate(
                prompt=None,
                conditions=conditions,
                max_gen_len=total_gen_len,
                use_sampling=True,
                temp=temperature,
                top_k=top_k,
                cfg_coef=cfg_coef,
            )

        audio = self.audiogen.compression_model.decode(gen_tokens, None)
        return [audio[i, 0].cpu().numpy() for i in range(audio.shape[0])]


def _load_taxonomy(path=_TAXONOMY_PATH):
    code_to_sci = {}
    code_to_common = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row.get("SPECIES_CODE", "").strip()
            sci = row.get("SCIENTIFIC_NAME", "").strip()
            common = row.get("COMMON_NAME", "").strip()
            if code:
                if sci:
                    code_to_sci[code] = sci
                if common:
                    code_to_common[code] = common
    return code_to_sci, code_to_common


class TextAudiogenGenerator:
    PROMPT_TEMPLATES = {
        "scientific": "Bird vocalization. {sci_name}.",
        "common": "Bird vocalization. {common_name}.",
        "descriptive": "Bird song in nature. {common_name} ({sci_name}).",
    }

    def __init__(
        self,
        ebird_codes,
        device=DEVICE,
        use_bf16=False,
        prompt_template="descriptive",
        taxonomy_path=_TAXONOMY_PATH,
    ):
        self.device = torch.device(device)
        self.audiogen = AudioGen.get_pretrained(AG_PRETRAINED, device=str(self.device))
        self.sample_rate = self.audiogen.sample_rate

        code_to_sci, code_to_common = _load_taxonomy(taxonomy_path)

        self.ebird_to_id = {code: i for i, code in enumerate(sorted(ebird_codes))}
        self.id_to_ebird = {i: code for code, i in self.ebird_to_id.items()}

        if prompt_template in self.PROMPT_TEMPLATES:
            self._template = self.PROMPT_TEMPLATES[prompt_template]
        else:
            self._template = prompt_template

        self._prompts = {}
        for code in ebird_codes:
            sci = code_to_sci.get(code, code)
            common = code_to_common.get(code, code)
            self._prompts[code] = self._template.format(
                sci_name=sci,
                common_name=common,
                ebird_code=code,
            )

        if use_bf16:
            if self.device.type != "cuda":
                raise ValueError("use_bf16=True requires CUDA.")
            self.audiogen.lm.to(torch.bfloat16)
            self.audiogen.autocast = TorchAutocast(
                enabled=True,
                device_type="cuda",
                dtype=torch.bfloat16,
            )

    def _prompt_for(self, species_id):
        code = self.id_to_ebird[species_id]
        return self._prompts[code]

    @torch.no_grad()
    def generate(
        self,
        species_id,
        duration=AG_GEN_DURATION,
        temperature=AG_GEN_TEMPERATURE,
        top_k=AG_GEN_TOP_K,
        cfg_coef=AG_GEN_CFG_COEF,
    ):
        self.audiogen.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            cfg_coef=cfg_coef,
            use_sampling=True,
        )
        prompt = self._prompt_for(species_id)
        wav = self.audiogen.generate([prompt])
        return wav[0, 0].cpu().numpy()

    @torch.no_grad()
    def generate_batch(
        self,
        species_id,
        k,
        duration=AG_GEN_DURATION,
        temperature=AG_GEN_TEMPERATURE,
        top_k=AG_GEN_TOP_K,
        cfg_coef=AG_GEN_CFG_COEF,
    ):
        self.audiogen.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            cfg_coef=cfg_coef,
            use_sampling=True,
        )
        prompt = self._prompt_for(species_id)
        wav = self.audiogen.generate([prompt] * k)
        return [wav[i, 0].cpu().numpy() for i in range(wav.shape[0])]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="generated/audiogen")
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--list-classes", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = AudiogenGenerator(
        args.checkpoint,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_bf16=args.bf16,
    )

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
            class_id if class_id is not None else np.random.randint(0, gen.n_species)
        )
        class_name = gen.id_to_ebird.get(sample_class, f"class_{sample_class}")

        audio = gen.generate(sample_class)
        wav = torch.from_numpy(audio).unsqueeze(0)

        fname = f"{class_name}_stage{gen.stage}_t{AG_GEN_TEMPERATURE}_{i:03d}.wav"
        filepath = output_dir / fname
        torchaudio.save(str(filepath), wav, gen.sample_rate)
        print(f"Saved: {filepath} ({audio.shape[0] / gen.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
