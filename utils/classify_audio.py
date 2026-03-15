import torch
import torch.nn.functional as F
import librosa
import numpy as np
from pathlib import Path

from config import DEVICE, FILTERED_DIR, SAMPLE_RATE, MAX_LENGTH, MODEL_CHECKPOINT
from preprocessing.judge import BirdClassifier
from preprocessing.code_translator import BirdTranslator
from utils.mapping_utils import load_ebird_mapping


def load_audio(path: str, target_sr: int = SAMPLE_RATE, max_length: float = MAX_LENGTH) -> torch.Tensor:
    audio, sr = librosa.load(path, sr=target_sr)
    target_samples = int(target_sr * max_length)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    elif len(audio) > target_samples:
        audio = audio[:target_samples]
    return torch.from_numpy(audio).float().unsqueeze(0)


def build_trained_class_mask(ebird_to_id, translator, num_xcl_classes, device):
    """Build a boolean mask: True for XCL indices that correspond to our 161 trained classes."""
    mask = torch.zeros(num_xcl_classes, dtype=torch.bool, device=device)
    for ebird_code in ebird_to_id:
        xcl_idx = translator._ebird_to_xcl_idx.get(ebird_code)
        if xcl_idx is not None:
            mask[xcl_idx] = True
    return mask


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading classifier ({MODEL_CHECKPOINT})...")
    classifier = BirdClassifier(device=DEVICE)
    translator = BirdTranslator(load_metadata=True)

    ebird_to_id, _ = load_ebird_mapping(Path(FILTERED_DIR))
    num_xcl = classifier.model.config.num_labels
    trained_mask = build_trained_class_mask(ebird_to_id, translator, num_xcl, DEVICE)
    print(f"Restricting to {trained_mask.sum().item()} / {num_xcl} XCL classes")

    audio_tensor = load_audio(args.audio)

    with torch.inference_mode():
        spec = classifier.preprocessor(audio_tensor.to(DEVICE))
        logits = classifier.model(spec).logits
        logits[:, ~trained_mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)

    topk = torch.topk(probs, k=args.top_k, dim=-1)

    print(f"\nFile: {args.audio}")
    print(f"{'Rank':<6} {'XCL ID':<8} {'eBird code':<16} {'Confidence':<12}")
    print("-" * 44)
    for i in range(args.top_k):
        xcl_idx = topk.indices[0, i].item()
        prob = topk.values[0, i].item()
        ebird = translator.xcl2ebird(xcl_idx) or "unknown"
        print(f"{i+1:<6} {xcl_idx:<8} {ebird:<16} {prob:.4f}")


if __name__ == "__main__":
    main()
