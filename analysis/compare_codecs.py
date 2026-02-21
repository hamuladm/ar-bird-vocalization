"""Compare EnCodec 24kHz vs SNAC 24kHz reconstruction quality on XCM samples.

Both codecs operate at 24 kHz for a fair comparison. Audio is loaded at its
native sample rate, resampled to 24 kHz for codec processing, and the
reconstructed waveforms are resampled back to native SR for comparison.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torchaudio.transforms as T
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import EncodecModel, AutoProcessor
from snac import SNAC
from tqdm import tqdm

import sys
sys.path.append("/home/dkham/Documents/year4/ar-bird-vocalization/")
from utils.audio import get_all_segments

CODEC_SR = 24000  # both codecs operate at 24 kHz


def load_segment_native(filepath: str, start: float, end: float):
    """Load an audio segment at its native sample rate. Returns (audio, sr)."""
    info = sf.info(filepath)
    sr = info.samplerate
    start_frame = int(start * sr)
    end_frame = int(end * sr)
    audio, sr = sf.read(filepath, start=start_frame, stop=end_frame)
    if audio.ndim > 1:
        audio = librosa.to_mono(audio.T)
    return audio.astype(np.float32), sr


def resample(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def collect_segments(dataset, n):
    """Gather the first *n* segments from the dataset."""
    segments = []
    for item in dataset:
        if len(segments) >= n:
            break
        for start, end in get_all_segments(item):
            segments.append((item["filepath"], start, end))
            if len(segments) >= n:
                break
    return segments[:n]


def encodec_roundtrip(audio_24k, model, processor, device):
    """Encode + decode with EnCodec 24 kHz."""
    inputs = processor(
        raw_audio=[audio_24k],
        sampling_rate=CODEC_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        encoded = model.encode(**inputs)
        decoded = model.decode(encoded.audio_codes, [None])
    return decoded.audio_values[0, 0].cpu().numpy()


def snac_roundtrip(audio_24k, model, device):
    """Encode + decode with SNAC 24 kHz."""
    x = torch.from_numpy(audio_24k).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        codes = model.encode(x)
        audio_hat = model.decode(codes)
    return audio_hat[0, 0].cpu().numpy()


def trim_or_pad(recon, target_len):
    """Match reconstructed audio length to the original."""
    if len(recon) > target_len:
        return recon[:target_len]
    if len(recon) < target_len:
        return np.pad(recon, (0, target_len - len(recon)), mode="constant")
    return recon


def create_mel_transform(sample_rate, device, n_fft=2048, hop_length=256, n_mels=128):
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    ).to(device)
    amplitude_to_db = T.AmplitudeToDB().to(device)
    return mel_transform, amplitude_to_db


def audio_to_logmel(audio_np, mel_transform, amplitude_to_db, device):
    x = torch.from_numpy(audio_np).float().unsqueeze(0).to(device)
    with torch.no_grad():
        mel = mel_transform(x)
        log_mel = amplitude_to_db(mel)
    return log_mel[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Compare EnCodec 24kHz vs SNAC 24kHz on XCM samples")
    parser.add_argument("-n", "--num_audios", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="data/codec_comparison")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "audios").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ds = load_dataset("DBD-research-group/BirdSet", "XCM", split="train", trust_remote_code=True)
    segments = collect_segments(ds, args.num_audios)

    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device).eval()
    encodec_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()

    mel_cache = {}

    for i, (filepath, start, end) in enumerate(tqdm(segments, desc="Processing")):
        audio_native, native_sr = load_segment_native(filepath, start, end)
        name = Path(filepath).stem
        prefix = f"sample_{i:03d}_{name}"

        # Resample to 24 kHz for both codecs
        audio_24k = resample(audio_native, native_sr, CODEC_SR)

        # Encode / decode
        recon_encodec_24k = encodec_roundtrip(audio_24k, encodec_model, encodec_processor, device)
        recon_snac_24k = snac_roundtrip(audio_24k, snac_model, device)

        # Match lengths at 24 kHz, then resample back to native SR
        recon_encodec_24k = trim_or_pad(recon_encodec_24k, len(audio_24k))
        recon_snac_24k = trim_or_pad(recon_snac_24k, len(audio_24k))

        recon_encodec = resample(recon_encodec_24k, CODEC_SR, native_sr)
        recon_snac = resample(recon_snac_24k, CODEC_SR, native_sr)

        recon_encodec = trim_or_pad(recon_encodec, len(audio_native))
        recon_snac = trim_or_pad(recon_snac, len(audio_native))

        # Save audio at native SR
        sf.write(str(out_dir / "audios" / f"{prefix}_original.wav"), audio_native, native_sr)
        sf.write(str(out_dir / "audios" / f"{prefix}_encodec.wav"), recon_encodec, native_sr)
        sf.write(str(out_dir / "audios" / f"{prefix}_snac.wav"), recon_snac, native_sr)

        # Mel spectrogram comparison at native SR
        if native_sr not in mel_cache:
            mel_cache[native_sr] = create_mel_transform(native_sr, device)
        mel_transform, amplitude_to_db = mel_cache[native_sr]

        mel_orig = audio_to_logmel(audio_native, mel_transform, amplitude_to_db, device)
        mel_encodec = audio_to_logmel(recon_encodec, mel_transform, amplitude_to_db, device)
        mel_snac = audio_to_logmel(recon_snac, mel_transform, amplitude_to_db, device)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        for ax, mel, title in zip(
            axes,
            [mel_orig, mel_encodec, mel_snac],
            ["Original", "EnCodec 24kHz", "SNAC 24kHz"],
        ):
            im = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
            ax.set_title(title)
            ax.set_ylabel("Mel bin")
            ax.set_xlabel("Frame")
        fig.colorbar(im, ax=axes, label="dB", shrink=0.8, pad=0.02)
        fig.suptitle(f"Mel spectrogram — {prefix}  (native SR = {native_sr} Hz)")
        fig.savefig(out_dir / "plots" / f"{prefix}_comparison.png", dpi=150)
        plt.close(fig)

    print(f"Done. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
