# Borrowed from: https://github.com/ixobert/birds-generation/blob/master/src/generate_samples.py

from collections import OrderedDict
from copy import deepcopy
import os
import random
import torch
import argparse
import glob
import librosa

from vqvae import VQVAE

import tqdm
import numpy as np
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument("--data_paths", type=str, default="")
parser.add_argument("--out_folder", type=str)
parser.add_argument("--augmentations", default="noise")
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--device", default="cpu")
parser.add_argument("--model_path", type=str)


class Augmentations:
    all_methods = [
        "noise",
        "interpolation",
    ]

    def __init__(self):
        pass

    def encode(self, model, img, device="cuda:0"):
        img = img.to(device)
        quant_top, quant_bottom, diff, id_top, id_bottom = model.encode(img)
        return (quant_top, quant_bottom, id_top, id_bottom)

    def load_audio(self, audio_path, sr=16384, seconds=4):
        audio, _sr = librosa.load(audio_path)
        audio = librosa.resample(y=audio, orig_sr=_sr, target_sr=sr)
        audio = librosa.util.fix_length(data=audio, size=seconds)
        return audio

    def load_sample_spectrogram(
        self, audio_path, window_length=16384 * 4, sr=16384, n_fft=1024
    ):
        audio = self.load_audio(audio_path, sr, window_length)
        features = librosa.feature.melspectrogram(y=audio, n_fft=n_fft)
        features = librosa.power_to_db(features)

        if features.shape[0] % 2 != 0:
            features = features[1:, :]
        if features.shape[1] % 2 != 0:
            features = features[:, 1:]
        return features

    def load_sample(self, filepath: str):
        spectrogram = self.load_sample_spectrogram(filepath)

        spectrogram = np.expand_dims(spectrogram, 0)
        spectrogram = np.expand_dims(spectrogram, 0)

        return spectrogram

    def decode(self, model: VQVAE, quant_top, quant_bottom):
        return model.decode(quant_top, quant_bottom).detach()

    def spectrogram_to_audio(self, spectrogram_db, sr=16384, n_fft=1024):
        spectrogram_power = librosa.db_to_power(spectrogram_db)
        audio = librosa.feature.inverse.mel_to_audio(
            spectrogram_power, sr=sr, n_fft=n_fft
        )
        return audio

    def noise(
        self,
        model,
        all_samples_paths,
        ratio=0.5,
        out_folder="",
        generation_count=10,
        device="cpu",
    ):
        all_samples_paths = [
            x for x in all_samples_paths if "noise" not in os.path.basename(x)
        ]

        for i, sample_path in tqdm.tqdm(enumerate(all_samples_paths)):
            for j in range(generation_count):
                spectrogram = torch.tensor(self.load_sample(sample_path))
                q_t, q_b, i_t, i_b = self.encode(model, spectrogram, device=device)
                new_q_t = ratio * torch.randn_like(q_t) + q_t
                new_q_b = ratio * torch.randn_like(q_b) + q_b
                reconstructed = self.decode(model, new_q_t, new_q_b).cpu().numpy()[0][0]
                reconstructed = reconstructed[:, :-4]

                filename, ext = os.path.splitext(sample_path)
                outfile = f"{filename}-{j}_noise{ratio:.2f}{ext}"
                os.makedirs(out_folder, exist_ok=True)
                outfile = os.path.join(out_folder, os.path.basename(outfile))
                np.save(outfile, reconstructed)

                audio = self.spectrogram_to_audio(reconstructed)
                wav_path = os.path.splitext(outfile)[0] + ".wav"
                sf.write(wav_path, audio, samplerate=16384)

    def interpolation(
        self,
        model,
        all_samples_paths,
        ratio=0.5,
        out_folder="",
        generation_count=10,
        device="cpu",
    ):
        all_samples_paths = [
            x for x in all_samples_paths if "interpolation" not in os.path.basename(x)
        ]

        for i, sample_path in tqdm.tqdm(enumerate(all_samples_paths)):
            count = 0
            tmp_all_samples_paths = deepcopy(all_samples_paths)
            random.shuffle(tmp_all_samples_paths)

            for k, sample_path1 in enumerate(tmp_all_samples_paths):
                if count >= generation_count:
                    break
                if sample_path == sample_path1:
                    continue
                sample_class = sample_path.split(os.sep)[-2]
                sample_class1 = sample_path1.split(os.sep)[-2]
                if sample_class == sample_class1:
                    ratio = random.random()  # Generate number in [0,1)
                    spectrogram = torch.tensor(self.load_sample(sample_path))
                    spectrogram1 = torch.tensor(self.load_sample(sample_path1))

                    q_t, q_b, i_t, i_b = self.encode(model, spectrogram, device=device)
                    q_t1, q_b1, i_t1, i_b1 = self.encode(
                        model, spectrogram1, device=device
                    )
                    new_q_t = (q_t1 - q_t) * ratio + q_t
                    new_q_b = (q_b1 - q_b) * ratio + q_b

                    reconstructed = (
                        self.decode(model, new_q_t, new_q_b).cpu().numpy()[0][0]
                    )
                    reconstructed = reconstructed[:, :-4]

                    filename, ext = os.path.splitext(sample_path)
                    outfile = f"{filename}-{k}_interpolation{ratio:.2f}-{os.path.basename(sample_path1)}{ext}"
                    np.save(outfile, reconstructed)

                    audio = self.spectrogram_to_audio(reconstructed)
                    wav_path = os.path.splitext(outfile)[0] + ".wav"
                    sf.write(wav_path, audio, samplerate=16384)
                    count += 1


def update_model_keys(old_model: OrderedDict, key_to_replace: str = "module."):
    new_model = OrderedDict()
    for key, value in old_model.items():
        if key.startswith(key):
            new_model[key.replace(key_to_replace, "", 1)] = value
        else:
            new_model[key] = value
    return new_model


def load_model(model, model_path, device="cuda:0"):
    weights = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model" in weights:
        weights = weights["model"]
    if "state_dict" in weights:
        weights = weights["state_dict"]
    weights = update_model_keys(weights, key_to_replace="net.")
    model.load_state_dict(weights)
    model = model.eval()
    model = model.to(device)
    return model


def main() -> None:
    args = parser.parse_args()
    augmentations = Augmentations()
    model = VQVAE(in_channel=1)
    if args.model_path:
        model = load_model(model, args.model_path, device=args.device)

    all_samples_paths = glob.glob(args.data_paths)
    aug_methods_names = args.augmentations.split(",")
    for aug_method_name in aug_methods_names:
        if aug_method_name not in Augmentations.all_methods:
            raise NotImplementedError

        func = getattr(augmentations, aug_method_name)
        func(
            model=model,
            all_samples_paths=all_samples_paths,
            ratio=0.5,
            out_folder=args.out_folder,
            generation_count=int(args.num_samples),
            device=args.device,
        )


if __name__ == "__main__":
    main()
