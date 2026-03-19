import numpy as np
import soundfile as sf
import librosa


def load_segment(filepath, start, end, target_sr=32000):
    info = sf.info(filepath)
    sr = info.samplerate
    start_frame = int(start * sr)
    end_frame = int(end * sr)
    audio, sr = sf.read(filepath, start=start_frame, stop=end_frame)
    if audio.ndim > 1:
        audio = librosa.to_mono(audio.T)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def apply_fade(audio, fade_samples):
    if len(audio) < 2 * fade_samples:
        return audio
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    audio = audio.copy()
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    return audio
