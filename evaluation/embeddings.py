import csv
import logging
import os
import shutil
import tempfile
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from tqdm import tqdm
from transformers import ConvNextForImageClassification

from config import (
    DEVICE,
    EVAL_SAMPLE_RATE,
    EVAL_MODEL_CHECKPOINT,
    EVAL_N_FFT,
    EVAL_HOP_LENGTH,
    EVAL_N_MELS,
    EVAL_SPEC_MEAN,
    EVAL_SPEC_STD,
    EVAL_BATCH_SIZE,
    EVAL_SPEC_SIZE,
    EVAL_CHUNK_SEC,
)
from utils.audio import load_segment

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}

_BIRDNET_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
_BIRDNET_MODEL_PATH = os.path.join(
    _BIRDNET_DIR, "BirdNET+_V3.0-preview3_Global_11K_FP32.pt"
)
_BIRDNET_LABELS_PATH = os.path.join(
    _BIRDNET_DIR, "BirdNET+_V3.0-preview3_Global_11K_Labels.csv"
)
_BIRDNET_MODEL_URL = "https://zenodo.org/records/18247420/files/BirdNET+_V3.0-preview3_Global_11K_FP32.pt?download=1"
_BIRDNET_LABELS_URL = "https://zenodo.org/records/18247420/files/BirdNET+_V3.0-preview3_Global_11K_Labels.csv?download=1"
_TAXONOMY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data_relaxed", "taxonomy", "ebird_taxonomy.csv"
)
_BIRDNET_SR = 32000
_BIRDNET_CHUNK_SEC = 3.0


def _download(url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(dst):
        return
    logger.info("downloading %s -> %s", url, dst)
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dst)) as tmp:
        tmp_path = tmp.name
        with urllib.request.urlopen(url) as r:
            shutil.copyfileobj(r, tmp)
    os.replace(tmp_path, dst)


def _load_ebird_taxonomy():
    sci_to_code = {}
    if not os.path.isfile(_TAXONOMY_PATH):
        logger.warning("eBird taxonomy not found at %s", _TAXONOMY_PATH)
        return sci_to_code
    with open(_TAXONOMY_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sci = row.get("SCIENTIFIC_NAME", "").strip()
            code = row.get("SPECIES_CODE", "").strip()
            if sci and code:
                sci_to_code[sci] = code
    return sci_to_code


class SpectrogramTransform:
    def __init__(self, device=DEVICE):
        self.device = device

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=EVAL_N_FFT,
            hop_length=EVAL_HOP_LENGTH,
            power=2.0,
        ).to(device)

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=EVAL_N_MELS,
            sample_rate=EVAL_SAMPLE_RATE,
            n_stft=EVAL_N_FFT // 2 + 1,
        ).to(device)

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        ).to(device)

    def __call__(self, waveform):
        waveform = waveform.to(self.device)
        spec = self.spectrogram(waveform)
        mel_spec = self.mel_scale(spec)
        log_mel = self.amplitude_to_db(mel_spec)

        log_mel = log_mel.unsqueeze(1)

        log_mel = F.interpolate(
            log_mel,
            size=EVAL_SPEC_SIZE,
            mode="bilinear",
            align_corners=False,
        )

        log_mel = (log_mel - EVAL_SPEC_MEAN) / EVAL_SPEC_STD
        return log_mel


class EvalEmbedder:
    sample_rate = EVAL_SAMPLE_RATE

    def __init__(self, checkpoint=EVAL_MODEL_CHECKPOINT, device=DEVICE):
        self.device = device
        self.model = (
            ConvNextForImageClassification.from_pretrained(
                checkpoint, ignore_mismatched_sizes=True
            )
            .to(device)
            .eval()
        )
        self.preprocessor = SpectrogramTransform(device=device)
        id2label = self.model.config.id2label
        self.idx_to_ebird = {int(k): str(v).strip() for k, v in id2label.items()}
        self._chunk_len = int(EVAL_CHUNK_SEC * EVAL_SAMPLE_RATE)

    def _chunk_waveform(self, waveform):
        n = waveform.shape[-1]
        if n <= self._chunk_len:
            return waveform.unsqueeze(0)
        chunks = []
        for start in range(0, n, self._chunk_len):
            seg = waveform[start : start + self._chunk_len]
            if seg.shape[-1] < self._chunk_len:
                seg = F.pad(seg, (0, self._chunk_len - seg.shape[-1]))
            chunks.append(seg)
        return torch.stack(chunks)

    @torch.inference_mode()
    def extract(self, waveforms):
        all_probs = []
        all_features = []
        for i in range(waveforms.shape[0]):
            chunks = self._chunk_waveform(waveforms[i]).to(self.device)
            spec = self.preprocessor(chunks)
            backbone_out = self.model.convnext(spec)
            features = backbone_out.pooler_output
            logits = self.model.classifier(features)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.mean(dim=0))
            all_features.append(features.mean(dim=0))
        return {
            "probs": torch.stack(all_probs),
            "features": torch.stack(all_features),
        }


class BirdNetEmbedder:
    sample_rate = _BIRDNET_SR

    def __init__(self, device=DEVICE):
        _download(_BIRDNET_MODEL_URL, _BIRDNET_MODEL_PATH)
        _download(_BIRDNET_LABELS_URL, _BIRDNET_LABELS_PATH)

        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = torch.jit.load(_BIRDNET_MODEL_PATH, map_location=self.device)
        self.model.eval()

        self.labels = self._load_labels()
        self.ebird_to_idx = {}
        self.idx_to_ebird = {}
        for i, lbl in enumerate(self.labels):
            code = lbl["ebird_code"]
            if code != "?":
                self.ebird_to_idx[code] = i
                self.idx_to_ebird[i] = code

        logger.info(
            "BirdNetEmbedder: %d labels, %d mapped to ebird codes",
            len(self.labels),
            len(self.ebird_to_idx),
        )

    def _load_labels(self):
        sci_to_code = _load_ebird_taxonomy()
        labels = []
        with open(_BIRDNET_LABELS_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                sci = row.get("sci_name", "").strip()
                com = row.get("com_name", "").strip()
                ebird = sci_to_code.get(sci, "?")
                labels.append({"sci_name": sci, "com_name": com, "ebird_code": ebird})
        return labels

    def _chunk_waveform(self, waveform):
        chunk_len = int(_BIRDNET_CHUNK_SEC * _BIRDNET_SR)
        n = waveform.shape[-1]
        chunks = []
        for start in range(0, n, chunk_len):
            seg = waveform[start : start + chunk_len]
            if seg.shape[-1] < chunk_len:
                seg = F.pad(seg, (0, chunk_len - seg.shape[-1]))
            chunks.append(seg)
        return torch.stack(chunks)

    @torch.inference_mode()
    def extract(self, waveforms):
        all_embeddings = []
        all_preds = []
        for i in range(waveforms.shape[0]):
            w = waveforms[i]
            nonzero = w.abs().flip(0).cumsum(0).flip(0) > 0
            w = w[nonzero] if nonzero.any() else w
            chunks = self._chunk_waveform(w).to(self.device)
            embeddings, predictions = self.model(chunks)
            all_embeddings.append(embeddings.mean(dim=0))
            all_preds.append(predictions.mean(dim=0))
        return {
            "probs": torch.stack(all_preds),
            "features": torch.stack(all_embeddings),
        }

    def get_target_probs(self, preds, ebird_code):
        idx = self.ebird_to_idx.get(ebird_code)
        if idx is None:
            return None
        return preds[:, idx]

    def get_top1(self, preds):
        top1_prob, top1_idx = preds.max(dim=-1)
        top1_ebird = [self.idx_to_ebird.get(i.item(), "?") for i in top1_idx]
        return top1_prob, top1_ebird


_ENCODEC_SR = 16000


class EncodecEmbedder:
    sample_rate = _ENCODEC_SR

    def __init__(self, device=DEVICE):
        from audiocraft.models import AudioGen

        self.device = torch.device(device) if isinstance(device, str) else device
        audiogen = AudioGen.get_pretrained(
            "facebook/audiogen-medium", device=self.device
        )
        self.encoder = audiogen.compression_model.encoder
        self.encoder.eval()
        del audiogen

    @torch.inference_mode()
    def extract(self, waveforms):
        waveforms = waveforms.to(self.device)
        if waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(1)
        enc_out = self.encoder(waveforms)
        features = enc_out.mean(dim=-1)
        return {"probs": None, "features": features}


def _collect_audio_paths(directory):
    return sorted(
        p for p in Path(directory).rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    )


def _load_and_resample(path, target_sr=EVAL_SAMPLE_RATE):
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0)


def _collate_waveforms(waveforms):
    max_len = max(w.shape[0] for w in waveforms)
    batch = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        batch[i, : w.shape[0]] = w
    return batch


def _extract_batched(
    waveform_source, embedder, batch_size, desc="embedding", total=None
):
    all_probs = []
    all_features = []

    if isinstance(waveform_source, (list, tuple)):
        total = len(waveform_source)
        source_iter = iter(waveform_source)
    else:
        source_iter = iter(waveform_source)

    n_batches = (total + batch_size - 1) // batch_size if total else None
    pbar = tqdm(desc=desc, total=n_batches, unit="batch")
    done = False
    while not done:
        batch_wavs = []
        for _ in range(batch_size):
            try:
                batch_wavs.append(next(source_iter))
            except StopIteration:
                done = True
                break
        if not batch_wavs:
            break
        batch = _collate_waveforms(batch_wavs).to(embedder.device)
        result = embedder.extract(batch)
        if result["probs"] is not None:
            all_probs.append(result["probs"].cpu().numpy())
        all_features.append(result["features"].cpu().numpy())
        del batch, batch_wavs, result
        pbar.update(1)
    pbar.close()

    out = {"features": np.concatenate(all_features, axis=0)}
    if all_probs:
        out["probs"] = np.concatenate(all_probs, axis=0)
    return out


def _iter_directory_waveforms(paths, target_sr):
    for p in paths:
        yield _load_and_resample(p, target_sr)


def extract_embeddings_from_directory(directory, embedder, batch_size=EVAL_BATCH_SIZE):
    target_sr = getattr(embedder, "sample_rate", EVAL_SAMPLE_RATE)
    paths = _collect_audio_paths(directory)
    logger.info("loading %d audio files from %s", len(paths), directory)
    source = _iter_directory_waveforms(paths, target_sr)
    return _extract_batched(
        source, embedder, batch_size, desc="dir embeddings", total=len(paths)
    )


def _iter_segment_waveforms(segments, target_sr):
    for seg in segments:
        seg["filepath"] = seg["filepath"].replace(
            "/workspace/.hf_home/", "/home/dkham/.cache/huggingface/"
        )
        audio_np = load_segment(seg["filepath"], seg["start"], seg["end"], target_sr)
        yield torch.from_numpy(audio_np).float()


def extract_embeddings_from_segments(segments, embedder, batch_size=EVAL_BATCH_SIZE):
    target_sr = getattr(embedder, "sample_rate", EVAL_SAMPLE_RATE)
    n = len(segments)
    logger.info("streaming %d reference segments (target_sr=%d)", n, target_sr)
    source = _iter_segment_waveforms(segments, target_sr)
    return _extract_batched(
        source, embedder, batch_size, desc="ref embeddings", total=n
    )


def extract_embeddings_from_arrays(arrays, embedder, batch_size=EVAL_BATCH_SIZE):
    waveforms = [torch.from_numpy(a).float() for a in arrays]
    return _extract_batched(waveforms, embedder, batch_size)


def _iter_shard_waveforms(shard_paths, target_sr, gt_labels_out):
    for path in shard_paths:
        ebird_code = path.stem
        data = np.load(path)
        samples = data["samples"]
        lengths = data["lengths"]
        sr = int(data["sample_rate"])
        for i in range(len(lengths)):
            w = torch.from_numpy(samples[i, : lengths[i]]).float()
            if sr != target_sr:
                w = torchaudio.functional.resample(
                    w.unsqueeze(0), sr, target_sr
                ).squeeze(0)
            gt_labels_out.append(ebird_code)
            yield w


def extract_embeddings_from_shards(directory, embedder, batch_size=EVAL_BATCH_SIZE):
    target_sr = getattr(embedder, "sample_rate", EVAL_SAMPLE_RATE)
    shard_paths = sorted(Path(directory).glob("*.npz"))
    logger.info("loading %d shards from %s", len(shard_paths), directory)

    total_waveforms = 0
    for path in shard_paths:
        data = np.load(path)
        total_waveforms += len(data["lengths"])
    logger.info("total waveforms across shards: %d", total_waveforms)

    gt_labels: list[str] = []
    source = _iter_shard_waveforms(shard_paths, target_sr, gt_labels)
    result = _extract_batched(
        source, embedder, batch_size, desc="gen embeddings", total=total_waveforms
    )
    result["gt_labels"] = gt_labels
    return result
