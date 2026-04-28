import logging

import torch
import torchaudio

from config import DEVICE

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, generator, embedder, device=DEVICE):
        self.generator = generator
        self.device = torch.device(device)
        self.embedder = embedder
        self._build_label_map()

    def _build_label_map(self):
        if hasattr(self.embedder, "ebird_to_idx"):
            self.ebird_to_disc = dict(self.embedder.ebird_to_idx)
            self.disc_to_ebird = dict(self.embedder.idx_to_ebird)
        else:
            id2label = self.embedder.model.config.id2label
            self.ebird_to_disc = {str(v).strip(): int(k) for k, v in id2label.items()}
            self.disc_to_ebird = {int(k): str(v).strip() for k, v in id2label.items()}

    def _collate_and_resample(self, waveforms):
        target_sr = getattr(self.embedder, "sample_rate", 32000)
        tensors = []
        for w in waveforms:
            t = torch.from_numpy(w).float()
            if self.generator.sample_rate != target_sr:
                t = torchaudio.functional.resample(
                    t.unsqueeze(0), self.generator.sample_rate, target_sr
                ).squeeze(0)
            tensors.append(t)

        max_len = max(t.shape[0] for t in tensors)
        batch = torch.zeros(len(tensors), max_len)
        for i, t in enumerate(tensors):
            batch[i, : t.shape[0]] = t
        return batch.to(self.device)

    @torch.inference_mode()
    def _score(self, waveforms, ebird_code):
        disc_idx = self.ebird_to_disc.get(ebird_code)
        if disc_idx is None:
            logger.warning("ebird code %s not in discriminator label set", ebird_code)
            return torch.zeros(len(waveforms), device=self.device)

        batch = self._collate_and_resample(waveforms)
        result = self.embedder.extract(batch)
        probs = result["probs"]

        top1_prob, top1_idx = probs.max(dim=-1)
        target_probs = probs[:, disc_idx]

        for i in range(len(waveforms)):
            pred_ebird = self.disc_to_ebird.get(top1_idx[i].item(), "?")
            correct = pred_ebird == ebird_code
            logger.info(
                "candidate %d/%d  target=%s  predicted=%s  "
                "top1_prob=%.4f  target_prob=%.4f  correct=%s",
                i + 1,
                len(waveforms),
                ebird_code,
                pred_ebird,
                top1_prob[i].item(),
                target_probs[i].item(),
                correct,
            )
        return target_probs

    def generate(self, class_id, k=8, **gen_kwargs):
        ebird_code = self.generator.id_to_ebird[class_id]
        logger.info(
            "reranking class_id=%d (%s) with k=%d candidates",
            class_id,
            ebird_code,
            k,
        )

        waveforms = self.generator.generate_batch(class_id, k, **gen_kwargs)
        logger.info(
            "generated %d valid waveforms out of %d requested", len(waveforms), k
        )

        if not waveforms:
            return None
        if len(waveforms) == 1:
            return waveforms[0]

        scores = self._score(waveforms, ebird_code)
        best_idx = scores.argmax().item()

        logger.info(
            "selected candidate %d/%d  target_prob=%.4f  (min=%.4f, mean=%.4f, max=%.4f)",
            best_idx + 1,
            len(waveforms),
            scores[best_idx].item(),
            scores.min().item(),
            scores.mean().item(),
            scores.max().item(),
        )

        return waveforms[best_idx]
