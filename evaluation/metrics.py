import numpy as np
import scipy.linalg


def inception_score(probs: np.ndarray, eps: float = 1e-10) -> float:
    p_y = probs.mean(axis=0)
    kl_divs = (probs * (np.log(probs + eps) - np.log(p_y + eps))).sum(axis=1)
    return float(np.exp(kl_divs.mean()))


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    diff = mu1 - mu2

    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        if np.allclose(covmean.imag, 0, atol=1e-3):
            covmean = covmean.real
        else:
            covmean = covmean.real

    fd = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return fd


def compute_fad(feats_gen: np.ndarray, feats_ref: np.ndarray) -> float:
    mu_gen = feats_gen.mean(axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)

    mu_ref = feats_ref.mean(axis=0)
    sigma_ref = np.cov(feats_ref, rowvar=False)

    return frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)
