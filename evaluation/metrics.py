import numpy as np
import scipy.linalg


def inception_score(probs, eps=1e-10):
    p_y = probs.mean(axis=0)
    kl_divs = (probs * (np.log(probs + eps) - np.log(p_y + eps))).sum(axis=1)
    return float(np.exp(kl_divs.mean()))


def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fad(feats_gen, feats_ref):
    mu_gen = feats_gen.mean(axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)
    mu_ref = feats_ref.mean(axis=0)
    sigma_ref = np.cov(feats_ref, rowvar=False)
    return frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)
