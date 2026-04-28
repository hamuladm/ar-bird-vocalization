import numpy as np
import scipy.linalg


def inception_score(probs, eps=1e-10):
    p_y = probs.mean(axis=0)
    kl_divs = (probs * (np.log(probs + eps) - np.log(p_y + eps))).sum(axis=1)
    return float(np.exp(kl_divs.mean()))


def inception_score_restricted(probs, keep_indices, eps=1e-10):
    restricted = probs[:, keep_indices].copy()
    row_sums = restricted.sum(axis=1, keepdims=True)
    restricted = restricted / (row_sums + eps)
    return inception_score(restricted, eps=eps)


def classification_accuracy(probs, gt_labels, idx_to_ebird, top_k=(1, 5)):
    n = len(gt_labels)
    ebird_to_idx = {v: k for k, v in idx_to_ebird.items()}
    sorted_idx = np.argsort(-probs, axis=1)
    results = {}
    for k in top_k:
        correct = 0
        for i in range(n):
            top_k_ebird = [idx_to_ebird.get(int(j), "?") for j in sorted_idx[i, :k]]
            if gt_labels[i] in top_k_ebird:
                correct += 1
        results[f"top{k}_accuracy"] = correct / n if n > 0 else 0.0
    target_probs = []
    for i in range(n):
        idx = ebird_to_idx.get(gt_labels[i])
        if idx is not None:
            target_probs.append(float(probs[i, idx]))
    results["mean_target_prob"] = float(np.mean(target_probs)) if target_probs else 0.0
    return results


def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fad(feats_gen, feats_ref):
    mu_gen = feats_gen.mean(axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)
    mu_ref = feats_ref.mean(axis=0)
    sigma_ref = np.cov(feats_ref, rowvar=False)
    return frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)
