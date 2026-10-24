"""Epipole-aware Bayesian fusion of a learned depth prior with spherical triangulation.

All quantities live in log-depth, where relative errors are additive:
  prior   ~ N(mu_p, sigma_p^2)          (learned monocular panoramic depth)
  stereo  ~ N(mu_s, sigma_s^2 / w)      (triangulation; sigma_s analytic, see uncertainty.py)
The posterior is the precision-weighted combination. w in [0, 1] down-weights
stereo where it is unreliable (low matching confidence) or inconsistent with the
prior (occlusion, mismatch); w is smooth so the whole thing is differentiable.
"""
import torch


def relative_to_log_std(depth, depth_std):
    """Std of log-depth from an absolute depth std (first-order: sigma_z / z)."""
    return depth_std / depth.clamp_min(1e-6)


def weighted_median(values, weights):
    """Weighted median of a 1-D tensor."""
    order = torch.argsort(values)
    v, w = values[order], weights[order]
    cdf = torch.cumsum(w, 0)
    idx = torch.searchsorted(cdf, 0.5 * cdf[-1].unsqueeze(0)).clamp(max=len(v) - 1)
    return v[idx].squeeze(0)


def align_prior_scale(mu_p, mu_s, sigma_s, confidence, max_sigma=0.05, min_confidence=0.5):
    """Global log-scale offset that aligns the prior to reliable triangulated rays.

    Monocular metric depth often has a per-image scale error; well-conditioned
    stereo rays fix it. Uses a precision-weighted median of (mu_s - mu_p) per image.
    Returns the offset (B,), 0 where no reliable ray exists.
    """
    offsets = []
    for b in range(mu_p.shape[0]):
        good = (sigma_s[b] < max_sigma) & (confidence[b] > min_confidence)
        if good.sum() < 10:
            offsets.append(mu_p.new_zeros(()))
            continue
        diff = (mu_s[b] - mu_p[b])[good]
        weight = (confidence[b][good] / sigma_s[b][good] ** 2)
        offsets.append(weighted_median(diff, weight))
    return torch.stack(offsets)


def fuse_log_depth(mu_p, sigma_p, mu_s=None, sigma_s=None, confidence=None, gate_k=3.0):
    """Posterior (mu, sigma, stereo_weight) in log-depth.

    With no stereo (single panorama) this returns the prior unchanged.
    The gate w = confidence / (1 + (delta / (gate_k * sigma_tot))^4) suppresses
    stereo that disagrees with the prior by more than ~gate_k combined sigmas.
    """
    if mu_s is None:
        return mu_p, sigma_p, torch.zeros_like(mu_p)
    if confidence is None:
        confidence = torch.ones_like(mu_s)
    var_p = sigma_p ** 2
    var_s = sigma_s.clamp_min(1e-6) ** 2
    delta = mu_s - mu_p
    z = delta / (gate_k * torch.sqrt(var_p + var_s))
    w = confidence / (1.0 + z ** 4)
    prec = 1.0 / var_p + w / var_s
    mu = (mu_p / var_p + w * mu_s / var_s) / prec
    return mu, torch.sqrt(1.0 / prec), w
