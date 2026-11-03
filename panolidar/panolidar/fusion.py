"""Epipole-aware Bayesian fusion of a learned depth prior with spherical triangulation.

All quantities live in log-depth, where relative errors are additive:
  prior            ~ N(mu_p, sigma_p^2)          learned monocular panoramic depth
  stereo, partner k ~ N(mu_k, sigma_k^2 / w_k)   triangulation; sigma_k is analytic (uncertainty.py)
The posterior is the precision-weighted combination. The weight w_k in [0, 1]
down-weights stereo that is unreliable (low matching confidence) or inconsistent
with the prior (occlusion, moving object, mismatch). Two gates are provided:
  'mixture'  w = c * P(inlier | delta), the posterior inlier probability under
             delta = mu_k - mu_p ~ pi N(0, sigma_p^2 + sigma_k^2) + (1 - pi) U(range);
  'quartic'  w = c / (1 + (delta / (k sigma_tot))^4), a heuristic kept for ablations.
Both are smooth, so the fusion is differentiable end to end.
"""
import math

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


def align_prior_scale(mu_p, mu_s, sigma_s, confidence, max_sigma=0.05, min_confidence=0.5, min_rays=10):
    """Global log-scale offset that aligns the prior to reliable triangulated rays.

    Monocular metric depth often has a per-image scale error; well-conditioned
    stereo rays fix it. Uses a precision-weighted median of (mu_s - mu_p) per image.
    Returns the offset (B,), 0 where too few reliable rays exist.
    """
    offsets = []
    for b in range(mu_p.shape[0]):
        good = (sigma_s[b] < max_sigma) & (confidence[b] > min_confidence)
        good &= torch.isfinite(mu_s[b]) & torch.isfinite(mu_p[b])
        if good.sum() < min_rays:
            offsets.append(mu_p.new_zeros(()))
            continue
        diff = (mu_s[b] - mu_p[b])[good]
        weight = confidence[b][good] / sigma_s[b][good] ** 2
        offsets.append(weighted_median(diff.detach(), weight.detach()))
    return torch.stack(offsets)


def inlier_weight(delta, var_tot, confidence, gate='mixture', gate_k=3.0, inlier_prior=0.9, log_range=6.0):
    """Stereo weight in [0, 1] from the stereo-prior disagreement delta (log-depth)."""
    if gate == 'none':
        return confidence
    if gate == 'quartic':
        z = delta / (gate_k * torch.sqrt(var_tot))
        return confidence / (1.0 + z ** 4)
    if gate == 'mixture':
        log_in = math.log(inlier_prior) - 0.5 * delta ** 2 / var_tot - 0.5 * torch.log(2 * math.pi * var_tot)
        log_out = torch.full_like(log_in, math.log(1.0 - inlier_prior) - math.log(log_range))
        return confidence * torch.sigmoid(log_in - log_out)
    raise ValueError(f'unknown gate {gate}')


def fuse_log_depth(mu_p, sigma_p, mu_s=None, sigma_s=None, confidence=None, gate_k=3.0, gate='mixture', **gate_kwargs):
    """Posterior (mu, sigma, stereo_weight) in log-depth for one stereo partner.

    With no stereo (single panorama) this returns the prior unchanged.
    """
    if mu_s is None:
        return mu_p, sigma_p, torch.zeros_like(mu_p)
    mu, sigma, weights = fuse_log_depth_multi(mu_p, sigma_p, [(mu_s, sigma_s, confidence)], gate=gate,
                                              gate_k=gate_k, **gate_kwargs)
    return mu, sigma, weights[0]


def fuse_log_depth_multi(mu_p, sigma_p, partners, gate='mixture', gate_k=3.0, **gate_kwargs):
    """Fuse a prior with any number of stereo partners [(mu_k, sigma_k, confidence_k), ...].

    Each partner is gated against the prior independently. Returns (mu, sigma, [w_k]).
    Partners share the reference image, so their errors are not fully independent;
    treating them as independent slightly over-states the precision with many partners.
    """
    var_p = sigma_p ** 2
    precision = 1.0 / var_p
    weighted_mean = mu_p / var_p
    weights = []
    for mu_k, sigma_k, conf_k in partners:
        if conf_k is None:
            conf_k = torch.ones_like(mu_k)
        var_k = sigma_k.clamp_min(1e-6) ** 2
        w = inlier_weight(mu_k - mu_p, var_p + var_k, conf_k, gate, gate_k, **gate_kwargs)
        precision = precision + w / var_k
        weighted_mean = weighted_mean + w * mu_k / var_k
        weights.append(w)
    return weighted_mean / precision, torch.sqrt(1.0 / precision), weights
