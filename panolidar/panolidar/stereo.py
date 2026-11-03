"""Two-panorama spherical plane-sweep stereo with coarse-to-fine refinement.

For every ray of the reference panorama and every log-depth hypothesis, the 3D
point is projected into the source panorama, which is sampled with longitude
wrap-around. Two matching costs are available:
  'l1'   mean absolute difference, box-aggregated over a window;
  'zncc' 1 - zero-mean normalised cross-correlation over the window, per channel,
         invariant to per-image gain and bias (exposure / white balance changes).
Hypotheses are processed in chunks so memory stays bounded at high resolution.
"""
import math

import torch
import torch.nn.functional as F

from .sphere import circular_pad, equirect_rays, rays_to_angles, sample_equirect


def box_filter(x, kernel):
    """Mean over a kernel x kernel window of (N, C, H, W) with longitude wrap-around."""
    if kernel <= 1:
        return x
    p = kernel // 2
    return F.avg_pool2d(circular_pad(x, p, p), kernel, stride=1)


aggregate = box_filter  # backwards-compatible name


def warp_to_reference(feat2, rays, log_depths, R12, t12):
    """Sample panorama-2 features for every camera-1 ray at every depth hypothesis.

    feat2: (B, C, H, W); rays: (H, W, 3) camera-1 rays; log_depths: (B, K, H, W);
    R12, t12: pose of camera 2 in camera-1 frame, (B, 3, 3) and (B, 3).
    Returns (B, K, C, H, W).
    """
    B, K, H, W = log_depths.shape
    C = feat2.shape[1]
    depth = log_depths.exp().unsqueeze(-1)  # (B, K, H, W, 1)
    x1 = rays.view(1, 1, H, W, 3) * depth
    x2 = torch.einsum('bkhwi,bij->bkhwj', x1 - t12.view(B, 1, 1, 1, 3), R12)  # R12^T (x1 - t)
    theta, phi = rays_to_angles(x2)
    src = feat2.unsqueeze(1).expand(B, K, *feat2.shape[1:])
    sampled = sample_equirect(src.reshape(B * K, C, *feat2.shape[2:]), theta.reshape(B * K, H, W),
                              phi.reshape(B * K, H, W))
    return sampled.view(B, K, C, H, W)


def matching_cost(ref, warped, kernel=5, cost='l1', eps=1e-4):
    """Cost volume (B, K, H, W) from reference (B, C, H, W) and warped source (B, K, C, H, W)."""
    B, K, C, H, W = warped.shape
    if cost == 'l1':
        c = (warped - ref.unsqueeze(1)).abs().mean(2)
        return box_filter(c, kernel)
    if cost == 'zncc':
        r = ref.unsqueeze(1).expand_as(warped).reshape(B * K, C, H, W)
        w = warped.reshape(B * K, C, H, W)
        mr, mw = box_filter(r, kernel), box_filter(w, kernel)
        var_r = (box_filter(r * r, kernel) - mr * mr).clamp_min(0)
        var_w = (box_filter(w * w, kernel) - mw * mw).clamp_min(0)
        cov = box_filter(r * w, kernel) - mr * mw
        zncc = cov / torch.sqrt(var_r * var_w + eps ** 2)
        return (0.5 * (1.0 - zncc)).mean(1).view(B, K, H, W)
    raise ValueError(f'unknown cost {cost}')


def sweep(feat1, feat2, log_depths, R12, t12, kernel=5, temperature=0.02, cost='l1', chunk=16):
    """One plane-sweep pass. Returns (soft log_depth, probabilities (B,K,H,W), cost volume)."""
    H, W = feat1.shape[-2:]
    rays = equirect_rays(H, W, device=feat1.device, dtype=feat1.dtype)
    costs = []
    for k0 in range(0, log_depths.shape[1], chunk):
        hyp = log_depths[:, k0:k0 + chunk]
        warped = warp_to_reference(feat2, rays, hyp, R12, t12)
        costs.append(matching_cost(feat1, warped, kernel, cost))
    cost_volume = torch.cat(costs, dim=1)
    prob = torch.softmax(-cost_volume / temperature, dim=1)
    log_depth = (prob * log_depths).sum(1)
    return log_depth, prob, cost_volume


def two_view_depth(feat1, feat2, R12, t12, min_depth=1.0, max_depth=200.0, num_hypotheses=64,
                   refine_iters=2, refine_hypotheses=16, kernel=5, temperature=0.02, cost='l1', chunk=16):
    """Depth of panorama 1 from panoramas 1+2 with known relative pose.

    Returns dict with 'depth' (B, H, W), 'log_depth', 'confidence' in [0, 1] (how much the best
    coarse hypothesis beats the median one), and 'min_cost' (B, H, W) of the final pass.
    """
    B, _, H, W = feat1.shape
    log_d = torch.linspace(math.log(min_depth), math.log(max_depth), num_hypotheses,
                           device=feat1.device, dtype=feat1.dtype)
    hyps = log_d.view(1, -1, 1, 1).expand(B, -1, H, W)
    step = (log_d[1] - log_d[0]).item()
    log_depth, prob, cost_volume = sweep(feat1, feat2, hyps, R12, t12, kernel, temperature, cost, chunk)
    # Distinctiveness of the best match over the whole depth range (coarse pass): ~0 where all
    # hypotheses look alike (epipoles, textureless) or nothing matches well (occlusion).
    coarse_min = cost_volume.min(1).values
    coarse_median = cost_volume.median(1).values
    confidence = ((coarse_median - coarse_min) / coarse_median.clamp_min(1e-6)).clamp(0.0, 1.0)
    # Initialise refinement from the arg-max (robust to multi-modal costs), not the soft mean.
    log_depth = torch.gather(hyps, 1, prob.argmax(1, keepdim=True)).squeeze(1)
    for _ in range(refine_iters):
        offsets = torch.linspace(-2 * step, 2 * step, refine_hypotheses, device=feat1.device, dtype=feat1.dtype)
        hyps = log_depth.unsqueeze(1) + offsets.view(1, -1, 1, 1)
        hyps = hyps.clamp(math.log(min_depth), math.log(max_depth))
        log_depth, prob, cost_volume = sweep(feat1, feat2, hyps, R12, t12, kernel, temperature, cost, chunk)
        step = 4 * step / (refine_hypotheses - 1)
    return {'depth': log_depth.exp(), 'log_depth': log_depth, 'confidence': confidence,
            'min_cost': cost_volume.min(1).values}
