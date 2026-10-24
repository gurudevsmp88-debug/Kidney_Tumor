"""Two-panorama spherical plane-sweep stereo with coarse-to-fine refinement."""
import math

import torch
import torch.nn.functional as F

from .sphere import circular_pad, equirect_rays, rays_to_angles, sample_equirect


def aggregate(cost, kernel=5):
    """Box-filter a (B, K, H, W) cost volume over the image (longitude wraps)."""
    if kernel <= 1:
        return cost
    p = kernel // 2
    return F.avg_pool2d(circular_pad(cost, p, p), kernel, stride=1)


def warp_to_reference(feat2, rays, log_depths, R12, t12):
    """Sample panorama-2 features for every camera-1 ray at every depth hypothesis.

    feat2: (B, C, H, W); rays: (H, W, 3) camera-1 rays; log_depths: (B, K, H, W);
    R12, t12: pose of camera 2 in camera-1 frame, (B, 3, 3) and (B, 3).
    Returns (B, K, C, H, W).
    """
    B, K, H, W = log_depths.shape
    depth = log_depths.exp().unsqueeze(-1)  # (B, K, H, W, 1)
    x1 = rays.view(1, 1, H, W, 3) * depth
    x2 = torch.einsum('bkhwi,bij->bkhwj', x1 - t12.view(B, 1, 1, 1, 3), R12)  # R12^T (x1 - t)
    theta, phi = rays_to_angles(x2)
    sampled = sample_equirect(
        feat2.unsqueeze(1).expand(B, K, *feat2.shape[1:]).reshape(B * K, *feat2.shape[1:]),
        theta.reshape(B * K, H, W), phi.reshape(B * K, H, W))
    return sampled.view(B, K, feat2.shape[1], H, W)


def sweep(feat1, feat2, log_depths, R12, t12, kernel=5, temperature=0.02):
    """One plane-sweep pass. Returns (log_depth, probabilities (B,K,H,W), cost)."""
    H, W = feat1.shape[-2:]
    rays = equirect_rays(H, W, device=feat1.device, dtype=feat1.dtype)
    warped = warp_to_reference(feat2, rays, log_depths, R12, t12)
    cost = (warped - feat1.unsqueeze(1)).abs().mean(2)  # (B, K, H, W)
    cost = aggregate(cost, kernel)
    prob = torch.softmax(-cost / temperature, dim=1)
    log_depth = (prob * log_depths).sum(1)
    return log_depth, prob, cost


def two_view_depth(feat1, feat2, R12, t12, min_depth=1.0, max_depth=200.0, num_hypotheses=64,
                   refine_iters=2, refine_hypotheses=16, kernel=5, temperature=0.02):
    """Depth of panorama 1 from panoramas 1+2 with known relative pose.

    Returns dict with 'depth' (B, H, W), 'log_depth', 'confidence' in [0, 1] (how much the best
    coarse hypothesis beats the median one), and 'min_cost' (B, H, W) of the final pass.
    """
    B, _, H, W = feat1.shape
    log_d = torch.linspace(math.log(min_depth), math.log(max_depth), num_hypotheses,
                           device=feat1.device, dtype=feat1.dtype)
    hyps = log_d.view(1, -1, 1, 1).expand(B, -1, H, W)
    step = (log_d[1] - log_d[0]).item()
    log_depth, prob, cost = sweep(feat1, feat2, hyps, R12, t12, kernel, temperature)
    # Distinctiveness of the best match over the whole depth range (coarse pass): ~0 where all
    # hypotheses look alike (epipoles, textureless) or nothing matches well (occlusion).
    coarse_min = cost.min(1).values
    coarse_median = cost.median(1).values
    confidence = ((coarse_median - coarse_min) / coarse_median.clamp_min(1e-6)).clamp(0.0, 1.0)
    # Initialise refinement from the arg-max (robust to multi-modal costs), not the soft mean.
    log_depth = torch.gather(hyps, 1, prob.argmax(1, keepdim=True)).squeeze(1)
    for _ in range(refine_iters):
        offsets = torch.linspace(-2 * step, 2 * step, refine_hypotheses, device=feat1.device, dtype=feat1.dtype)
        hyps = log_depth.unsqueeze(1) + offsets.view(1, -1, 1, 1)
        hyps = hyps.clamp(math.log(min_depth), math.log(max_depth))
        log_depth, prob, cost = sweep(feat1, feat2, hyps, R12, t12, kernel, temperature)
        step = 4 * step / (refine_hypotheses - 1)
    return {'depth': log_depth.exp(), 'log_depth': log_depth, 'confidence': confidence,
            'min_cost': cost.min(1).values}
