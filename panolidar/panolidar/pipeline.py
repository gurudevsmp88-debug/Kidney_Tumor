"""Training-free reconstruction: prior + spherical stereo + epipole-aware fusion.

Works with any panoramic depth prior (e.g. Depth Any Panoramas / DA360 outputs
saved as .npy). With one panorama it returns the prior; with two it fuses.
"""
import math

import torch

from .fusion import align_prior_scale, fuse_log_depth, relative_to_log_std
from .sphere import depth_to_points, equirect_rays, relative_pose
from .stereo import two_view_depth
from .uncertainty import triangulation_depth_std


def reconstruct(image1, pose1, prior_mu, prior_sigma, image2=None, pose2=None, cfg=None):
    """Fused log-depth for panorama 1.

    image*: (3, H, W) in [0, 1]; pose*: (R, C); prior_mu/prior_sigma: (H, W) log-depth prior.
    cfg: dict with keys of configs/default.yaml 'stereo' and 'fusion' sections (optional).
    Returns dict: depth, sigma_log, mu_log, stereo_depth, stereo_sigma_log, confidence,
    stereo_weight, scale_offset, points (H, W, 3) in world coordinates.
    """
    cfg = cfg or {}
    scfg, fcfg = cfg.get('stereo', {}), cfg.get('fusion', {})
    H, W = image1.shape[-2:]
    mu_p, sigma_p = prior_mu.unsqueeze(0), prior_sigma.unsqueeze(0)
    out = {'stereo_depth': None, 'stereo_sigma_log': None, 'confidence': None, 'scale_offset': 0.0}
    if image2 is None:
        mu, sigma, w = fuse_log_depth(mu_p, sigma_p)
    else:
        R12, t12 = relative_pose(pose1[0], pose1[1], pose2[0], pose2[1])
        st = two_view_depth(image1.unsqueeze(0), image2.unsqueeze(0), R12.unsqueeze(0), t12.unsqueeze(0),
                            min_depth=scfg.get('min_depth', 1.0), max_depth=scfg.get('max_depth', 200.0),
                            num_hypotheses=scfg.get('num_hypotheses', 64),
                            refine_iters=scfg.get('refine_iters', 2),
                            refine_hypotheses=scfg.get('refine_hypotheses', 16),
                            kernel=scfg.get('kernel', 5), temperature=scfg.get('temperature', 0.02))
        sigma_ang = scfg.get('sigma_px', 0.5) * 2 * math.pi / W
        rays = equirect_rays(H, W, dtype=image1.dtype)
        # Evaluate the analytic sigma at the prior depth: near the epipoles the stereo estimate
        # itself is unreliable, and a wrong (too small) depth would give a falsely small sigma.
        prior_depth = mu_p[0].exp()
        std = triangulation_depth_std(rays, prior_depth, t12, sigma_ang)
        sigma_s = relative_to_log_std(prior_depth, std).unsqueeze(0).clamp(max=10.0)
        offset = torch.zeros(1)
        if fcfg.get('align_scale', True):
            offset = align_prior_scale(mu_p, st['log_depth'], sigma_s, st['confidence'],
                                       max_sigma=fcfg.get('align_max_sigma', 0.05))
            mu_p = mu_p + offset.view(-1, 1, 1)
        mu, sigma, w = fuse_log_depth(mu_p, sigma_p, st['log_depth'], sigma_s, st['confidence'],
                                      gate_k=fcfg.get('gate_k', 3.0))
        out.update(stereo_depth=st['depth'][0], stereo_sigma_log=sigma_s[0], confidence=st['confidence'][0],
                   scale_offset=float(offset[0]))
    depth = mu[0].exp()
    out.update(depth=depth, mu_log=mu[0], sigma_log=sigma[0], stereo_weight=w[0],
               points=depth_to_points(depth, pose1[0], pose1[1]))
    return out
