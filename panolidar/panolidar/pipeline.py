"""Training-free reconstruction: prior + spherical stereo + epipole-aware fusion.

Works with any panoramic depth prior (e.g. Depth Any Panoramas / DA360 outputs
saved as .npy). With one panorama it returns the prior; with partners it fuses.

Ablation switches (cfg['fusion']):
  sigma_mode  'analytic'   closed-form pose-aware triangulation sigma (ours)
              'constant'   geometry-blind constant sigma (`constant_sigma`)
              'confidence' constant sigma / match confidence (learned-confidence style)
  sigma_at    'prior'      evaluate the analytic sigma at the prior depth (ours)
              'stereo'     ... at the stereo depth
  mode        'fuse'       precision-weighted posterior (ours)
              'select'     per ray, take whichever source has the smaller sigma
  gate        'mixture' | 'quartic' | 'none'
  align_scale true/false
"""
import torch

from .calibration import PrecisionModel, unit_and_pose_sigma
from .fusion import align_prior_scale, fuse_log_depth_multi, inlier_weight, relative_to_log_std
from .sphere import depth_to_points, equirect_rays, relative_pose
from .stereo import two_view_depth

STEREO_KEYS = ('min_depth', 'max_depth', 'num_hypotheses', 'refine_iters', 'refine_hypotheses', 'kernel',
               'temperature', 'cost', 'chunk')


def stereo_kwargs(scfg):
    return {k: scfg[k] for k in STEREO_KEYS if k in scfg}


def stereo_sigma(rays, ref_depth, stereo, t12, width, scfg, fcfg):
    """Log-depth std (H, W) of one stereo partner according to the configured sigma model."""
    mode = fcfg.get('sigma_mode', 'analytic')
    if mode == 'constant':
        return torch.full_like(ref_depth, fcfg.get('constant_sigma', 0.03))
    if mode == 'confidence':
        scale = fcfg.get('confidence_scale', fcfg.get('constant_sigma', 0.03))
        return scale / stereo['confidence'][0].clamp_min(1e-3)
    at = ref_depth if fcfg.get('sigma_at', 'prior') == 'prior' else stereo['depth'][0]
    U, P = unit_and_pose_sigma(rays, at, t12, width, fcfg.get('sigma_rot', 0.0), fcfg.get('sigma_trans', 0.0))
    # matching precision: scalar sigma_px (pure closed form) or a calibrated map (calibration.py)
    match = PrecisionModel.from_config(scfg.get('sigma_px', 0.5))(stereo['confidence'][0], U)
    return torch.sqrt(match ** 2 + P ** 2).clamp(max=10.0)


def reconstruct(image1, pose1, prior_mu, prior_sigma, image2=None, pose2=None, cfg=None, partners=None):
    """Fused log-depth for reference panorama 1.

    image*: (3, H, W) in [0, 1]; pose*: (R, C); prior_mu/prior_sigma: (H, W) log-depth prior.
    partners: optional list of (image, pose); image2/pose2 is shorthand for one partner.
    Returns dict: depth, sigma_log, mu_log, stereo (list of per-partner dicts), scale_offset,
    points (H, W, 3) world, plus stereo_depth / stereo_sigma_log / confidence / stereo_weight of
    the first partner for convenience.
    """
    cfg = cfg or {}
    scfg, fcfg = cfg.get('stereo', {}), cfg.get('fusion', {})
    partners = list(partners or [])
    if image2 is not None:
        partners.insert(0, (image2, pose2))
    H, W = image1.shape[-2:]
    mu_p, sigma_p = prior_mu.unsqueeze(0), prior_sigma.unsqueeze(0)
    rays = equirect_rays(H, W, dtype=image1.dtype)
    out = {'stereo': [], 'stereo_depth': None, 'stereo_sigma_log': None, 'confidence': None,
           'stereo_weight': torch.zeros(H, W), 'scale_offset': 0.0}

    results = []
    for image_k, pose_k in partners:
        R12, t12 = relative_pose(pose1[0], pose1[1], pose_k[0], pose_k[1])
        st = two_view_depth(image1.unsqueeze(0), image_k.unsqueeze(0), R12.unsqueeze(0), t12.unsqueeze(0),
                            **stereo_kwargs(scfg))
        results.append((st, t12))

    if results:
        # Scale alignment uses the best-conditioned partner rays; sigma at the (unaligned) prior depth.
        sigmas = [stereo_sigma(rays, mu_p[0].exp(), st, t12, W, scfg, fcfg) for st, t12 in results]
        offset = torch.zeros(1)
        if fcfg.get('align_scale', True):
            best = min(range(len(results)), key=lambda k: sigmas[k].median().item())
            st_b = results[best][0]
            offset = align_prior_scale(mu_p, st_b['log_depth'], sigmas[best].unsqueeze(0), st_b['confidence'],
                                       max_sigma=fcfg.get('align_max_sigma', 0.05),
                                       scale_prior_std=fcfg.get('scale_prior_std'))
            mu_p = mu_p + offset.view(-1, 1, 1)
            if fcfg.get('sigma_at', 'prior') == 'prior' and fcfg.get('sigma_mode', 'analytic') == 'analytic':
                sigmas = [stereo_sigma(rays, mu_p[0].exp(), st, t12, W, scfg, fcfg) for st, t12 in results]
        out['scale_offset'] = float(offset[0])
        entries = [(st['log_depth'], sig.unsqueeze(0), st['confidence']) for (st, _), sig in zip(results, sigmas)]
        gate_args = {'gate': fcfg.get('gate', 'mixture'), 'gate_k': fcfg.get('gate_k', 3.0)}
        if 'inlier_prior' in fcfg:
            gate_args['inlier_prior'] = fcfg['inlier_prior']
        if fcfg.get('mode', 'fuse') == 'select':
            mu, sigma, weights = _select(mu_p, sigma_p, entries, gate_args)
        else:
            mu, sigma, weights = fuse_log_depth_multi(mu_p, sigma_p, entries, **gate_args)
        for (st, _), sig, w in zip(results, sigmas, weights):
            out['stereo'].append({'depth': st['depth'][0], 'sigma_log': sig, 'confidence': st['confidence'][0],
                                  'weight': w[0]})
        first = out['stereo'][0]
        out.update(stereo_depth=first['depth'], stereo_sigma_log=first['sigma_log'],
                   confidence=first['confidence'], stereo_weight=first['weight'])
    else:
        mu, sigma = mu_p, sigma_p

    depth = mu[0].exp()
    out.update(depth=depth, mu_log=mu[0], sigma_log=sigma[0], prior_mu_aligned=mu_p[0],
               points=depth_to_points(depth, pose1[0], pose1[1]))
    return out


def _select(mu_p, sigma_p, entries, gate_args):
    """Hard per-ray selection of the lowest-sigma source (gated stereo only)."""
    mu, sigma = mu_p.clone(), sigma_p.clone()
    weights = []
    for mu_k, sigma_k, conf_k in entries:
        w = inlier_weight(mu_k - mu_p, sigma_p ** 2 + sigma_k ** 2, conf_k, gate_args['gate'], gate_args['gate_k'])
        take = (w > 0.5) & (sigma_k < sigma)
        mu = torch.where(take, mu_k, mu)
        sigma = torch.where(take, sigma_k, sigma)
        weights.append(take.to(mu.dtype))
    return mu, sigma, weights
