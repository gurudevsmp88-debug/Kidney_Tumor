"""Benchmark protocol used in the paper (synthetic streets; see experiments/run_paper_experiments.py).

Protocol
  * Calibration scenes (seeds >= 1000) are used only to fit the stereo precision model,
    standing in for the city teacher. Test scenes (seeds < 1000) are disjoint.
  * Reference panorama at the origin, partner panorama displaced along the street (+y),
    i.e. the realistic worst case: epipoles look down the street.
  * The monocular prior is simulated (synthetic.synthetic_prior): ~6% smooth relative error
    plus a global scale error up to +-10%, with a constant self-reported sigma.
"""
import math

import torch

from .calibration import PrecisionModel, fit_baseline_sigmas, unit_and_pose_sigma
from .metrics import depth_metrics, metrics_by_baseline_angle
from .pipeline import reconstruct, stereo_kwargs
from .sphere import equirect_rays
from .stereo import two_view_depth
from .synthetic import random_street, render_panorama, synthetic_prior

IDENTITY = torch.eye(3)


def render_pair(seed, height, width, baseline):
    scene = random_street(seed)
    img1, gt = render_panorama(scene, (0, 0, 0), height, width)
    img2, _ = render_panorama(scene, (0, baseline, 0), height, width)
    return img1, img2, gt


def calibrate(cfg, seeds, height, width, baseline, num_bins=6, fit_kappa=True):
    """Fit the precision model on calibration scenes (GT depth plays the teacher)."""
    rays = equirect_rays(height, width)
    b = torch.tensor([0.0, baseline, 0.0])
    E, U, C = [], [], []
    for seed in seeds:
        img1, img2, gt = render_pair(seed, height, width, baseline)
        st = two_view_depth(img1[None], img2[None], IDENTITY[None], b[None], **stereo_kwargs(cfg['stereo']))
        m = torch.isfinite(gt)
        u, _ = unit_and_pose_sigma(rays, torch.where(m, gt, torch.ones_like(gt)), b, width)
        E.append((st['log_depth'][0] - gt.log())[m])
        U.append(u[m])
        C.append(st['confidence'][0][m])
    E, U, C = torch.cat(E), torch.cat(U), torch.cat(C)
    precision = PrecisionModel.fit(E, U, C, num_bins=num_bins, fit_kappa=fit_kappa)
    precision.baselines = fit_baseline_sigmas(E, C)
    return precision


def method_configs(cfg, precision):
    """Name -> cfg for every compared method (all fusion methods share the same stereo + prior)."""
    def variant(sigma_px=None, **fusion):
        out = dict(cfg, fusion=dict(cfg['fusion'], **fusion), stereo=dict(cfg['stereo']))
        if sigma_px is not None:
            out['stereo']['sigma_px'] = sigma_px
        return out
    cal = precision.to_dict()
    fitted = getattr(precision, 'baselines', {})
    return {
        'fuse_constant': variant(sigma_mode='constant', **({'constant_sigma': fitted['constant_sigma']} if fitted else {})),
        'fuse_confidence': variant(sigma_mode='confidence',
                                   **({'confidence_scale': fitted['confidence_scale']} if fitted else {})),
        'select_ours': variant(sigma_px=cal, mode='select'),
        'ours_closed_form': variant(sigma_px=0.5),
        'ours_no_align': variant(sigma_px=cal, align_scale=False),
        'ours_quartic_gate': variant(sigma_px=cal, gate='quartic'),
        'ours_sigma_at_stereo': variant(sigma_px=cal, sigma_at='stereo'),
        'ours': variant(sigma_px=cal),
    }


def evaluate(cfg, precision, seeds, height, width, baseline, prior_rel_std=0.06, prior_scale_bias=0.1,
             methods=None):
    """Per-method metrics (overall and per baseline-angle bin), averaged over test scenes."""
    rays = equirect_rays(height, width)
    b = torch.tensor([0.0, baseline, 0.0])
    configs = method_configs(cfg, precision)
    if methods is not None:
        configs = {k: v for k, v in configs.items() if k in methods}
    per = {}
    for seed in seeds:
        img1, img2, gt = render_pair(seed, height, width, baseline)
        mu_p, sigma_p = synthetic_prior(gt, prior_rel_std, prior_scale_bias, seed=seed)
        preds = {}
        for name, c in configs.items():
            r = reconstruct(img1, (IDENTITY, torch.zeros(3)), mu_p, sigma_p, img2, (IDENTITY, b), c)
            preds[name] = (r['depth'], r['sigma_log'])
            if name == 'ours':
                preds['prior'] = (mu_p.exp(), sigma_p)
                preds['prior_aligned'] = (r['prior_mu_aligned'].exp(), sigma_p)
                preds['stereo'] = (r['stereo_depth'], r['stereo_sigma_log'])
        for name, (d, s) in preds.items():
            row = metrics_by_baseline_angle(d, gt, rays, b, s)
            row['all'] = depth_metrics(d, gt, s)
            per.setdefault(name, []).append(row)
    return {name: _average(rows) for name, rows in per.items()}


def _average(rows):
    out = {}
    for bin_ in rows[0]:
        keys = [k for k in rows[0][bin_] if k != 'count']
        out[bin_] = {k: sum(r[bin_][k] for r in rows) / len(rows) for k in keys}
        out[bin_]['count'] = sum(r[bin_].get('count', 0) for r in rows)
    return out


def _rotation(axis_angle):
    th = float(axis_angle.norm())
    if th < 1e-12:
        return torch.eye(3)
    k = axis_angle / th
    K = torch.tensor([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return torch.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def evaluate_robustness(cfg, precision, seeds, height, width, partners_y=(4.0,), sigma_rot=0.0, sigma_trans=0.0,
                        model_pose=True, prior_rel_std=0.06, prior_scale_bias=0.1, baseline_sigmas=None):
    """Robustness protocol: partners at the given offsets along the street, whose *estimated* poses
    are perturbed by rotation/translation noise (per-axis stds). Images are rendered with the true poses.
    Methods: ours with the pose term (assumes the pose noise level is known, e.g. from SfM/GNSS),
    ours without it, and the two fitted geometry-blind baselines."""
    rays = equirect_rays(height, width)
    base = method_configs(cfg, precision)
    if baseline_sigmas is not None:  # geometry-blind baselines recalibrated under the same pose noise
        base['fuse_constant']['fusion']['constant_sigma'] = baseline_sigmas['constant_sigma']
        base['fuse_confidence']['fusion']['confidence_scale'] = baseline_sigmas['confidence_scale']
    configs = {'fuse_constant': base['fuse_constant'], 'fuse_confidence': base['fuse_confidence'],
               'ours_no_pose_term': base['ours']}
    if model_pose:
        configs['ours'] = dict(base['ours'], fusion=dict(base['ours']['fusion'], sigma_rot=sigma_rot,
                                                         sigma_trans=sigma_trans))
    per = {}
    for seed in seeds:
        scene = random_street(seed)
        img1, gt = render_panorama(scene, (0, 0, 0), height, width)
        gen = torch.Generator().manual_seed(10_000 + seed)
        partners = []
        for y in partners_y:
            img, _ = render_panorama(scene, (0, y, 0), height, width)
            R = _rotation(sigma_rot * torch.randn(3, generator=gen))
            C = torch.tensor([0.0, y, 0.0]) + sigma_trans * torch.randn(3, generator=gen)
            partners.append((img, (R, C)))
        mu_p, sigma_p = synthetic_prior(gt, prior_rel_std, prior_scale_bias, seed=seed)
        b = torch.tensor([0.0, partners_y[0], 0.0])
        for name, c in configs.items():
            r = reconstruct(img1, (IDENTITY, torch.zeros(3)), mu_p, sigma_p, cfg=c, partners=partners)
            row = metrics_by_baseline_angle(r['depth'], gt, rays, b, r['sigma_log'])
            row['all'] = depth_metrics(r['depth'], gt, r['sigma_log'])
            per.setdefault(name, []).append(row)
    return {name: _average(rows) for name, rows in per.items()}


def calibrate_baselines_with_pose_noise(cfg, seeds, height, width, baseline, sigma_rot, sigma_trans):
    """Refit the geometry-blind baselines on calibration scenes whose partner poses carry the same noise."""
    from .sphere import relative_pose
    E, C = [], []
    for seed in seeds:
        scene = random_street(seed)
        img1, gt = render_panorama(scene, (0, 0, 0), height, width)
        img2, _ = render_panorama(scene, (0, baseline, 0), height, width)
        gen = torch.Generator().manual_seed(20_000 + seed)
        R = _rotation(sigma_rot * torch.randn(3, generator=gen))
        Cn = torch.tensor([0.0, baseline, 0.0]) + sigma_trans * torch.randn(3, generator=gen)
        R12, t12 = relative_pose(IDENTITY, torch.zeros(3), R, Cn)
        st = two_view_depth(img1[None], img2[None], R12[None], t12[None], **stereo_kwargs(cfg['stereo']))
        m = torch.isfinite(gt)
        E.append((st['log_depth'][0] - gt.log())[m])
        C.append(st['confidence'][0][m])
    return fit_baseline_sigmas(torch.cat(E), torch.cat(C))
