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


def _moved_scene(scene, shift=3.0, fraction=0.5, seed=0):
    """Copy of the scene in which a fraction of the boxes (cars) moved along the street."""
    g = torch.Generator().manual_seed(seed)
    moved = dict(scene)
    boxes = []
    for lo, hi in scene['boxes']:
        if torch.rand((), generator=g).item() < fraction:
            d = torch.tensor([0.0, shift, 0.0])
            boxes.append((lo + d, hi + d))
        else:
            boxes.append((lo, hi))
    moved['boxes'] = boxes
    return moved


def capture_effects(img1, img2, gain=1.0, bias=0.0, noise=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    noisy = lambda x: (x + noise * torch.randn(x.shape, generator=g)).clamp(0, 1) if noise > 0 else x
    return noisy(img1), noisy((gain * img2 + bias).clamp(0, 1))


ZNCC_STEREO = {'cost': 'zncc', 'temperature': 0.05, 'kernel': 7}


def zncc_config(cfg):
    return dict(cfg, stereo=dict(cfg['stereo'], **ZNCC_STEREO))


def evaluate_capture(cfg, precision, seeds, height, width, baseline=4.0, gain=1.0, bias=0.0, noise=0.0,
                     moving=False, prior_rel_std=0.06, prior_scale_bias=0.1, precision_zncc=None):
    """Capture-effect protocol: exposure change of the partner, sensor noise, moving cars.
    Methods: fitted confidence baseline and ours, each with the L1 and the ZNCC cost (each cost with its
    own calibration), and ours (ZNCC) without the outlier gate."""
    rays = equirect_rays(height, width)
    b = torch.tensor([0.0, baseline, 0.0])
    base = method_configs(cfg, precision)
    zbase = method_configs(zncc_config(cfg), precision_zncc or precision)
    configs = {
        'fuse_confidence_l1': base['fuse_confidence'],
        'fuse_confidence_zncc': zbase['fuse_confidence'],
        'ours_l1': base['ours'],
        'ours_zncc_no_gate': dict(zbase['ours'], fusion=dict(zbase['ours']['fusion'], gate='none')),
        'ours_zncc': zbase['ours'],
    }
    per = {}
    for seed in seeds:
        scene = random_street(seed)
        img1, gt = render_panorama(scene, (0, 0, 0), height, width)
        scene2 = _moved_scene(scene, seed=seed) if moving else scene
        img2, _ = render_panorama(scene2, (0, baseline, 0), height, width)
        img1, img2 = capture_effects(img1, img2, gain, bias, noise, seed)
        mu_p, sigma_p = synthetic_prior(gt, prior_rel_std, prior_scale_bias, seed=seed)
        for name, c in configs.items():
            r = reconstruct(img1, (IDENTITY, torch.zeros(3)), mu_p, sigma_p, img2, (IDENTITY, b), c)
            row = metrics_by_baseline_angle(r['depth'], gt, rays, b, r['sigma_log'])
            row['all'] = depth_metrics(r['depth'], gt, r['sigma_log'])
            per.setdefault(name, []).append(row)
    return {name: _average(rows) for name, rows in per.items()}


def calibrate_from_teacher(cfg, seeds, height, width, baseline, teacher_sigma_ang, mode='covariance',
                           n_teacher_views=7, spacing=3.0, min_views=2, num_bins=6):
    """Fit the precision model on *noisy teacher* depth instead of ground truth.

    The teacher is a denser capture along the street (7 panoramas, lateral offsets); its points are
    displaced by draws from their own triangulation covariance. Modes:
      'naive'      teacher depth treated as exact
      'min_views'  naive, restricted to teacher points seen by >= min_views views
      'covariance' teacher log-depth variance added in quadrature to the model (ours)
    """
    from .teacher import build_teacher_points, layered_targets, perturb_teacher
    rays = equirect_rays(height, width)
    b = torch.tensor([0.0, baseline, 0.0])
    E, U, C, T = [], [], [], []
    for seed in seeds:
        scene = random_street(seed)
        img1, _ = render_panorama(scene, (0, 0, 0), height, width)
        img2, _ = render_panorama(scene, (0, baseline, 0), height, width)
        g = torch.Generator().manual_seed(30_000 + seed)
        depths, poses = [], []
        for k in range(n_teacher_views):
            c = torch.tensor([1.5 * (2 * torch.rand((), generator=g).item() - 1),
                              (k - n_teacher_views // 2) * spacing, 0.0])
            depths.append(render_panorama(scene, tuple(c.tolist()), height, width)[1])
            poses.append((IDENTITY, c))
        teacher = build_teacher_points(depths, poses, teacher_sigma_ang, stride=1)
        noisy = perturb_teacher(teacher['points'], teacher['cov'], g)
        tgt = layered_targets(noisy, teacher['cov'], IDENTITY, torch.zeros(3), height, width, n_views=teacher['n_views'])
        st = two_view_depth(img1[None], img2[None], IDENTITY[None], b[None], **stereo_kwargs(cfg['stereo']))
        d = tgt['depth1']
        m = torch.isfinite(d) & torch.isfinite(tgt['var1'])
        if mode == 'min_views':
            m &= tgt['views1'] >= min_views
        u, _ = unit_and_pose_sigma(rays, torch.where(m, d, torch.ones_like(d)), b, width)
        E.append((st['log_depth'][0] - torch.log(torch.where(m, d, torch.ones_like(d))))[m])
        U.append(u[m])
        C.append(st['confidence'][0][m])
        T.append((tgt['var1'].clamp_min(0).sqrt() / d)[m] if mode == 'covariance' else torch.zeros(int(m.sum())))
    E, U, C, T = torch.cat(E), torch.cat(U), torch.cat(C), torch.cat(T)
    # rays whose teacher is far less certain than any plausible stereo error carry no information
    keep = T < 0.2
    return PrecisionModel.fit(E[keep], U[keep], C[keep], P=T[keep], num_bins=num_bins)
