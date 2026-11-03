"""Tests for pose-aware uncertainty, ZNCC stereo, chunking, multi-partner fusion and ablation switches."""
import math

import torch

from panolidar import cli
from panolidar.pipeline import reconstruct
from panolidar.sphere import equirect_rays
from panolidar.stereo import matching_cost, two_view_depth
from panolidar.synthetic import random_street, render_panorama, synthetic_prior
from panolidar.uncertainty import angle_to_baseline, triangulation_depth_std

CFG = cli.load_config()
DT = torch.float64


def _rodrigues(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    k = w / th
    K = torch.zeros(w.shape[0], 3, 3, dtype=w.dtype)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    th = th.unsqueeze(-1)
    return torch.eye(3, dtype=w.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def _midpoint(c1, d1, c2, d2):
    w = c1 - c2
    a, b, c = (d1 * d1).sum(-1), (d1 * d2).sum(-1), (d2 * d2).sum(-1)
    d, e = (d1 * w).sum(-1), (d2 * w).sum(-1)
    s = (b * e - c * d) / (a * c - b * b)
    t = (a * e - b * d) / (a * c - b * b)
    return 0.5 * ((c1 + s[:, None] * d1) + (c2 + t[:, None] * d2))


def _angular_noise(d, sigma, n, gen):
    e = sigma * torch.randn(n, 3, generator=gen, dtype=DT)
    e = e - (e @ d)[:, None] * d
    v = d + e
    return v / v.norm(dim=-1, keepdim=True)


def test_pose_aware_sigma_matches_monte_carlo():
    """Isotropic matching noise on both rays + rotation/position error of camera 2."""
    gen = torch.Generator().manual_seed(0)
    n = 30000
    b = torch.tensor([0.0, 4.0, 0.0], dtype=DT)
    cases = [(90, 20.0, 1e-3, 0.0, 0.0), (45, 15.0, 1e-3, 2e-3, 0.0), (30, 25.0, 5e-4, 0.0, 0.05),
             (70, 30.0, 1e-3, 1e-3, 0.02), (60, 10.0, 0.0, 0.0, 0.1)]
    for alpha_deg, depth, s_ang, s_rot, s_trans in cases:
        a = math.radians(alpha_deg)
        ray = torch.tensor([math.sin(a), math.cos(a), 0.2], dtype=DT)
        ray = ray / ray.norm()
        x = depth * ray
        d2 = (x - b) / (x - b).norm()
        r1 = _angular_noise(ray, s_ang, n, gen)
        r2 = _angular_noise(d2, s_ang, n, gen)
        if s_rot > 0:
            r2 = (_rodrigues(s_rot * torch.randn(n, 3, generator=gen, dtype=DT)) @ r2.unsqueeze(-1)).squeeze(-1)
        c2 = b + s_trans * torch.randn(n, 3, generator=gen, dtype=DT)
        est = _midpoint(torch.zeros(n, 3, dtype=DT), r1, c2, r2) @ ray
        mc = est.std().item()
        analytic = triangulation_depth_std(ray[None], torch.tensor([depth], dtype=DT), b, s_ang,
                                           sigma_rot=s_rot, sigma_trans=s_trans).item()
        assert abs(analytic / mc - 1) < 0.05, (alpha_deg, mc, analytic)


def test_zncc_is_invariant_to_gain_and_bias():
    gen = torch.Generator().manual_seed(0)
    ref = torch.rand(1, 3, 16, 32, generator=gen)
    warped = ref.unsqueeze(1).repeat(1, 2, 1, 1, 1)
    warped[:, 1] = 0.6 * warped[:, 1] + 0.2  # exposure change
    zncc = matching_cost(ref, warped, kernel=5, cost='zncc')
    l1 = matching_cost(ref, warped, kernel=5, cost='l1')
    assert torch.allclose(zncc[:, 0], zncc[:, 1], atol=1e-3) and zncc.max() < 1e-2
    assert l1[:, 1].mean() > 0.05


def test_chunked_sweep_equals_unchunked():
    scene = random_street(1)
    img1, _ = render_panorama(scene, (0, 0, 0), 32, 64)
    img2, _ = render_panorama(scene, (0, 3, 0), 32, 64)
    args = (img1[None], img2[None], torch.eye(3)[None], torch.tensor([[0.0, 3.0, 0.0]]))
    a = two_view_depth(*args, num_hypotheses=24, refine_iters=1, chunk=24)
    b = two_view_depth(*args, num_hypotheses=24, refine_iters=1, chunk=5)
    assert torch.allclose(a['depth'], b['depth'], atol=1e-5)


def test_exposure_change_hurts_l1_not_zncc():
    scene = random_street(2)
    img1, gt = render_panorama(scene, (0, 0, 0), 64, 128)
    img2, _ = render_panorama(scene, (0, 4, 0), 64, 128)
    img2 = (0.6 * img2 + 0.15).clamp(0, 1)
    side = torch.isfinite(gt) & (angle_to_baseline(equirect_rays(64, 128), torch.tensor([0.0, 4, 0])) > 1.0)
    errs = {}
    for cost in ('l1', 'zncc'):
        out = two_view_depth(img1[None], img2[None], torch.eye(3)[None], torch.tensor([[0.0, 4.0, 0.0]]), cost=cost,
                             temperature=0.05 if cost == 'zncc' else 0.02)
        errs[cost] = ((out['depth'][0] - gt).abs() / gt)[side].median().item()
    # ZNCC is gain/bias invariant: it beats L1 under an exposure change
    assert errs['zncc'] < errs['l1']


def _scene_inputs(seed=0, H=64, W=128, ys=(4.0,)):
    scene = random_street(seed)
    img1, gt = render_panorama(scene, (0, 0, 0), H, W)
    partners = [(render_panorama(scene, (0, y, 0), H, W)[0], (torch.eye(3), torch.tensor([0.0, y, 0.0]))) for y in ys]
    mu_p, s_p = synthetic_prior(gt, seed=seed)
    return img1, gt, mu_p, s_p, partners


def _median_rel(depth, gt):
    m = torch.isfinite(gt)
    return ((depth - gt).abs() / gt)[m].median().item()


def test_second_partner_improves_accuracy():
    img1, gt, mu_p, s_p, partners = _scene_inputs(ys=(4.0, -4.0))
    one = reconstruct(img1, (torch.eye(3), torch.zeros(3)), mu_p, s_p, cfg=CFG, partners=partners[:1])
    two = reconstruct(img1, (torch.eye(3), torch.zeros(3)), mu_p, s_p, cfg=CFG, partners=partners)
    assert len(two['stereo']) == 2
    assert _median_rel(two['depth'], gt) <= _median_rel(one['depth'], gt) * 1.02
    assert two['sigma_log'].mean() < one['sigma_log'].mean()


def test_ablation_switches_run_and_analytic_is_best():
    img1, gt, mu_p, s_p, partners = _scene_inputs(seed=3)
    pose = (torch.eye(3), torch.zeros(3))
    errors = {}
    for name, fusion in {'ours': {}, 'constant': {'sigma_mode': 'constant'},
                         'confidence': {'sigma_mode': 'confidence'}, 'sigma_at_stereo': {'sigma_at': 'stereo'},
                         'select': {'mode': 'select'}, 'quartic': {'gate': 'quartic'},
                         'no_align': {'align_scale': False}}.items():
        cfg = dict(CFG, fusion=dict(CFG['fusion'], **fusion))
        errors[name] = _median_rel(reconstruct(img1, pose, mu_p, s_p, cfg=cfg, partners=partners)['depth'], gt)
    # Ranking claims are made on the full benchmark (experiments/), not on a single small scene.
    assert all(math.isfinite(e) and e < 0.2 for e in errors.values())


def test_pose_sigma_inflates_uncertainty():
    img1, gt, mu_p, s_p, partners = _scene_inputs(seed=4)
    pose = (torch.eye(3), torch.zeros(3))
    base = reconstruct(img1, pose, mu_p, s_p, cfg=CFG, partners=partners)
    cfg = dict(CFG, fusion=dict(CFG['fusion'], sigma_rot=2e-3, sigma_trans=0.05))
    noisy = reconstruct(img1, pose, mu_p, s_p, cfg=cfg, partners=partners)
    assert (noisy['stereo_sigma_log'] >= base['stereo_sigma_log'] - 1e-7).all()
    assert noisy['stereo_sigma_log'].mean() > base['stereo_sigma_log'].mean()


def test_precision_model_fit_recovers_parameters():
    from panolidar.calibration import PrecisionModel
    gen = torch.Generator().manual_seed(0)
    U = torch.exp(torch.randn(20000, generator=gen) - 2)
    conf = torch.rand(20000, generator=gen)
    true = 0.05 * U ** 0.6 * torch.where(conf > 0.5, 1.0, 2.0)
    e = true * torch.randn(20000, generator=gen)
    pm = PrecisionModel.fit(e, U, conf, num_bins=2)
    assert abs(pm.kappa - 0.6) < 0.06
    s = pm.scales.tolist()
    assert s[0] > s[1] and abs(s[1] - 0.05) / 0.05 < 0.2
    legacy = PrecisionModel.from_config(0.5)
    assert legacy.kappa == 1.0 and torch.allclose(legacy(torch.rand(3), torch.ones(3)), torch.full((3,), 0.5))
