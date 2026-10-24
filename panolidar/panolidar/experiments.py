"""Synthetic experiments: training-free fusion benchmark and student training with a city teacher."""
import math

import torch

from .metrics import depth_metrics, metrics_by_baseline_angle
from .model import PanoLidarNet
from .pipeline import reconstruct
from .sphere import equirect_rays
from .synthetic import random_street, render_panorama, synthetic_prior
from .teacher import build_teacher_points, gaussian_nll_log, layered_targets

IDENTITY = torch.eye(3)


def _mean_dicts(dicts):
    keys = [k for k in dicts[0] if k != 'count']
    out = {k: sum(d[k] for d in dicts if k in d) / max(1, sum(k in d for d in dicts)) for k in keys}
    out['count'] = sum(d.get('count', 0) for d in dicts)
    return out


def fusion_benchmark(cfg, seeds=(0, 1, 2), baseline=4.0, prior_rel_std=0.06, prior_scale_bias=0.1):
    """Compare prior-only, stereo-only and fused depth on synthetic streets, per baseline-angle bin."""
    H, W = cfg['resolution']['height'], cfg['resolution']['width']
    b = torch.tensor([0.0, baseline, 0.0])
    rays = equirect_rays(H, W)
    names = ('prior', 'prior_aligned', 'stereo', 'fused')
    rows = {n: [] for n in names}
    overall = {n: [] for n in names}
    for seed in seeds:
        scene = random_street(seed)
        img1, gt = render_panorama(scene, (0, 0, 0), H, W)
        img2, _ = render_panorama(scene, tuple(b.tolist()), H, W)
        mu_p, sigma_p = synthetic_prior(gt, prior_rel_std, prior_scale_bias, seed=seed)
        res = reconstruct(img1, (IDENTITY, torch.zeros(3)), mu_p, sigma_p, img2, (IDENTITY, b), cfg)
        preds = {'prior': (mu_p.exp(), sigma_p), 'prior_aligned': ((mu_p + res['scale_offset']).exp(), sigma_p),
                 'stereo': (res['stereo_depth'], res['stereo_sigma_log']),
                 'fused': (res['depth'], res['sigma_log'])}
        for name, (pred, sig) in preds.items():
            rows[name].append(metrics_by_baseline_angle(pred, gt, rays, b, sig))
            overall[name].append(depth_metrics(pred, gt, sig))
    table = {}
    for name in rows:
        table[name] = {bin_: _mean_dicts([r[bin_] for r in rows[name]]) for bin_ in rows[name][0]}
        table[name]['all'] = _mean_dicts(overall[name])
    return table


def format_benchmark(table, keys=('median_rel', 'within_0.1m', 'within_0.2m', 'calib_1s')):
    bins = list(table['fused'].keys())
    lines = [f"{'method':>13} | {'bin':>9} | " + ' | '.join(f'{k:>11}' for k in keys)]
    for name in table:
        for bin_ in bins:
            r = table[name][bin_]
            lines.append(f'{name:>13} | {bin_:>9} | ' + ' | '.join(f"{r.get(k, float('nan')):>11.3f}" for k in keys))
    return '\n'.join(lines)


def make_training_sample(seed, height, width, sigma_ang, n_teacher_views=7, spacing=3.0):
    """One synthetic sample: two student panoramas plus layered targets from a denser 'city' capture."""
    g = torch.Generator().manual_seed(seed)
    scene = random_street(seed)
    baseline = 2.0 + 4.0 * torch.rand((), generator=g).item()
    c1, c2 = torch.zeros(3), torch.tensor([0.0, baseline, 0.0])
    img1, _ = render_panorama(scene, tuple(c1.tolist()), height, width)
    img2, _ = render_panorama(scene, tuple(c2.tolist()), height, width)
    # Teacher views: a denser drive along the street, with sideways offsets to see behind objects.
    depths, poses = [], []
    for k in range(n_teacher_views):
        y = (k - n_teacher_views // 2) * spacing
        x = 1.5 * (2 * torch.rand((), generator=g).item() - 1)
        c = torch.tensor([x, y, 0.0])
        depths.append(render_panorama(scene, tuple(c.tolist()), height, width)[1])
        poses.append((IDENTITY, c))
    teacher = build_teacher_points(depths, poses, sigma_ang)
    targets = layered_targets(teacher['points'], teacher['cov'], IDENTITY, c1, height, width)
    return {'image1': img1, 'image2': img2, 'pose1': (IDENTITY, c1), 'pose2': (IDENTITY, c2),
            'targets': targets, 'teacher_points': len(teacher['points'])}


def student_loss(out, targets):
    t = {k: v.unsqueeze(0) for k, v in targets.items()}
    loss_fused = gaussian_nll_log(out['mu'], out['sigma'], t['depth1'], t['var1'])
    loss_prior = gaussian_nll_log(out['prior_mu'], out['prior_sigma'], t['depth1'], t['var1'])
    loss_layer2 = gaussian_nll_log(out['layer2_mu'], out['layer2_sigma'], t['depth2'], t['var2'])
    return loss_fused + loss_prior + 0.5 * loss_layer2, {
        'fused': loss_fused.item(), 'prior': loss_prior.item(), 'layer2': loss_layer2.item()}


def train_synthetic(cfg, steps=100, lr=1e-3, seed=0, log_every=10, model=None):
    """Train the reference student on synthetic streets with city-teacher targets (CPU-friendly)."""
    torch.manual_seed(seed)
    H, W = cfg['train']['height'], cfg['train']['width']
    sigma_ang = cfg['stereo']['sigma_px'] * 2 * math.pi / W
    model = model or PanoLidarNet(sigma_px=cfg['stereo']['sigma_px'], stereo_kwargs=cfg['train']['stereo'])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for step in range(steps):
        sample = make_training_sample(seed * 100003 + step, H, W, sigma_ang)
        pose = lambda p: (p[0].unsqueeze(0), p[1].unsqueeze(0))
        out = model(sample['image1'].unsqueeze(0), sample['image2'].unsqueeze(0),
                    pose(sample['pose1']), pose(sample['pose2']))
        loss, parts = student_loss(out, sample['targets'])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append({'step': step, 'loss': loss.item(), **parts})
        if log_every and step % log_every == 0:
            print(f"step {step}: loss {loss.item():.4f} (fused {parts['fused']:.3f}, prior {parts['prior']:.3f}, "
                  f"layer2 {parts['layer2']:.3f})")
    return model, history
