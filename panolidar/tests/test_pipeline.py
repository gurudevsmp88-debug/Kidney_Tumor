import json
import math
import os

import numpy as np
import torch
from PIL import Image

from panolidar import cli
from panolidar.experiments import fusion_benchmark, make_training_sample, train_synthetic
from panolidar.fusion import align_prior_scale, fuse_log_depth
from panolidar.metrics import chamfer_distance, depth_metrics
from panolidar.model import PanoLidarNet
from panolidar.sphere import equirect_rays
from panolidar.stereo import two_view_depth
from panolidar.synthetic import random_street, render_panorama, synthetic_prior
from panolidar.teacher import gaussian_nll_log
from panolidar.uncertainty import angle_to_baseline

CFG = cli.load_config()


def test_stereo_is_accurate_away_from_epipoles():
    scene = random_street(0)
    img1, gt = render_panorama(scene, (0, 0, 0), 64, 128)
    img2, _ = render_panorama(scene, (0, 4, 0), 64, 128)
    out = two_view_depth(img1[None], img2[None], torch.eye(3)[None], torch.tensor([[0.0, 4.0, 0.0]]))
    ang = angle_to_baseline(equirect_rays(64, 128), torch.tensor([0.0, 4.0, 0.0]))
    side = torch.isfinite(gt) & (ang > math.radians(60))
    cone = torch.isfinite(gt) & (ang < math.radians(10))
    rel = (out['depth'][0] - gt).abs() / gt
    assert rel[side].median() < 0.05
    assert out['confidence'][0][side].mean() > out['confidence'][0][cone].mean() + 0.2


def test_fusion_math_and_gating():
    mu_p, s_p = torch.zeros(1, 2, 2), torch.full((1, 2, 2), 0.1)
    mu, s, w = fuse_log_depth(mu_p, s_p)
    assert torch.equal(mu, mu_p) and torch.equal(s, s_p) and not w.any()
    mu_s, s_s = torch.full((1, 2, 2), 0.05), torch.full((1, 2, 2), 0.1)
    mu, s, w = fuse_log_depth(mu_p, s_p, mu_s, s_s, torch.ones(1, 2, 2), gate='none')
    assert torch.allclose(mu, torch.full_like(mu, 0.025)) and torch.allclose(s, torch.full_like(s, 0.1 / math.sqrt(2)))
    outlier = torch.full((1, 2, 2), 2.0)  # ~14 combined sigmas away: stereo must be ignored
    for gate in ('mixture', 'quartic'):
        mu, _, w = fuse_log_depth(mu_p, s_p, outlier, s_s, torch.ones(1, 2, 2), gate=gate)
        assert w.max() < 1e-2 and mu.abs().max() < 1e-2
    inlier = torch.full((1, 2, 2), 0.05)  # well within the combined sigma: kept
    _, _, w = fuse_log_depth(mu_p, s_p, inlier, s_s, torch.ones(1, 2, 2), gate='mixture')
    assert w.min() > 0.9


def test_scale_alignment_recovers_offset():
    gen = torch.Generator().manual_seed(0)
    mu_s = torch.rand(1, 32, 32, generator=gen) * 3
    mu_p = mu_s - 0.08 + 0.01 * torch.randn(1, 32, 32, generator=gen)
    off = align_prior_scale(mu_p, mu_s, torch.full_like(mu_s, 0.01), torch.ones_like(mu_s))
    assert abs(off.item() - 0.08) < 0.01


def test_benchmark_fused_beats_sources_and_is_calibrated():
    table = fusion_benchmark(CFG, seeds=(0, 1))
    fused, prior, stereo = table['fused'], table['prior_aligned'], table['stereo']
    assert fused['all']['median_rel'] < prior['all']['median_rel']
    assert fused['0-10deg']['median_rel'] < 0.5 * stereo['0-10deg']['median_rel']
    assert fused['60-90deg']['median_rel'] < prior['60-90deg']['median_rel']
    assert 0.5 < fused['all']['calib_1s'] < 0.9


def test_teacher_targets_have_amodal_layer_and_finite_loss():
    sample = make_training_sample(0, 64, 128, 0.5 * 2 * math.pi / 128)
    t = sample['targets']
    both = torch.isfinite(t['depth1']) & torch.isfinite(t['depth2'])
    assert both.sum() > 20 and (t['depth2'][both] > t['depth1'][both]).all()
    target = t['depth1'].unsqueeze(0)
    good = gaussian_nll_log(torch.log(torch.where(torch.isfinite(target), target, torch.ones_like(target))),
                            torch.full_like(target, 0.05), target, t['var1'].unsqueeze(0))
    bad = gaussian_nll_log(torch.full_like(target, 0.0), torch.full_like(target, 0.05), target, t['var1'].unsqueeze(0))
    assert torch.isfinite(good) and good < bad


def test_model_forward_backward_two_and_one_panorama():
    model = PanoLidarNet(stereo_kwargs=CFG['train']['stereo'])
    sample = make_training_sample(1, 32, 64, 0.5 * 2 * math.pi / 64)
    p = lambda q: (q[0][None], q[1][None])
    out = model(sample['image1'][None], sample['image2'][None], p(sample['pose1']), p(sample['pose2']))
    loss = out['mu'].mean() + out['sigma'].mean() + out['layer2_mu'].mean()
    loss.backward()
    assert all(torch.isfinite(v).all() for v in out.values() if torch.is_tensor(v))
    single = model(sample['image1'][None], external_prior=torch.zeros(1, 32, 64))
    assert single['mu'].shape == (1, 32, 64) and 'stereo_mu' not in single


def test_train_synthetic_runs():
    cfg = dict(CFG, train=dict(CFG['train'], height=32, width=64))
    _, history = train_synthetic(cfg, steps=3, log_every=0)
    assert len(history) == 3 and all(math.isfinite(h['loss']) for h in history)


def test_metrics():
    gt = torch.full((4, 4), 10.0)
    m = depth_metrics(gt + 0.08, gt, torch.full((4, 4), 0.01))
    assert m['within_0.1m'] == 1.0 and m['within_0.05m'] == 0.0 and m['calib_1s'] == 1.0
    a = torch.rand(50, 3)
    assert chamfer_distance(a, a) == 0.0


def test_cli_reconstruct_on_posed_folder(tmp_path):
    scene = random_street(3)
    entries = []
    for i, y in enumerate((0.0, 4.0)):
        img, depth = render_panorama(scene, (0, y, 0), 64, 128)
        Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(tmp_path / f'pano_{i}.png')
        mu_p, _ = synthetic_prior(depth, seed=i)
        np.save(tmp_path / f'prior_{i}.npy', mu_p.exp().numpy())
        np.save(tmp_path / f'depth_{i}.npy', torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth)).numpy())
        entries.append({'name': str(i), 'image': f'pano_{i}.png', 'R': torch.eye(3).tolist(), 'C': [0.0, y, 0.0],
                        'prior_depth': f'prior_{i}.npy', 'depth': f'depth_{i}.npy'})
    (tmp_path / 'poses.json').write_text(json.dumps({'panoramas': entries}))
    out = tmp_path / 'ref0.ply'
    res = cli.main(['reconstruct', '--scene', str(tmp_path), '--ref', '0', '--height', '64', '--width', '128',
                    '--out', str(out)])
    assert out.exists() and os.path.exists(tmp_path / 'ref0_depth.npy')
    assert res['stereo_depth'] is not None  # partner found automatically
    single = cli.main(['reconstruct', '--scene', str(tmp_path), '--ref', '0', '--single', '--height', '64',
                       '--width', '128', '--out', str(tmp_path / 'single.ply')])
    assert single['stereo_depth'] is None
