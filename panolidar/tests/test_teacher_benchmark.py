"""Tests for teacher perturbation, view counts and benchmark scene effects."""
import math

import torch

from panolidar.benchmark import _moved_scene, capture_effects
from panolidar.synthetic import random_street
from panolidar.teacher import layered_targets, perturb_teacher
from panolidar.uncertainty import multiview_point_covariance


def test_perturb_teacher_follows_covariance():
    gen = torch.Generator().manual_seed(0)
    n = 20000
    pts = torch.tensor([[0.0, 20.0, 0.0]]).repeat(n, 1)
    centers = torch.tensor([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    cov = multiview_point_covariance(pts, centers, torch.ones(n, 2, dtype=torch.bool), 2e-3)
    d = (perturb_teacher(pts, cov, gen) - pts).double()
    emp = d.T @ d / n
    assert torch.allclose(emp, cov[0].double(), rtol=0.1, atol=1e-4)  # off-diagonal zeros: MC noise ~1e-5


def test_perturb_teacher_caps_single_view_points():
    gen = torch.Generator().manual_seed(1)
    pts = torch.tensor([[0.0, 10.0, 0.0]]).repeat(5000, 1)
    cov = multiview_point_covariance(pts, torch.zeros(1, 3), torch.ones(5000, 1, dtype=torch.bool), 1e-3)
    d = perturb_teacher(pts, cov, gen, max_rel_std=0.5) - pts
    assert d[:, 1].std() < 0.6 * 10.0 and d[:, 1].std() > 0.3 * 10.0  # along-ray std capped at 50%


def test_layered_targets_report_view_counts():
    pts = torch.tensor([[0.0, 10.0, 0.0], [0.0, 20.0, 0.0]])
    cov = torch.eye(3).repeat(2, 1, 1) * 1e-4
    t = layered_targets(pts, cov, torch.eye(3), torch.zeros(3), 32, 64, n_views=torch.tensor([3, 1]))
    m1, m2 = torch.isfinite(t['depth1']), torch.isfinite(t['depth2'])
    assert m1.sum() == 1 and m2.sum() == 1
    assert t['views1'][m1].item() == 3 and t['views2'][m2].item() == 1
    assert abs(t['depth1'][m1].item() - 10.0) < 1e-4 and abs(t['depth2'][m2].item() - 20.0) < 1e-4


def test_moved_scene_and_capture_effects():
    scene = random_street(0)
    moved = _moved_scene(scene, shift=3.0, fraction=1.0)
    assert all(abs((m[0] - o[0])[1].item() - 3.0) < 1e-6 for m, o in zip(moved['boxes'], scene['boxes']))
    assert scene['boxes'][0][0][1] != moved['boxes'][0][0][1]  # original untouched
    a, b = torch.full((3, 4, 4), 0.5), torch.full((3, 4, 4), 0.5)
    x, y = capture_effects(a, b, gain=0.6, bias=0.15)
    assert torch.equal(x, a) and torch.allclose(y, torch.full_like(y, 0.45))
    x, _ = capture_effects(a, b, noise=0.05, seed=3)
    assert not torch.equal(x, a) and x.min() >= 0 and x.max() <= 1
