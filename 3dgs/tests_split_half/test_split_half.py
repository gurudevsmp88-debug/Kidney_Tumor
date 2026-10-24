"""CPU tests for split-half noise-unbiased densification (no CUDA extensions needed)."""
import importlib.util
import os
import sys
import types
from argparse import ArgumentParser

import pytest
import torch

import split_half_toy as toy
from utils.split_half import (add_image_noise, cross_term, gradient_snr, sample_split_mask,
                              split_half_gradients, split_half_l1_losses, split_half_score)


def _scene(**kwargs):
    return toy.make_scene(**kwargs)


def _render(scene, means):
    return toy.render(means, scene['colors'], scene['sigma'], scene['size'], scene['size'])


def test_split_mask_properties():
    gen = torch.Generator().manual_seed(0)
    mask = sample_split_mask(37, 53, block_size=4, generator=gen)
    assert mask.shape == (37, 53) and mask.dtype == torch.bool
    assert mask.any() and (~mask).any()
    assert 0.3 < mask.float().mean().item() < 0.7
    blocks = mask[:36, :52].reshape(9, 4, 13, 4)
    assert (blocks == blocks[:, :1, :, :1]).all()  # constant inside each 4x4 block
    tiny = sample_split_mask(1, 2, generator=gen)  # degenerate size still yields two halves
    assert tiny.any() and (~tiny).any()


def test_half_losses_sum_to_weighted_l1():
    gen = torch.Generator().manual_seed(1)
    image, gt = torch.rand(3, 8, 8, generator=gen), torch.rand(3, 8, 8, generator=gen)
    mask = sample_split_mask(8, 8, generator=gen)
    loss_a, loss_b = split_half_l1_losses(image, gt, mask, weight=0.8)
    assert torch.allclose(loss_a + loss_b, 0.8 * torch.abs(image - gt).mean())


def test_half_gradients_add_up_to_full_gradient():
    scene = _scene(grid=3)
    means = scene['means'].clone().requires_grad_(True)
    image = _render(scene, means)
    gt = scene['target'] + 0.05 * torch.randn(scene['target'].shape, dtype=torch.float64,
                                               generator=torch.Generator().manual_seed(2))
    mask = sample_split_mask(scene['size'], scene['size'], generator=torch.Generator().manual_seed(3))
    loss_a, loss_b = split_half_l1_losses(image, gt, mask)
    grad_a, grad_b = split_half_gradients(loss_a, loss_b, means)
    grad_full, = torch.autograd.grad(torch.abs(image - gt).mean(), means)
    assert torch.allclose(grad_a + grad_b, grad_full, atol=1e-12)


def test_split_half_gradients_handle_unused_points():
    points = torch.zeros(4, 3, requires_grad=True)
    used = torch.ones(2, requires_grad=True)
    loss = (used ** 2).sum()
    grad_a, grad_b = split_half_gradients(loss, loss, points)
    assert grad_a.shape == (4, 2) and not grad_a.any() and not grad_b.any()


def test_cross_term_is_unbiased_under_pixel_noise():
    """E<g_A, g_B> = <E g_A, E g_B> for a fixed split, while E||g||^2 carries a noise term."""
    scene = _scene(grid=3)
    gen = torch.Generator().manual_seed(4)
    mask = sample_split_mask(scene['size'], scene['size'], generator=gen)
    means = scene['means'].clone().requires_grad_(True)
    image = _render(scene, means)
    n_real, noise_std = 1500, 0.1
    sum_a = torch.zeros_like(scene['means'])
    sum_b = torch.zeros_like(scene['means'])
    sum_cross = torch.zeros(len(means), 1, dtype=torch.float64)
    sum_sq = torch.zeros(len(means), 1, dtype=torch.float64)
    for _ in range(n_real):
        gt = scene['target'] + noise_std * torch.randn(scene['target'].shape, generator=gen, dtype=torch.float64)
        grad_a, grad_b = split_half_gradients(*split_half_l1_losses(image, gt, mask), means)
        sum_a += grad_a
        sum_b += grad_b
        sum_cross += cross_term(grad_a, grad_b)
        sum_sq += ((grad_a + grad_b) ** 2).sum(-1, keepdim=True)
    mean_cross = sum_cross / n_real
    expected = cross_term(sum_a / n_real, sum_b / n_real)
    mean_sq = sum_sq / n_real
    scale = mean_sq.max()
    # Cross term matches the noise-free product up to Monte Carlo error ...
    assert (mean_cross - expected).abs().max() < 0.02 * scale
    # ... while the vanilla squared norm on well-fit Gaussians is dominated by noise.
    well_fit = ~scene['misfit']
    assert mean_sq[well_fit].mean() > 20 * mean_cross[well_fit].abs().mean()


def test_score_and_snr_helpers():
    accum = torch.tensor([[4.0], [-1.0], [0.0]])
    denom = torch.tensor([[1.0], [1.0], [0.0]])
    assert torch.allclose(split_half_score(accum, denom), torch.tensor([[4.0], [0.0], [0.0]]))
    g = torch.tensor([[1.0, 0.0]])
    signal, noise, snr = gradient_snr(g, g)
    assert signal.item() == 4.0 and noise.item() == 0.0 and snr.item() > 1e10


def test_add_image_noise_is_seeded_and_clamped():
    image = torch.full((3, 4, 4), 0.5)
    a = add_image_noise(image, 0.3, torch.Generator().manual_seed(0))
    b = add_image_noise(image, 0.3, torch.Generator().manual_seed(0))
    assert torch.equal(a, b) and a.min() >= 0 and a.max() <= 1 and not torch.equal(a, image)


def test_toy_experiment_split_half_suppresses_noise_floor():
    results = toy.run(noise_levels=(0.0, 0.2), views=60)
    clean, noisy = results
    for name in ('vanilla', 'split_half'):
        assert clean[name]['score_well_fit'] == 0.0 and clean[name]['auc'] == 1.0
    assert noisy['split_half']['score_well_fit'] < 0.5 * noisy['vanilla']['score_well_fit']
    assert noisy['split_half']['auc'] >= noisy['vanilla']['auc']


@pytest.fixture
def gaussian_model_module(monkeypatch):
    """Import scene.gaussian_model with its CUDA-only dependencies stubbed out."""
    simple_knn = types.ModuleType('simple_knn')
    simple_knn._C = types.SimpleNamespace(distCUDA2=None)
    monkeypatch.setitem(sys.modules, 'simple_knn', simple_knn)
    monkeypatch.setitem(sys.modules, 'simple_knn._C', simple_knn._C)
    plyfile = types.ModuleType('plyfile')
    plyfile.PlyData = plyfile.PlyElement = None
    monkeypatch.setitem(sys.modules, 'plyfile', plyfile)
    # Load the file directly: scene/__init__.py pulls in dataset readers (cv2, PIL, ...) not needed here.
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scene', 'gaussian_model.py')
    spec = importlib.util.spec_from_file_location('gaussian_model_under_test', path)
    gm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm)
    return gm


def test_gaussian_model_accumulates_cross_terms(gaussian_model_module):
    model = gaussian_model_module.GaussianModel(sh_degree=3)
    n = 5
    model.xyz_gradient_accum = torch.zeros(n, 1)
    model.denom = torch.zeros(n, 1)
    grad_a = torch.randn(n, 2)
    grad_b = torch.randn(n, 2)
    visible = torch.tensor([True, False, True, True, False]).nonzero()  # same format as render()
    model.add_split_half_stats(grad_a, grad_b, visible)
    model.add_split_half_stats(grad_a, grad_b, visible)
    expected = 2 * (grad_a * grad_b).sum(-1, keepdim=True)
    idx = visible.squeeze(1)
    assert torch.allclose(model.xyz_gradient_accum[idx], expected[idx])
    assert model.denom[idx].eq(2).all() and model.denom[~torch.isin(torch.arange(n), idx)].eq(0).all()


def test_cli_flags_parse():
    from arguments import ModelParams, OptimizationParams
    parser = ArgumentParser()
    lp, op = ModelParams(parser), OptimizationParams(parser)
    args = parser.parse_args(['-s', 'scene', '--densify_score', 'split_half', '--split_half_block', '2',
                              '--split_half_every', '3', '--train_noise_std', '0.1'])
    opt, data = op.extract(args), lp.extract(args)
    assert (opt.densify_score, opt.split_half_block, opt.split_half_every) == ('split_half', 2, 3)
    assert data.train_noise_std == 0.1 and data.train_noise_seed == 0
    defaults = op.extract(parser.parse_args(['-s', 'scene']))
    assert defaults.densify_score == 'vanilla'
