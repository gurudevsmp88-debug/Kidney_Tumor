#
# Split-half noise-unbiased densification (SHD) for 3D Gaussian Splatting.
#
# Vanilla 3DGS densifies a Gaussian when the view-averaged norm of its
# screen-space positional gradient, mean_v ||g_v||, exceeds a threshold.
# With noisy training images that statistic is biased upwards: the image
# noise contributes a strictly positive term to every ||g_v|| even when the
# Gaussian already fits the clean signal, so noisy captures over-densify and
# grow primitives that fit noise.
#
# SHD splits the pixels of each training view into two random disjoint halves
# A and B and computes the positional gradients g_A and g_B of the per-pixel
# photometric loss restricted to each half. If pixel noise is independent
# across pixels, g_A and g_B are conditionally independent given the current
# render, so
#
#     E[<g_A, g_B>] = <E[g_A], E[g_B]>,
#
# i.e. the noise drops out of the cross term in expectation. With interleaved
# random halves E[g_A] ~= E[g_B] ~= E[g] / 2, hence 4 <g_A, g_B> is an unbiased
# estimate of ||E[g]||^2, the squared gradient of the noise-free objective.
# This holds for any per-pixel loss (L1, L2, Charbonnier), because only
# independence between disjoint pixel sets is used, not linearity.
#
# The densification score of a Gaussian is
#
#     s = 2 * sqrt( max(0, mean_v <g_A,v, g_B,v>) ),
#
# which replaces mean_v ||g_v|| in densify_and_prune. The difference
# ||g_A - g_B||^2 additionally estimates the noise energy of the gradient,
# giving a per-Gaussian gradient SNR that is logged as a diagnostic.
#
# Everything in this file is plain PyTorch (no CUDA extension) so it can be
# unit tested on CPU.
#

import torch


def sample_split_mask(height, width, block_size=1, generator=None, device=None):
    """Random half split of the image plane.

    Returns a bool tensor (H, W): True = half A, False = half B. Pixels are
    grouped into block_size x block_size blocks that are assigned together,
    which keeps both halves independent when the noise is spatially
    correlated over a few pixels (e.g. after demosaicing). Both halves are
    guaranteed to be non-empty.
    """
    bh = -(-height // block_size)
    bw = -(-width // block_size)
    blocks = torch.rand((bh, bw), generator=generator, device=device) < 0.5
    if blocks.all() or not blocks.any():
        blocks = blocks.clone()
        blocks.view(-1)[0] = ~blocks.view(-1)[0]
    mask = blocks.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    return mask[:height, :width]


def split_half_l1_losses(image, gt_image, mask, weight=1.0):
    """L1 losses on the two halves of `mask`.

    Both are normalized by the total number of elements, so
    loss_a + loss_b == weight * mean(|image - gt_image|) and the gradients
    are on the same scale as the vanilla 3DGS L1 term.
    """
    residual = torch.abs(image - gt_image)
    mask = mask.to(residual.dtype)
    if residual.dim() == 3:
        mask = mask.unsqueeze(0)
    scale = weight / residual.numel()
    loss_a = (residual * mask).sum() * scale
    loss_b = (residual * (1.0 - mask)).sum() * scale
    return loss_a, loss_b


def split_half_gradients(loss_a, loss_b, points):
    """Positional gradients (N, 2) of each half-loss w.r.t. the screen-space means.

    The graph is retained so that the regular loss.backward() can run after.
    Gaussians a half-loss does not reach get a zero gradient.
    """
    grad_a, = torch.autograd.grad(loss_a, points, retain_graph=True, allow_unused=True)
    grad_b, = torch.autograd.grad(loss_b, points, retain_graph=True, allow_unused=True)
    if grad_a is None:
        grad_a = torch.zeros_like(points)
    if grad_b is None:
        grad_b = torch.zeros_like(points)
    return grad_a[..., :2].detach(), grad_b[..., :2].detach()


def cross_term(grad_a, grad_b):
    """<g_A, g_B> per Gaussian, shape (..., 1). Unbiased (x 4) for ||E[g]||^2."""
    return (grad_a * grad_b).sum(dim=-1, keepdim=True)


def split_half_score(accum, denom):
    """Densification score from accumulated cross terms: 2 * sqrt(max(0, mean))."""
    mean = accum / denom
    mean[mean.isnan()] = 0.0
    return 2.0 * torch.sqrt(torch.clamp_min(mean, 0.0))


def gradient_snr(grad_a, grad_b, eps=1e-20):
    """Per-Gaussian (signal energy, noise energy) estimates and their ratio.

    signal ~ 4 <g_A, g_B>   (estimates ||E[g]||^2)
    noise  ~ ||g_A - g_B||^2 (estimates E||g - E[g]||^2 for interleaved halves)
    """
    signal = 4.0 * cross_term(grad_a, grad_b)
    noise = ((grad_a - grad_b) ** 2).sum(dim=-1, keepdim=True)
    return signal, noise, signal / (noise + eps)


def add_image_noise(image, std, generator=None):
    """Additive Gaussian noise for controlled noisy-capture experiments."""
    # Drawn on CPU so a CPU generator gives the same noise on any device.
    noise = torch.randn(image.shape, generator=generator, dtype=image.dtype).to(image.device)
    return (image + std * noise).clamp(0.0, 1.0)
