#
# CPU toy experiment for split-half noise-unbiased densification (SHD).
#
# A differentiable 2D splatting toy: K isotropic Gaussians on a grid render a
# grayscale image. Some Gaussians are displaced from their true position
# ("mis-fit": they really need densification); the rest sit exactly at the
# truth ("well-fit": densifying them only adds primitives that fit noise).
# Each "view" is an independent noisy observation of the clean target.
#
# For several noise levels the script compares the vanilla 3DGS score
# (mean_v ||g_v||) with the SHD score (2 sqrt(max(0, mean_v <g_A,v, g_B,v>))).
# Each score's threshold is calibrated once on clean data, then reused at
# every noise level, as a fixed densify_grad_threshold would be. It reports how
# often well-fit Gaussians would be densified (false positives) and how often
# mis-fit ones would be (true positives).
#
# Usage: python split_half_toy.py [--views 200] [--noise 0 0.05 0.1 0.2] [--out toy.json]
#

import json
from argparse import ArgumentParser

import torch

from utils.split_half import cross_term, sample_split_mask, split_half_l1_losses, split_half_score


def render(means, colors, sigma, height, width):
    ys = torch.arange(height, dtype=means.dtype).view(1, height, 1)
    xs = torch.arange(width, dtype=means.dtype).view(1, 1, width)
    d2 = (xs - means[:, 0].view(-1, 1, 1)) ** 2 + (ys - means[:, 1].view(-1, 1, 1)) ** 2
    # 3-sigma cutoff as in the 3DGS rasterizer, so footprints of separated Gaussians do not overlap
    weights = torch.exp(-d2 / (2.0 * sigma ** 2)) * (d2 <= (3.0 * sigma) ** 2)
    return (colors.view(-1, 1, 1) * weights).sum(0, keepdim=True)  # (1, H, W)


def make_scene(grid=6, spacing=20, sigma=2.5, misfit_every=2, displacement=(0.25, 1.5), seed=0):
    gen = torch.Generator().manual_seed(seed)
    size = grid * spacing
    coords = torch.arange(grid, dtype=torch.float64) * spacing + spacing / 2
    true_means = torch.cartesian_prod(coords, coords)
    colors = 0.3 + 0.5 * torch.rand(len(true_means), generator=gen, dtype=torch.float64)
    misfit = torch.zeros(len(true_means), dtype=torch.bool)
    misfit[::misfit_every] = True
    direction = torch.randn(int(misfit.sum()), 2, generator=gen, dtype=torch.float64)
    direction = direction / direction.norm(dim=1, keepdim=True)
    means = true_means.clone()
    lo, hi = displacement
    magnitude = lo + (hi - lo) * torch.rand(int(misfit.sum()), 1, generator=gen, dtype=torch.float64)
    means[misfit] += magnitude * direction
    target = render(true_means, colors, sigma, size, size)
    return dict(means=means, colors=colors, sigma=sigma, size=size, target=target, misfit=misfit)


def scores_for_noise(scene, noise_std, views, block_size=1, seed=0):
    """Return (vanilla, split_half) per-Gaussian scores accumulated over `views` noisy views."""
    gen = torch.Generator().manual_seed(seed)
    n = len(scene['means'])
    vanilla_accum = torch.zeros(n, 1, dtype=torch.float64)
    cross_accum = torch.zeros(n, 1, dtype=torch.float64)
    denom = torch.full((n, 1), float(views), dtype=torch.float64)
    for _ in range(views):
        gt = scene['target'] + noise_std * torch.randn(scene['target'].shape, generator=gen, dtype=torch.float64)
        means = scene['means'].clone().requires_grad_(True)
        image = render(means, scene['colors'], scene['sigma'], scene['size'], scene['size'])

        mask = sample_split_mask(scene['size'], scene['size'], block_size, generator=gen)
        loss_a, loss_b = split_half_l1_losses(image, gt, mask)
        grad_a, = torch.autograd.grad(loss_a, means, retain_graph=True)
        grad_b, = torch.autograd.grad(loss_b, means, retain_graph=True)
        grad_full, = torch.autograd.grad(torch.abs(image - gt).mean(), means)

        vanilla_accum += grad_full.norm(dim=-1, keepdim=True)
        cross_accum += cross_term(grad_a, grad_b)
    return (vanilla_accum / denom).squeeze(1), split_half_score(cross_accum, denom).squeeze(1)


def auc(positive, negative):
    """Probability a mis-fit Gaussian scores above a well-fit one (threshold-free separation)."""
    diff = positive.view(-1, 1) - negative.view(1, -1)
    return ((diff > 0).double() + 0.5 * (diff == 0).double()).mean().item()


def run(noise_levels=(0.0, 0.05, 0.1, 0.2), views=200, block_size=1, seed=0):
    scene = make_scene(seed=seed)
    misfit = scene['misfit']
    per_noise = {}
    for i, std in enumerate(noise_levels):
        per_noise[std] = scores_for_noise(scene, std, views, block_size, seed=seed + 1 + i)

    # Fixed threshold per method, calibrated on the cleanest level (like a fixed densify_grad_threshold):
    # a quarter of the mean mis-fit score.
    clean = per_noise[min(noise_levels)]
    thresholds = [0.25 * s[misfit].mean().item() for s in clean]

    results = []
    for std in noise_levels:
        row = {'noise_std': std}
        for name, score, tau in zip(('vanilla', 'split_half'), per_noise[std], thresholds):
            flagged = score >= tau
            row[name] = {
                'score_well_fit': score[~misfit].mean().item(),
                'score_mis_fit': score[misfit].mean().item(),
                'false_positive_rate': flagged[~misfit].float().mean().item(),
                'true_positive_rate': flagged[misfit].float().mean().item(),
                'auc': auc(score[misfit], score[~misfit]),
            }
        results.append(row)
    return results


def format_table(results):
    lines = [f"{'noise':>6} | {'method':>10} | {'score well-fit':>14} | {'score mis-fit':>13} | {'FP rate':>7} | {'TP rate':>7} | {'AUC':>5}"]
    for row in results:
        for name in ('vanilla', 'split_half'):
            r = row[name]
            lines.append(f"{row['noise_std']:>6.3f} | {name:>10} | {r['score_well_fit']:>14.3e} | "
                         f"{r['score_mis_fit']:>13.3e} | {r['false_positive_rate']:>7.2f} | {r['true_positive_rate']:>7.2f} | {r['auc']:>5.3f}")
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = ArgumentParser(description='Split-half densification toy experiment (CPU)')
    parser.add_argument('--views', type=int, default=200)
    parser.add_argument('--noise', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument('--block_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    results = run(args.noise, args.views, args.block_size, args.seed)
    print(format_table(results))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
