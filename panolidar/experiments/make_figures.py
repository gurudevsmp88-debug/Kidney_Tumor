"""Qualitative figure: panorama, closed-form sigma map, stereo-only and fused error maps."""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from panolidar.benchmark import IDENTITY, render_pair  # noqa: E402
from panolidar.calibration import unit_and_pose_sigma  # noqa: E402
from panolidar.cli import load_config  # noqa: E402
from panolidar.pipeline import reconstruct  # noqa: E402
from panolidar.sphere import equirect_rays  # noqa: E402
from panolidar.synthetic import synthetic_prior  # noqa: E402


def main(seed=3, H=256, W=512, baseline=4.0):
    cfg = load_config()
    cal = json.load(open(os.path.join(ROOT, 'results', 'calibration.json')))
    cfg['stereo']['sigma_px'] = {k: cal[k] for k in ('edges', 'scales', 'kappa')}
    img1, img2, gt = render_pair(seed, H, W, baseline)
    mu_p, s_p = synthetic_prior(gt, seed=seed)
    b = torch.tensor([0.0, baseline, 0.0])
    res = reconstruct(img1, (IDENTITY, torch.zeros(3)), mu_p, s_p, img2, (IDENTITY, b), cfg)
    valid = torch.isfinite(gt)
    U, _ = unit_and_pose_sigma(equirect_rays(H, W), torch.where(valid, gt, torch.ones_like(gt)), b, W)

    def rel(d):
        e = ((d - gt).abs() / gt).numpy()
        e[~valid.numpy()] = np.nan
        return e

    panels = [
        ('(a) reference panorama', img1.permute(1, 2, 0).numpy(), None, None),
        ('(b) closed-form $\\log_{10}\\,U$ (unit-precision $\\sigma$)',
         np.where(valid.numpy(), np.log10(U.numpy()), np.nan), 'viridis', (-2, 1)),
        ('(c) stereo only: rel. error', rel(res['stereo_depth']), 'magma', (0, 0.1)),
        ('(d) ours (fused): rel. error', rel(res['depth']), 'magma', (0, 0.1)),
    ]
    plt.rcParams.update({'font.size': 7, 'font.family': 'serif'})
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.25))
    for ax, (title, data, cmap, lim) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1],
                       interpolation='nearest')
        ax.set_title(title, fontsize=6.5, pad=2)
        ax.axis('off')
        if cmap is not None:
            cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
            cb.ax.tick_params(labelsize=5)
    for ax in axes[1:]:  # mark epipoles (forward: phi = 0, centre column; backward: phi = pi, image border)
        ax.plot([W / 2], [H / 2], marker='x', color='white', ms=3, mew=0.8)
        ax.plot([0, W - 1], [H / 2, H / 2], linestyle='none', marker='x', color='white', ms=3, mew=0.8)
    fig.tight_layout(pad=0.2, w_pad=0.4)
    out = os.path.join(ROOT, 'paper', 'figures', 'teaser.pdf')
    fig.savefig(out, dpi=300)
    print('wrote', out)


if __name__ == '__main__':
    main()
