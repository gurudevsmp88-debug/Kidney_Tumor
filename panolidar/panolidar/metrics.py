"""LiDAR-style evaluation: absolute accuracy, completeness and uncertainty calibration."""
import math

import torch

from .uncertainty import angle_to_baseline


def depth_metrics(pred, gt, sigma_log=None, mask=None, thresholds=(0.05, 0.10, 0.20)):
    """Metrics over pixels with finite GT. pred/gt (H, W) metres; sigma_log optional (H, W).

    within_Xm: fraction of pixels with |pred - gt| < X metres (LiDAR-style accuracy).
    calib_1s / calib_2s: fraction with |log pred - log gt| < 1 / 2 sigma (ideal 0.683 / 0.954).
    """
    valid = torch.isfinite(gt) & (gt > 0) & torch.isfinite(pred) & (pred > 0)
    if mask is not None:
        valid &= mask
    if valid.sum() == 0:
        return {'count': 0}
    p, g = pred[valid], gt[valid]
    err = (p - g).abs()
    out = {
        'count': int(valid.sum()),
        'abs_rel': (err / g).mean().item(),
        'median_rel': (err / g).median().item(),
        'rmse_m': torch.sqrt((err ** 2).mean()).item(),
        'mae_m': err.mean().item(),
    }
    for t in thresholds:
        out[f'within_{t:g}m'] = (err < t).float().mean().item()
    if sigma_log is not None:
        z = (p.log() - g.log()).abs() / sigma_log[valid].clamp_min(1e-9)
        out['calib_1s'] = (z < 1).float().mean().item()
        out['calib_2s'] = (z < 2).float().mean().item()
    return out


def metrics_by_baseline_angle(pred, gt, rays, baseline, sigma_log=None,
                              bins=((0, 10), (10, 30), (30, 60), (60, 90))):
    """depth_metrics per bin of angle between the ray and the baseline (degrees)."""
    ang = angle_to_baseline(rays, baseline)
    out = {}
    for lo, hi in bins:
        m = (ang >= math.radians(lo)) & (ang < math.radians(hi) + (1e-6 if hi == 90 else 0))
        out[f'{lo}-{hi}deg'] = depth_metrics(pred, gt, sigma_log, m)
    return out


def chamfer_distance(a, b, chunk=4096):
    """Symmetric mean nearest-neighbour distance between point clouds (N, 3), (M, 3)."""
    def one_way(x, y):
        mins = [torch.cdist(x[i:i + chunk], y, compute_mode='donot_use_mm_for_euclid_dist').min(1).values for i in range(0, len(x), chunk)]
        return torch.cat(mins).mean()
    return 0.5 * (one_way(a, b) + one_way(b, a)).item()
