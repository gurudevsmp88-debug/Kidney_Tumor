"""Geometry-conditioned calibration of the stereo uncertainty.

The closed-form triangulation sigma (uncertainty.py) gives, for a unit angular
matching precision (sigma_px = 1), a log-depth std U per ray that captures all
geometry: depth, baseline length and direction (epipoles), panorama resolution.
Real matching precision is not a constant number of pixels: window aggregation,
texture, the cost function and the bounded depth search make the observed error
grow sub-linearly in U. We therefore calibrate a monotone map

    sigma_match = a(c) * U^kappa,      sigma_stereo^2 = sigma_match^2 + P^2,

where c is the match confidence (a(c) piecewise constant, non-increasing in c),
kappa a shared exponent, and P the closed-form pose-error term (no free
parameters). (a, kappa) are fitted once on calibration data -- in practice the
city teacher -- by maximising a robust Student-t likelihood of the observed
log-depth errors, which tolerates occlusions and mismatches. kappa = 1 and a
single bin recovers the pure closed form with a constant sigma_px = a.
Because U normalises baseline, depth and resolution, the calibration transfers
across capture configurations (tested in the experiments).
"""
import math

import torch

from .fusion import relative_to_log_std
from .uncertainty import triangulation_depth_std

A_GRID = torch.logspace(math.log10(1e-3), math.log10(10.0), 161)
KAPPA_GRID = torch.linspace(0.3, 1.2, 37)


def unit_and_pose_sigma(rays, depth, t12, width, sigma_rot=0.0, sigma_trans=0.0, max_std=1e4):
    """(U, P): log-depth sigma for sigma_px = 1 without pose error, and the pose-only part."""
    unit_ang = 2 * math.pi / width
    U = relative_to_log_std(depth, triangulation_depth_std(rays, depth, t12, unit_ang, max_std=max_std))
    if sigma_rot == 0 and sigma_trans == 0:
        return U, torch.zeros_like(U)
    P = relative_to_log_std(depth, triangulation_depth_std(rays, depth, t12, 0.0, max_std=max_std,
                                                           sigma_rot=sigma_rot, sigma_trans=sigma_trans))
    return U, P


def student_t_nll(e, s, nu=3.0):
    return torch.log(s) + 0.5 * (nu + 1) * torch.log1p((e / s) ** 2 / nu)


def gaussian_nll(e, s):
    return torch.log(s) + 0.5 * (e / s) ** 2


def _best_scale(errors, base, P, nu, grid=A_GRID, family='t'):
    """Grid-search ML scale a for s = sqrt((a * base)^2 + P^2). Returns (a, mean nll)."""
    s = torch.sqrt((grid.view(-1, 1) * base.view(1, -1)) ** 2 + P.view(1, -1) ** 2).clamp_min(1e-9)
    e = errors.view(1, -1)
    nll = (student_t_nll(e, s, nu) if family == 't' else gaussian_nll(e, s)).mean(1)
    k = int(nll.argmin())
    return float(grid[k]), float(nll[k])


def _gaussian_scale(errors, base, P, nu, outlier_k=4.0):
    """Gaussian-consistent scale: robust t-fit to find inliers, then Gaussian ML on the inliers.

    The t-scale itself is ~20% below the std for Gaussian errors, and fusion needs a std.
    """
    a_t, _ = _best_scale(errors, base, P, nu)
    s_t = torch.sqrt((a_t * base) ** 2 + P ** 2)
    inl = errors.abs() < outlier_k * s_t
    if inl.sum() < 10:
        return a_t
    return _best_scale(errors[inl], base[inl], P[inl], nu, family='gauss')[0]


def fit_sigma_px(errors, U, P=None, nu=3.0):
    """Robust ML estimate of a constant sigma_px (kappa = 1) from log-depth errors."""
    P = torch.zeros_like(U) if P is None else P
    return _gaussian_scale(errors, U, P, nu)


class PrecisionModel:
    """sigma_match = a(confidence) * U^kappa (a piecewise constant in confidence)."""

    def __init__(self, edges, scales, kappa=1.0):
        self.edges = torch.as_tensor(edges, dtype=torch.float32)    # (K-1,) interior confidence edges
        self.scales = torch.as_tensor(scales, dtype=torch.float32)  # (K,) a per bin
        self.kappa = float(kappa)

    def scale(self, confidence):
        idx = torch.bucketize(confidence.contiguous(), self.edges.to(confidence.device))
        return self.scales.to(confidence.device, confidence.dtype)[idx]

    def __call__(self, confidence, U):
        """Matching part of the stereo log-depth sigma."""
        return self.scale(confidence) * U.clamp_min(1e-12) ** self.kappa

    def to_dict(self):
        return {'edges': self.edges.tolist(), 'scales': self.scales.tolist(), 'kappa': self.kappa}

    @classmethod
    def from_config(cls, sigma_px):
        """From a scalar sigma_px (pure closed form) or a to_dict() result."""
        if isinstance(sigma_px, dict):
            if 'scales' in sigma_px:
                return cls(sigma_px['edges'], sigma_px['scales'], sigma_px.get('kappa', 1.0))
            return cls(sigma_px['edges'], sigma_px['sigma_px'], 1.0)
        return cls([], [float(sigma_px)], 1.0)

    @classmethod
    def fit(cls, errors, U, confidence, P=None, num_bins=6, min_count=200, nu=3.0, fit_kappa=True,
            max_uncertainty=0.5, max_rays=100000, seed=0):
        """Fit on calibration rays (1-D tensors). Rays whose predicted sigma exceeds max_uncertainty
        (deep inside epipolar cones, where the bounded depth search truncates errors) are excluded."""
        P = torch.zeros_like(U) if P is None else P
        keep = torch.isfinite(errors) & torch.isfinite(U) & (U > 0) & torch.isfinite(confidence)
        errors, U, confidence, P = errors[keep], U[keep], confidence[keep], P[keep]
        if len(errors) > max_rays:  # random subsample: the fit has only a handful of parameters
            sel = torch.randperm(len(errors), generator=torch.Generator().manual_seed(seed))[:max_rays]
            errors, U, confidence, P = errors[sel], U[sel], confidence[sel], P[sel]
        kappas = KAPPA_GRID if fit_kappa else torch.tensor([1.0])

        def fit_global(e, u, p):
            best = (1.0, 1.0, math.inf)
            for kappa in kappas.tolist():
                a, nll = _best_scale(e, u ** kappa, p, nu)
                if nll < best[2]:
                    best = (kappa, a, nll)
            return best

        kappa, a, _ = fit_global(errors, U, P)
        informative = a * U ** kappa < max_uncertainty
        errors, U, confidence, P = errors[informative], U[informative], confidence[informative], P[informative]
        kappa, _, _ = fit_global(errors, U, P)
        a = _gaussian_scale(errors, U ** kappa, P, nu)
        if len(errors) < 2 * min_count or num_bins <= 1:
            return cls([], [a], kappa)
        edges = torch.quantile(confidence, torch.linspace(0, 1, num_bins + 1)[1:-1])
        idx = torch.bucketize(confidence, edges)
        scales = []
        for k in range(num_bins):
            m = idx == k
            scales.append(_gaussian_scale(errors[m], U[m] ** kappa, P[m], nu) if m.sum() >= min_count else a)
        for k in range(num_bins - 2, -1, -1):  # non-increasing in confidence
            scales[k] = max(scales[k], scales[k + 1])
        return cls(edges.tolist(), scales, kappa)


def fit_baseline_sigmas(errors, confidence, nu=3.0, max_rays=100000, seed=0):
    """Fair baselines fitted on the same calibration rays with the same robust estimator:
    a constant log-depth sigma, and sigma = s / confidence."""
    keep = torch.isfinite(errors) & torch.isfinite(confidence)
    errors, confidence = errors[keep], confidence[keep]
    if len(errors) > max_rays:
        sel = torch.randperm(len(errors), generator=torch.Generator().manual_seed(seed))[:max_rays]
        errors, confidence = errors[sel], confidence[sel]
    zeros = torch.zeros_like(errors)
    constant = _gaussian_scale(errors, torch.ones_like(errors), zeros, nu)
    conf_scale = _gaussian_scale(errors, 1.0 / confidence.clamp_min(1e-3), zeros, nu)
    return {'constant_sigma': constant, 'confidence_scale': conf_scale}
