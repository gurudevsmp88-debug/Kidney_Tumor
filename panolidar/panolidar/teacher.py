"""City-as-teacher supervision.

A dense city capture (many posed panoramas) is reconstructed once with
multi-view geometry. Each teacher point gets a triangulation covariance from
the views that see it. A student that receives only 1-2 panoramas is then
supervised with:
  layer 1: the first teacher surface along each of its rays (visible geometry),
  layer 2: the first teacher surface behind it (amodal geometry: behind parked
           cars, around corners), seen only by *other* panoramas of the city.
Losses are Gaussian NLLs in log-depth that add the teacher's own variance, so
poorly triangulated teacher points supervise weakly instead of wrongly.
"""
import math

import torch

from .sphere import angles_to_pixels, depth_to_points, project_points
from .uncertainty import multiview_point_covariance, ray_variance


def visibility(points, depths, poses, rel_tol=0.02):
    """(N, V) bool: point projects inside view v and agrees with its depth map (not occluded)."""
    vis = []
    for depth, (R, C) in zip(depths, poses):
        H, W = depth.shape
        theta, phi, dist = project_points(points, R, C)
        u, v = angles_to_pixels(theta, phi, H, W)
        ui = u.round().long() % W
        vi = v.round().long().clamp(0, H - 1)
        observed = depth[vi, ui]
        vis.append(torch.isfinite(observed) & ((dist - observed).abs() <= rel_tol * dist))
    return torch.stack(vis, dim=1)


def build_teacher_points(depths, poses, sigma_ang, stride=2, rel_tol=0.02):
    """Fuse per-view depth maps (from MVS, or from LiDAR where available) into teacher points.

    depths: list of (H, W) depth maps (inf = no depth); poses: list of (R, C).
    Returns dict with points (N, 3), cov (N, 3, 3), n_views (N,).
    """
    pts = []
    for depth, (R, C) in zip(depths, poses):
        d = depth[::stride, ::stride]
        full = depth_to_points(torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth)), R, C)
        p = full[::stride, ::stride][torch.isfinite(d)]
        pts.append(p)
    points = torch.cat(pts)
    vis = visibility(points, depths, poses, rel_tol)
    centers = torch.stack([C for _, C in poses])
    cov = multiview_point_covariance(points, centers, vis, sigma_ang)
    return {'points': points, 'cov': cov, 'n_views': vis.sum(1)}


def layered_targets(points, cov, R, C, height, width, gap_rel=0.05, gap_abs=0.5, n_views=None):
    """Per-pixel two-layer teacher targets for a student panorama at pose (R, C).

    Returns dict of (H, W) tensors: depth1, var1, depth2, var2 (inf / nan where no target), and
    views1/views2 (number of teacher views of the chosen point, 0 where none) if n_views is given.
    var is the teacher variance along the student ray.
    """
    theta, phi, dist = project_points(points, R, C)
    u, v = angles_to_pixels(theta, phi, height, width)
    pix = (v.round().long().clamp(0, height - 1) * width + (u.round().long() % width))
    rays = (points - C) / dist.unsqueeze(-1).clamp_min(1e-9)
    var = ray_variance(cov, rays)
    n = height * width

    def first_surface(candidate):
        depth = torch.full((n,), math.inf, dtype=dist.dtype)
        depth.scatter_reduce_(0, pix[candidate], dist[candidate], reduce='amin')
        chosen = candidate & (dist == depth[pix])
        out_var = torch.full((n,), math.nan, dtype=dist.dtype)
        out_var[pix[chosen]] = var[chosen]
        views = torch.zeros(n, dtype=torch.long)
        if n_views is not None:
            views[pix[chosen]] = n_views[chosen].long()
        return depth, out_var, views

    all_points = torch.ones_like(dist, dtype=torch.bool)
    depth1, var1, views1 = first_surface(all_points)
    behind = dist > depth1[pix] * (1 + gap_rel) + gap_abs
    depth2, var2, views2 = first_surface(behind)
    shape = (height, width)
    out = {'depth1': depth1.view(shape), 'var1': var1.view(shape),
           'depth2': depth2.view(shape), 'var2': var2.view(shape)}
    if n_views is not None:
        out['views1'], out['views2'] = views1.view(shape), views2.view(shape)
    return out


def perturb_teacher(points, cov, generator, max_rel_std=0.5):
    """Simulate a real multi-view teacher: displace each point by a draw from its own covariance.

    Ill-conditioned points (e.g. seen from a single view) get displacements capped at
    max_rel_std times their distance from the origin along each principal axis.
    """
    evals, evecs = torch.linalg.eigh(cov.double())
    cap = (max_rel_std * points.double().norm(dim=-1, keepdim=True)) ** 2
    std = torch.sqrt(torch.minimum(evals.clamp_min(0), cap))
    z = torch.randn(points.shape, generator=generator, dtype=torch.float64)
    return points + (evecs @ (std * z).unsqueeze(-1)).squeeze(-1).to(points.dtype)


def gaussian_nll_log(mu, sigma, target_depth, target_var=None, max_target_rel_std=0.2):
    """Mean Gaussian NLL in log-depth over valid targets; adds teacher variance.

    mu, sigma: predicted log-depth mean/std (B, H, W); target_depth (B, H, W) with inf = none.
    """
    valid = torch.isfinite(target_depth) & (target_depth > 0)
    target_log = torch.log(torch.where(valid, target_depth, torch.ones_like(target_depth)))
    t_var = torch.zeros_like(mu)
    if target_var is not None:
        rel_var = target_var / torch.where(valid, target_depth, torch.ones_like(target_depth)) ** 2
        valid = valid & torch.isfinite(rel_var) & (rel_var >= 0) & (rel_var < max_target_rel_std ** 2)
        t_var = torch.where(valid, rel_var, torch.zeros_like(rel_var))
    var = sigma ** 2 + t_var
    nll = 0.5 * ((mu - target_log) ** 2 / var + torch.log(var))
    if valid.sum() == 0:
        return mu.sum() * 0.0
    return nll[valid].mean()
