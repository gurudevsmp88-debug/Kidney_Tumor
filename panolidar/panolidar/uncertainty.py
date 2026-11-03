"""Analytic geometric uncertainty for spherical triangulation.

Two-view depth uncertainty (the core of the method)
---------------------------------------------------
Camera 1 at the origin, camera 2 at baseline vector b. A ray d from camera 1
with depth z hits X = z d. In the triangle (C1, C2, X), let
  alpha = angle at C1 between d and b,
  beta  = angle at C2 between (C1 - C2) and (X - C2),
  gamma = angle at X (the parallax angle), alpha + beta + gamma = pi.
The law of sines gives z = |b| sin(beta) / sin(gamma). Perturbing the observed
directions by small in-plane angular errors d_alpha, d_beta (matching noise)
gives
  dz/dbeta  = |b| sin(alpha) / sin^2(gamma)
  dz/dalpha = |b| sin(beta) cos(gamma) / sin^2(gamma)
so with independent angular noise sigma_ang on both rays
  sigma_z^2 = sigma_ang^2 |b|^2 (sin^2 alpha + sin^2 beta cos^2 gamma) / sin^4 gamma.
Far field (z >> |b|, beta -> pi - alpha): sigma_z ~= sqrt(2) z^2 sigma_ang / (|b| sin alpha).
Near the epipoles (alpha -> 0 or pi), gamma -> 0 and sigma_z -> infinity: depth
along the baseline direction is unobservable from the two panoramas.

Relative-pose error (camera 1 is the reference frame):
  * a rotation error of camera 2 with per-axis std sigma_rot turns ray 2 within the
    epipolar plane by the same angle: it adds sigma_rot^2 to ray 2's angular variance;
  * a position error of camera 2 with per-axis std sigma_trans shifts ray 2's line;
    only its in-plane component perpendicular to ray 2 matters, and it moves the
    triangulated depth by delta / sin(gamma).
Hence
  sigma_z^2 = |b|^2 / sin^4 gamma * [sigma_ang^2 sin^2 beta cos^2 gamma
                                     + (sigma_ang^2 + sigma_rot^2) sin^2 alpha]
              + sigma_trans^2 / sin^2 gamma.

Multi-view point covariance (teacher)
-------------------------------------
A point seen from centres C_i at distances r_i with angular noise sigma_ang has
information sum_i (I - u_i u_i^T) / (r_i sigma_ang)^2 (each ray constrains the
two directions perpendicular to it). Its inverse is the triangulation covariance.
"""
import math

import torch


def triangulation_depth_std(rays, depth, baseline, sigma_ang, max_std=1e4, sigma_rot=0.0, sigma_trans=0.0):
    """Analytic std of camera-1 depth for two-view spherical triangulation.

    rays: (..., 3) unit directions in camera-1 frame
    depth: (...) depth hypothesis along each ray (e.g. the prior depth)
    baseline: (3,) camera-2 centre in camera-1 frame
    sigma_ang: angular matching std in radians (e.g. sigma_px * 2*pi / width)
    sigma_rot: per-axis std (radians) of the relative rotation error of camera 2
    sigma_trans: per-axis std (depth units) of the relative position error of camera 2
    Returns (...) std in the units of depth, clamped to max_std.
    """
    b = baseline.to(rays.dtype)
    b_norm = b.norm()
    to_x = rays * depth.unsqueeze(-1) - b
    to_x_norm = to_x.norm(dim=-1).clamp_min(1e-12)
    # Sines from cross-product norms (numerically stable for tiny angles, unlike sqrt(1 - cos^2)).
    b_rep = b.expand_as(rays)
    sin_alpha = torch.linalg.cross(rays, b_rep, dim=-1).norm(dim=-1) / b_norm
    sin_beta = torch.linalg.cross(to_x, b_rep, dim=-1).norm(dim=-1) / (to_x_norm * b_norm)
    # parallax angle between the two rays to X
    sin_gamma = torch.linalg.cross(rays, to_x, dim=-1).norm(dim=-1) / to_x_norm
    cos_gamma = (rays * to_x).sum(-1) / to_x_norm
    sin_g = sin_gamma.clamp_min(1e-12)
    num = b_norm ** 2 * (sigma_ang ** 2 * (sin_beta * cos_gamma) ** 2
                         + (sigma_ang ** 2 + sigma_rot ** 2) * sin_alpha ** 2)
    var = num / sin_g ** 4 + sigma_trans ** 2 / sin_g ** 2
    return torch.sqrt(var).clamp(max=max_std)


def angle_to_baseline(rays, baseline):
    """Angle (radians, in [0, pi/2]) between each ray's line and the baseline line."""
    cos = (rays @ baseline.to(rays.dtype)).abs() / baseline.norm()
    return torch.acos(cos.clamp(0.0, 1.0)).clamp(max=math.pi / 2)


def multiview_point_covariance(points, centers, visible, sigma_ang, reg=1e-9):
    """Triangulation covariance (N, 3, 3) of points seen from several panoramas.

    points: (N, 3) world points; centers: (V, 3) camera centres;
    visible: (N, V) bool visibility; sigma_ang: angular std (radians).
    Points seen from a single view get a large covariance along that ray.
    """
    out_dtype = points.dtype
    # float64: points seen from one view give a near-singular information matrix
    points, centers = points.double(), centers.double()
    diff = points.unsqueeze(1) - centers.unsqueeze(0)  # (N, V, 3)
    r = diff.norm(dim=-1).clamp_min(1e-9)
    u = diff / r.unsqueeze(-1)
    eye = torch.eye(3, dtype=points.dtype, device=points.device)
    proj = eye - u.unsqueeze(-1) * u.unsqueeze(-2)  # (N, V, 3, 3)
    w = visible.to(points.dtype) / (r * sigma_ang) ** 2
    info = (w.unsqueeze(-1).unsqueeze(-1) * proj).sum(1)
    info = info + reg * eye
    cov = torch.linalg.inv(info)
    return (0.5 * (cov + cov.transpose(-1, -2))).to(out_dtype)


def ray_variance(cov, rays):
    """Variance of each point along a viewing ray: d^T Sigma d. cov (..., 3, 3), rays (..., 3)."""
    return (rays.unsqueeze(-2) @ cov @ rays.unsqueeze(-1)).squeeze(-1).squeeze(-1).clamp_min(0.0)
