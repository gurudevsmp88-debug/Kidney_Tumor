"""Procedural street-canyon scenes rendered as equirectangular panoramas with exact depth.

The scene: a ground plane, two building facades along the street (+y is the
direction of travel), end walls, and axis-aligned boxes (parked cars, kiosks).
Surfaces carry a view-independent procedural texture so stereo is well posed
wherever the geometry allows. Rays that hit nothing (sky) get depth = inf.
"""
import math

import torch

from .sphere import equirect_rays

SKY_COLOR = (0.55, 0.7, 0.9)


def random_street(seed=0, camera_height=2.5):
    """Random street layout dict (all lengths in metres)."""
    g = torch.Generator().manual_seed(seed)

    def rnd(lo, hi):
        return lo + (hi - lo) * torch.rand((), generator=g).item()

    left, right = rnd(5.0, 9.0), rnd(5.0, 9.0)
    boxes = []
    for _ in range(int(rnd(3, 7))):
        side = -1 if torch.rand((), generator=g).item() < 0.5 else 1
        width = side * (left if side < 0 else right)
        x0 = width - side * rnd(1.0, 2.5)
        y0 = rnd(-30.0, 30.0)
        lo = torch.tensor([min(x0, x0 - side * 1.8), y0, -camera_height])
        hi = torch.tensor([max(x0, x0 - side * 1.8), y0 + rnd(3.5, 5.0), -camera_height + rnd(1.4, 2.0)])
        boxes.append((lo, hi))
    freqs = torch.randn(24, 3, generator=g, dtype=torch.float32)
    freqs = freqs / freqs.norm(dim=1, keepdim=True) * torch.logspace(-0.3, 0.9, 24).unsqueeze(1)
    return {
        'camera_height': camera_height, 'left': left, 'right': right,
        'facade_top': rnd(10.0, 25.0), 'length': rnd(60.0, 120.0), 'boxes': boxes,
        'freqs': freqs, 'phases': 2 * math.pi * torch.rand(24, 3, generator=g),
        'albedo': 0.3 + 0.5 * torch.rand(4 + len(boxes), 3, generator=g),
    }


def _plane_hit(origins, dirs, axis, value, lo, hi):
    """Intersection distance with the plane coord[axis] = value, bounded by box lo/hi (inf if none)."""
    denom = dirs[:, axis]
    t = (value - origins[:, axis]) / torch.where(denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom)
    p = origins + t.unsqueeze(1) * dirs
    inside = torch.ones_like(t, dtype=torch.bool)
    for a in range(3):
        if a != axis:
            inside &= (p[:, a] >= lo[a] - 1e-6) & (p[:, a] <= hi[a] + 1e-6)
    return torch.where((t > 1e-6) & inside, t, torch.full_like(t, math.inf))


def _box_hit(origins, dirs, lo, hi):
    inv = 1.0 / torch.where(dirs.abs() < 1e-12, torch.full_like(dirs, 1e-12), dirs)
    t0, t1 = (lo - origins) * inv, (hi - origins) * inv
    tmin = torch.minimum(t0, t1).max(1).values
    tmax = torch.maximum(t0, t1).min(1).values
    hit = (tmax >= tmin) & (tmax > 1e-6)
    t = torch.where(tmin > 1e-6, tmin, tmax)
    return torch.where(hit, t, torch.full_like(t, math.inf))


def ray_cast(scene, origins, dirs):
    """Distance (N,) and surface id (N,) of the first hit; id -1 / inf for sky."""
    h, L, top = scene['camera_height'], scene['length'], scene['facade_top']
    xl, xr = -scene['left'], scene['right']
    big = 1e6
    surfaces = [
        _plane_hit(origins, dirs, 2, -h, (xl, -L, -h), (xr, L, -h)),               # ground
        _plane_hit(origins, dirs, 0, xl, (xl, -big, -h), (xl, big, top)),          # left facade
        _plane_hit(origins, dirs, 0, xr, (xr, -big, -h), (xr, big, top)),          # right facade
        torch.minimum(_plane_hit(origins, dirs, 1, L, (xl, L, -h), (xr, L, top)),  # end walls
                      _plane_hit(origins, dirs, 1, -L, (xl, -L, -h), (xr, -L, top))),
    ]
    surfaces += [_box_hit(origins, dirs, lo, hi) for lo, hi in scene['boxes']]
    ts = torch.stack(surfaces, dim=1)
    t, idx = ts.min(dim=1)
    idx = torch.where(torch.isfinite(t), idx, torch.full_like(idx, -1))
    return t, idx


def shade(scene, points, surface):
    """View-independent procedural colour (N, 3)."""
    proj = points @ scene['freqs'].T  # (N, 24)
    pattern = torch.sin(proj.unsqueeze(-1) + scene['phases'].unsqueeze(0)).mean(1)  # (N, 3)
    albedo = scene['albedo'][surface.clamp_min(0)]
    color = albedo * (0.55 + 0.9 * pattern)
    sky = torch.tensor(SKY_COLOR).expand_as(color)
    return torch.where((surface >= 0).unsqueeze(1), color, sky).clamp(0.0, 1.0)


def render_panorama(scene, center, height=128, width=256):
    """Render (image (3, H, W), depth (H, W)) from an axis-aligned camera at `center`."""
    dirs = equirect_rays(height, width).reshape(-1, 3)
    center = torch.as_tensor(center, dtype=torch.float32)
    origins = center.expand_as(dirs)
    t, surface = ray_cast(scene, origins, dirs)
    points = origins + torch.where(torch.isfinite(t), t, torch.zeros_like(t)).unsqueeze(1) * dirs
    image = shade(scene, points, surface).T.reshape(3, height, width)
    return image, t.reshape(height, width)


def synthetic_prior(depth, rel_std=0.06, scale_bias=0.1, seed=0):
    """Simulated monocular metric prior: smooth relative error plus a global scale bias.

    Mimics a panoramic depth foundation model (~6% AbsRel) for controlled experiments.
    Returns (mu_p, sigma_p) in log-depth; sky pixels get a far placeholder.
    """
    g = torch.Generator().manual_seed(seed)
    H, W = depth.shape
    coarse = torch.randn(1, 1, max(H // 16, 2), max(W // 16, 2), generator=g)
    smooth = torch.nn.functional.interpolate(coarse, size=(H, W), mode='bicubic', align_corners=False)[0, 0]
    smooth = smooth / smooth.std().clamp_min(1e-6)
    bias = scale_bias * (2 * torch.rand((), generator=g) - 1)
    finite = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, 1e3))
    mu_p = finite.log() + rel_std * smooth + bias
    return mu_p, torch.full_like(mu_p, rel_std)
