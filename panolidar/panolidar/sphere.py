"""Equirectangular (ERP) geometry.

Conventions (camera frame): x right, y forward, z up.
  longitude phi in [-pi, pi), phi = 0 looks forward (+y), increasing to the right (+x)
  latitude theta in [-pi/2, pi/2], theta = +pi/2 looks up (+z)
  pixel (u, v): u = column, v = row; v = 0 is the top row.
A pose is (R, C): R rotates camera-frame directions into the world frame, C is
the camera centre in world coordinates, so X_world = C + depth * R @ ray_cam.
"""
import math

import torch
import torch.nn.functional as F


def pixel_grid(height, width, device=None, dtype=torch.float32):
    """Longitude/latitude of every pixel centre: two (H, W) tensors."""
    v = torch.arange(height, device=device, dtype=dtype) + 0.5
    u = torch.arange(width, device=device, dtype=dtype) + 0.5
    theta = math.pi / 2 - v / height * math.pi
    phi = u / width * 2 * math.pi - math.pi
    return torch.meshgrid(theta, phi, indexing='ij')


def angles_to_rays(theta, phi):
    """Unit ray directions (..., 3) from latitude/longitude."""
    cos_t = torch.cos(theta)
    return torch.stack((cos_t * torch.sin(phi), cos_t * torch.cos(phi), torch.sin(theta)), dim=-1)


def rays_to_angles(rays):
    """(theta, phi) from (not necessarily unit) directions (..., 3)."""
    rays = rays / rays.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    theta = torch.asin(rays[..., 2].clamp(-1.0, 1.0))
    phi = torch.atan2(rays[..., 0], rays[..., 1])
    return theta, phi


def equirect_rays(height, width, device=None, dtype=torch.float32):
    """Unit camera-frame rays (H, W, 3) for every ERP pixel."""
    return angles_to_rays(*pixel_grid(height, width, device, dtype))


def angles_to_grid(theta, phi):
    """Normalised grid_sample coordinates (..., 2) for align_corners=False."""
    return torch.stack((phi / math.pi, -2.0 * theta / math.pi), dim=-1)


def circular_pad(x, pad_w, pad_h=0):
    """Pad (B, C, H, W) with longitude wrap-around and latitude replication."""
    if pad_w:
        x = torch.cat((x[..., -pad_w:], x, x[..., :pad_w]), dim=-1)
    if pad_h:
        x = F.pad(x, (0, 0, pad_h, pad_h), mode='replicate')
    return x


def sample_equirect(image, theta, phi, mode='bilinear'):
    """Sample (B, C, H, W) ERP images at directions (B, ...) with longitude wrap-around.

    theta/phi have shape (B, h, w) (any h, w); returns (B, C, h, w).
    """
    width = image.shape[-1]
    pad = 2
    padded = circular_pad(image, pad)
    grid = angles_to_grid(theta, phi)
    grid = torch.stack((grid[..., 0] * width / (width + 2 * pad), grid[..., 1]), dim=-1)
    return F.grid_sample(padded, grid, mode=mode, padding_mode='border', align_corners=False)


def relative_pose(R1, C1, R2, C2):
    """Pose of panorama 2 expressed in panorama 1's camera frame: (R_12, t_12).

    A point with camera-1 coordinates x1 has camera-2 coordinates R_12^T (x1 - t_12).
    """
    R12 = R1.transpose(-1, -2) @ R2
    t12 = (R1.transpose(-1, -2) @ (C2 - C1).unsqueeze(-1)).squeeze(-1)
    return R12, t12


def depth_to_points(depth, R=None, C=None):
    """World points (H, W, 3) from an ERP depth map (H, W) (distance along the ray)."""
    rays = equirect_rays(*depth.shape, device=depth.device, dtype=depth.dtype)
    if R is not None:
        rays = rays @ R.transpose(-1, -2)
    points = rays * depth.unsqueeze(-1)
    if C is not None:
        points = points + C
    return points


def project_points(points, R, C):
    """Project world points (..., 3) into a panorama: (theta, phi, distance)."""
    local = (points - C) @ R  # R^T (X - C) for row vectors
    dist = local.norm(dim=-1)
    theta, phi = rays_to_angles(local)
    return theta, phi, dist


def angles_to_pixels(theta, phi, height, width):
    """Continuous pixel coordinates (u, v) of directions."""
    u = (phi + math.pi) / (2 * math.pi) * width - 0.5
    v = (math.pi / 2 - theta) / math.pi * height - 0.5
    return u, v
