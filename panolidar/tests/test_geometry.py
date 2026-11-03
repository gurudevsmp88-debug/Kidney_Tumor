import math

import torch

from panolidar.sphere import (angles_to_pixels, depth_to_points, equirect_rays, pixel_grid, project_points,
                              rays_to_angles, relative_pose, sample_equirect)
from panolidar.uncertainty import (angle_to_baseline, multiview_point_covariance, ray_variance,
                                   triangulation_depth_std)


def test_ray_angle_roundtrip_and_conventions():
    theta, phi = pixel_grid(32, 64)
    t2, p2 = rays_to_angles(equirect_rays(32, 64))
    assert torch.allclose(theta, t2, atol=1e-5) and torch.allclose(phi, p2, atol=1e-5)
    centre = equirect_rays(32, 64)[16, 32]
    assert centre[1] > 0.99  # image centre looks forward (+y)
    assert equirect_rays(32, 64)[0, 32][2] > 0.99  # top row looks up


def test_sampling_reproduces_pixels_and_wraps_longitude():
    img = torch.rand(1, 3, 16, 32)
    theta, phi = pixel_grid(16, 32)
    out = sample_equirect(img, theta[None], phi[None])
    assert torch.allclose(out, img, atol=1e-5)
    # a direction exactly on the seam blends first and last columns
    seam = sample_equirect(img, theta[None, :, :1], torch.full((1, 16, 1), math.pi))
    assert torch.allclose(seam[..., 0], 0.5 * (img[..., 0] + img[..., -1]), atol=1e-5)


def test_project_inverts_backproject_with_rotation():
    angle = 0.3
    R = torch.tensor([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0, 0, 1.0]])
    C = torch.tensor([1.0, 2.0, 0.5])
    depth = 3.0 + torch.rand(16, 32)
    pts = depth_to_points(depth, R, C)
    theta, phi, dist = project_points(pts, R, C)
    u, v = angles_to_pixels(theta, phi, 16, 32)
    uu, vv = torch.meshgrid(torch.arange(16.0), torch.arange(32.0), indexing='ij')
    assert torch.allclose(dist, depth, atol=1e-4)
    wrapped = (u - vv + 16) % 32 - 16  # column difference modulo the width (seam-safe)
    assert torch.allclose(v, uu, atol=1e-3) and wrapped.abs().max() < 1e-3
    R12, t12 = relative_pose(torch.eye(3), torch.zeros(3), R, C)
    assert torch.allclose(t12, C) and torch.allclose(R12, R)


def _triangulate(c1, d1, c2, d2):
    """Depth along ray 1 of the closest-approach midpoint of two rays."""
    w = c1 - c2
    a, b, c = (d1 * d1).sum(-1), (d1 * d2).sum(-1), (d2 * d2).sum(-1)
    d, e = (d1 * w).sum(-1), (d2 * w).sum(-1)
    s = (b * e - c * d) / (a * c - b * b)
    t = (a * e - b * d) / (a * c - b * b)
    p = 0.5 * ((c1 + s.unsqueeze(-1) * d1) + (c2 + t.unsqueeze(-1) * d2))
    return ((p - c1) * d1).sum(-1)


def _perturb_in_plane(d, normal, sigma, gen):
    """Rotate unit directions d (N,3) by N(0, sigma) within the plane with given normal."""
    ang = sigma * torch.randn(d.shape[0], 1, generator=gen, dtype=d.dtype)
    perp = torch.cross(normal.expand_as(d), d, dim=-1)
    perp = perp / perp.norm(dim=-1, keepdim=True)
    return torch.cos(ang) * d + torch.sin(ang) * perp


def test_analytic_triangulation_std_matches_monte_carlo():
    gen = torch.Generator().manual_seed(0)
    b = torch.tensor([0.0, 4.0, 0.0], dtype=torch.float64)
    sigma = 1e-3
    n = 20000
    for alpha_deg, depth in [(90, 20.0), (45, 15.0), (20, 30.0)]:
        a = math.radians(alpha_deg)
        ray = torch.tensor([math.sin(a), math.cos(a), 0.0], dtype=torch.float64)
        x = depth * ray
        d2 = (x - b) / (x - b).norm()
        normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)  # epipolar plane is z = 0
        r1 = _perturb_in_plane(ray.expand(n, 3), normal, sigma, gen)
        r2 = _perturb_in_plane(d2.expand(n, 3), normal, sigma, gen)
        z = _triangulate(torch.zeros(3, dtype=torch.float64), r1, b, r2)
        mc = (z * (r1 @ ray) - depth).std().item()  # depth error projected on the true ray
        analytic = triangulation_depth_std(ray[None], torch.tensor([depth], dtype=torch.float64), b, sigma).item()
        assert abs(mc - analytic) / analytic < 0.1, (alpha_deg, mc, analytic)
        far_field = depth ** 2 * sigma / (b.norm().item() * math.sin(a))
        assert 0.5 < analytic / far_field < 2.0


def test_uncertainty_explodes_at_epipoles():
    b = torch.tensor([0.0, 4.0, 0.0])
    rays = equirect_rays(32, 64)
    std = triangulation_depth_std(rays, torch.full((32, 64), 20.0), b, 1e-3)
    ang = angle_to_baseline(rays, b)
    # far field: sigma ~ z^2 sigma_ang / (|b| sin(alpha)), i.e. grows as 1/sin(alpha) towards the epipoles
    far = triangulation_depth_std(rays, torch.full((32, 64), 400.0), b, 1e-3)  # z >> |b|
    scaled = far * torch.sin(ang)
    assert scaled.max() / scaled.min() < 1.1
    assert std[ang < math.radians(6)].min() > 8 * std[ang > math.radians(80)].max()


def test_multiview_covariance():
    pts = torch.tensor([[0.0, 10.0, 0.0]])
    centers = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]])
    two = multiview_point_covariance(pts, centers, torch.tensor([[True, True]]), 1e-3)
    one = multiview_point_covariance(pts, centers, torch.tensor([[True, False]]), 1e-3)
    ray = torch.tensor([[0.0, 1.0, 0.0]])
    assert ray_variance(two, ray).item() < 1e-3  # second view fixes depth along ray 1
    assert ray_variance(one, ray).item() > 1e3   # single view: depth unconstrained
    assert ray_variance(one, torch.tensor([[1.0, 0.0, 0.0]])).item() < 1e-3  # lateral is constrained
