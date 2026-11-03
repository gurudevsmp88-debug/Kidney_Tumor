"""Synthetic round-trip tests for the KITTI-360 adapter (no real data needed)."""
import math

import torch

from panolidar.kitti360 import (MEICamera, fisheyes_to_equirect, points_to_equirect_depth, read_cam_to_pose,
                                read_poses)
from panolidar.sphere import depth_to_points, rays_to_angles, sample_equirect
from panolidar.synthetic import random_street, render_panorama

# Plausible MEI parameters for a 1400x1400 180-degree fisheye (values in the style of KITTI-360).
CAM = MEICamera(xi=2.2, k1=0.016, k2=1.2, p1=0.0005, p2=-0.0006, gamma1=1336.0, gamma2=1335.0,
                u0=716.0, v0=705.0, width=1400, height=1400)


def _small(cam, scale=0.1):
    return MEICamera(cam.xi, cam.k1, cam.k2, cam.p1, cam.p2, cam.gamma1 * scale, cam.gamma2 * scale,
                     cam.u0 * scale, cam.v0 * scale, int(cam.width * scale), int(cam.height * scale))


def test_mei_project_unproject_roundtrip():
    gen = torch.Generator().manual_seed(0)
    r = 650 * torch.sqrt(torch.rand(500, generator=gen, dtype=torch.float64))  # inside the image circle
    a = 2 * math.pi * torch.rand(500, generator=gen, dtype=torch.float64)
    u, v = CAM.u0 + r * torch.cos(a), CAM.v0 + r * torch.sin(a)
    rays = CAM.unproject(u, v)
    u2, v2, valid = CAM.project(rays * 7.0)
    assert valid.all()
    assert (u2 - u).abs().max() < 1e-3 and (v2 - v).abs().max() < 1e-3


def test_yaml_and_text_parsers(tmp_path):
    (tmp_path / 'image_02.yaml').write_text(
        '%YAML:1.0\n---\nmodel_type: MEI\ncamera_name: image_02\nimage_width: 1400\nimage_height: 1400\n'
        'mirror_parameters:\n   xi: 2.2\ndistortion_parameters:\n   k1: 0.01\n   k2: 1.0\n   p1: 0.0\n   p2: 0.0\n'
        'projection_parameters:\n   gamma1: 1300.0\n   gamma2: 1301.0\n   u0: 700.0\n   v0: 701.0\n')
    cam = MEICamera.from_yaml(tmp_path / 'image_02.yaml')
    assert (cam.xi, cam.gamma2, cam.width) == (2.2, 1301.0, 1400)
    (tmp_path / 'cam_to_pose.txt').write_text('image_00: 1 0 0 1 0 1 0 2 0 0 1 3\nimage_02: 0 0 1 0 -1 0 0 0 0 -1 0 0\n')
    c2p = read_cam_to_pose(tmp_path / 'cam_to_pose.txt')
    assert torch.equal(c2p['image_00'][:3, 3], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    (tmp_path / 'poses.txt').write_text('5 1 0 0 0 0 1 0 0 0 0 1 0\n7 1 0 0 1 0 1 0 0 0 0 1 0\n')
    poses = read_poses(tmp_path / 'poses.txt')
    assert sorted(poses) == [5, 7] and poses[7][0, 3] == 1.0


def _render_fisheye(pano, cam, cam_to_rig):
    """Synthesise a fisheye image by sampling an ERP panorama along the fisheye's rays."""
    vv, uu = torch.meshgrid(torch.arange(cam.height, dtype=torch.float64), torch.arange(cam.width, dtype=torch.float64),
                            indexing='ij')
    rays_cam = cam.unproject(uu, vv)
    rays_rig = rays_cam @ cam_to_rig[:3, :3].T
    theta, phi = rays_to_angles(rays_rig.float())
    return sample_equirect(pano[None], theta[None], phi[None])[0]


def test_fisheye_pair_to_equirect_roundtrip():
    scene = random_street(0)
    pano, _ = render_panorama(scene, (0, 0, 0), 128, 256)
    cam = _small(CAM)
    # left camera looks along -x (ours), right along +x; OpenCV camera axes: x right, y down, z forward
    left = torch.eye(4, dtype=torch.float64)
    left[:3, :3] = torch.tensor([[0.0, 0, -1], [-1, 0, 0], [0, -1, 0]], dtype=torch.float64)
    right = torch.eye(4, dtype=torch.float64)
    right[:3, :3] = torch.tensor([[0.0, 0, 1], [1, 0, 0], [0, -1, 0]], dtype=torch.float64)
    fisheyes = [_render_fisheye(pano, cam, T) for T in (left, right)]
    rebuilt, covered = fisheyes_to_equirect(fisheyes, [cam, cam], [left, right], 128, 256)
    assert covered.float().mean() > 0.8  # two 180-degree fisheyes cover most of the sphere
    err = (rebuilt - pano).abs().mean(0)[covered]
    assert err.median() < 0.05


def test_points_to_equirect_depth_matches_render():
    scene = random_street(1)
    _, depth = render_panorama(scene, (0, 0, 0), 64, 128)
    finite = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
    pts = depth_to_points(finite.double())[torch.isfinite(depth)]
    proj = points_to_equirect_depth(pts, 64, 128)
    m = torch.isfinite(depth)
    assert torch.allclose(proj[m].float(), depth[m], rtol=1e-4)
    assert torch.isinf(proj[~m]).all()
