"""KITTI-360 adapter: fisheye pair -> equirectangular panorama, LiDAR -> ERP depth ground truth.

KITTI-360 (Liao et al., TPAMI 2022) provides two side-looking 180-degree fisheye cameras
(image_02 left, image_03 right) with the MEI unified camera model (Mei & Rives, ICRA 2007),
Velodyne scans and vehicle poses. Expected layout (as distributed):

  calibration/image_02.yaml, image_03.yaml   MEI intrinsics (OpenCV-YAML, '%YAML:1.0' header)
  calibration/calib_cam_to_pose.txt          'image_0X: r11 ... t3' (camera -> vehicle pose frame)
  calibration/calib_cam_to_velo.txt          12 numbers (camera 00 -> Velodyne)
  data_poses/<seq>/poses.txt                 'frame r11 ... t3' (vehicle pose frame -> world)
  data_2d_raw/<seq>/image_0X/data_rgb/%010d.png
  data_3d_raw/<seq>/velodyne_points/data/%010d.bin   float32 (x, y, z, intensity)

Frames: KITTI cameras are OpenCV-style (x right, y down, z forward); the vehicle pose frame is
x forward, y left, z up. Our panoramas use x right, y forward, z up (sphere.py), so
p_ours = VEHICLE_TO_OURS @ p_vehicle.

This module is unit-tested with synthetic round trips only; it has not yet been run on the real
dataset, so verify the frame conventions on the first real sequence (e.g. LiDAR overlay).
"""
import math
import os

import numpy as np
import torch

from .sphere import equirect_rays, project_points, angles_to_pixels

VEHICLE_TO_OURS = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------------------------
# MEI camera model
# ---------------------------------------------------------------------------------------------

class MEICamera:
    """Unified (MEI) omnidirectional camera with radial-tangential distortion."""

    def __init__(self, xi, k1, k2, p1, p2, gamma1, gamma2, u0, v0, width, height):
        self.xi, self.k1, self.k2, self.p1, self.p2 = xi, k1, k2, p1, p2
        self.gamma1, self.gamma2, self.u0, self.v0 = gamma1, gamma2, u0, v0
        self.width, self.height = width, height

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path) as f:
            text = ''.join(line for line in f if not line.startswith('%YAML'))
        d = yaml.safe_load(text)
        dist, proj = d['distortion_parameters'], d['projection_parameters']
        return cls(d['mirror_parameters']['xi'], dist['k1'], dist['k2'], dist['p1'], dist['p2'],
                   proj['gamma1'], proj['gamma2'], proj['u0'], proj['v0'], d['image_width'], d['image_height'])

    def _distort(self, x, y):
        r2 = x * x + y * y
        radial = 1 + self.k1 * r2 + self.k2 * r2 * r2
        xd = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        yd = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        return xd, yd

    def project(self, points):
        """Camera-frame points (..., 3) -> pixels (u, v) and a validity mask."""
        n = points / points.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        denom = n[..., 2] + self.xi
        valid = denom > 1e-6
        denom = torch.where(valid, denom, torch.ones_like(denom))
        xd, yd = self._distort(n[..., 0] / denom, n[..., 1] / denom)
        u, v = self.gamma1 * xd + self.u0, self.gamma2 * yd + self.v0
        valid &= (u >= 0) & (u <= self.width - 1) & (v >= 0) & (v <= self.height - 1)
        return u, v, valid

    def _distort_jacobian(self, x, y):
        r2 = x * x + y * y
        radial = 1 + self.k1 * r2 + self.k2 * r2 * r2
        g = 2 * (self.k1 + 2 * self.k2 * r2)
        a = radial + g * x * x + 2 * self.p1 * y + 6 * self.p2 * x
        b = g * x * y + 2 * self.p1 * x + 2 * self.p2 * y
        d = radial + g * y * y + 6 * self.p1 * y + 2 * self.p2 * x
        return a, b, b, d

    def unproject(self, u, v, iters=30):
        """Pixels -> unit rays (..., 3); distortion inverted with damped Newton iterations."""
        xd, yd = (u - self.u0) / self.gamma1, (v - self.v0) / self.gamma2
        x, y = xd.clone(), yd.clone()
        for _ in range(iters):
            ex, ey = self._distort(x, y)
            rx, ry = ex - xd, ey - yd
            a, b, c, d = self._distort_jacobian(x, y)
            det = a * d - b * c
            det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
            dx, dy = (d * rx - b * ry) / det, (-c * rx + a * ry) / det
            x, y = x - dx.clamp(-0.5, 0.5), y - dy.clamp(-0.5, 0.5)
        r2 = x * x + y * y
        factor = (self.xi + torch.sqrt((1 + (1 - self.xi ** 2) * r2).clamp_min(0))) / (r2 + 1)
        rays = torch.stack((factor * x, factor * y, factor - self.xi), dim=-1)
        return rays / rays.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------------------------
# Calibration / pose files
# ---------------------------------------------------------------------------------------------

def _rigid(values):
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :4] = torch.tensor(values, dtype=torch.float64).view(3, 4)
    return T


def read_cam_to_pose(path):
    """{'image_00': 4x4, ...}: camera -> vehicle pose frame."""
    out = {}
    with open(path) as f:
        for line in f:
            if ':' in line:
                name, vals = line.split(':', 1)
                out[name.strip()] = _rigid([float(x) for x in vals.split()])
    return out


def read_cam_to_velo(path):
    with open(path) as f:
        return _rigid([float(x) for x in f.read().split()][:12])


def read_poses(path):
    """{frame: 4x4 vehicle -> world}."""
    poses = {}
    with open(path) as f:
        for line in f:
            vals = line.split()
            if len(vals) >= 13:
                poses[int(vals[0])] = _rigid([float(x) for x in vals[1:13]])
    return poses


# ---------------------------------------------------------------------------------------------
# Panorama synthesis and LiDAR ground truth
# ---------------------------------------------------------------------------------------------

def fisheyes_to_equirect(images, cameras, cam_to_rig, height, width, feather=0.15):
    """Resample fisheye images into one ERP panorama in the rig frame (ours: x right, y fwd, z up).

    images: list of (3, h, w) tensors; cameras: list of MEICamera; cam_to_rig: list of 4x4
    (camera -> rig, rig in *our* axis convention). Overlaps are blended with weights that fall off
    towards the fisheye image border. Returns (panorama (3, H, W), coverage mask (H, W)).
    """
    rays = equirect_rays(height, width, dtype=torch.float64)
    acc = torch.zeros(3, height, width, dtype=torch.float64)
    wsum = torch.zeros(height, width, dtype=torch.float64)
    for image, cam, T in zip(images, cameras, cam_to_rig):
        R = T[:3, :3]
        local = rays @ R  # rig -> camera directions: R^T d
        u, v, valid = cam.project(local)
        cx, cy = (cam.width - 1) / 2, (cam.height - 1) / 2
        radius = torch.sqrt(((u - cx) / cx) ** 2 + ((v - cy) / cy) ** 2)
        weight = ((1 - radius) / feather).clamp(0, 1) * valid
        grid = torch.stack((2 * u / (cam.width - 1) - 1, 2 * v / (cam.height - 1) - 1), dim=-1)
        sampled = torch.nn.functional.grid_sample(image[None].double(), grid[None], mode='bilinear',
                                                  padding_mode='border', align_corners=True)[0]
        acc += sampled * weight
        wsum += weight
    pano = (acc / wsum.clamp_min(1e-9)).float()
    return pano, (wsum > 0)


def points_to_equirect_depth(points, height, width, R=None, C=None):
    """Z-buffer project 3D points (N, 3) into an ERP depth map (inf where empty)."""
    R = torch.eye(3, dtype=points.dtype) if R is None else R.to(points.dtype)
    C = torch.zeros(3, dtype=points.dtype) if C is None else C.to(points.dtype)
    theta, phi, dist = project_points(points, R, C)
    u, v = angles_to_pixels(theta, phi, height, width)
    idx = v.round().long().clamp(0, height - 1) * width + (u.round().long() % width)
    depth = torch.full((height * width,), math.inf, dtype=points.dtype)
    depth.scatter_reduce_(0, idx, dist, reduce='amin')
    return depth.view(height, width)


def read_velodyne(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64))


class KITTI360Sequence:
    """Minimal reader for one KITTI-360 sequence producing panoramas, poses and LiDAR depth."""

    def __init__(self, root, sequence, height=512, width=1024):
        self.root, self.seq, self.height, self.width = root, sequence, height, width
        cal = os.path.join(root, 'calibration')
        self.cameras = [MEICamera.from_yaml(os.path.join(cal, f'image_0{i}.yaml')) for i in (2, 3)]
        cam_to_pose = read_cam_to_pose(os.path.join(cal, 'calib_cam_to_pose.txt'))
        to_ours = torch.eye(4, dtype=torch.float64)
        to_ours[:3, :3] = VEHICLE_TO_OURS.double()
        self.cam_to_rig = [to_ours @ cam_to_pose[f'image_0{i}'] for i in (2, 3)]
        self.velo_to_rig = to_ours @ cam_to_pose['image_00'] @ torch.linalg.inv(
            read_cam_to_velo(os.path.join(cal, 'calib_cam_to_velo.txt')))
        self.poses = read_poses(os.path.join(root, 'data_poses', sequence, 'poses.txt'))
        self.to_ours = to_ours

    def frames(self):
        return sorted(self.poses)

    def rig_pose(self, frame):
        """(R, C) of the rig (our axes) in the world frame."""
        T = self.poses[frame] @ torch.linalg.inv(self.to_ours)
        return T[:3, :3].float(), T[:3, 3].float()

    def panorama(self, frame):
        from PIL import Image
        imgs = []
        for i in (2, 3):
            p = os.path.join(self.root, 'data_2d_raw', self.seq, f'image_0{i}', 'data_rgb', f'{frame:010d}.png')
            imgs.append(torch.from_numpy(np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255).permute(2, 0, 1))
        return fisheyes_to_equirect(imgs, self.cameras, self.cam_to_rig, self.height, self.width)

    def lidar_depth(self, frame, accumulate=()):
        """ERP depth from the frame's scan plus optional neighbouring scans (frame offsets)."""
        R, C = self.rig_pose(frame)
        pts = []
        for f in (frame, *[frame + d for d in accumulate]):
            path = os.path.join(self.root, 'data_3d_raw', self.seq, 'velodyne_points', 'data', f'{f:010d}.bin')
            if f not in self.poses or not os.path.exists(path):
                continue
            p = read_velodyne(path)
            rig = p @ self.velo_to_rig[:3, :3].T + self.velo_to_rig[:3, 3]
            Rf, Cf = self.rig_pose(f)
            pts.append(rig @ Rf.double().T + Cf.double())
        return points_to_equirect_depth(torch.cat(pts), self.height, self.width, R.double(), C.double()).float()
