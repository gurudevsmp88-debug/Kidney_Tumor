"""Posed panorama collections on disk (e.g. a city capture).

Layout:
  <scene>/poses.json
  <scene>/<images...>
poses.json:
  {"panoramas": [
     {"name": "000", "image": "pano_000.jpg",
      "R": [[...], [...], [...]],      # camera-to-world rotation (camera frame: x right, y forward, z up)
      "C": [x, y, z],                  # camera centre in metres (local metric frame, e.g. ENU from GPS/SfM)
      "prior_depth": "prior_000.npy",  # optional: metric depth prior (H, W) from a panoramic depth model
      "prior_sigma": 0.06,             # optional: relative std of that prior
      "depth": "lidar_000.npy"}        # optional: ground-truth / teacher depth (H, W), inf or 0 = none
  ]}
"""
import json
import math
import os

import numpy as np
import torch
from PIL import Image


def _resize_depth(depth, height, width):
    if depth.shape == (height, width):
        return depth
    finite = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
    out = torch.nn.functional.interpolate(finite[None, None], size=(height, width), mode='nearest')[0, 0]
    return torch.where(out > 0, out, torch.full_like(out, math.inf))


class PanoramaCollection:
    def __init__(self, root, height=None, width=None):
        self.root = root
        with open(os.path.join(root, 'poses.json'), 'r') as f:
            self.entries = json.load(f)['panoramas']
        self.height, self.width = height, width

    def __len__(self):
        return len(self.entries)

    def pose(self, i):
        e = self.entries[i]
        return torch.tensor(e['R'], dtype=torch.float32), torch.tensor(e['C'], dtype=torch.float32)

    def image(self, i):
        img = Image.open(os.path.join(self.root, self.entries[i]['image'])).convert('RGB')
        if self.height and self.width:
            img = img.resize((self.width, self.height), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()

    def _depth(self, i, key):
        name = self.entries[i].get(key)
        if not name:
            return None
        d = torch.from_numpy(np.load(os.path.join(self.root, name)).astype(np.float32))
        d = torch.where((d > 0) & torch.isfinite(d), d, torch.full_like(d, math.inf))
        if self.height and self.width:
            d = _resize_depth(d, self.height, self.width)
        return d

    def prior(self, i, default_sigma=0.06):
        """(mu_log, sigma_log) or None."""
        d = self._depth(i, 'prior_depth')
        if d is None:
            return None
        d = torch.where(torch.isfinite(d), d, torch.full_like(d, 1e3))
        return d.log(), torch.full_like(d, float(self.entries[i].get('prior_sigma', default_sigma)))

    def depth(self, i):
        return self._depth(i, 'depth')

    def nearest_partner(self, i, min_baseline=1.0, max_baseline=15.0):
        """Index of the closest other panorama within the baseline range, or None."""
        c = self.pose(i)[1]
        best, best_d = None, math.inf
        for j in range(len(self)):
            if j == i:
                continue
            d = (self.pose(j)[1] - c).norm().item()
            if min_baseline <= d <= max_baseline and d < best_d:
                best, best_d = j, d
        return best


def write_ply(path, points, colors=None):
    """Binary little-endian PLY of finite points (N, 3) with optional colours in [0, 1]."""
    points = points.reshape(-1, 3)
    keep = torch.isfinite(points).all(1)
    xyz = points[keep].numpy().astype('<f4')
    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    if colors is not None:
        fields += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    rec = np.empty(len(xyz), dtype=fields)
    rec['x'], rec['y'], rec['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    header = f'ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n'
    header += 'property float x\nproperty float y\nproperty float z\n'
    if colors is not None:
        rgb = (colors.reshape(-1, 3)[keep].clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        rec['red'], rec['green'], rec['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        header += 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
    header += 'end_header\n'
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(rec.tobytes())
    return len(xyz)
