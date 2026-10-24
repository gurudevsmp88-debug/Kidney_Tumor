"""Learned student: prior head + feature stereo + analytic uncertainty + fusion + amodal layer.

This is a deliberately small reference network (CPU-trainable on synthetic
streets). For real data, replace `PanoEncoder`/`prior_head` with a panoramic
depth foundation model and keep the geometric parts (stereo, analytic sigma,
fusion, teacher losses) unchanged: pass its log-depth as `external_prior`.
"""
import math

import torch
import torch.nn as nn

from .fusion import align_prior_scale, fuse_log_depth, relative_to_log_std
from .sphere import circular_pad, equirect_rays, relative_pose
from .stereo import two_view_depth
from .uncertainty import triangulation_depth_std


class EquiConv(nn.Module):
    """3x3 conv with longitude wrap-around padding, + GroupNorm + ReLU."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=0)
        self.norm = nn.GroupNorm(min(8, cout), cout)

    def forward(self, x):
        return torch.relu(self.norm(self.conv(circular_pad(x, 1, 1))))


class PanoEncoder(nn.Module):
    """Small U-Net-ish encoder returning matching features and context features (full resolution)."""

    def __init__(self, feat_dim=16, ctx_dim=32):
        super().__init__()
        self.stem = nn.Sequential(EquiConv(3, feat_dim), EquiConv(feat_dim, feat_dim))
        self.down = nn.Sequential(EquiConv(feat_dim, ctx_dim, stride=2), EquiConv(ctx_dim, ctx_dim),
                                  EquiConv(ctx_dim, ctx_dim, stride=2), EquiConv(ctx_dim, ctx_dim))
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.fuse = EquiConv(ctx_dim + feat_dim + 1, ctx_dim)

    def forward(self, image):
        f = self.stem(image)
        latitude = torch.linspace(1, -1, image.shape[-2], device=image.device, dtype=image.dtype)
        lat = latitude.view(1, 1, -1, 1).expand(image.shape[0], 1, *image.shape[-2:])
        ctx = self.fuse(torch.cat((self.up(self.down(f)), f, lat), dim=1))
        return f, ctx


class PanoLidarNet(nn.Module):
    def __init__(self, feat_dim=16, ctx_dim=32, sigma_px=0.5, stereo_kwargs=None):
        super().__init__()
        self.encoder = PanoEncoder(feat_dim, ctx_dim)
        self.prior_head = nn.Conv2d(ctx_dim, 2, 1)       # log-depth (or residual) and log-sigma
        self.layer2_head = nn.Conv2d(ctx_dim + 2, 2, 1)  # amodal second surface: log-gap and log-sigma
        self.log_sigma_scale = nn.Parameter(torch.zeros(()))  # global calibration of the analytic sigma
        self.sigma_px = sigma_px
        self.stereo_kwargs = stereo_kwargs or {}
        nn.init.zeros_(self.prior_head.weight)
        nn.init.constant_(self.prior_head.bias, 0.0)
        self.prior_head.bias.data[0] = math.log(10.0)
        self.prior_head.bias.data[1] = math.log(0.3)

    def forward(self, image1, image2=None, pose1=None, pose2=None, external_prior=None):
        """image*: (B, 3, H, W); pose*: ((B,3,3), (B,3)); external_prior: optional (B, H, W) log-depth."""
        feat1, ctx1 = self.encoder(image1)
        head = self.prior_head(ctx1)
        mu_p = head[:, 0] if external_prior is None else external_prior + head[:, 0] - self.prior_head.bias[0]
        sigma_p = torch.exp(head[:, 1]).clamp(1e-3, 3.0)
        out = {'prior_mu': mu_p, 'prior_sigma': sigma_p}

        if image2 is not None:
            feat2, _ = self.encoder(image2)
            B, _, H, W = image1.shape
            R12, t12 = relative_pose(pose1[0], pose1[1], pose2[0], pose2[1])
            st = two_view_depth(feat1, feat2, R12, t12, **self.stereo_kwargs)
            rays = equirect_rays(H, W, device=image1.device, dtype=image1.dtype)
            sigma_ang = self.sigma_px * 2 * math.pi / W
            # analytic sigma at the prior depth (see pipeline.reconstruct)
            prior_depth = mu_p.detach().exp()
            std = torch.stack([triangulation_depth_std(rays, prior_depth[b], t12[b], sigma_ang) for b in range(B)])
            sigma_s = (relative_to_log_std(prior_depth, std) * self.log_sigma_scale.exp()).clamp(max=10.0)
            offset = align_prior_scale(mu_p.detach(), st['log_depth'].detach(), sigma_s.detach(), st['confidence'])
            mu_p_aligned = mu_p + offset.view(-1, 1, 1)
            mu, sigma, w = fuse_log_depth(mu_p_aligned, sigma_p, st['log_depth'], sigma_s, st['confidence'])
            out.update(stereo_mu=st['log_depth'], stereo_sigma=sigma_s, confidence=st['confidence'],
                       stereo_weight=w, scale_offset=offset)
        else:
            mu, sigma, _ = fuse_log_depth(mu_p, sigma_p)
        out.update(mu=mu, sigma=sigma)

        l2 = self.layer2_head(torch.cat((ctx1, mu.detach().unsqueeze(1), sigma.detach().unsqueeze(1)), dim=1))
        # second surface is behind the first: log d2 = mu + softplus(gap)
        out['layer2_mu'] = mu.detach() + nn.functional.softplus(l2[:, 0]) + 1e-3
        out['layer2_sigma'] = torch.exp(l2[:, 1]).clamp(1e-3, 3.0)
        return out
