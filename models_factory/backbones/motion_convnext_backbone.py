"""A compact ConvNeXt-style encoder with local cross-frame motion fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES


class _GRN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return x + self.gamma * (x * nx) + self.beta


class _ConvNeXtBlock(nn.Module):
    def __init__(self, channels, expansion=4):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        hidden = channels * expansion
        self.pw1 = nn.Linear(channels, hidden)
        self.grn = _GRN(hidden)
        self.pw2 = nn.Linear(hidden, channels)
        self.scale = nn.Parameter(1e-6 * torch.ones(channels))

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw1(x)
        x = F.gelu(x)
        x = self.grn(x)
        x = self.pw2(x) * self.scale
        return residual + x.permute(0, 3, 1, 2)


class _LocalMotionFusion(nn.Module):
    """Feature correlation over nearby time offsets, with a residual motion gate."""

    def __init__(self, channels, radius):
        super().__init__()
        self.radius = radius
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x is [B, T, C, H, W]. Invalid temporal neighbors are masked, not wrapped.
        b, t, c, h, w = x.shape
        q = self.query(x.reshape(b * t, c, h, w)).reshape(b, t, c, h, w)
        k = self.key(x.reshape(b * t, c, h, w)).reshape(b, t, c, h, w)
        v = self.value(x.reshape(b * t, c, h, w)).reshape(b, t, c, h, w)
        # Cosine-style local correlation keeps the attention scale stable as
        # feature norms change across training stages.
        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)
        neighbors, valid = [], []
        for offset in range(-self.radius, self.radius + 1):
            indices = [min(max(i + offset, 0), t - 1) for i in range(t)]
            neighbors.append(k[:, indices])
            valid.append([0 if i + offset < 0 or i + offset >= t else 1 for i in range(t)])
        keys = torch.stack(neighbors, dim=2)  # [B,T,K,C,H,W]
        values = torch.stack([v[:, [min(max(i + o, 0), t - 1) for i in range(t)]]
                              for o in range(-self.radius, self.radius + 1)], dim=2)
        scores = (q.unsqueeze(2) * keys).mean(dim=3) / self.temperature.clamp_min(0.1)
        mask = torch.tensor(valid, device=x.device, dtype=torch.bool).t()[None, :, :, None, None]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=2).unsqueeze(3)
        aligned = (weights * values).sum(dim=2)
        motion = (x - aligned).abs()
        return x + self.gate(motion.reshape(b * t, c, h, w)).reshape(b, t, c, h, w) * aligned


@BACKBONES.register_module
class MotionConvNeXtBackbone(nn.Module):
    """Shared per-frame ConvNeXt encoder; returns half/quarter/eighth features."""

    def __init__(self, num_frames=3, in_channels=3, dims=(48, 96, 192), depths=(2, 2, 2)):
        super().__init__()
        if num_frames not in (3, 5):
            raise ValueError("MotionConvNeXtBackbone supports num_frames=3 or 5")
        self.num_frames = num_frames
        self.dims = tuple(dims)
        self.full_proj = nn.Sequential(nn.Conv2d(in_channels, dims[0], 3, padding=1), nn.GELU())
        self.stem = nn.Sequential(nn.Conv2d(in_channels, dims[0], 4, stride=2, padding=1), nn.GELU())
        self.stage1 = nn.Sequential(*[_ConvNeXtBlock(dims[0]) for _ in range(depths[0])])
        self.down2 = nn.Sequential(nn.GroupNorm(1, dims[0]), nn.Conv2d(dims[0], dims[1], 2, stride=2))
        self.stage2 = nn.Sequential(*[_ConvNeXtBlock(dims[1]) for _ in range(depths[1])])
        self.motion_quarter = _LocalMotionFusion(dims[1], radius=2)
        self.down3 = nn.Sequential(nn.GroupNorm(1, dims[1]), nn.Conv2d(dims[1], dims[2], 2, stride=2))
        self.stage3 = nn.Sequential(*[_ConvNeXtBlock(dims[2]) for _ in range(depths[2])])
        self.motion_eighth = _LocalMotionFusion(dims[2], radius=4)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.num_frames * 3:
            raise ValueError(f"expected [B,{self.num_frames * 3},H,W], got {tuple(x.shape)}")
        b, _, h, w = x.shape
        x = x.reshape(b, self.num_frames, 3, h, w)
        full = self.full_proj(x.reshape(b * self.num_frames, 3, h, w))
        full = full.reshape(b, self.num_frames, self.dims[0], h, w)
        y = self.stem(x.reshape(b * self.num_frames, 3, h, w))
        h2, w2 = y.shape[-2:]
        y = self.stage1(y).reshape(b, self.num_frames, self.dims[0], h2, w2)
        half = y
        y = self.down2(y.reshape(b * self.num_frames, self.dims[0], h2, w2))
        h4, w4 = y.shape[-2:]
        y = self.stage2(y).reshape(b, self.num_frames, self.dims[1], h4, w4)
        quarter = self.motion_quarter(y)
        y = self.down3(quarter.reshape(b * self.num_frames, self.dims[1], h4, w4))
        h8, w8 = y.shape[-2:]
        y = self.stage3(y).reshape(b, self.num_frames, self.dims[2], h8, w8)
        eighth = self.motion_eighth(y)
        return {"full": full, "half": half, "quarter": quarter, "eighth": eighth}
