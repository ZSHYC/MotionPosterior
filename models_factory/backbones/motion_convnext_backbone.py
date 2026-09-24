"""ConvNeXt encoder with global registration and local motion correlation.

The motion block is deliberately self contained: it estimates a bounded global
translation, searches a 3x3 spatial neighbourhood on the compensated feature,
and predicts a dense residual offset for a final refinement.  The local search
radius is conditioned on an inexpensive velocity/uncertainty state predicted
from each temporal pair, so fast or ambiguous motion can use a wider search
without paying for a dense full-frame cost volume.
"""

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


class _SpatialTemporalMotionFusion(nn.Module):
    """Register neighbouring frames and aggregate local spatio-temporal evidence."""

    def __init__(self, channels, radius, correlation_channels=32, max_global_shift=0.15):
        super().__init__()
        self.radius = radius
        self.correlation_channels = min(correlation_channels, channels)
        self.max_global_shift = max_global_shift
        self.query = nn.Conv2d(channels, self.correlation_channels, 1, bias=False)
        self.key = nn.Conv2d(channels, self.correlation_channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        hidden = max(16, channels // 4)
        self.global_shift = nn.Sequential(
            nn.Linear(channels * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
            nn.Tanh(),
        )
        # A pair-level state controls the local search radius.  The first two
        # values are a bounded velocity proxy and the last is uncertainty.  This
        # is intentionally feature-only: it keeps the backbone independent from
        # the task head while making the search motion aware.
        self.motion_state = nn.Sequential(
            nn.Linear(channels * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.offset = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 2, 3, padding=1),
            nn.Tanh(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        # A strict zero made every alignment path receive zero gradient at
        # initialization (the output was x + 0 * aligned).  A small positive
        # residual keeps the block near-identity while allowing q/k/value and
        # offset branches to learn from the first step.
        self.motion_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.global_shift[2].weight)
        nn.init.zeros_(self.global_shift[2].bias)
        # Keep the initial offset tiny without severing gradients through the
        # preceding offset features (a zero final convolution would do that).
        nn.init.normal_(self.offset[2].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.offset[2].bias)

    @staticmethod
    def _grid(height, width, device, dtype):
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((x, y), dim=-1).unsqueeze(0)

    def forward(self, x):
        # x is [B, T, C, H, W]. Invalid temporal neighbours are masked, not wrapped.
        b, t, c, h, w = x.shape
        x_flat = x.reshape(b * t, c, h, w)
        q = F.normalize(self.query(x_flat), dim=1)
        base_grid = self._grid(h, w, x.device, x.dtype)
        temporal_candidates, temporal_scores, valid = [], [], []
        center_desc = x.mean(dim=(-1, -2))
        for offset in range(-self.radius, self.radius + 1):
            if offset == 0:
                continue
            indices = torch.tensor(
                [min(max(i + offset, 0), t - 1) for i in range(t)],
                device=x.device,
                dtype=torch.long,
            )
            neighbour = x[:, indices]
            neighbour_desc = neighbour.mean(dim=(-1, -2))
            pair_desc = torch.cat(
                (center_desc, neighbour_desc, (center_desc - neighbour_desc).abs()), dim=-1
            )
            shift = self.global_shift(pair_desc.reshape(b * t, -1))
            shift = shift * self.max_global_shift
            motion_state = self.motion_state(pair_desc.reshape(b * t, -1))
            velocity_proxy = torch.tanh(motion_state[:, :2]).norm(dim=1)
            uncertainty = torch.sigmoid(motion_state[:, 2])
            # Radius is measured in the current feature map's pixel units.  The
            # base 3x3 neighbourhood remains available, while fast/uncertain
            # pairs expand it continuously (and differentiably) up to 2.5x.
            radius_scale = (1.0 + 0.75 * velocity_proxy + 0.75 * uncertainty).clamp(1.0, 2.5)
            grid = base_grid.expand(b * t, -1, -1, -1) + shift[:, None, None, :]
            warped = F.grid_sample(
                neighbour.reshape(b * t, c, h, w), grid,
                mode="bilinear", padding_mode="border", align_corners=True,
            )
            key = F.normalize(self.key(warped), dim=1)
            value = self.value(warped)

            # Sample keys at the expanded radius before scoring.  Scaling only
            # the expected displacement after a fixed 3x3 correlation would
            # still miss fast balls outside that original neighbourhood.
            displacements = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
            local_scores = []
            for dx, dy in displacements:
                sampling_grid = base_grid + torch.stack((
                    radius_scale * dx * (2.0 / max(w - 1, 1)),
                    radius_scale * dy * (2.0 / max(h - 1, 1)),
                ), dim=-1)[:, None, None, :]
                sampled_key = F.grid_sample(
                    key, sampling_grid, mode="bilinear",
                    padding_mode="border", align_corners=True,
                )
                local_scores.append((q * sampled_key).sum(dim=1))
            local_scores = torch.stack(local_scores, dim=1)
            local_weights = torch.softmax(
                local_scores / self.temperature.clamp_min(0.1), dim=1
            )
            dx = sum(local_weights[:, i] * disp[0] for i, disp in enumerate(displacements))
            dy = sum(local_weights[:, i] * disp[1] for i, disp in enumerate(displacements))
            dx = dx * radius_scale[:, None, None]
            dy = dy * radius_scale[:, None, None]
            local_grid = base_grid.expand(b * t, -1, -1, -1) + torch.stack(
                (dx * (2.0 / max(w - 1, 1)), dy * (2.0 / max(h - 1, 1))), dim=-1
            )
            local_aligned = F.grid_sample(
                value, local_grid, mode="bilinear", padding_mode="border", align_corners=True
            )
            temporal_candidates.append(local_aligned)
            temporal_scores.append((q * F.normalize(self.key(local_aligned), dim=1)).mean(dim=1))
            valid.append([0 if i + offset < 0 or i + offset >= t else 1 for i in range(t)])

        candidates = torch.stack(temporal_candidates, dim=1)  # [BT,K,C,H,W]
        scores = torch.stack(temporal_scores, dim=1)  # [BT,K,H,W]
        mask = torch.tensor(valid, device=x.device, dtype=torch.bool).t().reshape(1, t, -1, 1, 1)
        mask = mask.expand(b, -1, -1, -1, -1).reshape(b * t, -1, 1, 1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        temporal_weights = torch.softmax(scores, dim=1).unsqueeze(2)
        aligned = (temporal_weights * candidates).sum(dim=1)

        # Dense residual offset acts as a lightweight deformable alignment step.
        motion = x_flat - aligned
        residual_offset = self.offset(torch.cat((x_flat, aligned, motion), dim=1)) * 0.15
        refined_grid = base_grid.expand(b * t, -1, -1, -1) + residual_offset.permute(0, 2, 3, 1)
        aligned = F.grid_sample(
            aligned, refined_grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        gate = self.gate(torch.cat((x_flat, motion), dim=1))
        return (x_flat + self.motion_scale.clamp_min(0.01) * gate * aligned).reshape(b, t, c, h, w)


# Keep the old private name available for code importing it from early versions.
_LocalMotionFusion = _SpatialTemporalMotionFusion


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
        self.motion_half = _SpatialTemporalMotionFusion(dims[0], radius=1, correlation_channels=8)
        self.motion_quarter = _SpatialTemporalMotionFusion(dims[1], radius=2, correlation_channels=16)
        self.down3 = nn.Sequential(nn.GroupNorm(1, dims[1]), nn.Conv2d(dims[1], dims[2], 2, stride=2))
        self.stage3 = nn.Sequential(*[_ConvNeXtBlock(dims[2]) for _ in range(depths[2])])
        self.motion_eighth = _SpatialTemporalMotionFusion(dims[2], radius=4, correlation_channels=24)

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
        y = self.motion_half(y)
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
