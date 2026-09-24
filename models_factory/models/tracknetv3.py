"""PyTorch TrackNetV3 baseline from the released TrackNetV3 structure.

The official V3 repository uses this heatmap network together with a separate
1-D InpaintNet trajectory rectifier. Both modules are registered here so the
tracking baseline and its rectification stage can be built independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import MODELS


class _Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _Double2DConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            _Conv2DBlock(in_channels, out_channels),
            _Conv2DBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class _Triple2DConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            _Conv2DBlock(in_channels, out_channels),
            _Conv2DBlock(out_channels, out_channels),
            _Conv2DBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


@MODELS.register_module
class TrackNetV3(nn.Module):
    """Official V3 heatmap tracker: three-frame U-Net style encoder/decoder."""

    def __init__(self, in_dim=9, out_dim=3, num_frames=3):
        super().__init__()
        if num_frames != 3 or in_dim != 9 or out_dim != 3:
            raise ValueError("TrackNetV3 baseline requires 3 RGB frames and 3 heatmaps")
        self.num_frames = num_frames
        self.down_block_1 = _Double2DConv(in_dim, 64)
        self.down_block_2 = _Double2DConv(64, 128)
        self.down_block_3 = _Triple2DConv(128, 256)
        self.bottleneck = _Triple2DConv(256, 512)
        self.up_block_1 = _Triple2DConv(768, 256)
        self.up_block_2 = _Double2DConv(384, 128)
        self.up_block_3 = _Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, 1)

    @staticmethod
    def _up(x, target):
        return F.interpolate(x, size=target.shape[-2:], mode="nearest")

    def forward(self, x):
        x1 = self.down_block_1(x)
        x2 = self.down_block_2(F.max_pool2d(x1, 2))
        x3 = self.down_block_3(F.max_pool2d(x2, 2))
        x = self.bottleneck(F.max_pool2d(x3, 2))
        x = self.up_block_1(torch.cat([self._up(x, x3), x3], dim=1))
        x = self.up_block_2(torch.cat([self._up(x, x2), x2], dim=1))
        x = self.up_block_3(torch.cat([self._up(x, x1), x1], dim=1))
        return torch.sigmoid(self.predictor(x))


class _Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


@MODELS.register_module
class InpaintNetV3(nn.Module):
    """Official V3 trajectory rectifier for normalized coordinates and masks."""

    def __init__(self):
        super().__init__()
        self.down_1 = _Conv1DBlock(3, 32)
        self.down_2 = _Conv1DBlock(32, 64)
        self.down_3 = _Conv1DBlock(64, 128)
        self.bottleneck = nn.Sequential(_Conv1DBlock(128, 256), _Conv1DBlock(256, 256))
        self.up_1 = _Conv1DBlock(384, 128)
        self.up_2 = _Conv1DBlock(192, 64)
        self.up_3 = _Conv1DBlock(96, 32)
        self.predictor = nn.Conv1d(32, 2, 3, padding=1)

    def forward(self, coordinates, inpaint_mask):
        x = torch.cat([coordinates, inpaint_mask], dim=2).permute(0, 2, 1)
        x1 = self.down_1(x)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x = self.bottleneck(x3)
        x = self.up_1(torch.cat([x, x3], dim=1))
        x = self.up_2(torch.cat([x, x2], dim=1))
        x = self.up_3(torch.cat([x, x1], dim=1))
        return torch.sigmoid(self.predictor(x)).permute(0, 2, 1)
