import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import NECKS
from ..basic import BasicConvBlock as ConvBlock


@NECKS.register_module
class TrackNetV4Neck(nn.Module):
    """Official V4 three-stage decoder with skip connections."""

    def __init__(self):
        super().__init__()
        self.conv11 = ConvBlock(512 + 256, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.conv14 = ConvBlock(256 + 128, 128)
        self.conv15 = ConvBlock(128, 128)
        self.conv16 = ConvBlock(128 + 64, 64)
        self.conv17 = ConvBlock(64, 64)

    @staticmethod
    def _up(x, target):
        return F.interpolate(x, size=target.shape[-2:], mode="nearest")

    def forward(self, features):
        skip1, skip2, skip3 = features["skip1"], features["skip2"], features["skip3"]
        x = self._up(features["bottleneck"], skip3)
        x = self.conv13(self.conv12(self.conv11(torch.cat([x, skip3], dim=1))))
        x = self._up(x, skip2)
        x = self.conv15(self.conv14(torch.cat([x, skip2], dim=1)))
        x = self._up(x, skip1)
        return self.conv17(self.conv16(torch.cat([x, skip1], dim=1)))
