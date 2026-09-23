import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionConditionedGate(nn.Module):
    """Apply learned spatial and channel gates from signed motion maps."""

    def __init__(self, channels, motion_channels=4):
        super().__init__()
        hidden_channels = max(16, min(channels // 4, 64))
        self.encoder = nn.Sequential(
            nn.Conv2d(motion_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        self.spatial = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.channel = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.gate_scale = nn.Parameter(torch.zeros(1))

    def forward(self, features, motion_maps):
        if motion_maps is None:
            raise ValueError('motion_maps are required by MotionConditionedGate')
        motion_maps = F.interpolate(
            motion_maps, size=features.shape[-2:], mode='bilinear', align_corners=False
        )
        motion_features = self.encoder(motion_maps)
        spatial_gate = torch.sigmoid(self.spatial(motion_features))
        channel_gate = torch.sigmoid(
            self.channel(F.adaptive_avg_pool2d(motion_features, 1))
        )
        gate = spatial_gate * channel_gate
        return features * (1.0 + torch.tanh(self.gate_scale) * gate)
