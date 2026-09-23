import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import MODELS, build_backbone


@MODELS.register_module
class TrackNetMotion(nn.Module):
    """Multi-frame motion-aware heatmap model (3 or 5 RGB frames)."""

    def __init__(self, backbone=None, num_frames=3):
        super().__init__()
        if num_frames not in (3, 5):
            raise ValueError("TrackNetMotion supports num_frames=3 or 5")
        backbone = dict(backbone or {"type": "MotionConvNeXtBackbone"})
        backbone.setdefault("num_frames", num_frames)
        self.num_frames = num_frames
        self.backbone = build_backbone(backbone)
        d0, d1, d2 = self.backbone.dims
        self.up_quarter = nn.Conv2d(d2 + d1, d1, 3, padding=1)
        self.up_half = nn.Conv2d(d1 + d0, d0, 3, padding=1)
        self.up_full = nn.Conv2d(d0 + d0, d0, 3, padding=1)
        self.head = nn.Sequential(nn.Conv2d(d0, d0 // 2, 3, padding=1), nn.GELU(), nn.Conv2d(d0 // 2, 1, 1))

    def forward(self, x):
        features = self.backbone(x)
        b, t = x.shape[0], self.num_frames
        q, h, f, e = features["quarter"], features["half"], features["full"], features["eighth"]
        y = F.interpolate(e.reshape(b * t, e.shape[2], *e.shape[-2:]), size=q.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up_quarter(torch.cat([y, q.reshape(b * t, q.shape[2], *q.shape[-2:])], dim=1))
        y = F.interpolate(y, size=h.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up_half(torch.cat([y, h.reshape(b * t, h.shape[2], *h.shape[-2:])], dim=1))
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up_full(torch.cat([y, f.reshape(b * t, f.shape[2], *f.shape[-2:])], dim=1))
        return torch.sigmoid(self.head(y).reshape(b, t, *x.shape[-2:]).squeeze(2))
