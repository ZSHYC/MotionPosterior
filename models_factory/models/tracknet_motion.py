import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import MODELS, build_backbone


@MODELS.register_module
class TrackNetMotion(nn.Module):
    """Multi-frame motion-aware detector (3 or 5 RGB frames).

    ``return_aux=False`` keeps the original heatmap tensor contract for old
    callers.  The richer output mode exposes centre offsets, visibility and
    uncertainty so the upgraded loss can train the tracking geometry directly.
    """

    def __init__(self, backbone=None, num_frames=3, return_aux=False):
        super().__init__()
        if num_frames not in (3, 5):
            raise ValueError("TrackNetMotion supports num_frames=3 or 5")
        backbone = dict(backbone or {"type": "MotionConvNeXtBackbone"})
        if "num_frames" in backbone and backbone["num_frames"] != num_frames:
            raise ValueError(
                f"model num_frames={num_frames} conflicts with backbone "
                f"num_frames={backbone['num_frames']}"
            )
        backbone.setdefault("num_frames", num_frames)
        self.num_frames = num_frames
        self.return_aux = bool(return_aux)
        self.backbone = build_backbone(backbone)
        backbone_frames = getattr(self.backbone, "num_frames", num_frames)
        if backbone_frames != num_frames:
            raise ValueError(
                f"built backbone num_frames={backbone_frames} conflicts with model num_frames={num_frames}"
            )
        d0, d1, d2 = self.backbone.dims
        self.up_quarter = nn.Conv2d(d2 + d1, d1, 3, padding=1)
        self.up_half = nn.Conv2d(d1 + d0, d0, 3, padding=1)
        self.up_full = nn.Conv2d(d0 + d0, d0, 3, padding=1)
        self.head = nn.Sequential(nn.Conv2d(d0, d0 // 2, 3, padding=1), nn.GELU(), nn.Conv2d(d0 // 2, 1, 1))
        self.offset_head = nn.Sequential(
            nn.Linear(d0 * 2, d0 // 2), nn.GELU(), nn.Linear(d0 // 2, 2), nn.Tanh()
        )
        self.visibility_head = nn.Linear(d0 * 2, 1)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d0 * 2, d0 // 2), nn.GELU(), nn.Linear(d0 // 2, 1)
        )

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
        heat_logits = self.head(y).reshape(b, t, *x.shape[-2:]).squeeze(2)
        height, width = heat_logits.shape[-2:]
        spatial_weights = torch.softmax(heat_logits.flatten(2) * 10.0, dim=-1)
        xs = torch.linspace(0, 1, width, device=x.device, dtype=heat_logits.dtype)
        ys = torch.linspace(0, 1, height, device=x.device, dtype=heat_logits.dtype)
        coarse_x = (spatial_weights.reshape(b, t, height, width) * xs).sum(dim=(-2, -1))
        coarse_y = (spatial_weights.reshape(b, t, height, width) * ys[:, None]).sum(dim=(-2, -1))
        coarse_centres = torch.stack((coarse_x, coarse_y), dim=-1)
        frame_features = y.reshape(b, t, y.shape[1], height * width)
        local_context = torch.matmul(
            frame_features, spatial_weights.unsqueeze(-1)
        ).squeeze(-1)
        global_context = frame_features.mean(dim=-1)
        pooled = torch.cat((local_context, global_context), dim=-1)
        offsets = (coarse_centres + 0.1 * self.offset_head(pooled)).clamp(0.0, 1.0)
        visibility_logits = self.visibility_head(pooled).squeeze(-1)
        log_variance = self.uncertainty_head(pooled).squeeze(-1).clamp(-8.0, 4.0)
        uncertainty = torch.exp(0.5 * log_variance)
        velocity = offsets[:, 1:] - offsets[:, :-1]
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        heatmap = torch.sigmoid(heat_logits)
        if not self.return_aux:
            return heatmap
        return {
            "heatmap": heatmap,
            "heat_logits": heat_logits,
            "offset": offsets,
            "visibility_logits": visibility_logits,
            "uncertainty": uncertainty,
            "log_variance": log_variance,
            "velocity": velocity,
            "acceleration": acceleration,
        }

    def forward_heatmap(self, x):
        """Return only heatmaps for deployment code using the rich model."""
        output = self.forward(x)
        return output["heatmap"] if isinstance(output, dict) else output
