import torch
import torch.nn as nn

from ..builder import HEADS


@HEADS.register_module
class TrackNetV4Head(nn.Module):
    """Official V4 TypeA/TypeB motion fusion head."""

    def __init__(self, in_channels=64, out_channels=3, fusion_type="A"):
        super().__init__()
        if out_channels != 3 or fusion_type.upper() not in {"A", "B"}:
            raise ValueError("TrackNetV4Head requires out_channels=3 and fusion_type A or B")
        self.predictor = nn.Conv2d(in_channels, out_channels, 1)
        self.fusion_type = fusion_type.upper()

    def forward(self, x, attention):
        if attention.ndim != 4 or attention.shape[1] != 2:
            raise ValueError(f"TrackNetV4 expects two motion attention maps, got {tuple(attention.shape)}")
        feature_map = self.predictor(x)
        if self.fusion_type == "A":
            fused = torch.stack(
                [
                    feature_map[:, 0],
                    feature_map[:, 1] * attention[:, 0],
                    feature_map[:, 2] * attention[:, 1],
                ],
                dim=1,
            )
        else:
            fused = torch.stack(
                [
                    feature_map[:, 0] * attention[:, 0],
                    feature_map[:, 1] * (attention[:, 0] + attention[:, 1]) * 0.5,
                    feature_map[:, 2] * attention[:, 1],
                ],
                dim=1,
            )
        return torch.sigmoid(fused)
