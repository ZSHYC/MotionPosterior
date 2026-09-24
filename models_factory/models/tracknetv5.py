import torch.nn as nn
import torch
from ..builder import MODELS, build_backbone, build_neck, build_head
from ..basic import MDD

@MODELS.register_module
class TrackNetV5(nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.num_frames = 3
        self.mdd = MDD()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

    def forward(self, x):
        mvdr_attention = self.mdd(x)

        F1 = x[:, 0:3]
        F2 = x[:, 3:6]
        F3 = x[:, 6:9]
        att_12 = mvdr_attention[:, 0:2]
        att_23 = mvdr_attention[:, 2:4]

        x = torch.cat([
            F1,
            att_12,
            F2,
            att_23,
            F3
        ], dim=1)

        features = self.backbone(x, motion_maps=mvdr_attention)
        refined_features = self.neck(features)
        logits = self.head(refined_features, mvdr_attention)
        return logits
