import torch.nn as nn
from ..builder import MODELS, build_backbone, build_neck, build_head

@MODELS.register_module
class TrackNetV4(nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.num_frames = 3
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

    def forward(self, x):
        features = self.backbone(x)
        refined_features = self.neck(features)
        logits = self.head(refined_features, features['attention'])
        return logits
