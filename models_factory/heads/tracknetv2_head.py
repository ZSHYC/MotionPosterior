import torch
import torch.nn as nn
from ..builder import HEADS

class HeadConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # 2D卷积层
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1, 
                bias=True
            ),
            nn.Sigmoid()  # 变为sigmoid用于wbce损失
        )

    def forward(self, x):
        return self.conv(x)

@HEADS.register_module
class TrackNetV2Head(nn.Module):
    """
    它通过一个1x1卷积将输入特征图的通道数映射到任务所需的类别数。
    """

    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()
        self.head = HeadConvBlock(in_channels, out_channels)

    def forward(self, x):
        """
        输入: 来自Neck的精炼特征图, 形状为 [B, in_channels, H, W]
        输出: Logits, 形状为 [B, out_channels, H, W]
        """
        return self.head(x)
