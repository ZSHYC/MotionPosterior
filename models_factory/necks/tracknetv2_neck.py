import torch
import torch.nn as nn
from ..builder import NECKS

from ..basic import BasicConvBlock as ConvBlock


@NECKS.register_module
class TrackNetV2Neck(nn.Module):
    def __init__(self):
        super().__init__()
        # --- Decoder Layers ---
        self.ups1 = nn.Upsample(scale_factor=2, mode='nearest')
        # 输入通道: bottleneck(512) + skip3(256) = 768
        self.conv11 = ConvBlock(512 + 256, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)

        self.ups2 = nn.Upsample(scale_factor=2, mode='nearest')
        # 输入通道: 上一层输出(256) + skip2(128) = 384
        self.conv14 = ConvBlock(256 + 128, 128)
        self.conv15 = ConvBlock(128, 128)

        self.ups3 = nn.Upsample(scale_factor=2, mode='nearest')
        # 输入通道: 上一层输出(128) + skip1(64) = 192
        self.conv16 = ConvBlock(128 + 64, 64)
        self.conv17 = ConvBlock(64, 64)

    def forward(self, features):
        skip1_feat = features['skip1']
        skip2_feat = features['skip2']
        skip3_feat = features['skip3']
        bottleneck_feat = features['bottleneck']

        # --- Decoder with Skip Connections ---
        x = self.ups1(bottleneck_feat)
        x = torch.cat([x, skip3_feat], dim=1)  # 维度 H, W: 64x64
        x = self.conv13(self.conv12(self.conv11(x)))

        x = self.ups2(x)
        x = torch.cat([x, skip2_feat], dim=1)  # 维度 H, W: 128x128
        x = self.conv15(self.conv14(x))

        x = self.ups3(x)
        x = torch.cat([x, skip1_feat], dim=1)  # 维度 H, W: 256x256
        x = self.conv17(self.conv16(x))

        return x


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 1. 定义超参数和设备
    # 这些参数应该与 Backbone 测试中的参数保持一致
    batch_size = 2
    height = 640
    width = 360
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用的设备: {device}")

    # 2. 初始化 Neck 网络
    model = TrackNetV2Neck().to(device)
    model.eval()

    # 3. 创建模拟的输入特征字典 (Mock Input Features)
    # 这些形状是 Backbone 模块的预期输出
    mock_features = {
        'skip1': torch.randn(batch_size, 64, height, width).to(device),
        'skip2': torch.randn(batch_size, 128, height // 2, width // 2).to(device),
        'skip3': torch.randn(batch_size, 256, height // 4, width // 4).to(device),
        'bottleneck': torch.randn(batch_size, 512, height // 8, width // 8).to(device)
    }

    print("\n--- 模拟输入特征形状 ---")
    for name, tensor in mock_features.items():
        print(f"  - {name}: {tensor.shape}")

    # 4. 定义预期的最终输出形状
    # 经过三次上采样，最终输出的特征图尺寸应与 skip1 相同
    # 输出通道数由最后一个 ConvBlock (conv17) 决定, 为 64
    expected_output_shape = (batch_size, 64, height, width)

    print(f"\n预期的最终输出形状: {expected_output_shape}")
    print("\n--- 开始测试 ---")

    # 5. 执行前向传播
    with torch.no_grad():
        output_tensor = model(mock_features)

    # 6. 检查输出形状
    actual_output_shape = output_tensor.shape
    print(f"实际的最终输出形状: {actual_output_shape}")

    # 7. 输出最终测试结果
    print("\n--- 测试结论 ---")
    if actual_output_shape == expected_output_shape:
        print("✅ 测试通过")
    else:
        print("❌ 测试不通过")
