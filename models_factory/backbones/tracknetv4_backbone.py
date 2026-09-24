import torch
import torch.nn as nn
from ..builder import BACKBONES
from ..basic.conv_block import BasicConvBlock as ConvBlock

import torch
import torch.nn as nn


def power_normalization(input, a, b):
    """幂归一化函数"""
    return 1 / (1 + torch.exp(
        -(5 / (0.45 * torch.abs(torch.tanh(a)) + 1e-6)) * (torch.abs(input) - 0.6 * torch.tanh(b))))


class MotionPromptLayer(nn.Module):

    def __init__(self, penalty_weight=0.0, input_scale=0.225, input_bias=0.45):
        super(MotionPromptLayer, self).__init__()
        # Default configs
        self.input_color_order = "RGB"
        self.color_map = {'R': 0, 'G': 1, 'B': 2}
        self.register_buffer('gray_scale', torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32))

        # Power normalization parameters (trainable scalars)
        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.0))

        # Temporal attention variation regularization parameter
        self.lambda1 = penalty_weight
        self.input_scale = float(input_scale)
        self.input_bias = float(input_bias)

    def forward(self, video_seq):
        # Initialize loss
        loss = torch.tensor(0.0, device=video_seq.device)

        # Input dimension: [b, 9, h, w] -> reshape to [b, 3, 3, h, w] (3 frames, each with 3 RGB channels)
        B, C, H, W = video_seq.shape
        if C != 9:
            raise ValueError(f"Expected input channels=9 (3 frames × 3 RGB channels), but got {C}")

        # Reshape back to video sequence format
        norm_seq = video_seq.view(B, 3, 3, H, W)
        norm_seq = norm_seq * self.input_scale + self.input_bias

        # Transfer to grayscale
        idx_list = [self.color_map[idx] for idx in self.input_color_order]
        gray_scale_tensor = self.gray_scale[idx_list]
        weights = gray_scale_tensor.to(norm_seq.dtype).to(norm_seq.device)

        # Grayscale conversion for each frame: [B, T, C, H, W] -> [B, T, H, W]
        grayscale_video_seq = torch.einsum("btchw,c->bthw", norm_seq, weights)

        # Frame difference - for 3 frames, we get 2 frame differences
        # grayscale_video_seq shape: [B, 3, H, W]
        frame_diff = grayscale_video_seq[:, 1:] - grayscale_video_seq[:, :-1]  # [B, 2, H, W]

        # Power normalization
        attention_map = power_normalization(frame_diff, self.a, self.b)  # [B, 2, H, W]
        norm_attention = attention_map

        # Temporal attention variation regularization
        if self.training:
            temp_diff = norm_attention[:, 1:] - norm_attention[:, :-1]  # [B, 1, 1, H, W]
            temporal_loss = torch.sum(temp_diff ** 2) / (H * W * B)
            loss = self.lambda1 * temporal_loss

        return attention_map, loss

@BACKBONES.register_module
class TrackNetV4Backbone(nn.Module):
    def __init__(self, in_channels=9):
        super().__init__()
        self.motion = MotionPromptLayer()

        # --- Encoder Layers ---
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)

    def forward(self, x):
        features = {}
        attn, _ = self.motion(x)
        features['attention'] = attn

        # --- Encoder ---
        x = self.conv1(x)
        x = self.conv2(x)
        features['skip1'] = x

        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        features['skip2'] = x

        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        features['skip3'] = x

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        features['bottleneck'] = x

        return features


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 1. 定义超参数和设备
    # 假设输入图像尺寸为 256x256
    # 批大小(batch_size)为 2
    # 输入通道数(in_channels)为 9
    batch_size = 2
    input_height = 640
    input_width = 360
    in_channels = 9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用的设备: {device}")

    # 2. 初始化骨干网络
    model = TrackNetV4Backbone(in_channels=in_channels).to(device)
    model.eval()  # 设置为评估模式

    # 3. 创建一个测试张量
    # 张量形状: (batch_size, channels, height, width)
    test_tensor = torch.randn(batch_size, in_channels, input_height, input_width).to(device)
    print(f"输入张量形状: {test_tensor.shape}")

    # 4. 定义预期的输出形状
    # 根据网络结构计算:
    # skip1: H, W 不变
    # skip2: H, W 减半 (256 -> 128)
    # skip3: H, W 再减半 (128 -> 64)
    # bottleneck: H, W 再减半 (64 -> 32)
    expected_shapes = {
        'skip1': (batch_size, 64, input_height, input_width),
        'skip2': (batch_size, 128, input_height // 2, input_width // 2),
        'skip3': (batch_size, 256, input_height // 4, input_width // 4),
        'bottleneck': (batch_size, 512, input_height // 8, input_width // 8)
    }

    print("\n--- 开始测试 ---")

    # 5. 执行前向传播并获取输出
    with torch.no_grad():  # 在测试阶段不计算梯度
        output_features = model(test_tensor)

    # 6. 检查形状是否符合预期
    test_passed = True
    print("检查输出特征图的形状:")
    for name, feature_map in output_features.items():
        actual_shape = feature_map.shape
        expected_shape = expected_shapes[name]

        if actual_shape == expected_shape:
            print(f"  - 特征 '{name}': 形状 {actual_shape} 符合预期。")
        else:
            print(f"  - [失败] 特征 '{name}': 形状 {actual_shape}，但预期为 {expected_shape}。")
            test_passed = False

    # 7. 输出最终测试结果
    print("\n--- 测试结论 ---")
    if test_passed:
        print("✅ 测试通过")
    else:
        print("❌ 测试不通过")
