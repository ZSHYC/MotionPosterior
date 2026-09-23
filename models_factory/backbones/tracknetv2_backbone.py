import torch
import torch.nn as nn
from ..builder import BACKBONES

from ..basic import BasicConvBlock as ConvBlock, MotionConditionedGate

@BACKBONES.register_module
class TrackNetV2Backbone(nn.Module):
    def __init__(self, in_channels=9, use_motion_gates=False, motion_channels=4):
        super().__init__()
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
        self.motion_gates = nn.ModuleDict({
            name: MotionConditionedGate(channels, motion_channels)
            for name, channels in (
                ('skip1', 64), ('skip2', 128), ('skip3', 256), ('bottleneck', 512)
            )
        }) if use_motion_gates else None

    def forward(self, x, motion_maps=None):
        features = {}

        # --- Encoder ---
        x = self.conv1(x)
        x = self.conv2(x)
        if self.motion_gates is not None:
            x = self.motion_gates['skip1'](x, motion_maps)
        features['skip1'] = x

        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if self.motion_gates is not None:
            x = self.motion_gates['skip2'](x, motion_maps)
        features['skip2'] = x

        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        if self.motion_gates is not None:
            x = self.motion_gates['skip3'](x, motion_maps)
        features['skip3'] = x

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        if self.motion_gates is not None:
            x = self.motion_gates['bottleneck'](x, motion_maps)
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
    in_channels = 13
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用的设备: {device}")

    # 2. 初始化骨干网络
    model = TrackNetV2Backbone(in_channels=in_channels).to(device)
    model.eval()  # 设置为评估模式
    print(model)

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
