# -*- coding: utf-8 -*-
import torch
import cv2
import numpy as np
import math
from tqdm import tqdm
from .pipeline import BallPoint  # 导入数据包定义

# 导入项目内的构建组件
from models_factory.builder import build_model
from datasets_factory.transforms.tracknet_transforms import Resize, ConcatChannels

# --- 1. “模型配置库” (已硬编码至 Detector 内部) ---
MODEL_CONFIGS = {
    'v2': dict(
        type='TrackNetV2',
        backbone=dict(type='TrackNetV2Backbone', in_channels=9),
        neck=dict(type='TrackNetV2Neck'),
        head=dict(type='TrackNetV2Head', in_channels=64, out_channels=3)
    ),
    'v5': dict(
        type='TrackNetV5',
        backbone=dict(type='TrackNetV2Backbone', in_channels=13),
        neck=dict(type='TrackNetV2Neck'),
        head=dict(type='R_STRHead', in_channels=64, out_channels=3)
    )
}

class TrackNetDetector:
    def __init__(self, arch, weights_path, device='cuda:0', threshold=0.5):
        """
        Stage 1: 检测器
        :param arch: 架构版本 ('v2', 'v5')
        :param weights_path: .pth 权重文件路径
        :param device: 设备 (如 'cuda:0' 或 'cpu')
        :param threshold: 热力图激活阈值
        """
        self.threshold = float(threshold)
        if not math.isfinite(self.threshold) or not 0 <= self.threshold < 1:
            raise ValueError(f"threshold must be a finite value in [0, 1), got {threshold}")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = (288, 512)

        # 1. 从内部配置库获取配置
        model_cfg = MODEL_CONFIGS.get(arch)
        if model_cfg is None:
            raise ValueError(f"Unknown architecture: {arch}. Available: {list(MODEL_CONFIGS.keys())}")

        # 2. 构建并加载模型
        self.model = build_model(model_cfg)
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        self.model.to(self.device).eval()
        print(f"✅ Detector initialized with [{arch.upper()}] model on {self.device}")

        # 3. 初始化变换算子
        self.resizer = Resize(keys=['p', 'c', 'n'], size=self.input_size)
        self.concator = ConcatChannels(keys=['p', 'c', 'n'], output_key='img')

    def detect_video(self, video_path: str) -> list:
        """执行全视频扫描并返回原始 BallPoint 列表"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0 or width * 9 != height * 16:
            cap.release()
            raise ValueError(f"TrackNet requires 16:9 video, got {width}x{height}: {video_path}")

        raw_points = []

        pbar = tqdm(total=total_frames or None, desc="[Stage 1] Neural Inference")

        try:
            while cap.isOpened():
                frames = []
                for _ in range(3):
                    readable, frame = cap.read()
                    if not readable:
                        break
                    frames.append(frame)
                if not frames:
                    break
                actual_frame_count = len(frames)
                while len(frames) < 3:
                    frames.append(frames[-1])

                batch_data = {
                    'p': cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB),
                    'c': cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB),
                    'n': cv2.cvtColor(frames[2], cv2.COLOR_BGR2RGB)
                }
                batch_data = self.concator(self.resizer(batch_data))
                img_tensor = torch.from_numpy(batch_data['img'].transpose(2, 0, 1))
                img_tensor = img_tensor.float().div(255).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    heatmap_preds = self.model(img_tensor).squeeze(0).cpu().numpy()
                if heatmap_preds.shape != (3, *self.input_size):
                    raise ValueError(
                        f"Unexpected model output shape: {heatmap_preds.shape}, "
                        f"expected {(3, *self.input_size)}"
                    )

                for heatmap in heatmap_preds[:actual_frame_count]:
                    point = self._heatmap_to_point(heatmap)
                    if point.is_detected:
                        point.x *= width / self.input_size[1]
                        point.y *= height / self.input_size[0]
                    raw_points.append(point)

                pbar.update(actual_frame_count)
                if actual_frame_count < 3:
                    break
        finally:
            pbar.close()
            cap.release()

        return raw_points

    def _heatmap_to_point(self, heatmap: np.ndarray) -> BallPoint:
        """将热力图矩阵转换为 BallPoint 数据包"""
        # 转为 uint8 以便 OpenCV 处理
        h_uint8 = (heatmap * 255).astype(np.uint8)
        thresh_val = int(self.threshold * 255)
        
        _, binary = cv2.threshold(h_uint8, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return BallPoint(is_detected=False)

        # 获取最大连通域作为目标
        largest_cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_cnt)
        
        if M["m00"] <= 0:
            return BallPoint(is_detected=False)

        # 计算几何中心
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 提取置信度：取连通域内的峰值
        mask = np.zeros(h_uint8.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest_cnt], -1, 255, -1)
        _, max_val, _, _ = cv2.minMaxLoc(h_uint8, mask=mask)
        conf = round(max_val / 255.0, 4)

        return BallPoint(x=cx, y=cy, conf=conf, is_detected=True)
