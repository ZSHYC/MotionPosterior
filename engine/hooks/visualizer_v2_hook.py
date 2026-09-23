import cv2
import torch
import numpy as np
import math
from pathlib import Path

from .base_hook import BaseHook
from ..builder import HOOKS

try:
    from metrics_factory.metrics.tracknetv2_metric import _heatmap_to_coords
except ImportError:
    print("Warning: _heatmap_to_coords not found. Visualization will not show predicted coordinates.")


    def _heatmap_to_coords(heatmap, threshold):
        return None, None


@HOOKS.register_module
class ValidationVisualizerV2Hook(BaseHook):
    """
    一个在验证阶段，用于将模型预测结果（包括最终坐标）可视化的钩子。
    """

    def __init__(self, num_samples_to_save=5, heatmap_threshold=127, original_size=(720, 1280)):
        self.num_samples_to_save = num_samples_to_save
        self.heatmap_threshold = heatmap_threshold
        self.original_h, self.original_w = original_size  # 存储原始视频尺寸 (高, 宽)
        self.vis_count = 0

    def before_val_epoch(self, runner):
        """在每次验证epoch开始前，重置计数器。"""
        self.vis_count = 0
        self.vis_dir = runner.work_dir / f'val_epoch_{runner.epoch + 1}'
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def after_val_iter(self, runner):
        """在每次验证迭代后，检查是否需要进行可视化。"""
        if self.vis_count >= self.num_samples_to_save:
            return

        # 从 runner 中获取数据
        batch = runner.outputs['val_batch']
        logits = runner.outputs['val_logits']

        input_tensor = batch['image'].cpu()
        target_tensor = batch['target'].cpu()
        coords_gt_batch = batch['coords']
        # print(logits.shape)
        pred_tensor = logits.cpu() * 255

        for i in range(input_tensor.size(0)):
            if self.vis_count >= self.num_samples_to_save:
                break

            num_frames = target_tensor.shape[1]
            # --- 准备时序原图（支持三帧/五帧） ---
            input_frames = []
            for frame_idx in range(num_frames):
                # 获取当前帧的RGB通道 (3个通道)
                frame_rgb = input_tensor[i, frame_idx * 3:(frame_idx + 1) * 3, :, :].permute(1, 2, 0).numpy()
                frame_rgb = (frame_rgb * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                input_frames.append(frame_bgr)
            h, w, _ = input_frames[0].shape  # 获取单帧尺寸

            # --- 准备GT热力图 ---
            gt_heatmaps = []
            for frame_idx in range(num_frames):
                gt_heatmap = target_tensor[i, frame_idx].numpy().astype(np.uint8)
                gt_heatmap_color = cv2.applyColorMap(gt_heatmap, cv2.COLORMAP_JET)
                gt_heatmaps.append(gt_heatmap_color)

            # --- 准备预测热力图 ---
            pred_heatmaps = []
            pred_coords = []  # 存储每帧的预测坐标
            for frame_idx in range(num_frames):
                pred_heatmap_np = pred_tensor[i, frame_idx].numpy().astype(np.uint8)
                pred_heatmap_color = cv2.applyColorMap(pred_heatmap_np, cv2.COLORMAP_JET)
                pred_heatmaps.append(pred_heatmap_color)

                # 获取每帧的预测坐标
                x_pred, y_pred = _heatmap_to_coords(pred_heatmap_np, threshold=self.heatmap_threshold)
                pred_coords.append((x_pred, y_pred))

            # --- 准备准确点 ---
            x_gt = []
            y_gt = []
            for frame_idx in range(num_frames):
                x_gt_raw = coords_gt_batch[frame_idx][0][i].item()
                y_gt_raw = coords_gt_batch[frame_idx][1][i].item()
                x_gt.append(x_gt_raw)
                y_gt.append(y_gt_raw)

            # 在所有帧原图上绘制标记
            frames_with_marks = []
            for frame_idx in range(num_frames):
                input_img = input_frames[frame_idx]

                # 绘制绿色的真实标记
                if not math.isnan(x_gt[frame_idx]) and not math.isnan(y_gt[frame_idx]):
                    x_gt_scaled = int(x_gt[frame_idx] * (w / self.original_w))
                    y_gt_scaled = int(y_gt[frame_idx] * (h / self.original_h))
                    cv2.drawMarker(input_img, (x_gt_scaled, y_gt_scaled),
                                   color=(0, 255, 0), markerType=cv2.MARKER_CROSS,
                                   markerSize=15, thickness=2)

                # 绘制红色的预测标记
                x_pred, y_pred = pred_coords[frame_idx]
                if x_pred is not None:
                    cv2.drawMarker(input_img, (int(x_pred), int(y_pred)),
                                   color=(0, 0, 255), markerType=cv2.MARKER_CROSS,
                                   markerSize=15, thickness=2)


            # --- 拼接画布 ---
            canvas = np.zeros((h * num_frames, w * 3, 3), dtype=np.uint8)
            for frame_idx in range(num_frames):
                canvas[frame_idx * h:(frame_idx+1) * h, 0:w] = input_frames[frame_idx]
                canvas[frame_idx * h:(frame_idx+1) * h, w:2 * w] = pred_heatmaps[frame_idx]
                canvas[frame_idx * h:(frame_idx+1) * h, 2 * w:3 * w] = gt_heatmaps[frame_idx]

            cv2.putText(canvas, 'Input (Red:Pred, Green:GT)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                        2)
            cv2.putText(canvas, 'Prediction', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(canvas, 'Ground Truth', (2 * w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 保存图像
            save_path = self.vis_dir / f'sample_{self.vis_count}.jpg'
            cv2.imwrite(str(save_path), canvas)
            self.vis_count += 1
