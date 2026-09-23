import cv2
import numpy as np
from scipy.spatial import distance
import torch
import math
from ..builder import METRICS



def _heatmap_to_coords(heatmap: np.ndarray, threshold: int = 127):
    """
    一个鲁棒的坐标提取函数。
    它对热力图进行二值化，然后寻找最大轮廓的质心作为坐标。
    """
    if heatmap.dtype != np.uint8:
        heatmap = heatmap.astype(np.uint8)

    _, binary_map = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy

    return None, None


@METRICS.register_module
class TrackNetV2Metric:
    """
    一个专为UTrackNetV1设计的、用于计算 F1, Precision, Recall 的计分员。
    它内部封装了从热力图到坐标的转换逻辑。
    """

    def __init__(self, min_dist: int = 10, heatmap_threshold: int = 127, original_size=(720, 1280)):
        self.min_dist = min_dist
        self.heatmap_threshold = heatmap_threshold
        self.original_h, self.original_w = original_size
        self.reset()

    def reset(self):
        """清空计分板。"""
        self.tp, self.fp1, self.fp2, self.fp, self.tn, self.fn = 0, 0, 0, 0, 0, 0
        self.pixel_errors = []

    def update(self, logits: torch.Tensor, batch: dict):
        """根据一个批次的数据，更新计分板。"""
        if isinstance(logits, dict):
            logits = logits["heatmap"]
        predictions = logits.detach().cpu().numpy() * 255
        _, c, h, w = predictions.shape
        scale_h = h / self.original_h
        scale_w = w / self.original_w
        coords_gt = batch['coords']
        visibility_gt = batch['visibility']

        for i in range(len(predictions)):
            for j in range(c):
                # 直接调用本文件内的辅助函数
                x_pred, y_pred = _heatmap_to_coords(predictions[i][j], threshold=self.heatmap_threshold)

                # 当vis==0时，x_gt和y_gt为nan
                x_gt, y_gt = coords_gt[j][0][i].item(), coords_gt[j][1][i].item()
                vis = visibility_gt[j][i].item()

                if x_pred is not None:
                    if vis != 0:
                        try:
                            x_gt_scaled = int(x_gt * scale_w)
                            y_gt_scaled = int(y_gt * scale_h)
                        except:
                            self.fp2 += 1
                            continue
                        dist = distance.euclidean((x_pred, y_pred), (x_gt_scaled, y_gt_scaled))
                        self.pixel_errors.append(float(dist))
                        if dist < self.min_dist:
                            self.tp += 1
                        else:
                            self.fp1 += 1
                    else:
                        self.fp2 += 1
                else:
                    if vis != 0:
                        self.fn += 1
                    else:
                        self.tn += 1

    def compute(self) -> dict:
        """计算并返回最终的评估结果字典。"""
        eps = 1e-15
        self.fp = self.fp1 + self.fp2
        total = self.tp + self.fp + self.tn + self.fn
        accuracy = (self.tp + self.tn) / (total + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        mean_error = float(np.mean(self.pixel_errors)) if self.pixel_errors else float("nan")
        p90_error = float(np.percentile(self.pixel_errors, 90)) if self.pixel_errors else float("nan")

        return {
            'Total': total,
            'TP': self.tp,
            'FP1': self.fp1,
            'FP2': self.fp2,
            'FP': self.fp,
            'TN': self.tn,
            'FN': self.fn,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'MeanPixelError': mean_error,
            'P90PixelError': p90_error,
        }
