import cv2
import numpy as np
from scipy.spatial import distance
import torch
from ..builder import METRICS
from core.postprocess import decode_prediction



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
        self.visible_total = 0
        self.visible_detected = 0

    @staticmethod
    def _batch_time(value, batch_size, time_steps):
        """Normalize metadata to [B,T,...] for old and new collate layouts."""
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.ndim >= 2 and value.shape[0] == time_steps and value.shape[1] == batch_size:
            value = value.transpose(1, 0, *range(2, value.ndim))
        if value.ndim >= 1 and value.shape[0] != batch_size and value.size == batch_size * time_steps:
            value = value.reshape(batch_size, time_steps)
        return value

    def update(self, logits: torch.Tensor, batch: dict):
        """根据一个批次的数据，更新计分板。"""
        predictions = logits["heatmap"].detach().cpu().numpy() if isinstance(logits, dict) else logits.detach().cpu().numpy()
        _, c, h, w = predictions.shape
        batch_size = predictions.shape[0]
        coords_gt = self._batch_time(batch.get('coords'), batch_size, c)
        visibility_gt = self._batch_time(batch.get('visibility'), batch_size, c)
        threshold = self.heatmap_threshold / 255.0 if self.heatmap_threshold > 1 else self.heatmap_threshold

        for i in range(batch_size):
            for j in range(c):
                decoded = decode_prediction(logits, j, threshold=threshold, batch_index=i)
                x_pred, y_pred = (decoded[:2] if decoded is not None else (None, None))
                if coords_gt is None or visibility_gt is None:
                    continue
                x_gt, y_gt = np.asarray(coords_gt[i, j, :2], dtype=float)
                vis = float(np.asarray(visibility_gt[i, j]).reshape(-1)[0])
                visible = vis > 0.5 and np.isfinite([x_gt, y_gt]).all()
                if visible:
                    self.visible_total += 1

                if x_pred is not None:
                    if visible:
                        # New Finalize emits model-space coordinates.  Keep a
                        # compatibility conversion for callers still passing
                        # source-resolution metadata.
                        if x_gt > w or y_gt > h:
                            x_gt *= w / self.original_w
                            y_gt *= h / self.original_h
                        dist = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        self.pixel_errors.append(float(dist))
                        if dist < self.min_dist:
                            self.tp += 1
                            self.visible_detected += 1
                        else:
                            self.fp1 += 1
                    else:
                        self.fp2 += 1
                else:
                    if visible:
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
        p50_error = float(np.percentile(self.pixel_errors, 50)) if self.pixel_errors else float("nan")
        p95_error = float(np.percentile(self.pixel_errors, 95)) if self.pixel_errors else float("nan")

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
            'P50PixelError': p50_error,
            'P90PixelError': p90_error,
            'P95PixelError': p95_error,
            'VisibleRecall': self.visible_detected / (self.visible_total + eps),
            'MissRate': self.fn / (self.visible_total + eps),
        }
