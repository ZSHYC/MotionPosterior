"""Deployment-time decoding for heatmap and motion-aware predictions.

The model's rich output is intentionally decoded here so every inference
entrypoint uses the same visibility gate and coordinate convention.  The
legacy heatmap decoder remains available for older checkpoints.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import cv2
import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def heatmap_frame(prediction: Any, frame_index: int = 0, batch_index: int = 0) -> np.ndarray:
    """Return one probability heatmap from tensor or rich model output."""
    value = prediction.get("heatmap") if isinstance(prediction, dict) else prediction
    heatmap = _numpy(value)
    if heatmap.ndim == 4:  # [B, T, H, W]
        heatmap = heatmap[batch_index, frame_index]
    elif heatmap.ndim == 3:  # [T, H, W]
        heatmap = heatmap[frame_index]
    elif heatmap.ndim != 2:
        raise ValueError(f"expected heatmap [H,W], [T,H,W] or [B,T,H,W], got {heatmap.shape}")
    return np.nan_to_num(heatmap.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def decode_heatmap(heatmap: np.ndarray, threshold: float = 0.5) -> Optional[Tuple[float, float, float]]:
    """Legacy contour decoder, returning model-space ``(x, y, confidence)``."""
    heatmap = np.nan_to_num(np.asarray(heatmap, dtype=np.float32), nan=0.0).clip(0.0, 1.0)
    if heatmap.ndim != 2 or heatmap.size == 0:
        raise ValueError(f"expected a non-empty [H,W] heatmap, got {heatmap.shape}")
    peak = float(heatmap.max())
    if peak < threshold:
        return None
    heatmap_uint8 = np.rint(heatmap * 255.0).astype(np.uint8)
    _, binary = cv2.threshold(heatmap_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    mask = np.zeros_like(heatmap_uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    _, max_value, _, _ = cv2.minMaxLoc(heatmap_uint8, mask=mask)
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]), float(max_value / 255.0)


def _frame_value(value: Any, frame_index: int, batch_index: int = 0) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = _numpy(value)
    if array.ndim == 2 and array.shape[-1] == 2:
        # Unbatched offset sequence: [T, 2].
        array = array[frame_index]
    elif array.ndim == 2:
        # Batched scalar sequence: [B, T].
        array = array[batch_index if array.shape[0] > 1 else 0, frame_index]
    elif array.ndim >= 3:
        array = array[batch_index if array.shape[0] > 1 else 0, frame_index]
    elif array.ndim == 1:
        array = array[frame_index]
    return np.asarray(array)


def decode_prediction(
    prediction: Any,
    frame_index: int = 0,
    threshold: float = 0.5,
    visibility_threshold: Optional[float] = None,
    batch_index: int = 0,
    uncertainty_threshold: Optional[float] = None,
) -> Optional[Tuple[float, float, float]]:
    """Decode tensor or rich prediction into a model-space point.

    Rich outputs use the model's normalized ``offset`` as the refined centre,
    while the heatmap supplies the candidate and confidence gate.  A missing
    or low visibility logit abstains instead of inventing a position.
    """
    if not isinstance(prediction, dict):
        return decode_heatmap(heatmap_frame(prediction, frame_index, batch_index), threshold)

    heatmap = heatmap_frame(prediction, frame_index, batch_index)
    peak = float(heatmap.max())
    if not math.isfinite(peak) or peak < threshold:
        return None

    visibility_threshold = threshold if visibility_threshold is None else float(visibility_threshold)
    visibility = _frame_value(prediction.get("visibility_logits"), frame_index, batch_index)
    if visibility is not None:
        visibility_value = float(1.0 / (1.0 + np.exp(-float(np.ravel(visibility)[0]))))
        if not math.isfinite(visibility_value) or visibility_value < visibility_threshold:
            return None
    else:
        visibility_value = 1.0

    uncertainty = _frame_value(prediction.get("uncertainty"), frame_index, batch_index)
    uncertainty_value = None
    if uncertainty is not None and np.asarray(uncertainty).size:
        uncertainty_value = float(np.ravel(uncertainty)[0])
        if not math.isfinite(uncertainty_value):
            return None
        uncertainty_value = max(0.0, uncertainty_value)
        if uncertainty_threshold is not None and uncertainty_value > float(uncertainty_threshold):
            return None
    confidence = min(peak, visibility_value)
    if uncertainty_value is not None:
        confidence *= math.exp(-min(uncertainty_value, 8.0))

    offset = _frame_value(prediction.get("offset"), frame_index, batch_index)
    if offset is not None and np.asarray(offset).size >= 2:
        x_norm, y_norm = (float(v) for v in np.ravel(offset)[:2])
        if math.isfinite(x_norm) and math.isfinite(y_norm):
            height, width = heatmap.shape
            x = float(np.clip(x_norm, 0.0, 1.0) * max(width - 1, 1))
            y = float(np.clip(y_norm, 0.0, 1.0) * max(height - 1, 1))
            return x, y, float(confidence)

    decoded = decode_heatmap(heatmap, threshold)
    if decoded is None:
        return None
    return decoded[0], decoded[1], float(confidence)
