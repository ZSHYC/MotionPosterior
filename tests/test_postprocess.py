import numpy as np
import torch

from core.postprocess import decode_prediction


def test_rich_prediction_uses_refined_offset_and_visibility_confidence():
    heatmap = torch.zeros((1, 3, 8, 12))
    heatmap[:, :, 2:4, 5:7] = 0.9
    prediction = {
        "heatmap": heatmap,
        "offset": torch.tensor([[[0.25, 0.5], [0.75, 0.25], [0.5, 0.5]]]),
        "visibility_logits": torch.tensor([[4.0, 2.0, 4.0]]),
        "uncertainty": torch.tensor([[0.0, 0.0, 0.0]]),
    }

    point = decode_prediction(prediction, frame_index=1, threshold=0.5)

    assert point is not None
    assert np.allclose(point[:2], (0.75 * 11, 0.25 * 7))
    assert point[2] < 0.9
    assert point[2] > 0.8


def test_rich_prediction_uncertainty_reduces_confidence_and_can_abstain():
    prediction = {
        "heatmap": torch.ones((1, 1, 4, 4)),
        "offset": torch.tensor([[[0.5, 0.5]]]),
        "visibility_logits": torch.tensor([[5.0]]),
        "uncertainty": torch.tensor([[2.0]]),
    }
    point = decode_prediction(prediction, threshold=0.5)
    assert point is not None and point[2] < 0.2
    assert decode_prediction(prediction, threshold=0.5, uncertainty_threshold=1.0) is None


def test_rich_prediction_abstains_when_visibility_is_low():
    prediction = {
        "heatmap": torch.ones((1, 1, 4, 4)),
        "offset": torch.tensor([[[0.5, 0.5]]]),
        "visibility_logits": torch.tensor([[-4.0]]),
    }

    assert decode_prediction(prediction, threshold=0.5) is None


def test_legacy_tensor_prediction_keeps_contour_decoder():
    heatmap = torch.zeros((1, 1, 8, 12))
    heatmap[:, :, 3:5, 6:8] = 1.0

    point = decode_prediction(heatmap, threshold=0.5)

    assert point is not None
    assert point[2] == 1.0
    assert 6 <= point[0] <= 7
    assert 3 <= point[1] <= 4
