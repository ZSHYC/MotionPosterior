import torch
import pytest

from metrics_factory.metrics.tracknetv2_metric import TrackNetV2Metric


def test_metric_consumes_batched_motion_posterior():
    metric = TrackNetV2Metric(min_dist=1, original_size=(8, 12))
    heatmap = torch.zeros(1, 1, 8, 12)
    heatmap[:, :, 3, 6] = 0.95
    prediction = {
        "heatmap": heatmap,
        "offset": torch.tensor([[[6.0 / 11.0, 3.0 / 7.0]]]),
        "visibility_logits": torch.tensor([[5.0]]),
    }
    metric.update(
        prediction,
        {
            "coords": torch.tensor([[[6.0, 3.0]]]),
            "visibility": torch.tensor([[1.0]]),
        },
    )
    result = metric.compute()
    assert result["TP"] == 1
    assert result["VisibleRecall"] == pytest.approx(1.0)
