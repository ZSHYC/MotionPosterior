import torch

from losses_factory.losses.tracknetv2_loss import TrackNetV2Loss


def _prediction(logits):
    return {
        "heat_logits": logits,
        "offset": torch.tensor([[[0.20, 0.20], [0.45, 0.30], [0.70, 0.40]]]),
        "visibility_logits": torch.zeros(1, 3),
        "log_variance": torch.zeros(1, 3),
    }


def test_probability_and_logit_heatmap_contract_match():
    target = torch.zeros(1, 3, 4, 4)
    target[:, :, 1, 2] = 1.0
    logits = torch.full_like(target, -2.0)
    loss = TrackNetV2Loss(aux_weight=0.0)
    expected = loss(logits, target)
    actual = loss(torch.sigmoid(logits), target)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_metadata_visibility_masks_geometry_and_uses_pixel_coordinates():
    target = torch.zeros(1, 3, 4, 4)
    target[:, :, 1, 2] = 255.0
    pred = _prediction(torch.zeros_like(target))
    # Default-collate layout: [T][B, 2] / [T][B]. Middle frame is invisible.
    coords = [torch.tensor([[2.0, 1.0]]), torch.tensor([[99.0, 99.0]]), torch.tensor([[2.0, 1.0]])]
    visibility = [torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0])]
    criterion = TrackNetV2Loss(trajectory_weight=1.0)
    value = criterion(pred, target, coords=coords, visibility=visibility, fps=50.0)
    assert torch.isfinite(value)
    centres, visible = criterion._metadata_targets(target, coords, visibility)
    assert visible.tolist() == [[True, False, True]]
    assert torch.allclose(centres[0, 0], torch.tensor([2 / 3, 1 / 3]))


def test_missing_metadata_falls_back_to_heatmap_mass():
    target = torch.zeros(1, 2, 4, 4)
    target[0, 0, 1, 2] = 255.0
    pred = {
        "heatmap": torch.full_like(target, 0.5),
        "offset": torch.zeros(1, 2, 2),
    }
    value = TrackNetV2Loss()(pred, target)
    assert torch.isfinite(value)
