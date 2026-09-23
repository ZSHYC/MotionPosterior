import torch

from models_factory.backbones.motion_convnext_backbone import (
    MotionConvNeXtBackbone,
    _SpatialTemporalMotionFusion,
)


def test_motion_fusion_has_nonzero_alignment_gradients():
    torch.manual_seed(7)
    fusion = _SpatialTemporalMotionFusion(channels=8, radius=1, correlation_channels=4)
    inputs = torch.randn(2, 3, 8, 8, 8, requires_grad=True)

    fusion(inputs).square().mean().backward()

    for name in ("query.weight", "key.weight", "offset.0.weight", "offset.2.weight"):
        gradient = dict(fusion.named_parameters())[name].grad
        assert gradient is not None, f"missing gradient for {name}"
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum().item() > 0.0, f"zero gradient for {name}"

    assert fusion.motion_scale.detach().item() > 0.0


def test_motion_backbone_keeps_three_and_five_frame_contracts():
    for frames in (3, 5):
        model = MotionConvNeXtBackbone(
            num_frames=frames, dims=(8, 12, 16), depths=(1, 1, 1)
        )
        outputs = model(torch.randn(1, frames * 3, 32, 32))
        assert outputs["full"].shape[:2] == (1, frames)
        assert outputs["half"].shape[:2] == (1, frames)
        assert outputs["quarter"].shape[:2] == (1, frames)
        assert outputs["eighth"].shape[:2] == (1, frames)
