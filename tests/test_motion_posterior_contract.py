import torch

from models_factory.builder import build_model
from models_factory.models.motion_posterior import MotionPosteriorNet
from core.postprocess import decode_prediction


def test_motion_model_rich_output_and_decoder_contract():
    model = MotionPosteriorNet(
        num_frames=3,
        return_aux=True,
        backbone=dict(
            type="MotionConvNeXtBackbone",
            num_frames=3,
            dims=(8, 12, 16),
            depths=(1, 1, 1),
        ),
    )
    prediction = model(torch.randn(1, 9, 32, 32))

    assert prediction["heatmap"].shape == (1, 3, 32, 32)
    assert prediction["offset"].shape == (1, 3, 2)
    assert prediction["visibility_logits"].shape == (1, 3)
    assert prediction["uncertainty"].shape == (1, 3)
    assert torch.isfinite(prediction["heatmap"]).all()

    # Decoder must accept the actual batched rich output used by deployment.
    point = decode_prediction(prediction, frame_index=1, threshold=0.0)
    assert point is None or len(point) == 3


def test_public_motionposterior_registry_contract():
    config = {
        "type": "MotionPosteriorNet",
        "num_frames": 3,
        "return_aux": True,
        "backbone": {
            "type": "MotionConvNeXtBackbone",
            "num_frames": 3,
            "dims": (8, 12, 16),
            "depths": (1, 1, 1),
        },
    }
    model = build_model(config)
    assert isinstance(model, MotionPosteriorNet)
    assert model.num_frames == 3
