import torch

import models_factory
from models_factory.builder import build_model
from track import MODEL_CONFIGS


def _small_config(name):
    config = dict(MODEL_CONFIGS[name])
    if name == "v5":
        config["head"] = {
            **config["head"],
            "img_size": (32, 32),
            "patch_size": 8,
            "embed_dim": 32,
            "num_transformer_layers": 1,
            "num_transformer_heads": 2,
        }
    return config


@torch.no_grad()
def test_v2_v3_v4_v5_baselines_build_and_emit_three_heatmaps():
    for name in ("v2", "v3", "v4", "v4_typeb", "v5"):
        model = build_model(_small_config(name)).eval()
        output = model(torch.randn(1, 9, 32, 32))
        assert output.shape == (1, 3, 32, 32)
        assert torch.isfinite(output).all()


@torch.no_grad()
def test_official_v3_inpaintnet_contract():
    model = build_model({"type": "InpaintNetV3"}).eval()
    output = model(torch.rand(2, 8, 2), torch.ones(2, 8, 1))
    assert output.shape == (2, 8, 2)
    assert torch.isfinite(output).all()
