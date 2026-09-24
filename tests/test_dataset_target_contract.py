import cv2
import numpy as np
import torch

from datasets_factory.transforms.tracknet_transforms import (
    Finalize,
    LoadAndFormatMultiTargets,
    Resize,
)


def test_resize_keeps_coordinates_in_model_pixel_space():
    results = {
        "path_prev": np.zeros((100, 200, 3), dtype=np.uint8),
        "path": np.zeros((100, 200, 3), dtype=np.uint8),
        "path_next": np.zeros((100, 200, 3), dtype=np.uint8),
        "coords": [(100.5, 50.25), (0.0, 99.0), (float("nan"), float("nan"))],
    }
    Resize(keys=["path_prev", "path", "path_next"], size=(50, 80))(results)
    assert results["path"].shape == (50, 80, 3)
    assert np.allclose(results["coords"][0], (40.2, 25.125))
    assert np.isnan(results["coords"][2][0])


def test_finalize_returns_stable_metadata_tensors():
    results = {
        "image": np.zeros((4, 5, 3), dtype=np.uint8),
        "target": torch.zeros(3, 4, 5),
        "coords": [(1.5, 2.0), (float("nan"), float("nan")), (3.0, 1.0)],
        "visibility": [1, float("nan"), 0],
    }
    output = Finalize()(results)
    assert output["coords"].shape == (3, 2)
    assert output["coords"].dtype == torch.float32
    assert output["visibility"].shape == (3,)
    assert output["visibility"].dtype == torch.float32
    assert torch.isnan(output["coords"][1]).all()
    assert output["visibility"].tolist() == [1.0, 0.0, 0.0]


def test_target_resize_is_float_and_preserves_subpixel_mass(tmp_path):
    source = np.zeros((4, 4), dtype=np.uint8)
    source[1:3, 1:3] = 255
    path = tmp_path / "target.png"
    assert cv2.imwrite(str(path), source)
    results = {
        "input_width": 8,
        "input_height": 8,
        "gt_path_prev": path,
        "gt_path": path,
        "gt_path_next": path,
    }
    LoadAndFormatMultiTargets()(results)
    target = results["target"]
    assert target.shape == (3, 8, 8)
    assert target.dtype == torch.float32
    assert float(target.max()) == 255.0
    # Linear interpolation produces a continuous target instead of a nearest-neighbor block.
    assert float(target[0, 1, 1]) < 255.0

