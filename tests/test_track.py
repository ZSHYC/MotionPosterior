import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import track
import infer
import core.detector as detector_module
import core.pipeline as pipeline_module
from core.detector import MODEL_CONFIGS as CORE_MODEL_CONFIGS, TrackNetDetector
from core.pipeline import BallPoint, TennisPipeline
from datasets_factory.transforms.tracknet_transforms import ConcatChannels, Resize
from track import canonical_row, process_video


def test_canonical_row_scales_model_coordinates_and_preserves_metadata():
    row = canonical_row(
        sample_id="20260706_001",
        video_name="20260706_001.mp4",
        frame_number=7,
        coords=(269, 203, 0.7725),
        fps=20.0,
        width=1280,
        height=720,
    )

    assert row == {
        "benchmark_id": "20260706_001",
        "video_name": "20260706_001.mp4",
        "frame_number": 7,
        "detected": 1,
        "x_512": 269,
        "y_288": 203,
        "x_orig": 672.5,
        "y_orig": 507.5,
        "conf": 0.7725,
        "fps": 20.0,
        "width": 1280,
        "height": 720,
    }

    missing = canonical_row(
        "20260706_001",
        "20260706_001.mp4",
        8,
        None,
        20.0,
        1280,
        720,
    )
    assert missing["detected"] == 0
    assert missing["conf"] == 0.0
    assert all(missing[key] is None for key in ("x_512", "y_288", "x_orig", "y_orig"))


def _write_video(path, frame_count=4, fps=20.0, size=(1280, 720)):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        size,
    )
    assert writer.isOpened()
    for _ in range(frame_count):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


class _FakeModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        result = torch.zeros((1, 3, 288, 512), device=image.device)
        result[:, :, 50:53, 100:103] = 1
        return result


@pytest.mark.parametrize(
    ("frame_count", "expected_detections"),
    [(4, [1, 1, 1, 1]), (5, [1, 1, 1, 1, 1])],
)
def test_process_video_writes_complete_canonical_csv_without_summary_or_videos(
    tmp_path,
    frame_count,
    expected_detections,
):
    video = tmp_path / "20260706_001.avi"
    output = tmp_path / "tracks"
    _write_video(video, frame_count=frame_count)

    model = _FakeModel()
    stats = process_video(
        video,
        model,
        torch.device("cpu"),
        SimpleNamespace(threshold=0.5, visualization_dir=None),
        output,
    )

    csv_path = output / "20260706_001.csv"
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert list(rows[0]) == [
        "benchmark_id",
        "video_name",
        "frame_number",
        "detected",
        "x_512",
        "y_288",
        "x_orig",
        "y_orig",
        "conf",
        "fps",
        "width",
        "height",
    ]
    assert [int(row["frame_number"]) for row in rows] == list(range(frame_count))
    assert [int(row["detected"]) for row in rows] == expected_detections
    assert float(rows[0]["x_orig"]) == 252.5
    assert float(rows[0]["y_orig"]) == 127.5
    assert float(rows[0]["conf"]) == 1.0
    assert "total_detected_frame" not in csv_path.read_text(encoding="utf-8")
    assert not list(output.rglob("*.mp4"))
    assert stats["total_frames"] == frame_count
    assert stats["detected_frames"] == frame_count
    assert model.calls == 2


@pytest.mark.parametrize("threshold", [-0.001, 1.0, 1.001, float("nan"), float("inf")])
def test_process_video_rejects_invalid_threshold_before_opening_video(tmp_path, threshold):
    with pytest.raises(ValueError, match="threshold"):
        process_video(
            tmp_path / "missing.mp4",
            _FakeModel(),
            torch.device("cpu"),
            SimpleNamespace(threshold=threshold, visualization_dir=None),
            tmp_path / "tracks",
        )


def test_process_video_rejects_non_16_by_9_video(tmp_path):
    video = tmp_path / "sample.avi"
    _write_video(video, size=(640, 480))

    with pytest.raises(ValueError, match="16:9"):
        process_video(
            video,
            _FakeModel(),
            torch.device("cpu"),
            SimpleNamespace(threshold=0.5, visualization_dir=None),
            tmp_path / "tracks",
        )


def test_cli_requires_canonical_output_directory():
    parser = track.build_parser()

    args = parser.parse_args([
        "videos",
        "weights.pth",
        "--arch",
        "v5",
        "--output-dir",
        "tracks",
    ])

    assert args.output_dir.name == "tracks"
    assert args.visualization_dir is None


def test_cli_accepts_public_motionposterior_architecture():
    parser = track.build_parser()
    args = parser.parse_args([
        "videos",
        "weights.pth",
        "--arch",
        "motion_posterior5",
        "--output-dir",
        "tracks",
    ])
    assert args.arch == "motion_posterior5"
    assert track.MODEL_CONFIGS[args.arch]["type"] == "MotionPosteriorNet"
    assert CORE_MODEL_CONFIGS[args.arch]["type"] == "MotionPosteriorNet"


@pytest.mark.parametrize("arch", ["v3", "v4", "v4_typea", "v4_typeb"])
def test_cli_accepts_official_baseline_architectures(arch):
    parser = track.build_parser()
    args = parser.parse_args([
        "videos",
        "weights.pth",
        "--arch",
        arch,
        "--output-dir",
        "tracks",
    ])
    assert args.arch == arch


def test_legacy_infer_entrypoint_uses_canonical_track_cli():
    assert infer.main is track.main


def test_main_validates_paths_before_building_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "track.py",
            str(tmp_path / "missing"),
            str(tmp_path / "missing.pth"),
            "--arch",
            "v5",
            "--output-dir",
            str(tmp_path / "tracks"),
        ],
    )
    monkeypatch.setattr(
        track,
        "build_model",
        lambda _config: pytest.fail("model must not be built before path validation"),
    )

    with pytest.raises(SystemExit):
        track.main()


def test_main_rejects_empty_video_directory_before_building_model(tmp_path, monkeypatch):
    weights = tmp_path / "weights.pth"
    weights.touch()
    monkeypatch.setattr(
        "sys.argv",
        [
            "track.py",
            str(tmp_path),
            str(weights),
            "--arch",
            "v5",
            "--output-dir",
            str(tmp_path / "tracks"),
        ],
    )
    monkeypatch.setattr(
        track,
        "build_model",
        lambda _config: pytest.fail("model must not be built without input videos"),
    )

    with pytest.raises(SystemExit):
        track.main()


@pytest.mark.parametrize("frame_count", [4, 5])
def test_core_detector_keeps_tail_frames_and_returns_original_coordinates(
    tmp_path,
    frame_count,
):
    video = tmp_path / "sample.avi"
    _write_video(video, frame_count=frame_count)
    detector = TrackNetDetector.__new__(TrackNetDetector)
    detector.device = torch.device("cpu")
    detector.threshold = 0.5
    detector.input_size = (288, 512)
    detector.model = _FakeModel()
    detector.resizer = Resize(keys=["p", "c", "n"], size=detector.input_size)
    detector.concator = ConcatChannels(keys=["p", "c", "n"], output_key="img")

    points = detector.detect_video(str(video))

    assert len(points) == frame_count
    assert all(point.is_detected for point in points)
    assert points[0].x == 252.5
    assert points[0].y == 127.5
    assert detector.model.calls == 2


def test_core_detector_exposes_official_baselines_and_rejects_invalid_threshold(monkeypatch):
    assert {"v2", "v3", "v4", "v4_typea", "v4_typeb", "v5"} <= set(CORE_MODEL_CONFIGS)
    monkeypatch.setattr(
        detector_module,
        "build_model",
        lambda _config: pytest.fail("model must not be built for invalid threshold"),
    )

    with pytest.raises(ValueError, match="threshold"):
        TrackNetDetector("v5", "unused.pth", device="cpu", threshold=1.0)


class _FakeRenderCapture:
    def __init__(self):
        self.read_count = 0

    def isOpened(self):
        return True

    def get(self, key):
        return {
            cv2.CAP_PROP_FPS: 29.97,
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FRAME_COUNT: 1,
        }[key]

    def read(self):
        self.read_count += 1
        if self.read_count == 1:
            return True, np.zeros((720, 1280, 3), dtype=np.uint8)
        return False, None

    def release(self):
        pass


class _FakeRenderWriter:
    def __init__(self, opened=True):
        self.opened = opened
        self.frames = []

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        pass


def test_core_pipeline_preserves_fractional_fps_and_checks_writer(monkeypatch):
    captured = {}
    writer = _FakeRenderWriter()
    monkeypatch.setattr(
        pipeline_module.cv2,
        "VideoCapture",
        lambda _path: _FakeRenderCapture(),
    )

    def make_writer(_path, _fourcc, fps, _size):
        captured["fps"] = fps
        return writer

    monkeypatch.setattr(pipeline_module.cv2, "VideoWriter", make_writer)
    visualizer = SimpleNamespace(render=lambda frame, _point: frame)
    pipeline = TennisPipeline(None, None, visualizer)

    pipeline._pass3_rendering(
        Path("sample.mp4"),
        "output.mp4",
        {0: BallPoint(is_detected=False)},
    )

    assert captured["fps"] == 29.97
    assert len(writer.frames) == 1

    monkeypatch.setattr(
        pipeline_module.cv2,
        "VideoWriter",
        lambda *_args: _FakeRenderWriter(opened=False),
    )
    with pytest.raises(ValueError, match="writer"):
        pipeline._pass3_rendering(Path("sample.mp4"), "output.mp4", {})
