# MotionPosterior

**[中文版 / Chinese](README_CN.md)**

This repository contains the PyTorch implementation and research baseline suite for
**MotionPosterior**, a posterior-aware detector for tiny, fast-moving point targets.
The current benchmark is tennis-ball tracking; the model contract is not tied to a
particular sport or to the old `rally` name.

`MotionPosteriorNet` is the final mainline model. `TrackNetV2`, `TrackNetV3`,
`TrackNetV4`, and `TrackNetV5` remain available as historical baselines and for
checkpoint/script compatibility. The repository does not use `TrackNetMotion` as a
public model.

## Model choices

| CLI entry | Temporal contract | Role |
| --- | --- | --- |
| `motion_posterior3` | 3 RGB frames in, 3 heatmaps out | Three-frame reference |
| `motion_posterior5` | `[t-2,t-1,t,t+1,t+2]` in, 5 outputs | Offline mainline |
| `motion_posterior5_causal` | `[t-4,t-3,t-2,t-1,t]` in, 5 outputs | Online mainline |
| `v2` | TrackNetV2-style 3-frame heatmaps | Lightweight baseline |
| `v3` | Official TrackNetV3-style 3-frame heatmaps | Recovered V3 baseline |
| `v4`, `v4_typea`, `v4_typeb` | PyTorch TrackNetV4-style 3-frame motion fusion | V4 ablation baselines |
| `v5` | MDD + R-STR, 3-frame heatmaps | Motion-aware historical baseline |

All entries are built through the same PyTorch model factory. Baseline construction
tests only establish that the architecture runs; accuracy claims require matching
weights, the same video-level split, preprocessing, threshold, and metrics.

## Why V5 is still in the repository

V5 is not the direction of the final architecture, but it is not useless. The released
[TrackNetV5 paper](https://arxiv.org/abs/2512.02789) proposes Motion Direction
Decoupling (MDD) and a Residual-Driven Spatio-Temporal Refinement (R-STR) head. In
this repository, the V5 path implements the corresponding low-cost idea: signed
luminance-difference maps condition the V2-style backbone, and motion tokens provide
context to the three-frame refinement head.

Its value is as a controlled historical motion baseline and an ablation point. Its
limits are also clear: it is fixed to three frames, represents motion mainly through
pixel luminance differences, has no explicit camera/feature alignment, and does not
produce the posterior quantities used by the mainline (`offset`, `visibility`,
`uncertainty`, velocity, and acceleration). The paper's reported numbers are not
treated as reproduced results for this checkout. Further architecture work therefore
goes into `MotionPosteriorNet`; V5 is kept stable for comparison and compatibility.

## Mainline architecture

`MotionPosteriorNet` accepts `[B, 3T, H, W]` with `T=3` or `T=5` and returns one
full-resolution heatmap per input frame plus auxiliary posterior heads:

* ConvNeXt V2-inspired hierarchical encoder with `full/half/quarter/eighth` features;
* confidence-gated global translation compensation at multiple scales;
* velocity/uncertainty-conditioned local cross-frame correlation and dense offset refinement;
* high-resolution skip paths for tiny-object detail;
* heatmap logits, normalized center offset, visibility, uncertainty, velocity, and acceleration.

Deployment decodes the rich output in this order:

```text
heatmap candidate -> offset refinement -> visibility gate -> uncertainty
```

The single architecture document contains the tensor contracts, losses, motion design,
baseline recovery notes, and literature references: [docs/MotionPosterior.md](docs/MotionPosterior.md).

## Environment

Python 3.10 is the recommended interpreter. PyTorch and torchvision are intentionally
not hard-pinned in `requirements.txt`; install a build that matches the available CUDA
runtime. The remaining pinned dependencies are listed in [requirements.txt](requirements.txt).

```bash
pip install -r requirements.txt
```

## Data preparation

The preprocessing tool renders soft Gaussian heatmaps and writes grouped train/validation
CSVs. Splitting is performed by clip to avoid adjacent-window leakage.

Three-frame context:

```bash
python tools/preprocess_data_gauss.py \
  --input_dir <raw> --output_dir <data> \
  --mode context --num-frames 3 --train_rate 0.8
```

Five-frame offline context:

```bash
python tools/preprocess_data_gauss.py \
  --input_dir <raw> --output_dir <data> \
  --mode context --num-frames 5 --window-type center --train_rate 0.8
```

Five-frame causal context:

```bash
python tools/preprocess_data_gauss.py \
  --input_dir <raw> --output_dir <data> \
  --mode context --num-frames 5 --window-type causal --train_rate 0.8
```

The causal command writes `labels_causal5_train.csv` and
`labels_causal5_val.csv`; the centered command writes `labels_context5_train.csv`
and `labels_context5_val.csv`.

## Training

`train.py` lists the configuration files in `configs/`; choose one when prompted.
The mainline configurations are:

* `configs/motionposterior_convnext_3frames.py`
* `configs/motionposterior_convnext_5frames.py`
* `configs/motionposterior_convnext_5frames_causal.py`

```bash
python train.py
```

The loss consumes real `[T,2]` coordinates and `[T]` visibility when available. Position,
uncertainty, velocity, and acceleration terms are visibility-masked. Checkpoints include
model, optimizer, scheduler, progress, configuration, and RNG state for resume.

## Inference

The batch pipeline writes one canonical CSV per video. The CLI currently resizes to
`512x288`, requires 16:9 input videos, and scales `x_512/y_288` back to the original
video metadata for `x_orig/y_orig`.

Five-frame offline inference:

```bash
python track.py <input_dir> <weights.pth> \
  --arch motion_posterior5 \
  --window-mode center \
  --output-dir <trajectory_csv_dir> \
  --threshold 0.5 --device cuda:0
```

Five-frame online inference:

```bash
python track.py <input_dir> <weights.pth> \
  --arch motion_posterior5_causal \
  --window-mode causal \
  --output-dir <trajectory_csv_dir> \
  --threshold 0.5 --device cuda:0
```

Use `--window-mode chunk` when the equal-T input/output behavior of the legacy pipeline
is required. Add `--visualization-dir <dir>` only when trajectory and heatmap videos are
needed.

CSV fields are:

```text
benchmark_id,video_name,frame_number,detected,x_512,y_288,x_orig,y_orig,conf,fps,width,height
```

## Evaluation guidance

Report three-frame center, five-frame center, five-frame causal, and chunk protocols
separately. At minimum use the same video-level split and report PCK/EPE, visible-target
recall, miss rate, false-positive rate, visibility recall, P50/P90/P95 pixel error, and
end-to-end latency. A model smoke test or a paper-reported score is not a substitute for
retraining and evaluating on the project's data.

## License and data

This SDK and its model weights/data are proprietary to Shanghai Code Zero Sports
Technology Co., Ltd. Weights and training data are not included in this repository.

© 2025 Shanghai Code Zero Sports Technology Co., Ltd.
