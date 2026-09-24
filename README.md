# MotionPosterior

**[中文版 (Chinese Version)](README_CN.md)**

This repository is the official SDK for **MotionPosterior**, a motion-aware posterior estimation framework for tiny, fast-moving point targets. The current benchmark is tennis-ball tracking, but the public architecture is not limited to rally or sports footage. Developed and maintained by **Shanghai Code Zero Sports Technology Co., Ltd.**

`MotionPosteriorNet` is the final model name. `TrackNetV2` and `TrackNetV5` remain
available as historical baselines for existing scripts and checkpoints.

The core architecture and algorithmic logic of TrackNetV5 are based on our latest research:

* **Title**: *TrackNetV5: Residual-Driven Spatio-Temporal Refinement and Motion Direction Decoupling for Fast Object Tracking*
* **Paper**: [arXiv:2512.02789](https://arxiv.org/abs/2512.02789)

## Core Specifications

* **Architecture Support**: Supports the public `MotionPosteriorNet` 3/5-frame paths, plus TrackNetV2, official TrackNetV3, and PyTorch TrackNetV4 TypeA/TypeB and TrackNetV5 baselines.
* **Integrated Features**: Encapsulates configurable three- or five-frame inference, Gaussian heatmap centroid extraction, trajectory enhancement visualization, and an industrial-grade training pipeline.
* **Confidentiality Notice**: Model weights and training datasets are proprietary assets of the company and are currently not open to the public.

---

## 1. Environment Configuration

This project is optimized for specific computing environments. To ensure system stability, please use the following recommended versions:

### Core Dependencies

| Component | Recommended Version |
| --- | --- |
| **Python** | 3.10 |
| **CUDA** | 12.6 |
| **PyTorch** | 2.9.0+cu126 |
| **Torchvision** | 0.24.0+cu126 |

### Installation

```bash
# 1. Install basic scientific computing and image processing libraries
pip install -r requirements.txt

# 2. Install specific PyTorch ecosystem versions
# It is recommended to download the corresponding .whl files from the official PyTorch website

```

---

## 2. Data Preparation

The SDK utilizes **Gaussian Heatmaps** as the supervision signal.

### Dataset Standards

Please strictly follow the directory structure and labeling specifications of the following repository for custom datasets:

* **Reference**: `WASB-TrainingOK` Dataset Specification.

### Preprocessing Script

Use `tools/preprocess_data_gauss.py` to convert raw video frames and `Label.csv` into the spatio-temporal context tensors required by the model.

```bash
python tools/preprocess_data_gauss.py \
    --input_dir <path_to_raw_data> \
    --output_dir <path_to_output> \
    --mode context \
    --num-frames 3 \
    --train_rate 0.8 \
    --height 1080 --width 1920

```

* **Key Parameters**:
* `--mode`: Use `context` for centered temporal windows.
* `--num-frames`: Use `3` for the legacy CSV contract or `5` to generate `labels_context5_train.csv` and `labels_context5_val.csv`.
* `--size` & `--variance`: Controls the radius and variance of the generated Gaussian spots.



---

## 3. Training Guide

Training tasks are dispatched via `train.py`, which utilizes a **Factory Pattern** for dynamic component construction.

### Start Training Queue

```bash
python train.py

```

### Operational Steps

1. **Auto-Scan**: The system lists all `.py` configuration files in the `./configs/` directory.
2. **Selection**: Enter the configuration index (supports space-separated multi-task queues, e.g., `1 3 5`).
3. **Execution**: The `Runner` orchestrator automatically handles instantiation, Learning Rate Warmup, Gradient Clipping (`GradClip`), and Hook plugin mounting.

---

## 4. Inference Pipeline

The inference module supports batch video processing and structured data export.

### Execution Command

```bash
python track.py <input_dir> <weights_path> \
  --arch motion_posterior5 \
  --output-dir <trajectory_csv_dir> \
  --threshold 0.5 \
  --device cuda:0
```

### Output Description

Each input video produces one same-stem CSV in `trajectory_csv_dir`; for example,
`20260706_001.mp4` produces `20260706_001.csv`. The schema is:

```text
benchmark_id,video_name,frame_number,detected,x_512,y_288,x_orig,y_orig,conf,fps,width,height
```

Frame numbers are zero-based and every decoded frame has one row.
`x_512/y_288` are model-space coordinates; `x_orig/y_orig` are scaled to the
source-video metadata. TrackNetV5 has a fixed `512×288` model input, so source
videos must be 16:9. Detection statistics are printed to the terminal and are
not appended to the CSV.

Visualization videos are disabled by default. Add
`--visualization-dir <visualization_dir>` when needed.

---

## 5. Architecture Deep-Dive & Resources

### Recovered open-source baselines

The inference choices include `v2`, official `v3`, `v4`/`v4_typea`, `v4_typeb`, and
the project `v5` baseline. V3 also registers its separate `InpaintNetV3` trajectory
rectifier. V4 TypeA and TypeB are PyTorch implementations of the official
[TrackNetV4 repository](https://github.com/TrackNetV4/TrackNetV4); TensorFlow is not a
runtime dependency. These baselines are structural smoke-test targets and still need
their matching checkpoints and a common evaluation split for accuracy claims.

### Motion-aware V5 upgrade

The V5 path keeps its three RGB frames, four-channel MDD prompt, three heatmap outputs, and inference contract. It now adds two internal motion paths:

* `MotionConditionedGate` conditions `skip1`, `skip2`, `skip3`, and `bottleneck` on the signed MDD maps with spatial and channel gates.
* `R_STRHead` patchifies the motion maps into context tokens and lets the existing temporal Transformer attend to them without changing the three-frame output shape or loss interface.

The upgrade deliberately avoids optical flow, deformable convolution, and new dependencies. It follows the short-term temporal-difference direction used in [LSTFE-Net (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Xiao_LSTFE-NetLong_Short-Term_Feature_Enhancement_Network_for_Video_Small_Object_Detection_CVPR_2023_paper.pdf), [Temporal Difference Learning (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Feng_Mutual_Information-Based_Temporal_Difference_Learning_for_Human_Pose_Estimation_in_CVPR_2023_paper.pdf), and [Look Back and Forth (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Isobe_Look_Back_and_Forth_Video_Super-Resolution_With_Explicit_Temporal_Difference_CVPR2022_paper.pdf). BasicVSR++ is kept as a future, heavier alignment direction rather than a direct dependency.

The engineering design patterns, TrackNetV5 model details, and underlying inference logic are documented in our exclusive **Obsidian Visual Knowledge Base**.

### MotionPosterior architecture

`MotionPosteriorNet` is the final motion-aware architecture. It accepts either `[B, 9, H, W]` or
`[B, 15, H, W]` and returns the same number of full-resolution heatmaps:

```bash
# Five-frame preprocessing and training configuration
python tools/preprocess_data_gauss.py --input_dir <raw> --output_dir <data> \
  --mode context --num-frames 5 --train_rate 0.8
python train.py  # select configs/motionposterior_convnext_5frames.py

# Five-frame inference
python track.py <input_dir> <weights_path> --arch motion_posterior5 \
  --output-dir <trajectory_csv_dir> --threshold 0.5 --device cuda:0
```

For an online five-frame model, preprocess with `--window-type causal` and use
`configs/motionposterior_convnext_5frames_causal.py`; its input is
`[t-4,t-3,t-2,t-1,t]` and never includes future frames.

`motion_posterior3` and `motion_posterior5` default to `chunk`, preserving equal T-frame input/output
windows. Use `--window-mode center` for centered sliding evaluation or
`--window-mode causal` for online inference. Training checkpoints include model,
optimizer, scheduler, and progress state; set `resume_from` to continue.

The new backbone is a ConvNeXt V2-inspired hierarchical encoder with GRN,
full/half/quarter/eighth-resolution features. At half, quarter, and eighth scales it
estimates bounded global translation, performs 3x3 spatial-temporal correlation,
and applies dense offset refinement with valid temporal masks. It does not require
an optical-flow dependency. The quarter and half scales preserve tiny-ball detail. See [`docs/模型升级方案.md`](docs/模型升级方案.md) for the design rationale,
constraints, and literature links.

> [!IMPORTANT]
> **Access**: The Obsidian repository is a private resource. For in-depth development, architectural study, or technical exchange, please contact the author via **Email** to request authorization.

---
## License
This SDK is proprietary software. All rights reserved by Shanghai Code Zero Sports Technology Co., Ltd. The source code is provided for technical exchange and academic study only.

© 2025 Shanghai Code Zero Sports Technology Co., Ltd.

---
