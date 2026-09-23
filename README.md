# TrackNetV5 SDK Documentation

**[中文版 (Chinese Version)](README_CN.md)**

This repository is the official Software Development Kit (SDK) for **TrackNetV5**, providing a standardized engineering implementation of the tennis ball tracking algorithm. Developed and maintained by **Shanghai Code Zero Sports Technology Co., Ltd.**

The core architecture and algorithmic logic of TrackNetV5 are based on our latest research:

* **Title**: *TrackNetV5: Residual-Driven Spatio-Temporal Refinement and Motion Direction Decoupling for Fast Object Tracking*
* **Paper**: [arXiv:2512.02789](https://arxiv.org/abs/2512.02789)

## Core Specifications

* **Architecture Support**: Supports TrackNetV5 and V2. The unregistered V4 implementation is not exposed as an inference option.
* **Integrated Features**: Encapsulates three-frame sliding window inference, Gaussian heatmap centroid extraction, trajectory enhancement visualization, and an industrial-grade training pipeline.
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

Use `tools/preprocess_data_gauss.py` to convert raw video frames and `Label.csv` into the spatio-temporal context tensors required by the V5 architecture.

```bash
python tools/preprocess_data_gauss.py \
    --input_dir <path_to_raw_data> \
    --output_dir <path_to_output> \
    --mode context \
    --train_rate 0.8 \
    --height 1080 --width 1920

```

* **Key Parameters**:
* `--mode`: Must be set to `context` (generates associative indices for the three-frame sliding window inference).
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
  --arch v5 \
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

### Motion-aware V5 upgrade

The V5 path keeps its three RGB frames, four-channel MDD prompt, three heatmap outputs, and inference contract. It now adds two internal motion paths:

* `MotionConditionedGate` conditions `skip1`, `skip2`, `skip3`, and `bottleneck` on the signed MDD maps with spatial and channel gates.
* `R_STRHead` patchifies the motion maps into context tokens and lets the existing temporal Transformer attend to them without changing the three-frame output shape or loss interface.

The upgrade deliberately avoids optical flow, deformable convolution, and new dependencies. It follows the short-term temporal-difference direction used in [LSTFE-Net (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Xiao_LSTFE-NetLong_Short-Term_Feature_Enhancement_Network_for_Video_Small_Object_Detection_CVPR_2023_paper.pdf), [Temporal Difference Learning (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Feng_Mutual_Information-Based_Temporal_Difference_Learning_for_Human_Pose_Estimation_in_CVPR_2023_paper.pdf), and [Look Back and Forth (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Isobe_Look_Back_and_Forth_Video_Super-Resolution_With_Explicit_Temporal_Difference_CVPR2022_paper.pdf). BasicVSR++ is kept as a future, heavier alignment direction rather than a direct dependency.

The engineering design patterns, TrackNetV5 model details, and underlying inference logic are documented in our exclusive **Obsidian Visual Knowledge Base**.

> [!IMPORTANT]
> **Access**: The Obsidian repository is a private resource. For in-depth development, architectural study, or technical exchange, please contact the author via **Email** to request authorization.

---
## License
This SDK is proprietary software. All rights reserved by Shanghai Code Zero Sports Technology Co., Ltd. The source code is provided for technical exchange and academic study only.

© 2025 Shanghai Code Zero Sports Technology Co., Ltd.

---
