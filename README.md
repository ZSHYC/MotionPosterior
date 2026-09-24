# MotionPosterior: Motion-Conditioned Posterior Estimation for Fast Small-Object Tracking

**PyTorch implementation of MotionPosterior, an iterative extension of TrackNetV5 for motion-conditioned posterior estimation of fast-moving small objects.**

This repository contains the model implementation, training pipeline, preprocessing tools, and evaluation interfaces used to study dense heatmap localization with explicit motion and geometric uncertainty. The proposed model preserves the efficient three-frame formulation of TrackNetV5 and extends it to symmetric and causal five-frame inference.

## Abstract

TrackNetV5 combines signed motion-direction cues with residual spatio-temporal refinement for fast object tracking. MotionPosterior builds on that formulation by replacing image-difference-only interaction with feature-level temporal registration and local correspondence. A hierarchical ConvNeXt V2-style encoder produces high-resolution and contextual features at four scales. Confidence-gated global translation compensation, velocity-conditioned local correlation, and dense offset refinement provide a motion-conditioned representation. The decoder predicts a heatmap posterior together with center offset, visibility, localization uncertainty, velocity, and acceleration. The same architecture supports three-frame and five-frame windows while preserving dense per-frame outputs.

## Method

```text
RGB window (T = 3 or 5)
        │
        ▼
Shared hierarchical encoder
full ─ half ─ quarter ─ eighth features
        │
        ├── confidence-gated global translation compensation
        ├── velocity/uncertainty-conditioned local correlation
        ├── temporal-valid masking and motion residual gating
        └── dense offset refinement
        │
        ▼
Multi-scale decoder with high-resolution skip connections
        │
        ├── heatmap logits
        ├── center offset
        ├── visibility logits
        ├── localization uncertainty
        ├── velocity
        └── acceleration
```

The deployment decoder consumes the outputs in the following order:

```text
heatmap candidate → offset refinement → visibility gate → uncertainty
```

Detailed tensor contracts, supervision, and module definitions are documented in
[`docs/MotionPosterior.md`](docs/MotionPosterior.md).

## Temporal protocols

| Configuration | Input window | Output |
| --- | --- | --- |
| `motion_posterior3` | `[t-1, t, t+1]` | 3 heatmaps and posterior heads |
| `motion_posterior5` | `[t-2, t-1, t, t+1, t+2]` | 5 heatmaps and posterior heads |
| `motion_posterior5_causal` | `[t-4, t-3, t-2, t-1, t]` | 5 heatmaps and posterior heads |

Centered and causal protocols are evaluated independently. `chunk` inference is available when an equal-length input/output window is required by an existing pipeline.

## Installation

Python 3.10 is recommended. Install a PyTorch/torchvision build compatible with the local CUDA runtime, then install the remaining dependencies:

```bash
pip install torch torchvision
pip install -r requirements.txt
```

The repository does not require a third-party optical-flow package. The complete dependency list is in [`requirements.txt`](requirements.txt).

## Data preparation

The preprocessing script converts frame directories and `Label.csv` annotations into soft Gaussian heatmaps and grouped train/validation CSV files. Train/validation splitting is performed at clip level.

Three-frame context:

```bash
python tools/preprocess_data_gauss.py \
  --input_dir <raw_data> \
  --output_dir <processed_data> \
  --mode context \
  --num-frames 3 \
  --train_rate 0.8
```

Five-frame centered context:

```bash
python tools/preprocess_data_gauss.py \
  --input_dir <raw_data> \
  --output_dir <processed_data> \
  --mode context \
  --num-frames 5 \
  --window-type center \
  --train_rate 0.8
```

Five-frame causal context:

```bash
python tools/preprocess_data_gauss.py \
  --input_dir <raw_data> \
  --output_dir <processed_data> \
  --mode context \
  --num-frames 5 \
  --window-type causal \
  --train_rate 0.8
```

The centered five-frame split is written as `labels_context5_{train,val}.csv`; the causal split is written as `labels_causal5_{train,val}.csv`.

## Training

Training is configured through the files under [`configs/`](configs/). The main experiments are:

* `motionposterior_convnext_3frames.py`
* `motionposterior_convnext_5frames.py`
* `motionposterior_convnext_5frames_causal.py`

Select a configuration from the training launcher:

```bash
python train.py
```

The training objective combines heatmap supervision with visibility-masked geometric terms for center offset, heteroscedastic uncertainty, velocity, and acceleration. When available, real coordinate and visibility annotations are used directly. Checkpoints store the model, optimizer, scheduler, progress state, configuration, and random-number-generator state.

## Inference

The batch inference interface accepts a model checkpoint and a directory of videos. The current pipeline resizes frames to `512 × 288` and writes one CSV per video.

Offline five-frame inference:

```bash
python track.py <video_dir> <checkpoint.pth> \
  --arch motion_posterior5 \
  --window-mode center \
  --output-dir <output_dir> \
  --threshold 0.5 \
  --device cuda:0
```

Online causal inference:

```bash
python track.py <video_dir> <checkpoint.pth> \
  --arch motion_posterior5_causal \
  --window-mode causal \
  --output-dir <output_dir> \
  --threshold 0.5 \
  --device cuda:0
```

The canonical output schema is:

```text
benchmark_id,video_name,frame_number,detected,x_512,y_288,x_orig,y_orig,conf,fps,width,height
```

Use `--visualization-dir <dir>` to additionally write trajectory and heatmap videos.

## Evaluation protocol

For a reproducible comparison, keep the video-level split, input resolution, Gaussian target parameters, threshold, and temporal protocol fixed. Report at least:

* PCK/EPE and pixel localization error;
* visible-target recall, miss rate, false-positive rate, and visibility recall;
* P50/P90/P95 localization error;
* throughput and end-to-end latency.

Three-frame centered, five-frame centered, five-frame causal, and `chunk` results should be reported separately. Structural smoke tests and paper-reported numbers do not replace retraining and evaluation on the target split.

## Baseline and implementation references

The repository includes a PyTorch TrackNetV5 configuration for direct comparison with the predecessor architecture. MotionPosterior is the forward model developed from that baseline. The recovered TrackNetV3 and TrackNetV4 implementations are provided as additional architectural references.

* [TrackNetV5: Residual-Driven Spatio-Temporal Refinement and Motion Direction Decoupling](https://arxiv.org/abs/2512.02789)
* [TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps](https://arxiv.org/abs/2409.14543)
* [TrackNetV3 repository](https://github.com/qaz812345/TrackNetV3)
* [ConvNeXt V2](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html)
* [LSTFE-Net](https://openaccess.thecvf.com/content/CVPR2023/html/Xiao_LSTFE-NetLong_Short-Term_Feature_Enhancement_Network_for_Video_Small_Object_Detection_CVPR_2023_paper.html)

## Citation

If you use this code, please cite the predecessor work and the corresponding MotionPosterior paper when available:

```bibtex
@article{tang2025tracknetv5,
  title   = {TrackNetV5: Residual-Driven Spatio-Temporal Refinement and Motion Direction Decoupling for Fast Object Tracking},
  author  = {Tang, Haonan and Chen, Yanjun and Jiang, Lezhi},
  journal = {arXiv preprint arXiv:2512.02789},
  year    = {2025}
}
```

## License

The code, model weights, and data are distributed under the terms specified by the project owner. Training data and pretrained checkpoints are not included in this repository.
