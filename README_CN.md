# MotionPosterior

**[English README](README.md)**

本仓库是 **MotionPosterior** 的 PyTorch 实现与历史基线集合，面向高速、小尺寸、短时序运动点目标。当前基准任务是网球追踪，但模型契约不绑定具体体育项目，也不再使用旧的 `rally` 作为项目主名称。

`MotionPosteriorNet` 是最终主线模型。`TrackNetV2`、`TrackNetV3`、`TrackNetV4`、`TrackNetV5` 作为历史基线和脚本/权重兼容入口保留；`TrackNetMotion` 不作为公开模型维护。

## 模型入口

| CLI 入口 | 时序契约 | 定位 |
| --- | --- | --- |
| `motion_posterior3` | 输入 3 帧 RGB，输出 3 张热力图 | 三帧参考版本 |
| `motion_posterior5` | 输入 `[t-2,t-1,t,t+1,t+2]`，输出 5 帧 | 离线主线 |
| `motion_posterior5_causal` | 输入 `[t-4,t-3,t-2,t-1,t]`，输出 5 帧 | 在线主线 |
| `v2` | TrackNetV2 风格三帧热力图 | 轻量基线 |
| `v3` | 官方 TrackNetV3 风格三帧热力图 | 恢复的 V3 基线 |
| `v4`、`v4_typea`、`v4_typeb` | PyTorch TrackNetV4 风格三帧运动融合 | V4 对比基线 |
| `v5` | MDD + R-STR，三帧热力图 | 运动感知历史基线 |

所有入口都通过同一个 PyTorch model factory 构建。基线的前向 smoke test 只能证明结构可运行；精度比较必须使用匹配权重、相同视频级数据划分、相同预处理、阈值和指标。

## V5 motion-aware 到底还有没有用

V5 不是最终架构方向，但也没有必要删除。[TrackNetV5 论文](https://arxiv.org/abs/2512.02789)提出了 Motion Direction Decoupling（MDD）和 Residual-Driven Spatio-Temporal Refinement（R-STR）。本仓库的 V5 路径实现了这个低成本思路：用带正负极性的亮度差分图调制 V2 风格骨干，再用运动 token 为三帧精修头提供上下文。

它的价值是作为一个清晰的历史运动基线和消融参照。它的边界也很明确：固定三帧，主要依赖像素亮度差分，没有显式的相机/特征对齐，也不输出主线模型中的 `offset`、`visibility`、`uncertainty`、速度和加速度。论文中的指标不等于本仓库已经复现的结果。因此，后续模型架构投入应放在 `MotionPosteriorNet`；V5 保持稳定，用于兼容、对比和消融。

## 主线模型架构

`MotionPosteriorNet` 接受 `[B, 3T, H, W]`，其中 `T=3` 或 `T=5`，为每个输入帧输出全分辨率热力图以及后验辅助量：

* ConvNeXt V2 风格的分层骨干，输出 `full/half/quarter/eighth` 多尺度特征；
* 多尺度置信度门控全局平移补偿；
* 由速度和不确定度调节的局部跨帧相关与 dense offset 精修；
* 面向小目标细节的高分辨率跳连；
* heatmap logits、归一化中心 offset、visibility、uncertainty、velocity、acceleration。

部署解码顺序固定为：

```text
heatmap candidate → offset refinement → visibility gate → uncertainty
```

张量契约、运动表示、损失、基线恢复和文献依据统一记录在唯一的[主线模型文档](docs/MotionPosterior.md)中。

## 环境

推荐使用 Python 3.10。`requirements.txt` 没有强行锁定 PyTorch/torchvision 版本，请根据机器上的 CUDA 运行时安装匹配的 PyTorch 构建；其余依赖见 [requirements.txt](requirements.txt)。

```bash
pip install -r requirements.txt
```

## 数据准备

预处理脚本生成软高斯热力图和训练/验证 CSV，并按 clip 划分，避免相邻窗口泄漏。

三帧上下文：

```bash
python tools/preprocess_data_gauss.py \
  --input_dir <原始数据> --output_dir <预处理数据> \
  --mode context --num-frames 3 --train_rate 0.8
```

五帧离线上下文：

```bash
python tools/preprocess_data_gauss.py \
  --input_dir <原始数据> --output_dir <预处理数据> \
  --mode context --num-frames 5 --window-type center --train_rate 0.8
```

五帧因果上下文：

```bash
python tools/preprocess_data_gauss.py \
  --input_dir <原始数据> --output_dir <预处理数据> \
  --mode context --num-frames 5 --window-type causal --train_rate 0.8
```

因果版本生成 `labels_causal5_train.csv` 和 `labels_causal5_val.csv`；中心版本生成 `labels_context5_train.csv` 和 `labels_context5_val.csv`。

## 训练

`train.py` 会列出 `configs/` 下的配置文件并交互选择。主线配置为：

* `configs/motionposterior_convnext_3frames.py`
* `configs/motionposterior_convnext_5frames.py`
* `configs/motionposterior_convnext_5frames_causal.py`

```bash
python train.py
```

损失优先使用真实的 `[T,2]` 坐标和 `[T]` visibility。位置、不确定度、速度和加速度项都会按 visibility 屏蔽。checkpoint 保存模型、优化器、调度器、进度、配置和随机数状态，支持继续训练。

## 推理

批量推理为每个视频写出一个标准 CSV。当前 CLI 统一缩放到 `512×288`，要求输入视频为 16:9，并根据视频元数据把 `x_512/y_288` 换算为 `x_orig/y_orig`。

五帧离线推理：

```bash
python track.py <视频目录> <weights.pth> \
  --arch motion_posterior5 \
  --window-mode center \
  --output-dir <轨迹CSV目录> \
  --threshold 0.5 --device cuda:0
```

五帧在线推理：

```bash
python track.py <视频目录> <weights.pth> \
  --arch motion_posterior5_causal \
  --window-mode causal \
  --output-dir <轨迹CSV目录> \
  --threshold 0.5 --device cuda:0
```

如果需要历史流程的等长 T 帧输入/输出，使用 `--window-mode chunk`。只有需要轨迹和热力图视频时才增加 `--visualization-dir <目录>`。

CSV 字段为：

```text
benchmark_id,video_name,frame_number,detected,x_512,y_288,x_orig,y_orig,conf,fps,width,height
```

## 评估建议

三帧中心、五帧中心、五帧因果和 chunk 协议必须分开报告。在相同视频级划分上至少报告 PCK/EPE、可见目标召回率、miss rate、误检率、visibility recall、P50/P90/P95 像素误差和端到端延迟。结构 smoke test 或论文中的分数不能替代在本项目数据上的重新训练与评估。

## 许可证与数据

本 SDK、模型权重和训练数据属于上海代号零体育科技有限公司，仓库不包含权重和训练数据。

© 2025 Shanghai Code Zero Sports Technology Co., Ltd.
