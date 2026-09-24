# MotionPosterior

本仓库是 **MotionPosterior** 专用软件开发工具包（SDK），提供面向高速小目标的运动感知后验估计与轨迹追踪实现。当前基准任务是网球追踪，但公开架构不限定于 rally 或体育场景。由 **上海代号零体育科技有限公司** 开发并持有。

最终模型名称为 `MotionPosteriorNet`。`TrackNetV2` 和 `TrackNetV5` 继续保留，作为已有脚本和权重的历史基线。

TrackNetV5 的核心架构与算法逻辑基于公司最新研究成果：
- 论文标题: TrackNetV5: Residual-Driven Spatio-Temporal Refinement and Motion Direction Decoupling for Fast Object Tracking
- 论文地址: [arXiv:2512.02789](https://arxiv.org/abs/2512.02789)

## 核心规格

* **架构支持**：支持公开的 `MotionPosteriorNet` 三帧/五帧路径，同时提供 TrackNetV2、官方 TrackNetV3、PyTorch 版 TrackNetV4 TypeA/TypeB 和 TrackNetV5 基线。
* **功能集成**：封装了可配置的三帧/五帧时序推理、高斯热力图质心提取、轨迹增强可视化及训练流水线。
* **保密声明**：模型权重与训练数据集属于公司内部核心资产，暂不公开。

---

## 1. 环境配置 (Environment)

本项目针对特定计算环境进行了深度优化，请确保依赖版本一致以维持系统稳定性。

### 核心依赖

| 组件 | 推荐版本 |
| --- | --- |
| **Python** | 3.10 |
| **CUDA** | 12.6 |
| **PyTorch** | 2.9.0+cu126 |
| **Torchvision** | 0.24.0+cu126 |

### 安装步骤

```bash
# 1. 安装基础科学计算与图像处理库
pip install -r requirements.txt

# 2. 手动安装指定版本的 Torch 生态,建议直接去官网找对应版本下载
```

---

## 2. 数据准备 (Data Preparation)

本 SDK 采用 **高斯热力图（Gaussian Heatmap）** 作为监督信号。

### 数据集标准

自定义数据集的准备请严格参考以下标注规范：

* **参考仓库**：`WASB-TrainingOK` 数据集规范。

### 预处理脚本

使用 `tools/preprocess_data_gauss.py` 将原始视频帧与 `Label.csv` 转换为模型所需的时空上下文关联张量。

```bash
python tools/preprocess_data_gauss.py \
    --input_dir <原始数据路径> \
    --output_dir <预处理输出路径> \
    --mode context \
    --num-frames 3 \
    --train_rate 0.8 \
    --height 1080 --width 1920

```

* **关键参数**：
* `--mode`: 使用 `context` 生成中心时序窗口。
* `--num-frames`: `3` 保持旧 CSV 契约；`5` 生成 `labels_context5_train.csv` 与 `labels_context5_val.csv`。
* `--size & --variance`: 控制生成高斯斑点的半径与方差。



---

## 3. 训练指南 (Training)

训练任务采用 **工厂模式（Factory Pattern）** 动态构建，由 `train.py` 统一调度。

### 启动训练队列

```bash
python train.py

```

### 操作流程

1. **自动扫描**：系统将列出 `./configs/` 目录下所有 `.py` 配置文件。
2. **序号选择**：输入配置序号（支持空格分隔的多任务队列，如 `1 3 5`）。
3. **引擎运行**：`Runner` 指挥官将自动执行实例化、学习率预热（Warmup）、梯度裁剪（GradClip）及 Hook 插件挂载。

---

## 4. 推理流水线 (Inference)

推理模块支持批量视频处理及结构化数据导出。

### 执行命令

```bash
python track.py <input_dir> <weights_path> \
  --arch motion_posterior5 \
  --output-dir <trajectory_csv_dir> \
  --threshold 0.5 \
  --device cuda:0
```

### 输出产物说明

`trajectory_csv_dir` 中每个视频生成一个同名 CSV，例如
`20260706_001.mp4` 对应 `20260706_001.csv`。字段固定为：

```text
benchmark_id,video_name,frame_number,detected,x_512,y_288,x_orig,y_orig,conf,fps,width,height
```

帧号从 0 开始，CSV 保证每个实际解码帧一行；`x_512/y_288` 是模型空间坐标，
`x_orig/y_orig` 是按视频元数据换算后的原分辨率坐标。TrackNetV5 的模型输入固定为
`512×288`，因此输入视频必须是 16:9。检测统计只打印到终端，不混入 CSV。

默认不生成可视化视频；需要时增加
`--visualization-dir <visualization_dir>`。

---

## 5. 架构讲解与资源获取

### 已恢复的开源基线

推理入口包含 `v2`、官方 `v3`、`v4`/`v4_typea`、`v4_typeb` 以及本项目的 `v5` 基线。
V3 另外注册了独立的 `InpaintNetV3` 轨迹修正器。V4 TypeA/TypeB 根据官方
[TrackNetV4 仓库](https://github.com/TrackNetV4/TrackNetV4) 的结构用 PyTorch 重写，
不引入 TensorFlow 运行依赖。当前基线已通过结构和前向 smoke test；要比较精度仍需
分别加载匹配权重，并使用统一数据划分、阈值和指标。

### V5 motion-aware 升级

当前 V5 保留三帧 RGB、MDD 四通道亮暗差分、三帧热图输出和原有推理接口，同时在模型内部增加两层运动建模：

* `MotionConditionedGate` 将 MDD 的 signed motion map 缩放到 `skip1/skip2/skip3/bottleneck` 四个尺度，联合生成空间门控和通道门控，让运动提示参与高分辨率细节和低分辨率语义。
* `R_STRHead` 将差分图 patch 化为 motion tokens，与三帧 draft tokens 一起进入已有 Transformer。motion token 只作为上下文，不改变三帧输出的 token 数量、热图尺寸或损失接口。

这次升级没有引入光流、DCN 或额外依赖。对于小球这类目标，优先保留短期邻帧的局部差分，避免把背景运动直接当成目标位移；后续若有数据和算力，再考虑局部对齐或短时记忆。

设计参考：

* [LSTFE-Net, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Xiao_LSTFE-NetLong_Short-Term_Feature_Enhancement_Network_for_Video_Small_Object_Detection_CVPR_2023_paper.pdf)：短期邻帧与长期上下文分层融合，用于视频小目标。
* [Mutual Information-Based Temporal Difference Learning, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Feng_Mutual_Information-Based_Temporal_Difference_Learning_for_Human_Pose_Estimation_in_CVPR_2023_paper.pdf)：逐级时间差分编码和运动表示解耦。
* [Look Back and Forth, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Isobe_Look_Back_and_Forth_Video_Super-Resolution_With_Explicit_Temporal_Difference_CVPR2022_paper.pdf)：显式建模前后帧差异和残差细化。
* [BasicVSR++, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Chan_BasicVSR_Improving_Video_Super-Resolution_With_Enhanced_Propagation_and_Alignment_CVPR_2022_paper.html)：更重的传播与对齐方向，暂不直接移植到三帧热图任务。

本仓库的工程设计模式、模型细节及底层逻辑已整理至专属的 **Obsidian 可视化知识库**。

### MotionPosterior 升级架构

最终的 `MotionPosteriorNet` 支持：

* 输入 `[B, 9, H, W]`（三帧）或 `[B, 15, H, W]`（五帧）；
* 输出与输入帧数相同的全分辨率热力图；
* ConvNeXt V2 思路的分层骨干（GELU、GRN、深度卷积残差块）；
* 在 1/2、1/4、1/8 尺度先做可学习全局平移补偿，再执行 3×3 局部跨帧相关和密集 offset 精修；
* 运动模型额外预测中心 offset、visibility、uncertainty，并用速度/加速度一致性损失训练；
* 保留高分辨率分支，避免高速小球在深层下采样中消失。

五帧训练与推理：

```bash
python tools/preprocess_data_gauss.py --input_dir <raw> --output_dir <data> \
  --mode context --num-frames 5 --train_rate 0.8
python train.py  # 选择 configs/motionposterior_convnext_5frames.py
python track.py <input_dir> <weights_path> --arch motion_posterior5 \
  --output-dir <trajectory_csv_dir> --threshold 0.5 --device cuda:0
```

若要训练严格在线版本，将预处理参数改为 `--window-type causal`，并选择
`configs/motionposterior_convnext_5frames_causal.py`；该版本输入为
`[t-4,t-3,t-2,t-1,t]`，不会读未来帧。

`motion_posterior3/motion_posterior5` 默认使用 `chunk`，每个窗口输入/输出帧数一致。需要中心滑动或实时因果评估时分别使用 `--window-mode center` / `--window-mode causal`。训练保存完整 checkpoint（模型、优化器、调度器和进度），配置 `resume_from` 可续训。

完整设计取舍、边界条件与顶会文献链接见 [`docs/MotionPosterior.md`](docs/MotionPosterior.md)。

> [!IMPORTANT]
> **获取途径**：该 Obsidian 仓库属于非公开资源。如有深度开发、架构学习或技术交流需求，请通过 **Email** 联系作者申请授权。

---
## 许可证
本 SDK 属于公司私有软件。上海代号零体育科技有限公司保留所有权利。源码仅供技术交流与学术学习使用。

© 2025 上海代号零体育科技有限公司 | Shanghai Code Zero Sports Technology Co., Ltd.

---
