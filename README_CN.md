# TrackNetV5 SDK Documentation

本仓库是 **TrackNetV5** 专用软件开发工具包（SDK），提供网球追踪算法的标准工程化实现。由 **上海代号零体育科技有限公司** 开发并持有。

TrackNetV5 的核心架构与算法逻辑基于公司最新研究成果：
- 论文标题: TrackNetV5: Residual-Driven Spatio-Temporal Refinement and Motion Direction Decoupling for Fast Object Tracking
- 论文地址: [arXiv:2512.02789](https://arxiv.org/abs/2512.02789)

## 核心规格

* **架构支持**：支持 TrackNetV5 与 V2；未完成注册的 V4 不作为公开推理选项。
* **功能集成**：封装了三帧滑动窗口推理、高斯热力图质心提取、轨迹增强可视化及工业级训练流水线。
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

使用 `tools/preprocess_data_gauss.py` 将原始视频帧与 `Label.csv` 转换为 V5 架构所需的时空上下文关联张量。

```bash
python tools/preprocess_data_gauss.py \
    --input_dir <原始数据路径> \
    --output_dir <预处理输出路径> \
    --mode context \
    --train_rate 0.8 \
    --height 1080 --width 1920

```

* **关键参数**：
* `--mode`: 必须指定为 `context`（生成三帧滑动推理所需的关联索引）。
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
  --arch v5 \
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

本仓库的工程设计模式、模型细节及底层逻辑已整理至专属的 **Obsidian 可视化知识库**。

> [!IMPORTANT]
> **获取途径**：该 Obsidian 仓库属于非公开资源。如有深度开发、架构学习或技术交流需求，请通过 **Email** 联系作者申请授权。

---
## 许可证
本 SDK 属于公司私有软件。上海代号零体育科技有限公司保留所有权利。源码仅供技术交流与学术学习使用。

© 2025 上海代号零体育科技有限公司 | Shanghai Code Zero Sports Technology Co., Ltd.

---
