# MotionPosterior 主线模型

本文是仓库唯一的模型设计文档，描述当前主线模型 `MotionPosteriorNet` 的架构、输入输出契约、运动表示、训练监督和推理方式。数据集目前以网球为主，但模型不依赖 `rally` 命名，也不把方法限制在体育场景；目标是高速、小尺寸、短时序运动点目标。

## 1. 模型定位与公开入口

`MotionPosteriorNet` 是最终模型。`TrackNetV2`、`TrackNetV3`、`TrackNetV4` 和 `TrackNetV5` 只作为可构建的历史/对比基线保留；仓库不再维护中间模型 `TrackNetMotion`。

| CLI 架构 | 输入窗口 | 输出 | 用途 |
| --- | --- | --- | --- |
| `motion_posterior3` | `[t-1,t,t+1]` | 3 张热力图及后验量 | 三帧离线基线 |
| `motion_posterior5` | `[t-2,t-1,t,t+1,t+2]` | 5 张热力图及后验量 | 五帧离线主线 |
| `motion_posterior5_causal` | `[t-4,t-3,t-2,t-1,t]` | 5 张热力图及后验量 | 五帧在线版本 |

三种协议必须分别评估。中心窗口用于离线精度，因果窗口只读过去帧；`chunk` 模式保持输入和输出都是 T 帧，`center`/`causal` 模式用于只评价中心帧的滑动窗口。

## 2. 总体数据流

```text
T 帧 RGB (T=3 或 5)
        │ 共享帧编码器
        ▼
多尺度空间特征：full / half / quarter / eighth
        │
        ├─ 相机运动候选与置信度
        ├─ 速度/不确定度条件的局部跨帧相关
        ├─ 有效邻帧 mask 与运动残差门控
        └─ dense offset refinement
        │
        ▼
逐帧 U-Net 式解码 + 高分辨率跳连
        │
        ├─ heatmap logits
        ├─ center offset
        ├─ visibility logits
        ├─ log variance / uncertainty
        ├─ velocity
        └─ acceleration
```

主线模型的目标不是把所有帧压成一个分类 token，而是为每个输入帧保留一个完整的空间后验。这样热力图负责候选发现，几何头负责亚像素修正，visibility 和 uncertainty 负责在模糊、遮挡或出界时控制解码风险。

## 3. Backbone：MotionConvNeXtBackbone

### 3.1 分层编码

每帧使用共享参数的 `MotionConvNeXtBackbone`，输出四个尺度：

* `full`：保留小球边缘和运动模糊的高频细节；
* `half`：主预测分支，兼顾定位精度与感受野；
* `quarter`、`eighth`：建模球场、球员和相机背景的长程上下文。

基本块采用 ConvNeXt V2 风格的 depthwise convolution、LayerNorm、GELU、pointwise expansion 和 GRN（Global Response Normalization）。高分辨率特征通过跳连回到输出端，避免小目标在连续下采样后只剩一个不稳定峰值。

### 3.2 为什么不用单一旧式 VGG 路径

V2/V3 的热力图 U-Net 适合作为可解释基线，但其感受野和跨帧交互有限；V5 的运动提示仍主要来自相邻图像差分。主线模型将空间编码、运动对应和不确定度解耦，既保留 dense prediction 的像素精度，又为高速位移提供显式条件。

## 4. Motion：从帧差到后验运动表示

### 4.1 相机补偿与有效性

在 `half/quarter/eighth` 尺度，邻帧先生成有界的全局平移候选及置信度，再通过可微 `grid_sample` 得到对齐特征。补偿只在置信度足够时参与融合，防止把相机摇摄、球员挥拍或运动模糊误当成球的位移。首尾帧缺少邻居时使用显式 temporal-valid mask，不用循环索引伪造邻帧。

当前实现使用平移级补偿；若数据出现明显透视变化，应以同一评估协议比较 homography/ECC 或几何相机 token，不能直接假设光流始终可靠。

### 4.2 局部跨帧相关

对每个尺度的中心特征 `F_t` 和对齐邻帧 `F_j`，在局部窗口计算相关体：

```text
C_t,j(δ) = < norm(F_t(x)), norm(F_j(x + δ)) >,
δ ∈ N_r,  r = 1, 2, 4  (由尺度决定)
```

相关聚合之后预测 dense 2D offset，完成一次局部精修。搜索半径由速度代理和预测不确定度连续放大，而不是整个视频使用一个固定半径；这对高速球的跨帧大位移更合适，也避免全图全局 attention 的显存开销。

### 4.3 Motion residual 与门控

对齐特征与中心特征的绝对差形成 `motion residual`，经过深度卷积和通道门控后调制对齐特征。最终融合同时包含：

1. 原始 RGB 外观，防止运动模糊时只剩差分噪声；
2. 对齐后的邻帧证据，提供短期轨迹上下文；
3. 局部相关峰和 offset，表达候选位移；
4. residual/confidence，表达“哪里在动”和“这次对应是否可信”。

整个模块只依赖 PyTorch，训练时端到端反向传播，不依赖预计算光流或第三方光流网络。它吸收了 feature-level temporal difference、局部稀疏对应和小目标局部相关体的思想，同时保留本项目的轻量 T=3/T=5 契约。

## 5. 解码器与输出后验

逐帧 decoder 在 `eighth → quarter → half → full` 路径上逐级上采样，并融合同尺度 skip feature。输出头为共享结构、逐帧预测：

* `heatmap_logits[T, H, W]`：目标存在的空间分布；
* `offset[T, 2, H, W]`：归一化中心残差，约束在稳定范围；
* `visibility_logits[T]`：该帧是否有可靠目标；
* `log_variance[T, 2]`：x/y 定位方差，表示后验不确定度；
* `velocity[T, 2]`、`acceleration[T, 2]`：用于短期运动一致性和下游滤波。

部署解码顺序固定为：

```text
heatmap candidate → offset refinement → visibility gate → uncertainty
```

低 visibility 或高不确定度时允许 abstain，避免把背景峰值强行写入轨迹。旧的阈值加最大轮廓解码仍可用于 V2/V3/V5 基线和仅返回 tensor 的旧权重，但不代表主线模型的完整能力。

## 6. 训练监督与坐标契约

训练优先使用数据集提供的真实 `coords[T,2]` 和 `visibility[T]`，而不是从量化高斯热力图反推中心。`Resize` 必须同步缩放坐标；缺少旧 metadata 时才回退到热力图质量中心。

总损失由以下项组成：

```text
L = L_heatmap + λ_pos L_pos + λ_vis L_vis
    + λ_unc L_heteroscedastic + λ_v L_velocity + λ_a L_acceleration
```

* `L_heatmap` 在 logits 上使用稳定 BCE/Focal 形式；
* `L_pos` 监督 offset 或中心坐标；
* `L_heteroscedastic` 用预测方差调节定位残差；
* visibility 为 0 的帧不计入位置、方差和运动损失；
* velocity/acceleration 只在相邻帧均有效时计算，并使用 dataloader 提供的 `dt`，否则回退到 `1/fps` 或单位帧间隔；
* 加速度项不能强迫击球、落地等 change point 满足匀速假设，应使用鲁棒权重并允许可见性门控。

## 7. 五帧数据与推理

```bash
# 离线五帧
python tools/preprocess_data_gauss.py --input_dir <raw> --output_dir <data> \
  --mode context --num-frames 5 --train_rate 0.8
python track.py <input_dir> <weights_path> --arch motion_posterior5 \
  --output-dir <trajectory_csv_dir> --threshold 0.5 --device cuda:0

# 严格在线五帧
python tools/preprocess_data_gauss.py --input_dir <raw> --output_dir <data> \
  --mode context --num-frames 5 --window-type causal --train_rate 0.8
```

对应训练配置为：

* `configs/motionposterior_convnext_3frames.py`；
* `configs/motionposterior_convnext_5frames.py`；
* `configs/motionposterior_convnext_5frames_causal.py`。

边界窗口只复制边界帧，不把不存在的帧写入轨迹 CSV。五帧中心、五帧因果和等长 `chunk` 必须分开报告，不能把输出五帧的模型与只输出中心帧的模型直接混算。

## 8. 基线恢复矩阵

| 入口 | 结构 | 在本项目中的定位 |
| --- | --- | --- |
| `v2` | TrackNetV2 三帧热力图 U-Net | 轻量历史基线 |
| `v3` | 官方 TrackNetV3 heatmap stage；`InpaintNetV3` 独立注册 | 官方 V3 对比；CLI 默认 heatmap stage |
| `v4` / `v4_typea` | PyTorch MotionPrompt + TypeA Fusion | 官方 V4 TypeA 思路的无 TensorFlow 实现 |
| `v4_typeb` | PyTorch MotionPrompt + TypeB Fusion | 官方 V4 TypeB 对比 |
| `v5` | MDD + R-STR | 现有研究基线 |
| `motion_posterior3/5` | ConvNeXt V2 风格骨干 + 局部相关 + 后验头 | 最终主线模型 |

官方结构参考：[TrackNetV3](https://github.com/qaz812345/TrackNetV3) 及其[论文](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf)，[TrackNetV4](https://github.com/TrackNetV4/TrackNetV4) 的[结构定义](https://github.com/TrackNetV4/TrackNetV4/blob/main/src/models/TrackNetV4.py)和 [arXiv 论文](https://arxiv.org/abs/2409.14543)。相关设计依据包括 [TrackNet](https://arxiv.org/abs/1907.03698)、[LSTFE-Net](https://openaccess.thecvf.com/content/CVPR2023/html/Xiao_LSTFE-NetLong_Short-Term_Feature_Enhancement_Network_for_Video_Small_Object_Detection_CVPR_2023_paper.html)、[TEA](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_TEA_Temporal_Excitation_and_Aggregation_for_Action_Recognition_CVPR_2020_paper.html)、[PSLA](https://openaccess.thecvf.com/content_ICCV_2019/html/Guo_Progressive_Sparse_Local_Attention_for_Video_Object_Detection_ICCV_2019_paper.html)、[ConvNeXt V2](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html)、[QueryDet](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html) 和 [TAPIR](https://openaccess.thecvf.com/content/ICCV2023/html/Doersch_TAPIR_Tracking_Any_Point_with_Per-Frame_Initialization_and_Temporal_Refinement_ICCV_2023_paper.html)。这些论文是设计参考，不代表本仓库已复现其全部实验或权重。

## 9. 实验与复现边界

正式比较应固定视频级划分、预处理尺寸、阈值和窗口协议，至少报告可见球召回、miss rate、误检率、PCK/EPE、visibility recall、P50/P90/P95 像素误差和端到端延迟。训练 checkpoint 保存模型、优化器、调度器、进度、配置以及 Python/NumPy/PyTorch/CUDA RNG 状态，以便恢复同一实验阶段。

当前代码和测试验证的是张量形状、Motion 梯度、后验解码、坐标缩放、visibility mask、三/五帧入口和基线构建契约；它们不等价于真实视频精度提升。任何论文级结论都必须在相同数据划分上重新训练并报告完整指标。
