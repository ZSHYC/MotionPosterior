"""
config for tracknetv5 with b2e30 and adamw lr1e-4
"""

from pathlib import Path

# ------------------- 1. 模型定义 (Model) -------------------
model = dict(
    type='TrackNetV5',
    backbone=dict(
        type='TrackNetV2Backbone', # OK
        in_channels=13,
        motion_channels=4,
        use_motion_gates=True
    ),
    neck=dict(
        type='TrackNetV2Neck'# OK
    ),
    head=dict( 
        type='R_STRHead',
        in_channels=64,
        out_channels=3,
        motion_channels=4,
        use_motion_tokens=True
    )
)

# ------------------- 2. 数据定义 (Data) -------------------
# --- 2.1 通用参数 ---
input_size = (288, 512)  # (height, width)
original_size = (1080, 1920) # 原图片大小(height, width)
# ‼️ 请务必将此路径修改为您自己电脑上的正确路径
data_root = './data/benchmark'

# --- 2.2 数据处理流水线定义 ---
pipeline = [
    dict(type='LoadMultiImagesFromPaths', to_rgb=True),
    dict(type='Resize', keys=['path_prev', 'path', 'path_next'], size=input_size),
    # dict(type='GenerateMotionAttention', threshold=40),
    dict(type='ConcatChannels',
         keys=['path_prev', 'path', 'path_next'],
         output_key='image'),
    dict(type='LoadAndFormatMultiTargets',  # 使用新的多目标加载器
         keys=['gt_path_prev', 'gt_path', 'gt_path_next'],  # 指定三个gt路径
         output_key='target'),
    dict(type='Finalize',
         image_key='image',
         final_keys=['image', 'target', 'coords', 'visibility', 'original_info'])
]

# --- 2.3 数据加载器配置 ---
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='TennisDataset',
        data_dir=data_root,
        csv_path=f'{data_root}/labels_context_train.csv',
        input_height=input_size[0],
        input_width=input_size[1],
        pipeline=pipeline
    ),
    val=dict(
        type='TennisDataset',
        data_dir=data_root,
        csv_path=f'{data_root}/labels_context_val.csv',
        input_height=input_size[0],
        input_width=input_size[1],
        pipeline=pipeline
    )
)

# ------------------- 3. 损失函数定义 (Loss) -------------------

loss = dict(
    type='TrackNetV2Loss'
)

# ------------------- 4. 优化策略定义 (Optimization) -------------------
optimizer = dict(type='AdamW', lr=1e-4)

# (2) 定义优化器配置：添加梯度裁剪 (防止梯度爆炸)
optimizer_config = dict(
    grad_clip=dict(max_norm=1.0)
)

# (3) 定义学习率配置：Warmup + Step Decay
lr_config = dict(
    # 策略：使用 Step Decay
    policy='Step',
    # 线性预热：保证 Transformer 稳定启动
    # warmup='linear',          # 使用线性预热
    # warmup_iters=50*200,          # 预热轮数（前 50 个 epoch）
    # warmup_ratio=1e-6,        # 初始学习率 (从接近 0 开始预热)
    # 学习率衰减步长 (epoch)
    step=[20, 25],          # 在第 20 轮和第 25 轮结束时触发衰减
    # 衰减因子
    gamma=0.1                 # 每次衰减时，学习率乘以 0.1
)

# ------------------- 5. 评估策略定义 (Evaluation) -------------------
evaluation = dict(
    interval=1,
    metric=dict(
        type='TrackNetV2Metric',
        min_dist=4,
        original_size=original_size
    )
)

# ------------------- 6. 运行时定义 (Runtime) -------------------
total_epochs = 30
work_dir = f'./workdirs/{Path(__file__).stem}' # 注意自定义workdir

# ✨ 修正三：根据您的要求，添加每轮最大迭代次数
# steps_per_epoch = 200

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

custom_hooks = [
    dict(type='ValidationVisualizerV2Hook', num_samples_to_save=100, original_size=original_size)
]

seed = 42
resume_from = None
