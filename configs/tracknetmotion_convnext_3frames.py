"""Three-frame TrackNetMotion configuration for an apples-to-apples V5 upgrade."""

from pathlib import Path

input_size = (288, 512)
original_size = (1080, 1920)
data_root = './data/benchmark'
num_frames = 3
frame_keys = ['path_prev', 'path', 'path_next']
target_keys = ['gt_path_prev', 'gt_path', 'gt_path_next']

model = dict(
    type='TrackNetMotion',
    num_frames=num_frames,
    return_aux=True,
    backbone=dict(type='MotionConvNeXtBackbone', num_frames=num_frames),
)

pipeline = [
    dict(type='LoadMultiImagesFromPaths', to_rgb=True),
    dict(type='Resize', keys=frame_keys, size=input_size),
    dict(type='ConcatChannels', keys=frame_keys, output_key='image'),
    dict(type='LoadAndFormatMultiTargets', keys=target_keys, output_key='target'),
    dict(type='Finalize', image_key='image', final_keys=['image', 'target', 'coords', 'visibility', 'original_info']),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(type='TennisDataset', data_dir=data_root,
               csv_path=f'{data_root}/labels_context_train.csv',
               input_height=input_size[0], input_width=input_size[1], pipeline=pipeline),
    val=dict(type='TennisDataset', data_dir=data_root,
             csv_path=f'{data_root}/labels_context_val.csv',
             input_height=input_size[0], input_width=input_size[1], pipeline=pipeline),
)

loss = dict(type='TrackNetV2Loss', aux_weight=0.25, offset_weight=0.2,
            visibility_weight=0.1, uncertainty_weight=0.05, trajectory_weight=0.05)
optimizer = dict(type='AdamW', lr=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
lr_config = dict(policy='Step', step=[20, 25], gamma=0.1)
evaluation = dict(interval=1, metric=dict(type='TrackNetV2Metric', min_dist=4, original_size=original_size))
total_epochs = 30
work_dir = f'./workdirs/{Path(__file__).stem}'
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
seed = 42
deterministic = True
resume_from = None
