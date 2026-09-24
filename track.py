# -*- coding: utf-8 -*-
import torch
import cv2
import numpy as np
import argparse
import math
import csv
from pathlib import Path
from tqdm import tqdm
from collections import deque
import time

# 导入你项目里的构建器和模型！
from models_factory.builder import build_model
from datasets_factory.transforms.tracknet_transforms import (
    Resize, ConcatChannels
)
from core.windowing import iter_sliding_windows
from core.postprocess import decode_heatmap, decode_prediction

# --- 1. “模型配置库” ---
MODEL_CONFIGS = {
    'v2': dict(
        type='TrackNetV2',
        backbone=dict(type='TrackNetV2Backbone', in_channels=9),
        neck=dict(type='TrackNetV2Neck'),
        head=dict(type='TrackNetV2Head', in_channels=64, out_channels=3)
    ),
    'v5': dict(
        type='TrackNetV5',
        backbone=dict(type='TrackNetV2Backbone', in_channels=13, motion_channels=4, use_motion_gates=True),
        neck=dict(type='TrackNetV2Neck'),
        head=dict(
            type='R_STRHead',
            in_channels=64,
            out_channels=3,
            motion_channels=4,
            use_motion_tokens=True,
        )
    ),
    # Motion-aware ConvNeXt-style model.  ``motion3`` preserves the usual
    # three-frame contract; ``motion5`` is the symmetric five-frame option.
    'motion3': dict(
        type='TrackNetMotion',
        num_frames=3,
        return_aux=True,
        backbone=dict(type='MotionConvNeXtBackbone', num_frames=3),
    ),
    'motion5': dict(
        type='TrackNetMotion',
        num_frames=5,
        return_aux=True,
        backbone=dict(type='MotionConvNeXtBackbone', num_frames=5),
    ),
    'motion5_causal': dict(
        type='TrackNetMotion', num_frames=5, return_aux=True,
        backbone=dict(type='MotionConvNeXtBackbone', num_frames=5),
    ),
    # Canonical public names. Legacy ``motion*`` keys remain supported.
    'motion_posterior3': dict(
        type='MotionPosteriorNet', num_frames=3, return_aux=True,
        backbone=dict(type='MotionConvNeXtBackbone', num_frames=3),
    ),
    'motion_posterior5': dict(
        type='MotionPosteriorNet', num_frames=5, return_aux=True,
        backbone=dict(type='MotionConvNeXtBackbone', num_frames=5),
    ),
    'motion_posterior5_causal': dict(
        type='MotionPosteriorNet', num_frames=5, return_aux=True,
        backbone=dict(type='MotionConvNeXtBackbone', num_frames=5),
    ),
}

INPUT_HEIGHT = 288
INPUT_WIDTH = 512
CANONICAL_FIELDS = [
    'benchmark_id',
    'video_name',
    'frame_number',
    'detected',
    'x_512',
    'y_288',
    'x_orig',
    'y_orig',
    'conf',
    'fps',
    'width',
    'height',
]


def _load_model_weights(model, weights_path, device='cpu'):
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return checkpoint


def canonical_row(sample_id, video_name, frame_number, coords, fps, width, height):
    """Build one project-compatible trajectory row."""
    detected = coords is not None
    x_model, y_model, conf = coords if detected else (None, None, 0.0)
    return {
        'benchmark_id': sample_id,
        'video_name': video_name,
        'frame_number': frame_number,
        'detected': int(detected),
        'x_512': x_model,
        'y_288': y_model,
        'x_orig': x_model * width / INPUT_WIDTH if detected else None,
        'y_orig': y_model * height / INPUT_HEIGHT if detected else None,
        'conf': conf,
        'fps': fps,
        'width': width,
        'height': height,
    }


# --- 2. 辅助函数 (✨ 已修改，与你的 Metric 脚本对齐) ---
def _heatmap_to_coords(heatmap: np.ndarray, threshold: int = 127):
    array = np.asarray(heatmap)
    scale = 255.0 if array.dtype == np.uint8 or (array.size and float(np.nanmax(array)) > 1.5) else 1.0
    decoded = decode_heatmap(array.astype(np.float32) / scale, threshold / 255.0 if scale > 1 else threshold)
    if decoded is None:
        return None
    return int(decoded[0]), int(decoded[1]), round(decoded[2], 4)

def draw_comet_tail(frame, points_deque, head_radius=8):
    """
    基于半径衰减的圆点轨迹可视化 (无连线版)
    :param frame: 当前图像帧 (BGR)
    :param points_deque: 存储坐标的队列，支持 (x, y) 或 (x, y, vis)
    :param head_radius: 最前端圆点的最大半径
    """
    # 无需创建全黑 overlay，因为我们直接在原图绘制实心圆
    # 如果需要半透明效果，可以保留 overlay 逻辑，这里采用你要求的直接绘制
    
    q_len = len(points_deque)
    if q_len == 0:
        return frame

    for i, pt in enumerate(points_deque):
        # 1. 安全检查：跳过空点或 NaN
        if pt is None:
            continue
            
        # 兼容处理：支持 (x, y) 或 (x, y, vis/conf)
        if len(pt) >= 3:
            tx, ty, tvis = pt[:3]
            if tvis == 0 or tvis is None: continue 
        else:
            tx, ty = pt
            
        if tx is None or ty is None:
            continue

        # 2. 计算半径衰减 (核心逻辑)
        # i=0 (最旧) -> scale 最小; i=q_len-1 (最新) -> scale=1.0
        scale = (i + 1) / q_len
        current_radius = int(head_radius * scale)

        # 确保半径至少为 1
        current_radius = max(1, current_radius)

        # 3. 绘制实心球
        # 使用 LINE_AA 开启抗锯齿，让圆点边缘更丝滑
        cv2.circle(
            frame, 
            (int(tx), int(ty)), 
            current_radius, 
            (0, 0, 255), # 纯红色
            -1,          # 实心
            lineType=cv2.LINE_AA
        )

    return frame


# --- 3. “核心加工车间”: ✨ process_video (✨ 已修改) ✨ ---
def process_video(video_path: Path, model, device, args, output_root_dir: Path) -> dict:
    """Run inference and write one canonical trajectory CSV."""
    threshold = float(args.threshold)
    if not math.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError(f"threshold must be a finite value in [0, 1), got {args.threshold}")
    print(f"\n🏭 Processing video: {video_path.name}")
    output_root_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    metadata_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError(f"Invalid video metadata: {video_path}")
    if width * 9 != height * 16:
        cap.release()
        raise ValueError(
            f"TrackNetV5 requires 16:9 video, got {width}x{height}: {video_path}"
        )
    resolution_str = f"{width}x{height}"

    csv_path = output_root_dir / f"{video_path.stem}.csv"
    visualization_dir = getattr(args, 'visualization_dir', None)
    writer_traj = None
    writer_comp = None
    if visualization_dir:
        video_output_dir = Path(visualization_dir) / video_path.stem
        video_output_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer_traj = cv2.VideoWriter(
            str(video_output_dir / f"{video_path.stem}_trajectory.mp4"),
            fourcc,
            fps,
            (INPUT_WIDTH, INPUT_HEIGHT),
        )
        writer_comp = cv2.VideoWriter(
            str(video_output_dir / f"{video_path.stem}_comparison.mp4"),
            fourcc,
            fps,
            (INPUT_WIDTH * 2, INPUT_HEIGHT),
        )
        if not writer_traj.isOpened() or not writer_comp.isOpened():
            cap.release()
            writer_traj.release()
            writer_comp.release()
            raise ValueError(f"Unable to open visualization writers: {video_output_dir}")

    num_frames = int(getattr(model, 'num_frames', 3))
    if num_frames not in (3, 5):
        raise ValueError(f'model.num_frames must be 3 or 5, got {num_frames}')
    frame_keys = [f'frame_{idx}' for idx in range(num_frames)]
    trajectory_points = deque(maxlen=max(1, round(fps)))
    csv_data = []
    detected_frames_count = 0
    resizer = Resize(keys=frame_keys, size=(INPUT_HEIGHT, INPUT_WIDTH))
    concatenator = ConcatChannels(keys=frame_keys, output_key='image')
    decoded_frame_count = 0
    pbar = tqdm(total=metadata_frame_count or None, desc=f"Processing {video_path.stem}")
    start_time = time.time()

    def run_window(frames, output_positions):
        nonlocal decoded_frame_count, detected_frames_count
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        data_dict = dict(zip(frame_keys, rgb_frames))
        data_dict = concatenator(resizer(data_dict))
        resized_frames = [data_dict[key] for key in frame_keys]
        image_np = data_dict['image']
        image_tensor = torch.from_numpy(
            image_np.transpose(2, 0, 1)
        ).float().div(255).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(image_tensor)
            if isinstance(prediction, dict):
                rich_prediction = prediction
                heatmaps_np = prediction['heatmap'].squeeze(0).detach().cpu().numpy()
            else:
                rich_prediction = None
                heatmaps_np = prediction.squeeze(0).cpu().numpy()
        if heatmaps_np.shape != (num_frames, INPUT_HEIGHT, INPUT_WIDTH):
            raise ValueError(
                f"Unexpected model output shape: {heatmaps_np.shape}, "
                f"expected {(num_frames, INPUT_HEIGHT, INPUT_WIDTH)}"
            )
        threshold_uint8 = int(threshold * 255)

        for output_position, frame_number in output_positions:
            single_heatmap_np = heatmaps_np[output_position]
            heatmap_uint8 = (single_heatmap_np * 255).astype(np.uint8)
            decoded = decode_prediction(
                prediction, output_position, threshold=threshold,
            ) if rich_prediction is not None else _heatmap_to_coords(
                heatmap_uint8, threshold=threshold_uint8,
            )
            coords = decoded
            if coords is not None:
                detected_frames_count += 1
                trajectory_points.append(coords)
            else:
                trajectory_points.append(None)
            csv_data.append(canonical_row(
                video_path.stem, video_path.name, frame_number, coords,
                fps, width, height,
            ))

            if writer_traj is not None and writer_comp is not None:
                frame_to_draw = cv2.cvtColor(resized_frames[output_position], cv2.COLOR_RGB2BGR)
                final_traj_frame = draw_comet_tail(frame_to_draw, trajectory_points)
                writer_traj.write(final_traj_frame)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                writer_comp.write(np.hstack((final_traj_frame, heatmap_color)))

    try:
        window_mode = getattr(args, 'window_mode', None)
        if window_mode is None:
            # Keep the T-frame input/output contract by default. Center and
            # causal modes are explicit single-frame sliding alternatives.
            window_mode = 'chunk'
        if getattr(args, 'arch', '').endswith('5_causal'):
            window_mode = 'causal'
        if window_mode in ('center', 'causal'):
            for frames, output_position, frame_number in iter_sliding_windows(cap, num_frames, window_mode):
                run_window(frames, [(output_position, frame_number)])
                decoded_frame_count += 1
                pbar.update(1)
        else:
            while cap.isOpened():
                frames = []
                for _ in range(num_frames):
                    readable, frame = cap.read()
                    if not readable:
                        break
                    frames.append(frame)
                if not frames:
                    break
                actual_frame_count = len(frames)
                while len(frames) < num_frames:
                    frames.append(frames[-1])
                run_window(
                    frames,
                    [(offset, decoded_frame_count + offset) for offset in range(actual_frame_count)],
                )
                decoded_frame_count += actual_frame_count
                pbar.update(actual_frame_count)
                if actual_frame_count < num_frames:
                    break
    finally:
        pbar.close()
        cap.release()
        if writer_traj is not None:
            writer_traj.release()
        if writer_comp is not None:
            writer_comp.release()

    total_duration = time.time() - start_time
    processing_fps = decoded_frame_count / total_duration if total_duration > 0 else 0
    print(
        f"⏱️  Processed {decoded_frame_count} frames of {resolution_str} "
        f"in {total_duration:.2f} seconds. Avg: {processing_fps:.2f} frames/sec."
    )

    detection_ratio = (
        detected_frames_count / decoded_frame_count if decoded_frame_count > 0 else 0
    )
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(csv_data)
    print(f"✅ Finished processing. Trajectory saved to: {csv_path}")

    stats = {
        'video_name': video_path.name,
        'detected_frames': detected_frames_count,
        'total_frames': decoded_frame_count,
        'detection_ratio': round(detection_ratio, 4)
    }
    return stats


# --- 4. “总调度室”: ✨ main (✨ 已修改) ✨ ---
def build_parser():
    parser = argparse.ArgumentParser(description="MotionPosterior batch video inference pipeline")
    parser.add_argument('input_dir', type=str, help='Path to the directory containing input videos.')
    parser.add_argument('weights_path', type=str, help='Path to the model weights (.pth file).')
    
    # ✨ 新增架构选择参数
    parser.add_argument(
        '--arch', 
        type=str, 
        required=True, 
        choices=sorted(MODEL_CONFIGS),
        help='Canonical aliases: motion_posterior3/5; legacy motion3/5 remain supported.'
    )
    
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for detection [0, 1).',
    )
    parser.add_argument(
        '--window-mode', choices=('chunk', 'center', 'causal'), default=None,
        help='Temporal inference semantics. Default is chunk (non-overlap); '
             'center is offline symmetric, causal uses only past frames.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory for canonical per-video trajectory CSV files.',
    )
    parser.add_argument(
        '--visualization-dir',
        type=Path,
        default=None,
        help='Optional directory for trajectory and heatmap videos.',
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    weights_path = Path(args.weights_path)
    if not input_dir.is_dir():
        parser.error(f"input_dir is not a directory: {input_dir}")
    if not weights_path.is_file():
        parser.error(f"weights_path is not a file: {weights_path}")
    if not math.isfinite(args.threshold) or not 0 <= args.threshold < 1:
        parser.error(f"threshold must be a finite value in [0, 1), got {args.threshold}")
    
    print("🔎 Searching for .mp4 and .mov files...")
    video_files = []
    supported_formats = ['*.mp4', '*.mov', '*.MOV', '*.MP4']
    for fmt in supported_formats:
        video_files.extend(input_dir.glob(fmt))
    
    if not video_files:
        parser.error(f"No supported video files (.mp4, .mov) found in {input_dir}")
        
    video_files = sorted(set(video_files))
    duplicate_stems = sorted(
        stem for stem in {path.stem for path in video_files}
        if sum(path.stem == stem for path in video_files) > 1
    )
    if duplicate_stems:
        parser.error(f"video stems must be unique: {duplicate_stems}")
    print(f"Found {len(video_files)} videos to process.")

    model_cfg = MODEL_CONFIGS[args.arch]
    print(f"🚀 Starting MotionPosterior batch inference for [{args.arch.upper()}]...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg)
    _load_model_weights(model, weights_path, device='cpu')
    model.to(device).eval()
    print(f"✅ Model loaded from {weights_path} and sent to {device}.")

    output_root_dir = args.output_dir
    output_root_dir.mkdir(parents=True, exist_ok=True)
    
    # ✨ 1. 初始化汇总列表
    summary_data_list = [] 
    
    for video_path in video_files:
        summary_data_list.append(
            process_video(video_path, model, device, args, output_root_dir)
        )

    total_frames = sum(item['total_frames'] for item in summary_data_list)
    detected_frames = sum(item['detected_frames'] for item in summary_data_list)
    print(
        f"\nAll videos processed: videos={len(summary_data_list)}, "
        f"frames={total_frames}, detected={detected_frames}, output={output_root_dir}"
    )


if __name__ == '__main__':
    main()
