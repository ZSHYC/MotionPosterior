# 文件路径: ./scripts/preprocess_data.py (已修正无球帧的处理逻辑)

import numpy as np
import pandas as pd
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def create_gaussian_kernel(size, variance):
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x ** 2 + y ** 2) / float(2 * variance))
    g = g * 255 / g.max()
    return g.astype(np.uint8)


def process_data(input_dir: Path, output_dir: Path, mode: str, config: dict):
    num_frames = config.get('num_frames', 3)
    window_type = config.get('window_type', 'center')
    if num_frames not in (3, 5):
        raise ValueError('num_frames must be 3 or 5')
    if num_frames == 5 and mode != 'context':
        raise ValueError('5-frame preprocessing is available for context mode only')
    if window_type not in ('center', 'causal'):
        raise ValueError('window_type must be center or causal')
    if window_type == 'causal' and (mode != 'context' or num_frames != 5):
        raise ValueError('causal preprocessing currently requires 5-frame context mode')
    gaussian_kernel = create_gaussian_kernel(config['size'], config['variance'])
    kernel_size = config['size']
    height, width = config['height'], config['width']

    label_files = sorted(list(input_dir.glob('**/Label.csv')))

    if not label_files:
        print(f"❌ Error: No 'Label.csv' files found in the directory: {input_dir}")
        return

    all_clip_dfs = []

    print(f"🚀 Starting data preprocessing for mode: '{mode}'...")
    for label_path in tqdm(label_files, desc="Processing Clips"):
        clip_df = pd.read_csv(label_path)
        clip_root = label_path.parent

        gt_clip_output_dir = output_dir / 'gts' / clip_root.relative_to(input_dir)
        gt_clip_output_dir.mkdir(parents=True, exist_ok=True)

        gt_paths = []
        # ✨✨✨ 核心改动区域开始 ✨✨✨
        for _, row in clip_df.iterrows():
            gt_path = gt_clip_output_dir / row['file name']
            gt_paths.append(str(gt_path.relative_to(output_dir)))

            # 仅在热力图文件不存在时创建，避免重复工作
            if not gt_path.exists():
                # 1. 首先，创建一个纯黑的画布
                heatmap = np.zeros((height, width), dtype=np.uint8)

                # 2. 只有当球可见且坐标存在时，才在画布上画高斯斑点
                if row['visibility'] != 0 and pd.notna(row['x-coordinate']):
                    x, y = int(row['x-coordinate']), int(row['y-coordinate'])

                    x_min, x_max = max(0, x - kernel_size), min(width, x + kernel_size + 1)
                    y_min, y_max = max(0, y - kernel_size), min(height, y + kernel_size + 1)

                    kernel_x_min = max(0, kernel_size - (x - x_min))
                    kernel_x_max = kernel_size + (x_max - x)
                    kernel_y_min = max(0, kernel_size - (y - y_min))
                    kernel_y_max = kernel_size + (y_max - y)

                    if x_max > x_min and y_max > y_min:
                        heatmap[y_min:y_max, x_min:x_max] = gaussian_kernel[kernel_y_min:kernel_y_max,
                                                            kernel_x_min:kernel_x_max]
                # 3. 无论画布上是否有斑点，都将它保存下来
                cv2.imwrite(str(gt_path), heatmap)
        # ✨✨✨ 核心改动区域结束 ✨✨✨

        clip_df['gt_path'] = gt_paths
        base_path_col = clip_root.relative_to(input_dir)
        clip_df['_clip_id'] = str(base_path_col)
        clip_df['path'] = [str(base_path_col / fname) for fname in clip_df['file name']]

        # ✨✨✨ 新增：为每帧添加前后帧的信息 ✨✨✨
        if mode == 'past':
            # 添加前一帧和后一帧的路径和标签信息
            clip_df['path_prev'] = clip_df['path'].shift(1)
            clip_df['path_next'] = clip_df['path'].shift(-1)

            # 添加前一帧和后一帧的gt路径
            clip_df['gt_path_prev'] = clip_df['gt_path'].shift(1)
            clip_df['gt_path_next'] = clip_df['gt_path'].shift(-1)

            # 添加前一帧和后一帧的坐标信息
            clip_df['x_prev'] = clip_df['x-coordinate'].shift(1)
            clip_df['y_prev'] = clip_df['y-coordinate'].shift(1)
            clip_df['x_next'] = clip_df['x-coordinate'].shift(-1)
            clip_df['y_next'] = clip_df['y-coordinate'].shift(-1)

            # 添加前一帧和后一帧的visibility和status
            clip_df['visibility_prev'] = clip_df['visibility'].shift(1)
            clip_df['status_prev'] = clip_df['status'].shift(1)
            clip_df['visibility_next'] = clip_df['visibility'].shift(-1)
            clip_df['status_next'] = clip_df['status'].shift(-1)

        elif mode == 'context' and num_frames == 3:
            # 添加前一帧和后一帧的路径和标签信息
            clip_df['path_prev'] = clip_df['path'].shift(1)
            clip_df['path_next'] = clip_df['path'].shift(-1)

            # 添加前一帧和后一帧的gt路径
            clip_df['gt_path_prev'] = clip_df['gt_path'].shift(1)
            clip_df['gt_path_next'] = clip_df['gt_path'].shift(-1)

            # 添加前一帧和后一帧的坐标信息
            clip_df['x_prev'] = clip_df['x-coordinate'].shift(1)
            clip_df['y_prev'] = clip_df['y-coordinate'].shift(1)
            clip_df['x_next'] = clip_df['x-coordinate'].shift(-1)
            clip_df['y_next'] = clip_df['y-coordinate'].shift(-1)

            # 添加前一帧和后一帧的visibility和status
            clip_df['visibility_prev'] = clip_df['visibility'].shift(1)
            clip_df['status_prev'] = clip_df['status'].shift(1)
            clip_df['visibility_next'] = clip_df['visibility'].shift(-1)
            clip_df['status_next'] = clip_df['status'].shift(-1)

        elif mode == 'context' and num_frames == 5:
            neighbours = (
                [('prev4', 4), ('prev3', 3), ('prev2', 2), ('prev', 1)]
                if window_type == 'causal' else
                [('prev2', 2), ('prev', 1), ('next', -1), ('next2', -2)]
            )
            for suffix, offset in neighbours:
                clip_df[f'path_{suffix}'] = clip_df['path'].shift(offset)
                clip_df[f'gt_path_{suffix}'] = clip_df['gt_path'].shift(offset)
                clip_df[f'x_{suffix}'] = clip_df['x-coordinate'].shift(offset)
                clip_df[f'y_{suffix}'] = clip_df['y-coordinate'].shift(offset)
                clip_df[f'visibility_{suffix}'] = clip_df['visibility'].shift(offset)
                clip_df[f'status_{suffix}'] = clip_df['status'].shift(offset)

        # 删除没有完整时序上下文的首尾帧。
        if window_type == 'causal':
            clip_df = clip_df.iloc[num_frames - 1:]
        else:
            radius = num_frames // 2
            clip_df = clip_df.iloc[radius:-radius]

        all_clip_dfs.append(clip_df)

    print("✅ All clips processed. Concatenating and creating temporal relationships...")
    master_df = pd.concat(all_clip_dfs, ignore_index=True)

    # ✨✨✨ 修改最终的列选择 ✨✨✨
    if mode == 'past':
        final_columns = [
            'path_prev', 'path', 'path_next',  # 三张图片路径：前一帧、当前帧、后一帧
            'gt_path_prev', 'gt_path', 'gt_path_next',  # 三张对应的gt图路径
            'x_prev', 'y_prev', 'x-coordinate', 'y-coordinate', 'x_next', 'y_next',  # 三个x,y坐标
            'visibility_prev', 'visibility', 'visibility_next',  # 三个visibility
            'status_prev', 'status', 'status_next'  # 三个status
        ]
    elif mode == 'context' and num_frames == 3:
        final_columns = [
            'path_prev', 'path', 'path_next',  # 三张图片路径：前一帧、当前帧、后一帧
            'gt_path_prev', 'gt_path', 'gt_path_next',  # 三张对应的gt图路径
            'x_prev', 'y_prev', 'x-coordinate', 'y-coordinate', 'x_next', 'y_next',  # 三个x,y坐标
            'visibility_prev', 'visibility', 'visibility_next',  # 三个visibility
            'status_prev', 'status', 'status_next'  # 三个status
        ]
    elif mode == 'context' and num_frames == 5 and window_type == 'causal':
        final_columns = [
            'path_prev4', 'path_prev3', 'path_prev2', 'path_prev', 'path',
            'gt_path_prev4', 'gt_path_prev3', 'gt_path_prev2', 'gt_path_prev', 'gt_path',
            'x_prev4', 'y_prev4', 'x_prev3', 'y_prev3', 'x_prev2', 'y_prev2',
            'x_prev', 'y_prev', 'x-coordinate', 'y-coordinate',
            'visibility_prev4', 'visibility_prev3', 'visibility_prev2', 'visibility_prev', 'visibility',
            'status_prev4', 'status_prev3', 'status_prev2', 'status_prev', 'status',
        ]
    elif mode == 'context' and num_frames == 5:
        final_columns = [
            'path_prev2', 'path_prev', 'path', 'path_next', 'path_next2',
            'gt_path_prev2', 'gt_path_prev', 'gt_path', 'gt_path_next', 'gt_path_next2',
            'x_prev2', 'y_prev2', 'x_prev', 'y_prev', 'x-coordinate', 'y-coordinate',
            'x_next', 'y_next', 'x_next2', 'y_next2',
            'visibility_prev2', 'visibility_prev', 'visibility', 'visibility_next', 'visibility_next2',
            'status_prev2', 'status_prev', 'status', 'status_next', 'status_next2'
        ]

    final_df = master_df[final_columns + ['_clip_id']]

    # 重命名列以保持一致性
    column_rename = {
        'x-coordinate': 'x_current',
        'y-coordinate': 'y_current',
        'visibility': 'visibility_current',
        'status': 'status_current'
    }
    final_df = final_df.rename(columns=column_rename)

    train_rate = float(config['train_rate'])
    if not 0.0 < train_rate < 1.0:
        raise ValueError('train_rate must be strictly between 0 and 1')
    # Split by clip, never by adjacent windows.  Random row splitting leaks
    # nearly identical neighbouring frames into validation.
    clip_ids = np.array(sorted(final_df['_clip_id'].unique()))
    if len(clip_ids) < 2:
        raise ValueError(
            'Grouped train/validation split requires at least two clips; '
            'add another clip instead of reporting a leaked or empty validation set.'
        )
    rng = np.random.default_rng(42)
    rng.shuffle(clip_ids)
    if len(clip_ids) > 1:
        num_train_clips = min(max(1, int(round(len(clip_ids) * train_rate))), len(clip_ids) - 1)
        train_ids = set(clip_ids[:num_train_clips])
        df_train = final_df[final_df['_clip_id'].isin(train_ids)]
        df_val = final_df[~final_df['_clip_id'].isin(train_ids)]
    df_train = df_train.drop(columns=['_clip_id']).reset_index(drop=True)
    df_val = df_val.drop(columns=['_clip_id']).reset_index(drop=True)

    label_suffix = ('causal5' if window_type == 'causal' else f'{mode}5') if mode == 'context' and num_frames == 5 else mode
    train_csv_path = output_dir / f"labels_{label_suffix}_train.csv"
    val_csv_path = output_dir / f"labels_{label_suffix}_val.csv"

    df_train.to_csv(train_csv_path, index=False)
    df_val.to_csv(val_csv_path, index=False)

    print(f"🎉 Preprocessing for mode '{mode}' complete!")
    print(f"Train samples: {len(df_train)}, saved to {train_csv_path}")
    print(f"Validation samples: {len(df_val)}, saved to {val_csv_path}")

    # 打印第一行数据作为示例
    print("\n📊 Example of first row in final dataset:")
    if not df_train.empty:
        sample = df_train.iloc[0]
        frame_labels = (('prev4', 'prev3', 'prev2', 'prev', 'current') if window_type == 'causal' else ('prev2', 'prev', 'current', 'next', 'next2')) if num_frames == 5 else ('prev', 'current', 'next')
        print("Image paths:", ', '.join(str(sample['path' if label == 'current' else f'path_{label}']) for label in frame_labels))
        print("GT paths:", ', '.join(str(sample['gt_path' if label == 'current' else f'gt_path_{label}']) for label in frame_labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TrackNet Dataset Preprocessing Script")
    parser.add_argument('--input_dir', '-in', type=str, required=True, help='Path to the raw data directory.')
    parser.add_argument('--output_dir', '-out', type=str, required=True,
                        help='Path to save the processed data and labels.')
    parser.add_argument('--mode', '-m', type=str, required=True, choices=['past', 'context'], help="Processing mode.")
    parser.add_argument('--num-frames', type=int, choices=[3, 5], default=3,
                        help='Temporal context length. 5 is supported for context mode.')
    parser.add_argument('--window-type', choices=['center', 'causal'], default='center',
                        help='Five-frame training window layout.')
    parser.add_argument('--height', type=int, default=1080, help='Target image height.')
    parser.add_argument('--width', type=int, default=1920, help='Target image width.')
    parser.add_argument('--size', type=int, default=40, help='Radius of the Gaussian kernel.')
    parser.add_argument('--variance', type=float, default=10, help='Variance of the Gaussian kernel.')
    parser.add_argument('--train_rate', type=float, default=0.8, help='Proportion of clips to use for training.')

    args = parser.parse_args()

    config = {
        'height': args.height, 'width': args.width,
        'size': args.size, 'variance': args.variance,
        'train_rate': args.train_rate, 'num_frames': args.num_frames,
        'window_type': args.window_type,
    }

    print(config)

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    process_data(input_path, output_path, args.mode, config)
