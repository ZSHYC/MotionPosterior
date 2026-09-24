import cv2
import numpy as np
import torch
from pathlib import Path

# 导入我们自己的TRANSFORMS注册表
from ..builder import TRANSFORMS

@TRANSFORMS.register_module
class LoadMultiImagesFromPaths:
    """从results字典中的文件路径加载多张图像。"""
    def __init__(self, to_rgb=True):
        self.to_rgb = to_rgb

    def __call__(self, results: dict) -> dict:
        # 'img_fields' 是我们在Dataset中准备好的、需要加载的图片键名列表
        for key in results['img_fields']:
            img_path = results[key]
            if not isinstance(img_path, Path) or not img_path.exists():
                 raise FileNotFoundError(f"Image file not found at path: {img_path} for key: {key}")
            
            img = cv2.imread(str(img_path)) # 把图片读进来，BGR转RGB
            if self.to_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results[key] = img # 原本存字符串（图片路径）的，现在直接直接存读出来的img，后传给resize
        return results

@TRANSFORMS.register_module
class Resize:
    """对字典中的多张图像进行缩放。"""
    def __init__(self, keys, size: tuple):
        # size a tuple of (height, width)
        self.keys = keys
        self.size_wh = (size[1], size[0]) # cv2.resize expects (width, height)

    def __call__(self, results: dict) -> dict:
        original_shape = None
        for key in self.keys:
            if key in results:
                image = results[key]
                if original_shape is None:
                    original_shape = image.shape[:2]
                results[key] = cv2.resize(image, self.size_wh) # 调整到模型需要的输入尺寸
        # 坐标来自原始帧像素坐标，必须和图像使用同一几何缩放。
        coords = results.get('coords')
        if coords is not None and original_shape is not None:
            old_h, old_w = original_shape
            new_w, new_h = self.size_wh
            scale_x = new_w / max(float(old_w), 1.0)
            scale_y = new_h / max(float(old_h), 1.0)
            scaled = []
            for pair in coords:
                try:
                    x, y = float(pair[0]), float(pair[1])
                except (TypeError, ValueError, IndexError):
                    x, y = float('nan'), float('nan')
                scaled.append((x * scale_x, y * scale_y))
            results['coords'] = scaled
        return results

@TRANSFORMS.register_module
class GenerateMotionAttention:
    """根据前后帧生成运动注意力图。"""
    def __init__(self, threshold=40):
        self.threshold = threshold

    def __call__(self, results: dict) -> dict:
        # 这个transform假设 'path_prev', 'path', 'path_next' 的图像已经被加载
        img_prev, img_curr, img_next = results['path_prev'], results['path'], results['path_next']
        
        gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_RGB2GRAY).astype(np.int16)
        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_RGB2GRAY).astype(np.int16)
        gray_next = cv2.cvtColor(img_next, cv2.COLOR_RGB2GRAY).astype(np.int16)

        diff1 = gray_curr - gray_prev
        diff2 = gray_next - gray_curr
        
        brighten1 = ((diff1 > self.threshold) * 255).astype(np.uint8)
        darken1 = ((diff1 < -self.threshold) * 255).astype(np.uint8)
        brighten2 = ((diff2 > self.threshold) * 255).astype(np.uint8)
        darken2 = ((diff2 < -self.threshold) * 255).astype(np.uint8)
        
        results['att_prev_to_curr'] = np.stack([brighten1, darken1], axis=-1)
        results['att_curr_to_next'] = np.stack([brighten2, darken2], axis=-1)
        return results

@TRANSFORMS.register_module
class ConcatChannels:
    """按指定顺序拼接通道，形成最终的模型输入。"""
    def __init__(self, keys, output_key='image'):
        self.keys = keys # 这里是按顺序排列的，需要concatenate的keys
        self.output_key = output_key # 用一个'image' 存放concatenate之后的输入

    def __call__(self, results: dict) -> dict:
        imgs_to_stack = [results[key] for key in self.keys] # 先stack起来 [360, 640, 9] x n, [H, W, C]是cv2的读取格式
        results[self.output_key] = np.concatenate(imgs_to_stack, axis=2) # 然后沿着通道维度拼接
        return results

@TRANSFORMS.register_module
class LoadAndFormatTarget:
    """加载、缩放并格式化GT热力图为Tensor。"""
    def __init__(self, key='gt_path', output_key='target'):
        self.key = key
        self.output_key = output_key

    def __call__(self, results: dict) -> dict:
        gt_path = results[self.key]
        size = (results['input_width'], results['input_height'])
        target_np = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) # 以灰度图读取
        target_np = cv2.resize(target_np.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        results[self.output_key] = torch.from_numpy(target_np)
        return results

@TRANSFORMS.register_module
class LoadAndFormatMultiTargets:
    """加载、缩放并格式化多张GT热力图为Tensor，维度为[3, h, w]。"""

    def __init__(self, keys=['gt_path_prev', 'gt_path', 'gt_path_next'], output_key='target'):
        self.keys = keys
        self.output_key = output_key

    def __call__(self, results: dict) -> dict:
        targets = []
        size = (results['input_width'], results['input_height'])

        for key in self.keys:
            gt_path = results[key]
            target_np = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            target_np = cv2.resize(target_np.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
            targets.append(target_np)

        # 堆叠成 [3, H, W] 维度
        target_stack = np.stack(targets, axis=0).astype(np.float32)  # 形状: (T, H, W)
        results[self.output_key] = torch.from_numpy(target_stack)
        return results


@TRANSFORMS.register_module
class Finalize:
    """
    将数据转为Tensor并收集最终需要的键值对，作为dataloader的最终输出。
    """
    def __init__(self, image_key='image', final_keys=['image', 'target', 'coords', 'visibility']):
        self.image_key = image_key
        self.final_keys = final_keys
    
    def __call__(self, results: dict) -> dict:
        # 将最终的输入图像转为 PyTorch 需要的 (C, H, W) 格式 Tensor
        img = results[self.image_key]
        results[self.image_key] = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255) # 转成需要的pytorch需要的维度格式[C,H,W], 数据格式0-1之间
        
        # 固定 metadata 契约：[T, 2] float32 和 [T] float32。这里做转换而
        # 非依赖默认 collate 的列表转置，单样本和批处理拥有同一语义。
        if 'coords' in results:
            coords = np.asarray(results['coords'], dtype=np.float32)
            if coords.size == 0:
                coords = np.full((0, 2), np.nan, dtype=np.float32)
            coords = coords.reshape(-1, 2)
            results['coords'] = torch.from_numpy(coords)
        if 'visibility' in results:
            visibility = np.asarray(results['visibility'], dtype=np.float32).reshape(-1)
            visibility = np.nan_to_num(visibility, nan=0.0, posinf=0.0, neginf=0.0)
            results['visibility'] = torch.from_numpy(visibility)

        # 从“周转箱”中只挑选出模型训练/评估需要的最终数据
        final_data = {}
        for key in self.final_keys:
            if key in results:
                final_data[key] = results[key]
        return final_data
