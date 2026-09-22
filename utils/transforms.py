"""同步几何变换：图像和原始 mask 始终使用同一个裁剪框与翻转方向。"""

import random
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance

from utils.constants import IMAGENET_MEAN, IMAGENET_STD, RAW_IGNORE_INDEX


def validate_image_size(image_size: Tuple[int, int]) -> Tuple[int, int]:
    """尺寸统一使用 (高度, 宽度)，Pillow resize 时再交换顺序。"""
    if len(image_size) != 2:
        raise ValueError("image_size 必须是 (height, width)。")
    height, width = image_size
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in image_size
    ):
        raise TypeError("image_size 中的高度、宽度必须是整数。")
    if height <= 0 or width <= 0:
        raise ValueError("image_size 中的高度、宽度必须大于 0。")
    return height, width


def preprocess_image(
    image: Image.Image,
    image_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """RGB → 可选 bilinear resize → [0, 1] → ImageNet normalization。

    返回 float32 [3, H, W]。image_size=None 保持原尺寸，适合全图滑窗；
    resize 对照策略显式传入训练尺寸，训练与推理使用同一预处理。
    """
    image = image.convert("RGB")
    if image_size is not None:
        height, width = validate_image_size(image_size)
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    array = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def transform_image_and_mask(
    image: Image.Image,
    raw_mask: Optional[np.ndarray],
    image_size: Tuple[int, int],
    spatial_mode: str,
    augment: bool = False,
    color_jitter: float = 0.0,
    train_scales: Sequence[float] = (1.0,),
    horizontal_flip_probability: float = 0.5,
    vertical_flip_probability: float = 0.5,
    rotate90_probability: float = 1.0,
    class_aware_crop_probability: float = 0.0,
    class_aware_raw_classes: Sequence[int] = (2, 3, 4, 5),
    min_target_pixels: int = 512,
    max_crop_attempts: int = 8,
    augmentation_mode: str = "independent",
    one_of_probability: float = 0.75,
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """同步执行多尺度、crop、翻转和直角旋转；标签始终保留 0~7。

    原始 mask 的尺寸、类型和标签范围由 Dataset 在进入这里前检查。
    只为全 ignore 的 crop 有限重采样，不对类别比例做强制筛选。
    """
    crop_height, crop_width = validate_image_size(image_size)
    image = image.convert("RGB")

    if not train_scales or any(scale <= 0 for scale in train_scales):
        raise ValueError("train_scales 必须包含正数。")
    if spatial_mode == "crop":
        scale = float(random.choice(tuple(train_scales)))
        if scale != 1.0:
            scaled_width = max(1, round(image.width * scale))
            scaled_height = max(1, round(image.height * scale))
            image = image.resize(
                (scaled_width, scaled_height), resample=Image.Resampling.BILINEAR
            )
            if raw_mask is not None:
                raw_mask = np.array(
                    Image.fromarray(raw_mask.astype(np.uint8)).resize(
                        (scaled_width, scaled_height),
                        resample=Image.Resampling.NEAREST,
                    )
                )

    if spatial_mode == "crop":
        width, height = image.size
        pad_height = max(crop_height - height, 0)
        pad_width = max(crop_width - width, 0)
        if pad_height or pad_width:
            # 小图只在右侧/底部补边。图像复制边缘，mask 补原始 ignore 值 0。
            padded_image = np.pad(
                np.array(image),
                ((0, pad_height), (0, pad_width), (0, 0)),
                mode="edge",
            )
            image = Image.fromarray(padded_image)
            if raw_mask is not None:
                raw_mask = np.pad(
                    raw_mask,
                    ((0, pad_height), (0, pad_width)),
                    mode="constant",
                    constant_values=RAW_IGNORE_INDEX,
                )
        width, height = image.size
        cropped_mask = None
        use_class_aware = (
            raw_mask is not None
            and class_aware_crop_probability > 0
            and random.random() < class_aware_crop_probability
        )
        if use_class_aware:
            present_values = set(np.unique(raw_mask).tolist())
            present_classes = [
                value for value in class_aware_raw_classes if value in present_values
            ]
            position_cache: dict[int, np.ndarray] = {}
            for _ in range(max_crop_attempts):
                if not present_classes:
                    break
                target_class = random.choice(present_classes)
                if target_class not in position_cache:
                    position_cache[target_class] = np.flatnonzero(
                        raw_mask == target_class
                    )
                positions = position_cache[target_class]
                if positions.size == 0:
                    continue
                position = int(random.choice(positions))
                center_y, center_x = divmod(position, width)
                top_min = max(0, center_y - crop_height + 1)
                top_max = min(center_y, height - crop_height)
                left_min = max(0, center_x - crop_width + 1)
                left_max = min(center_x, width - crop_width)
                top = random.randint(top_min, top_max)
                left = random.randint(left_min, left_max)
                candidate = raw_mask[top:top + crop_height, left:left + crop_width]
                if np.count_nonzero(candidate == target_class) >= min_target_pixels:
                    cropped_mask = candidate
                    break
        if cropped_mask is None:
            # class-aware 尝试失败时回退到普通随机裁块，避免改变分布或卡住。
            for _ in range(10):
                top = random.randint(0, height - crop_height)
                left = random.randint(0, width - crop_width)
                if raw_mask is None:
                    break
                cropped_mask = raw_mask[top:top + crop_height, left:left + crop_width]
                if np.any(cropped_mask != RAW_IGNORE_INDEX):
                    break
        # 即使整张原图都是 ignore，也只尝试 10 次，随后由损失/训练循环处理。
        image = image.crop((left, top, left + crop_width, top + crop_height))
        raw_mask = cropped_mask
    elif spatial_mode == "resize":
        image = image.resize(
            (crop_width, crop_height), resample=Image.Resampling.BILINEAR
        )
        if raw_mask is not None:
            mask_image = Image.fromarray(raw_mask.astype(np.uint8))
            raw_mask = np.array(
                mask_image.resize(
                    (crop_width, crop_height), resample=Image.Resampling.NEAREST
                )
            )
    elif spatial_mode != "full":
        raise ValueError("spatial_mode 必须是 'crop'、'resize' 或 'full'。")

    if augmentation_mode not in ("independent", "one_of"):
        raise ValueError("augmentation_mode 仅支持 independent 或 one_of。")
    if not 0.0 <= one_of_probability <= 1.0:
        raise ValueError("one_of_probability 必须位于 [0,1]。")

    if augment:
        horizontal_flip = False
        vertical_flip = False
        turns = 0
        if augmentation_mode == "one_of":
            if random.random() < one_of_probability:
                operation = random.choice(("horizontal", "vertical", "rotate90"))
                horizontal_flip = operation == "horizontal"
                vertical_flip = operation == "vertical"
                if operation == "rotate90":
                    turns = random.choice((1, 2, 3)) if image.width == image.height else 2
        else:
            horizontal_flip = random.random() < horizontal_flip_probability
            vertical_flip = random.random() < vertical_flip_probability
            if random.random() < rotate90_probability:
                turns = random.choice(
                    (0, 1, 2, 3) if image.width == image.height else (0, 2)
                )

        if horizontal_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if raw_mask is not None:
                raw_mask = np.fliplr(raw_mask)
        if vertical_flip:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if raw_mask is not None:
                raw_mask = np.flipud(raw_mask)
        rotations = {
            1: Image.Transpose.ROTATE_90,
            2: Image.Transpose.ROTATE_180,
            3: Image.Transpose.ROTATE_270,
        }
        if turns:
            image = image.transpose(rotations[turns])
            if raw_mask is not None:
                raw_mask = np.rot90(raw_mask, k=turns)

        # 颜色增强只作用于 RGB；默认 0 关闭，避免改变遥感场景颜色过多。
        if color_jitter > 0:
            for enhancer in (
                ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color,
            ):
                factor = random.uniform(1.0 - color_jitter, 1.0 + color_jitter)
                image = enhancer(image).enhance(factor)

    return image, raw_mask
