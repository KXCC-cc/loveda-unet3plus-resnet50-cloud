"""LoveDA 数据读取与训练、推理共用的图像预处理。"""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.constants import (
    map_raw_mask,
    validate_raw_mask,
)
from utils.transforms import (
    preprocess_image,
    transform_image_and_mask,
    validate_image_size,
)


class LoveDADataset(Dataset):
    """读取 root/{Train,Val,Test}/{Rural,Urban} 下的 LoveDA PNG。

    Train / Val 返回 (image, mask)：
        image: float32，[3, H, W]，完成 ImageNet normalization。
        mask: long，[H, W]，原始 0 → 255，原始 1~7 → 0~6。

    Train 默认原分辨率随机 crop，并同步做翻转/直角旋转。
    Val / Test 默认保持完整图像，禁止随机 crop 和训练增强；全图 loader 使用 batch=1。
    spatial_mode='resize' 保留整图缩放的对照策略，mask 始终使用 nearest。
    Test 没有真实标签，返回 (image, str(image_path))。
    samples 保留每个原始文件一项，samples_per_image 只扩展训练迭代长度。
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "Train",
        image_size: Tuple[int, int] = (256, 256),
        spatial_mode: str = "crop",
        augment: Optional[bool] = None,
        samples_per_image: int = 1,
        color_jitter: float = 0.0,
        train_scales: Sequence[float] = (1.0,),
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        rotate90_probability: float = 1.0,
        class_aware_crop_probability: float = 0.0,
        class_aware_classes: Sequence[int] = (1, 2, 3, 4),
        min_target_pixels: int = 512,
        max_crop_attempts: int = 8,
        augmentation_mode: str = "independent",
        one_of_probability: float = 0.75,
    ) -> None:
        self.root = Path(root)
        if split not in ("Train", "Val", "Test"):
            raise ValueError("split 必须是 'Train'、'Val' 或 'Test'。")
        self.split = split
        self.image_size = validate_image_size(image_size)
        if spatial_mode not in ("crop", "resize", "full"):
            raise ValueError("spatial_mode 必须是 'crop'、'resize' 或 'full'。")
        # 默认参数 crop 只适用于 Train；Val/Test 自动改为确定性的 full。
        self.spatial_mode = (
            "full" if split != "Train" and spatial_mode == "crop"
            else spatial_mode
        )
        self.augment = split == "Train" if augment is None else bool(augment)
        if split != "Train" and self.augment:
            raise ValueError("Val/Test 不允许训练增强。")
        if isinstance(samples_per_image, bool) or not isinstance(
            samples_per_image, int
        ):
            raise TypeError("samples_per_image 必须是正整数。")
        if samples_per_image <= 0:
            raise ValueError("samples_per_image 必须大于 0。")
        if split != "Train" and samples_per_image != 1:
            raise ValueError("Val/Test 的 samples_per_image 必须为 1，确保每张图只计一次。")
        if not 0.0 <= color_jitter <= 1.0:
            raise ValueError("color_jitter 必须位于 [0, 1]，0 表示关闭。")
        if split != "Train" and color_jitter != 0.0:
            raise ValueError("Val/Test 不允许颜色增强。")
        self.samples_per_image = samples_per_image
        self.color_jitter = float(color_jitter)
        self.train_scales = tuple(float(value) for value in train_scales)
        if not self.train_scales or any(value <= 0 for value in self.train_scales):
            raise ValueError("train_scales 必须包含正数。")
        for name, probability in (
            ("horizontal_flip_probability", horizontal_flip_probability),
            ("vertical_flip_probability", vertical_flip_probability),
            ("rotate90_probability", rotate90_probability),
            ("class_aware_crop_probability", class_aware_crop_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} 必须位于 [0,1]。")
        if any(value < 0 or value >= 7 for value in class_aware_classes):
            raise ValueError("class_aware_classes 使用训练索引 0~6。")
        if min_target_pixels < 1 or max_crop_attempts < 1:
            raise ValueError("min_target_pixels 和 max_crop_attempts 必须为正整数。")
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.vertical_flip_probability = float(vertical_flip_probability)
        self.rotate90_probability = float(rotate90_probability)
        self.class_aware_crop_probability = float(class_aware_crop_probability)
        # Dataset 对外使用训练索引，变换内部读取原始 mask，故统一加 1。
        self.class_aware_raw_classes = tuple(int(value) + 1 for value in class_aware_classes)
        self.min_target_pixels = int(min_target_pixels)
        self.max_crop_attempts = int(max_crop_attempts)
        if augmentation_mode not in ("independent", "one_of"):
            raise ValueError("augmentation_mode 仅支持 independent 或 one_of。")
        if not 0.0 <= one_of_probability <= 1.0:
            raise ValueError("one_of_probability 必须位于 [0,1]。")
        self.augmentation_mode = augmentation_mode
        self.one_of_probability = float(one_of_probability)
        self.has_masks = split != "Test"
        self.samples: list[Tuple[Path, Optional[Path]]] = []

        for region in ("Rural", "Urban"):
            region_dir = self.root / split / region
            image_dir = region_dir / "images_png"
            if not image_dir.is_dir():
                raise FileNotFoundError(f"图像目录不存在：{image_dir}")
            image_paths = sorted(
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".png"
            )
            if not image_paths:
                raise ValueError(f"图像目录中没有 PNG 文件：{image_dir}")

            mask_paths = {}
            if self.has_masks:
                mask_dir = region_dir / "masks_png"
                if not mask_dir.is_dir():
                    raise FileNotFoundError(f"标签目录不存在：{mask_dir}")
                mask_paths = {
                    path.name: path
                    for path in mask_dir.iterdir()
                    if path.is_file() and path.suffix.lower() == ".png"
                }
                image_names = {path.name for path in image_paths}
                mask_names = set(mask_paths)
                missing_masks = sorted(image_names - mask_names)
                extra_masks = sorted(mask_names - image_names)
                if missing_masks or extra_masks:
                    raise ValueError(
                        f"{region_dir} 的 image/mask 文件名不匹配。"
                        f"缺少 mask 的图像（最多列出 5 个）：{missing_masks[:5]}；"
                        f"没有对应图像的 mask（最多列出 5 个）：{extra_masks[:5]}"
                    )

            for image_path in image_paths:
                mask_path: Optional[Path] = (
                    mask_paths[image_path.name] if self.has_masks else None
                )
                self.samples.append((image_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples) * self.samples_per_image

    def __getitem__(
        self, index: int,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, str]]:
        """读取原始文件，同步变换后返回图像与标签或 Test 文件路径。"""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("LoveDADataset 索引越界。")
        # 同一原图每轮可抽取多个随机 crop，不复制文件、不预先生成裁剪数据。
        image_path, mask_path = self.samples[index % len(self.samples)]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        raw_mask = None
        if self.has_masks:
            with Image.open(mask_path) as mask_image:
                if mask_image.size != image.size:
                    raise ValueError(
                        f"原始图像与标签尺寸不同：{image_path}={image.size}，"
                        f"{mask_path}={mask_image.size}"
                    )
                # 不 convert('L')：P 模式的调色板索引本身就是类别值。
                raw_mask = np.array(mask_image)
            try:
                validate_raw_mask(raw_mask)
            except (TypeError, ValueError) as error:
                raise ValueError(f"标签文件无效：{mask_path}；{error}") from error

        image, raw_mask = transform_image_and_mask(
            image,
            raw_mask,
            image_size=self.image_size,
            spatial_mode=self.spatial_mode,
            augment=self.augment,
            color_jitter=self.color_jitter,
            train_scales=self.train_scales,
            horizontal_flip_probability=self.horizontal_flip_probability,
            vertical_flip_probability=self.vertical_flip_probability,
            rotate90_probability=self.rotate90_probability,
            class_aware_crop_probability=self.class_aware_crop_probability,
            class_aware_raw_classes=self.class_aware_raw_classes,
            min_target_pixels=self.min_target_pixels,
            max_crop_attempts=self.max_crop_attempts,
            augmentation_mode=self.augmentation_mode,
            one_of_probability=self.one_of_probability,
        )
        image_tensor = preprocess_image(image)  # 只归一化，不再次缩放。
        if not self.has_masks:
            return image_tensor, str(image_path)
        mask_tensor = torch.from_numpy(map_raw_mask(raw_mask))  # long，[H, W]。
        return image_tensor, mask_tensor
