"""基于完整验证集累计混淆矩阵的确定性评估。"""

from __future__ import annotations

import math
from typing import Any

import torch
from tqdm import tqdm

from utils.inference import multi_scale_sliding_window_logits
from utils.metrics import SegmentationMetrics


@torch.inference_mode()
def evaluate_full_validation(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
    config: dict[str, Any],
    description: str = "Full Val",
    max_images: int | None = None,
) -> tuple[float, dict, torch.Tensor]:
    """逐图滑窗评估；IoU/Dice 最后由全 Val 混淆矩阵统一计算。"""
    model.eval()
    validation = config["validation"]
    tile_size = tuple(validation.get("tile_size", (512, 512)))
    stride = tuple(validation.get("stride", tile_size))
    scales = tuple(float(value) for value in validation.get("scales", (1.0,)))
    metrics = SegmentationMetrics(device="cpu")
    total_loss = 0.0
    loss_images = 0

    progress = tqdm(loader, desc=description, leave=False)
    for index, (images, masks) in enumerate(progress):
        if max_images is not None and index >= max_images:
            break
        image = images[0].float().cpu()
        target = masks.long().cpu()
        logits = multi_scale_sliding_window_logits(
            model=model,
            image=image,
            scales=scales,
            tile_size=tile_size,
            stride=stride,
            device=device,
            amp_enabled=bool(config["runtime"].get("amp", True) and device.type == "cuda"),
            tile_batch_size=int(validation.get("patch_batch_size", 1)),
            horizontal_flip=bool(validation.get("horizontal_flip_tta", False)),
            accumulate_on_device=bool(validation.get("accumulate_on_device", False)),
        )
        # criterion 的 class-weight buffer 位于训练设备；仅损失计算时传入 GPU。
        loss = criterion(
            logits.to(device, non_blocking=True),
            target.to(device, non_blocking=True),
        )
        if math.isfinite(float(loss)):
            total_loss += float(loss)
            loss_images += 1
        metrics.update(logits.argmax(dim=1).long(), target)

    if loss_images == 0:
        raise RuntimeError("验证集没有可计算损失的图像。")
    scores = metrics.compute()
    matrix = metrics.confusion_matrix.clone()
    return total_loss / loss_images, scores, matrix
