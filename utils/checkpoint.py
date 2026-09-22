"""保存/恢复完整训练状态，或只加载模型权重开始独立微调。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.unet3plus import MODEL_VERSION
from utils.constants import (
    CLASS_NAMES,
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
)


def load_model_weights(
    path: Path, model: torch.nn.Module,
) -> dict[str, Any]:
    """加载全部模型权重和 BN 统计量，供一个新的微调实验使用。

    类别顺序、标签与归一化规则必须一致；裁剪尺寸、学习率、epoch 等
    新实验参数可以调整。此处不恢复 optimizer、scheduler、scaler 或随机
    状态，也不沿用旧实验的最佳分数。完整断点续训仍使用下面的 resume。
    """
    path = path.resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("初始化 checkpoint 必须包含 model_state_dict。")
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(
            "初始化 checkpoint 的模型版本不兼容："
            f"保存={checkpoint.get('model_version')!r}，"
            f"当前={MODEL_VERSION!r}。不能忽略缺失参数加载。"
        )
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        raise ValueError("初始化 checkpoint 的 config 必须是字典。")
    expected_metadata = {
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "ignore_index": IGNORE_INDEX,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }
    for key, expected in expected_metadata.items():
        if saved.get(key) != expected:
            raise ValueError(
                f"初始化 checkpoint 的 {key} 与当前数据定义不一致："
                f"保存={saved.get(key)!r}，当前={expected!r}。"
            )
    source_epoch = int(checkpoint.get("epoch", 0))
    # strict=True 同时检查主干、所有监督头及 BN 缓冲区，不做部分加载。
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(
            "初始化 checkpoint 权重与当前 U-Net 3+ 不兼容，不能部分加载。"
        ) from error
    # 返回轻量来源信息；不把旧 optimizer 等大块张量留在训练主程序中。
    return {"checkpoint": str(path), "epoch": source_epoch, "config": saved}


def capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    """只保存 Tensor/基础类型，兼容 torch.load(weights_only=True)。"""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
            int(numpy_state[3]), float(numpy_state[4]),
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def restore_rng_state(
    state: dict[str, Any], generator: torch.Generator,
) -> None:
    """恢复主进程随机状态；不包含 persistent workers 内部的随机流。"""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
        numpy_state[2], numpy_state[3], numpy_state[4],
    ))
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    cuda_states = state["cuda"]
    if torch.cuda.is_available() and cuda_states:
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
        else:
            print("GPU 数量与保存时不同：保留本次种子，不恢复 CUDA 随机流。")


def load_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: dict[str, Any],
    generator: torch.Generator,
) -> tuple[int, float]:
    """恢复本版本 checkpoint；拒绝把旧四输出模型当作完整 resume。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(
            "checkpoint 不是当前五输出模型版本，不能直接 resume。"
            "旧版四输出权重缺少 E5 分类头，请为本版本重新训练。"
        )
    saved = checkpoint["config"]
    consistent_keys = (
        "num_classes", "image_size", "data_strategy", "eval_stride",
        "samples_per_image", "augment", "color_jitter", "epochs",
        "batch_size", "lr", "weight_decay", "aux_weights",
        "ce_weight", "dice_weight", "focal_weight", "focal_gamma",
        "class_weights", "grad_clip", "ignore_index", "class_names",
        "mean", "std", "data_root", "visualization_samples",
        "fixed_validation_samples",
    )
    for key in consistent_keys:
        if saved.get(key) != config.get(key):
            raise ValueError(
                f"resume 设置不一致：{key}，保存={saved.get(key)}，"
                f"当前={config.get(key)}。请沿用原训练参数。"
            )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    # PyTorch 会按对应参数的设备恢复 optimizer 内的动量张量。
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler.is_enabled():
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        elif saved.get("amp_enabled"):
            raise ValueError("保存时启用了 AMP，但 checkpoint 缺少 scaler 状态。")
        else:
            print("原训练未启用 AMP：本次使用新的 GradScaler。")
    restore_rng_state(checkpoint["rng_state"], generator)
    if "initialization" in saved:
        # resume 不必重新指定初始化文件，但仍保留本次微调的权重来源。
        config["initialization"] = saved["initialization"]
        config["init_checkpoint"] = saved.get("init_checkpoint")
    return int(checkpoint["epoch"]), float(checkpoint["best_miou"])


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    """先写临时文件再替换，降低保存中断损坏上一份 checkpoint 的风险。"""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_experiment_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    current_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """恢复新训练引擎在 update 边界保存的完整状态。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "micro_step",
        "optimizer_step",
        "best_miou",
        "config",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"checkpoint 缺少字段：{sorted(missing)}")
    if current_config is not None:
        saved_config = checkpoint["config"]
        for section in ("model", "data", "loss", "optimizer", "scheduler", "training"):
            if saved_config.get(section) != current_config.get(section):
                raise ValueError(
                    f"resume 配置不一致：{section}。请使用原实验 config.json/YAML。"
                )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler.is_enabled() and scaler_state:
        scaler.load_state_dict(scaler_state)
    return checkpoint
