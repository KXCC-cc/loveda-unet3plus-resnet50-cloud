"""训练实验记录、曲线、混淆矩阵与固定验证样本可视化。"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

# 训练通常运行在无桌面的 WSL/服务器中，显式使用不依赖窗口系统的后端。
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np
from PIL import Image
import torch

from utils.constants import (
    CLASS_NAMES,
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
)


# LoveDA 训练索引 0~6 的固定颜色；所有 epoch、GT 和预测都共用此表。
CLASS_COLORS = np.asarray(
    [
        (0, 0, 0),        # background
        (220, 20, 60),    # building
        (255, 215, 0),    # road
        (30, 144, 255),   # water
        (186, 85, 211),   # barren
        (34, 139, 34),    # forest
        (255, 140, 0),    # agricultural
    ],
    dtype=np.uint8,
)
IGNORE_COLOR = np.asarray((128, 128, 128), dtype=np.uint8)


def metric_fieldnames(class_names: Sequence[str] = CLASS_NAMES) -> list[str]:
    """返回 metrics.csv 的固定列顺序。"""
    fields = [
        "epoch",
        "train_loss",
        "val_main_loss",
        "mean_dice",
        "miou",
        "learning_rate",
    ]
    fields.extend(f"{name}_iou" for name in class_names)
    fields.extend(f"{name}_dice" for name in class_names)
    return fields


def select_fixed_validation_samples(
    samples: Sequence[tuple[Path, Any]],
    count: int,
) -> dict[int, str]:
    """从 Rural/Urban 各自按等间隔固定选择样本索引。

    选择只依赖 Val 文件列表，不依赖模型结果，因此不会产生挑选偏差。
    返回值的 key 是 Val Dataset 索引，value 是稳定且可用于文件名的样本名。
    """
    if not 5 <= count <= 10:
        raise ValueError("固定验证可视化样本数必须位于 5~10。")

    domain_indices: dict[str, list[int]] = {"Rural": [], "Urban": []}
    for index, (image_path, _) in enumerate(samples):
        domain = image_path.parent.parent.name
        if domain in domain_indices:
            domain_indices[domain].append(index)
    if any(not indices for indices in domain_indices.values()):
        raise ValueError("Val 必须同时包含 Rural 和 Urban，才能固定选择展示样本。")

    rural_count = (count + 1) // 2
    requested = {"Rural": rural_count, "Urban": count - rural_count}
    selected: dict[int, str] = {}
    for domain in ("Rural", "Urban"):
        indices = domain_indices[domain]
        sample_count = requested[domain]
        if sample_count > len(indices):
            raise ValueError(f"Val/{domain} 不足 {sample_count} 张，无法选择固定样本。")
        if sample_count == 1:
            positions = [len(indices) // 2]
        else:
            positions = [
                round(position * (len(indices) - 1) / (sample_count - 1))
                for position in range(sample_count)
            ]
        for position in positions:
            dataset_index = indices[position]
            image_path = samples[dataset_index][0]
            selected[dataset_index] = f"{domain}_{image_path.stem}"
    return dict(sorted(selected.items()))


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """ImageNet 标准化的 [3,H,W] Tensor 转成可保存的 RGB uint8。"""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("可视化图像必须是 [3, H, W]。")
    image = image.detach().cpu().float()
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    image = (image * std + mean).clamp(0.0, 1.0)
    array = image.permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def make_visualization_sample(
    name: str,
    image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> dict[str, Any]:
    """将一个验证样本压缩为 CPU uint8，避免长期保留大 Tensor。"""
    target_array = target.detach().cpu().numpy().astype(np.uint8, copy=True)
    prediction_array = (
        prediction.detach().cpu().numpy().astype(np.uint8, copy=True)
    )
    if target_array.shape != prediction_array.shape:
        raise ValueError("可视化的 GT 与预测尺寸必须一致。")
    return {
        "name": name,
        "rgb": denormalize_image(image),
        "target": target_array,
        "prediction": prediction_array,
    }


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """把 0~6 类别图映射到固定颜色，255 显示为灰色。"""
    if mask.ndim != 2:
        raise ValueError("mask 可视化输入必须是二维类别索引。")
    valid = (mask >= 0) & (mask < NUM_CLASSES)
    ignored = mask == IGNORE_INDEX
    if not np.all(valid | ignored):
        raise ValueError("mask 只允许类别 0~6 和 ignore_index=255。")
    colored = np.empty((*mask.shape, 3), dtype=np.uint8)
    colored[ignored] = IGNORE_COLOR
    colored[valid] = CLASS_COLORS[mask[valid]]
    return colored


def save_validation_visualizations(
    samples: Sequence[Mapping[str, Any]],
    output_dir: Path,
    epoch: int,
) -> None:
    """保存 RGB、彩色 GT、彩色预测，以及带图例的三联对比图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    legend_handles = [
        Patch(color=CLASS_COLORS[index] / 255.0, label=name)
        for index, name in enumerate(CLASS_NAMES)
    ]
    legend_handles.append(Patch(color=IGNORE_COLOR / 255.0, label="ignore"))

    for sample in samples:
        name = str(sample["name"])
        rgb = np.asarray(sample["rgb"], dtype=np.uint8)
        target_color = colorize_mask(np.asarray(sample["target"]))
        prediction_color = colorize_mask(np.asarray(sample["prediction"]))

        # 单图便于后续页面直接使用；comparison 图用于人工快速检查。
        Image.fromarray(rgb).save(output_dir / f"{name}_rgb.png")
        Image.fromarray(target_color).save(output_dir / f"{name}_gt.png")
        Image.fromarray(prediction_color).save(
            output_dir / f"{name}_prediction.png"
        )

        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        for axis, content, title in zip(
            axes,
            (rgb, target_color, prediction_color),
            ("RGB", "Ground Truth", "Prediction"),
        ):
            axis.imshow(content)
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(f"{name} | epoch {epoch}")
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.02),
        )
        figure.tight_layout(rect=(0, 0.10, 1, 0.95))
        figure.savefig(
            output_dir / f"{name}_comparison.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)


def write_json(path: Path, content: Mapping[str, Any]) -> None:
    """先写临时文件再替换，避免中断留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def prepare_metrics_csv(
    path: Path,
    start_epoch: int,
    class_names: Sequence[str] = CLASS_NAMES,
) -> list[dict[str, str]]:
    """新训练写表头；resume 时读取并核对已有 epoch，避免重复记录。"""
    fields = metric_fieldnames(class_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    if start_epoch == 0:
        with path.open("w", encoding="utf-8", newline="") as file:
            csv.DictWriter(file, fieldnames=fields).writeheader()
        return []
    if not path.is_file():
        raise FileNotFoundError(
            "resume 需要同一实验目录中的 metrics.csv；缺少历史记录时无法生成"
            "可靠的最佳 epoch、曲线和 summary。"
        )

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != fields:
            raise ValueError("已有 metrics.csv 的列与当前版本不一致。")
        rows = list(reader)
    if not rows:
        raise ValueError("resume 的 metrics.csv 没有历史 epoch 记录。")
    if rows and int(rows[-1]["epoch"]) > start_epoch:
        # 指标已写入、checkpoint 尚未保存时发生中断：checkpoint 是恢复基准。
        rows = [row for row in rows if int(row["epoch"]) <= start_epoch]
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print("提示：已按 checkpoint epoch 移除 metrics.csv 中未提交的后续记录。")
    recorded_epochs = [int(row["epoch"]) for row in rows]
    if recorded_epochs != list(range(1, start_epoch + 1)):
        raise ValueError(
            "metrics.csv 必须完整记录 epoch 1 到 checkpoint epoch，"
            f"当前记录={recorded_epochs[:3]}...{recorded_epochs[-3:]}，"
            f"checkpoint={start_epoch}。"
        )
    return rows


def append_metrics_row(
    path: Path,
    row: Mapping[str, Any],
    class_names: Sequence[str] = CLASS_NAMES,
) -> None:
    """每轮验证后立即追加并刷入磁盘。"""
    fields = metric_fieldnames(class_names)
    if set(row) != set(fields):
        raise ValueError("指标记录字段与 metrics.csv 表头不一致。")
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writerow({field: row[field] for field in fields})
        file.flush()
        os.fsync(file.fileno())


def read_metrics_csv(path: Path) -> list[dict[str, str]]:
    """读取已经落盘的每轮记录，供曲线和 summary 使用。"""
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _plot_lines(
    epochs: Sequence[int],
    series: Sequence[tuple[str, Sequence[float]]],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """统一保存训练曲线，NaN 会自然形成断点。"""
    figure, axis = plt.subplots(figsize=(8, 5))
    for label, values in series:
        axis.plot(epochs, values, label=label, linewidth=1.8)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    if len(series) > 1:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_training_curves(
    metrics_path: Path,
    curves_dir: Path,
    class_names: Sequence[str] = CLASS_NAMES,
) -> None:
    """训练结束后从 metrics.csv 生成总体与逐类曲线。"""
    rows = read_metrics_csv(metrics_path)
    if not rows:
        return
    curves_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in rows]

    _plot_lines(
        epochs,
        [
            ("train loss", [float(row["train_loss"]) for row in rows]),
            ("val main loss", [float(row["val_main_loss"]) for row in rows]),
        ],
        "Loss",
        "Training and Validation Loss",
        curves_dir / "loss_curve.png",
    )
    _plot_lines(
        epochs,
        [("mIoU", [float(row["miou"]) for row in rows])],
        "mIoU",
        "Validation mIoU",
        curves_dir / "miou_curve.png",
    )
    _plot_lines(
        epochs,
        [("mean Dice", [float(row["mean_dice"]) for row in rows])],
        "Dice",
        "Validation Mean Dice",
        curves_dir / "dice_curve.png",
    )
    _plot_lines(
        epochs,
        [("learning rate", [float(row["learning_rate"]) for row in rows])],
        "Learning rate",
        "Learning Rate",
        curves_dir / "lr_curve.png",
    )
    _plot_lines(
        epochs,
        [
            (name, [float(row[f"{name}_iou"]) for row in rows])
            for name in class_names
        ],
        "IoU",
        "Per-class Validation IoU",
        curves_dir / "per_class_iou_curve.png",
    )
    _plot_lines(
        epochs,
        [
            (name, [float(row[f"{name}_dice"]) for row in rows])
            for name in class_names
        ],
        "Dice",
        "Per-class Validation Dice",
        curves_dir / "per_class_dice_curve.png",
    )


def save_confusion_matrix(
    matrix: torch.Tensor,
    output_dir: Path,
    class_names: Sequence[str] = CLASS_NAMES,
) -> None:
    """保存原始像素计数 CSV 与按真实类别归一化的热力图。"""
    values = matrix.detach().cpu().numpy().astype(np.int64, copy=False)
    if values.shape != (len(class_names), len(class_names)):
        raise ValueError("混淆矩阵尺寸必须与类别数一致。")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "confusion_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *class_names])
        for name, row in zip(class_names, values):
            writer.writerow([name, *(int(value) for value in row)])

    row_sums = values.sum(axis=1, keepdims=True)
    normalized = np.divide(
        values,
        row_sums,
        out=np.zeros_like(values, dtype=np.float64),
        where=row_sums != 0,
    )
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(
        range(len(class_names)), labels=class_names, rotation=45, ha="right"
    )
    axis.set_yticks(range(len(class_names)), labels=class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title("Validation Confusion Matrix (row-normalized)")
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{value:.1%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value >= 0.5 else "black",
            )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def _finite_or_none(value: Any) -> Optional[float]:
    """JSON 不写 NaN/Inf；空类指标用 null 表示。"""
    number = float(value)
    return number if math.isfinite(number) else None


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    best_model_path: Path,
    status: str,
    class_names: Sequence[str] = CLASS_NAMES,
    baseline_reference_miou: float | None = None,
) -> dict[str, Any]:
    """根据已记录 epoch 汇总最佳与最终指标。最佳 epoch 按 mIoU 选择。"""
    if not rows:
        raise ValueError("至少需要一轮指标才能生成 summary.json。")
    best_row = max(rows, key=lambda row: float(row["miou"]))
    best_dice_row = max(rows, key=lambda row: float(row["mean_dice"]))
    best_loss_row = min(rows, key=lambda row: float(row["val_main_loss"]))
    final_row = rows[-1]
    per_class_best_iou = {}
    for name in class_names:
        finite_values = [
            float(row[f"{name}_iou"])
            for row in rows
            if math.isfinite(float(row[f"{name}_iou"]))
        ]
        per_class_best_iou[name] = max(finite_values) if finite_values else None

    best_miou = _finite_or_none(best_row["miou"])
    summary = {
        "status": status,
        "best_epoch": int(best_row["epoch"]),
        "best_miou": best_miou,
        "best_dice": _finite_or_none(best_dice_row["mean_dice"]),
        "best_dice_epoch": int(best_dice_row["epoch"]),
        "best_val_loss": _finite_or_none(best_loss_row["val_main_loss"]),
        "best_val_loss_epoch": int(best_loss_row["epoch"]),
        "best_miou_epoch_dice": _finite_or_none(best_row["mean_dice"]),
        "best_miou_epoch_val_loss": _finite_or_none(best_row["val_main_loss"]),
        "best_epoch_per_class_iou": {
            name: _finite_or_none(best_row[f"{name}_iou"])
            for name in class_names
        },
        "per_class_best_iou": per_class_best_iou,
        "final_epoch": int(final_row["epoch"]),
        "final_per_class_iou": {
            name: _finite_or_none(final_row[f"{name}_iou"])
            for name in class_names
        },
        "best_model": str(best_model_path.resolve()),
    }
    if baseline_reference_miou is not None:
        reference = float(baseline_reference_miou)
        if not math.isfinite(reference):
            raise ValueError("baseline_reference_miou 必须是有限数。")
        summary["baseline_reference_miou"] = reference
        summary["delta_vs_resnet34"] = (
            None if best_miou is None else best_miou - reference
        )
    return summary
