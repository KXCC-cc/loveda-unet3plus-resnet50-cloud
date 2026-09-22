"""用已有权重比较滑窗重叠与 BN 校准；不反传、不续训、不覆盖原实验。"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models import MODEL_VERSION, UNet3Plus
from utils.constants import (
    CLASS_NAMES, IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES,
)
from utils.dataset import LoveDADataset
from utils.device import autocast_context, print_device_info, resolve_device
from utils.experiment import (
    make_visualization_sample,
    save_confusion_matrix,
    save_validation_visualizations,
    select_fixed_validation_samples,
    write_json,
)
from utils.inference import sliding_window_logits
from utils.losses import CombinedLoss
from utils.metrics import SegmentationMetrics
from utils.transforms import validate_image_size


PROJECT_DIR = Path(__file__).resolve().parent
MODES = ("baseline", "overlap", "bn", "bn_overlap")
DOMAINS = ("Rural", "Urban")


def parse_args() -> argparse.Namespace:
    """诊断参数独立于 train.py，不改变已有训练入口的默认行为。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_DIR / "dataset")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=None,
        help="默认沿用 checkpoint；CPU 自动关闭 AMP",
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--val-samples", type=int, default=100,
        help="固定抽取的 Val 图像总数，两域尽量等量；0 表示完整 Val",
    )
    parser.add_argument(
        "--overlap-stride", type=int, nargs=2, default=None, metavar=("H", "W"),
        help="重叠组步长，默认 checkpoint 裁块尺寸的一半",
    )
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument(
        "--accumulate-on-device", action=argparse.BooleanOptionalAction,
        default=True, help="在计算设备上融合 FP32 logits，结束后传回 CPU",
    )
    parser.add_argument("--bn-batches", type=int, default=128)
    parser.add_argument("--bn-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.val_samples != 0 and args.val_samples < 6:
        parser.error("val-samples 应为 0（完整 Val）或至少 6。")
    if min(args.tile_batch_size, args.bn_batches, args.bn_batch_size) <= 0:
        parser.error("tile-batch-size、bn-batches、bn-batch-size 必须为正整数。")
    if args.num_workers < 0 or not 0 <= args.seed < 2**32:
        parser.error("num-workers 不能为负数；seed 应位于 [0, 2**32)。")
    return args


def seed_worker(worker_id: int) -> None:
    """给 Train 裁块的 Python/NumPy 随机数设置 worker 种子。"""
    del worker_id
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def select_indices(dataset: LoveDADataset, count: int, seed: int) -> list[int]:
    """仅按文件列表和种子取样，不根据类别标签或预测好坏筛选。"""
    if count == 0:
        return list(range(len(dataset.samples)))
    rng = random.Random(seed)
    selected = []
    for domain, requested in zip(DOMAINS, ((count + 1) // 2, count // 2)):
        candidates = [
            index for index, (path, _) in enumerate(dataset.samples)
            if path.parent.parent.name == domain
        ]
        if requested > len(candidates):
            raise ValueError(
                f"{dataset.split}/{domain} 只有 {len(candidates)} 张，"
                f"不能无重复地抽取 {requested} 张，请降低样本数。"
            )
        selected.extend(rng.sample(candidates, requested))
    return sorted(selected)


def sample_manifest(
    dataset: LoveDADataset, indices: list[int],
) -> list[dict[str, Any]]:
    """记录相对文件名，方便不同实验确认使用了同一批图片。"""
    return [
        {
            "dataset_index": index,
            "image": dataset.samples[index][0].relative_to(dataset.root).as_posix(),
            "mask": dataset.samples[index][1].relative_to(dataset.root).as_posix(),
        }
        for index in indices
    ]


def synchronize(device: torch.device) -> None:
    """阶段计时前后等待 GPU 完成，避免只量到异步提交时间。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def finite_or_none(value: float) -> float | None:
    """无真实/预测像素的类别用 JSON null、CSV 空值表示。"""
    value = float(value)
    return value if math.isfinite(value) else None


@torch.no_grad()
def calibrate_batch_norm(
    model: UNet3Plus,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    """仅用 Train 的固定数量随机裁块，重新估计 BN 的均值与方差。

    卷积权重、BN 的 affine 参数均不更新。只让 BN 进入 train 模式，
    其余模块保持 eval；无 optimizer、无 backward，也不读取 Val 进行校准。
    momentum=None 表示等权累计各 batch 的统计，而非只强调最后几个 batch。
    """
    model.eval()
    layers = [
        module for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
        and module.track_running_stats
    ]
    if not layers:
        raise ValueError("模型没有可校准的 BatchNorm2d。")
    original_momenta = [layer.momentum for layer in layers]
    synchronize(device)
    start = time.perf_counter()
    batches = 0
    samples = 0
    try:
        for layer in layers:
            layer.reset_running_stats()
            layer.momentum = None
            layer.train()
        for images, _ in tqdm(loader, desc="BN calibration (Train only)"):
            images = images.to(device, non_blocking=device.type == "cuda")
            with autocast_context(device, amp_enabled):
                outputs = model(images, return_aux=False)
            batches += 1
            samples += images.shape[0]
            del images, outputs
        for layer in layers:
            if (
                not torch.isfinite(layer.running_mean).all().item()
                or not torch.isfinite(layer.running_var).all().item()
            ):
                raise RuntimeError("BN 统计出现 NaN/Inf，请关闭 AMP 后重新诊断。")
    finally:
        for layer, momentum in zip(layers, original_momenta):
            layer.momentum = momentum
        model.eval()
    synchronize(device)
    return {
        "seconds": time.perf_counter() - start,
        "batches": batches,
        "samples": samples,
        "bn_layers": len(layers),
        "source_split": "Train",
    }


@torch.no_grad()
def evaluate_mode(
    model: UNet3Plus,
    loader: DataLoader,
    samples: list[tuple[Path, Any]],
    criterion: CombinedLoss,
    device: torch.device,
    tile_size: tuple[int, int],
    stride: tuple[int, int],
    amp_enabled: bool,
    args: argparse.Namespace,
    mode: str,
    output_dir: Path,
    epoch: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """同一 Val 子集分别累计 Overall/Rural/Urban 指标，复用现有损失。"""
    model.eval()
    metrics = {
        domain: SegmentationMetrics(device=device) for domain in DOMAINS
    }
    loss_sums = dict.fromkeys(DOMAINS, 0.0)
    valid_images = dict.fromkeys(DOMAINS, 0)
    image_counts = dict.fromkeys(DOMAINS, 0)
    capture = select_fixed_validation_samples(samples, 6)
    visualizations = []
    synchronize(device)
    start = time.perf_counter()

    for position, (images, masks) in enumerate(tqdm(loader, desc=mode)):
        domain = samples[position][0].parent.parent.name
        image_counts[domain] += 1
        has_valid_target = (masks != IGNORE_INDEX).any().item()
        if not has_valid_target and position not in capture:
            continue
        logits = sliding_window_logits(
            model, images[0], tile_size=tile_size, stride=stride,
            device=device, amp_enabled=amp_enabled,
            tile_batch_size=args.tile_batch_size,
            accumulate_on_device=args.accumulate_on_device,
        ).to(device, non_blocking=device.type == "cuda")
        targets = masks.to(device, non_blocking=device.type == "cuda")
        predictions = logits.argmax(dim=1)
        if has_valid_target:
            loss = criterion(logits, targets)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"{mode} 的验证损失出现 NaN/Inf。")
            loss_sums[domain] += loss.item()
            valid_images[domain] += 1
            metrics[domain].update(predictions, targets)
            del loss
        if position in capture:
            visualizations.append(make_visualization_sample(
                capture[position], images[0], masks[0], predictions[0],
            ))
        del logits, targets, predictions

    synchronize(device)
    evaluation_seconds = time.perf_counter() - start
    if sum(valid_images.values()) == 0:
        raise RuntimeError("所选 Val 图片没有有效标签。")
    overall = SegmentationMetrics(device=device)
    for domain in DOMAINS:
        overall.confusion_matrix += metrics[domain].confusion_matrix
    metrics = {"Overall": overall, **metrics}
    loss_sums["Overall"] = sum(loss_sums.values())
    valid_images["Overall"] = sum(valid_images.values())
    image_counts["Overall"] = sum(image_counts.values())

    # 绘图/写盘单独计时，不混入模型评估耗时；每组只保存相同的 6 张图。
    artifact_start = time.perf_counter()
    rows = []
    for domain, meter in metrics.items():
        scores = meter.compute()
        row = {
            "mode": mode,
            "domain": domain,
            "images": image_counts[domain],
            "valid_images": valid_images[domain],
            "val_main_loss": (
                loss_sums[domain] / valid_images[domain]
                if valid_images[domain] else None
            ),
            "miou": finite_or_none(scores["mean_iou"]),
            "mean_dice": finite_or_none(scores["mean_dice"]),
        }
        for metric_name in ("iou", "dice"):
            for index, name in enumerate(CLASS_NAMES):
                row[f"{name}_{metric_name}"] = finite_or_none(
                    scores[f"per_class_{metric_name}"][index].item(),
                )
        rows.append(row)
        save_confusion_matrix(meter.confusion_matrix, output_dir / domain)
    save_validation_visualizations(
        visualizations, output_dir / "predictions", epoch,
    )
    return rows, {
        "evaluation_seconds": evaluation_seconds,
        "artifact_seconds": time.perf_counter() - artifact_start,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    load_start = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError("checkpoint 与当前五输出 U-Net 3+ 版本不一致。")
    config = checkpoint["config"]
    expected_metadata = {
        "class_names": list(CLASS_NAMES), "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX, "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD), "data_strategy": "crop",
    }
    for key, expected in expected_metadata.items():
        if config.get(key) != expected:
            raise ValueError(f"诊断要求 checkpoint 的 {key}={expected!r}。")
    tile_size = validate_image_size(tuple(config["image_size"]))
    baseline_stride = validate_image_size(tuple(config["eval_stride"]))
    overlap_stride = validate_image_size(tuple(
        args.overlap_stride or [size // 2 for size in tile_size]
    ))
    if any(size < 32 or size % 16 for size in tile_size):
        raise ValueError("checkpoint 的裁块高宽须至少为 32，且是 16 的倍数。")
    for stride in (baseline_stride, overlap_stride):
        if any(step > size for step, size in zip(stride, tile_size)):
            raise ValueError("滑窗步长不能超过对应裁块尺寸。")
    if overlap_stride == baseline_stride:
        print("提示：baseline 与 overlap 步长相同，两组只作为重复对照。")
    amp_requested = (
        config.get("amp_enabled", False) if args.amp is None else args.amp
    )
    amp_enabled = device.type == "cuda" and bool(amp_requested)
    print_device_info(device, amp_enabled)
    model = UNet3Plus(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    epoch = int(checkpoint["epoch"])
    del checkpoint  # 不保留无用的 optimizer/scaler 状态。
    model = model.to(device).eval()
    model.requires_grad_(False)
    criterion = CombinedLoss(
        ce_weight=config["ce_weight"], dice_weight=config["dice_weight"],
        class_weights=config["class_weights"],
        focal_weight=config["focal_weight"], focal_gamma=config["focal_gamma"],
    ).to(device)
    synchronize(device)
    load_seconds = time.perf_counter() - load_start

    val_dataset = LoveDADataset(
        args.data_root, split="Val", image_size=tile_size,
        spatial_mode="full", augment=False,
    )
    val_indices = select_indices(val_dataset, args.val_samples, args.seed)
    val_samples = [val_dataset.samples[index] for index in val_indices]
    # 提前检查两域都至少有 3 张，保证四组对照保存相同 6 张可视化。
    select_fixed_validation_samples(val_samples, 6)
    val_loader = DataLoader(
        Subset(val_dataset, val_indices), batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    bn_loader = None
    train_manifest = []
    if any(mode in args.modes for mode in ("bn", "bn_overlap")):
        train_dataset = LoveDADataset(
            args.data_root, split="Train", image_size=tile_size,
            spatial_mode="crop", augment=False, samples_per_image=1,
        )
        train_indices = select_indices(
            train_dataset, args.bn_batches * args.bn_batch_size, args.seed,
        )
        # 打散 Rural/Urban，避免所有乡村或城市裁块集中在校准的末段。
        random.Random(args.seed).shuffle(train_indices)
        train_manifest = sample_manifest(train_dataset, train_indices)
        bn_loader = DataLoader(
            Subset(train_dataset, train_indices), batch_size=args.bn_batch_size,
            shuffle=False, num_workers=args.num_workers,
            pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(args.seed),
        )

    output_dir = (args.output_dir or (
        PROJECT_DIR / "checkpoints"
        / f"diagnostics_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )).resolve()
    if not output_dir.is_relative_to(PROJECT_DIR):
        raise ValueError("output-dir 必须位于当前项目中。")
    for protected in (args.data_root.resolve(), PROJECT_DIR / "dataset"):
        if output_dir.is_relative_to(protected.resolve()):
            raise ValueError("诊断输出不能写入原始 dataset。")
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"诊断结果目录：{output_dir}", flush=True)
    print(
        f"权重来自 epoch {epoch} | Val 图片 {len(val_indices)} | "
        f"tile={tile_size} | baseline stride={baseline_stride} | "
        f"overlap stride={overlap_stride}"
    )
    write_json(output_dir / "config.json", {
        "arguments": {
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "output_dir": str(output_dir),
        "checkpoint_config": config,
        "checkpoint_epoch": epoch,
        "amp_enabled": amp_enabled,
        "baseline_stride": list(baseline_stride),
        "overlap_stride": list(overlap_stride),
        "torch_version": str(torch.__version__),
    })
    write_json(output_dir / "samples.json", {
        "validation": sample_manifest(val_dataset, val_indices),
        "bn_calibration_train": train_manifest,
        "seed": args.seed,
        "bn_geometry": "native random crop; no flips/rotation/color jitter",
        "note": "相同 seed、worker 数和文件列表重建校准裁块；不依赖 Val 标签选择。",
    })

    results = []
    timings: dict[str, Any] = {"load_checkpoint_seconds": load_seconds}
    calibrated = False
    requested_modes = [mode for mode in MODES if mode in args.modes]
    for mode in requested_modes:
        if mode in ("bn", "bn_overlap") and not calibrated:
            # 独立 generator + 重置主进程随机数，确保省略 baseline 组时
            # 校准仍使用同一批裁块；校准只做一次，两个 BN 组共用统计。
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            timings["bn_calibration"] = calibrate_batch_norm(
                model, bn_loader, device, amp_enabled,
            )
            calibrated = True
        stride = (
            overlap_stride if mode in ("overlap", "bn_overlap")
            else baseline_stride
        )
        rows, stage_timing = evaluate_mode(
            model, val_loader, val_samples, criterion, device, tile_size,
            stride, amp_enabled, args, mode, output_dir / mode, epoch,
        )
        results.extend(rows)
        timings[mode] = stage_timing
        # 每完成一组就落盘；即使用户中途停止，也保留已完成的对照结果。
        temporary_csv = output_dir / "comparison.csv.tmp"
        with temporary_csv.open(
            "w", encoding="utf-8", newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        temporary_csv.replace(output_dir / "comparison.csv")
        write_json(output_dir / "report.json", {
            "status": "completed" if mode == requested_modes[-1] else "in_progress",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": epoch,
            "validation_images": len(val_indices),
            "full_validation": args.val_samples == 0,
            "results": results,
            "timings": timings,
            "note": "子集仅用于同批对照，不能与全量 Val 分数直接比较；BN 校准不保证提升。",
        })
        overall = rows[0]
        print(
            f"{mode}: mIoU={overall['miou']:.4f}, "
            f"Dice={overall['mean_dice']:.4f}, "
            f"Val Main Loss={overall['val_main_loss']:.4f}, "
            f"评估耗时={stage_timing['evaluation_seconds']:.1f}s",
            flush=True,
        )
    print(f"对照完成：{output_dir / 'comparison.csv'}")
    print("原始 checkpoint 未覆盖；校准的 BN 统计只在本次进程内使用。")


if __name__ == "__main__":
    main()
