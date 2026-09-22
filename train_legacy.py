"""LoveDA U-Net 3+：原生分辨率裁块训练、整图滑窗验证、CUDA/AMP。"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import MODEL_VERSION, UNet3Plus
from utils.checkpoint import (
    capture_rng_state,
    load_model_weights,
    load_training_checkpoint,
    save_checkpoint,
)
from utils.constants import (
    CLASS_NAMES,
    DEFAULT_AUX_WEIGHTS,
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
)
from utils.dataset import LoveDADataset
from utils.device import (
    autocast_context,
    create_grad_scaler,
    print_device_info,
    resolve_device,
)
from utils.inference import sliding_window_logits
from utils.experiment import (
    append_metrics_row,
    build_summary,
    make_visualization_sample,
    prepare_metrics_csv,
    read_metrics_csv,
    save_confusion_matrix,
    save_training_curves,
    save_validation_visualizations,
    select_fixed_validation_samples,
    write_json,
)
from utils.losses import CombinedLoss, deep_supervision_loss
from utils.metrics import SegmentationMetrics


PROJECT_DIR = Path(__file__).resolve().parent


def load_auto_class_weights(stats_path: Path) -> tuple[list[float], list[float]]:
    """读取 Train 总频率，返回频率及均值为 1 的平方根倒数权重。"""
    with stats_path.open("r", encoding="utf-8-sig") as file:
        stats = json.load(file)

    if not isinstance(stats, dict) or stats.get("split") != "Train":
        raise ValueError("class_stats.json 必须是 Train 的类别统计。")
    if (
        stats.get("class_ids") != list(range(NUM_CLASSES))
        or stats.get("class_names") != list(CLASS_NAMES)
        or stats.get("ignore_index") != IGNORE_INDEX
    ):
        raise ValueError("统计文件的类别索引、名称顺序或 ignore_index 不匹配。")

    # 使用 Rural + Urban 的总像素频率，不对两个域的频率做简单平均。
    # 统计脚本的分母是 valid_pixels，原始 0 / 映射后 255 不计入频率。
    total = stats.get("total")
    if not isinstance(total, dict):
        raise ValueError("统计文件缺少 total 总计。")
    frequencies = total.get("class_frequencies")
    if not isinstance(frequencies, list) or len(frequencies) != NUM_CLASSES:
        raise ValueError("total.class_frequencies 必须包含七类频率。")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= 1
        for value in frequencies
    ):
        raise ValueError("七类频率必须为 (0, 1] 内的有限数，不能为零或 null。")
    frequencies = [float(value) for value in frequencies]
    if not math.isclose(math.fsum(frequencies), 1.0, rel_tol=1e-6):
        raise ValueError("七类有效像素频率之和应为 1，不能包含 ignore 比例。")

    # 温和地提高低频类权重；零频率明确报错，不靠随意 epsilon 产生巨大权重。
    inverse_sqrt = [1.0 / math.sqrt(value) for value in frequencies]
    mean_weight = math.fsum(inverse_sqrt) / NUM_CLASSES
    weights = [weight / mean_weight for weight in inverse_sqrt]
    return frequencies, weights


def parse_args() -> argparse.Namespace:
    """训练参数集中在此；不使用额外配置框架。"""
    parser = argparse.ArgumentParser(description="LoveDA U-Net 3+ GPU 训练")
    parser.add_argument("--data-root", type=Path, default=PROJECT_DIR / "dataset")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=PROJECT_DIR / "checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--image-size", "--crop-size", dest="image_size", type=int,
        nargs=2, default=[256, 256], metavar=("H", "W"),
        help="crop 模式的裁块尺寸，或 resize baseline 的缩放尺寸",
    )
    parser.add_argument(
        "--data-strategy", choices=("crop", "resize"), default="crop",
    )
    parser.add_argument(
        "--samples-per-image", type=int, default=4,
        help="每轮每张训练图随机取样次数；resize baseline 建议设为 1",
    )
    parser.add_argument(
        "--eval-stride", type=int, nargs=2, default=None, metavar=("H", "W"),
        help="验证滑窗步长，默认各为 image-size 的一半",
    )
    parser.add_argument(
        "--eval-tile-batch-size", type=int, default=1,
        help="一次前向的验证窗口数量；与训练 batch-size 独立",
    )
    parser.add_argument(
        "--eval-accumulate-on-device", action=argparse.BooleanOptionalAction,
        default=False, help="在计算设备融合 FP32 logits，默认仍在 CPU 融合",
    )
    parser.add_argument(
        "--augment", action=argparse.BooleanOptionalAction, default=True,
        help="Train 同步翻转/直角旋转；--no-augment 关闭",
    )
    parser.add_argument(
        "--color-jitter", type=float, default=0.0,
        help="只作用于图像的轻度颜色扰动幅度，默认关闭",
    )
    # Windows 从 0 开始；>0 时启用 persistent_workers 避免每轮重建进程。
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--aux-weights", type=float, nargs=4, default=list(DEFAULT_AUX_WEIGHTS),
        metavar=("D2", "D3", "D4", "E5"),
    )
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--class-weights", nargs="+", default=None, metavar="WEIGHT",
        help=(
            "按类别 0~6 顺序输入七个 CE/Focal 权重；或输入 auto，"
            "读取项目 class_stats.json，使用均值为 1 的平方根倒数权重；"
            "默认不使用类别权重"
        ),
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="梯度全局 L2 范数上限，设为 0 关闭",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda:0",
        help="默认 cuda:0；可选 cuda:N 或显式 cpu，不静默回退 CPU",
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="CUDA 默认启用 AMP；--no-amp 关闭；CPU 自动关闭",
    )
    parser.add_argument(
        "--visualization-interval", type=int, default=5,
        help="每隔多少个 epoch 保存固定验证样本，默认 5",
    )
    parser.add_argument(
        "--visualization-samples", type=int, default=6,
        help="固定验证可视化样本数，必须位于 5~10，默认 6",
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path, default=None)
    initialization.add_argument(
        "--init-checkpoint", type=Path, default=None,
        help="只载入模型权重，在新目录开始微调；不恢复旧优化器、LR 或轮数",
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.samples_per_image <= 0:
        parser.error("epochs、batch-size、samples-per-image 必须为正整数。")
    if args.eval_tile_batch_size <= 0:
        parser.error("eval-tile-batch-size 必须为正整数。")
    if args.num_workers < 0 or not 0 <= args.seed < 2**32:
        parser.error("num-workers 不能为负数；seed 应位于 [0, 2**32)。")
    if args.visualization_interval <= 0:
        parser.error("visualization-interval 必须为正整数。")
    if not 5 <= args.visualization_samples <= 10:
        parser.error("visualization-samples 必须位于 5~10。")
    if any(size < 32 or size % 16 != 0 for size in args.image_size):
        parser.error("image-size 的 H/W 至少为 32，且均为 16 的倍数。")
    if args.eval_stride is None:
        args.eval_stride = [size // 2 for size in args.image_size]
    if any(
        stride <= 0 or stride > size
        for stride, size in zip(args.eval_stride, args.image_size)
    ):
        parser.error("eval-stride 每一维必须大于 0，且不超过对应裁块尺寸。")
    nonnegative = (
        args.weight_decay, args.ce_weight, args.dice_weight,
        args.focal_weight, args.focal_gamma, args.grad_clip, *args.aux_weights,
    )
    if any(not math.isfinite(value) or value < 0 for value in nonnegative):
        parser.error("损失权重、衰减、gamma 和 grad-clip 必须为有限非负数。")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr 必须为有限正数。")
    if args.ce_weight + args.dice_weight + args.focal_weight <= 0:
        parser.error("至少需要启用一个分割损失项。")
    if not 0 <= args.color_jitter <= 0.5:
        parser.error("color-jitter 应在 [0, 0.5] 内。")

    args.class_frequencies = None
    if args.class_weights == ["auto"]:
        stats_path = PROJECT_DIR / "class_stats.json"
        try:
            args.class_frequencies, args.class_weights = load_auto_class_weights(
                stats_path,
            )
        except (OSError, ValueError) as error:
            parser.error(f"无法加载自动类别权重：{stats_path}，{error}")
    elif args.class_weights is not None:
        if len(args.class_weights) != NUM_CLASSES:
            parser.error("class-weights 必须是 auto 或七个有限正数。")
        try:
            args.class_weights = [float(value) for value in args.class_weights]
        except ValueError:
            parser.error("class-weights 必须是 auto 或七个有限正数。")
    # 自动权重和手动权重统一转成数值列表，供 CE/Focal 及 checkpoint 使用。
    if args.class_weights is not None and any(
        not math.isfinite(value) or value <= 0 for value in args.class_weights
    ):
        parser.error("class-weights 必须是七个有限正数。")
    output_dir = args.checkpoint_dir.resolve()
    if not output_dir.is_relative_to(PROJECT_DIR):
        parser.error("checkpoint-dir 必须位于项目目录内。")
    for data_dir in (PROJECT_DIR / "dataset", args.data_root.resolve()):
        if output_dir.is_relative_to(data_dir.resolve()):
            parser.error("不能把 checkpoint 写入 dataset 原始数据目录。")
    if args.resume is None and output_dir.exists():
        if not output_dir.is_dir() or any(
            path.name != ".gitkeep" for path in output_dir.iterdir()
        ):
            parser.error(
                "新训练/微调必须使用新的或空的 checkpoint-dir，避免覆盖已有实验。"
                "恢复同一实验请使用 --resume。"
            )
    if args.resume is not None and args.resume.resolve().parent != output_dir:
        parser.error("resume 的 checkpoint 必须位于 checkpoint-dir 中，并保留该实验日志。")
    return args


def set_seed(seed: int) -> None:
    """基础随机种子；不同 CUDA 算法/硬件仍可能有数值差异。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Windows 可 pickle 的顶层函数，给每个 worker 的增强设置种子。"""
    del worker_id  # worker 的不同种子已包含在 torch.initial_seed() 中。
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def synchronize_device(device: torch.device) -> None:
    """阶段计时前等待 CUDA 完成，避免只计入异步提交的时间。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_run_summary(
    path: Path,
    rows: list[dict[str, Any]],
    best_model_path: Path,
    status: str,
    initial_validation: dict[str, Any] | None,
) -> None:
    """训练轮次照常汇总；微调前基线单独记录，不伪造 epoch 0 的训练行。"""
    summary = build_summary(rows, best_model_path, status, CLASS_NAMES)
    if initial_validation is not None:
        initial_miou = initial_validation["miou"]
        summary.update({
            "initial_validation_miou": initial_miou,
            "initial_checkpoint": initial_validation["source_checkpoint"],
            "best_miou_gain_over_initial": summary["best_miou"] - initial_miou,
            "improved_over_initial": summary["best_miou"] > initial_miou,
        })
    write_json(path, summary)


def train_one_epoch(
    model: UNet3Plus,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    aux_weights: Sequence[float],
    scaler: Any,
    amp_enabled: bool,
    grad_clip: float,
) -> float:
    """images/masks → GPU → 五输出 → 深监督损失 → 反传 → 更新。"""
    model.train()
    loss_sum = 0.0
    sample_count = 0
    update_count = 0
    ignored_batches = 0
    overflow_batches = 0
    progress = tqdm(loader, desc="Train", leave=False)

    for images, masks in progress:
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        if not (masks != IGNORE_INDEX).any().item():
            ignored_batches += 1  # 全 ignore 时连 AdamW 权重衰减也不执行。
            continue

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            outputs = model(images)
            loss = deep_supervision_loss(outputs, masks, criterion, aux_weights)
        if not torch.isfinite(loss).item():
            raise RuntimeError("训练 loss 出现 NaN/Inf，请检查数据或关闭 AMP 定位。")

        scaler.scale(loss).backward()
        if grad_clip > 0:
            # 必须先取消梯度缩放，才可按真实范数裁剪。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip,
                error_if_nonfinite=not amp_enabled,
            )
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            overflow_batches += 1  # AMP 检测到非有限梯度，自动跳过本次更新。
        else:
            update_count += 1

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=f"{loss_sum / sample_count:.4f}")
        del outputs, loss  # 不把本批输出引用带入下一个前向。

    if sample_count == 0:
        raise RuntimeError("训练集本轮全部为 ignore，没有有效监督。")
    if update_count == 0:
        raise RuntimeError("本轮没有成功更新参数，请检查 AMP 溢出与梯度。")
    if ignored_batches or overflow_batches:
        print(f"跳过批次：全 ignore={ignored_batches}，AMP 溢出={overflow_batches}")
    return loss_sum / sample_count


@torch.no_grad()
def validate(
    model: UNet3Plus,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    image_size: tuple[int, int],
    eval_stride: tuple[int, int],
    amp_enabled: bool,
    visualization_indices: dict[int, str],
    tile_batch_size: int = 1,
    accumulate_on_device: bool = False,
) -> tuple[float, dict[str, Any], torch.Tensor, list[dict[str, Any]]]:
    """完整拼接 main logits 后算指标，每个原图像素只计一次。

    Val loader 固定 batch_size=1。crop 策略保留原图；resize baseline
    的 Dataset 已确定性缩小 image/mask，其指标相应在缩小网格上计算。
    验证损失仅为 main 的 CombinedLoss，不含训练专用的辅助损失项。
    """
    model.eval()
    loss_sum = 0.0
    image_count = 0
    metrics = SegmentationMetrics(NUM_CLASSES, IGNORE_INDEX, device=device)
    visualization_samples: list[dict[str, Any]] = []

    for dataset_index, (images, masks) in enumerate(
        tqdm(loader, desc="Val", leave=False)
    ):
        capture_sample = dataset_index in visualization_indices
        has_valid_target = (masks != IGNORE_INDEX).any().item()
        if not has_valid_target and not capture_sample:
            continue
        # 展示样本先保留 CPU 引用；后续 mask 会被移动到 GPU。
        cpu_target = masks[0] if capture_sample else None
        # 完整 RGB 留在 CPU；窗口可合并前向，融合位置由显式参数控制。
        logits = sliding_window_logits(
            model, images[0], tile_size=image_size, stride=eval_stride,
            device=device, amp_enabled=amp_enabled,
            tile_batch_size=tile_batch_size,
            accumulate_on_device=accumulate_on_device,
        )
        logits = logits.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        predictions = logits.argmax(dim=1)  # 整图 [1, H, W]，最后才取类别。
        if has_valid_target:
            loss = criterion(logits, masks)
            if not torch.isfinite(loss).item():
                raise RuntimeError("验证 loss 出现 NaN/Inf。")
            loss_sum += loss.item()
            image_count += 1
            metrics.update(predictions, masks)
        if capture_sample:
            visualization_samples.append(
                make_visualization_sample(
                    visualization_indices[dataset_index],
                    images[0],
                    cpu_target,
                    predictions[0],
                )
            )
        del logits, predictions  # 下一张图开始前释放完整输出。
        if has_valid_target:
            del loss

    if image_count == 0:
        raise RuntimeError("验证集没有有效标签，无法选择最佳模型。")
    scores = metrics.compute()
    if not math.isfinite(scores["mean_iou"]):
        raise RuntimeError("验证 mIoU 不是有限数值。")
    if len(visualization_samples) != len(visualization_indices):
        raise RuntimeError("固定验证样本没有全部生成可视化结果。")
    confusion_matrix = metrics.confusion_matrix.detach().cpu().clone()
    return (
        loss_sum / image_count,
        scores,
        confusion_matrix,
        visualization_samples,
    )


def main() -> None:
    args = parse_args()
    if args.class_frequencies is not None:
        print(f"Auto class weights: {PROJECT_DIR / 'class_stats.json'}")
        print(f"Train 有效像素频率（排除 ignore_index={IGNORE_INDEX}），权重均值为 1")
        print(f"{'Class':<18} {'Frequency':>12} {'Class weight':>14}")
        for name, frequency, weight in zip(
            CLASS_NAMES, args.class_frequencies, args.class_weights,
        ):
            print(f"{name:<18} {frequency:>12.8f} {weight:>14.6f}")
    device = resolve_device(args.device)
    amp_enabled = args.amp and device.type == "cuda"
    print_device_info(device, amp_enabled)
    set_seed(args.seed)
    image_size = tuple(args.image_size)
    eval_stride = tuple(args.eval_stride)

    train_dataset = LoveDADataset(
        args.data_root, split="Train", image_size=image_size,
        spatial_mode=args.data_strategy, augment=args.augment,
        samples_per_image=args.samples_per_image, color_jitter=args.color_jitter,
    )
    val_dataset = LoveDADataset(
        args.data_root, split="Val", image_size=image_size,
        spatial_mode="full" if args.data_strategy == "crop" else "resize",
        augment=False,
    )
    visualization_indices = select_fixed_validation_samples(
        val_dataset.samples, args.visualization_samples,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0, drop_last=False,
        worker_init_fn=seed_worker, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )

    model = UNet3Plus(in_channels=3, num_classes=NUM_CLASSES).to(device)
    initialization_info = None
    if args.init_checkpoint is not None:
        initialization_info = load_model_weights(args.init_checkpoint, model)
        print(
            f"微调初始化：{initialization_info['checkpoint']}，"
            f"来源 epoch={initialization_info['epoch']}。"
        )
        print("已加载模型权重及原始 BN 统计；使用新的 optimizer/scheduler/scaler。")
    criterion = CombinedLoss(
        num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX,
        ce_weight=args.ce_weight, dice_weight=args.dice_weight,
        class_weights=args.class_weights,
        focal_weight=args.focal_weight, focal_gamma=args.focal_gamma,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.0,
    )
    scaler = create_grad_scaler(amp_enabled)
    # 参数以普通字典存入 checkpoint；Path 转字符串以便安全加载。
    config = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "resume"
    }
    config.update({
        "initialization": initialization_info,
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "class_names": list(CLASS_NAMES),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "amp_enabled": amp_enabled,
        "model_version": MODEL_VERSION,
        "val_loss_heads": "main",
        # 同时保存更直观的字段名，便于脱离命令行参数阅读 config.json。
        "crop_size": list(image_size),
        "learning_rate": args.lr,
        "loss_weights": {
            "cross_entropy": args.ce_weight,
            "dice": args.dice_weight,
            "focal": args.focal_weight,
            "focal_gamma": args.focal_gamma,
            "deep_supervision_aux": list(args.aux_weights),
        },
        "resume_from": (
            str(args.resume.resolve()) if args.resume is not None else None
        ),
        "fixed_validation_samples": [
            {
                "dataset_index": index,
                "name": name,
                "image_path": str(val_dataset.samples[index][0].resolve()),
            }
            for index, name in visualization_indices.items()
        ],
    })
    start_epoch, best_miou = 0, -math.inf
    if args.resume is not None:
        start_epoch, best_miou = load_training_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, config, generator,
        )
        print(f"恢复到已完成 {start_epoch} 轮，历史最佳 mIoU={best_miou:.4f}")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.checkpoint_dir / "config.json"
    metrics_path = args.checkpoint_dir / "metrics.csv"
    summary_path = args.checkpoint_dir / "summary.json"
    curves_dir = args.checkpoint_dir / "curves"
    predictions_dir = args.checkpoint_dir / "predictions"
    best_model_path = args.checkpoint_dir / "best_model.pth"
    initial_validation_path = args.checkpoint_dir / "initial_validation.json"
    timing_path = args.checkpoint_dir / "timings.json"
    initial_validation = None
    timings: dict[str, Any] = {"epochs": {}}
    if args.resume is not None:
        if initial_validation_path.is_file():
            with initial_validation_path.open("r", encoding="utf-8") as file:
                initial_validation = json.load(file)
        elif config.get("initialization") is not None:
            raise FileNotFoundError("微调 resume 需要保留 initial_validation.json。")
        if timing_path.is_file():
            with timing_path.open("r", encoding="utf-8") as file:
                timings = json.load(file)
            # 以 checkpoint 的已完成 epoch 为准，裁掉更晚的耗时记录。
            timings["epochs"] = {
                key: value for key, value in timings["epochs"].items()
                if int(key) <= start_epoch
            }
    metric_rows = prepare_metrics_csv(metrics_path, start_epoch, CLASS_NAMES)
    write_json(config_path, config)
    if start_epoch >= args.epochs:
        if metric_rows:
            save_training_curves(metrics_path, curves_dir, CLASS_NAMES)
            write_run_summary(
                summary_path, metric_rows, best_model_path, "completed",
                initial_validation,
            )
        print("checkpoint 已达到设定总轮数。")
        return

    print(
        f"Train images: {len(train_dataset.samples)} | "
        f"Samples/epoch: {len(train_dataset)} | Val images: {len(val_dataset)}"
    )
    print(
        f"Strategy: {args.data_strategy} | Patch: {image_size} | "
        f"Batch: {args.batch_size} | Eval stride: {eval_stride} | "
        f"Eval tile batch: {args.eval_tile_batch_size} | "
        f"Device accumulation: {args.eval_accumulate_on_device}"
    )
    print("Train Loss 含四个辅助头；Val Main Loss 仅使用重建后的主输出。")

    if args.init_checkpoint is not None:
        # 用本次完整 Val 协议评估微调起点；不拿旧日志/诊断子集分数代替。
        # 此处尚未训练，不把它写进 metrics.csv 的 epoch 序列。
        print("先评估微调起点：完整 Val，仅前向，不更新参数或 BN 统计。")
        synchronize_device(device)
        initial_start = time.perf_counter()
        initial_loss, initial_scores, _, _ = validate(
            model, val_loader, criterion, device, image_size, eval_stride,
            amp_enabled, {}, args.eval_tile_batch_size,
            args.eval_accumulate_on_device,
        )
        synchronize_device(device)
        initial_validation = {
            "source_checkpoint": initialization_info["checkpoint"],
            "source_epoch": initialization_info["epoch"],
            "val_images": len(val_dataset),
            "full_validation": True,
            "image_size": list(image_size),
            "eval_stride": list(eval_stride),
            "eval_tile_batch_size": args.eval_tile_batch_size,
            "eval_accumulate_on_device": args.eval_accumulate_on_device,
            "amp_enabled": amp_enabled,
            "val_main_loss": initial_loss,
            "miou": initial_scores["mean_iou"],
            "mean_dice": initial_scores["mean_dice"],
            "per_class_iou": {
                name: (
                    float(value) if math.isfinite(float(value)) else None
                )
                for name, value in zip(
                    CLASS_NAMES, initial_scores["per_class_iou"].tolist(),
                )
            },
        }
        timings["initial_validation_seconds"] = time.perf_counter() - initial_start
        write_json(initial_validation_path, initial_validation)
        write_json(timing_path, timings)
        print(f"微调起点完整 Val mIoU={initial_validation['miou']:.4f}")

    for epoch in range(start_epoch, args.epochs):
        synchronize_device(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.aux_weights,
            scaler, amp_enabled, args.grad_clip,
        )
        synchronize_device(device)
        train_seconds = time.perf_counter() - train_start
        # 验证不需要训练梯度，及时释放最后一批 parameter.grad 的显存。
        optimizer.zero_grad(set_to_none=True)
        validation_start = time.perf_counter()
        val_loss, scores, confusion_matrix, visualization_samples = validate(
            model, val_loader, criterion, device, image_size, eval_stride,
            amp_enabled, visualization_indices,
            args.eval_tile_batch_size, args.eval_accumulate_on_device,
        )
        synchronize_device(device)
        validation_seconds = time.perf_counter() - validation_start
        logging_start = time.perf_counter()
        current_miou = scores["mean_iou"]
        is_best = current_miou > best_miou
        best_miou = max(best_miou, current_miou)
        scheduler.step()  # 本轮 optimizer 更新完成后，准备下一轮 LR。

        checkpoint = {
            "model_version": MODEL_VERSION,
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if amp_enabled else None,
            "best_miou": best_miou,
            "val_loss": val_loss,
            "val_miou": current_miou,
            "val_dice": scores["mean_dice"],
            "config": config,
            "rng_state": capture_rng_state(generator),
        }
        per_class_iou = [
            scores["per_class_iou"][index].item()
            for index in range(NUM_CLASSES)
        ]
        per_class_dice = [
            scores["per_class_dice"][index].item()
            for index in range(NUM_CLASSES)
        ]
        metric_row: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_main_loss": val_loss,
            "mean_dice": scores["mean_dice"],
            "miou": current_miou,
            "learning_rate": learning_rate,
        }
        metric_row.update({
            f"{name}_iou": per_class_iou[index]
            for index, name in enumerate(CLASS_NAMES)
        })
        metric_row.update({
            f"{name}_dice": per_class_dice[index]
            for index, name in enumerate(CLASS_NAMES)
        })
        append_metrics_row(metrics_path, metric_row, CLASS_NAMES)
        # CSV 先落盘；checkpoint 随后保存同一 epoch。若保存中断，resume
        # 会以 checkpoint 为准裁掉 CSV 中多出的一轮，避免重复 epoch。
        save_checkpoint(args.checkpoint_dir / "last_checkpoint.pth", checkpoint)
        if is_best:
            save_checkpoint(best_model_path, checkpoint)

        # 根目录混淆矩阵始终对应最近完成的一轮完整 Val。
        save_confusion_matrix(confusion_matrix, args.checkpoint_dir, CLASS_NAMES)
        should_save_periodic = (
            (epoch + 1) % args.visualization_interval == 0
            or epoch + 1 == args.epochs
        )
        if should_save_periodic:
            save_validation_visualizations(
                visualization_samples,
                predictions_dir / f"epoch_{epoch + 1:04d}",
                epoch + 1,
            )
        if is_best:
            save_validation_visualizations(
                visualization_samples,
                predictions_dir / "best",
                epoch + 1,
            )

        # summary 每轮更新，训练意外中断时仍保留已完成 epoch 的汇总。
        metric_rows = read_metrics_csv(metrics_path)
        write_run_summary(
            summary_path, metric_rows, best_model_path,
            "completed" if epoch + 1 == args.epochs else "in_progress",
            initial_validation,
        )
        synchronize_device(device)
        logging_seconds = time.perf_counter() - logging_start
        timings["epochs"][str(epoch + 1)] = {
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "logging_seconds": logging_seconds,
            "peak_cuda_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda" else None
            ),
        }
        write_json(timing_path, timings)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Main Loss: {val_loss:.4f} | Dice: {scores['mean_dice']:.4f} | "
            f"mIoU: {current_miou:.4f} | LR: {learning_rate:.6g}"
        )
        print(
            f"耗时：Train {train_seconds:.1f}s | Val {validation_seconds:.1f}s | "
            f"保存/绘图 {logging_seconds:.1f}s"
        )
        if initial_validation is not None:
            gain = (current_miou - initial_validation["miou"]) * 100
            print(f"相对微调起点完整 Val：本轮 mIoU {gain:+.2f} 个百分点")
        for index, name in enumerate(CLASS_NAMES):
            iou = per_class_iou[index]
            dice = per_class_dice[index]
            if math.isnan(iou):
                print(f"  {index} {name}: IoU=N/A | Dice=N/A（该类无真实/预测像素）")
            else:
                print(f"  {index} {name}: IoU={iou:.4f} | Dice={dice:.4f}")
        if is_best:
            print(f"已保存 best_model.pth，mIoU={best_miou:.4f}")

    # 曲线只在训练完整结束后生成，不占用每个 batch 或每个 epoch 的绘图时间。
    save_training_curves(metrics_path, curves_dir, CLASS_NAMES)
    metric_rows = read_metrics_csv(metrics_path)
    write_run_summary(
        summary_path, metric_rows, best_model_path, "completed",
        initial_validation,
    )
    print(f"实验指标与可视化已保存到：{args.checkpoint_dir.resolve()}")


if __name__ == "__main__":
    # Windows 多进程必须保留入口保护，导入此模块不会启动训练。
    main()
