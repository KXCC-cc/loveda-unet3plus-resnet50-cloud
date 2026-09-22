"""配置化训练引擎：AMP、梯度累积、按 update 调度和完整 Val。"""

from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import RESNET_MODEL_VERSION, build_model
from utils.checkpoint import (
    capture_rng_state,
    load_experiment_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from utils.class_weights import load_inverse_sqrt_weights
from utils.constants import (
    CLASS_NAMES,
    DEFAULT_AUX_WEIGHTS,
    RESNET34_REFERENCE_MIOU,
)
from utils.dataset import LoveDADataset
from utils.device import autocast_context, create_grad_scaler, print_device_info, resolve_device
from utils.experiment import (
    append_metrics_row,
    build_summary,
    make_visualization_sample,
    metric_fieldnames,
    save_confusion_matrix,
    save_training_curves,
    save_validation_visualizations,
    select_fixed_validation_samples,
    write_json,
)
from utils.inference import multi_scale_sliding_window_logits
from utils.losses import CombinedLoss, deep_supervision_loss
from utils.schedulers import UpdateLRScheduler

from .evaluator import evaluate_full_validation


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _make_datasets(config: dict) -> tuple[LoveDADataset, LoveDADataset]:
    data = config["data"]
    augmentation = data.get("augmentation", {})
    class_aware = data.get("class_aware_crop", {})
    train_dataset = LoveDADataset(
        root=data["root"],
        split="Train",
        image_size=tuple(data.get("crop_size", (512, 512))),
        spatial_mode=data.get("strategy", "crop"),
        augment=True,
        samples_per_image=int(data.get("samples_per_image", 1)),
        color_jitter=float(augmentation.get("color_jitter", 0.0)),
        train_scales=tuple(data.get("train_scales", (1.0,))),
        horizontal_flip_probability=float(augmentation.get("horizontal_flip", 0.5)),
        vertical_flip_probability=float(augmentation.get("vertical_flip", 0.5)),
        rotate90_probability=float(augmentation.get("rotate90", 1.0)),
        class_aware_crop_probability=float(class_aware.get("probability", 0.0)),
        class_aware_classes=tuple(class_aware.get("classes", (1, 2, 3, 4))),
        min_target_pixels=int(class_aware.get("min_target_pixels", 512)),
        max_crop_attempts=int(class_aware.get("max_attempts", 8)),
        augmentation_mode=augmentation.get("mode", "independent"),
        one_of_probability=float(augmentation.get("one_of_probability", 0.75)),
    )
    val_dataset = LoveDADataset(
        root=data["root"], split="Val", spatial_mode="full", augment=False
    )
    return train_dataset, val_dataset


def _make_loaders(config: dict, generator: torch.Generator):
    train_dataset, val_dataset = _make_datasets(config)
    loader = config["data"]["loader"]
    workers = int(loader.get("num_workers", 0))
    common = {
        "num_workers": workers,
        "pin_memory": bool(loader.get("pin_memory", True)),
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if workers > 0:
        common["persistent_workers"] = bool(loader.get("persistent_workers", True))
        common["prefetch_factor"] = int(loader.get("prefetch_factor", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(loader.get("batch_size", 1)),
        shuffle=True,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=common["pin_memory"],
        persistent_workers=common.get("persistent_workers", False),
        prefetch_factor=common.get("prefetch_factor") if workers > 0 else None,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def _resolve_class_weights(config: dict) -> tuple[list[float] | None, list[float] | None]:
    value = config["loss"].get("class_weights")
    if value is None or value == "none":
        return None, None
    if value == "auto":
        return load_inverse_sqrt_weights(Path(config["loss"]["class_stats_path"]))
    weights = [float(item) for item in value]
    if len(weights) != 7:
        raise ValueError("手动 class_weights 必须恰好包含 7 个数。")
    return None, weights


def _make_optimizer(config: dict, model: torch.nn.Module, effective_batch: int):
    settings = config["optimizer"]
    base_lr = float(settings["lr"])
    if settings.get("lr_scale", "none") == "linear":
        base_lr *= effective_batch / int(settings.get("reference_batch_size", 16))
    elif settings.get("lr_scale", "none") != "none":
        raise ValueError("optimizer.lr_scale 仅支持 none 或 linear。")

    differential = settings.get("differential_lr")
    if differential and hasattr(model, "encoder"):
        encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
        encoder_parameters = [p for p in model.parameters() if id(p) in encoder_ids and p.requires_grad]
        decoder_parameters = [p for p in model.parameters() if id(p) not in encoder_ids and p.requires_grad]
        if not encoder_parameters or not decoder_parameters:
            raise ValueError("差分学习率要求 encoder 和 decoder 参数组都非空。")
        encoder_lr = float(differential["encoder_lr"])
        decoder_lr = float(differential["decoder_lr"])
        if encoder_lr <= 0 or decoder_lr <= 0:
            raise ValueError("encoder_lr 和 decoder_lr 必须大于 0。")
        param_groups = [
            {"params": encoder_parameters, "lr": encoder_lr, "name": "encoder"},
            {"params": decoder_parameters, "lr": decoder_lr, "name": "decoder"},
        ]
    else:
        param_groups = [{
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": base_lr,
            "name": "all",
        }]

    name = settings.get("name", "sgd").lower()
    if name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=base_lr,
            momentum=float(settings.get("momentum", 0.9)),
            weight_decay=float(settings.get("weight_decay", 1e-4)),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=float(settings.get("weight_decay", 1e-4)),
        )
    else:
        raise ValueError("optimizer.name 仅支持 sgd 或 adamw。")
    return optimizer, base_lr


def _optimizer_group_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """返回实际参数组初始 LR，供 config.json 与日志核查。"""
    result = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        if name in result:
            raise ValueError(f"optimizer 参数组名称重复：{name}")
        result[name] = float(group["lr"])
    return result


def _checkpoint_payload(
    model, optimizer, scheduler, scaler, config, generator,
    epoch: int, micro_step: int, optimizer_step: int, best_miou: float,
) -> dict:
    return {
        "model_version": RESNET_MODEL_VERSION if hasattr(model, "encoder") else "unet3plus_loveda_v2",
        "epoch": epoch,
        "micro_step": micro_step,
        "optimizer_step": optimizer_step,
        "best_miou": best_miou,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
        "rng_state": capture_rng_state(generator),
        "config": config,
    }


def _save_epoch_checkpoint(
    output_dir: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    config,
    generator,
    epoch: int,
    micro_step: int,
    optimizer_step: int,
    best_miou: float,
    save_best: bool = False,
) -> dict:
    """在完整 epoch 边界保存可恢复状态；验证提升时先写 best，再写 last。"""
    payload = _checkpoint_payload(
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        generator,
        epoch,
        micro_step,
        optimizer_step,
        best_miou,
    )
    if save_best:
        save_checkpoint(output_dir / "best_model.pth", payload)
    save_checkpoint(output_dir / "last_checkpoint.pth", payload)
    print(
        f"Saved epoch checkpoint: epoch={epoch}, "
        f"optimizer={optimizer_step} -> {output_dir / 'last_checkpoint.pth'}"
    )
    return payload


def _save_fixed_predictions(model, dataset, selected, config, device, output_dir, epoch):
    samples = []
    validation = config["validation"]
    for index, name in selected.items():
        image, target = dataset[index]
        logits = multi_scale_sliding_window_logits(
            model=model,
            image=image,
            scales=(1.0,),
            tile_size=tuple(validation.get("tile_size", (512, 512))),
            stride=tuple(validation.get("stride", (512, 512))),
            device=device,
            amp_enabled=bool(config["runtime"].get("amp", True) and device.type == "cuda"),
            tile_batch_size=int(validation.get("patch_batch_size", 1)),
        )
        samples.append(make_visualization_sample(name, image, target, logits.argmax(1)[0]))
    save_validation_visualizations(samples, output_dir, epoch)


def _write_metrics_header(path: Path, resume: bool) -> None:
    if resume:
        if not path.is_file():
            raise FileNotFoundError("resume 时缺少 metrics.csv。")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        csv.DictWriter(file, fieldnames=metric_fieldnames()).writeheader()


def _prepare_output_directory(output_dir: Path, resume: Path | None) -> None:
    """在任何实验文件写入前检查输出目录，避免误覆盖已有 run。"""
    if resume is not None:
        resume_path = resume.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint 不存在：{resume_path}")
        if resume_path.parent != output_dir:
            raise ValueError(
                "resume checkpoint 必须位于本次 run 的 output_dir 中："
                f"checkpoint={resume_path}，output_dir={output_dir}"
            )
        return

    if output_dir.exists():
        existing_items = sorted(path.name for path in output_dir.iterdir())
        if existing_items:
            preview = ", ".join(existing_items[:8])
            if len(existing_items) > 8:
                preview += ", ..."
            raise FileExistsError(
                "新训练拒绝写入非空实验目录，以免覆盖已有权重和指标："
                f"{output_dir}（已有：{preview}）。请修改 run.output_dir；"
                "如需断点续训，请同时使用原 output_dir 和 --resume。"
            )


def run_training(config: dict[str, Any], resume: Path | None = None) -> None:
    seed = int(config.get("seed", 42))
    _seed_everything(seed)
    runtime = config["runtime"]
    device = resolve_device(runtime.get("device", "auto"))
    amp_enabled = bool(runtime.get("amp", True) and device.type == "cuda")
    print_device_info(device, amp_enabled)
    torch.backends.cudnn.benchmark = bool(runtime.get("cudnn_benchmark", True))

    output_dir = Path(config["run"]["output_dir"]).resolve()
    _prepare_output_directory(output_dir, resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "curves").mkdir(exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, train_loader, val_loader = _make_loaders(config, generator)
    physical_batch = int(config["data"]["loader"].get("batch_size", 1))
    accumulation = int(config["training"].get("accumulation_steps", 1))
    effective_batch = physical_batch * accumulation
    updates_per_epoch = len(train_loader) // accumulation
    if updates_per_epoch < 1:
        raise ValueError("训练 loader 不足以完成一次梯度累积。")
    max_updates = config["training"].get("max_iters")
    if max_updates is None:
        max_updates = int(config["training"]["epochs"]) * updates_per_epoch
    max_updates = int(max_updates)

    model = build_model(config["model"]).to(device)
    if runtime.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    actual_weights = getattr(getattr(model, "encoder", None), "weights_name", None)
    config["model"]["resolved_weights"] = actual_weights

    frequencies, class_weights = _resolve_class_weights(config)
    criterion = CombinedLoss(
        ce_weight=float(config["loss"].get("ce_weight", 1.0)),
        dice_weight=float(config["loss"].get("dice_weight", 0.0)),
        focal_weight=float(config["loss"].get("focal_weight", 0.0)),
        focal_gamma=float(config["loss"].get("focal_gamma", 2.0)),
        class_weights=class_weights,
    ).to(device)
    optimizer, resolved_lr = _make_optimizer(config, model, effective_batch)
    resolved_group_lrs = _optimizer_group_lrs(optimizer)
    scheduler = UpdateLRScheduler(
        optimizer,
        max_updates=max_updates,
        name=config["scheduler"].get("name", "poly"),
        power=float(config["scheduler"].get("power", 0.9)),
        warmup_updates=int(config["scheduler"].get("warmup_updates", 0)),
        warmup_start_factor=float(
            config["scheduler"].get("warmup_start_factor", 0.01)
        ),
    )
    scaler = create_grad_scaler(amp_enabled)

    config["resolved"] = {
        "physical_batch": physical_batch,
        "accumulation_steps": accumulation,
        "effective_batch": effective_batch,
        "updates_per_epoch": updates_per_epoch,
        "max_optimizer_steps": max_updates,
        "learning_rate": resolved_lr,
        "optimizer_group_lrs": resolved_group_lrs,
        "encoder_lr": resolved_group_lrs.get("encoder", resolved_group_lrs.get("all")),
        "decoder_lr": resolved_group_lrs.get("decoder", resolved_group_lrs.get("all")),
        "scheduler": config["scheduler"].get("name", "poly"),
        "warmup_updates": scheduler.warmup_updates,
        "warmup_start_factor": scheduler.warmup_start_factor,
        "baseline_reference_miou": float(
            config.get("baseline_reference_miou", RESNET34_REFERENCE_MIOU)
        ),
        "class_weights": class_weights,
        "class_frequencies": frequencies,
    }
    print(
        f"Train images: {len(train_dataset.samples)} | Val images: {len(val_dataset.samples)} | "
        f"physical batch={physical_batch}, accumulation={accumulation}, effective batch={effective_batch}"
    )
    print(f"Max optimizer steps: {max_updates} | resolved lr: {resolved_lr:g}")
    print(
        "Optimizer parameter-group base LRs: "
        + ", ".join(f"{name}={lr:g}" for name, lr in resolved_group_lrs.items())
    )
    if scheduler.warmup_updates:
        print(
            f"LR warmup: {scheduler.warmup_updates} optimizer updates | "
            f"start factor={scheduler.warmup_start_factor:g}"
        )
    print(f"Backbone weights: {actual_weights}")
    if class_weights is not None:
        print("Class weights:")
        for index, name in enumerate(CLASS_NAMES):
            frequency = "manual" if frequencies is None else f"{frequencies[index]:.8f}"
            print(f"  {name:12s} frequency={frequency} weight={class_weights[index]:.6f}")

    start_epoch = 0
    micro_step = 0
    optimizer_step = 0
    best_miou = -math.inf
    if resume is not None:
        state = load_experiment_checkpoint(
            resume, model, optimizer, scheduler, scaler, current_config=config
        )
        start_epoch = int(state["epoch"])
        micro_step = int(state["micro_step"])
        optimizer_step = int(state["optimizer_step"])
        best_miou = float(state["best_miou"])
        if "rng_state" in state:
            restore_rng_state(state["rng_state"], generator)
        print(f"Resumed: epoch={start_epoch}, optimizer step={optimizer_step}")

    # resume 参数核对通过后再写，避免错误配置覆盖原实验记录。
    write_json(output_dir / "config.json", config)

    metrics_path = output_dir / "metrics.csv"
    _write_metrics_header(metrics_path, resume is not None)
    selected = select_fixed_validation_samples(
        val_dataset.samples, int(config["validation"].get("visualization_samples", 6))
    )
    raw_model = model
    train_model = model
    if runtime.get("compile", False):
        train_model = torch.compile(model)

    aux_weights = tuple(config["loss"].get("aux_weights", DEFAULT_AUX_WEIGHTS))
    grad_clip = float(config["training"].get("gradient_clip", 1.0))
    val_interval = int(config["validation"].get("interval_epochs", 2))
    visual_interval = int(config["validation"].get("visualization_interval_epochs", 5))
    overflow_count = 0
    epoch = start_epoch

    while optimizer_step < max_updates:
        epoch += 1
        train_model.train()
        epoch_loss = 0.0
        used_micro_batches = 0
        iterator = iter(train_loader)
        started = time.perf_counter()
        for _ in range(updates_per_epoch):
            if optimizer_step >= max_updates:
                break
            optimizer.zero_grad(set_to_none=True)
            update_loss = 0.0
            for _ in range(accumulation):
                images, masks = next(iterator)
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                if runtime.get("channels_last", False):
                    images = images.contiguous(memory_format=torch.channels_last)
                with autocast_context(device, amp_enabled):
                    outputs = train_model(images, return_aux=True)
                    loss = deep_supervision_loss(outputs, masks, criterion, aux_weights)
                scaler.scale(loss / accumulation).backward()
                update_loss += float(loss.detach())
                micro_step += 1
                used_micro_batches += 1

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            overflow = scaler.is_enabled() and scaler.get_scale() < scale_before
            if overflow:
                overflow_count += 1
                if overflow_count % 5 == 0:
                    print(f"警告：累计发生 {overflow_count} 次 AMP overflow。")
            else:
                optimizer_step += 1
                scheduler.step(optimizer_step)
            epoch_loss += update_loss / accumulation
            if optimizer_step % int(config["training"].get("log_interval", 50)) == 0:
                print(
                    f"Epoch {epoch} | micro {micro_step} | optimizer "
                    f"{optimizer_step}/{max_updates} | loss {update_loss / accumulation:.4f} | "
                    f"lr {optimizer.param_groups[0]['lr']:.6g}"
                )

        train_loss = epoch_loss / max(used_micro_batches / accumulation, 1)
        should_validate = epoch % val_interval == 0 or optimizer_step >= max_updates
        print(
            f"Epoch {epoch} complete | optimizer {optimizer_step}/{max_updates} | "
            f"train loss {train_loss:.4f} | {time.perf_counter() - started:.1f}s"
        )
        if not should_validate:
            _save_epoch_checkpoint(
                output_dir,
                raw_model,
                optimizer,
                scheduler,
                scaler,
                config,
                generator,
                epoch,
                micro_step,
                optimizer_step,
                best_miou,
            )
            continue

        val_loss, scores, matrix = evaluate_full_validation(
            raw_model, val_loader, criterion, device, config
        )
        miou = float(scores["mean_iou"])
        mean_dice = float(scores["mean_dice"])
        per_iou = scores["per_class_iou"].cpu().tolist()
        per_dice = scores["per_class_dice"].cpu().tolist()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_main_loss": val_loss,
            "mean_dice": mean_dice,
            "miou": miou,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        row.update({f"{name}_iou": value for name, value in zip(CLASS_NAMES, per_iou)})
        row.update({f"{name}_dice": value for name, value in zip(CLASS_NAMES, per_dice)})
        append_metrics_row(metrics_path, row)
        print(f"Full Val | loss={val_loss:.4f} | Dice={mean_dice:.4f} | mIoU={miou:.4f}")

        improved = math.isfinite(miou) and miou > best_miou
        if improved:
            best_miou = miou
        _save_epoch_checkpoint(
            output_dir,
            raw_model,
            optimizer,
            scheduler,
            scaler,
            config,
            generator,
            epoch,
            micro_step,
            optimizer_step,
            best_miou,
            save_best=improved,
        )
        save_confusion_matrix(matrix, output_dir)
        save_training_curves(metrics_path, output_dir / "curves")

        if epoch % visual_interval == 0:
            _save_fixed_predictions(
                raw_model, val_dataset, selected, config, device,
                output_dir / "predictions" / f"epoch_{epoch:04d}", epoch,
            )
        if improved:
            _save_fixed_predictions(
                raw_model, val_dataset, selected, config, device,
                output_dir / "predictions" / "best", epoch,
            )
        with metrics_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        write_json(
            output_dir / "summary.json",
            build_summary(
                rows,
                output_dir / "best_model.pth",
                "running",
                baseline_reference_miou=config["resolved"]["baseline_reference_miou"],
            ),
        )

    with metrics_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if rows:
        write_json(
            output_dir / "summary.json",
            build_summary(
                rows,
                output_dir / "best_model.pth",
                "complete",
                baseline_reference_miou=config["resolved"]["baseline_reference_miou"],
            ),
        )
