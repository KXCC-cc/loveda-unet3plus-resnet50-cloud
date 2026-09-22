"""在完整 LoveDA Val 上执行单尺度或多尺度滑窗评估。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from engine.evaluator import evaluate_full_validation
from models import build_model
from utils.constants import CLASS_NAMES
from utils.dataset import LoveDADataset
from utils.device import print_device_info, resolve_device
from utils.experiment import save_confusion_matrix, write_json
from utils.losses import CombinedLoss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--patch-batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    config["validation"]["scales"] = args.scales
    config["validation"]["horizontal_flip_tta"] = args.horizontal_flip
    if args.patch_batch_size is not None:
        config["validation"]["patch_batch_size"] = args.patch_batch_size
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    print_device_info(device, bool(config["runtime"].get("amp", True)))

    dataset = LoveDADataset(config["data"]["root"], split="Val", spatial_mode="full", augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    loss_config = config["loss"]
    class_weights = config.get("resolved", {}).get("class_weights")
    criterion = CombinedLoss(
        ce_weight=float(loss_config.get("ce_weight", 1.0)),
        dice_weight=float(loss_config.get("dice_weight", 0.0)),
        focal_weight=float(loss_config.get("focal_weight", 0.0)),
        focal_gamma=float(loss_config.get("focal_gamma", 2.0)),
        class_weights=class_weights,
    ).to(device)
    loss, scores, matrix = evaluate_full_validation(model, loader, criterion, device, config)
    label = "multiscale" if len(args.scales) > 1 or args.horizontal_flip else "single_scale"
    output_dir = args.output_dir or args.checkpoint.resolve().parent / f"evaluation_{label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "scales": args.scales,
        "horizontal_flip": args.horizontal_flip,
        "val_loss": loss,
        "miou": scores["mean_iou"],
        "mean_dice": scores["mean_dice"],
        "per_class_iou": dict(zip(CLASS_NAMES, scores["per_class_iou"].tolist())),
        "per_class_dice": dict(zip(CLASS_NAMES, scores["per_class_dice"].tolist())),
    }
    write_json(output_dir / "metrics.json", result)
    save_confusion_matrix(matrix, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
