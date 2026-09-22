"""新旧 U-Net 3+ checkpoint 的单图滑窗推理入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models import build_model
from utils.constants import encode_loveda_mask
from utils.device import print_device_info, resolve_device
from utils.experiment import colorize_mask
from utils.inference import multi_scale_sliding_window_logits
from utils.transforms import preprocess_image


def main() -> None:
    parser = argparse.ArgumentParser(description="LoveDA 单张图片推理")
    parser.add_argument("image", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("prediction.png"))
    parser.add_argument("--color-output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--patch-batch-size", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError(
            "该 checkpoint 是旧命令行格式，请使用 predict_legacy.py；"
            "新入口要求 checkpoint.config.model。"
        )
    device = resolve_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    print_device_info(device, amp)
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    with Image.open(args.image) as source:
        image = preprocess_image(source.convert("RGB"))
    validation = config["validation"]
    logits = multi_scale_sliding_window_logits(
        model=model,
        image=image,
        scales=tuple(args.scales),
        tile_size=tuple(validation.get("tile_size", (512, 512))),
        stride=tuple(validation.get("stride", (512, 512))),
        device=device,
        amp_enabled=amp,
        tile_batch_size=args.patch_batch_size or int(validation.get("patch_batch_size", 1)),
        horizontal_flip=args.horizontal_flip,
        accumulate_on_device=bool(validation.get("accumulate_on_device", False)),
    )
    prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encode_loveda_mask(prediction)).save(args.output)
    if args.color_output is not None:
        args.color_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(colorize_mask(prediction)).save(args.color_output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
