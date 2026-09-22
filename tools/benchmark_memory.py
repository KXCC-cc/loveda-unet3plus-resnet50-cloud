"""极小的 512 forward/backward 显存测试；不会启动正式训练。"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

import torch

from models import UNet3PlusResNet
from utils.device import autocast_context, create_grad_scaler
from utils.losses import CombinedLoss, deep_supervision_loss


def benchmark(backbone: str, batch_size: int, image_size: int, pretrained: bool) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = optimizer = scaler = images = masks = outputs = loss = criterion = None
    try:
        model = UNet3PlusResNet(
            backbone=backbone,
            pretrained=pretrained,
            weights="IMAGENET1K_V1",
        ).cuda().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scaler = create_grad_scaler(True)
        images = torch.randn(batch_size, 3, image_size, image_size, device="cuda")
        masks = torch.randint(0, 7, (batch_size, image_size, image_size), device="cuda")
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(torch.device("cuda"), True):
            outputs = model(images, return_aux=True)
            # 与 4060 performance profile 一致：主头和辅助头均使用 CE + Dice。
            criterion = CombinedLoss(ce_weight=1.0, dice_weight=1.0).cuda()
            loss = deep_supervision_loss(outputs, masks, criterion, (0.5, 0.25, 0.125, 0.0625))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        return {
            "backbone": backbone,
            "batch_size": batch_size,
            "image_size": image_size,
            "status": "ok",
            "weights": model.encoder.weights_name,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
    except (torch.OutOfMemoryError, torch.AcceleratorError) as error:
        return {
            "backbone": backbone,
            "batch_size": batch_size,
            "image_size": image_size,
            "status": "oom",
            "error": str(error).splitlines()[0],
        }
    finally:
        del model, optimizer, scaler, images, masks, outputs, loss, criterion
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except torch.AcceleratorError:
            # OOM 后 CUDA context 可能保持 error 状态；单场景子进程即将退出。
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("memory_benchmark.json"))
    parser.add_argument("--backbone", choices=("resnet34", "resnet50"), default=None)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("显存测试要求 CUDA。")
    if args.backbone is not None:
        if args.batch_size is None:
            parser.error("--backbone 与 --batch-size 必须同时提供。")
        result = benchmark(args.backbone, args.batch_size, args.image_size, not args.no_pretrained)
        print("BENCHMARK_RESULT=" + json.dumps(result))
        return

    # 每个场景使用独立进程；某一组 OOM 不会污染后续 CUDA context。
    results = []
    for backbone in ("resnet34", "resnet50"):
        for batch in (1, 2):
            command = [
                sys.executable,
                "-m",
                "tools.benchmark_memory",
                "--backbone",
                backbone,
                "--batch-size",
                str(batch),
                "--image-size",
                str(args.image_size),
            ]
            if args.no_pretrained:
                command.append("--no-pretrained")
            completed = subprocess.run(command, capture_output=True, text=True)
            marker_lines = [
                line for line in completed.stdout.splitlines()
                if line.startswith("BENCHMARK_RESULT=")
            ]
            if marker_lines:
                results.append(json.loads(marker_lines[-1].split("=", 1)[1]))
            else:
                results.append({
                    "backbone": backbone,
                    "batch_size": batch,
                    "image_size": args.image_size,
                    "status": "failed",
                    "error": (completed.stderr or completed.stdout).strip()[-1000:],
                })
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
