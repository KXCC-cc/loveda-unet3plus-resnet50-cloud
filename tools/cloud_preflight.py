"""正式云训练前检查 CUDA、显存和 LoveDA Train/Val 文件数量。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


EXPECTED = {"Train": 2522, "Val": 1669}


def _png_names(directory: Path) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在：{directory}")
    return {path.name for path in directory.glob("*.png") if path.is_file()}


def validate_dataset(root: Path) -> dict[str, int]:
    counts = {}
    for split, expected in EXPECTED.items():
        total = 0
        for region in ("Rural", "Urban"):
            base = root / split / region
            images = _png_names(base / "images_png")
            masks = _png_names(base / "masks_png")
            if images != masks:
                missing = sorted(images - masks)[:5]
                extra = sorted(masks - images)[:5]
                raise RuntimeError(
                    f"{split}/{region} image/mask 不匹配："
                    f"缺 mask={missing}，多余 mask={extra}"
                )
            total += len(images)
        if total != expected:
            raise RuntimeError(
                f"{split} images={total}，期望 {expected}；拒绝启动正式训练。"
            )
        counts[split] = total
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("未检测到 CUDA GPU，拒绝启动正式训练。")
    properties = torch.cuda.get_device_properties(0)
    print("GPU:", properties.name)
    print(f"Total VRAM: {properties.total_memory / 1024**3:.2f} GiB")
    counts = validate_dataset(args.dataset_root.resolve())
    print(f"Train images: {counts['Train']}")
    print(f"Val images: {counts['Val']}")
    print("Preflight status: OK")


if __name__ == "__main__":
    main()
