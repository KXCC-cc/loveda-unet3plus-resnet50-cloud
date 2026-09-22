"""手动统计 LoveDA 原始 mask 的类别频率；导入本文件不会开始统计。

默认只统计 Train，不缩放、不裁剪、不增强，不修改 image / mask 原始数据。
此脚本只提供真实像素计数与频率，不自动猜测训练权重。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.constants import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    RAW_IGNORE_INDEX,
    map_raw_mask,
)
from utils.dataset import LoveDADataset


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手动统计 LoveDA 原始类别像素频率")
    parser.add_argument("--data-root", type=Path, default=PROJECT_DIR / "dataset")
    parser.add_argument("--split", choices=["Train", "Val"], default="Train")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_DIR / "class_stats.json"
    )
    args = parser.parse_args()

    output_path = args.output.resolve()
    if PROJECT_DIR not in output_path.parents:
        parser.error("统计结果必须保存在本项目目录内。")
    for dataset_root in (
        args.data_root.resolve(),
        (PROJECT_DIR / "dataset").resolve(),
    ):
        if output_path == dataset_root or dataset_root in output_path.parents:
            parser.error("统计结果不能写入 dataset 目录或数据集所在目录。")
    if output_path.suffix.lower() != ".json":
        parser.error("output 必须使用 .json 扩展名。")
    args.output = output_path
    return args


def empty_counts() -> dict[str, Any]:
    return {
        "num_masks": 0,
        "total_pixels": 0,
        "ignored_pixels": 0,
        "class_counts": np.zeros(NUM_CLASSES, dtype=np.int64),
    }


def summarize_counts(counts: dict[str, Any]) -> dict[str, Any]:
    """转换成 JSON 可保存的基础类型，频率的分母只包括有效类别像素。"""
    class_counts = counts["class_counts"].tolist()
    valid_pixels = sum(class_counts)
    frequencies = [
        count / valid_pixels if valid_pixels > 0 else None for count in class_counts
    ]
    return {
        "num_masks": counts["num_masks"],
        "total_pixels": counts["total_pixels"],
        "ignored_pixels": counts["ignored_pixels"],
        "valid_pixels": valid_pixels,
        "class_counts": class_counts,
        "class_frequencies": frequencies,
    }


def main() -> None:
    args = parse_args()
    # 只重用 Dataset 的文件发现和严格 image-mask 配对；不调用 __getitem__。
    # samples 每张原图只有一项，不受训练时 samples_per_image 影响。
    dataset = LoveDADataset(
        args.data_root,
        split=args.split,
        spatial_mode="full",
        augment=False,
    )
    domains = {domain: empty_counts() for domain in ("Rural", "Urban")}
    total = empty_counts()

    for image_path, mask_path in tqdm(dataset.samples, desc=f"Stats {args.split}"):
        if mask_path is None:
            raise ValueError("统计需要真实 mask，不能用于无标注的 Test。")
        domain = image_path.parent.parent.name
        if domain not in domains:
            raise ValueError(f"无法识别样本所属域：{image_path}")
        with Image.open(mask_path) as mask_image:
            # 不转灰度、不做 resize；P 模式调色板 mask 保留原始类别索引。
            raw_mask = np.array(mask_image, copy=True)
        try:
            # map_raw_mask 内部调用 validate_raw_mask，校验非空二维整数 0~7。
            mapped_mask = map_raw_mask(raw_mask)
        except (TypeError, ValueError) as error:
            raise ValueError(f"无效的原始 mask：{mask_path}，{error}") from error

        valid = mapped_mask != IGNORE_INDEX
        class_counts = np.bincount(mapped_mask[valid], minlength=NUM_CLASSES)
        ignored_pixels = int(np.count_nonzero(raw_mask == RAW_IGNORE_INDEX))
        for counts in (domains[domain], total):
            counts["num_masks"] += 1
            counts["total_pixels"] += int(raw_mask.size)
            counts["ignored_pixels"] += ignored_pixels
            counts["class_counts"] += class_counts

    report = {
        "dataset_root": str(args.data_root.resolve()),
        "split": args.split,
        "resolution": "original; no resize, crop or augmentation",
        "raw_ignore_index": RAW_IGNORE_INDEX,
        "ignore_index": IGNORE_INDEX,
        "class_ids": list(range(NUM_CLASSES)),
        "raw_class_ids": list(range(1, NUM_CLASSES + 1)),
        "class_names": list(CLASS_NAMES),
        "frequency_definition": (
            "class_counts[i] / valid_pixels; null if no valid pixels"
        ),
        "domains": {
            name: summarize_counts(counts) for name, counts in domains.items()
        },
        "total": summarize_counts(total),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"统计结果已保存：{args.output}")
    summaries = [*report["domains"].items(), ("Total", report["total"])]
    for name, summary in summaries:
        print(
            f"{name}: masks={summary['num_masks']}, "
            f"valid_pixels={summary['valid_pixels']}, "
            f"ignored_pixels={summary['ignored_pixels']}"
        )
        for class_id, class_name in enumerate(CLASS_NAMES):
            frequency = summary["class_frequencies"][class_id]
            frequency_text = (
                f"{frequency:.6%}" if frequency is not None else "N/A"
            )
            print(
                f"  {class_id} {class_name}: "
                f"pixels={summary['class_counts'][class_id]}, "
                f"frequency={frequency_text}"
            )


if __name__ == "__main__":
    main()
