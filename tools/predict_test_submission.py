"""批量预测 LoveDA Test，并生成可上传的单尺度 submission ZIP。

输出 PNG 为单通道 uint8，像素值严格限定为 0~6。ZIP 根目录直接
包含所有 PNG，不保存彩色图、叠加图、模型文件或额外目录。
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models import build_model
from utils.constants import CLASS_NAMES, NUM_CLASSES
from utils.device import print_device_info, resolve_device
from utils.inference import multi_scale_sliding_window_logits
from utils.transforms import preprocess_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "resnet34_pretrained_512_randomcrop"
    / "best_model.pth"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "submissions" / "loveda_test_single_scale"
DEFAULT_ZIP_PATH = PROJECT_ROOT / "submissions" / "loveda_test_single_scale.zip"
DOMAINS = ("Rural", "Urban")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 LoveDA Test 单尺度官方评测提交包"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-batch-size", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="重新预测已经存在且有效的 PNG；默认支持断点续跑",
    )
    return parser.parse_args()


def collect_test_images(data_root: Path) -> list[tuple[str, Path]]:
    """按 Rural、Urban 和文件名排序，并确保平铺后不会重名。"""
    samples: list[tuple[str, Path]] = []
    for domain in DOMAINS:
        image_dir = data_root / "Test" / domain / "images_png"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"找不到 Test 图片目录：{image_dir}")
        domain_images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )
        if not domain_images:
            raise ValueError(f"{image_dir} 中没有 PNG 图片。")
        samples.extend((domain, path) for path in domain_images)

    names = [path.name for _, path in samples]
    duplicate_names = sorted(
        name for name in set(names) if names.count(name) > 1
    )
    if duplicate_names:
        preview = ", ".join(duplicate_names[:5])
        raise ValueError(
            "Rural 与 Urban 存在重名图片，不能平铺提交："
            f"{preview}"
        )
    return samples


def load_runtime(
    checkpoint_path: Path,
    device_name: str,
    amp_requested: bool,
    patch_batch_size: int | None,
) -> tuple[
    torch.nn.Module,
    dict[str, Any],
    torch.device,
    bool,
    int,
]:
    """加载 checkpoint；关闭预训练下载，只恢复其中保存的完整权重。"""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint.get("config")
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError("checkpoint 缺少 config.model，无法构建模型。")

    model_config = copy.deepcopy(config["model"])
    model_config["pretrained"] = False
    model_config["weights"] = None
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    device = resolve_device(device_name)
    amp_enabled = device.type == "cuda" and amp_requested
    model.to(device).eval()

    validation = config.get("validation", {})
    resolved_patch_batch = (
        patch_batch_size
        if patch_batch_size is not None
        else int(validation.get("patch_batch_size", 1))
    )
    if resolved_patch_batch < 1:
        raise ValueError("patch_batch_size 必须大于等于 1。")
    return model, config, device, amp_enabled, resolved_patch_batch


def read_prediction(path: Path) -> np.ndarray:
    """读取提交 mask，同时拒绝 RGB 或越界标签。"""
    with Image.open(path) as image:
        if image.mode not in ("L", "P"):
            raise ValueError(f"{path.name} 不是单通道 PNG，mode={image.mode}。")
        prediction = np.asarray(image, dtype=np.uint8)
    if prediction.ndim != 2:
        raise ValueError(f"{path.name} 必须是二维单通道 mask。")
    if prediction.size == 0:
        raise ValueError(f"{path.name} 是空图片。")
    minimum = int(prediction.min())
    maximum = int(prediction.max())
    if minimum < 0 or maximum >= NUM_CLASSES:
        raise ValueError(
            f"{path.name} 标签范围为 {minimum}~{maximum}，要求 0~6。"
        )
    return prediction


def existing_prediction_is_valid(
    prediction_path: Path,
    expected_size: tuple[int, int],
) -> bool:
    """断点续跑时只跳过尺寸和标签均正确的已有结果。"""
    if not prediction_path.is_file():
        return False
    try:
        prediction = read_prediction(prediction_path)
    except (OSError, ValueError):
        return False
    expected_width, expected_height = expected_size
    return prediction.shape == (expected_height, expected_width)


def predict_all(
    samples: list[tuple[str, Path]],
    output_dir: Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
    patch_batch_size: int,
    overwrite: bool,
) -> int:
    """单尺度滑窗预测；模型内部的 0~6 索引直接保存为提交标签。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = config.get("validation", {})
    tile_size = tuple(validation.get("tile_size", (512, 512)))
    stride = tuple(validation.get("stride", tile_size))
    accumulate_on_device = (
        device.type == "cuda"
        and bool(validation.get("accumulate_on_device", False))
    )

    skipped = 0
    for _domain, image_path in tqdm(
        samples,
        desc="Test single-scale",
        unit="image",
    ):
        output_path = output_dir / image_path.name
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")
        if (
            not overwrite
            and existing_prediction_is_valid(output_path, source.size)
        ):
            skipped += 1
            continue

        image_tensor = preprocess_image(source)
        logits = multi_scale_sliding_window_logits(
            model=model,
            image=image_tensor,
            scales=(1.0,),
            tile_size=tile_size,
            stride=stride,
            device=device,
            amp_enabled=amp_enabled,
            tile_batch_size=patch_batch_size,
            horizontal_flip=False,
            accumulate_on_device=accumulate_on_device,
        )
        prediction = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
        if prediction.shape != (source.height, source.width):
            raise RuntimeError(
                f"{image_path.name} 输出尺寸 {prediction.shape} 与原图"
                f" {(source.height, source.width)} 不一致。"
            )
        if int(prediction.max()) >= NUM_CLASSES:
            raise RuntimeError(f"{image_path.name} 出现 0~6 之外的标签。")

        temporary_path = output_path.with_suffix(".tmp.png")
        Image.fromarray(prediction, mode="L").save(temporary_path)
        temporary_path.replace(output_path)

    return skipped


def validate_outputs(
    samples: list[tuple[str, Path]],
    output_dir: Path,
) -> list[Path]:
    """逐张核对名称、尺寸、通道和标签范围。"""
    expected_names = {path.name for _, path in samples}
    actual_paths = sorted(output_dir.glob("*.png"))
    actual_names = {path.name for path in actual_paths}
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing:
        raise RuntimeError(f"缺少 {len(missing)} 个预测文件，例如 {missing[:5]}。")
    if unexpected:
        raise RuntimeError(
            f"存在 {len(unexpected)} 个多余 PNG，例如 {unexpected[:5]}。"
        )

    source_by_name = {path.name: path for _, path in samples}
    for prediction_path in tqdm(
        actual_paths,
        desc="Validate masks",
        unit="image",
    ):
        source_path = source_by_name[prediction_path.name]
        with Image.open(source_path) as source:
            expected_size = source.size
        prediction = read_prediction(prediction_path)
        expected_width, expected_height = expected_size
        if prediction.shape != (expected_height, expected_width):
            raise RuntimeError(
                f"{prediction_path.name} 尺寸 {prediction.shape} 错误，"
                f"期望 {(expected_height, expected_width)}。"
            )
    return actual_paths


def create_flat_zip(prediction_paths: list[Path], zip_path: Path) -> None:
    """ZIP 根目录直接放 PNG，避免多套一层目录。"""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_zip = zip_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(
        temporary_zip,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in tqdm(prediction_paths, desc="Create ZIP", unit="file"):
            archive.write(path, arcname=path.name)
    temporary_zip.replace(zip_path)


def write_report(
    report_path: Path,
    checkpoint_path: Path,
    samples: list[tuple[str, Path]],
    output_dir: Path,
    zip_path: Path,
    elapsed_seconds: float,
    skipped: int,
) -> None:
    counts = {
        domain: sum(1 for sample_domain, _ in samples if sample_domain == domain)
        for domain in DOMAINS
    }
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "inference": "single_scale",
        "scales": [1.0],
        "horizontal_flip": False,
        "test_counts": counts,
        "total_png": len(samples),
        "label_encoding": {
            str(index): name for index, name in enumerate(CLASS_NAMES)
        },
        "mask_format": "single-channel uint8 PNG",
        "output_dir": str(output_dir.resolve()),
        "zip_path": str(zip_path.resolve()),
        "zip_flat_root": True,
        "skipped_existing": skipped,
        "elapsed_seconds": elapsed_seconds,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    zip_path = args.zip_path.resolve()
    if zip_path.suffix.lower() != ".zip":
        raise ValueError("--zip-path 必须以 .zip 结尾。")

    samples = collect_test_images(data_root)
    domain_counts = {
        domain: sum(1 for sample_domain, _ in samples if sample_domain == domain)
        for domain in DOMAINS
    }
    print(
        f"Test images: {len(samples)} | "
        f"Rural={domain_counts['Rural']} | Urban={domain_counts['Urban']}"
    )

    model, config, device, amp_enabled, patch_batch_size = load_runtime(
        checkpoint_path=checkpoint_path,
        device_name=args.device,
        amp_requested=not args.no_amp,
        patch_batch_size=args.patch_batch_size,
    )
    print_device_info(device, amp_enabled)
    print("Inference: single scale 1.0 | horizontal flip: False")
    print(f"Output masks: {output_dir}")
    print(f"Submission ZIP: {zip_path}")

    started = time.perf_counter()
    skipped = predict_all(
        samples=samples,
        output_dir=output_dir,
        model=model,
        config=config,
        device=device,
        amp_enabled=amp_enabled,
        patch_batch_size=patch_batch_size,
        overwrite=args.overwrite,
    )
    prediction_paths = validate_outputs(samples, output_dir)
    create_flat_zip(prediction_paths, zip_path)
    elapsed = time.perf_counter() - started

    report_path = zip_path.with_name(f"{zip_path.stem}_report.json")
    write_report(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        samples=samples,
        output_dir=output_dir,
        zip_path=zip_path,
        elapsed_seconds=elapsed,
        skipped=skipped,
    )
    print("Submission ready.")
    print(f"PNG count: {len(prediction_paths)}")
    print(f"ZIP: {zip_path}")
    print(f"Report: {report_path}")
    print(f"Elapsed: {elapsed / 60.0:.1f} minutes")


if __name__ == "__main__":
    main()
