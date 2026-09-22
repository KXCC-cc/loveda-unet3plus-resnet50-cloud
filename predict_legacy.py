"""单张图像推理：加载 checkpoint，输出 0~6 的类别索引 mask。"""

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.unet3plus import MODEL_VERSION, UNet3Plus
from utils.constants import (
    CLASS_NAMES,
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    encode_loveda_mask,
)
from utils.device import print_device_info, resolve_device
from utils.inference import sliding_window_logits
from utils.transforms import preprocess_image, validate_image_size


PROJECT_ROOT = Path(__file__).resolve().parent


def predict_image(
    image_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    restore_original_size: bool = True,
    amp_enabled: Optional[bool] = None,
) -> np.ndarray:
    """返回 uint8 的 [H, W] mask，像素值为训练类别索引 0~6。

    crop checkpoint：保持原图分辨率，滑窗融合完整图像 logits。
    resize checkpoint：先按训练尺寸缩放 RGB，再预测并恢复 logits 尺寸。
    默认返回原图尺寸；restore_original_size=False 仅适用于 resize 对照策略。
    此函数只返回数组，不写文件。默认沿用 checkpoint 的 AMP 设置，CPU 关闭。
    """
    image_path = Path(image_path)
    checkpoint_path = Path(checkpoint_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"输入图像不存在：{image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"模型 checkpoint 不存在：{checkpoint_path}")

    selected_device = resolve_device("auto" if device is None else str(device))
    # 显式使用 weights_only，checkpoint 的 config 只包含普通 Python 数据类型。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint 必须包含 model_state_dict，请使用 train.py 保存的文件。")
    saved_version = checkpoint.get("model_version")
    if saved_version != MODEL_VERSION:
        raise ValueError(
            f"checkpoint 模型版本不兼容：保存版本={saved_version!r}，"
            f"当前版本={MODEL_VERSION!r}。旧版四输出头的权重不能直接用于当前模型；"
            "请使用当前版本训练的 checkpoint，或使用与旧权重匹配的旧版代码。"
        )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint 的 config 必须是字典。")
    expected_metadata = {
        "class_names": list(CLASS_NAMES),
        "ignore_index": IGNORE_INDEX,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }
    for key, expected in expected_metadata.items():
        if config.get(key) != expected:
            raise ValueError(
                f"checkpoint 的 {key} 与当前数据定义不一致，"
                "请使用训练时对应的类别顺序与归一化配置。"
            )
    requested_amp = (
        bool(config.get("amp_enabled", False))
        if amp_enabled is None else amp_enabled
    )
    use_amp = selected_device.type == "cuda" and requested_amp
    print_device_info(selected_device, use_amp)
    if "image_size" not in config:
        raise ValueError("checkpoint 的 config 缺少 image_size，无法确定训练预处理。")
    image_size = validate_image_size(tuple(config["image_size"]))
    stride = config.get("eval_stride")
    if stride is None:
        stride = (max(image_size[0] // 2, 1), max(image_size[1] // 2, 1))
    stride = validate_image_size(tuple(stride))
    data_strategy = config.get("data_strategy")
    if data_strategy not in ("crop", "resize"):
        raise ValueError("checkpoint 的 config.data_strategy 必须为 crop 或 resize。")
    if data_strategy == "crop" and not restore_original_size:
        raise ValueError(
            "crop 策略必须返回完整原图尺寸；--keep-training-size / "
            "restore_original_size=False 仅适用于 resize 对照策略。"
        )
    num_classes = config.get("num_classes")
    if num_classes != NUM_CLASSES:
        raise ValueError(f"LoveDA 推理要求 num_classes=7，当前为 {num_classes}。")

    model = UNet3Plus(num_classes=num_classes)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(
            "checkpoint 权重结构与当前 U-Net 3+ 不兼容，不能忽略缺失参数加载。"
        ) from error
    del checkpoint  # 推理不需要 optimizer 等训练状态，及时释放这部分内存。
    model = model.to(selected_device)
    model.eval()

    with Image.open(image_path) as image:
        original_width, original_height = image.size
        # crop 策略不缩小原图；resize 策略与训练的 bilinear 预处理保持一致。
        image_tensor = preprocess_image(
            image, image_size if data_strategy == "resize" else None
        )

    logits = sliding_window_logits(
        model,
        image_tensor,
        tile_size=image_size,
        stride=stride,
        device=selected_device,
        amp_enabled=use_amp,
    )
    target_size = (
        (original_height, original_width) if restore_original_size else image_size
    )
    if logits.shape[-2:] != target_size:
        # 始终先调整 logits 再 argmax；不对离散类别编号做 bilinear 插值。
        logits = F.interpolate(
            logits, size=target_size, mode="bilinear", align_corners=False
        )
    prediction = logits.argmax(dim=1)[0]
    return prediction.numpy().astype(np.uint8)


def _inside_directory(path: Path, directory: Path) -> bool:
    """resolve 后判断输出是否落在受保护目录中。"""
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="LoveDA U-Net 3+ 单张图像推理")
    parser.add_argument("image", type=Path, help="输入 RGB 图像路径")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "best_model.pth",
        help="train.py 保存的 checkpoint 路径",
    )
    parser.add_argument(
        "--device", default="auto", help="auto、cpu、cuda、cuda:0；默认 auto"
    )
    parser.add_argument("--no-amp", action="store_true", help="关闭 CUDA 混合精度推理")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 PNG 路径；默认 predictions/<图像名>_prediction.png",
    )
    parser.add_argument(
        "--keep-training-size",
        action="store_true",
        help="仅供 resize 对照策略保留训练尺寸；crop 策略禁止该选项，默认输出原图尺寸",
    )
    parser.add_argument(
        "--loveda-labels",
        action="store_true",
        help="输出原始 LoveDA 类别值 1~7；默认输出训练类别索引 0~6",
    )
    args = parser.parse_args()

    image_path = args.image.resolve()
    output_path = (
        args.output
        if args.output is not None
        else PROJECT_ROOT / "predictions" / f"{image_path.stem}_prediction.png"
    ).resolve()
    if output_path.suffix.lower() != ".png":
        parser.error("输出必须使用 .png 扩展名，以无损保存类别索引。")
    if not _inside_directory(output_path, PROJECT_ROOT):
        parser.error(f"输出必须位于项目目录 {PROJECT_ROOT} 中。")
    if output_path == image_path:
        parser.error("输出路径不能覆盖输入图像。")
    if _inside_directory(output_path, (PROJECT_ROOT / "dataset").resolve()):
        parser.error("不能向 dataset 目录写入推理结果。")
    if output_path.exists():
        parser.error(f"输出文件已存在，请指定新的 --output 路径：{output_path}")

    mask = predict_image(
        image_path=image_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        restore_original_size=not args.keep_training_size,
        amp_enabled=False if args.no_amp else None,
    )
    if args.loveda_labels:
        # Test 无真实 mask，不能推断 no-data；每个像素均输出一个实际类别 1~7。
        mask = encode_loveda_mask(mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path)
    label_range = "1~7（LoveDA 原始类别值）" if args.loveda_labels else "0~6（训练类别索引）"
    print(f"已保存预测：{output_path}")
    print(f"mask shape：{mask.shape}；类别范围：{label_range}")


if __name__ == "__main__":
    main()
