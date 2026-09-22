"""手动检查五个模型输出的 shape；不会被训练入口自动执行。"""

import argparse

import torch

from models import MODEL_VERSION, UNet3Plus
from utils.constants import NUM_CLASSES
from utils.device import print_device_info, resolve_device


def parse_args() -> argparse.Namespace:
    """读取结构检查参数；仅由 main 调用。"""
    parser = argparse.ArgumentParser(description="检查 U-Net 3+ 深监督输出形状")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto、cpu、cuda 或 cuda:N，例如 cuda:0",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("H", "W"),
        help="输入高和宽，均须至少 32 且为 16 的倍数",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于 0。")
    if any(size < 32 or size % 16 for size in args.image_size):
        parser.error("--image-size 的 H 和 W 须至少为 32 且为 16 的倍数。")
    return args


def main() -> None:
    """构造随机输入，打印并检查主输出与四个辅助输出。"""
    args = parse_args()
    device = resolve_device(args.device)
    print_device_info(device, amp_enabled=False)
    print("Model version:", MODEL_VERSION)

    # 与训练使用同一模型；结构检查不启用混合精度。
    model = UNet3Plus(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.eval()
    height, width = args.image_size
    images = torch.randn(args.batch_size, 3, height, width, device=device)

    with torch.no_grad():
        outputs = model(images, return_aux=True)

    expected_shape = (args.batch_size, NUM_CLASSES, height, width)
    print("main output shape:", tuple(outputs["main"].shape))
    assert tuple(outputs["main"].shape) == expected_shape
    assert len(outputs["aux"]) == 4
    for index, aux in enumerate(outputs["aux"], start=2):
        print(f"d{index} aux output shape:", tuple(aux.shape))
        assert tuple(aux.shape) == expected_shape


if __name__ == "__main__":
    main()
