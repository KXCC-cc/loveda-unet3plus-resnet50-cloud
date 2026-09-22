"""确定性全图滑窗：按小批窗口运行模型，平均 logits 后返回完整图像。"""

from typing import List, Tuple

import torch
import torch.nn.functional as F

from utils.constants import NUM_CLASSES
from utils.device import autocast_context
from utils.transforms import validate_image_size


def _window_starts(
    length: int, tile_length: int, stride: int,
) -> List[int]:
    """最后一个窗口锚定图像末端，保证非整除尺寸也完整覆盖。"""
    starts = list(range(0, length - tile_length + 1, stride))
    last_start = length - tile_length
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


@torch.inference_mode()
def sliding_window_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    tile_size: Tuple[int, int],
    stride: Tuple[int, int],
    device: torch.device,
    amp_enabled: bool = False,
    tile_batch_size: int = 1,
    accumulate_on_device: bool = False,
) -> torch.Tensor:
    """输入已归一化的 CPU float32 [3,H,W]，返回 CPU float32 [1,7,H,W]。

    调用方先 model.eval()。默认一个窗口对应一次前向，在 CPU 累加结果。
    tile_batch_size 可把多个窗口合并前向，增大时需要更多显存。
    accumulate_on_device=True 时在 device 累加，仅在最后传回完整 CPU logits；
    这会额外占用整图 logits 和覆盖次数的显存，但减少逐窗口的数据传输。
    重叠区域先累加 logits、再除以覆盖次数；本函数不会提前 argmax。
    Val 与 predict 共用此函数，确保边缘覆盖和融合方式一致。
    """
    if model.training:
        raise ValueError("滑窗验证/推理前必须先调用 model.eval()。")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image 必须是 [3, H, W] 的 RGB 张量。")
    if image.device.type != "cpu" or image.dtype != torch.float32:
        raise ValueError("image 必须是已归一化的 CPU float32 张量。")
    if not isinstance(tile_batch_size, int) or tile_batch_size < 1:
        raise ValueError("tile_batch_size 必须是正整数。")
    tile_height, tile_width = validate_image_size(tile_size)
    stride_height, stride_width = validate_image_size(stride)
    if min(tile_height, tile_width) < 32 or tile_height % 16 or tile_width % 16:
        raise ValueError("tile 的高、宽至少为 32，且必须是 16 的倍数，与模型输入要求一致。")
    if stride_height > tile_height or stride_width > tile_width:
        raise ValueError("stride 不能大于 tile_size，否则窗口之间会遗漏像素。")
    original_height, original_width = image.shape[-2:]
    if original_height <= 0 or original_width <= 0:
        raise ValueError("image 不能为空。")

    padded_height = max(original_height, tile_height)
    padded_width = max(original_width, tile_width)
    padded_image = image.unsqueeze(0)  # [1,3,H,W]，始终保留在 CPU。
    if padded_height != original_height or padded_width != original_width:
        padded_image = F.pad(
            padded_image,
            (0, padded_width - original_width, 0, padded_height - original_height),
            mode="replicate",
        )

    accumulation_device = device if accumulate_on_device else torch.device("cpu")
    logits_sum = torch.zeros(
        (1, NUM_CLASSES, padded_height, padded_width),
        dtype=torch.float32, device=accumulation_device,
    )
    coverage = torch.zeros(
        (1, 1, padded_height, padded_width),
        dtype=torch.float32, device=accumulation_device,
    )
    row_starts = _window_starts(padded_height, tile_height, stride_height)
    column_starts = _window_starts(padded_width, tile_width, stride_width)

    # 顺序仍然是从上到下、从左到右；同一图像的窗口合并为一个小 batch。
    windows = [(top, left) for top in row_starts for left in column_starts]
    for start in range(0, len(windows), tile_batch_size):
        batch_windows = windows[start:start + tile_batch_size]
        tiles = torch.cat(
            [
                padded_image[
                    :, :, top:top + tile_height, left:left + tile_width
                ]
                for top, left in batch_windows
            ],
            dim=0,
        )  # [N,3,tile_H,tile_W]，最后一批的 N 可以小于 tile_batch_size。
        tiles = tiles.to(device, non_blocking=True)
        with autocast_context(device, enabled=amp_enabled):
            # 验证/推理只计算 main 分类头，避免分配没有使用的 aux logits。
            outputs = model(tiles, return_aux=False)
            logits = outputs["main"]
        expected_shape = (
            len(batch_windows), NUM_CLASSES, tile_height, tile_width
        )
        if logits.shape != expected_shape:
            raise ValueError(
                f"模型 main 输出尺寸异常：{tuple(logits.shape)}，"
                f"期望 {expected_shape}。"
            )

        # AMP 前向可能输出 float16；融合始终用 float32，减小累加误差。
        batch_logits = logits.float().to(accumulation_device)
        for index, (top, left) in enumerate(batch_windows):
            bottom = top + tile_height
            right = left + tile_width
            logits_sum[:, :, top:bottom, left:right] += batch_logits[
                index:index + 1
            ]
            coverage[:, :, top:bottom, left:right] += 1.0
        del tiles, outputs, logits, batch_logits

    # 窗口步长校验与末端锚定保证 coverage>0；保留显式检查便于学习/排错。
    if torch.any(coverage == 0):
        raise RuntimeError("滑窗存在未覆盖像素，请检查 tile_size 与 stride。")
    logits_sum.div_(coverage)
    return logits_sum[:, :, :original_height, :original_width].contiguous().cpu()


@torch.inference_mode()
def multi_scale_sliding_window_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    scales: tuple[float, ...],
    tile_size: Tuple[int, int],
    stride: Tuple[int, int],
    device: torch.device,
    amp_enabled: bool,
    tile_batch_size: int,
    horizontal_flip: bool = False,
    accumulate_on_device: bool = False,
) -> torch.Tensor:
    """各尺度（及可选水平镜像）先恢复原尺寸，再平均 logits。

    最终 argmax 由调用方执行，严禁缩放离散类别编号。
    """
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("scales 必须包含正数。")
    original_size = image.shape[-2:]
    logits_sum = None
    count = 0
    for scale in scales:
        scaled_size = (
            max(1, round(original_size[0] * scale)),
            max(1, round(original_size[1] * scale)),
        )
        scaled = F.interpolate(
            image.unsqueeze(0), size=scaled_size, mode="bilinear", align_corners=False
        ).squeeze(0).contiguous()
        variants = ((scaled, False),)
        if horizontal_flip:
            variants += ((torch.flip(scaled, dims=(2,)).contiguous(), True),)
        for variant, flipped in variants:
            logits = sliding_window_logits(
                model=model,
                image=variant,
                tile_size=tile_size,
                stride=stride,
                device=device,
                amp_enabled=amp_enabled,
                tile_batch_size=tile_batch_size,
                accumulate_on_device=accumulate_on_device,
            )
            if flipped:
                logits = torch.flip(logits, dims=(3,))
            logits = F.interpolate(
                logits, size=original_size, mode="bilinear", align_corners=False
            )
            logits_sum = logits if logits_sum is None else logits_sum + logits
            count += 1
    return logits_sum / count
