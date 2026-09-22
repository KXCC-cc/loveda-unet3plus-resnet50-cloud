"""训练、推理和结构检查共用的设备与 AMP 设置。"""

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Union

import torch


def resolve_device(
    requested: Union[str, torch.device] = "auto",
) -> torch.device:
    """规范化为 cpu 或 cuda:N；显式请求不可用 CUDA 时直接报错。"""
    name = str(requested)
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("本项目支持 cpu、cuda、cuda:N 或 auto。")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "请求了 CUDA，但当前 PyTorch 没有可用 CUDA。"
                "请检查 NVIDIA 驱动及 PyTorch CUDA 版本；"
                "若要使用 CPU，请明确传入 --device cpu。"
            )
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"cuda:{index} 超出当前可用 GPU 数量。")
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def autocast_context(
    device: torch.device, enabled: bool,
) -> AbstractContextManager:
    """CUDA 使用 FP16 autocast，CPU 始终保持普通 FP32。"""
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def create_grad_scaler(enabled: bool) -> Any:
    """优先使用新版 API，保留 PyTorch 2.0/2.1 的兼容分支。"""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def print_device_info(device: torch.device, amp_enabled: bool) -> None:
    """只在用户实际启动入口后报告硬件信息。"""
    available = torch.cuda.is_available()
    print(f"PyTorch: {torch.__version__} | CUDA available: {available}")
    print(f"Device: {device} | CUDA device count: {torch.cuda.device_count()}")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "N/A"
    print(f"GPU name: {gpu_name} | AMP enabled: {amp_enabled}")
