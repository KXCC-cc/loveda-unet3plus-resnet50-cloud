"""从真实 LoveDA 像素统计生成温和的类别权重。"""

from __future__ import annotations

import json
import math
from pathlib import Path

from utils.constants import CLASS_NAMES, NUM_CLASSES


def load_inverse_sqrt_weights(path: Path) -> tuple[list[float], list[float]]:
    """weight=1/sqrt(freq)，再归一化到七类均值为 1。"""
    stats = json.loads(path.read_text(encoding="utf-8"))
    if stats.get("class_names") != list(CLASS_NAMES):
        raise ValueError("class_stats.json 的类别顺序与项目常量不一致。")
    frequencies = [float(value) for value in stats["total"]["class_frequencies"]]
    if len(frequencies) != NUM_CLASSES or any(
        not math.isfinite(value) or value <= 0 for value in frequencies
    ):
        raise ValueError("class_stats.json 必须包含七个有限正频率。")
    raw = [1.0 / math.sqrt(value) for value in frequencies]
    mean = sum(raw) / len(raw)
    return frequencies, [value / mean for value in raw]
