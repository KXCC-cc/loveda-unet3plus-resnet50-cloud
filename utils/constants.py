"""LoveDA 标签规则与项目各模块共用的常量。"""

import numpy as np


CLASS_NAMES = (
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agricultural",
)
NUM_CLASSES = 7
RAW_IGNORE_INDEX = 0
IGNORE_INDEX = 255
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_AUX_WEIGHTS = (0.5, 0.25, 0.125, 0.0625)
RESNET34_REFERENCE_MIOU = 0.4669311579969285


def validate_raw_mask(raw_mask: np.ndarray) -> None:
    """原始标签必须是非空二维整数数组，像素值只允许 0~7。"""
    if not isinstance(raw_mask, np.ndarray):
        raise TypeError("原始 mask 必须是 numpy.ndarray。")
    if raw_mask.ndim != 2 or raw_mask.size == 0:
        raise ValueError("原始 mask 必须是非空的单通道二维类别索引图。")
    if not np.issubdtype(raw_mask.dtype, np.integer):
        raise TypeError("原始 mask 必须包含整数类别索引。")
    if raw_mask.min() < RAW_IGNORE_INDEX or raw_mask.max() > NUM_CLASSES:
        raise ValueError(
            f"LoveDA 原始 mask 只允许 0~7，实际值为 {np.unique(raw_mask).tolist()}。"
        )


def map_raw_mask(raw_mask: np.ndarray) -> np.ndarray:
    """原始 0 → ignore 255；原始 1~7 → 训练索引 0~6；返回 int64。"""
    validate_raw_mask(raw_mask)
    mapped = np.full(raw_mask.shape, IGNORE_INDEX, dtype=np.int64)
    valid = raw_mask != RAW_IGNORE_INDEX
    mapped[valid] = raw_mask[valid].astype(np.int64) - 1
    return mapped


def encode_loveda_mask(prediction: np.ndarray) -> np.ndarray:
    """训练预测索引 0~6 → LoveDA 原始类别 1~7；返回 uint8。

    预测没有真实 no-data mask，不能据此恢复原始 0 类。
    """
    if not isinstance(prediction, np.ndarray):
        raise TypeError("prediction 必须是 numpy.ndarray。")
    if prediction.ndim != 2 or prediction.size == 0:
        raise ValueError("prediction 必须是非空二维类别索引数组。")
    if not np.issubdtype(prediction.dtype, np.integer):
        raise TypeError("prediction 必须包含整数类别索引。")
    if prediction.min() < 0 or prediction.max() >= NUM_CLASSES:
        raise ValueError("预测类别索引只允许 0~6。")
    return prediction.astype(np.uint8) + np.uint8(1)
