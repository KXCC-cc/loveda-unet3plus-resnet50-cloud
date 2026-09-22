"""可用于语义分割的 ImageNet 预训练骨干。"""

from .resnet import FrozenBatchNorm2d, ResNetEncoder

__all__ = ["FrozenBatchNorm2d", "ResNetEncoder"]
