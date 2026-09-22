"""U-Net 3+ 使用的基础卷积块。"""

from torch import nn


def make_group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """选择不超过 max_groups 且可整除通道数的最大分组数。"""
    if channels <= 0:
        raise ValueError("channels 必须大于 0。")
    for groups in (32, 16, 8, 4, 2, 1):
        if groups <= max_groups and channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ConvBNReLU(nn.Sequential):
    """一个卷积投影或融合层：Conv2d → BatchNorm2d → ReLU。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    """Encoder 中连续两次 Conv → BN → ReLU，空间尺寸保持不变。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            ConvBNReLU(in_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )


class ConvGNReLU(nn.Sequential):
    """小 batch Decoder 使用的 Conv2d → GroupNorm → ReLU。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_group_norm(out_channels),
            nn.ReLU(inplace=True),
        )
