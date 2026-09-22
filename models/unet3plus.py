"""适用于 LoveDA 七分类分割的 U-Net 3+。

Decoder 显式写出五路全尺度连接，便于逐层追踪数据流。
模型始终返回原始 logits；softmax 由损失函数负责，推理使用 argmax。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import NUM_CLASSES

from .blocks import ConvBNReLU, DoubleConv

MODEL_VERSION = "unet3plus_loveda_v2"


class UNet3Plus(nn.Module):
    """五层 Encoder、四层全尺度 Decoder，带五个深监督输出。

    输入：[B, in_channels, H, W]，H/W 至少为 32，且均为 16 的倍数。
    默认输出：{"main": d1, "aux": [d2, d3, d4, d5]}。
    d5 来自最深层 E5；五个输出均为 [B, num_classes, H, W] 的 logits。
    return_aux=False 时只计算主分类头，aux 返回空列表。

    本项目保留 1 × 1 分类头；原论文分类头为 3 × 3。
    LoveDA 为七类语义分割，此处不加入原论文用于器官有无判断的 CGM。
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        if in_channels < 1 or num_classes < 2:
            raise ValueError("in_channels 必须为正整数，num_classes 至少为 2。")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.cat_channels = 64
        self.up_channels = self.cat_channels * 5  # 五路各 64 通道，共 320 通道。

        # Encoder：五层双卷积，四次 2 倍最大池化。
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        self.encoder5 = DoubleConv(512, 1024)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool8 = nn.MaxPool2d(kernel_size=8, stride=8)

        # D4 的五路：E1、E2、E3、E4、E5 → H/8 × W/8。
        self.d4_from_e1 = ConvBNReLU(64, 64)
        self.d4_from_e2 = ConvBNReLU(128, 64)
        self.d4_from_e3 = ConvBNReLU(256, 64)
        self.d4_from_e4 = ConvBNReLU(512, 64)
        self.d4_from_e5 = ConvBNReLU(1024, 64)
        self.d4_fusion = ConvBNReLU(320, 320)

        # D3 的五路：E1、E2、E3、D4、E5 → H/4 × W/4。
        self.d3_from_e1 = ConvBNReLU(64, 64)
        self.d3_from_e2 = ConvBNReLU(128, 64)
        self.d3_from_e3 = ConvBNReLU(256, 64)
        self.d3_from_d4 = ConvBNReLU(320, 64)
        self.d3_from_e5 = ConvBNReLU(1024, 64)
        self.d3_fusion = ConvBNReLU(320, 320)

        # D2 的五路：E1、E2、D3、D4、E5 → H/2 × W/2。
        self.d2_from_e1 = ConvBNReLU(64, 64)
        self.d2_from_e2 = ConvBNReLU(128, 64)
        self.d2_from_d3 = ConvBNReLU(320, 64)
        self.d2_from_d4 = ConvBNReLU(320, 64)
        self.d2_from_e5 = ConvBNReLU(1024, 64)
        self.d2_fusion = ConvBNReLU(320, 320)

        # D1 的五路：E1、D2、D3、D4、E5 → H × W。
        self.d1_from_e1 = ConvBNReLU(64, 64)
        self.d1_from_d2 = ConvBNReLU(320, 64)
        self.d1_from_d3 = ConvBNReLU(320, 64)
        self.d1_from_d4 = ConvBNReLU(320, 64)
        self.d1_from_e5 = ConvBNReLU(1024, 64)
        self.d1_fusion = ConvBNReLU(320, 320)

        # 四个 Decoder 和最深 Encoder 各自输出分类 logits。
        # D1～D4 特征为 320 通道，E5 特征为 1024 通道。
        self.head_d1 = nn.Conv2d(320, num_classes, kernel_size=1)
        self.head_d2 = nn.Conv2d(320, num_classes, kernel_size=1)
        self.head_d3 = nn.Conv2d(320, num_classes, kernel_size=1)
        self.head_d4 = nn.Conv2d(320, num_classes, kernel_size=1)
        self.head_d5 = nn.Conv2d(1024, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, return_aux: bool = True) -> dict:
        """显式融合全尺度特征，可按需省略四个辅助分类头。"""
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"输入必须为 [B, {self.in_channels}, H, W]。")
        height, width = x.shape[-2:]
        if height < 32 or width < 32 or height % 16 or width % 16:
            raise ValueError("输入 H 和 W 至少为 32，且必须均为 16 的倍数。")
        input_size = (height, width)

        # H/W 表示输入高宽，不必相等；注释中的具体数字以 256 × 256 为例。
        # e1: [B, 64, H, W]，例如 [B, 64, 256, 256]。
        e1 = self.encoder1(x)
        # e2: [B, 128, H/2, W/2]，例如 [B, 128, 128, 128]。
        e2 = self.encoder2(self.pool2(e1))
        # e3: [B, 256, H/4, W/4]，例如 [B, 256, 64, 64]。
        e3 = self.encoder3(self.pool2(e2))
        # e4: [B, 512, H/8, W/8]，例如 [B, 512, 32, 32]。
        e4 = self.encoder4(self.pool2(e3))
        # e5: [B, 1024, H/16, W/16]，例如 [B, 1024, 16, 16]。
        e5 = self.encoder5(self.pool2(e4))

        # D4：先调整各路空间尺寸，再分别 Conv → BN → ReLU 到 64 通道。
        # 五路统一为 [B, 64, H/8, W/8]，例如 [B, 64, 32, 32]。
        # E1 下采样 8 倍：256 → 32；d4_e1: [B, 64, 32, 32]。
        d4_e1 = self.d4_from_e1(self.pool8(e1))
        # E2 下采样 4 倍：128 → 32；d4_e2: [B, 64, 32, 32]。
        d4_e2 = self.d4_from_e2(self.pool4(e2))
        # E3 下采样 2 倍：64 → 32；d4_e3: [B, 64, 32, 32]。
        d4_e3 = self.d4_from_e3(self.pool2(e3))
        d4_e4 = self.d4_from_e4(e4)            # d4_e4: [B, 64, 32, 32]
        # E5 上采样 2 倍：16 → 32；d4_e5: [B, 64, 32, 32]。
        d4_e5 = self.d4_from_e5(
            F.interpolate(
                e5,
                size=e4.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        d4_cat = torch.cat([d4_e1, d4_e2, d4_e3, d4_e4, d4_e5], dim=1)
        # d4_cat: [B, 320, H/8, W/8]，例如 [B, 320, 32, 32]。
        d4 = self.d4_fusion(d4_cat)            # d4: [B, 320, 32, 32]

        # D3：浅层 Encoder + 同层 Encoder + 更深 Decoder + 最深 Encoder。
        # 五路统一为 [B, 64, H/4, W/4]，例如 [B, 64, 64, 64]。
        d3_e1 = self.d3_from_e1(self.pool4(e1))  # d3_e1: [B, 64, 64, 64]
        d3_e2 = self.d3_from_e2(self.pool2(e2))  # d3_e2: [B, 64, 64, 64]
        d3_e3 = self.d3_from_e3(e3)             # d3_e3: [B, 64, 64, 64]
        # D4 上采样 2 倍：32 → 64；d3_d4: [B, 64, 64, 64]。
        d3_d4 = self.d3_from_d4(
            F.interpolate(
                d4,
                size=e3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        # E5 上采样 4 倍：16 → 64；d3_e5: [B, 64, 64, 64]。
        d3_e5 = self.d3_from_e5(
            F.interpolate(
                e5,
                size=e3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        d3_cat = torch.cat([d3_e1, d3_e2, d3_e3, d3_d4, d3_e5], dim=1)
        # d3_cat: [B, 320, H/4, W/4]，例如 [B, 320, 64, 64]。
        d3 = self.d3_fusion(d3_cat)             # d3: [B, 320, 64, 64]

        # D2：以 E2 的 H/2 × W/2 作为目标空间尺寸。
        # 五路统一为 [B, 64, H/2, W/2]，例如 [B, 64, 128, 128]。
        d2_e1 = self.d2_from_e1(self.pool2(e1))  # d2_e1: [B, 64, 128, 128]
        d2_e2 = self.d2_from_e2(e2)             # d2_e2: [B, 64, 128, 128]
        # D3 上采样 2 倍；d2_d3: [B, 64, 128, 128]。
        d2_d3 = self.d2_from_d3(
            F.interpolate(
                d3,
                size=e2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        # D4 上采样 4 倍；d2_d4: [B, 64, 128, 128]。
        d2_d4 = self.d2_from_d4(
            F.interpolate(
                d4,
                size=e2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        # E5 上采样 8 倍；d2_e5: [B, 64, 128, 128]。
        d2_e5 = self.d2_from_e5(
            F.interpolate(
                e5,
                size=e2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        d2_cat = torch.cat([d2_e1, d2_e2, d2_d3, d2_d4, d2_e5], dim=1)
        # d2_cat: [B, 320, H/2, W/2]，例如 [B, 320, 128, 128]。
        d2 = self.d2_fusion(d2_cat)             # d2: [B, 320, 128, 128]

        # D1：以输入的 H × W 作为目标空间尺寸。
        # 五路统一为 [B, 64, H, W]，例如 [B, 64, 256, 256]。
        d1_e1 = self.d1_from_e1(e1)             # d1_e1: [B, 64, 256, 256]
        # D2 上采样 2 倍；d1_d2: [B, 64, 256, 256]。
        d1_d2 = self.d1_from_d2(
            F.interpolate(
                d2, size=input_size, mode="bilinear", align_corners=False
            )
        )
        # D3 上采样 4 倍；d1_d3: [B, 64, 256, 256]。
        d1_d3 = self.d1_from_d3(
            F.interpolate(
                d3, size=input_size, mode="bilinear", align_corners=False
            )
        )
        # D4 上采样 8 倍；d1_d4: [B, 64, 256, 256]。
        d1_d4 = self.d1_from_d4(
            F.interpolate(
                d4, size=input_size, mode="bilinear", align_corners=False
            )
        )
        # E5 上采样 16 倍；d1_e5: [B, 64, 256, 256]。
        d1_e5 = self.d1_from_e5(
            F.interpolate(
                e5, size=input_size, mode="bilinear", align_corners=False
            )
        )
        d1_cat = torch.cat([d1_e1, d1_d2, d1_d3, d1_d4, d1_e5], dim=1)
        # d1_cat: [B, 320, H, W]，例如 [B, 320, 256, 256]。
        d1 = self.d1_fusion(d1_cat)             # d1: [B, 320, 256, 256]

        # 主分类头：[B, C, H, W]；C = num_classes，LoveDA 默认 C = 7。
        d1_logits = self.head_d1(d1)  # 例如 [B, 7, 256, 256]。
        if not return_aux:
            # 推理仍需完整 Decoder，仅省略辅助分类头及其结果的上采样。
            return {"main": d1_logits, "aux": []}

        # 深监督：另外四个独立的 1 × 1 分类头，所有输出都是 logits。
        d2_logits = self.head_d2(d2)  # [B, C, H/2, W/2]；例如 128 × 128。
        d3_logits = self.head_d3(d3)  # [B, C, H/4, W/4]；例如 64 × 64。
        d4_logits = self.head_d4(d4)  # [B, C, H/8, W/8]；例如 32 × 32。
        d5_logits = self.head_d5(e5)  # [B, C, H/16, W/16]；例如 16 × 16。
        d2_logits = F.interpolate(
            d2_logits, size=input_size, mode="bilinear", align_corners=False
        )                                     # [B, 7, 256, 256]
        d3_logits = F.interpolate(
            d3_logits, size=input_size, mode="bilinear", align_corners=False
        )                                     # [B, 7, 256, 256]
        d4_logits = F.interpolate(
            d4_logits, size=input_size, mode="bilinear", align_corners=False
        )                                     # [B, 7, 256, 256]
        d5_logits = F.interpolate(
            d5_logits, size=input_size, mode="bilinear", align_corners=False
        )                                     # [B, 7, 256, 256]
        # 所有辅助输出：[B, C, H, W]，例如 [B, 7, 256, 256]。
        return {
            "main": d1_logits,
            "aux": [d2_logits, d3_logits, d4_logits, d5_logits],
        }
