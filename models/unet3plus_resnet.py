"""ImageNet ResNet Encoder + U-Net 3+ 全尺度 Decoder。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from utils.constants import NUM_CLASSES

from .backbones import ResNetEncoder
from .blocks import ConvGNReLU

RESNET_MODEL_VERSION = "unet3plus_resnet_v1"


class UNet3PlusResNet(nn.Module):
    """保留 U-Net 3+ full-scale skip connection 与五头深监督。

    默认 segmentation stem 下，输入 [B,3,H,W] 的五层尺度为 H、H/2、H/4、
    H/8、H/16。Decoder 的每条支路均显式定义，便于核查数据流。
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        backbone: str = "resnet34",
        pretrained: bool = True,
        weights: str = "IMAGENET1K_V1",
        standard_resnet_stem: bool = False,
        freeze_encoder_bn: bool = True,
        cat_channels: int = 64,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cat_channels = cat_channels
        self.up_channels = cat_channels * 5
        self.encoder = ResNetEncoder(
            backbone=backbone,
            pretrained=pretrained,
            weights=weights,
            standard_resnet_stem=standard_resnet_stem,
            freeze_bn=freeze_encoder_bn,
        )
        c1, c2, c3, c4, c5 = self.encoder.out_channels
        c = cat_channels
        up = self.up_channels

        # D4：E1、E2、E3、E4、E5 -> E4 的空间尺寸。
        self.d4_from_e1 = ConvGNReLU(c1, c)
        self.d4_from_e2 = ConvGNReLU(c2, c)
        self.d4_from_e3 = ConvGNReLU(c3, c)
        self.d4_from_e4 = ConvGNReLU(c4, c)
        self.d4_from_e5 = ConvGNReLU(c5, c)
        self.d4_fusion = ConvGNReLU(up, up)

        # D3：E1、E2、E3、D4、E5 -> E3 的空间尺寸。
        self.d3_from_e1 = ConvGNReLU(c1, c)
        self.d3_from_e2 = ConvGNReLU(c2, c)
        self.d3_from_e3 = ConvGNReLU(c3, c)
        self.d3_from_d4 = ConvGNReLU(up, c)
        self.d3_from_e5 = ConvGNReLU(c5, c)
        self.d3_fusion = ConvGNReLU(up, up)

        # D2：E1、E2、D3、D4、E5 -> E2 的空间尺寸。
        self.d2_from_e1 = ConvGNReLU(c1, c)
        self.d2_from_e2 = ConvGNReLU(c2, c)
        self.d2_from_d3 = ConvGNReLU(up, c)
        self.d2_from_d4 = ConvGNReLU(up, c)
        self.d2_from_e5 = ConvGNReLU(c5, c)
        self.d2_fusion = ConvGNReLU(up, up)

        # D1：E1、D2、D3、D4、E5 -> E1 的空间尺寸。
        self.d1_from_e1 = ConvGNReLU(c1, c)
        self.d1_from_d2 = ConvGNReLU(up, c)
        self.d1_from_d3 = ConvGNReLU(up, c)
        self.d1_from_d4 = ConvGNReLU(up, c)
        self.d1_from_e5 = ConvGNReLU(c5, c)
        self.d1_fusion = ConvGNReLU(up, up)

        self.head_d1 = nn.Conv2d(up, num_classes, kernel_size=1)
        self.head_d2 = nn.Conv2d(up, num_classes, kernel_size=1)
        self.head_d3 = nn.Conv2d(up, num_classes, kernel_size=1)
        self.head_d4 = nn.Conv2d(up, num_classes, kernel_size=1)
        self.head_d5 = nn.Conv2d(c5, num_classes, kernel_size=1)

    @staticmethod
    def _downsample(feature: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.adaptive_max_pool2d(feature, output_size=size)

    @staticmethod
    def _upsample(feature: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, return_aux: bool = True) -> dict[str, object]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("输入必须是 [B,3,H,W]。")
        input_size = x.shape[-2:]

        # 默认 segmentation stem，以 512 输入为例：
        e1, e2, e3, e4, e5 = self.encoder(x)
        # e1: ResNet34/50 [B,64,H,W]                  -> [B,64,512,512]
        # e2: ResNet34 [B,64,H/2,W/2] / ResNet50 [B,256,...] -> 256×256
        # e3: ResNet34 [B,128,H/4,W/4] / ResNet50 [B,512,...] -> 128×128
        # e4: ResNet34 [B,256,H/8,W/8] / ResNet50 [B,1024,...] -> 64×64
        # e5: ResNet34 [B,512,H/16,W/16] / ResNet50 [B,2048,...] -> 32×32

        # D4：五路全部对齐到 E4，再投影成 64 通道。
        size4 = e4.shape[-2:]
        d4_e1 = self.d4_from_e1(self._downsample(e1, size4))  # [B,64,H/8,W/8]
        d4_e2 = self.d4_from_e2(self._downsample(e2, size4))  # [B,64,H/8,W/8]
        d4_e3 = self.d4_from_e3(self._downsample(e3, size4))  # [B,64,H/8,W/8]
        d4_e4 = self.d4_from_e4(e4)                          # [B,64,H/8,W/8]
        d4_e5 = self.d4_from_e5(self._upsample(e5, size4))   # [B,64,H/8,W/8]
        d4_cat = torch.cat((d4_e1, d4_e2, d4_e3, d4_e4, d4_e5), dim=1)
        # d4_cat: [B,320,H/8,W/8]；512 输入为 [B,320,64,64]。
        d4 = self.d4_fusion(d4_cat)

        # D3：E1、E2、E3、D4、E5 全尺度融合到 H/4。
        size3 = e3.shape[-2:]
        d3_e1 = self.d3_from_e1(self._downsample(e1, size3))  # [B,64,H/4,W/4]
        d3_e2 = self.d3_from_e2(self._downsample(e2, size3))  # [B,64,H/4,W/4]
        d3_e3 = self.d3_from_e3(e3)                          # [B,64,H/4,W/4]
        d3_d4 = self.d3_from_d4(self._upsample(d4, size3))   # [B,64,H/4,W/4]
        d3_e5 = self.d3_from_e5(self._upsample(e5, size3))   # [B,64,H/4,W/4]
        d3_cat = torch.cat((d3_e1, d3_e2, d3_e3, d3_d4, d3_e5), dim=1)
        # d3_cat: [B,320,H/4,W/4]；512 输入为 [B,320,128,128]。
        d3 = self.d3_fusion(d3_cat)

        # D2：E1、E2、D3、D4、E5 全尺度融合到 H/2。
        size2 = e2.shape[-2:]
        d2_e1 = self.d2_from_e1(self._downsample(e1, size2))  # [B,64,H/2,W/2]
        d2_e2 = self.d2_from_e2(e2)                          # [B,64,H/2,W/2]
        d2_d3 = self.d2_from_d3(self._upsample(d3, size2))   # [B,64,H/2,W/2]
        d2_d4 = self.d2_from_d4(self._upsample(d4, size2))   # [B,64,H/2,W/2]
        d2_e5 = self.d2_from_e5(self._upsample(e5, size2))   # [B,64,H/2,W/2]
        d2_cat = torch.cat((d2_e1, d2_e2, d2_d3, d2_d4, d2_e5), dim=1)
        # d2_cat: [B,320,H/2,W/2]；512 输入为 [B,320,256,256]。
        d2 = self.d2_fusion(d2_cat)

        # D1：E1、D2、D3、D4、E5 全尺度融合到 E1 尺度。
        size1 = e1.shape[-2:]
        d1_e1 = self.d1_from_e1(e1)                          # [B,64,H,W]
        d1_d2 = self.d1_from_d2(self._upsample(d2, size1))   # [B,64,H,W]
        d1_d3 = self.d1_from_d3(self._upsample(d3, size1))   # [B,64,H,W]
        d1_d4 = self.d1_from_d4(self._upsample(d4, size1))   # [B,64,H,W]
        d1_e5 = self.d1_from_e5(self._upsample(e5, size1))   # [B,64,H,W]
        d1_cat = torch.cat((d1_e1, d1_d2, d1_d3, d1_d4, d1_e5), dim=1)
        # d1_cat: [B,320,H,W]；512 输入为 [B,320,512,512]。
        d1 = self.d1_fusion(d1_cat)

        main = self.head_d1(d1)
        if main.shape[-2:] != input_size:  # standard ResNet stem 消融时恢复原分辨率。
            main = self._upsample(main, input_size)
        if not return_aux:
            return {"main": main, "aux": []}

        # Deep Supervision：D2、D3、D4、E5 先产生 logits，再上采样到 GT 尺寸。
        aux = [
            self._upsample(self.head_d2(d2), input_size),
            self._upsample(self.head_d3(d3), input_size),
            self._upsample(self.head_d4(d4), input_size),
            self._upsample(self.head_d5(e5), input_size),
        ]
        return {"main": main, "aux": aux}
