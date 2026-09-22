"""把 torchvision ResNet 改造成五尺度语义分割 Encoder。"""

from __future__ import annotations

import torch
from torch import nn


class FrozenBatchNorm2d(nn.Module):
    """固定 affine 参数和 running statistics 的 BatchNorm2d。

    它没有训练态统计更新，因此外层调用 ``model.train()`` 后仍然稳定，适合
    512 裁块、physical batch 为 1～2 的训练。
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.weight * torch.rsqrt(self.running_var + self.eps)
        bias = self.bias - self.running_mean * scale
        # autocast 下保持激活 dtype，避免冻结 BN 意外把整条 Encoder 提升回 FP32。
        scale = scale.to(dtype=x.dtype)
        bias = bias.to(dtype=x.dtype)
        return x * scale.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)


def _replace_batch_norm(module: nn.Module) -> None:
    """原地把已载入预训练统计的 BatchNorm2d 转为 FrozenBatchNorm2d。"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, child.eps)
            with torch.no_grad():
                frozen.weight.copy_(child.weight.detach())
                frozen.bias.copy_(child.bias.detach())
                frozen.running_mean.copy_(child.running_mean.detach())
                frozen.running_var.copy_(child.running_var.detach())
            setattr(module, name, frozen)
        else:
            _replace_batch_norm(child)


class ResNetEncoder(nn.Module):
    """ResNet34/50 的五尺度特征提取器。

    segmentation stem（默认）把 conv1 stride 从 2 改为 1，保持预训练的
    7×7 权重，同时得到 H、H/2、H/4、H/8、H/16 五个尺度。
    standard_resnet_stem=True 时保留分类 ResNet 的 H/2～H/32 尺度，供消融。
    """

    CHANNELS = {
        "resnet34": (64, 64, 128, 256, 512),
        "resnet50": (64, 256, 512, 1024, 2048),
    }

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        weights: str = "IMAGENET1K_V1",
        standard_resnet_stem: bool = False,
        freeze_bn: bool = True,
    ) -> None:
        super().__init__()
        if backbone not in self.CHANNELS:
            raise ValueError("backbone 仅支持 'resnet34' 或 'resnet50'。")

        try:
            from torchvision.models import (
                ResNet34_Weights,
                ResNet50_Weights,
                resnet34,
                resnet50,
            )
        except ImportError as exc:
            raise ImportError("UNet3PlusResNet 需要 torchvision。") from exc

        enum_class = ResNet34_Weights if backbone == "resnet34" else ResNet50_Weights
        if pretrained:
            try:
                selected_weights = enum_class.DEFAULT if weights == "DEFAULT" else enum_class[weights]
            except KeyError as exc:
                choices = ", ".join(item.name for item in enum_class)
                raise ValueError(f"{backbone} weights={weights!r} 无效，可选：{choices}、DEFAULT。") from exc
        else:
            selected_weights = None

        network = (resnet34 if backbone == "resnet34" else resnet50)(
            weights=selected_weights
        )
        if not standard_resnet_stem:
            network.conv1.stride = (1, 1)
        if freeze_bn:
            _replace_batch_norm(network)

        self.backbone_name = backbone
        self.pretrained = pretrained
        self.weights_name = selected_weights.name if selected_weights is not None else None
        self.standard_resnet_stem = standard_resnet_stem
        self.freeze_bn = freeze_bn
        self.out_channels = self.CHANNELS[backbone]

        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.layer4 = network.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        e1 = self.relu(self.bn1(self.conv1(x)))
        e2 = self.layer1(self.maxpool(e1))
        e3 = self.layer2(e2)
        e4 = self.layer3(e3)
        e5 = self.layer4(e4)
        return e1, e2, e3, e4, e5
