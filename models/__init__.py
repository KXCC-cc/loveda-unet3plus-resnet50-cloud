"""分割模型构建入口；旧 scratch 模型保持可用。"""

from .unet3plus import MODEL_VERSION, UNet3Plus
from .unet3plus_resnet import RESNET_MODEL_VERSION, UNet3PlusResNet


def build_model(config: dict):
    """按配置构建旧 scratch 或新增 ResNet U-Net 3+。"""
    model_type = config.get("type", "unet3plus_resnet")
    if model_type == "unet3plus_scratch":
        return UNet3Plus(num_classes=int(config.get("num_classes", 7)))
    if model_type == "unet3plus_resnet":
        return UNet3PlusResNet(
            num_classes=int(config.get("num_classes", 7)),
            backbone=config.get("backbone", "resnet34"),
            pretrained=bool(config.get("pretrained", True)),
            weights=config.get("weights", "IMAGENET1K_V1"),
            standard_resnet_stem=bool(config.get("standard_resnet_stem", False)),
            freeze_encoder_bn=bool(config.get("freeze_encoder_bn", True)),
            cat_channels=int(config.get("cat_channels", 64)),
        )
    raise ValueError(f"未知 model.type：{model_type!r}")


__all__ = [
    "MODEL_VERSION",
    "RESNET_MODEL_VERSION",
    "UNet3Plus",
    "UNet3PlusResNet",
    "build_model",
]
