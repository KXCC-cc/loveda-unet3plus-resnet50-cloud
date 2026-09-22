"""LoveDA 分割损失：默认 CE + Dice，可选类别权重与 Focal Loss。"""

import math
from typing import Mapping, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from utils.constants import DEFAULT_AUX_WEIGHTS, IGNORE_INDEX, NUM_CLASSES


def _validate_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> None:
    """检查基本 shape、dtype 和设备，不反复读取 GPU 标量。

    标签范围由 Dataset 的集中映射校验，并由 CE / one_hot / gather 再约束。
    """
    if logits.ndim != 4 or targets.ndim != 3:
        raise ValueError("logits 必须是 [B, C, H, W]，targets 必须是 [B, H, W]。")
    if logits.shape[1] != num_classes:
        raise ValueError(f"输出通道应为 {num_classes}，实际为 {logits.shape[1]}。")
    if (logits.shape[0], *logits.shape[2:]) != tuple(targets.shape):
        raise ValueError("logits 和 targets 的 batch、H、W 必须一致。")
    if not logits.is_floating_point() or targets.dtype != torch.long:
        raise TypeError("logits 必须为浮点张量，targets 必须为 torch.long。")
    if logits.device != targets.device:
        raise ValueError("logits 和 targets 必须位于同一设备。")


def _differentiable_zero(logits: torch.Tensor) -> torch.Tensor:
    """空切片求和保留计算图，不对半精度 logits 大量求和，避免溢出。"""
    return logits[..., :0].float().sum()


def _make_class_weights(
    class_weights: Optional[Sequence[float]], num_classes: int
) -> Optional[torch.Tensor]:
    """权重只在构造损失时检查；默认 7 个类别，不推测数据的真实频率。"""
    if num_classes <= 0:
        raise ValueError("num_classes 必须大于 0。")
    if class_weights is None:
        return None
    if len(class_weights) != num_classes:
        raise ValueError(f"class_weights 必须包含 {num_classes} 个权重。")
    values = [float(weight) for weight in class_weights]
    if any(not math.isfinite(weight) or weight <= 0 for weight in values):
        raise ValueError("每个类别权重必须是有限正数，不能为 0。")
    weights = torch.tensor(values, dtype=torch.float32)
    # 同时拒绝转换成 FP32 后溢出为 Inf 或下溢为 0 的极端输入。
    if not torch.isfinite(weights).all().item() or not (weights > 0).all().item():
        raise ValueError("类别权重必须能够表示为 FP32 有限正数。")
    return weights


class CrossEntropyLoss(nn.Module):
    """FP32 交叉熵，支持类别权重，并对有效像素进行加权平均。"""

    def __init__(
        self,
        ignore_index: int = IGNORE_INDEX,
        class_weights: Optional[Sequence[float]] = None,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # buffer 会随 criterion.to(device) 一同移动，并写入 criterion.state_dict()。
        self.register_buffer(
            "class_weights", _make_class_weights(class_weights, num_classes)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_inputs(logits, targets, self.num_classes)
        valid = targets != self.ignore_index
        # 显式转为 FP32，损失计算不依赖调用方是否处于 autocast 上下文。
        pixel_loss = F.cross_entropy(
            logits.float(),
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        if self.class_weights is None:
            denominator = valid.float().sum()
        else:
            safe_targets = targets.masked_fill(~valid, 0)
            denominator = (self.class_weights[safe_targets] * valid.float()).sum()

        # 加权 CE 的分母是有效像素对应的权重和，而不是总像素数。
        # 全 ignore 时分母临时置 1；torch.where 避免逐输出头的 GPU .item() 同步。
        safe_denominator = torch.where(
            denominator > 0, denominator, torch.ones_like(denominator)
        )
        loss = pixel_loss.sum() / safe_denominator
        return torch.where(denominator > 0, loss, _differentiable_zero(logits))


class FocalLoss(nn.Module):
    """多分类 Focal Loss：-(1-p_t)^gamma * log(p_t)。

    先从未加权的 log_softmax 获得 p_t，再乘类别权重。有效像素按类别权重
    之和归一化；没有类别权重时按有效像素数平均。gamma=0 时等价于 CE。
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        gamma: float = 2.0,
        class_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("focal gamma 必须为有限非负数。")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.register_buffer(
            "class_weights", _make_class_weights(class_weights, num_classes)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_inputs(logits, targets, self.num_classes)
        valid = targets != self.ignore_index
        safe_targets = targets.masked_fill(~valid, 0)
        log_probabilities = F.log_softmax(logits.float(), dim=1)
        log_pt = log_probabilities.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        # p_t 舍入到 1 时，正 epsilon 避免 0<gamma<1 的 pow 导数变成 Inf。
        focal_factor = (1.0 - pt).clamp_min(torch.finfo(pt.dtype).eps).pow(self.gamma)
        pixel_loss = -focal_factor * log_pt

        pixel_weights = valid.float()
        if self.class_weights is not None:
            pixel_weights = pixel_weights * self.class_weights[safe_targets]
        denominator = pixel_weights.sum()
        safe_denominator = torch.where(
            denominator > 0, denominator, torch.ones_like(denominator)
        )
        loss = (pixel_loss * pixel_weights).sum() / safe_denominator
        return torch.where(denominator > 0, loss, _differentiable_zero(logits))


class DiceLoss(nn.Module):
    """对每个类别计算 soft Dice，再对类别取平均。

    batch 和空间维度共同参与求和；7 个实际类别等权参与计算，不使用 CE 类别权重。
    ignore 像素在预测概率和 one-hot 标签中都置零，不计入分子或分母。
    没有真实像素的类别仍参与损失，其预测概率视为误报；这与指标的空类规则不同。
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes 必须大于 0。")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon 必须为有限正数。")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_inputs(logits, targets, self.num_classes)
        valid = targets != self.ignore_index  # [B, H, W]
        # 损失内部使用 softmax；模型本身始终输出 logits。
        probabilities = F.softmax(logits.float(), dim=1)  # [B, C, H, W]

        # one_hot 不能接收 255，先临时替换为 0，再将对应位置屏蔽。
        safe_targets = targets.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_targets, num_classes=self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).to(probabilities.dtype)
        valid_mask = valid.unsqueeze(1).to(probabilities.dtype)  # [B, 1, H, W]
        probabilities = probabilities * valid_mask
        one_hot = one_hot * valid_mask

        reduce_dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dim=reduce_dims)  # [C]
        denominator = probabilities.sum(dim=reduce_dims) + one_hot.sum(dim=reduce_dims)
        dice = (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)
        loss = 1.0 - dice.mean()
        return torch.where(valid.any(), loss, _differentiable_zero(logits))


class CombinedLoss(nn.Module):
    """默认 CE + Dice；可配置为 ce_weight*CE + dice_weight*Dice + focal_weight*Focal。

    class_weights 只用于 CE / Focal，顺序与 constants.CLASS_NAMES 一致。
    Focal 默认权重为 0，因此不会增加默认训练的计算量。
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[Sequence[float]] = None,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        weights = (ce_weight, dice_weight, focal_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("CE、Dice 和 Focal 权重必须为有限非负数。")
        if sum(weights) == 0:
            raise ValueError("CE、Dice 和 Focal 权重不能全部为 0。")
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ce = CrossEntropyLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=class_weights,
        )
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.focal = FocalLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            gamma=focal_gamma,
            class_weights=class_weights,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = _differentiable_zero(logits)
        if self.ce_weight > 0:
            loss = loss + self.ce_weight * self.ce(logits, targets)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, targets)
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self.focal(logits, targets)
        return loss


def deep_supervision_loss(
    outputs: Mapping,
    targets: torch.Tensor,
    criterion: nn.Module,
    aux_weights: Sequence[float] = DEFAULT_AUX_WEIGHTS,
) -> torch.Tensor:
    """main_loss + 0.5*d2_loss + 0.25*d3_loss + 0.125*d4_loss + 0.0625*e5_loss。

    所有输出头都应已由模型调整到标签的 H、W，不能在此缩放 mask。
    """
    if "main" not in outputs or "aux" not in outputs:
        raise ValueError("模型输出必须包含 'main' 和 'aux'。")
    auxiliary_outputs = outputs["aux"]
    if not isinstance(auxiliary_outputs, (list, tuple)):
        raise TypeError("outputs['aux'] 必须为辅助 logits 的列表或元组。")
    if len(auxiliary_outputs) != len(aux_weights):
        raise ValueError("辅助输出数量与 aux_weights 长度必须完全一致。")
    if any(not math.isfinite(weight) or weight < 0 for weight in aux_weights):
        raise ValueError("辅助输出权重必须为有限非负数。")

    total_loss = criterion(outputs["main"], targets)
    for weight, auxiliary_logits in zip(aux_weights, auxiliary_outputs):
        if weight > 0:
            total_loss = total_loss + weight * criterion(auxiliary_logits, targets)
    return total_loss
