"""基于整个数据集混淆矩阵的多分类 IoU / Dice。"""

from typing import Dict, Optional, Union

import torch

from utils.constants import IGNORE_INDEX, NUM_CLASSES


class SegmentationMetrics:
    """逐 batch 累计计数，在验证结束后统一计算指标。

    混淆矩阵的行代表真实类别，列代表预测类别。
    与逐 batch 求均值相比，先累计计数可避免小 batch 改变指标权重。
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes 必须大于 0。")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        # int64 累计像素个数，避免使用浮点计数导致精度损失。
        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64, device=self.device
        )

    def reset(self) -> None:
        """每次新的验证循环开始前清空累计计数。"""
        self.confusion_matrix.zero_()

    @torch.no_grad()
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """接收预测类别 [B,H,W]，也支持模型 logits [B,C,H,W]。"""
        if predictions.ndim == 4:
            if predictions.shape[1] != self.num_classes:
                raise ValueError("logits 通道数必须等于 num_classes。")
            if not predictions.is_floating_point():
                raise TypeError("logits 必须为浮点张量。")
            predictions = predictions.argmax(dim=1)
        if predictions.ndim != 3 or targets.ndim != 3:
            raise ValueError("预测类别与标签必须均为 [B, H, W]。")
        if predictions.shape != targets.shape:
            raise ValueError("预测类别与标签的 shape 必须一致。")
        if predictions.dtype != torch.long or targets.dtype != torch.long:
            raise TypeError("预测类别与标签必须均为 torch.long。")

        predictions = predictions.detach().to(self.device)
        targets = targets.detach().to(self.device)
        valid = targets != self.ignore_index
        targets = targets[valid]
        predictions = predictions[valid]
        if targets.numel() == 0:
            return

        invalid_targets = (targets < 0) | (targets >= self.num_classes)
        invalid_predictions = (predictions < 0) | (predictions >= self.num_classes)
        if invalid_targets.any().item() or invalid_predictions.any().item():
            raise ValueError(
                f"有效位置的预测类别与真实类别必须位于 0~{self.num_classes - 1}。"
            )

        # 编码 (真实类别, 预测类别) 为一个整数，再统计每种组合出现的次数。
        encoded = targets * self.num_classes + predictions
        counts = torch.bincount(
            encoded, minlength=self.num_classes * self.num_classes
        )
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, Union[torch.Tensor, float]]:
        """返回逐类和宏平均指标；指标范围为 0~1。

        若某类既无真实像素也无预测像素，该类指标记为 NaN，宏平均排除它。
        只有预测、没有真实像素的类别仍计入平均，其指标为 0。
        若整个验证集均为 ignore，则两个均值都为 NaN，不能据此保存最优模型。
        """
        matrix = self.confusion_matrix.to(torch.float64)
        true_positive = matrix.diag()
        target_count = matrix.sum(dim=1)  # TP + FN
        prediction_count = matrix.sum(dim=0)  # TP + FP

        # IoU = TP / (TP + FP + FN)
        union = target_count + prediction_count - true_positive
        # Dice = 2*TP / (2*TP + FP + FN)
        dice_denominator = target_count + prediction_count

        per_class_iou = torch.full_like(true_positive, float("nan"))
        per_class_dice = torch.full_like(true_positive, float("nan"))
        valid_iou = union > 0
        valid_dice = dice_denominator > 0
        per_class_iou[valid_iou] = true_positive[valid_iou] / union[valid_iou]
        per_class_dice[valid_dice] = (
            2.0 * true_positive[valid_dice] / dice_denominator[valid_dice]
        )

        mean_iou = (
            per_class_iou[valid_iou].mean().item()
            if valid_iou.any().item()
            else float("nan")
        )
        mean_dice = (
            per_class_dice[valid_dice].mean().item()
            if valid_dice.any().item()
            else float("nan")
        )
        return {
            "per_class_iou": per_class_iou,
            "mean_iou": mean_iou,
            "per_class_dice": per_class_dice,
            "mean_dice": mean_dice,
        }
