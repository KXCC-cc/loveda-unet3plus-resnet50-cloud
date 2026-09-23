"""四模型 Web Demo 登记表与 checkpoint 身份校验。"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLOUD_PROJECT_ROOT = Path(
    "/mnt/d/loveda-unet3plus-resnet50-cloud-final/"
    "loveda-unet3plus-resnet50-cloud_full_except_data_submission/"
    "loveda-unet3plus-resnet50-cloud"
)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    subtitle: str
    feature: str
    checkpoint: Path
    test_miou: float
    expected_backbone: str
    expected_cat_channels: int
    expected_run_name: str
    requires_differential_lr: bool = False
    final: bool = False

    def public_info(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "subtitle": self.subtitle,
            "feature": self.feature,
            "test_miou": self.test_miou,
            "test_miou_percent": f"{self.test_miou * 100:.2f}%",
            "backbone": {"resnet34": "ResNet34", "resnet50": "ResNet50"}[self.expected_backbone],
            "cat_channels": self.expected_cat_channels,
            "decoder": "U-Net 3+",
            "final": self.final,
        }


MODEL_SPECS = (
    ModelSpec(
        id="resnet34_baseline",
        name="ResNet34 Baseline",
        subtitle="ImageNet ResNet34 · U-Net 3+",
        feature="baseline",
        checkpoint=PROJECT_ROOT
        / "runs/resnet34_pretrained_512_randomcrop/best_model.pth",
        test_miou=0.457311,
        expected_backbone="resnet34",
        expected_cat_channels=64,
        expected_run_name="resnet34_pretrained_512_randomcrop",
    ),
    ModelSpec(
        id="resnet50_cat48",
        name="ResNet50 Cat48",
        subtitle="ImageNet ResNet50 · U-Net 3+",
        feature="cat=48，本地显存妥协版",
        checkpoint=PROJECT_ROOT
        / "runs/resnet50_pretrained_512_cat48_fast/best_model.pth",
        test_miou=0.446118,
        expected_backbone="resnet50",
        expected_cat_channels=48,
        expected_run_name="resnet50_pretrained_512_cat48_fast",
    ),
    ModelSpec(
        id="resnet34_diff_lr",
        name="ResNet34 Diff-LR",
        subtitle="ImageNet ResNet34 · U-Net 3+",
        feature="differential LR ablation",
        checkpoint=PROJECT_ROOT / "runs/resnet34_512_diff_lr/best_model.pth",
        test_miou=0.454974,
        expected_backbone="resnet34",
        expected_cat_channels=64,
        expected_run_name="resnet34_512_diff_lr",
        requires_differential_lr=True,
    ),
    ModelSpec(
        id="resnet50_cloud_final",
        name="ResNet50 Cloud Final",
        subtitle="ImageNet ResNet50 · U-Net 3+",
        feature="cat=64，云端正式模型",
        checkpoint=CLOUD_PROJECT_ROOT
        / "runs/cloud_resnet50_24gb_eff4_40ep/best_model.pth",
        test_miou=0.469813,
        expected_backbone="resnet50",
        expected_cat_channels=64,
        expected_run_name="cloud_resnet50_24gb",
        final=True,
    ),
)


class ModelRegistryError(RuntimeError):
    """一个或多个正式模型缺失或身份不匹配。"""


def _validate_loaded_checkpoint(
    spec: ModelSpec,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint 缺少 config。")
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint 缺少 config.model。")
    if model_config.get("type") != "unet3plus_resnet":
        raise ValueError(
            f"model.type={model_config.get('type')!r}，要求 unet3plus_resnet。"
        )

    backbone = str(model_config.get("backbone", "")).lower()
    cat_channels = int(model_config.get("cat_channels", -1))
    if backbone != spec.expected_backbone:
        raise ValueError(
            f"backbone={backbone!r}，期望 {spec.expected_backbone!r}。"
        )
    if cat_channels != spec.expected_cat_channels:
        raise ValueError(
            f"cat_channels={cat_channels}，期望 {spec.expected_cat_channels}。"
        )

    run_name = str(config.get("run", {}).get("name", ""))
    if run_name != spec.expected_run_name:
        raise ValueError(
            f"run.name={run_name!r}，期望 {spec.expected_run_name!r}。"
        )

    differential_lr = config.get("optimizer", {}).get("differential_lr")
    if spec.requires_differential_lr and not isinstance(differential_lr, dict):
        raise ValueError("该模型应包含 differential_lr 配置，但 checkpoint 中没有。")
    if not spec.requires_differential_lr and differential_lr:
        raise ValueError("该模型不应使用 differential_lr，但 checkpoint 中已启用。")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("checkpoint 缺少 model_state_dict。")
    head = state_dict.get("head_d1.weight")
    expected_head_shape = (7, spec.expected_cat_channels * 5, 1, 1)
    if head is None or tuple(head.shape) != expected_head_shape:
        actual = None if head is None else tuple(head.shape)
        raise ValueError(
            f"head_d1.weight shape={actual}，期望 {expected_head_shape}。"
        )

    return {
        "path": str(spec.checkpoint.resolve()),
        "backbone": backbone,
        "cat_channels": cat_channels,
        "run_name": run_name,
        "weights": model_config.get("resolved_weights", model_config.get("weights")),
        "best_val_miou": float(checkpoint.get("best_miou", float("nan"))),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "optimizer_step": int(checkpoint.get("optimizer_step", 0)),
        "differential_lr": differential_lr,
    }


def validate_model_spec(spec: ModelSpec) -> dict[str, Any]:
    path = spec.checkpoint.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"[MISSING] {spec.name}:\nexpected path: {path}"
        )
    checkpoint = None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint 顶层不是字典。")
        return _validate_loaded_checkpoint(spec, checkpoint)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"[INVALID] {spec.name}: {path}\n{exc}") from exc
    finally:
        del checkpoint
        gc.collect()


def validate_model_registry(
    specs: tuple[ModelSpec, ...] = MODEL_SPECS,
) -> dict[str, dict[str, Any]]:
    validated: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    ids: set[str] = set()
    for spec in specs:
        if spec.id in ids:
            errors.append(f"[INVALID] duplicate model id: {spec.id}")
            continue
        ids.add(spec.id)
        try:
            validated[spec.id] = validate_model_spec(spec)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
    if errors:
        raise ModelRegistryError(
            "四模型登记检查失败，Web Demo 未启动：\n\n" + "\n\n".join(errors)
        )
    return validated
