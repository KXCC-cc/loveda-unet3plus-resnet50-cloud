"""LoveDA U-Net 3+ 本地网页演示。

默认使用 CPU，避免与正在运行的训练或验证任务争用 GPU。
训练/验证结束后可通过 --device cuda:0 启用 GPU 推理。
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

from models import build_model
from utils.constants import CLASS_NAMES, NUM_CLASSES, encode_loveda_mask
from utils.device import resolve_device
from utils.experiment import CLASS_COLORS, colorize_mask
from utils.inference import multi_scale_sliding_window_logits
from utils.transforms import preprocess_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "resnet34_pretrained_512_randomcrop"
    / "best_model.pth"
)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


def image_to_data_url(image: Image.Image, mode: str = "PNG") -> str:
    """将 PIL 图像编码为浏览器可直接显示和下载的 data URL。"""
    buffer = io.BytesIO()
    image.save(buffer, format=mode)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{mode.lower()};base64,{encoded}"


class SegmentationRuntime:
    """模型只加载一次；锁保证同一时刻只执行一张图片的推理。"""

    def __init__(
        self,
        checkpoint_path: Path,
        device_name: str,
        amp_enabled: bool,
        patch_batch_size: int | None,
    ) -> None:
        self.checkpoint_path = checkpoint_path.resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到 checkpoint：{self.checkpoint_path}")

        self.device = resolve_device(device_name)
        self.amp_enabled = self.device.type == "cuda" and amp_enabled
        self._lock = threading.Lock()

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        config = checkpoint.get("config")
        if not isinstance(config, dict) or "model" not in config:
            raise ValueError("网页演示只支持包含 config.model 的新版 checkpoint。")
        self.config: dict[str, Any] = config

        # checkpoint 已包含完整权重。构建时关闭 pretrained，避免重复联网下载。
        model_config = copy.deepcopy(config["model"])
        model_config["pretrained"] = False
        model_config["weights"] = None
        self.model = build_model(model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device).eval()

        validation = config.get("validation", {})
        self.tile_size = tuple(validation.get("tile_size", (512, 512)))
        self.stride = tuple(validation.get("stride", self.tile_size))
        self.patch_batch_size = (
            int(patch_batch_size)
            if patch_batch_size is not None
            else int(validation.get("patch_batch_size", 1))
        )
        # CPU 累加可减少显存占用，也能避免大图在 GPU 上长期保存 logits。
        self.accumulate_on_device = (
            self.device.type == "cuda"
            and bool(validation.get("accumulate_on_device", False))
        )
        self.backbone = str(config["model"].get("backbone", "unknown"))
        self.best_miou = float(checkpoint.get("best_miou", 0.0))
        self.best_epoch = int(checkpoint.get("epoch", 0))

    def predict(self, source: Image.Image) -> tuple[np.ndarray, float]:
        """返回训练索引 0~6 的预测图和推理耗时。"""
        image_tensor = preprocess_image(source.convert("RGB"))
        started = time.perf_counter()
        with self._lock:
            logits = multi_scale_sliding_window_logits(
                model=self.model,
                image=image_tensor,
                scales=(1.0,),
                tile_size=self.tile_size,
                stride=self.stride,
                device=self.device,
                amp_enabled=self.amp_enabled,
                tile_batch_size=self.patch_batch_size,
                horizontal_flip=False,
                accumulate_on_device=self.accumulate_on_device,
            )
        elapsed = time.perf_counter() - started
        prediction = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
        return prediction, elapsed

    def public_info(self) -> dict[str, str]:
        """提供给页面显示的非敏感运行信息。"""
        return {
            "model": f"U-Net 3+ · {self.backbone}",
            "device": str(self.device),
            "amp": "开启" if self.amp_enabled else "关闭",
            "tile": f"{self.tile_size[0]} × {self.tile_size[1]}",
            "best": f"{self.best_miou * 100:.2f}%",
            "epoch": str(self.best_epoch),
        }


def build_result(
    source: Image.Image,
    filename: str,
    prediction: np.ndarray,
    elapsed: float,
) -> dict[str, Any]:
    """生成页面所需的原图、彩色 mask、叠加图、标签图和类别占比。"""
    rgb = source.convert("RGB")
    mask_rgb = Image.fromarray(colorize_mask(prediction), mode="RGB")
    overlay = Image.blend(rgb, mask_rgb, alpha=0.48)
    loveda_mask = Image.fromarray(encode_loveda_mask(prediction), mode="L")

    counts = np.bincount(prediction.reshape(-1), minlength=NUM_CLASSES)
    total = int(counts.sum())
    classes = []
    for index, name in enumerate(CLASS_NAMES):
        percentage = float(counts[index]) * 100.0 / total
        classes.append(
            {
                "name": name,
                "pixels": f"{int(counts[index]):,}",
                "percentage": percentage,
                "percentage_text": f"{percentage:.2f}%",
                "color": "#{:02x}{:02x}{:02x}".format(
                    *CLASS_COLORS[index].tolist()
                ),
            }
        )

    return {
        "filename": filename,
        "width": rgb.width,
        "height": rgb.height,
        "elapsed": f"{elapsed:.2f}",
        "original": image_to_data_url(rgb),
        "mask": image_to_data_url(mask_rgb),
        "overlay": image_to_data_url(overlay),
        "raw_mask": image_to_data_url(loveda_mask),
        "classes": classes,
    }


def create_app(runtime: SegmentationRuntime) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            result=None,
            error=None,
            runtime=runtime.public_info(),
        )

    @app.post("/predict")
    def predict():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return render_template(
                "index.html",
                result=None,
                error="请选择一张遥感图片。",
                runtime=runtime.public_info(),
            ), 400

        try:
            payload = uploaded.read()
            with Image.open(io.BytesIO(payload)) as opened:
                opened.load()
                source = opened.convert("RGB")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    "图片像素过大，请上传不超过 2500 万像素的图片。"
                )
            prediction, elapsed = runtime.predict(source)
            result = build_result(
                source=source,
                filename=Path(uploaded.filename).name,
                prediction=prediction,
                elapsed=elapsed,
            )
            return render_template(
                "index.html",
                result=result,
                error=None,
                runtime=runtime.public_info(),
            )
        except (UnidentifiedImageError, OSError):
            message = "无法读取该文件，请上传有效的 PNG、JPG、JPEG、TIF 或 TIFF 图片。"
        except ValueError as exc:
            message = str(exc)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            message = "GPU 显存不足。请减小 patch batch，或使用 --device cpu 启动。"
        return render_template(
            "index.html",
            result=None,
            error=message,
            runtime=runtime.public_info(),
        ), 400

    @app.errorhandler(413)
    def upload_too_large(_error):
        return render_template(
            "index.html",
            result=None,
            error="上传文件超过 20 MB，请压缩后重试。",
            runtime=runtime.public_info(),
        ), 413

    @app.get("/health")
    def health():
        return {"status": "ok", **runtime.public_info()}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoveDA 语义分割网页演示")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="新版 best_model.pth 路径",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="默认 cpu，不抢训练/验证 GPU；结束后可指定 cuda:0",
    )
    parser.add_argument(
        "--patch-batch-size",
        type=int,
        default=None,
        help="滑窗小批量，留空则使用 checkpoint 配置",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = SegmentationRuntime(
        checkpoint_path=args.checkpoint,
        device_name=args.device,
        amp_enabled=not args.no_amp,
        patch_batch_size=args.patch_batch_size,
    )
    app = create_app(runtime)
    print(f"Web demo: http://{args.host}:{args.port}")
    print(
        f"Model: {runtime.backbone} | device={runtime.device} | "
        f"AMP={runtime.amp_enabled}"
    )
    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
