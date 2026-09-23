"""LoveDA U-Net 3+ 四模型横向对比 Web Demo。"""

from __future__ import annotations

import argparse
import base64
import copy
import gc
import io
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

from models import build_model
from utils.constants import CLASS_NAMES, encode_loveda_mask
from utils.device import resolve_device
from utils.experiment import CLASS_COLORS, colorize_mask
from utils.inference import multi_scale_sliding_window_logits
from utils.transforms import preprocess_image, validate_image_size
from web_demo.model_registry import (
    MODEL_SPECS,
    ModelRegistryError,
    ModelSpec,
    validate_model_registry,
)


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
OVERLAY_ALPHA = 0.48


def image_to_data_url(image: Image.Image, mode: str = "PNG") -> str:
    """将 PIL 图像编码为浏览器可显示和下载的 data URL。"""
    buffer = io.BytesIO()
    image.save(buffer, format=mode)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{mode.lower()};base64,{encoded}"


def _legend_items() -> list[dict[str, str]]:
    return [
        {
            "name": name,
            "color": "#{:02x}{:02x}{:02x}".format(
                *CLASS_COLORS[index].tolist()
            ),
        }
        for index, name in enumerate(CLASS_NAMES)
    ]


def build_model_visual(
    source: Image.Image,
    prediction: np.ndarray,
    spec: ModelSpec,
    elapsed: float,
    load_elapsed: float,
) -> dict[str, Any]:
    """用统一颜色表生成一个模型的 mask、overlay 和原始标签图。"""
    rgb = source.convert("RGB")
    mask_rgb = Image.fromarray(colorize_mask(prediction), mode="RGB")
    overlay = Image.blend(rgb, mask_rgb, alpha=OVERLAY_ALPHA)
    loveda_mask = Image.fromarray(encode_loveda_mask(prediction), mode="L")
    result = spec.public_info()
    result.update(
        {
            "success": True,
            "checkpoint": str(spec.checkpoint.resolve()),
            "mask": image_to_data_url(mask_rgb),
            "overlay": image_to_data_url(overlay),
            "raw_mask": image_to_data_url(loveda_mask),
            "elapsed": elapsed,
            "elapsed_text": f"{elapsed:.2f} s",
            "load_elapsed": load_elapsed,
            "load_elapsed_text": f"{load_elapsed:.2f} s",
            "error": None,
        }
    )
    return result


class MultiModelSegmentationRuntime:
    """四个模型顺序进入设备；任何时刻 GPU 只驻留一个模型。"""

    def __init__(
        self,
        device_name: str,
        amp_enabled: bool = True,
        patch_batch_size: int = 1,
        tile_size: tuple[int, int] = (512, 512),
        stride: tuple[int, int] = (512, 512),
        model_specs: Sequence[ModelSpec] = MODEL_SPECS,
    ) -> None:
        self.device = resolve_device(device_name)
        self.amp_enabled = self.device.type == "cuda" and amp_enabled
        self.patch_batch_size = int(patch_batch_size)
        if self.patch_batch_size < 1:
            raise ValueError("patch_batch_size 必须大于等于 1。")
        self.tile_size = validate_image_size(tile_size)
        self.stride = validate_image_size(stride)
        if (
            self.stride[0] > self.tile_size[0]
            or self.stride[1] > self.tile_size[1]
        ):
            raise ValueError("stride 不能大于 tile_size。")

        self.model_specs = tuple(model_specs)
        self.validated = validate_model_registry(self.model_specs)
        self._lock = threading.Lock()
        self.legend = _legend_items()

    def public_info(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "amp": "开启" if self.amp_enabled else "关闭",
            "model_count": len(self.model_specs),
            "models": [spec.public_info() for spec in self.model_specs],
            "tile": f"{self.tile_size[0]} × {self.tile_size[1]}",
            "stride": f"{self.stride[0]} × {self.stride[1]}",
            "protocol": "单尺度 1.0 · 无镜像 TTA · logits 融合后 argmax",
        }

    def _release_cuda(self, model: torch.nn.Module | None) -> None:
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _predict_one(
        self,
        spec: ModelSpec,
        image_tensor: torch.Tensor,
    ) -> tuple[np.ndarray, float, float]:
        model = None
        checkpoint = None
        logits = None
        load_started = time.perf_counter()
        try:
            checkpoint = torch.load(
                spec.checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            config = checkpoint["config"]
            model_config = copy.deepcopy(config["model"])
            model_config["pretrained"] = False
            model_config["weights"] = None
            model = build_model(model_config)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            checkpoint = None
            gc.collect()

            model.to(self.device).eval()
            load_elapsed = time.perf_counter() - load_started
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            inference_started = time.perf_counter()
            logits = multi_scale_sliding_window_logits(
                model=model,
                image=image_tensor,
                scales=(1.0,),
                tile_size=self.tile_size,
                stride=self.stride,
                device=self.device,
                amp_enabled=self.amp_enabled,
                tile_batch_size=self.patch_batch_size,
                horizontal_flip=False,
                accumulate_on_device=False,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_elapsed = time.perf_counter() - inference_started
            prediction = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
            return prediction, inference_elapsed, load_elapsed
        finally:
            del logits
            del checkpoint
            self._release_cuda(model)
            del model

    def predict_all(self, source: Image.Image, filename: str) -> dict[str, Any]:
        rgb = source.convert("RGB")
        image_tensor = preprocess_image(rgb)
        model_results: list[dict[str, Any]] = []
        total_started = time.perf_counter()

        with self._lock:
            for spec in self.model_specs:
                try:
                    prediction, elapsed, load_elapsed = self._predict_one(
                        spec, image_tensor
                    )
                    if prediction.shape != (rgb.height, rgb.width):
                        raise RuntimeError(
                            f"输出尺寸 {prediction.shape} 与输入"
                            f" {(rgb.height, rgb.width)} 不一致。"
                        )
                    model_results.append(
                        build_model_visual(
                            rgb,
                            prediction,
                            spec,
                            elapsed,
                            load_elapsed,
                        )
                    )
                except torch.cuda.OutOfMemoryError:
                    self._release_cuda(None)
                    failed = spec.public_info()
                    failed.update(
                        {
                            "success": False,
                            "checkpoint": str(spec.checkpoint.resolve()),
                            "error": (
                                "GPU 显存不足。当前模型已释放；"
                                "请使用 --patch-batch-size 1 或 --device cpu 重试。"
                            ),
                        }
                    )
                    model_results.append(failed)
                except Exception as exc:
                    self._release_cuda(None)
                    failed = spec.public_info()
                    failed.update(
                        {
                            "success": False,
                            "checkpoint": str(spec.checkpoint.resolve()),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    model_results.append(failed)

        total_elapsed = time.perf_counter() - total_started
        return {
            "filename": filename,
            "width": rgb.width,
            "height": rgb.height,
            "original": image_to_data_url(rgb),
            "total_elapsed": total_elapsed,
            "total_elapsed_text": f"{total_elapsed:.2f} s",
            "models": model_results,
            "legend": self.legend,
            "successful_models": sum(
                1 for item in model_results if item["success"]
            ),
        }


def create_app(runtime: MultiModelSegmentationRuntime) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    def render(result=None, error=None, status=200):
        return (
            render_template(
                "index.html",
                result=result,
                error=error,
                runtime=runtime.public_info(),
                legend=runtime.legend,
            ),
            status,
        )

    @app.get("/")
    def index():
        return render()[0]

    @app.post("/predict")
    def predict():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return render(error="请选择一张遥感图片。", status=400)

        try:
            payload = uploaded.read()
            with Image.open(io.BytesIO(payload)) as opened:
                opened.load()
                source = opened.convert("RGB")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    "图片像素过大，请上传不超过 2500 万像素的图片。"
                )
            result = runtime.predict_all(
                source=source,
                filename=Path(uploaded.filename).name,
            )
            return render(result=result)
        except (UnidentifiedImageError, OSError):
            message = "无法读取该文件，请上传有效的 PNG、JPG、JPEG、TIF 或 TIFF 图片。"
        except ValueError as exc:
            message = str(exc)
        return render(error=message, status=400)

    @app.errorhandler(413)
    def upload_too_large(_error):
        return render(error="上传文件超过 20 MB，请压缩后重试。", status=413)

    @app.get("/health")
    def health():
        info = runtime.public_info()
        return {
            "status": "ok",
            "device": info["device"],
            "amp": info["amp"],
            "model_count": info["model_count"],
            "models": [
                {
                    "id": spec.id,
                    "name": spec.name,
                    "checkpoint": str(spec.checkpoint.resolve()),
                    "validated": spec.id in runtime.validated,
                }
                for spec in runtime.model_specs
            ],
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoveDA 四模型语义分割对比网页")
    parser.add_argument(
        "--device",
        default="cpu",
        help="推荐训练结束后使用 cuda:0；CPU 仅用于调试。",
    )
    parser.add_argument(
        "--patch-batch-size",
        type=int,
        default=1,
        help="四个模型共用的滑窗 batch，8GB GPU 建议保持 1。",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        runtime = MultiModelSegmentationRuntime(
            device_name=args.device,
            amp_enabled=not args.no_amp,
            patch_batch_size=args.patch_batch_size,
            tile_size=(args.tile_size, args.tile_size),
            stride=(args.stride, args.stride),
        )
    except ModelRegistryError as exc:
        print(exc)
        raise SystemExit(2) from exc

    print("四模型 checkpoint 检查通过：")
    for spec in runtime.model_specs:
        metadata = runtime.validated[spec.id]
        print(
            f"[OK] {spec.name}: {metadata['path']} | "
            f"{metadata['backbone']} | cat={metadata['cat_channels']}"
        )
    print(f"Web demo: http://{args.host}:{args.port}")
    print(
        f"Protocol: scale=1.0 | tile={runtime.tile_size} | "
        f"stride={runtime.stride} | flip=False | "
        f"device={runtime.device} | AMP={runtime.amp_enabled}"
    )
    app = create_app(runtime)
    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
