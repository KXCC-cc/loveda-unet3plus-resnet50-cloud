"""简单 YAML 配置加载、覆盖和运行环境记录。"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path) -> dict[str, Any]:
    """读取 YAML；可用 `_base_` 引用一个相对路径基础配置。"""
    path = path.resolve()
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise ValueError("配置文件顶层必须是字典。")
    base_name = content.pop("_base_", None)
    if base_name is not None:
        base = load_config((path.parent / base_name).resolve())
        content = _deep_merge(base, content)
    content["config_path"] = str(path)
    return content


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """应用 `a.b=value` 命令行覆盖；value 使用 YAML 语法解析。"""
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"覆盖项必须是 key=value：{item!r}")
        dotted_key, raw_value = item.split("=", 1)
        target = result
        parts = dotted_key.split(".")
        for key in parts[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[parts[-1]] = yaml.safe_load(raw_value)
    return result


def collect_environment() -> dict[str, Any]:
    """收集复现实验所需的软件、GPU 和 Git 版本。"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_device_count": torch.cuda.device_count(),
    }


def json_safe_config(config: dict) -> dict:
    """借助 JSON round-trip 确认最终配置可稳定写入 checkpoint。"""
    return json.loads(json.dumps(config, ensure_ascii=False))
