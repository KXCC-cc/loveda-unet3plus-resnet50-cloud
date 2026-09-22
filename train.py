"""新配置化训练入口；旧命令行训练器保存在 train_legacy.py。"""

from __future__ import annotations

import argparse
from pathlib import Path

from engine.trainer import run_training
from utils.config import apply_overrides, collect_environment, json_safe_config, load_config


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoveDA ResNet U-Net 3+ training")
    parser.add_argument("--config", type=Path, required=True, help="YAML 实验配置")
    parser.add_argument("--resume", type=Path, default=None, help="恢复 last_checkpoint.pth")
    parser.add_argument("--output-dir", type=Path, default=None, help="覆盖本次 run 目录")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖配置，可重复，如 --set data.loader.batch_size=2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    if args.output_dir is not None:
        config["run"]["output_dir"] = str(args.output_dir)
    for section, key in (("data", "root"), ("loss", "class_stats_path")):
        value = config.get(section, {}).get(key)
        if value is not None and not Path(value).is_absolute():
            config[section][key] = str((PROJECT_ROOT / value).resolve())
    output = Path(config["run"]["output_dir"])
    if not output.is_absolute():
        config["run"]["output_dir"] = str((PROJECT_ROOT / output).resolve())
    config["environment"] = collect_environment()
    config = json_safe_config(config)
    run_training(config, args.resume)


if __name__ == "__main__":
    main()
