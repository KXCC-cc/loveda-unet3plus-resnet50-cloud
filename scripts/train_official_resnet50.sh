#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/unet3/bin/activate"

# 官方 batch=16 配置通常超出 8GB 显存；本脚本用于足够显存的严格对照环境。
python train.py --config configs/loveda_official_resnet50.yaml "$@"
