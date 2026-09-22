#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/unet3/bin/activate"

# RTX 4060 8GB：512、physical batch 1、accum 4、AMP。
# 配置与 ResNet34 对照实验一致，仅将 ImageNet 编码器换为 ResNet50。
python train.py --config configs/loveda_4060_resnet50.yaml "$@"
