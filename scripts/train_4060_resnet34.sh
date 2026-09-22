#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/unet3/bin/activate"

# 第一轮推荐：512、physical batch 1、accum 4、AMP、单尺度训练。
python train.py --config configs/loveda_4060_resnet34.yaml "$@"
