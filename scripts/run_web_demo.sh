#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/unet3/bin/activate"

checkpoint="runs/resnet34_pretrained_512_randomcrop/best_model.pth"
device="cpu"
if [[ $# -ge 1 ]]; then checkpoint="$1"; fi
if [[ $# -ge 2 ]]; then device="$2"; fi

python -m web_demo.app --checkpoint "$checkpoint" --device "$device"
