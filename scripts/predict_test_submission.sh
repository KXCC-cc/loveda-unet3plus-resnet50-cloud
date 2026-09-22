#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/unet3/bin/activate"

python -m tools.predict_test_submission   --checkpoint runs/resnet34_pretrained_512_randomcrop/best_model.pth   --data-root dataset   --output-dir submissions/loveda_test_single_scale   --zip-path submissions/loveda_test_single_scale.zip   --device cuda:0
