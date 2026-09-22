#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ALLOC_CONF=expandable_segments:True
python train.py --config configs/cloud/resnet50_16gb.yaml "$@"
