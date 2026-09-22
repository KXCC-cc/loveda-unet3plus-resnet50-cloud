#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install --upgrade pip
python -m pip install -r requirements-cloud.txt
bash scripts/download_loveda_release.sh

python - <<'PY'
import torch
import torchvision
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("未检测到CUDA GPU，请在云平台中启用GPU运行时。")
print("GPU:", torch.cuda.get_device_name(0))
PY
