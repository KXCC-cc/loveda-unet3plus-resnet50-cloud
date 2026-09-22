#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/unet3/bin/activate"

checkpoint="${1:?usage: $0 RUN_DIR/best_model.pth [extra args]}"
shift
python -m tools.eval_checkpoint "$checkpoint" \
  --scales 0.5 0.75 1.0 1.25 1.5 1.75 "$@"
