#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/unet3/bin/activate"

checkpoint="${1:?usage: $0 RUN_DIR/best_model.pth [extra args]}"
shift
python -m tools.eval_checkpoint "$checkpoint" --scales 1.0 "$@"
