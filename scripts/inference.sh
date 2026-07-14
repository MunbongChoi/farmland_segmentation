#!/usr/bin/env bash
set -euo pipefail
python -m src.infer --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt "$@"

