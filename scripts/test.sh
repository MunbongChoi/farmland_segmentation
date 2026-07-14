#!/usr/bin/env bash
set -euo pipefail
python -m src.test --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt "$@"

