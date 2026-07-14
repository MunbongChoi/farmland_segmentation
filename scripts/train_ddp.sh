#!/usr/bin/env bash
set -euo pipefail
: "${NPROC_PER_NODE:=2}"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m src.train --config configs/default.yaml "$@"

