#!/bin/bash
# Example Usage:
# ./mila_train.sh l40s sh/mila/scripts/encoder.sh
# ./mila_train.sh h100 sh/mila/scripts/generator.sh

set -euo pipefail

GPU="${1:-}"
SCRIPT="${2:-}"

if [[ -z "$GPU" || -z "$SCRIPT" ]]; then
    echo "Usage: $0 <gpu_type> <training_script>"
    echo "  gpu_type: h100, l40s"
    exit 1
fi

PROJ_DIR="/home/mila/a/alexander.tong/discrete_normalizing_flow_text8"
cd "$PROJ_DIR"

[[ "$SCRIPT" = /* ]] || SCRIPT="$PROJ_DIR/$SCRIPT"

export DNF_TRAIN_SCRIPT="$SCRIPT"
sbatch "$PROJ_DIR/sh/mila/slurm/${GPU}.sbatch"
