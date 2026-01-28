#!/bin/bash
# Usage: ./train <gpu_type> <training_script>
# Example: ./train l40s sh/mila/scripts/encoder.sh

set -euo pipefail

GPU="${1:-}"
SCRIPT="${2:-}"

if [[ -z "$GPU" || -z "$SCRIPT" ]]; then
    echo "Usage: $0 <gpu_type> <training_script>"
    echo "  gpu_type: h100, l40s"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJ_DIR="/home/mila/a/alexander.tong/discrete_normalizing_flow_text8"
cd "$PROJ_DIR"

# Resolve training script path
[[ "$SCRIPT" = /* ]] || SCRIPT="$PROJ_DIR/$SCRIPT"

export DNF_TRAIN_SCRIPT="$SCRIPT"
sbatch "$SCRIPT_DIR/slurm/${GPU}.sbatch"
