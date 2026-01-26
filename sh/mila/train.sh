#!/bin/bash
#SBATCH -J DNF_
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --mem-per-gpu=128G
#SBATCH -t 03:00:00
#SBATCH -c 64
#SBATCH --open-mode=append
#SBATCH --partition=short-unkillable
#SBATCH --array=1-99%1

set -euo pipefail

# ---- go to project dir (explicit, since you gave it) ----
PROJ_DIR="/home/mila/a/alexander.tong/discrete_normalizing_flow_text8"
cd "$PROJ_DIR"


export UV_PROJECT_ENVIRONMENT="${SCRATCH}/uv-envs/discrete_normalizing_flow_text8"
export UV_CACHE_DIR="${SCRATCH}/uv-cache"
mkdir -p "$UV_CACHE_DIR"


# If the venv exists, don't touch it. If not, recreate exactly from lock.
if [[ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
  echo "[uv] venv not found at ${UV_PROJECT_ENVIRONMENT}, creating from uv.lock..."
  uv sync --frozen
else
  echo "[uv] using existing venv: ${UV_PROJECT_ENVIRONMENT}"
fi

uv run python train.py \
  data.batch_size=128 \
  model.hidden_dim=512 \
  encoder.n_layers=6 encoder.n_heads=8 \
  flow.n_blocks=6