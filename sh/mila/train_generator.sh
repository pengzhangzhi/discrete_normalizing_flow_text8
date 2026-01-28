#!/bin/bash
#SBATCH -J DNF_gen
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

# ---- go to project dir ----
PROJ_DIR="/home/mila/a/alexander.tong/discrete_normalizing_flow_text8"
cd "$PROJ_DIR"

export UV_PROJECT_ENVIRONMENT="${SCRATCH}/uv-envs/discrete_normalizing_flow_text8"
export UV_CACHE_DIR="${SCRATCH}/uv-cache"
mkdir -p "$UV_CACHE_DIR"

# Always sync to ensure new dependencies are installed
echo "[uv] syncing environment at ${UV_PROJECT_ENVIRONMENT}..."
uv sync --frozen


EXP_NAME="Generator_256_8blocks"
CKPT_DIR="${SCRATCH}/OLM/${EXP_NAME}"
mkdir -p "$CKPT_DIR"


ENCODER_EXP="NFEncoder_256_4enc_4flow"
ENCODER_CKPT="${SCRATCH}/OLM/${ENCODER_EXP}/last.ckpt"

# Train Generator (Stage B)
uv run python train.py +experiment=train_generator \
  ckpt_dir="$CKPT_DIR" \
  logging.run_name="$EXP_NAME" \
  data.batch_size=128 \
  data.num_workers=2 \
  model.hidden_dim=256 \
  model.n_blocks=8 \
  model.n_heads=8 \
  training.encoder_ckpt="$ENCODER_CKPT" \
  training.align_weight=0.0
