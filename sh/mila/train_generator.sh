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

# Always sync to ensure new dependencies are installed (fast if up-to-date)
echo "[uv] syncing environment at ${UV_PROJECT_ENVIRONMENT}..."
uv sync --frozen

# Path to frozen Stage A encoder checkpoint
ENCODER_CKPT="checkpoints/width_256_encoder_depth_4_flow_depth_4_mlm/last.ckpt"
uv run python train_generator.py   data.batch_size=128   data.num_workers=2   model.hidden_dim=256   train.precision='bf16-mixed'   encoder.n_layers=4 encoder.n_heads=8   flow.n_blocks=4   generator.encoder_ckpt="$ENCODER_CKPT"   generator.align_weight=0   logging.save_dir=checkpoints/generator_256_8blocks   logging.run_name=generator_256_8blocks