#!/bin/bash
#SBATCH -J DNF
#SBATCH -o watch_folder/l40s/%x_%j.out
#SBATCH -e watch_folder/l40s/%x_%j.err
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=128G
#SBATCH -t 7-00:00:00
#SBATCH --open-mode=append


PROJ_DIR="/home/mila/a/alexander.tong/discrete_normalizing_flow_text8"
cd "$PROJ_DIR"


export UV_PROJECT_ENVIRONMENT="${SCRATCH}/uv-envs/discrete_normalizing_flow_text8"
export UV_CACHE_DIR="${SCRATCH}/uv-cache"
mkdir -p "$UV_CACHE_DIR"

# Always sync to ensure new dependencies are installed (fast if up-to-date)
echo "[uv] syncing environment at ${UV_PROJECT_ENVIRONMENT}..."
uv sync --frozen

uv run python train.py \
  data.batch_size=128 \
  data.num_workers=2 \
  model.hidden_dim=768 \
  train.precision='bf16-mixed' \
  encoder.n_layers=12 encoder.n_heads=16 \
  flow.n_blocks=12 \
  mlm.enabled=false \
  logging.save_dir=checkpoints/width_768_encoder_depth_12_flow_depth_12 \
  logging.run_name=width_768_encoder_depth_12_flow_depth_12
