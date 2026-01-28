#!/bin/bash

PROJ_DIR="/home/mila/a/alexander.tong/discrete_normalizing_flow_text8"
cd "$PROJ_DIR"

export UV_PROJECT_ENVIRONMENT="${SCRATCH}/uv-envs/discrete_normalizing_flow_text8"
export UV_CACHE_DIR="${SCRATCH}/uv-cache"
mkdir -p "$UV_CACHE_DIR"

export WANDB_MODE=online

echo "[uv] syncing environment at ${UV_PROJECT_ENVIRONMENT}..."
uv sync --frozen

EXP_NAME="NFEncoder_256_4enc_4flow"
CKPT_DIR="${SCRATCH}/OLM/${EXP_NAME}"
mkdir -p "$CKPT_DIR"

uv run python train.py +experiment=train_encoder \
  ckpt_dir="$CKPT_DIR" \
  logging.run_name="$EXP_NAME" \
  data.batch_size=128 \
  data.num_workers=2 \
  model.hidden_dim=256 \
  model.encoder_layers=4 \
  model.encoder_heads=4 \
  model.flow_blocks=4 \
  training.mlm_enabled=false
