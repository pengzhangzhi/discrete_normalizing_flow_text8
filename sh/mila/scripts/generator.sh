#!/bin/bash

PROJ_DIR="/home/mila/a/alexander.tong/discrete_normalizing_flow_text8"
cd "$PROJ_DIR"

export UV_PROJECT_ENVIRONMENT="${SCRATCH}/uv-envs/discrete_normalizing_flow_text8"
export UV_CACHE_DIR="${SCRATCH}/uv-cache"
mkdir -p "$UV_CACHE_DIR"

export WANDB_MODE=online

echo "[uv] syncing environment at ${UV_PROJECT_ENVIRONMENT}..."
uv sync --frozen

EXP_NAME="Generator_256_4blocks"
CKPT_DIR="${SCRATCH}/OLM/${EXP_NAME}"
mkdir -p "$CKPT_DIR"

ENCODER_EXP="NFEncoder_256_4enc_4flow"
ENCODER_CKPT="${SCRATCH}/OLM/${ENCODER_EXP}/last.ckpt"

uv run python train.py +experiment=train_generator \
  ckpt_dir="$CKPT_DIR" \
  logging.run_name="$EXP_NAME" \
  data.batch_size=128 \
  data.num_workers=2 \
  model.hidden_dim=256 \
  model.n_blocks=4 \
  model.n_heads=4 \
  training.encoder_ckpt="$ENCODER_CKPT" \
  training.align_weight=0.0
