#!/bin/bash

PROJ_DIR="/home/mila/a/alexander.tong/discrete_normalizing_flow_text8"
cd "$PROJ_DIR"

export UV_PROJECT_ENVIRONMENT="${SCRATCH}/uv-envs/discrete_normalizing_flow_text8"
export UV_CACHE_DIR="${SCRATCH}/uv-cache"
mkdir -p "$UV_CACHE_DIR"

export WANDB_MODE=online

echo "[uv] syncing environment at ${UV_PROJECT_ENVIRONMENT}..."
uv sync --frozen

EXP_NAME="NFEncoder_512_8enc_8flow_ae_w_1"
CKPT_DIR="${SCRATCH}/OLM/${EXP_NAME}"
mkdir -p "$CKPT_DIR"

uv run python train.py +experiment=train_encoder \
  ckpt_dir="$CKPT_DIR" \
  logging.run_name="$EXP_NAME" \
  data.batch_size=128 \
  data.num_workers=2 \
  model.hidden_dim=512 \
  model.encoder_layers=8 \
  model.encoder_heads=8 \
  model.flow_blocks=8 \
  training.mlm_enabled=false \
  training.ae_enabled=true \
  training.ae_loss_weight=1.0
