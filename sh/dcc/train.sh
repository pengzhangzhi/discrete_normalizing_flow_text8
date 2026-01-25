#!/bin/bash
#SBATCH --job-name=discrete-normalizing-flow
#SBATCH --partition=cellbio-dgx-low
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --output=watch_folder/%x-%j.out
#SBATCH --error=watch_folder/%x-%j.err

# Exit on error
set -e

# Project directory
PROJECT_DIR=/hpc/group/dallagolab/fred/projects/github_projs/discrete_normalizing_flow
cd $PROJECT_DIR

# Create output directory
mkdir -p watch_folder

# Run training (uv auto-detects .venv in project dir)
uv run python train.py data.batch_size=256
