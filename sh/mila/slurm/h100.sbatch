#!/bin/bash
#SBATCH -J DNF
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --mem-per-gpu=128G
#SBATCH -t 03:00:00
#SBATCH -c 64
#SBATCH --partition=short-unkillable
#SBATCH --array=1-99%1
#SBATCH --open-mode=append

source "$DNF_TRAIN_SCRIPT"
