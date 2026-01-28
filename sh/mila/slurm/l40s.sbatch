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

source "$DNF_TRAIN_SCRIPT"
