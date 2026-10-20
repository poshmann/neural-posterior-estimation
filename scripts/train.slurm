#!/bin/bash
# Train one RQ3 arm on one node with 4 GPUs. Resumes automatically from the newest checkpoint,
# so the same script can simply be submitted again if a job ever stops early.
#   sbatch train.slurm rq3_arm1_fakelr        (or rq3_arm3_s2)
#
# HAICORE@KIT: partition 'normal', max 3 days, 1 node. GPU type: see --gres line.
#SBATCH --job-name=rq3_train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:full:4      # full A100-40GB; plain gpu:4 can give MIG slices
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
ARM="${1:?usage: sbatch train.slurm <config name, e.g. rq3_arm1_fakelr>}"
cd "${SLURM_SUBMIT_DIR}"            # = the folder you ran sbatch from
source env.sh
source "$HOME/miniforge3/bin/activate" "$RQ3_VENV"
mkdir -p logs "$RQ3_RUNS"

echo "arm=$ARM  node=$(hostname)  start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
export OMP_NUM_THREADS=8

torchrun --standalone --nproc_per_node=4 sr.py -c "config/${ARM}.json" -p train
echo "end=$(date)"
