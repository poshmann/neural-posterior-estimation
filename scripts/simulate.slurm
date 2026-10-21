#!/bin/bash
# Simulation records for the NPE, as a job array (one GPU per piece).
#   sbatch --array=0-3 simulate.slurm train 10      # 5,000 train tiles x 10 draws = 50,000 records
#   sbatch --array=0 simulate.slurm timing 5        # timing test: 20 tiles x 5 draws
#   sbatch --array=0-1 simulate.slurm val 4         # held-out simulations (500 val tiles)
#   sbatch --array=0-1 simulate.slurm baseline 1    # theta=0 context fingerprint for ALL train/val/test tiles
#   sbatch --array=0-1 simulate.slurm real 1        # REAL Sentinel-2 fingerprint for ALL train/val/test tiles
# Output: $RQ3_RUNS/sims/<split>_shardXXofYY.csv  (resumes if resubmitted)
#
# HAICORE@KIT: partition normal, full A100 via gpu:full
#SBATCH --job-name=rq3_sim
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
SPLIT="${1:?usage: sbatch --array=0-N simulate.slurm <train|val|test|timing> <draws per tile>}"
DRAWS="${2:-10}"
cd "${SLURM_SUBMIT_DIR}"
source env.sh
source "$HOME/miniforge3/bin/activate" "$RQ3_VENV"
NSHARDS=$(( ${SLURM_ARRAY_TASK_MAX:-0} + 1 ))
EXTRA=""
if [ "$SPLIT" = "timing" ]; then SPLIT=train; EXTRA="--n-tiles 20"; OUTNAME=timing; else OUTNAME=$SPLIT; fi
if [ "$SPLIT" = "baseline" ] || [ "$SPLIT" = "real" ]; then
  for SP in train val test; do
    python simulate.py --split "$SP" --$SPLIT --shard "${SLURM_ARRAY_TASK_ID}" --nshards "$NSHARDS" \
      --device cuda --out "$RQ3_RUNS/sims/${SPLIT}_${SP}.csv"
  done
  exit 0
fi
export OMP_NUM_THREADS=8
echo "split=$SPLIT draws=$DRAWS shard=${SLURM_ARRAY_TASK_ID}/$NSHARDS node=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
python simulate.py --split "$SPLIT" --draws "$DRAWS" $EXTRA \
  --shard "${SLURM_ARRAY_TASK_ID}" --nshards "$NSHARDS" --device cuda \
  --out "$RQ3_RUNS/sims/${OUTNAME}.csv"
