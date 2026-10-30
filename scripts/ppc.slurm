#!/bin/bash
# Posterior predictive check (RQ2), 1 GPU.
#   sbatch ppc.slurm <samples.npz> <real fingerprints csv> <out dir> [extra ppc.py args]
#   e.g. sbatch ppc.slurm $RQ3_RUNS/npe_v3/samples_sentinel.npz $RQ3_RUNS/npe_v3/sentinel.csv $RQ3_RUNS/ppc_v3
# Default 300 tiles x 20 draws (ppc.py defaults); change with --max-tiles / --draws.
#
# HAICORE@KIT: partition normal, full A100 via gpu:full
#SBATCH --job-name=rq2_ppc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
SAMPLES="${1:?usage: sbatch ppc.slurm <samples.npz> <real csv> <out dir> [extra args]}"
REAL="${2:?real fingerprints csv missing}"
OUT="${3:?out dir missing}"
shift 3
cd "${SLURM_SUBMIT_DIR}"
source env.sh
source "$HOME/miniforge3/bin/activate" "$RQ3_VENV"
mkdir -p logs "$OUT"
export OMP_NUM_THREADS=8
echo "samples=$SAMPLES real=$REAL out=$OUT extra=$* node=$(hostname) start=$(date)"
python ppc.py --samples "$SAMPLES" --real "$REAL" --device cuda --out "$OUT" "$@"
echo "end=$(date)"
