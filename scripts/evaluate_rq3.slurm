#!/bin/bash
# Score all arms on the 128 test tiles (after eval.slurm has written the SR tiles). 1 GPU, ~10 min.
#   sbatch evaluate_rq3.slurm
# Arms whose eval folder is missing are skipped. Output: $RQ3_RUNS/rq3_eval/
#
# HAICORE@KIT: partition normal, full A100 via gpu:full
#SBATCH --job-name=rq3_score
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source env.sh
source "$HOME/miniforge3/bin/activate" "$RQ3_VENV"
ARMS=(--arm "input_s2=$RQ3_TEST/sr_60_200")
for A in rq3_arm1_fakelr rq3_arm2_s2like rq3_arm3_s2; do
  D="$RQ3_RUNS/$A/eval_s2"
  [ -d "$D" ] && ARMS+=(--arm "${A#rq3_}=$D") || echo "skip $A (no $D)"
done
REF=arm3_s2; [ -d "$RQ3_RUNS/rq3_arm3_s2/eval_s2" ] || REF=input_s2
python evaluate_rq3.py --hr "$RQ3_TEST/hr_200" "${ARMS[@]}" --reference "$REF" --device cuda \
  --out "$RQ3_RUNS/rq3_eval"
