#!/bin/bash
# Super-resolve the 128 test tiles with the BEST checkpoint of one arm (1 GPU, ~15-30 min).
#   sbatch eval.slurm rq3_arm1_fakelr                 -> input = real Sentinel-2 (default)
#   sbatch eval.slurm rq3_arm1_fakelr own             -> input = the arm's own LR type
# Output: $RQ3_RUNS/<arm>/eval_<input>/<stem>.tif (float32, H,W,C, reflectance) + quick_psnr.csv
#
# HAICORE@KIT: partition normal, full A100 via gpu:full
#SBATCH --job-name=rq3_eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1      # full A100; plain gpu:1 can give a MIG slice
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
ARM="${1:?usage: sbatch eval.slurm <arm config name> [s2|own]}"
INPUT="${2:-s2}"
cd "${SLURM_SUBMIT_DIR}"
source env.sh
source "$HOME/miniforge3/bin/activate" "$RQ3_VENV"
mkdir -p logs

case "$INPUT" in
  s2)  export EVAL_INPUT_DIR="$RQ3_TEST/sr_60_200" ;;
  own) case "$ARM" in
         rq3_arm1_fakelr) export EVAL_INPUT_DIR="$RQ3_DATA/rq3_fake_lr" ;;
         rq3_arm2_s2like) export EVAL_INPUT_DIR="$RQ3_DATA/rq3_s2like" ;;
         rq3_arm3_s2)     export EVAL_INPUT_DIR="$RQ3_TEST/sr_60_200" ;;
       esac ;;
  *) echo "input must be s2 or own"; exit 1 ;;
esac

BEST="$RQ3_RUNS/$ARM/checkpoint/${ARM}_best"
test -f "${BEST}_gen.pth" || { echo "no best checkpoint at ${BEST}_gen.pth"; exit 1; }
echo "arm=$ARM  input=$EVAL_INPUT_DIR  checkpoint=$BEST"

python sr.py -c "config/${ARM}.json" -p val -gpu 0 \
  --resume_state "$BEST" \
  --results_dir "$RQ3_RUNS/$ARM/eval_${INPUT}"
