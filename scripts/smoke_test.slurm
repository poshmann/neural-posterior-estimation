#!/bin/bash
# 10-minute end-to-end test before the real runs (debug mode: 12 iterations, 10 diffusion
# steps, tiny data). Runs train -> resume -> eval for arm 1 and arm 3 on 1 GPU.
#   sbatch smoke_test.slurm
# Then read logs/rq3_smoke_<jobid>.out: it must end with "SMOKE TEST PASSED".
#
# HAICORE@KIT: partition normal, full A100 via gpu:full
#SBATCH --job-name=rq3_smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1      # full A100; plain gpu:1 can give a MIG slice
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source env.sh
source "$HOME/miniforge3/bin/activate" "$RQ3_VENV"
export RQ3_RUNS="$RQ3_RUNS/_smoke"          # keep test runs away from the real ones
rm -rf "$RQ3_RUNS"

for ARM in rq3_arm1_fakelr rq3_arm3_s2; do
  echo "=== $ARM: train (12 iterations) ==="
  python sr.py -c "config/${ARM}.json" -p train -gpu 0 -debug
  echo "=== $ARM: second call must RESUME and stop at once ==="
  python sr.py -c "config/${ARM}.json" -p train -gpu 0 -debug 2>&1 | tee /tmp/resume_$$.txt
  grep -q "Resuming training" /tmp/resume_$$.txt || { echo "RESUME FAILED"; exit 1; }
  CK="$RQ3_RUNS/${ARM}_debug/checkpoint"
  LAST=$(ls "$CK"/*_I*_gen.pth | sort -V | tail -1); LAST="${LAST%_gen.pth}"
  echo "=== $ARM: eval on real Sentinel-2 test input with $LAST ==="
  python sr.py -c "config/${ARM}.json" -p val -gpu 0 -debug \
    --resume_state "$LAST" --results_dir "$RQ3_RUNS/${ARM}_debug/eval_s2"
  n=$(ls "$RQ3_RUNS/${ARM}_debug/eval_s2"/*.tif | wc -l)
  echo "$ARM: $n SR tiles written"; [ "$n" -eq 128 ] || { echo "EVAL WROTE $n TILES"; exit 1; }
done
python - <<'EOF'
import glob, os, tifffile, numpy as np
f = sorted(glob.glob(os.path.join(os.environ["RQ3_RUNS"], "rq3_arm3_s2_debug", "eval_s2", "*.tif")))[0]
a = tifffile.imread(f); print("example output", os.path.basename(f), a.shape, a.dtype, float(a.min()), float(a.max()))
assert a.shape == (192, 192, 4) and np.isfinite(a).all()
EOF
echo "SMOKE TEST PASSED"
