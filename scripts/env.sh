# Source this before every RQ3 command on HAICORE:   source env.sh
# Paths filled in for HAICORE@KIT, workspace 'srdata' (user hgf_sma8852).

# folder you uploaded, containing hr_200/, sr_60_200/ (and later rq3_fake_lr/)
export RQ3_DATA="${RQ3_DATA:-/hkfs/work/workspace_haic/scratch/hgf_sma8852-srdata}"
# folder containing the 128 test tiles: hr_200/, sr_60_200/
export RQ3_TEST="${RQ3_TEST:-/hkfs/work/workspace_haic/scratch/hgf_sma8852-srdata/Test}"
# where runs are written (checkpoints ~ several GB per arm) -- a workspace, not $HOME
export RQ3_RUNS="${RQ3_RUNS:-/hkfs/work/workspace_haic/scratch/hgf_sma8852-srdata/rq3_runs}"
# python environment made by setup_env.sh
export RQ3_VENV="${RQ3_VENV:-$HOME/miniforge3/envs/wavecm}"

# fixed, relative to this folder
export RQ3_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RQ3_SPLITS="$RQ3_CODE/splits"
# input used at test time; default = real Sentinel-2 (same input for all three models)
export EVAL_INPUT_DIR="${EVAL_INPUT_DIR:-$RQ3_TEST/sr_60_200}"
