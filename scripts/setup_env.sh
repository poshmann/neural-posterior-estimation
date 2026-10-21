#!/bin/bash
# One-time: Python 3.12 environment on a HAICORE LOGIN node (needs internet).
# The system python is 3.9 and there is no python module, so we install Miniforge in $HOME.
#   bash setup_env.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [ ! -x "$HOME/miniforge3/bin/conda" ]; then
  curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
fi
[ -x "$RQ3_VENV/bin/python" ] || "$HOME/miniforge3/bin/conda" create -y -n wavecm python=3.12
PY="$RQ3_VENV/bin/python"
"$PY" -m pip install --upgrade pip wheel
# torch wheels for CUDA 12.8 come from the PyTorch index; the rest from PyPI
"$PY" -m pip install -r "$RQ3_CODE/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu128
# NPE: sbi (its dependency nflows only builds without build isolation); keep the cu128 torch
"$PY" -m pip install --no-build-isolation nflows
"$PY" -m pip install sbi "torch==2.9.1" --extra-index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip freeze > "$RQ3_CODE/requirements.lock.txt"     # exact versions, for the thesis
"$PY" - <<'PYEOF'
import torch, pytorch_wavelets, tifffile, cv2, skimage, lpips, clip
print("torch", torch.__version__, "| CUDA build", torch.version.cuda)
PYEOF
echo "Environment ready: $RQ3_VENV"
