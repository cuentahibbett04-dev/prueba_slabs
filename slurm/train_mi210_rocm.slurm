#!/bin/bash
#SBATCH --job-name=deepmc_mi210_e20
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST_PATH="${MANIFEST_PATH:-$SCRIPT_DIR/manifest.env}"
if [[ -f "$MANIFEST_PATH" ]]; then
  # shellcheck disable=SC1090
  set +u
  source "$MANIFEST_PATH"
  set -u
fi

# Optional module init (won't fail if modules are unavailable)
if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
  module purge || true
  module load rocm/5.7 || module load rocm || true
fi

PROJECT_DIR=${PROJECT_DIR:-$REPO_DIR}
DATA_DIR=${DATA_DIR:-${SCRATCH:-$HOME}/prueba_slabs_data}
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$TMPDIR" 2>/dev/null || true

cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH=$PROJECT_DIR/src
mkdir -p logs

python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
PY

python scripts/train.py \
  --data-root "$DATA_DIR/data_real_photon_1000_multinoise" \
  --out-dir "$DATA_DIR/artifacts_real_photon_1000_multinoise_e20_mi210" \
  --epochs 20 \
  --batch-size 2 \
  --device cuda \
  --workers 8 \
  --amp \
  --loss-alpha 3.0 \
  --patience 5 \
  --min-delta 0.0 \
  --save-every 5 \
  --input-norm-mode per_channel_max \
  --input-dose-scale 1.0 \
  --output-activation softplus
