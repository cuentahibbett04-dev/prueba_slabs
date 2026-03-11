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
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
MANIFEST_PATH="${MANIFEST_PATH:-$SUBMIT_DIR/slurm/manifest.env}"
if [[ ! -f "$MANIFEST_PATH" && -f "$SCRIPT_DIR/manifest.env" ]]; then
  MANIFEST_PATH="$SCRIPT_DIR/manifest.env"
fi
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

PROJECT_DIR=${PROJECT_DIR:-$SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-${SCRATCH:-$HOME}/prueba_slabs_data}
EVENTS_HIGH=${EVENTS_HIGH:-1000000}
DATA_ROOT=${DATA_ROOT:-$DATA_DIR/data_real_photon_${EVENTS_HIGH}_multinoise}
ARTIFACTS_DIR=${ARTIFACTS_DIR:-$DATA_DIR/artifacts_real_photon_${EVENTS_HIGH}_multinoise_e20_mi210}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-20}
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
  --data-root "$DATA_ROOT" \
  --out-dir "$ARTIFACTS_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  --workers 8 \
  --amp \
  --amp-dtype bf16 \
  --loss-alpha 3.0 \
  --loss-min-weight 0.05 \
  --background-threshold 0.02 \
  --background-lambda 0.2 \
  --patience 5 \
  --min-delta 0.0 \
  --save-every 5 \
  --save-epochs 1 \
  --input-norm-mode per_channel_max \
  --input-dose-scale 1.0 \
  --output-activation relu
