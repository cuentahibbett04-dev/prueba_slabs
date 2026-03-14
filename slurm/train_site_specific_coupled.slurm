#!/bin/bash
#SBATCH --job-name=site_coupled
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
  set +u
  # shellcheck disable=SC1090
  source "$MANIFEST_PATH"
  set -u
fi

if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
  module purge || true
  module load rocm/5.7 2>/dev/null || module load rocm 2>/dev/null || true
fi

PROJECT_DIR=${PROJECT_DIR:-$SUBMIT_DIR}
VENV_DIR=${VENV_DIR:-$PROJECT_DIR/.venv}
if [[ ! -f "$VENV_DIR/bin/activate" && -f "$(dirname "$PROJECT_DIR")/.venv/bin/activate" ]]; then
  VENV_DIR="$(dirname "$PROJECT_DIR")/.venv"
fi
DATA_DIR=${DATA_DIR:-/lustre/home/acastaneda/prueba_slabs_data}
SITE=${SITE:-lung}
DATA_ROOT=${DATA_ROOT:-$DATA_DIR/data_ct8k_geom_1Mhigh_by_site/$SITE}
OUT_DIR=${OUT_DIR:-$DATA_DIR/artifacts_${SITE}_coupled_v1}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-4}
WORKERS=${WORKERS:-8}
BASE_CHANNELS=${BASE_CHANNELS:-16}
OUTPUT_ACTIVATION=${OUTPUT_ACTIVATION:-identity}
CROP_D=${CROP_D:-64}
CROP_H=${CROP_H:-128}
CROP_W=${CROP_W:-128}
PATIENCE=${PATIENCE:-8}
MIN_DELTA=${MIN_DELTA:-1e-7}

unset TMPDIR TMP TEMP MIOPEN_CACHE_DIR MIOPEN_USER_DB_PATH MIOPEN_CUSTOM_CACHE_DIR
export TMPDIR=${TMPDIR:-/lustre/home/acastaneda/tmp/${SLURM_JOB_ID:-manual}}
export MIOPEN_CACHE_DIR=${MIOPEN_CACHE_DIR:-/lustre/home/acastaneda/miopen_cache}
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-$MIOPEN_CACHE_DIR}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR:-$MIOPEN_CACHE_DIR}
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
mkdir -p "$TMPDIR" "$MIOPEN_CACHE_DIR" "$PROJECT_DIR/logs"

cd "$PROJECT_DIR"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: virtualenv not found at $VENV_DIR"
  exit 1
fi
source "$VENV_DIR/bin/activate"
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH

echo "PROJECT_DIR=$PROJECT_DIR"
echo "VENV_DIR=$VENV_DIR"
echo "DATA_ROOT=$DATA_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "SITE=$SITE"
echo "TMPDIR=$TMPDIR"
echo "MIOPEN_CACHE_DIR=$MIOPEN_CACHE_DIR"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"

python - <<'PY'
import os
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
for k in ['TMPDIR','MIOPEN_CACHE_DIR','MIOPEN_USER_DB_PATH','MIOPEN_CUSTOM_CACHE_DIR']:
    print(k, os.environ.get(k))
PY

python scripts/train.py \
  --data-root "$DATA_ROOT" \
  --out-dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --input-norm-mode coupled_target_max \
  --crop-shape "$CROP_D" "$CROP_H" "$CROP_W" \
  --crop-focus center \
  --arch resunet3d \
  --base-channels "$BASE_CHANNELS" \
  --output-activation "$OUTPUT_ACTIVATION" \
  --patience "$PATIENCE" \
  --min-delta "$MIN_DELTA"
