#!/bin/bash
#SBATCH --job-name=colorectal_to50
#SBATCH --output=/lustre/home/acastaneda/Fernando/prueba_slabs_main_e86708d/run_8ct_10k/logs/%x_%j.out
#SBATCH --error=/lustre/home/acastaneda/Fernando/prueba_slabs_main_e86708d/run_8ct_10k/logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# ---- User-configurable paths ----
PROJECT_DIR=${PROJECT_DIR:-/lustre/home/acastaneda/Fernando/prueba_slabs_main_e86708d/run_8ct_10k}
VENV_DIR=${VENV_DIR:-/lustre/home/acastaneda/Fernando/prueba_slabs_main_e86708d/.venv}
DATA_ROOT=${DATA_ROOT:-/lustre/home/acastaneda/prueba_slabs_data/data_ct8k_geom_1Mhigh_by_site/colorectal}
BASE_RUN=${BASE_RUN:-/lustre/home/acastaneda/prueba_slabs_data/artifacts_colorectal_final_72x160x160_softplus_20260314_174339}
TARGET_EPOCH=${TARGET_EPOCH:-50}
RESUME_CKPT=${RESUME_CKPT:-}

# ---- Train hyperparameters ----
BATCH_SIZE=${BATCH_SIZE:-4}
WORKERS=${WORKERS:-8}
AMP_DTYPE=${AMP_DTYPE:-bf16}
DEVICE=${DEVICE:-cuda}
PATIENCE=${PATIENCE:-10}
MIN_DELTA=${MIN_DELTA:-0}
BASE_CHANNELS=${BASE_CHANNELS:-12}
OUTPUT_ACTIVATION=${OUTPUT_ACTIVATION:-softplus}
SAVE_EVERY=${SAVE_EVERY:-1}

# Optional module init
if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
  module purge || true
  module load rocm/5.7 2>/dev/null || module load rocm 2>/dev/null || true
fi

# ROCm runtime dirs
unset TMPDIR TMP TEMP MIOPEN_CACHE_DIR MIOPEN_USER_DB_PATH MIOPEN_CUSTOM_CACHE_DIR
export TMPDIR=${TMPDIR:-/lustre/home/acastaneda/tmp/${SLURM_JOB_ID:-manual}}
export MIOPEN_CACHE_DIR=${MIOPEN_CACHE_DIR:-/lustre/home/acastaneda/miopen_cache}
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-$MIOPEN_CACHE_DIR}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR:-$MIOPEN_CACHE_DIR}
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
mkdir -p "$TMPDIR" "$MIOPEN_CACHE_DIR" "$PROJECT_DIR/logs"

TRACE_LOG="$PROJECT_DIR/logs/${SLURM_JOB_NAME:-colorectal_to50}_${SLURM_JOB_ID:-manual}.trace.log"
exec > >(tee -a "$TRACE_LOG") 2>&1
trap 'echo "ERROR: command failed at line $LINENO"' ERR

cd "$PROJECT_DIR"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: virtualenv not found at $VENV_DIR"
  exit 1
fi
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

# Keep original working model untouched: write into a new run folder
RUN_STAMP="continue_to50_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
NEW_RUN="${BASE_RUN}_${RUN_STAMP}"
mkdir -p "$NEW_RUN"

# Seed artifacts for traceability
cp "$BASE_RUN/best_model.pt" "$NEW_RUN/best_model_seed.pt"
cp "$BASE_RUN/run_config.json" "$NEW_RUN/run_config_seed.json" 2>/dev/null || true
cp "$BASE_RUN/history.json" "$NEW_RUN/history_seed.json" 2>/dev/null || true

# Pick resume checkpoint
if [[ -n "$RESUME_CKPT" ]]; then
  CKPT="$RESUME_CKPT"
else
  shopt -s nullglob
  ckpt_candidates=("$BASE_RUN"/checkpoint_epoch_*.pt)
  shopt -u nullglob
  if [[ ${#ckpt_candidates[@]} -gt 0 ]]; then
    CKPT="$(printf '%s\n' "${ckpt_candidates[@]}" | sort -V | tail -n 1)"
  else
    CKPT="$BASE_RUN/best_model.pt"
  fi
fi
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: resume checkpoint not found: $CKPT"
  exit 1
fi

CURRENT_EPOCH="$(python3 - <<PY
import torch
ckpt = torch.load("$CKPT", map_location="cpu")
print(int(ckpt.get("epoch", 0)))
PY
)"
if [[ -z "$CURRENT_EPOCH" ]]; then
  echo "ERROR: CURRENT_EPOCH is empty (CKPT=$CKPT)"
  exit 1
fi
ADD_EPOCHS=$((TARGET_EPOCH - CURRENT_EPOCH))

echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATA_ROOT=$DATA_ROOT"
echo "BASE_RUN=$BASE_RUN"
echo "NEW_RUN=$NEW_RUN"
echo "CKPT=$CKPT"
echo "CURRENT_EPOCH=$CURRENT_EPOCH"
echo "TARGET_EPOCH=$TARGET_EPOCH"
echo "ADD_EPOCHS=$ADD_EPOCHS"
echo "DEVICE=$DEVICE"
echo "TMPDIR=$TMPDIR"
echo "MIOPEN_CACHE_DIR=$MIOPEN_CACHE_DIR"

python3 - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
PY

if [[ "$ADD_EPOCHS" -le 0 ]]; then
  echo "Already at epoch >= $TARGET_EPOCH. Exiting without training."
  exit 0
fi

TRAIN_ARGS=()
if [[ "$DEVICE" == "cuda" ]]; then
  TRAIN_ARGS+=(--amp --amp-dtype "$AMP_DTYPE")
fi

python3 scripts/train.py \
  --data-root "$DATA_ROOT" \
  --out-dir "$NEW_RUN" \
  --epochs "$ADD_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --device "$DEVICE" \
  --input-norm-mode coupled_target_max \
  --input-dose-scale 1.0 \
  --crop-shape 72 160 160 \
  --crop-focus maxdose \
  --arch resunet3d \
  --base-channels "$BASE_CHANNELS" \
  --output-activation "$OUTPUT_ACTIVATION" \
  --patience "$PATIENCE" \
  --min-delta "$MIN_DELTA" \
  --save-every "$SAVE_EVERY" \
  --resume-checkpoint "$CKPT" \
  --start-epoch "$CURRENT_EPOCH" \
  --resume-history \
  "${TRAIN_ARGS[@]}"

echo "DONE. New run dir: $NEW_RUN"