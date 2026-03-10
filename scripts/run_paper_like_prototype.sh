#!/usr/bin/env bash
set -euo pipefail

# Paper-like prototype pipeline:
# 1) Generate multinoise MC pairs (2k/5k/10k/20k -> 100k)
# 2) Build NPZ dataset with optional angle-holdout split
# 3) Train baseline denoiser model
#
# Example:
#   bash scripts/run_paper_like_prototype.sh \
#     --campaign-config mc/campaign.opengate.photon.multinoise.combat2k.json \
#     --dataset-root data_real_photon_1000_multinoise_combat2k \
#     --artifacts artifacts_combat2k_resunet \
#     --batch-size 16 --workers 8 --epochs 20

CAMPAIGN_CONFIG="mc/campaign.opengate.photon.multinoise.combat2k.json"
DATASET_ROOT="data_real_photon_1000_multinoise_combat2k"
ARTIFACTS_DIR="artifacts_combat2k_resunet"
ARCH="resunet3d"
BATCH_SIZE=16
WORKERS=8
EPOCHS=20
DEVICE="cuda"
AMP_DTYPE="bf16"
BASE_CHANNELS=16

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-config) CAMPAIGN_CONFIG="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --artifacts) ARTIFACTS_DIR="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --amp-dtype) AMP_DTYPE="$2"; shift 2 ;;
    --base-channels) BASE_CHANNELS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

export PYTHONPATH="${PYTHONPATH:-$PWD/src}"

# 200x200x200 mm phantom at 2 mm voxel size.
export SLAB_NX="${SLAB_NX:-100}"
export SLAB_NY="${SLAB_NY:-100}"
export SLAB_NZ="${SLAB_NZ:-100}"
export SLAB_SX="${SLAB_SX:-2.0}"
export SLAB_SY="${SLAB_SY:-2.0}"
export SLAB_SZ="${SLAB_SZ:-2.0}"

echo "[1/3] Generating multinoise MC pairs"
python -u scripts/run_multinoise_campaign.py --config "$CAMPAIGN_CONFIG"

PAIR_ROOT=$(python - <<'PY' "$CAMPAIGN_CONFIG"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = json.load(f)
print(cfg['output_root'].rstrip('/') + '/pairs')
PY
)

echo "[2/3] Building NPZ dataset with angle holdout split"
python -u scripts/build_dataset_from_mc.py \
  --mc-root "$PAIR_ROOT" \
  --out-root "$DATASET_ROOT" \
  --train-ratio 0.75 \
  --val-ratio 0.125 \
  --holdout-val-by-angle \
  --val-angle-ratio 0.2 \
  --rescale-low-by-history-ratio

echo "[3/3] Training model"
python -u scripts/train.py \
  --arch "$ARCH" \
  --data-root "$DATASET_ROOT" \
  --out-dir "$ARTIFACTS_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --amp \
  --amp-dtype "$AMP_DTYPE" \
  --loss-alpha 3.0 \
  --loss-min-weight 0.05 \
  --background-threshold 0.02 \
  --background-lambda 0.2 \
  --patience 5 \
  --save-every 5 \
  --save-epochs 1 \
  --input-norm-mode per_channel_max \
  --output-activation relu \
  --base-channels "$BASE_CHANNELS"

echo "Pipeline complete"
echo "Dataset:   $DATASET_ROOT"
echo "Artifacts: $ARTIFACTS_DIR"
