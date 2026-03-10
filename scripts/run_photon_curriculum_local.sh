#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

source .venv/bin/activate
export PYTHONPATH="$repo_root/src"

# Usage:
#   bash scripts/run_photon_curriculum_local.sh [data_root] [out_root] [device]
# Example:
#   bash scripts/run_photon_curriculum_local.sh data_real_photon_1000_multinoise artifacts_photon_curriculum cuda

data_root="${1:-data_real_photon_1000_multinoise}"
out_root="${2:-artifacts_photon_curriculum}"
device="${3:-cuda}"

if [[ ! -d "$data_root/train" ]]; then
  echo "ERROR: dataset split not found at $data_root/train" >&2
  exit 1
fi

mkdir -p "$out_root"

echo "[Curriculum] Phase 1/4: only 2k"
python scripts/train.py \
  --data-root "$data_root" \
  --out-dir "$out_root/phase1_2k" \
  --epochs 6 \
  --batch-size 2 \
  --device "$device" \
  --workers 4 \
  --amp \
  --loss-alpha 3.0 \
  --patience 3 \
  --min-delta 0.0 \
  --save-every 3 \
  --input-norm-mode per_channel_max \
  --output-activation softplus \
  --low-events-allow 2000

echo "[Curriculum] Phase 2/4: 2k + 5k"
python scripts/train.py \
  --data-root "$data_root" \
  --out-dir "$out_root/phase2_2k5k" \
  --epochs 6 \
  --batch-size 2 \
  --device "$device" \
  --workers 4 \
  --amp \
  --loss-alpha 3.0 \
  --patience 3 \
  --min-delta 0.0 \
  --save-every 3 \
  --input-norm-mode per_channel_max \
  --output-activation softplus \
  --low-events-allow 2000 5000 \
  --resume-checkpoint "$out_root/phase1_2k/best_model.pt"

echo "[Curriculum] Phase 3/4: 2k + 5k + 10k"
python scripts/train.py \
  --data-root "$data_root" \
  --out-dir "$out_root/phase3_2k5k10k" \
  --epochs 8 \
  --batch-size 2 \
  --device "$device" \
  --workers 4 \
  --amp \
  --loss-alpha 3.0 \
  --patience 4 \
  --min-delta 0.0 \
  --save-every 4 \
  --input-norm-mode per_channel_max \
  --output-activation softplus \
  --low-events-allow 2000 5000 10000 \
  --resume-checkpoint "$out_root/phase2_2k5k/best_model.pt"

echo "[Curriculum] Phase 4/4: full (2k + 5k + 10k + 20k)"
python scripts/train.py \
  --data-root "$data_root" \
  --out-dir "$out_root/phase4_full" \
  --epochs 10 \
  --batch-size 2 \
  --device "$device" \
  --workers 4 \
  --amp \
  --loss-alpha 3.0 \
  --patience 5 \
  --min-delta 0.0 \
  --save-every 5 \
  --input-norm-mode per_channel_max \
  --output-activation softplus \
  --low-events-allow 2000 5000 10000 20000 \
  --resume-checkpoint "$out_root/phase3_2k5k10k/best_model.pt"

rm -rf "$out_root/phase4_full/validation_latest"
python scripts/validate.py \
  --data-root "$data_root" \
  --checkpoint "$out_root/phase4_full/best_model.pt" \
  --out-dir "$out_root/phase4_full/validation_latest" \
  --plot-samples 2 \
  --device "$device" \
  --input-norm-mode per_channel_max

echo "Done"
echo "Final model: $out_root/phase4_full/best_model.pt"
echo "Validation: $out_root/phase4_full/validation_latest/metrics.csv"
