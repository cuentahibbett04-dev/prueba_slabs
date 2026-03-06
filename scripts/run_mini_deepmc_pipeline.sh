#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

source .venv/bin/activate
export PYTHONPATH="$repo_root/src"

cfg="${1:-mc/campaign.opengate.photon.mini_deepmc.json}"
train_out="${2:-artifacts_mini_deepmc}"

# Rebuild mini dataset from scratch so files are replaced, not accumulated.
python3 scripts/run_mini_deepmc_experiment.py --config "$cfg" --clean

# Keep training compact for quick iteration while debugging scale/normalization.
rm -rf "$train_out"
python3 scripts/train.py \
  --data-root mini_deepmc_photon/dataset \
  --out-dir "$train_out" \
  --epochs 12 \
  --batch-size 2 \
  --device cuda \
  --workers 4 \
  --amp \
  --loss-alpha 3.0 \
  --patience 4 \
  --min-delta 0.0 \
  --save-every 4 \
  --input-norm-mode per_channel_max \
  --input-dose-scale 1.0 \
  --output-activation softplus

# Replace previous evaluation artifacts to avoid PNG growth.
rm -rf "$train_out/validation_latest"
python3 scripts/validate.py \
  --data-root mini_deepmc_photon/dataset \
  --checkpoint "$train_out/best_model.pt" \
  --out-dir "$train_out/validation_latest" \
  --plot-samples 2 \
  --device cuda \
  --input-norm-mode per_channel_max \
  --input-dose-scale 1.0

echo "Mini pipeline complete"
echo "Model: $train_out/best_model.pt"
echo "Metrics: $train_out/validation_latest/metrics.csv"
