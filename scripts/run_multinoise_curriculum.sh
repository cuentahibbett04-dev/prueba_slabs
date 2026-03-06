#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

source .venv/bin/activate
export PYTHONPATH="$repo_root/src"

cfg="${1:-mc/campaign.opengate.photon.multinoise.small.json}"
out_root="${2:-artifacts_multinoise_curriculum}"

# 1) Generate multi-noise pairs.
python3 scripts/run_multinoise_campaign.py --config "$cfg" --clean

# 2) Build normalized dataset and rescale low-dose by history ratio (100k/low_k).
output_root="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1],"r",encoding="utf-8")).get("output_root","multinoise_deepmc_small"))' "$cfg")"
mc_root="$output_root/pairs"

dataset_root="$(dirname "$mc_root")/dataset"
rm -rf "$dataset_root"
python3 scripts/build_dataset_from_mc.py \
  --mc-root "$mc_root" \
  --out-root "$dataset_root" \
  --train-ratio 0.75 \
  --val-ratio 0.125 \
  --seed 42 \
  --rescale-low-by-history-ratio \
  --default-events-low 2000 \
  --default-events-high 100000

# 3) Curriculum training: easy(20k) -> mid(10k,20k) -> full(2k,5k,10k,20k)
rm -rf "$out_root"
mkdir -p "$out_root"

python3 scripts/train.py \
  --data-root "$dataset_root" \
  --out-dir "$out_root/phase1_20k" \
  --epochs 8 --batch-size 2 --device cuda --workers 4 --amp \
  --loss-alpha 3.0 --patience 3 --min-delta 0.0 \
  --save-every 4 \
  --input-norm-mode per_channel_max \
  --output-activation softplus \
  --low-events-allow 20000

python3 scripts/train.py \
  --data-root "$dataset_root" \
  --out-dir "$out_root/phase2_10k20k" \
  --epochs 8 --batch-size 2 --device cuda --workers 4 --amp \
  --loss-alpha 3.0 --patience 3 --min-delta 0.0 \
  --save-every 4 \
  --input-norm-mode per_channel_max \
  --output-activation softplus \
  --low-events-allow 10000 20000 \
  --resume-checkpoint "$out_root/phase1_20k/best_model.pt"

python3 scripts/train.py \
  --data-root "$dataset_root" \
  --out-dir "$out_root/phase3_full" \
  --epochs 12 --batch-size 2 --device cuda --workers 4 --amp \
  --loss-alpha 3.0 --patience 4 --min-delta 0.0 \
  --save-every 4 \
  --input-norm-mode per_channel_max \
  --output-activation softplus \
  --low-events-allow 2000 5000 10000 20000 \
  --resume-checkpoint "$out_root/phase2_10k20k/best_model.pt"

# 4) Compact validation on final phase.
rm -rf "$out_root/phase3_full/validation_latest"
python3 scripts/validate.py \
  --data-root "$dataset_root" \
  --checkpoint "$out_root/phase3_full/best_model.pt" \
  --out-dir "$out_root/phase3_full/validation_latest" \
  --plot-samples 2 \
  --device cuda \
  --input-norm-mode per_channel_max

echo "Done"
echo "Dataset: $dataset_root"
echo "Final model: $out_root/phase3_full/best_model.pt"
echo "Metrics: $out_root/phase3_full/validation_latest/metrics.csv"
