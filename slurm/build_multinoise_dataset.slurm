#!/bin/bash
#SBATCH --job-name=build_multi_ds
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/prueba_slabs}
DATA_DIR=${DATA_DIR:-$SCRATCH/prueba_slabs_data}

cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH=$PROJECT_DIR/src
mkdir -p logs

python scripts/extend_mc_multinoise_from_existing.py \
  --mc-root "$DATA_DIR/mc_runs_opengate_photon_1000" \
  --pairs-root "$DATA_DIR/mc_runs_opengate_photon_1000_multinoise_pairs" \
  --dataset-out "$DATA_DIR/data_real_photon_1000_multinoise" \
  --max-parallel 8 \
  --clean-pairs

echo "DONE dataset build"
