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

PROJECT_DIR=${PROJECT_DIR:-$SUBMIT_DIR}
DATA_DIR=${DATA_DIR:-${SCRATCH:-$HOME}/prueba_slabs_data}
EVENTS_HIGH=${EVENTS_HIGH:-1000000}
MAX_SAMPLES=${MAX_SAMPLES:-1001}
MC_ROOT=${MC_ROOT:-$DATA_DIR/mc_runs_opengate_photon_${EVENTS_HIGH}}
PAIRS_ROOT=${PAIRS_ROOT:-$DATA_DIR/mc_runs_opengate_photon_${EVENTS_HIGH}_multinoise_pairs}
DATASET_OUT=${DATASET_OUT:-$DATA_DIR/data_real_photon_${EVENTS_HIGH}_multinoise}
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$TMPDIR" 2>/dev/null || true

cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH=$PROJECT_DIR/src
mkdir -p logs

python scripts/extend_mc_multinoise_from_existing.py \
  --mc-root "$MC_ROOT" \
  --pairs-root "$PAIRS_ROOT" \
  --dataset-out "$DATASET_OUT" \
  --events-high "$EVENTS_HIGH" \
  --max-samples "$MAX_SAMPLES" \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --max-parallel 8 \
  --clean-pairs

echo "DONE dataset build"
