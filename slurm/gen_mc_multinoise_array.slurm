#!/bin/bash
#SBATCH --job-name=mc_multi_1000
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-999%200

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
MC_ROOT=${MC_ROOT:-$DATA_DIR/mc_runs_opengate_photon_1000}
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$TMPDIR" 2>/dev/null || true
SIM_SCRIPT="$PROJECT_DIR/mc/run_photon_sim.sh"

if [[ ! -f "$SIM_SCRIPT" ]]; then
  echo "ERROR: simulation script not found: $SIM_SCRIPT" >&2
  exit 1
fi

# Set this in sbatch environment if python3 is not your OpenGATE environment python.
# Example:
# sbatch --export=ALL,OPENGATE_PYTHON=/path/to/opengate_env/bin/python slurm/gen_mc_multinoise_array.slurm
export OPENGATE_PYTHON=${OPENGATE_PYTHON:-python3}

cd "$PROJECT_DIR"
mkdir -p logs "$MC_ROOT"

idx=$SLURM_ARRAY_TASK_ID
energies=(1 2 4 6 8)

energy_idx=$(( idx / 200 ))
rep=$(( idx % 200 ))
energy=${energies[$energy_idx]}

sample=$(printf "E%03d_r%03d" "$energy" "$rep")
sample_dir="$MC_ROOT/$sample"
mkdir -p "$sample_dir"

run_if_missing() {
  local out_dir="$1"
  local events="$2"
  local seed="$3"
  if [[ ! -f "$out_dir/dose.npy" ]]; then
    bash "$SIM_SCRIPT" --energy "$energy" --events "$events" --out "$out_dir" --seed "$seed"
  fi
}

# high target once
run_if_missing "$sample_dir/high" 100000 $((100000 + idx))

# low levels
run_if_missing "$sample_dir/low" 2000 $((2000 + idx))
run_if_missing "$sample_dir/low_e05000" 5000 $((5000 + idx))
run_if_missing "$sample_dir/low_e10000" 10000 $((10000 + idx))
run_if_missing "$sample_dir/low_e20000" 20000 $((20000 + idx))

cat > "$sample_dir/meta.json" <<EOF
{
  "energy_mev": $energy,
  "events_low": 2000,
  "events_high": 100000
}
EOF

echo "DONE $sample"
