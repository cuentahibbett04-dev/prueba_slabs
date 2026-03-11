#!/bin/bash
#SBATCH --job-name=mc_multi
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-2499%200

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
MC_ROOT=${MC_ROOT:-$DATA_DIR/mc_runs_opengate_photon_${EVENTS_HIGH}}
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
SIM_TIMEOUT_SEC=${SIM_TIMEOUT_SEC:-1200}
SIM_KILL_AFTER_SEC=${SIM_KILL_AFTER_SEC:-60}
DEBUG_TRACE=${DEBUG_TRACE:-0}

if [[ "$DEBUG_TRACE" == "1" ]]; then
  set -x
fi

cd "$PROJECT_DIR"
mkdir -p logs "$MC_ROOT"

idx_offset=${IDX_OFFSET:-0}
idx=$(( SLURM_ARRAY_TASK_ID + idx_offset ))
energies=(1 2 4 6 8)
samples_per_energy=${SAMPLES_PER_ENERGY:-500}

if (( samples_per_energy <= 0 )); then
  echo "ERROR: SAMPLES_PER_ENERGY must be > 0, got: $samples_per_energy" >&2
  exit 1
fi

if (( idx_offset < 0 )); then
  echo "ERROR: IDX_OFFSET must be >= 0, got: $idx_offset" >&2
  exit 1
fi

energy_idx=$(( idx / samples_per_energy ))
rep=$(( idx % samples_per_energy ))
if (( energy_idx < 0 || energy_idx >= ${#energies[@]} )); then
  echo "ERROR: idx=$idx maps to invalid energy index=$energy_idx (len=${#energies[@]}, samples_per_energy=$samples_per_energy)" >&2
  exit 1
fi
energy=${energies[$energy_idx]}

sample=$(printf "E%03d_r%03d" "$energy" "$rep")
sample_dir="$MC_ROOT/$sample"
mkdir -p "$sample_dir"

log() {
  echo "[$(date '+%F %T')] $*"
}

run_if_missing() {
  local out_dir="$1"
  local events="$2"
  local seed="$3"
  if [[ -f "$out_dir/dose.npy" ]]; then
    log "SKIP existing dose: $out_dir/dose.npy"
    return 0
  fi

  log "START sample=$sample events=$events out=$out_dir seed=$seed"
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after="${SIM_KILL_AFTER_SEC}s" "${SIM_TIMEOUT_SEC}s" \
      bash "$SIM_SCRIPT" --energy "$energy" --events "$events" --out "$out_dir" --seed "$seed"
  else
    # Fallback when timeout command is unavailable on cluster.
    bash "$SIM_SCRIPT" --energy "$energy" --events "$events" --out "$out_dir" --seed "$seed"
  fi

  local rc=$?
  if (( rc == 124 )); then
    log "TIMEOUT sample=$sample events=$events after ${SIM_TIMEOUT_SEC}s"
    return 124
  fi
  if (( rc != 0 )); then
    log "ERROR sample=$sample events=$events rc=$rc"
    return "$rc"
  fi

  if [[ ! -f "$out_dir/dose.npy" ]]; then
    log "ERROR sample=$sample events=$events finished but dose.npy missing: $out_dir/dose.npy"
    return 2
  fi

  log "DONE sample=$sample events=$events"
}

# high target once
run_if_missing "$sample_dir/high" "$EVENTS_HIGH" $((EVENTS_HIGH + idx))

# low levels
run_if_missing "$sample_dir/low" 2000 $((2000 + idx))
run_if_missing "$sample_dir/low_e05000" 5000 $((5000 + idx))
run_if_missing "$sample_dir/low_e10000" 10000 $((10000 + idx))
run_if_missing "$sample_dir/low_e20000" 20000 $((20000 + idx))

cat > "$sample_dir/meta.json" <<EOF
{
  "energy_mev": $energy,
  "events_low": 2000,
  "events_high": $EVENTS_HIGH
}
EOF

echo "DONE $sample"
