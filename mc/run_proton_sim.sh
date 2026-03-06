#!/usr/bin/env bash
set -euo pipefail

# Example wrapper expected by scripts/run_mc_campaign.py.
# Replace the body with your own Geant4/GATE application call.
# Required outputs per run:
#   $out/dose.npy
# Optional:
#   $out/spr.npy

energy=""
events=""
out=""
seed=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --energy)
      energy="$2"; shift 2 ;;
    --events)
      events="$2"; shift 2 ;;
    --out)
      out="$2"; shift 2 ;;
    --seed)
      seed="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"; exit 2 ;;
  esac
done

mkdir -p "$out"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
opengate_python="/home/fer/fer/ProtonAI/PrAI/.venv/bin/python"
opengate_script="/home/fer/fer/ProtonAI/PrAI/scripts/gate_voxelized_ct_experiment.py"
ct_mhd="$repo_root/mc/generated/slab_ct.mhd"

if [[ ! -x "$opengate_python" ]]; then
  echo "OpenGATE python not found: $opengate_python" >&2
  exit 1
fi

if [[ ! -f "$opengate_script" ]]; then
  echo "OpenGATE script not found: $opengate_script" >&2
  exit 1
fi

# Build slab CT once and reuse across runs.
if [[ ! -f "$ct_mhd" ]]; then
  "$opengate_python" "$repo_root/scripts/create_slab_ct_mhd.py" --out "$ct_mhd"
fi

"$opengate_python" "$opengate_script" \
  --ct-mhd "$ct_mhd" \
  --output-dir "$out" \
  --energy-mev "$energy" \
  --n-events "$events" \
  --seed "$seed" \
  --source-mode point \
  --source-z-cm -30 \
  --source-x-mm 0 \
  --source-y-mm 0 \
  --event-modulo 0 \
  --run-verbose 0 \
  --event-verbose 0

"$opengate_python" "$repo_root/scripts/postprocess_gate_output.py" \
  --dose-mhd "$out/dose_voxelized_ct_edep.mhd" \
  --ct-mhd "$ct_mhd" \
  --out-dir "$out"
