#!/usr/bin/env bash
set -euo pipefail

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
opengate_python="${OPENGATE_PYTHON:-python3}"
sample_root="$(dirname "$out")"
sample_id="$(basename "$sample_root")"
ct_mhd="$repo_root/mc/generated/$sample_id/slab_ct.mhd"

read -r layer_order layer_thickness source_x source_y source_z <<< "$($opengate_python "$repo_root/scripts/sample_variant_config.py" --sample-id "$sample_id" --global-seed 42)"

mkdir -p "$repo_root/mc/generated/$sample_id"
"$opengate_python" "$repo_root/scripts/create_slab_ct_mhd.py" \
  --out "$ct_mhd" \
  --layer-order "$layer_order" \
  --layer-thickness-mm "$layer_thickness"

cat > "$sample_root/variant.json" <<EOF
{
  "sample_id": "$sample_id",
  "layer_order": "$layer_order",
  "layer_thickness_mm": "$layer_thickness",
  "source_x_mm": $source_x,
  "source_y_mm": $source_y,
  "source_z_cm": $source_z
}
EOF

"$opengate_python" "$repo_root/scripts/gate_voxelized_ct_beam.py" \
  --ct-mhd "$ct_mhd" \
  --output-dir "$out" \
  --particle gamma \
  --energy-mev "$energy" \
  --n-events "$events" \
  --seed "$seed" \
  --source-z-cm "$source_z" \
  --source-x-mm "$source_x" \
  --source-y-mm "$source_y"

"$opengate_python" "$repo_root/scripts/postprocess_gate_output.py" \
  --dose-mhd "$out/dose_voxelized_ct_edep.mhd" \
  --ct-mhd "$ct_mhd" \
  --out-dir "$out"
