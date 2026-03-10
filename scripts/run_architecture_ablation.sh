#!/usr/bin/env bash
set -euo pipefail

# Compare architecture candidates under identical settings.
# Usage:
#   bash scripts/run_architecture_ablation.sh \
#     --data-root /path/to/data_real_photon_1000_multinoise \
#     --out-root artifacts_arch_ablation \
#     --epochs 20 --batch-size 2 --workers 4 --device cuda

DATA_ROOT="data_real_photon_1000_multinoise"
OUT_ROOT="artifacts_arch_ablation"
EPOCHS=20
BATCH_SIZE=2
WORKERS=4
DEVICE="cuda"
BASE_CHANNELS=16

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --base-channels) BASE_CHANNELS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

export PYTHONPATH="${PYTHONPATH:-$PWD/src}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export MIOPEN_ENABLE_LOGGING="${MIOPEN_ENABLE_LOGGING:-0}"
export TMPDIR="${TMPDIR:-${SCRATCH:-$HOME}/tmp}"
export MIOPEN_CACHE_DIR="${MIOPEN_CACHE_DIR:-${SCRATCH:-$HOME}/miopen_cache}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-${SCRATCH:-$HOME}/miopen_cache}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-${SCRATCH:-$HOME}/miopen_cache}"
mkdir -p "$TMPDIR" "$MIOPEN_CACHE_DIR" "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"

run_train () {
  local arch="$1"
  local out_dir="$2"
  local extra_args=("${@:3}")
  local train_log="$out_dir/train.log"
  local val_log="$out_dir/validate.log"

  mkdir -p "$out_dir"

  echo "=== Training arch=${arch} out=${out_dir} ==="
  local train_cmd=(
    python -u scripts/train.py
    --arch "$arch"
    --data-root "$DATA_ROOT"
    --out-dir "$out_dir"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --device "$DEVICE"
    --workers "$WORKERS"
    --amp
    --amp-dtype bf16
    --loss-alpha 3.0
    --loss-min-weight 0.05
    --background-threshold 0.02
    --background-lambda 0.2
    --patience 5
    --save-every 5
    --save-epochs 1
    --input-norm-mode per_channel_max
    --output-activation relu
    --base-channels "$BASE_CHANNELS"
  )
  if [[ ${#extra_args[@]} -gt 0 ]]; then
    train_cmd+=("${extra_args[@]}")
  fi

  {
    echo "[CMD] ${train_cmd[*]}"
  } >"$train_log"

  set +e
  "${train_cmd[@]}" >>"$train_log" 2>&1
  local train_rc=$?
  set -e
  if [[ $train_rc -ne 0 ]]; then
    echo "ERROR: train failed for arch=${arch} (rc=$train_rc)" >&2
    echo "Last train log lines:" >&2
    tail -n 120 "$train_log" >&2 || true
    return "$train_rc"
  fi

  if [[ ! -s "$train_log" ]]; then
    echo "ERROR: empty train log for arch=${arch}; train command produced no output" >&2
    return 1
  fi

  if [[ ! -f "$out_dir/history.json" ]]; then
    echo "ERROR: training produced no history.json for arch=${arch} in $out_dir" >&2
    echo "Last train log lines:" >&2
    tail -n 80 "$train_log" >&2 || true
    return 1
  fi

  local ckpt_path="$out_dir/best_model.pt"
  if [[ ! -f "$ckpt_path" ]]; then
    # Some runs can finish without best_model.pt (e.g., val_loss=nan throughout).
    # Fall back to latest periodic checkpoint to keep the ablation running.
    local latest_ckpt
    latest_ckpt=$(ls -1 "$out_dir"/checkpoint_epoch_*.pt 2>/dev/null | sort | tail -n 1 || true)
    if [[ -n "$latest_ckpt" && -f "$latest_ckpt" ]]; then
      ckpt_path="$latest_ckpt"
      echo "WARNING: missing best_model.pt for arch=${arch}; using latest checkpoint: $ckpt_path"
    else
      echo "ERROR: no checkpoint found for arch=${arch} in $out_dir" >&2
      if [[ -f "$train_log" ]]; then
        echo "Last train log lines:" >&2
        tail -n 80 "$train_log" >&2 || true
      fi
      if [[ -f "$out_dir/history.json" ]]; then
        echo "Last history rows:" >&2
        tail -n 40 "$out_dir/history.json" >&2 || true
      fi
      return 1
    fi
  fi

  local val_cmd=(
    python -u scripts/validate.py
    --data-root "$DATA_ROOT"
    --checkpoint "$ckpt_path"
    --out-dir "$out_dir/validation_latest"
    --device "$DEVICE"
    --input-norm-mode per_channel_max
  )

  {
    echo "[CMD] ${val_cmd[*]}"
  } >"$val_log"

  set +e
  "${val_cmd[@]}" >>"$val_log" 2>&1
  local val_rc=$?
  set -e
  if [[ $val_rc -ne 0 ]]; then
    echo "ERROR: validate failed for arch=${arch} (rc=$val_rc)" >&2
    echo "Last validate log lines:" >&2
    tail -n 120 "$val_log" >&2 || true
    return "$val_rc"
  fi

  if [[ ! -s "$val_log" ]]; then
    echo "WARNING: empty validate log for arch=${arch}" >&2
  fi

  echo "Train log: $train_log"
  echo "Val log:   $val_log"
}

run_train "resunet3d" "$OUT_ROOT/resunet3d"
run_train "attention_unet3d" "$OUT_ROOT/attention_unet3d"

# Optional: SwinUNETR only if MONAI is available.
if python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("monai") else 1)
PY
then
  run_train "swin_unetr" "$OUT_ROOT/swin_unetr" --base-channels 24
else
  echo "Skipping swin_unetr: MONAI not installed"
fi

echo "=== Done. Results under: $OUT_ROOT ==="
