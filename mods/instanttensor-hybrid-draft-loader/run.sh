#!/bin/bash
# instanttensor-hybrid-draft-loader — use lazy safetensors for speculative draft models
# while keeping InstantTensor for the target model.
#
# Problem: DSpark/Eagle3 draft models share the same checkpoint as the target.
# When both use InstantTensor (memory-mapped), they each mmap the full 155 GB file,
# doubling memory pressure and OOMing on DGX Spark unified memory (128 GB shared).
#
# Fix: patch vLLM's model loader to detect speculative draft loads and switch them
# to lazy safetensors — only reads the draft layers (40-42) instead of the whole file.
#
# From: eugr/spark-vllm-docker mods/instanttensor-hybrid-draft-loader

set -euo pipefail
PREFIX="[instanttensor-hybrid-draft-loader]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON_ROOT="/usr/local/lib/python3.12/dist-packages"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-$DEFAULT_PYTHON_ROOT}}"
VLLM_ROOT="$PYTHON_ROOT/vllm"
TARGET="$VLLM_ROOT/model_executor/model_loader/__init__.py"
PATCHER="$MOD_DIR/patch_model_loader.py"
MODE="${INSTANTTENSOR_DRAFT_LOADER:-auto}"
MODE_NORMALIZED="$(printf '%s' "$MODE" | tr '[:upper:]' '[:lower:]')"

echo "=== InstantTensor hybrid speculative-draft loader mod ==="

case "$MODE_NORMALIZED" in
  auto|safetensors|instanttensor) ;;
  *)
    echo "$PREFIX INSTANTTENSOR_DRAFT_LOADER must be auto, safetensors, or instanttensor; got '$MODE'." >&2
    exit 1
    ;;
esac

if [[ ! -d "$VLLM_ROOT" ]]; then
  echo "$PREFIX vLLM package not found at $VLLM_ROOT" >&2
  exit 1
fi

if [[ ! -f "$PATCHER" ]]; then
  echo "$PREFIX patcher not found at $PATCHER" >&2
  exit 1
fi

if [[ ! -f "$TARGET" ]]; then
  echo "$PREFIX vLLM model-loader module not found at $TARGET" >&2
  exit 1
fi

python3 "$PATCHER" --check "$TARGET"
python3 "$PATCHER" "$TARGET"
python3 "$PATCHER" --check "$TARGET"

find "$(dirname "$TARGET")" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "$PREFIX Enabled with INSTANTTENSOR_DRAFT_LOADER=$MODE_NORMALIZED."
echo "=== OK: target loads stay on InstantTensor; selected drafts use lazy safetensors ==="
