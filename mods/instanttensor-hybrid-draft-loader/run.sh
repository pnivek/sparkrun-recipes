#!/bin/bash
set -euo pipefail
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
TARGET="$PYTHON_ROOT/vllm/model_executor/model_loader/__init__.py"
PATCHER="$MOD_DIR/patch_model_loader.py"

[ -f "$TARGET" ] || { echo "[instanttensor-hybrid-draft-loader] target not found: $TARGET" >&2; exit 1; }
[ -f "$PATCHER" ] || { echo "[instanttensor-hybrid-draft-loader] patcher not found: $PATCHER" >&2; exit 1; }

python3 "$PATCHER" --check "$TARGET"
python3 "$PATCHER" "$TARGET"
python3 "$PATCHER" --check "$TARGET"

find "$(dirname "$TARGET")" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
echo "[instanttensor-hybrid-draft-loader] done"
