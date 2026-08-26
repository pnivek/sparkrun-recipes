#!/bin/bash
# Backport SparkInfer 4448acf / PR #106 onto the pinned 272a84b runtime.
# Corrects compressed-MLA decode/prefill dispatch to honor the tensor's
# physical page stride for both contiguous payload and padded layouts.
#
# SparkInfer >= 1.2.x renamed the package sparkinfer -> b12x and ships the
# fix upstream. This mod detects both and no-ops when the fix is present.
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$MOD_DIR/pr106.patch"
MARKER="physical page stride"

PKG_ROOT="$(python3 - <<'PY'
import importlib.util, pathlib
for name in ("sparkinfer", "b12x"):
    spec = importlib.util.find_spec(name)
    if spec and spec.origin:
        print(pathlib.Path(spec.origin).resolve().parent)
        break
PY
)"

if [ -z "$PKG_ROOT" ]; then
  echo "[sparkinfer-compressed-physical-stride] ERROR: neither sparkinfer nor b12x package found" >&2
  exit 1
fi

KERNEL="$PKG_ROOT/attention/_shared/mla/kernel.py"

if [ -f "$KERNEL" ] && grep -q "$MARKER" "$KERNEL"; then
  echo "[sparkinfer-compressed-physical-stride] skip (PR #106 already present in $(basename "$PKG_ROOT"))"
  exit 0
fi

if [ ! -f "$KERNEL" ]; then
  echo "[sparkinfer-compressed-physical-stride] ERROR: $KERNEL not found — package layout changed; re-qualify this mod" >&2
  exit 1
fi

SITE_ROOT="$(dirname "$PKG_ROOT")"
echo "[sparkinfer-compressed-physical-stride] validating exact PR #106 backport"
patch --dry-run --forward --batch -p1 -d "$SITE_ROOT" < "$PATCH_FILE"
patch --forward --batch -p1 -d "$SITE_ROOT" < "$PATCH_FILE"

python3 - "$PKG_ROOT" <<'PY'
import py_compile
import sys
from pathlib import Path
root = Path(sys.argv[1])
pkg = root.name
files = [
    "attention/_shared/mla/compressed_api.py",
    "attention/_shared/mla/compressed_reference.py",
    "attention/_shared/mla/kernel.py",
    "attention/_shared/mla/prefill.py",
    "attention/_shared/mla/prefill_mg.py",
]
for rel in files:
    py_compile.compile(str(root / rel), doraise=True)

import importlib
import torch
capi = importlib.import_module(f"{pkg}.attention._shared.mla.compressed_api")
cref = importlib.import_module(f"{pkg}.attention._shared.mla.compressed_reference")
kern = importlib.import_module(f"{pkg}.attention._shared.mla.kernel")
pref = importlib.import_module(f"{pkg}.attention._shared.mla.prefill")
pref_mg = importlib.import_module(f"{pkg}.attention._shared.mla.prefill_mg")

page_size = cref.COMPRESSED_MLA_DSV4_PAGE_SIZE
payload = page_size * cref.COMPRESSED_MLA_BYTES_PER_TOKEN
padded = cref.compressed_mla_page_nbytes(page_size)
for physical in {payload, padded}:
    storage = torch.empty(2 * physical, dtype=torch.uint8)
    cache = torch.as_strided(storage, size=(2, payload), stride=(physical, 1))
    capi._validate_compressed_cache_layout(cache, page_size=page_size, name="cache")
    assert kern._cache_block_stride_bytes(cache, page_size=page_size, model_type=0) == physical
    assert pref._cache_block_stride_bytes(cache, page_size=page_size, model_type=0) == physical
    assert pref_mg._cache_block_stride_bytes(cache, page_size=page_size, is_glm=False) == physical
print(f"[sparkinfer-compressed-physical-stride] byte-compile + stride probes OK: payload={payload}, padded={padded}")
PY

echo "[sparkinfer-compressed-physical-stride] applied upstream 4448acf / PR #106"
