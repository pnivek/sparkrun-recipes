#!/bin/bash
# Backport SparkInfer 4448acf / PR #106 onto the pinned 272a84b runtime.
# Corrects compressed-MLA decode/prefill dispatch to honor the tensor's
# physical page stride for both contiguous payload and padded layouts.
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$MOD_DIR/pr106.patch"
SPARKINFER_ROOT="$(python3 - <<'PY'
import pathlib, sparkinfer
print(pathlib.Path(sparkinfer.__file__).resolve().parent)
PY
)"
SITE_ROOT="$(dirname "$SPARKINFER_ROOT")"
MARKER='Use the tensor\x27s physical page stride for exact-payload and padded views.'
KERNEL="$SPARKINFER_ROOT/attention/_shared/mla/kernel.py"

if grep -Fq "Use the tensor's physical page stride for exact-payload and padded views." "$KERNEL"; then
  echo "[sparkinfer-compressed-physical-stride] skip (PR #106 already applied)"
else
  echo "[sparkinfer-compressed-physical-stride] validating exact PR #106 backport"
  patch --dry-run --forward --batch -p1 -d "$SITE_ROOT" < "$PATCH_FILE"
  patch --forward --batch -p1 -d "$SITE_ROOT" < "$PATCH_FILE"
fi

python3 - "$SPARKINFER_ROOT" <<'PY'
import py_compile
import sys
from pathlib import Path
root = Path(sys.argv[1])
files = [
    "attention/_shared/mla/compressed_api.py",
    "attention/_shared/mla/compressed_reference.py",
    "attention/_shared/mla/kernel.py",
    "attention/_shared/mla/prefill.py",
    "attention/_shared/mla/prefill_mg.py",
]
for rel in files:
    py_compile.compile(str(root / rel), doraise=True)

import torch
from sparkinfer.attention._shared.mla.compressed_api import _validate_compressed_cache_layout
from sparkinfer.attention._shared.mla.compressed_reference import (
    COMPRESSED_MLA_BYTES_PER_TOKEN,
    COMPRESSED_MLA_DSV4_PAGE_SIZE,
    compressed_mla_page_nbytes,
)
from sparkinfer.attention._shared.mla.kernel import _cache_block_stride_bytes as decode_stride
from sparkinfer.attention._shared.mla.prefill import _cache_block_stride_bytes as prefill_stride
from sparkinfer.attention._shared.mla.prefill_mg import _cache_block_stride_bytes as prefill_mg_stride

page_size = COMPRESSED_MLA_DSV4_PAGE_SIZE
payload = page_size * COMPRESSED_MLA_BYTES_PER_TOKEN
padded = compressed_mla_page_nbytes(page_size)
for physical in {payload, padded}:
    storage = torch.empty(2 * physical, dtype=torch.uint8)
    cache = torch.as_strided(storage, size=(2, payload), stride=(physical, 1))
    _validate_compressed_cache_layout(cache, page_size=page_size, name="cache")
    assert decode_stride(cache, page_size=page_size, model_type=0) == physical
    assert prefill_stride(cache, page_size=page_size, model_type=0) == physical
    assert prefill_mg_stride(cache, page_size=page_size, is_glm=False) == physical
print(f"[sparkinfer-compressed-physical-stride] byte-compile + stride probes OK: payload={payload}, padded={padded}")
PY

echo "[sparkinfer-compressed-physical-stride] applied upstream 4448acf / PR #106"
