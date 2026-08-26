#!/bin/bash
# =============================================================================
# dsv4-kv-memory-estimate — repair DeepSeek-V4 per-request KV memory estimate
# =============================================================================
# INDEPENDENT MOD: fixes concurrency for BOTH fp8 and nvfp4 KV.
# Stock `_max_memory_usage_bytes_from_groups` (DSV4 special case) computed
#   num_layer_tuples * pages * sum-of-distinct-page-sizes, summed per group
# -> ~7x per-request over-estimate -> concurrency capped at ~1.5x instead of
# ~9x. Fix: per-request = full_mla_pages * real packed bytes-per-block, using
# the SAME divisor as the allocator (_pool_bytes_per_block).
#
# Apply standalone (no nvfp4 mod needed): on a stock eugr b12x fp8 stack this
# alone lifts the pool from ~1.5x to ~9x concurrency.
# =============================================================================
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
VLLM="$PYTHON_ROOT/vllm"
TARGET="$VLLM/v1/core/kv_cache_utils.py"

if [ ! -f "$TARGET" ]; then
  echo "[dsv4-kv-memory-estimate] missing $TARGET — wrong vLLM tree?" >&2
  exit 1
fi

python3 - "$VLLM" <<'PY'
import py_compile
import sys
from pathlib import Path

root = Path(sys.argv[1])


def replace(path: str, old: str, new: str, what: str) -> None:
    p = root / path
    text = p.read_text()
    if new in text:
        print(f"[dsv4-kv-memory-estimate] skip  {what} (already applied)")
        return
    if old not in text:
        raise SystemExit(f"[dsv4-kv-memory-estimate] FAIL {what}: anchor missing in {path}")
    p.write_text(text.replace(old, new, 1))
    print(f"[dsv4-kv-memory-estimate] ok    {what} -> {path}")


replace(
    "v1/core/kv_cache_utils.py",
    """        full_mla_spec = cast(UniformTypeKVCacheSpecs, kv_cache_groups[0].kv_cache_spec)
        layer_tuple_bytes = sum(full_mla_spec.get_page_sizes())
        num_layer_tuples = max(
            cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).get_num_layer_tuples()
            for group in kv_cache_groups
        )

        total_max_mem_usage_bytes = 0
        for group in kv_cache_groups:
            group_spec = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec)
            g_max_mem_usage_pages = group_spec.max_memory_usage_pages(vllm_config)
            g_max_mem_usage_page_bytes = (
                num_layer_tuples * g_max_mem_usage_pages * layer_tuple_bytes
            )
            total_max_mem_usage_bytes += g_max_mem_usage_page_bytes
        return total_max_mem_usage_bytes
""",
    """        full_mla_spec = cast(UniformTypeKVCacheSpecs, kv_cache_groups[0].kv_cache_spec)
        max_pages = full_mla_spec.max_memory_usage_pages(vllm_config)
        return max_pages * _pool_bytes_per_block(vllm_config, kv_cache_groups)
""",
    "memory estimate: real packed per-request bytes",
)

py_compile.compile(str(root / "v1/core/kv_cache_utils.py"), doraise=True)
print("[dsv4-kv-memory-estimate] patched + byte-compiled OK")
PY

echo "[dsv4-kv-memory-estimate] done"
