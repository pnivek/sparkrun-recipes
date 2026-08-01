#!/bin/bash
# fix-dspark-sm120-topk — let DSpark serve on SM120 (DGX Spark / GB10, sm_121 → sm120).
#
# Problem: DeepSeek-V4-Flash DSpark draft issues a sparse-MLA decode with topk=256,
# but FlashInfer's SM120 decode_dsv4 kernel only has compiled buckets {128, 512, 1024}.
# topk=256 falls through to the paged kernel which asserts num_tokens > 64 and crashes
# during warmup with:
#   tvm.error.InternalError: Check failed: num_tokens > 64 (5 vs. 64)
#
# Fix: round unsupported draft topk DOWN to nearest supported bucket (256 → 128)
# before dispatch. Correctness-safe: the DSpark draft is a SPECULATOR — every token it
# proposes is verified by the full target model before emission, so this only affects
# accept-length (speed), never the emitted output.
#
# Based on gitmarecki's v2 fix (anchored inside _paged_attention closure).
# Original PR: https://github.com/eugr/spark-vllm-docker/pull/319

set -e

# --- Fix /tmp/.cache permissions ---
# Known sparkrun issue: containers run as host user but /tmp/.cache/ can be
# root-owned from previous runs, breaking FlashInfer JIT and vLLM model cache.
# https://forums.developer.nvidia.com/t/360832?page=5
if [ -d /tmp/.cache ]; then
    chown -R $(id -u):$(id -g) /tmp/.cache/ 2>/dev/null || true
    echo "[fix-dspark-sm120-topk] /tmp/.cache permissions fixed"
fi

# --- Patch FlashInfer SM120 topk dispatch ---
F="$(python3 -c 'import os,flashinfer; print(os.path.join(os.path.dirname(flashinfer.__file__),"mla","_sparse_mla_sm120.py"))' 2>/dev/null)"

[ -f "$F" ] || {
  echo "[fix-dspark-sm120-topk] ERROR: flashinfer _sparse_mla_sm120.py not found ($F)" >&2
  exit 1
}

python3 - "$F" <<'PY'
import sys
f = sys.argv[1]
s = open(f).read()

MARKER = "fix-dspark-sm120-topk-v2"
if MARKER in s:
    print("[fix-dspark-sm120-topk] already applied")
    raise SystemExit(0)

# Anchor INSIDE _paged_attention (8-space indent) where model_type/topk_length are in scope.
# The original PR anchored in get_sparse_mla_sm120_module() (4-space indent) which caused
# NameError: name 'model_type' is not defined at cache-build time.
anchor = (
    "        num_tokens, num_heads, d_qk = q.shape\n"
    "        topk = indices.shape[-1]\n"
)

if anchor not in s:
    raise SystemExit(
        "[fix-dspark-sm120-topk] ERROR: anchor not found — flashinfer version drift"
    )

inject = (
    "        # " + MARKER + ": round an unsupported topk down to the nearest\n"
    "        # bucket _decode_dsv4_dispatchable() supports. Draft-only — target\n"
    "        # verifies every proposal, so this affects accept-length, not output.\n"
    "        if model_type == _MODEL_TYPE_DSV4:\n"
    "            _DSV4_TOPK_OK = (128, 512, 1024)\n"
    "            if topk not in _DSV4_TOPK_OK:\n"
    "                _lo = [b for b in _DSV4_TOPK_OK if b <= topk]\n"
    "                _tgt = max(_lo) if _lo else min(_DSV4_TOPK_OK)\n"
    "                indices = indices[..., :_tgt].contiguous()\n"
    "                if topk_length is not None:\n"
    "                    topk_length = topk_length.clamp(max=_tgt)\n"
    "                topk = _tgt\n"
)

open(f, "w").write(s.replace(anchor, anchor + inject, 1))
print("[fix-dspark-sm120-topk] applied inside _paged_attention (v2)")
PY

python3 -c "import ast; ast.parse(open('$F').read()); print('[fix-dspark-sm120-topk] verified (AST valid)')"
