#!/bin/bash
# =============================================================================
# nvfp4-dsv4-kv — enable nvfp4_ds_mla KV cache on DeepSeek-V4 b12x
# =============================================================================
# DEPENDS ON: dsv4-kv-memory-estimate (the concurrency fix) for the pool to
# size correctly, but applies independently of it.
#
# The base image already ships the nvfp4 infrastructure (CacheDType entry,
# concat_and_cache_nvfp4_mla writer + .so, B12X canonicalizer). What the
# stock code blocks:
#   1. dtype gate: _resolve_dsv4_kv_cache_dtype hard-asserts fp8 only.
#   2. page alignment: packed uint8 layouts need 576B (b12x kernels hardcode
#      it); stock gives nvfp4 512B -> 192B/block storage overrun crash.
#   3. backend dtype/shape: DeepseekV4FlashMLABackend doesn't advertise
#      nvfp4_ds_mla nor shape the 584B record.
#   4. page size: deepseek_v4 nvfp4 spec must be storage_block_size*584 to
#      match what the b12x compressed-MLA kernels read.
#
# After this mod, run vllm with: --kv-cache-dtype nvfp4_ds_mla
#
# ENVELOPE TOGGLE: VLLM_NVFP4_ENVELOPE=<584|432> (default 584)
#   584 = fp8-compatible 584B/token page (b12x compressed-MLA read path was
#         qualified on this; current default, matches fp8_ds_mla pool sizing)
#   432 = true NVFP4 MLA record (256B E2M1 NoPE + 32B E4M3 group-16 scales +
#         16B pad + 128B BF16 RoPE). The writer kernel
#         (concat_and_cache_nvfp4_mla_kernel) hardcodes/asserts this layout,
#         so the Python page math is the only place 584 is imposed. Flipping
#         to 432 increases pool capacity ~1.35x (3.11x -> ~4.2x @1M) but the
#         b12x sparse-MLA READ kernel page-stride must accept the tighter
#         record — verify with a boot + needle + bench before relying on it.
# =============================================================================
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
VLLM="$PYTHON_ROOT/vllm"
# Envelope: 584 (default, current qualified path) or 432 (true NVFP4 record)
ENVELOPE="${VLLM_NVFP4_ENVELOPE:-584}"
case "$ENVELOPE" in
  584|432) ;;
  *)
    echo "[nvfp4-dsv4-kv] ERROR: VLLM_NVFP4_ENVELOPE must be 584 or 432, got '$ENVELOPE'" >&2
    exit 2
    ;;
esac
echo "[nvfp4-dsv4-kv] envelope: ${ENVELOPE}B/token"

NEEDED_FILES=(
  "$VLLM/models/deepseek_v4/attention.py"
  "$VLLM/models/deepseek_v4/sparse_mla.py"
  "$VLLM/v1/attention/backends/mla/sparse_swa.py"
  "$VLLM/v1/kv_cache_interface.py"
)
for f in "${NEEDED_FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "[nvfp4-dsv4-kv] missing $f — wrong vLLM tree?" >&2
    exit 1
  fi
done

python3 - "$VLLM" "$ENVELOPE" <<'PY'
import py_compile
import sys
from pathlib import Path

root = Path(sys.argv[1])
envelope = sys.argv[2]  # "584" or "432"
envelope_bytes = int(envelope)


def replace(path: str, old: str, new: str, what: str) -> None:
    p = root / path
    text = p.read_text()
    if new in text:
        print(f"[nvfp4-dsv4-kv] skip  {what} (already applied)")
        return
    if old not in text:
        raise SystemExit(f"[nvfp4-dsv4-kv] FAIL {what}: anchor missing in {path}")
    p.write_text(text.replace(old, new, 1))
    print(f"[nvfp4-dsv4-kv] ok    {what} -> {path}")


# ---- 1. dtype gate: accept nvfp4/nvfp4_ds_mla in the packed-uint8 path ----
replace(
    "models/deepseek_v4/attention.py",
    """    if use_fp8_ds_mla_layout:
        # fp8_ds_mla block format: UE8M0 block-scaled fp8 packed as uint8.
        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, "
            f"got {kv_cache_dtype}"
        )
        if kv_cache_dtype != "fp8_ds_mla":
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        return kv_cache_dtype, torch.uint8
""",
    """    if use_fp8_ds_mla_layout:
        # fp8_ds_mla block format: UE8M0 block-scaled fp8 packed as uint8.
        # nvfp4_ds_mla: packed NVFP4 MLA record (B12X reads natively).
        if kv_cache_dtype in ("nvfp4", "nvfp4_ds_mla"):
            if cache_config is not None:
                cache_config.cache_dtype = "nvfp4_ds_mla"
            kv_cache_dtype = "nvfp4_ds_mla"
            logger.info_once("Using DeepSeek V4 nvfp4_ds_mla KV cache format.")
            return kv_cache_dtype, torch.uint8
        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, "
            f"got {kv_cache_dtype}"
        )
        if kv_cache_dtype != "fp8_ds_mla":
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        return kv_cache_dtype, torch.uint8
""",
    "dtype gate: nvfp4 accepted",
)

# ---- 2. alignment: 576B for nvfp4 in the main MLA spec ----
replace(
    "models/deepseek_v4/attention.py",
    """        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
""",
    """        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        uses_packed_uint8_layout = uses_fp8_ds_mla_layout or self.kv_cache_dtype in (
            "nvfp4",
            "nvfp4_ds_mla",
        )
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_packed_uint8_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_packed_uint8_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
""",
    "alignment: 576B for nvfp4 (main MLA)",
)

# ---- 3. alignment: 576B for nvfp4 in the SWA cache spec ----
replace(
    "v1/attention/backends/mla/sparse_swa.py",
    """        uses_fp8_ds_mla_layout = self.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.cache_config.cache_dtype),
        )
""",
    """        uses_fp8_ds_mla_layout = self.cache_config.cache_dtype == "fp8_ds_mla"
        uses_packed_uint8_layout = uses_fp8_ds_mla_layout or self.cache_config.cache_dtype in (
            "nvfp4",
            "nvfp4_ds_mla",
        )
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
            alignment=576 if uses_packed_uint8_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.cache_config.cache_dtype),
        )
""",
    "alignment: 576B for nvfp4 (SWA cache)",
)

# ---- 3b. SWA KV shape: envelope-aware page size (must stay <= main MLA) ----
replace(
    "v1/attention/backends/mla/sparse_swa.py",
    """        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 SWA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        else:
            return (num_blocks, block_size, head_size)
""",
    f"""        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 SWA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        if cache_dtype_str == "nvfp4_ds_mla":
            # SWA page must stay <= the full-MLA page or the uniform-groups
            # invariant breaks (kv_cache_utils: max(sm) <= max(all)). Follow
            # VLLM_NVFP4_ENVELOPE ({envelope}: 584 default, 432 true record).
            return (num_blocks, block_size, {envelope_bytes})
        else:
            return (num_blocks, block_size, head_size)
""",
    "SWA KV shape: nvfp4 envelope",
)

# ---- 3c. SWA real_page_size_bytes: honor envelope for nvfp4 ----
# SlidingWindowMLASpec.real_page_size_bytes only branches on fp8_ds_mla and
# falls through to the head_size*dtype formula for nvfp4 — producing a page
# size that mismatches the main-MLA envelope and breaks the uniform-groups
# invariant (assert max(sm) <= max(all) in kv_cache_utils). Patch the nvfp4
# branch to follow the same envelope as MLAAttentionSpec.
replace(
    "v1/kv_cache_interface.py",
    """        if self.model_version == "deepseek_v4" and self.cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 FlashMLA: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B
            # per token. FlashInfer's contiguous bf16/fp8 cache falls through to
            # the element-size formula below.
            return self.storage_block_size * 584
        assert self.model_version in (None, "deepseek_v4"), (
""",
    f"""        if self.model_version == "deepseek_v4" and self.cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 FlashMLA: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B
            # per token. FlashInfer's contiguous bf16/fp8 cache falls through to
            # the element-size formula below.
            return self.storage_block_size * 584
        if self.model_version == "deepseek_v4" and self.cache_dtype_str == "nvfp4_ds_mla":
            # NVFP4 MLA latent: same record as the main MLA (256B E2M1 NoPE +
            # 32B E4M3 group-16 scales + 16B pad + 128B BF16 RoPE). Follow
            # VLLM_NVFP4_ENVELOPE ({envelope}: 584 default, 432 true record)
            # so the SWA page stays <= the full-MLA page (grouping invariant).
            return self.storage_block_size * {envelope_bytes}
        assert self.model_version in (None, "deepseek_v4"), (
""",
    "SWA real_page_size_bytes: nvfp4 envelope",
)

# ---- 4. backend: advertise nvfp4_ds_mla + KV shape branch ----
replace(
    "models/deepseek_v4/sparse_mla.py",
    """    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]
""",
    """    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
        "nvfp4_ds_mla",
    ]
""",
    "backend advertises nvfp4_ds_mla",
)

replace(
    "models/deepseek_v4/sparse_mla.py",
    """        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        else:
            return (num_blocks, block_size, head_size)
""",
    f"""        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        if cache_dtype_str == "nvfp4_ds_mla":
            # True NVFP4 record is 432B (256B E2M1 NoPE + 32B scales + 16B pad
            # + 128B BF16 RoPE) — the writer kernel asserts exactly 432. When
            # VLLM_NVFP4_ENVELOPE=584 we keep the fp8-compatible page so the
            # b12x read kernels stride 584B (qualified path). With 432 the
            # pool fits ~1.35x more tokens.
            return (num_blocks, block_size, {envelope_bytes})
        else:
            return (num_blocks, block_size, head_size)
""",
    "backend KV shape: nvfp4 envelope",
)

# ---- 5. page size: envelope-aware (584 default, 432 true NVFP4 record) ----
replace(
    "v1/kv_cache_interface.py",
    """            if self.model_version == "deepseek_v4":
                return self.storage_block_size * 432
            if self.model_version == "glm_fp8_rope":
""",
    f"""            if self.model_version == "deepseek_v4":
                # NVFP4 MLA latent: 432B/token (256B E2M1 NoPE + 32B E4M3
                # group-16 scales + 16B pad + 128B BF16 RoPE). Envelope
                # VLLM_NVFP4_ENVELOPE={envelope}: 584 = fp8-compatible page
                # (b12x read-kernel qualified); 432 = true NVFP4 record
                # (writer kernel asserts exactly 432).
                return self.storage_block_size * {envelope_bytes}
            if self.model_version == "glm_fp8_rope":
""",
    f"page size: {envelope}B envelope for deepseek_v4 nvfp4",
)

# --- top-level VllmConfig validator (added upstream ~2026-08: rejects any
# cache_dtype.startswith("nvfp4") when use_mla, which over-matches the packed
# nvfp4_ds_mla DSV4 layout this mod enables). Neutralize for nvfp4_ds_mla only;
# plain nvfp4 stays rejected as upstream intends. Older trees without the
# validator skip cleanly.
config_vllm = root / "config/vllm.py"
if config_vllm.exists():
    src = config_vllm.read_text()
    if "validate_nvfp4_kv_cache_with_mla" in src:
        replace(
            "config/vllm.py",
            """        if (
            self.cache_config.cache_dtype.startswith("nvfp4")
            and self.model_config.use_mla
        ):
""",
            """        if (
            self.cache_config.cache_dtype.startswith("nvfp4")
            and self.cache_config.cache_dtype != "nvfp4_ds_mla"
            and self.model_config.use_mla
        ):
""",
            "VllmConfig nvfp4+MLA validator: exempt nvfp4_ds_mla",
        )
    else:
        print("[nvfp4-dsv4-kv] skip  VllmConfig nvfp4+MLA validator (not present in this tree)")

print("[nvfp4-dsv4-kv] verifying byte-compile...")
for rel in (
    "models/deepseek_v4/attention.py",
    "models/deepseek_v4/sparse_mla.py",
    "v1/attention/backends/mla/sparse_swa.py",
    "v1/kv_cache_interface.py",
    "config/vllm.py",
):
    py_compile.compile(str(root / rel), doraise=True)
print("[nvfp4-dsv4-kv] all patches applied + byte-compiled OK")
PY

echo "[nvfp4-dsv4-kv] done. run vllm with: --kv-cache-dtype nvfp4_ds_mla"
