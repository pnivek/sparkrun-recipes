#!/usr/bin/env python3
"""Add hybrid loader policy for speculative draft models.
PATCHER V3: uses AST to find get_model() instead of exact string match.
Works across vLLM versions as long as get_model() exists as a module-level function."""
from __future__ import annotations
import argparse
import ast
import sys
from pathlib import Path
import textwrap

MARKER = "# spark-vllm mod: instanttensor-hybrid-draft-loader v1"

PATCH_CODE = textwrap.dedent("""
def _instanttensor_draft_load_config(
    vllm_config: VllmConfig,
    model_config: ModelConfig,
    load_config: LoadConfig | None,
) -> LoadConfig:
    \"\"\"Resolve the loader for one model without changing the target loader.\"\"\"
    import os
    from vllm.config import replace
    mode = os.environ.get("INSTANTTENSOR_DRAFT_LOADER", "auto").strip().lower()
    allowed_modes = ("auto", "safetensors", "instanttensor")
    if mode not in allowed_modes:
        raise ValueError(
            "INSTANTTENSOR_DRAFT_LOADER must be one of "
            f"{{', '.join(allowed_modes)}}; got {{mode!r}}"
        )
    effective = load_config or vllm_config.load_config
    load_format = getattr(effective.load_format, "value", effective.load_format)
    if mode == "instanttensor" or str(load_format).lower() != "instanttensor":
        return effective
    speculative_config = getattr(vllm_config, "speculative_config", None)
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    if draft_model_config is None or model_config is not draft_model_config:
        return effective
    if mode == "auto":
        target_model_config = (
            getattr(speculative_config, "target_model_config", None)
            or vllm_config.model_config
        )
        draft_source = (
            getattr(draft_model_config, "model", None),
            getattr(draft_model_config, "revision", None),
        )
        target_source = (
            getattr(target_model_config, "model", None),
            getattr(target_model_config, "revision", None),
        )
        if draft_source != target_source:
            return effective
    logger.info_once(
        "Hybrid draft loading: using lazy safetensors for speculative draft "
        "weights while preserving InstantTensor for the target model "
        "(INSTANTTENSOR_DRAFT_LOADER=%s).",
        mode,
    )
    return replace(
        effective,
        load_format="safetensors",
        safetensors_load_strategy="lazy",
    )
""").strip()


def patch_file(target_path: Path) -> None:
    source = target_path.read_text()

    if MARKER in source:
        print("[instanttensor-hybrid-draft-loader] already patched.")
        return

    # Parse the source
    tree = ast.parse(source)

    # Find get_model function
    get_model_func = None
    get_model_idx = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_model":
            get_model_func = node
            break

    if get_model_func is None:
        print("[instanttensor-hybrid-draft-loader ERROR] get_model() not found in module", file=sys.stderr)
        sys.exit(1)

    # Get the line number where get_model starts (1-indexed)
    lines = source.splitlines(keepends=True)

    # Find where to insert the helper function: right before get_model
    insert_line = get_model_func.lineno - 1  # 0-indexed

    # Build the patch
    helper_def = f"\n{MARKER}\n{PATCH_CODE}\n\n"
    patched_lines = lines[:insert_line] + [helper_def] + lines[insert_line:]

    # Now patch get_model's body to use the helper
    patched_source = "".join(patched_lines)

    # Find the get_model body and add the resolved_load_config line
    # Look for "loader = get_model_loader(" pattern
    old_loader_call = "loader = get_model_loader(load_config or vllm_config.load_config)"
    new_loader_block = (
        "if model_config is None:\n"
        "        model_config = vllm_config.model_config\n"
        "    resolved_load_config = _instanttensor_draft_load_config(\n"
        "        vllm_config, model_config, load_config\n"
        "    )\n"
        "    loader = get_model_loader(resolved_load_config)"
    )

    if old_loader_call in patched_source:
        patched_source = patched_source.replace(old_loader_call, new_loader_block, 1)
    else:
        # Try alternative format (no space after comma)
        old_loader_call_v2 = "loader = get_model_loader(load_config or vllm_config.load_config)"
        if old_loader_call_v2 not in patched_source:
            print("[instanttensor-hybrid-draft-loader ERROR] could not find loader call in get_model()", file=sys.stderr)
            print("Looking for:", repr(old_loader_call), file=sys.stderr)
            sys.exit(1)
        patched_source = patched_source.replace(old_loader_call_v2, new_loader_block, 1)

    # Validate
    ast.parse(patched_source)

    tmp = target_path.with_suffix(target_path.suffix + ".modtmp")
    tmp.write_text(patched_source)
    tmp.replace(target_path)
    print("[instanttensor-hybrid-draft-loader] Patched successfully.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not args.target.is_file():
        print(f"[instanttensor-hybrid-draft-loader ERROR] not found: {args.target}", file=sys.stderr)
        return 1

    if args.check:
        source = args.target.read_text()
        if MARKER in source:
            print(f"[instanttensor-hybrid-draft-loader] {args.target} is already patched.")
        else:
            print(f"[instanttensor-hybrid-draft-loader] {args.target} is compatible.")
        return 0

    patch_file(args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
