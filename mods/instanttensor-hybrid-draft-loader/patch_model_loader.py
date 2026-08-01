#!/usr/bin/env python3
"""Add hybrid loader policy for speculative draft models.
PATCHER V3: uses AST to find get_model() instead of exact string match.
Replaces the entire get_model function body with the patched version."""
from __future__ import annotations
import argparse
import ast
import sys
from pathlib import Path
import textwrap

MARKER = "# spark-vllm mod: instanttensor-hybrid-draft-loader v1"

HELPER_FUNC = textwrap.dedent("""\
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
            + ", ".join(allowed_modes) + "; got " + repr(mode)
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
    lines = source.splitlines(keepends=True)

    if MARKER in source:
        print("[instanttensor-hybrid-draft-loader] already patched.")
        return

    tree = ast.parse(source)

    # Find get_model function
    get_model_func = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_model":
            get_model_func = node
            break

    if get_model_func is None:
        print("[instanttensor-hybrid-draft-loader ERROR] get_model() not found", file=sys.stderr)
        sys.exit(1)

    func_start = get_model_func.lineno - 1  # 0-indexed first line of def
    func_end = get_model_func.end_lineno     # 1-indexed last line

    # Build the new get_model body
    new_body = textwrap.dedent("""\
    if model_config is None:
        model_config = vllm_config.model_config
    resolved_load_config = _instanttensor_draft_load_config(
        vllm_config, model_config, load_config
    )
    loader = get_model_loader(resolved_load_config)
    return loader.load_model(
        vllm_config=vllm_config, model_config=model_config, prefix=prefix
    )
    """)

    # Get indentation from the original function body
    # Find the first statement in the function body to determine indent
    first_stmt = get_model_func.body[0]
    body_indent = " " * (first_stmt.col_offset)

    # Indent the new body
    indented_body = "\n".join(
        body_indent + line if line.strip() else ""
        for line in new_body.strip().split("\n")
    )

    # Get the function signature lines (def line + parameter lines)
    sig_lines = []
    for i in range(func_start, func_end):
        line = lines[i]
        sig_lines.append(line)
        if line.rstrip().endswith("):") or line.rstrip().endswith("->") or line.rstrip() == ")":
            # Check if the line after "):" starts the body
            pass
        if "):" in line or ") ->" in line:
            break

    # Actually, simpler approach: take everything from func_start to func_end,
    # find where the signature ends (the first "):" or ") ->"), and replace
    # everything after that with the new body

    func_text = "".join(lines[func_start:func_end])

    # Find the end of the signature — look for "):" on a line
    sig_end_offset = None
    for i, line in enumerate(lines[func_start:func_end]):
        stripped = line.rstrip()
        if stripped.endswith("):") or stripped.endswith(") -> nn.Module:") or stripped.endswith(") -> \"nn.Module\":"):
            sig_end_offset = i
            break
        # Also handle multi-line: if the stripped line ends with ) and the
        # previous line started the signature
        if stripped == ") -> nn.Module:" or stripped == "):":
            sig_end_offset = i
            break

    if sig_end_offset is None:
        # Fallback: find the last line before the body starts
        for i, line in enumerate(lines[func_start:func_end]):
            if line.rstrip().endswith(":"):
                sig_end_offset = i
                break

    if sig_end_offset is None:
        print("[instanttensor-hybrid-draft-loader ERROR] could not find signature end", file=sys.stderr)
        sys.exit(1)

    # Build the new function
    sig_part = "".join(lines[func_start:func_start + sig_end_offset + 1])
    new_func = sig_part + "\n" + indented_body + "\n"

    # Replace the old function with the new one
    patched = source[:func_start] + new_func + source[func_end:]

    # Now insert the helper function before get_model
    # Find where to insert — before the marker in the new function
    marker_pos = patched.find(f"\n{MARKER}\n")
    if marker_pos == -1:
        # Insert helper right before the patched get_model
        insert_pos = patched.find(new_func)
        helper_block = f"{MARKER}\n{HELPER_FUNC}\n\n"
        patched = patched[:insert_pos] + helper_block + patched[insert_pos:]
    else:
        # Already inserted
        pass

    # Validate
    try:
        ast.parse(patched)
    except SyntaxError as e:
        print(f"[instanttensor-hybrid-draft-loader ERROR] syntax error after patch: {e}", file=sys.stderr)
        sys.exit(1)

    tmp = target_path.with_suffix(target_path.suffix + ".modtmp")
    tmp.write_text(patched)
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
