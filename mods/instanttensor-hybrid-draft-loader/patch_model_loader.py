#!/usr/bin/env python3
"""Patch vLLM model loader to use lazy safetensors for DSpark draft.
Simple approach: find def get_model( by string, replace body, insert helper."""
import argparse
import re
import sys
from pathlib import Path

MARKER = "# spark-vllm mod: instanttensor-hybrid-draft-loader v1"

HELPER = '''def _instanttensor_draft_load_config(
    vllm_config: "VllmConfig",
    model_config: "ModelConfig",
    load_config: "LoadConfig | None",
) -> "LoadConfig":
    """Resolve the loader for one model without changing the target loader."""
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


'''


def patch_file(target: Path) -> None:
    source = target.read_text()

    if MARKER in source:
        print("[instanttensor-hybrid-draft-loader] already patched.")
        return

    # Find "def get_model(" — must be at column 0 (top-level function)
    pattern = r'^def get_model\('
    match = re.search(pattern, source, re.MULTILINE)
    if not match:
        print("[instanttensor-hybrid-draft-loader ERROR] def get_model( not found", file=sys.stderr)
        sys.exit(1)

    func_start = match.start()
    # Find the end of the function: next top-level def/class, or end of file
    # A top-level definition starts at column 0 with def or class
    next_def = re.search(r'^(def |class |if __name__)', source[func_start + 4:], re.MULTILINE)
    if next_def:
        func_end = func_start + 4 + next_def.start()
    else:
        func_end = len(source)

    # Extract the signature (everything up to the "):" or ") -> ... :")
    sig_end = source.find('):', func_start)
    if sig_end == -1:
        # Multi-line return type like ") -> nn.Module:"
        sig_end = source.find(') ->', func_start)
        if sig_end != -1:
            # Extend to the colon at end of this line
            colon = source.find(':', sig_end)
            if colon != -1:
                sig_end = colon
    if sig_end == -1:
        print("[instanttensor-hybrid-draft-loader ERROR] cannot find signature end", file=sys.stderr)
        sys.exit(1)
    sig_end += 1  # include the ":"

    signature = source[func_start:sig_end]

    # Determine body indent from the first line after signature
    rest = source[sig_end:func_end].lstrip('\n')
    indent_match = re.match(r'^(\s+)', rest)
    body_indent = indent_match.group(1) if indent_match else '    '

    # Build new function body
    new_body_lines = [
        f'{body_indent}if model_config is None:',
        f'{body_indent}{body_indent}model_config = vllm_config.model_config',
        f'{body_indent}resolved_load_config = _instanttensor_draft_load_config(',
        f'{body_indent}{body_indent}vllm_config, model_config, load_config',
        f'{body_indent})',
        f'{body_indent}loader = get_model_loader(resolved_load_config)',
        f'{body_indent}return loader.load_model(',
        f'{body_indent}{body_indent}vllm_config=vllm_config, model_config=model_config, prefix=prefix',
        f'{body_indent})',
    ]
    new_body = '\n'.join(new_body_lines)
    new_func = signature + '\n' + new_body + '\n'

    # Assemble
    patched = source[:func_start] + new_func + source[func_end:]

    # Insert helper right before get_model
    helper_block = f'{MARKER}\n{HELPER}'
    insert_pos = patched.find(new_func)
    if insert_pos == -1:
        print("[instanttensor-hybrid-draft-loader ERROR] couldn't find patched function", file=sys.stderr)
        sys.exit(1)
    patched = patched[:insert_pos] + helper_block + patched[insert_pos:]

    # Validate
    try:
        compile(patched, '<patched>', 'exec')
    except SyntaxError as e:
        print(f"[instanttensor-hybrid-draft-loader ERROR] syntax error: {e}", file=sys.stderr)
        # Dump context
        lines = patched.split('\n')
        lo = max(0, e.lineno - 3)
        hi = min(len(lines), e.lineno + 2)
        for i in range(lo, hi):
            print(f"  {i+1:4d}: {lines[i]}", file=sys.stderr)
        sys.exit(1)

    tmp = target.with_suffix(target.suffix + '.modtmp')
    tmp.write_text(patched)
    tmp.replace(target)
    print("[instanttensor-hybrid-draft-loader] Patched successfully.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('target', type=Path)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()

    if not args.target.is_file():
        print(f"ERROR: {args.target} not found", file=sys.stderr)
        return 1

    if args.check:
        src = args.target.read_text()
        if MARKER in src:
            print(f"{args.target} is already patched.")
        else:
            print(f"{args.target} is compatible.")
        return 0

    patch_file(args.target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
