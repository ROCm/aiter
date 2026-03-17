#!/usr/bin/env python3
"""Collect (M, N, K) shapes for GEMM kernel tuning.

Gathers shapes from two sources:
1. Config files at aiter/ops/triton/configs/gemm/gfx950-GEMM-*.json
2. Model shapes from op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json

Outputs a deduplicated, sorted JSON list of {"M", "N", "K"} dicts.
"""

import argparse
import json
import os
import re
import sys
import warnings

STANDARD_M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

KERNEL_CONFIG_MAP = {
    "a16w16": {"config_pattern": "A16W16", "model_key": "gemm_a16w16"},
    "a16w16_agnostic": {"config_pattern": "A16W16", "model_key": "gemm_a16w16"},
    "a16w16_atomic": {"config_pattern": "A16W16", "model_key": "gemm_a16w16"},
    "a16w16_gated": {"config_pattern": "A16W16_GATED", "model_key": "gemm_a16w16"},
    "a16w8_blockscale": {"config_pattern": "A16W8_BLOCKSCALE", "model_key": None},
    "a16wfp4": {"config_pattern": "A16WFP4", "model_key": None},
    "a8w8": {"config_pattern": "A8W8", "model_key": None},
    "a8w8_blockscale": {"config_pattern": "A8W8_BLOCKSCALE", "model_key": "gemm_a8w8_blockscale"},
    "a8w8_per_token_scale": {"config_pattern": "A8W8_PER_TOKEN_SCALE", "model_key": "gemm_a8w8_per_token_scale"},
    "a8wfp4": {"config_pattern": "A8WFP4", "model_key": None},
    "afp4wfp4": {"config_pattern": "AFP4WFP4", "model_key": "gemm_afp4wfp4"},
    "afp4wfp4_pre_quant_atomic": {"config_pattern": "AFP4WFP4", "model_key": "gemm_afp4wfp4"},
}


def next_power_of_2(x):
    """Round x up to the next power of 2."""
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


def get_m_values_from_config_keys(keys):
    """Extract M values from config JSON keys like 'M_LEQ_64' and 'any'.

    For M_LEQ_<val> keys, the threshold value is treated as the M value.
    For 'any', we use the full standard M list.
    """
    m_values = set()
    has_any = False

    for key in keys:
        match = re.match(r"M_LEQ_(\d+)", key)
        if match:
            m_values.add(int(match.group(1)))
        elif key == "any":
            has_any = True

    if has_any:
        m_values.update(STANDARD_M_VALUES)

    return m_values


def collect_from_configs(kernel_name, repo_root):
    """Collect (M, N, K) shapes from gfx950-GEMM config files."""
    config_info = KERNEL_CONFIG_MAP.get(kernel_name)
    if config_info is None:
        warnings.warn(f"Unknown kernel: {kernel_name}")
        return set()

    config_pattern = config_info["config_pattern"]
    config_dir = os.path.join(repo_root, "aiter", "ops", "triton", "configs", "gemm")

    if not os.path.isdir(config_dir):
        warnings.warn(f"Config directory not found: {config_dir}")
        return set()

    # Match files like gfx950-GEMM-A8W8_BLOCKSCALE-N=7168-K=2048.json
    # The pattern must match exactly to avoid e.g. A8W8 matching A8W8_BLOCKSCALE
    nk_pattern = re.compile(
        rf"^gfx950-GEMM-{re.escape(config_pattern)}-N=(\d+)-K=(\d+)\.json$"
    )

    shapes = set()
    matched_files = []

    for filename in os.listdir(config_dir):
        match = nk_pattern.match(filename)
        if match:
            n_val = int(match.group(1))
            k_val = int(match.group(2))
            filepath = os.path.join(config_dir, filename)
            matched_files.append((filepath, n_val, k_val))

    # Also check the base config file (no N,K suffix) for M keys
    base_filename = f"gfx950-GEMM-{config_pattern}.json"
    base_filepath = os.path.join(config_dir, base_filename)

    for filepath, n_val, k_val in matched_files:
        try:
            with open(filepath) as f:
                config = json.load(f)
            m_values = get_m_values_from_config_keys(config.keys())
            for m in m_values:
                shapes.add((next_power_of_2(m), n_val, k_val))
        except (json.JSONDecodeError, OSError) as e:
            warnings.warn(f"Error reading {filepath}: {e}")

    # If the base config exists, it defines M ranges but no specific N,K.
    # We still parse M keys from it for any matched N,K pairs.
    # If no N,K-specific files were found, the base config alone
    # does not contribute shapes (no N,K to pair with).
    if os.path.isfile(base_filepath) and not matched_files:
        # No N,K specific configs; base config alone cannot produce shapes
        pass

    return shapes


def collect_from_model_shapes(kernel_name, repo_root):
    """Collect (M, N, K) shapes from model_shapes.json."""
    config_info = KERNEL_CONFIG_MAP.get(kernel_name)
    if config_info is None:
        return set()

    model_key = config_info.get("model_key")
    if model_key is None:
        return set()

    model_shapes_path = os.path.join(
        repo_root,
        "op_tests",
        "op_benchmarks",
        "triton",
        "model_benchmarking_tool",
        "model_shapes.json",
    )

    if not os.path.isfile(model_shapes_path):
        warnings.warn(f"Model shapes file not found: {model_shapes_path}")
        return set()

    try:
        with open(model_shapes_path) as f:
            all_models = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        warnings.warn(f"Error reading {model_shapes_path}: {e}")
        return set()

    shapes = set()
    for model_name, model_data in all_models.items():
        entries = model_data.get(model_key, [])
        for entry in entries:
            n_val = entry.get("N")
            k_val = entry.get("K")
            if n_val is not None and k_val is not None:
                for m in STANDARD_M_VALUES:
                    shapes.add((next_power_of_2(m), n_val, k_val))

    return shapes


# Fallback shapes for kernels without config files or model_shapes entries.
# These are common shapes used across most GEMM test/bench scripts.
FALLBACK_NK_PAIRS = [
    (1280, 8192), (8192, 1024), (4096, 4096), (7168, 2048), (2048, 7168),
    (2112, 7168), (7168, 2112), (3584, 8192), (8192, 3584),
]


def collect_shapes(kernel_name, repo_root):
    """Collect all shapes for a kernel from all sources."""
    shapes = set()
    shapes.update(collect_from_configs(kernel_name, repo_root))
    shapes.update(collect_from_model_shapes(kernel_name, repo_root))

    # If no shapes found from configs or model_shapes, use fallback shapes
    if not shapes:
        for n, k in FALLBACK_NK_PAIRS:
            for m in STANDARD_M_VALUES:
                shapes.add((next_power_of_2(m), n, k))

    return shapes


def shapes_to_list(shapes):
    """Convert set of (M, N, K) tuples to sorted list of dicts."""
    sorted_shapes = sorted(shapes, key=lambda x: (x[1], x[2], x[0]))
    return [{"M": m, "N": n, "K": k} for m, n, k in sorted_shapes]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect (M, N, K) shapes for GEMM kernel tuning."
    )
    parser.add_argument(
        "--kernel",
        type=str,
        required=True,
        help='Kernel name (e.g. "a8w8", "a16w16") or "all" for all kernels.',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write output shape files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve repo root relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Script is at: <repo>/aiter/ops/triton/utils/_triton/tunning/collect_shapes.py
    repo_root = os.path.normpath(
        os.path.join(script_dir, "..", "..", "..", "..", "..", "..")
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.kernel == "all":
        kernels = list(KERNEL_CONFIG_MAP.keys())
    else:
        if args.kernel not in KERNEL_CONFIG_MAP:
            print(f"Error: unknown kernel '{args.kernel}'", file=sys.stderr)
            print(f"Available kernels: {', '.join(sorted(KERNEL_CONFIG_MAP.keys()))}", file=sys.stderr)
            sys.exit(1)
        kernels = [args.kernel]

    for kernel in kernels:
        shapes = collect_shapes(kernel, repo_root)
        if not shapes:
            print(f"[{kernel}] No shapes found, skipping.")
            continue

        shape_list = shapes_to_list(shapes)
        output_path = os.path.join(args.output_dir, f"shapes_gemm_{kernel}.json")
        with open(output_path, "w") as f:
            json.dump(shape_list, f, indent=2)
            f.write("\n")

        print(f"[{kernel}] Wrote {len(shape_list)} shapes to {output_path}")


if __name__ == "__main__":
    main()
