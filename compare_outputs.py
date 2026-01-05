#!/usr/bin/env python3
"""
Compare saved output tensors from different attention methods.

Usage:
    python compare_outputs.py output_fav3_*.pt output_fav3_fp8_*.pt
    python compare_outputs.py output_fav3_*.pt output_sage_v1_triton_*.pt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from op_tests.triton_tests.utils.accuarcy_analysis import compare_accuracy


def sanitize_file_component(value: Any) -> str:
    """Sanitize a value for use as a filename component."""
    return str(value).replace(os.sep, "-").replace(" ", "-")


def move_to_cpu(obj: Any) -> Any:
    """Recursively move tensors to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_cpu(item) for item in obj)
    if isinstance(obj, dict):
        return {key: move_to_cpu(val) for key, val in obj.items()}
    return obj


def primary_output(result: Any) -> Any:
    """Extract the primary output tensor from a result."""
    if isinstance(result, torch.Tensor):
        return result
    if (
        isinstance(result, (list, tuple))
        and len(result) == 2
        and all(isinstance(item, torch.Tensor) for item in result)
    ):
        return result[0]
    return result


def build_output_path(
    args: argparse.Namespace,
    model: str,
    batch: int,
    hq: int,
    hk: int,
    sq: int,
    sk: int,
    d_head: int,
    d_head_v: int,
    dtype: torch.dtype,
    causal: bool,
    mode: str,
    varlen: bool,
) -> Path:
    """Build output file path from benchmark parameters."""
    dtype_str = str(dtype).split(".")[-1]
    parts = []
    if model:
        parts.append(f"model-{sanitize_file_component(model)}")
    parts.extend(
        [
            f"B{batch}",
            f"HQ{hq}",
            f"HK{hk}",
            f"SQ{sq}",
            f"SK{sk}",
            f"D{d_head}",
            f"DV{d_head_v}",
            f"causal-{int(bool(causal))}",
            f"mode-{mode}",
            f"layout-{args.layout}",
            f"dtype-{dtype_str}",
            "varlen" if varlen else "dense",
        ]
    )
    if args.fp8:
        parts.append("fp8")
    if args.qk_int8:
        parts.append("qk-int8")
    filename = "_".join(parts) + ".pt"
    return Path(args.output_dir) / filename


def save_benchmark_output(
    args: argparse.Namespace,
    fn: Callable[[], Any],
    saved_output_keys: set,
    model: str,
    batch: int,
    hq: int,
    hk: int,
    sq: int,
    sk: int,
    d_head: int,
    d_head_v: int,
    dtype: torch.dtype,
    causal: bool,
    mode: str,
    varlen: bool,
) -> None:
    """Save benchmark output tensor if not already saved for this configuration."""
    save_key = (
        batch,
        hq,
        hk,
        sq,
        sk,
        d_head,
        d_head_v,
        str(dtype),
        bool(causal),
        mode,
        args.layout,
        args.fp8,
        args.qk_int8,
        varlen,
        model,
    )
    if save_key in saved_output_keys:
        return

    saved_output_keys.add(save_key)
    output_value = fn()
    primary = primary_output(output_value)
    prepared = move_to_cpu(primary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = build_output_path(
        args, model, batch, hq, hk, sq, sk, d_head, d_head_v, dtype, causal, mode, varlen
    )
    torch.save(prepared, output_path)
    print(f"Saved output tensor to {output_path}")


def _is_qk_int8_file(file_path: str) -> bool:
    """Heuristic to detect SageAttn/QK-INT8 artifact from filename."""
    stem = Path(file_path).stem
    return "qk-int8" in stem or "qk_int8" in stem


def compare_outputs(ref_file: str, test_file: str):
    """Compare two saved output tensors"""

    # Load tensors
    ref_output = torch.load(ref_file)
    test_output = torch.load(test_file)

    # Convert to float for comparison
    ref_output = ref_output.float()
    test_output = test_output.float()

    print(f"{'='*60}")

    print(f"Files:")
    print(f"  Reference: {ref_file}")
    print(f"  Test:      {test_file}")

    # Ensure same layout (prefer dense BSHD layout to mirror inline checks)
    if ref_output.shape != test_output.shape:
        print(
            f"\nWarning: Shape mismatch - ref: {ref_output.shape}, test: {test_output.shape}"
        )
        if (
            ref_output.ndim == 4
            and test_output.ndim == 4
            and ref_output.shape[0] == test_output.shape[0]
            and ref_output.shape[3] == test_output.shape[3]
            and ref_output.shape[1] == test_output.shape[2]
            and ref_output.shape[2] == test_output.shape[1]
        ):
            ref_is_qk = _is_qk_int8_file(ref_file)
            test_is_qk = _is_qk_int8_file(test_file)
            if ref_is_qk and not test_is_qk:
                print(
                    "  Transposing reference output dimensions 1 and 2 to match dense layout"
                )
                ref_output = ref_output.transpose(1, 2).contiguous()
            elif test_is_qk and not ref_is_qk:
                print(
                    "  Transposing test output dimensions 1 and 2 to match dense layout"
                )
                test_output = test_output.transpose(1, 2).contiguous()
            else:
                print(
                    "  Could not infer layout ownership; transposing both tensors for comparison"
                )
                ref_output = ref_output.transpose(1, 2).contiguous()
                test_output = test_output.transpose(1, 2).contiguous()

    # Re-run shape check after normalization
    if ref_output.shape != test_output.shape:
        raise ValueError(
            f"Unable to align tensor shapes for comparison: {ref_output.shape} vs {test_output.shape}"
        )

    # Shared stats + correctness summary (keeps inline + file-based outputs consistent)
    compare_accuracy(test_output, ref_output)


def main():
    parser = argparse.ArgumentParser(description='Compare saved attention output tensors')
    parser.add_argument('--reference', type=str, help='Reference output file (.pt)')
    parser.add_argument('--test', type=str, help='Test output file (.pt)')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.reference).exists():
        print(f"Error: Reference file not found: {args.reference}")
        sys.exit(1)
    
    if not Path(args.test).exists():
        print(f"Error: Test file not found: {args.test}")
        sys.exit(1)
    
    # Compare outputs
    compare_outputs(args.reference, args.test)


if __name__ == '__main__':
    main()
