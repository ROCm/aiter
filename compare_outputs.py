#!/usr/bin/env python3
"""
Compare saved output tensors from different attention methods.

Usage:
    python compare_outputs.py output_fa3_*.pt output_fa3_fp8_*.pt
    python compare_outputs.py output_fa3_*.pt output_sage_v1_triton_*.pt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch


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
    fn: callable,
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


def compare_outputs(ref_file: str, test_file: str):
    """Compare two saved output tensors"""
    
    # Load tensors
    ref_output = torch.load(ref_file)
    test_output = torch.load(test_file)
    
    # Convert to float for comparison
    ref_output = ref_output.float()
    test_output = test_output.float()
    
    # Extract method names from filenames
    ref_method = Path(ref_file).stem.split('_b')[0]
    test_method = Path(test_file).stem.split('_b')[0]
    
    # print(f"\n{'='*60}")
    # print(f"Comparing {test_method} vs {ref_method} (reference)")
    print(f"{'='*60}")
    
    print(f"Files:")
    print(f"  Reference: {ref_file}")
    print(f"  Test:      {test_file}")
    
    # Ensure same shape (handle layout differences)
    if ref_output.shape != test_output.shape:
        print(f"\nWarning: Shape mismatch - ref: {ref_output.shape}, test: {test_output.shape}")
        # Try transposing if needed (e.g., BHSD vs BSHD)
        if len(ref_output.shape) == 4 and len(test_output.shape) == 4:
            if ref_output.shape[1] == test_output.shape[2] and ref_output.shape[2] == test_output.shape[1]:
                print(f"  Transposing test output dimensions 1 and 2")
                test_output = test_output.transpose(1, 2)
    
    # Print stats
    print(f"Output Tensor Stats:")
    print(f"  Reference ({ref_output.shape}): min={ref_output.min().item():.6f}, max={ref_output.max().item():.6f}, mean={ref_output.mean().item():.6f}, std={ref_output.std().item():.6f}")
    print(f"  Test ({test_output.shape}):      min={test_output.min().item():.6f}, max={test_output.max().item():.6f}, mean={test_output.mean().item():.6f}, std={test_output.std().item():.6f}")
    
    # Compute error metrics
    abs_diff = torch.abs(ref_output - test_output)
    
    print(f"Absolute Error:")
    print(f"  Mean: {abs_diff.mean().item():.6e}")
    print(f"  Max:  {abs_diff.max().item():.6e}")
    print(f"  Std:  {abs_diff.std().item():.6e}")
    
    # rel_diff = abs_diff / (torch.abs(ref_output) + 1e-8)
    # print(f"\nRelative Error:")
    # print(f"  Mean: {rel_diff.mean().item():.6e}")
    # print(f"  Max:  {rel_diff.max().item():.6e}")
    # print(f"  Std:  {rel_diff.std().item():.6e}")
    
    # Compute cosine similarity
    ref_flat = ref_output.reshape(-1)
    test_flat = test_output.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(ref_flat.unsqueeze(0), test_flat.unsqueeze(0))
    print(f"Cosine Similarity: {cos_sim.item():.8f}")
    
    # # Check numerical closeness at different tolerances
    # print(f"\nNumerical Closeness:")
    # for rtol, atol in [(1e-1, 1e-2), (1e-2, 1e-3), (1e-3, 1e-4), (1e-4, 1e-5), (1e-5, 1e-6)]:
    #     is_close = torch.allclose(ref_output, test_output, rtol=rtol, atol=atol)
    #     print(f"  rtol={rtol}, atol={atol}: {is_close}")
    
    # print(f"{'='*60}\n")
    
    return {
        'abs_mean': abs_diff.mean().item(),
        'abs_max': abs_diff.max().item(),
        'abs_std': abs_diff.std().item(),
        # 'rel_mean': rel_diff.mean().item(),
        # 'rel_max': rel_diff.max().item(),
        # 'rel_std': rel_diff.std().item(),
        'cosine_similarity': cos_sim.item(),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare saved attention output tensors')
    parser.add_argument('reference', type=str, help='Reference output file (.pt)')
    parser.add_argument('test', type=str, help='Test output file (.pt)')
    parser.add_argument('--save_json', type=str, default=None, help='Save comparison results to JSON')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.reference).exists():
        print(f"Error: Reference file not found: {args.reference}")
        sys.exit(1)
    
    if not Path(args.test).exists():
        print(f"Error: Test file not found: {args.test}")
        sys.exit(1)
    
    # Compare outputs
    metrics = compare_outputs(args.reference, args.test)
    
    # Optionally save to JSON
    if args.save_json:
        import json
        with open(args.save_json, 'w') as f:
            json.dump({
                'reference_file': args.reference,
                'test_file': args.test,
                **metrics
            }, f, indent=2)
        print(f"Saved comparison metrics to {args.save_json}")


if __name__ == '__main__':
    main()
