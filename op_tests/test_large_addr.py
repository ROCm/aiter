#!/usr/bin/env python3
"""
Test suite for validating 64-bit address fixes in FMHA backward kernels.

This test suite covers all backward kernel variants from fmha_bwd_dqdkdv.csv
to ensure they work correctly with large tensor sizes that exceed 32-bit address range.

Usage:
    python test_large_addr.py                    # Run all tests
    python test_large_addr.py --kernel bf16_a32  # Filter tests by kernel name substring
    python test_large_addr.py --list             # List all available tests
    python test_large_addr.py --csv              # Show CSV kernel mapping
"""

import argparse
import csv
import os
import sys
from pathlib import Path
import torch

# Set environment to disable v3 forward (use CK for forward, v3 for backward)
os.environ["AITER_DISABLE_V3_FWD"] = "1"

import aiter
from aiter.test_common import checkAllclose, run_perftest
from aiter.test_mha_common import (
    attention_ref,
    generate_qkv,
)


def get_device_arch():
    """Get the GPU architecture (gfx942, gfx950, etc.)"""
    props = torch.cuda.get_device_properties(0)
    gcn_arch = props.gcnArchName if hasattr(props, 'gcnArchName') else "unknown"
    return gcn_arch


def get_hsa_path():
    """Get path to HSA kernels based on GPU architecture."""
    arch = get_device_arch()
    # Determine the base arch (e.g., gfx950 from gfx950:sramecc+:xnack-)
    base_arch = arch.split(":")[0] if ":" in arch else arch
    return Path(aiter.__file__).parent.parent / "hsa" / base_arch / "fmha_v3_bwd"


def load_kernel_configs():
    """Load kernel configurations from CSV file."""
    hsa_path = get_hsa_path()
    csv_path = hsa_path / "fmha_bwd_dqdkdv.csv"
    
    if not csv_path.exists():
        print(f"Warning: CSV file not found at {csv_path}")
        return []
    
    configs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            configs.append(row)
    
    return configs


def run_torch(
    q, k, v, dout,
    causal=False,
    window_size=(-1, -1),
    upcast=True,
    reorder_ops=False,
):
    """
    Run PyTorch reference implementation (aligned with test_mha.py).
    Uses attention_ref from aiter.test_mha_common.
    """
    out, _, softmax_lse = attention_ref(
        q, k, v,
        None,  # query_padding_mask
        None,  # key_padding_mask
        None,  # attn_bias
        0.0,   # dropout_p
        None,  # dropout_mask
        causal=causal,
        window_size=window_size,
        upcast=upcast,
        reorder_ops=reorder_ops,
    )
    
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    return out, softmax_lse, dq, dk, dv


def print_mismatch_info(name, tensor_aiter, tensor_ref, tensor_pt, tol):
    """Print detailed mismatch information (aligned with test_mha.py)."""
    diff = (tensor_aiter - tensor_ref).abs()
    max_diff = diff.max().item()
    
    if max_diff > tol:
        # Find max diff location
        max_idx = torch.unravel_index(torch.argmax(diff), diff.shape)
        coords = tuple(idx.item() for idx in max_idx)
        print(f"\n--- {name} Mismatch Details ---")
        print(f"  Max diff: {max_diff}")
        print(f"  Max diff coords (batch, seq, head, dim): {coords}")
        print(f"  Aiter value:  {tensor_aiter[max_idx].item()}")
        print(f"  Ref value:    {tensor_ref[max_idx].item()}")
        if tensor_pt is not None:
            print(f"  PyTorch value: {tensor_pt[max_idx].item()}")
            print(f"  Aiter-Ref diff: {(tensor_aiter[max_idx] - tensor_ref[max_idx]).item()}")
            print(f"  PyTorch-Ref diff: {(tensor_pt[max_idx] - tensor_ref[max_idx]).item()}")
        
        # Print top 5 largest mismatches (handle large tensors)
        flat_diff = diff.flatten()
        top_k = 5
        if flat_diff.numel() <= 2**31 - 1:
            top_k = min(top_k, flat_diff.numel())
            top_vals, top_indices = torch.topk(flat_diff, top_k)
            print(f"  Top {top_k} mismatches:")
            for i in range(top_k):
                idx = torch.unravel_index(top_indices[i], diff.shape)
                coords = tuple(ix.item() for ix in idx)
                print(f"    [{i+1}] coords={coords}, diff={top_vals[i].item():.6f}, "
                      f"aiter={tensor_aiter[idx].item():.6f}, ref={tensor_ref[idx].item():.6f}")
        else:
            print(f"  (Tensor too large for top-k analysis, showing max only)")


def get_test_params_for_kernel(config):
    """
    Generate test parameters for a given kernel configuration.
    Uses large sequence lengths to trigger 64-bit address overflow.
    
    Returns dict with test parameters or None if kernel should be skipped.
    """
    dtype_str = config['dtype']
    hdim_q = int(config['hdim_q'])
    hdim_v = int(config['hdim_v'])
    mask = int(config['mask'])  # 0=no mask, 1=causal, 2=causal_br
    atomic32 = int(config['atomic32'])  # 0=a16, 1=a32
    mode = int(config['mode'])  # 0=normal, 1=group/varlen
    co_name = config['co_name']
    
    # Map dtype string to torch dtype
    dtype = torch.bfloat16 if dtype_str == 'bf16' else torch.float16
    
    # Determine causal type
    causal = mask > 0
    if mask == 2:
        causal_type = "bottom_right"
    elif mask == 1:
        causal_type = "top_left"
    else:
        causal_type = None
    
    # deterministic=False is required to use ASM backend kernels
    # atomic32 field determines a16 vs a32 kernel via is_v3_atomic_fp32 flag
    # is_v3_atomic_fp32: True=A32 kernel, False=A16 kernel
    deterministic = False
    is_v3_atomic_fp32 = (atomic32 == 1)  # 1=a32, 0=a16
    
    # Calculate sequence length to trigger 32-bit overflow
    # For batch=8, heads=40, hdim=128, dtype=2bytes:
    # Total size = 8 * seqlen * 40 * 128 * 2 = 81920 * seqlen bytes
    # Need > 4GB = 4294967296 bytes
    # seqlen > 4294967296 / 81920 = 52428
    # Use seqlen_k = 75600 to ensure overflow (gives ~6.2GB)
    batch_size = 8
    nheads = 40
    seqlen_q = 256
    seqlen_k = 75600  # Large enough to trigger 32-bit address overflow
    
    return {
        "batch_size": batch_size,
        "nheads": nheads,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "hdim_q": hdim_q,
        "hdim_v": hdim_v,
        "dtype": dtype,
        "causal": causal,
        "causal_type": causal_type,
        "deterministic": deterministic,
        "co_name": co_name,
        "dtype_str": dtype_str,
        "atomic32": atomic32,
        "is_v3_atomic_fp32": is_v3_atomic_fp32,  # True=A32, False=A16
        "is_varlen": (mode == 1),  # group mode uses varlen API
    }


def run_mha_backward_test(
    batch_size: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    hdim_q: int,
    hdim_v: int,
    dtype: torch.dtype,
    causal: bool,
    causal_type: str = None,
    deterministic: bool = False,
    test_name: str = "unknown",
    co_name: str = "",
    is_v3_atomic_fp32: bool = True,  # True=A32 kernel, False=A16 kernel
    **kwargs,
):
    """
    Run a single MHA backward test and return results.
    Fully aligned with test_mha.py methodology.
    """
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    causal_flag = "" if causal else "--no-causal"
    det_flag = "--deterministic" if deterministic else "--no-deterministic"
    cmd = (f"AITER_DISABLE_V3_FWD=1 python test_mha.py -b {batch_size} -n {nheads} "
           f"-q {seqlen_q} -k {seqlen_k} -d_qk_v {hdim_q},{hdim_v} -d {dtype_str} "
           f"{causal_flag} --no-local {det_flag} -m mha")
    
    print(f"\n{'='*70}")
    print(f"Test: {test_name}")
    print(f"CO file: {co_name}")
    print(f"Command: {cmd}")
    print(f"Config: batch={batch_size}, heads={nheads}, seq_q={seqlen_q}, seq_k={seqlen_k}")
    print(f"        hdim_q={hdim_q}, hdim_v={hdim_v}, dtype={dtype}")
    print(f"        causal={causal}, causal_type={causal_type}, deterministic={deterministic}")
    print(f"        is_v3_atomic_fp32={is_v3_atomic_fp32} ({'A32' if is_v3_atomic_fp32 else 'A16'} kernel)")
    print(f"{'='*70}")
    
    device = "cuda"
    
    # Set window size for causal (aligned with test_mha.py)
    if causal:
        if causal_type == "bottom_right":
            window_size = (seqlen_k - seqlen_q, 0)
        else:  # top_left
            window_size = (-1, 0)
    else:
        window_size = (-1, -1)
    
    # Create input tensors (aligned with test_mha.py)
    q = torch.randn(batch_size, seqlen_q, nheads, hdim_q, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, hdim_q, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, hdim_v, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(batch_size, seqlen_q, nheads, hdim_v, device=device, dtype=dtype, requires_grad=True)
    
    # Run aiter forward + backward (aligned with test_mha.py's run_ck)
    try:
        (out, softmax_lse, S_dmask), _ = run_perftest(
            aiter.flash_attn_func,
            q, k, v,
            0.0,   # dropout_p
            None,  # softmax_scale
            causal,
            window_size,
            None,  # bias
            None,  # alibi_slopes
            deterministic,
            return_lse=True,
            return_attn_probs=True,  # Aligned with test_mha.py
            how_v3_bf16_cvt=2,
            is_v3_atomic_fp32=is_v3_atomic_fp32,  # True=A32, False=A16
            num_rotate_args=1,
        )
        
        # Compute gradients
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        
    except Exception as e:
        import traceback
        error_msg = str(e) if str(e) else f"{type(e).__name__}"
        print(f"  ERROR: Aiter failed with: {error_msg}")
        traceback.print_exc()
        return {"passed": False, "error": error_msg or "Unknown error", "co_name": co_name}
    
    # Run PyTorch reference in float32 (upcast=True) - aligned with test_mha.py
    out_ref, softmax_lse_ref, dq_ref, dk_ref, dv_ref = run_torch(
        q, k, v, dout, causal=causal, window_size=window_size, upcast=True
    )
    
    # Run PyTorch in original dtype with reorder_ops (upcast=False, reorder_ops=True)
    # This is exactly what test_mha.py does for tolerance calculation
    out_pt, softmax_lse_pt, dq_pt, dk_pt, dv_pt = run_torch(
        q, k, v, dout, causal=causal, window_size=window_size, upcast=False, reorder_ops=True
    )
    
    # Print diff values (aligned with test_mha.py)
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"softmax_lse max diff: {(softmax_lse - softmax_lse_ref).abs().max().item()}")
    print(f"softmax_lse Pytorch max diff: {(softmax_lse_pt - softmax_lse_ref).abs().max().item()}")
    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
    
    # Tolerance calculation (aligned with test_mha.py: 10x PyTorch diff, min 0.01)
    dq_tol = max(10 * (dq_pt - dq_ref).abs().max().item(), 0.01)
    dk_tol = max(10 * (dk_pt - dk_ref).abs().max().item(), 0.01)
    dv_tol = max(10 * (dv_pt - dv_ref).abs().max().item(), 0.01)
    
    dq_diff = (dq - dq_ref).abs().max().item()
    dk_diff = (dk - dk_ref).abs().max().item()
    dv_diff = (dv - dv_ref).abs().max().item()
    
    dq_passed = dq_diff <= dq_tol
    dk_passed = dk_diff <= dk_tol
    dv_passed = dv_diff <= dv_tol
    passed = dq_passed and dk_passed and dv_passed
    
    status_q = '✓' if dq_passed else '✗'
    status_k = '✓' if dk_passed else '✗'
    status_v = '✓' if dv_passed else '✗'
    
    print(f"\nTolerance: dQ_tol={dq_tol:.6f}, dK_tol={dk_tol:.6f}, dV_tol={dv_tol:.6f}")
    print(f"  dQ: diff={dq_diff:.6f} vs tol={dq_tol:.6f} {status_q}")
    print(f"  dK: diff={dk_diff:.6f} vs tol={dk_tol:.6f} {status_k}")
    print(f"  dV: diff={dv_diff:.6f} vs tol={dv_tol:.6f} {status_v}")
    print(f"  Result: {'PASSED' if passed else 'FAILED'}")
    
    # Print mismatch details if failed (aligned with test_mha.py)
    print_mismatch_info("dQ", dq, dq_ref, dq_pt, dq_tol)
    print_mismatch_info("dK", dk, dk_ref, dk_pt, dk_tol)
    print_mismatch_info("dV", dv, dv_ref, dv_pt, dv_tol)
    
    return {
        "passed": passed,
        "dq_diff": dq_diff,
        "dk_diff": dk_diff,
        "dv_diff": dv_diff,
        "dq_tol": dq_tol,
        "dk_tol": dk_tol,
        "dv_tol": dv_tol,
        "co_name": co_name,
    }


def run_mha_varlen_backward_test(
    batch_size: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    hdim_q: int,
    hdim_v: int,
    dtype: torch.dtype,
    causal: bool,
    causal_type: str = None,
    deterministic: bool = False,
    test_name: str = "unknown",
    co_name: str = "",
    is_v3_atomic_fp32: bool = True,  # True=A32 kernel, False=A16 kernel
    **kwargs,
):
    """
    Run a single MHA varlen (group mode) backward test and return results.
    Aligned with test_mha.py methodology.
    """
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    
    print(f"\n{'='*70}")
    print(f"Test: {test_name} (VARLEN/GROUP MODE)")
    print(f"CO file: {co_name}")
    print(f"Command: python test_large_addr.py --kernel {co_name.replace('.co', '')}")
    print(f"Config: batch={batch_size}, heads={nheads}, seq_q={seqlen_q}, seq_k={seqlen_k}")
    print(f"        hdim_q={hdim_q}, hdim_v={hdim_v}, dtype={dtype}")
    print(f"        causal={causal}, causal_type={causal_type}, deterministic={deterministic}")
    print(f"        is_v3_atomic_fp32={is_v3_atomic_fp32} ({'A32' if is_v3_atomic_fp32 else 'A16'} kernel)")
    print(f"{'='*70}")
    
    device = "cuda"
    
    # Set window size for causal
    if causal:
        if causal_type == "bottom_right":
            window_size = (seqlen_k - seqlen_q, 0)
        else:  # top_left
            window_size = (-1, 0)
    else:
        window_size = (-1, -1)
    
    # For varlen mode, create packed tensors
    total_q = batch_size * seqlen_q
    total_k = batch_size * seqlen_k
    
    # Create cumulative sequence length tensors
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, seqlen_q, device=device, dtype=torch.int32)
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, seqlen_k, device=device, dtype=torch.int32)
    
    # Create packed input tensors (no batch dimension)
    q = torch.randn(total_q, nheads, hdim_q, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(total_k, nheads, hdim_q, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total_k, nheads, hdim_v, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(total_q, nheads, hdim_v, device=device, dtype=dtype, requires_grad=True)
    
    # Run aiter forward + backward
    try:
        out, softmax_lse, *_ = aiter.flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            dropout_p=0.0,
            softmax_scale=None,
            causal=causal,
            window_size=window_size,
            return_lse=True,
            return_attn_probs=False,
            deterministic=deterministic,
            is_v3_atomic_fp32=is_v3_atomic_fp32,  # True=A32, False=A16
        )
        
        # Compute gradients
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        
    except Exception as e:
        import traceback
        error_msg = str(e) if str(e) else f"{type(e).__name__}"
        print(f"  ERROR: Aiter failed with: {error_msg}")
        traceback.print_exc()
        return {"passed": False, "error": error_msg or "Unknown error", "co_name": co_name}
    
    # For reference, reshape to batch format and compute using attention_ref
    q_batch = q.reshape(batch_size, seqlen_q, nheads, hdim_q).requires_grad_(True)
    k_batch = k.reshape(batch_size, seqlen_k, nheads, hdim_q).requires_grad_(True)
    v_batch = v.reshape(batch_size, seqlen_k, nheads, hdim_v).requires_grad_(True)
    dout_batch = dout.reshape(batch_size, seqlen_q, nheads, hdim_v)
    
    # Run PyTorch reference in float32 (upcast=True)
    out_ref, softmax_lse_ref, dq_ref_batch, dk_ref_batch, dv_ref_batch = run_torch(
        q_batch, k_batch, v_batch, dout_batch, causal=causal, window_size=window_size, upcast=True
    )
    
    # Run PyTorch in original dtype with reorder_ops
    out_pt, softmax_lse_pt, dq_pt_batch, dk_pt_batch, dv_pt_batch = run_torch(
        q_batch, k_batch, v_batch, dout_batch, causal=causal, window_size=window_size, upcast=False, reorder_ops=True
    )
    
    # Reshape reference results to match varlen format
    dq_ref = dq_ref_batch.reshape(total_q, nheads, hdim_q)
    dk_ref = dk_ref_batch.reshape(total_k, nheads, hdim_q)
    dv_ref = dv_ref_batch.reshape(total_k, nheads, hdim_v)
    dq_pt = dq_pt_batch.reshape(total_q, nheads, hdim_q)
    dk_pt = dk_pt_batch.reshape(total_k, nheads, hdim_q)
    dv_pt = dv_pt_batch.reshape(total_k, nheads, hdim_v)
    
    # Print diff values (aligned with test_mha.py)
    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
    
    # Tolerance calculation (aligned with test_mha.py)
    dq_tol = max(10 * (dq_pt - dq_ref).abs().max().item(), 0.01)
    dk_tol = max(10 * (dk_pt - dk_ref).abs().max().item(), 0.01)
    dv_tol = max(10 * (dv_pt - dv_ref).abs().max().item(), 0.01)
    
    dq_diff = (dq - dq_ref).abs().max().item()
    dk_diff = (dk - dk_ref).abs().max().item()
    dv_diff = (dv - dv_ref).abs().max().item()
    
    dq_passed = dq_diff <= dq_tol
    dk_passed = dk_diff <= dk_tol
    dv_passed = dv_diff <= dv_tol
    passed = dq_passed and dk_passed and dv_passed
    
    status_q = '✓' if dq_passed else '✗'
    status_k = '✓' if dk_passed else '✗'
    status_v = '✓' if dv_passed else '✗'
    
    print(f"\nTolerance: dQ_tol={dq_tol:.6f}, dK_tol={dk_tol:.6f}, dV_tol={dv_tol:.6f}")
    print(f"  dQ: diff={dq_diff:.6f} vs tol={dq_tol:.6f} {status_q}")
    print(f"  dK: diff={dk_diff:.6f} vs tol={dk_tol:.6f} {status_k}")
    print(f"  dV: diff={dv_diff:.6f} vs tol={dv_tol:.6f} {status_v}")
    print(f"  Result: {'PASSED' if passed else 'FAILED'}")
    
    # Print mismatch details if failed
    print_mismatch_info("dQ", dq, dq_ref, dq_pt, dq_tol)
    print_mismatch_info("dK", dk, dk_ref, dk_pt, dk_tol)
    print_mismatch_info("dV", dv, dv_ref, dv_pt, dv_tol)
    
    return {
        "passed": passed,
        "dq_diff": dq_diff,
        "dk_diff": dk_diff,
        "dv_diff": dv_diff,
        "dq_tol": dq_tol,
        "dk_tol": dk_tol,
        "dv_tol": dv_tol,
        "co_name": co_name,
    }


def run_all_tests(kernel_filter=None):
    """Run all kernel tests or a filtered subset."""
    print(f"\nRunning on: {get_device_arch()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    configs = load_kernel_configs()
    
    if not configs:
        print("No kernel configurations found!")
        return
    
    # Filter configs if requested
    if kernel_filter:
        filter_lower = kernel_filter.lower()
        filtered = []
        for c in configs:
            co_name = c['co_name'].lower()
            # Check if filter matches
            if filter_lower in co_name:
                # If filter doesn't contain 'group', exclude group kernels
                # unless the filter is an exact match for the kernel name
                kernel_base = co_name.replace('bwd_', '').replace('.co', '')
                filter_base = filter_lower.replace('bwd_', '').replace('.co', '')
                
                if 'group' not in filter_lower and kernel_base.endswith('_group'):
                    # Filter doesn't want group kernels, skip
                    continue
                filtered.append(c)
        configs = filtered
        if not configs:
            print(f"No kernels matching '{kernel_filter}' found!")
            return
    
    results = []
    passed = 0
    failed = 0
    skipped = 0
    
    for config in configs:
        params = get_test_params_for_kernel(config)
        if params is None:
            skipped += 1
            continue
        
        test_name = params['co_name'].replace('.co', '')
        
        # Choose test function based on mode
        if params.get('is_varlen', False):
            result = run_mha_varlen_backward_test(
                **params,
                test_name=test_name,
            )
        else:
            result = run_mha_backward_test(
                **params,
                test_name=test_name,
            )
        
        results.append(result)
        if result.get('passed', False):
            passed += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total:   {len(results)}")
    print(f"{'='*70}")
    
    if failed > 0:
        print(f"\nFailed tests:")
        for r in results:
            if not r.get('passed', False):
                if 'error' in r:
                    print(f"  - {r['co_name']}: {r['error']}")
                else:
                    dq = r.get('dq_diff', float('nan'))
                    dk = r.get('dk_diff', float('nan'))
                    dv = r.get('dv_diff', float('nan'))
                    dq_tol = r.get('dq_tol', 0.01)
                    dk_tol = r.get('dk_tol', 0.01)
                    dv_tol = r.get('dv_tol', 0.01)
                    # Format properly
                    if isinstance(dq, (int, float)) and not (dq != dq):  # not NaN
                        dq_str = f"dQ={dq:.4f}(tol={dq_tol:.4f})"
                    else:
                        dq_str = f"dQ={dq}"
                    if isinstance(dk, (int, float)) and not (dk != dk):
                        dk_str = f"dK={dk:.4f}(tol={dk_tol:.4f})"
                    else:
                        dk_str = f"dK={dk}"
                    if isinstance(dv, (int, float)) and not (dv != dv):
                        dv_str = f"dV={dv:.4f}(tol={dv_tol:.4f})"
                    else:
                        dv_str = f"dV={dv}"
                    print(f"  - {r['co_name']}: {dq_str}, {dk_str}, {dv_str}")


def list_tests():
    """List all available tests."""
    configs = load_kernel_configs()
    
    print(f"\nAvailable kernel tests ({len(configs)} total):")
    print(f"{'='*70}")
    
    for config in configs:
        mode_str = "group" if config['mode'] == '1' else "batch"
        atomic_str = "a32" if config['atomic32'] == '1' else "a16"
        mask_str = ["none", "causal", "causal_br"][int(config['mask'])]
        print(f"  {config['co_name']}: {config['dtype']} {atomic_str} hdim={config['hdim_q']}/{config['hdim_v']} "
              f"mask={mask_str} mode={mode_str}")


def show_csv():
    """Show CSV kernel mapping."""
    configs = load_kernel_configs()
    
    print(f"\nCSV kernel configurations ({len(configs)} entries):")
    print(f"{'='*100}")
    
    headers = list(configs[0].keys()) if configs else []
    print("  " + " | ".join(f"{h:>10}" for h in headers))
    print(f"  {'-'*90}")
    
    for config in configs:
        values = [config.get(h, '') for h in headers]
        print("  " + " | ".join(f"{v:>10}" for v in values))


def main():
    parser = argparse.ArgumentParser(
        description="Test 64-bit address fixes in FMHA backward kernels"
    )
    parser.add_argument(
        "--kernel", "-k",
        type=str,
        default=None,
        help="Filter tests by kernel name substring (e.g., 'bf16_a32', 'hd128')"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available tests"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Show CSV kernel mapping"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_tests()
    elif args.csv:
        show_csv()
    else:
        run_all_tests(kernel_filter=args.kernel)


if __name__ == "__main__":
    main()
