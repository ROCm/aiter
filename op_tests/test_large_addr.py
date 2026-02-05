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
from aiter.test_common import checkAllclose


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
    if mask == 0:
        causal = False
        causal_type = None
    elif mask == 1:
        causal = True
        causal_type = "top_left"
    else:  # mask == 2
        causal = True
        causal_type = "bottom_right"
    
    # deterministic=False is required to use ASM backend kernels
    # atomic32 field determines a16 vs a32 kernel via is_v3_atomic_fp32 flag
    # is_v3_atomic_fp32: True=A32 kernel, False=A16 kernel
    deterministic = False
    is_v3_atomic_fp32 = (atomic32 == 1)  # 1=a32, 0=a16
    
    # Calculate sequence length to trigger 32-bit overflow
    # Target: batch * seq * heads * hdim * sizeof(dtype) > 4GB
    # For bf16/fp16: sizeof = 2 bytes
    # Example: 8 * 75600 * 40 * 128 * 2 = 6.2 GB > 4GB
    
    # Adjust seq_k based on hdim to ensure overflow while staying within memory limits
    if hdim_q == 64:
        batch_size = 4
        nheads = 32
        seqlen_q = 256
        seqlen_k = 100000  # Larger for smaller hdim
    elif hdim_q == 128:
        # batch_size = 8
        # nheads = 40
        # seqlen_q = 256
        # seqlen_k = 75600

        batch_size = 8
        nheads = 40
        seqlen_q = 256
        seqlen_k = 75600
    else:  # hdim_q == 192
        batch_size = 8
        nheads = 40
        seqlen_q = 256
        seqlen_k = 50000  # Smaller for larger hdim
    
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
    
    Returns:
        dict with test results including max diffs and pass/fail status
    """
    # Build equivalent command string
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    causal_flag = "" if causal else "--no-causal"
    det_flag = "--deterministic" if deterministic else "--no-deterministic"
    cmd = (f"AITER_DISABLE_V3_FWD=1 python test_mha.py -b {batch_size} -n {nheads} "
           f"-q {seqlen_q} -k {seqlen_k} -d_qk_v {hdim_q},{hdim_v} -d {dtype_str} "
           f"{causal_flag} --no-local {det_flag} -m mha".strip())
    
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
    
    # Create input tensors
    q = torch.randn(batch_size, seqlen_q, nheads, hdim_q, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, hdim_q, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, hdim_v, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(batch_size, seqlen_q, nheads, hdim_v, device=device, dtype=dtype)
    
    # Set window size for causal
    if causal:
        if causal_type == "bottom_right":
            window_size_left = seqlen_k - seqlen_q
            window_size_right = 0
        else:  # top_left
            window_size_left = -1
            window_size_right = 0
    else:
        window_size_left = -1
        window_size_right = -1
    
    # Run aiter forward + backward
    try:
        out, softmax_lse, *_ = aiter.flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            softmax_scale=None,
            causal=causal,
            window_size=(window_size_left, window_size_right),
            return_lse=True,  # Required for backward pass
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
    
    # Run PyTorch reference
    q_ref = q.detach().clone().float().requires_grad_(True)
    k_ref = k.detach().clone().float().requires_grad_(True)
    v_ref = v.detach().clone().float().requires_grad_(True)
    
    # Simple attention reference in float32 for accuracy
    scale = 1.0 / (hdim_q ** 0.5)
    q_ref_t = q_ref.transpose(1, 2)  # B, H, S_q, D
    k_ref_t = k_ref.transpose(1, 2)  # B, H, S_k, D
    v_ref_t = v_ref.transpose(1, 2)  # B, H, S_k, D_v
    
    attn = torch.matmul(q_ref_t, k_ref_t.transpose(-2, -1)) * scale
    
    if causal:
        # Create causal mask
        if causal_type == "bottom_right":
            row_idx = torch.arange(seqlen_q, device=device).unsqueeze(1)
            col_idx = torch.arange(seqlen_k, device=device).unsqueeze(0)
            offset = seqlen_k - seqlen_q
            mask = col_idx <= row_idx + offset
        else:
            row_idx = torch.arange(seqlen_q, device=device).unsqueeze(1)
            col_idx = torch.arange(seqlen_k, device=device).unsqueeze(0)
            mask = col_idx <= row_idx
        
        attn = attn.masked_fill(~mask, float('-inf'))
    
    attn = torch.softmax(attn, dim=-1)
    out_ref = torch.matmul(attn, v_ref_t)
    out_ref = out_ref.transpose(1, 2)  # B, S_q, H, D_v
    
    dout_ref = dout.clone().float()
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout_ref)
    
    # Convert back to original dtype for comparison
    dq_ref = dq_ref.to(dtype)
    dk_ref = dk_ref.to(dtype)
    dv_ref = dv_ref.to(dtype)
    
    # Compare results
    dq_diff = (dq - dq_ref).abs().max().item()
    dk_diff = (dk - dk_ref).abs().max().item()
    dv_diff = (dv - dv_ref).abs().max().item()
    
    # Tolerance based on dtype
    if dtype == torch.bfloat16:
        tol = 0.02  # bf16 has lower precision
    else:  # float16
        tol = 0.01
    
    passed = dq_diff <= tol and dk_diff <= tol and dv_diff <= tol
    
    status_q = '✓' if dq_diff <= tol else '✗'
    status_k = '✓' if dk_diff <= tol else '✗'
    status_v = '✓' if dv_diff <= tol else '✗'
    
    print(f"  dQ max diff: {dq_diff:.6f} {status_q}")
    print(f"  dK max diff: {dk_diff:.6f} {status_k}")
    print(f"  dV max diff: {dv_diff:.6f} {status_v}")
    print(f"  Result: {'PASSED' if passed else 'FAILED'}")
    
    # If failed, show mismatch details
    if not passed:
        for name, aiter_tensor, ref_tensor in [
            ("dQ", dq, dq_ref),
            ("dK", dk, dk_ref),
            ("dV", dv, dv_ref),
        ]:
            diff = (aiter_tensor - ref_tensor).abs()
            max_diff = diff.max().item()
            if max_diff > tol:
                max_idx = torch.unravel_index(torch.argmax(diff), diff.shape)
                coords = tuple(idx.item() for idx in max_idx)
                print(f"  {name} max diff at {coords}: aiter={aiter_tensor[max_idx].item():.6f}, ref={ref_tensor[max_idx].item():.6f}")
    
    return {
        "passed": passed,
        "dq_diff": dq_diff,
        "dk_diff": dk_diff,
        "dv_diff": dv_diff,
        "tol": tol,
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
    
    Returns:
        dict with test results including max diffs and pass/fail status
    """
    # Build equivalent command string (note: varlen tests use flash_attn_varlen_func directly)
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    causal_flag = "" if causal else "--no-causal"
    det_flag = "--deterministic" if deterministic else "--no-deterministic"
    # Note: test_mha.py doesn't have --varlen flag, this test uses flash_attn_varlen_func directly
    cmd = (f"python test_large_addr.py --kernel {co_name.replace('.co', '')}")
    
    print(f"\n{'='*70}")
    print(f"Test: {test_name} (VARLEN/GROUP MODE)")
    print(f"CO file: {co_name}")
    print(f"Command: {cmd}")
    print(f"Config: batch={batch_size}, heads={nheads}, seq_q={seqlen_q}, seq_k={seqlen_k}")
    print(f"        hdim_q={hdim_q}, hdim_v={hdim_v}, dtype={dtype}")
    print(f"        causal={causal}, causal_type={causal_type}, deterministic={deterministic}")
    print(f"        is_v3_atomic_fp32={is_v3_atomic_fp32} ({'A32' if is_v3_atomic_fp32 else 'A16'} kernel)")
    print(f"{'='*70}")
    
    device = "cuda"
    
    # For varlen, we create tensors in (total_tokens, nheads, hdim) format
    # with cu_seqlens arrays to mark batch boundaries
    total_q = batch_size * seqlen_q
    total_k = batch_size * seqlen_k
    
    # Create input tensors (packed format)
    q = torch.randn(total_q, nheads, hdim_q, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(total_k, nheads, hdim_q, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total_k, nheads, hdim_v, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(total_q, nheads, hdim_v, device=device, dtype=dtype)
    
    # Create cumulative sequence length arrays
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, seqlen_q, device=device, dtype=torch.int32)
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, seqlen_k, device=device, dtype=torch.int32)
    
    max_seqlen_q = seqlen_q
    max_seqlen_k = seqlen_k
    
    # Set window size for causal
    if causal:
        if causal_type == "bottom_right":
            window_size_left = seqlen_k - seqlen_q
            window_size_right = 0
        else:  # top_left
            window_size_left = -1
            window_size_right = 0
    else:
        window_size_left = -1
        window_size_right = -1
    
    # Run aiter varlen forward + backward
    try:
        out, softmax_lse, *_ = aiter.flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=None,
            causal=causal,
            window_size=(window_size_left, window_size_right),
            return_lse=True,  # Required for backward pass
            return_attn_probs=False,
            deterministic=deterministic,
            is_v3_atomic_fp32=is_v3_atomic_fp32,  # True=A32, False=A16
        )
        
        # Compute gradients
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        
    except Exception as e:
        import traceback
        error_msg = str(e) if str(e) else f"{type(e).__name__}"
        print(f"  ERROR: Aiter varlen failed with: {error_msg}")
        traceback.print_exc()
        return {"passed": False, "error": error_msg or "Unknown error", "co_name": co_name}
    
    # Run PyTorch reference (unpack, compute, compare)
    # Reshape to batch format for reference computation
    q_batch = q.reshape(batch_size, seqlen_q, nheads, hdim_q)
    k_batch = k.reshape(batch_size, seqlen_k, nheads, hdim_q)
    v_batch = v.reshape(batch_size, seqlen_k, nheads, hdim_v)
    dout_batch = dout.reshape(batch_size, seqlen_q, nheads, hdim_v)
    
    q_ref = q_batch.detach().clone().float().requires_grad_(True)
    k_ref = k_batch.detach().clone().float().requires_grad_(True)
    v_ref = v_batch.detach().clone().float().requires_grad_(True)
    
    # Simple attention reference in float32 for accuracy
    scale = 1.0 / (hdim_q ** 0.5)
    q_ref_t = q_ref.transpose(1, 2)  # B, H, S_q, D
    k_ref_t = k_ref.transpose(1, 2)  # B, H, S_k, D
    v_ref_t = v_ref.transpose(1, 2)  # B, H, S_k, D_v
    
    attn = torch.matmul(q_ref_t, k_ref_t.transpose(-2, -1)) * scale
    
    if causal:
        # Create causal mask
        if causal_type == "bottom_right":
            row_idx = torch.arange(seqlen_q, device=device).unsqueeze(1)
            col_idx = torch.arange(seqlen_k, device=device).unsqueeze(0)
            offset = seqlen_k - seqlen_q
            mask = col_idx <= row_idx + offset
        else:
            row_idx = torch.arange(seqlen_q, device=device).unsqueeze(1)
            col_idx = torch.arange(seqlen_k, device=device).unsqueeze(0)
            mask = col_idx <= row_idx
        
        attn = attn.masked_fill(~mask, float('-inf'))
    
    attn = torch.softmax(attn, dim=-1)
    out_ref = torch.matmul(attn, v_ref_t)
    out_ref = out_ref.transpose(1, 2)  # B, S_q, H, D_v
    
    dout_ref = dout_batch.clone().float()
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout_ref)
    
    # Convert back to original dtype and packed format for comparison
    # Use reshape instead of view since tensors may not be contiguous after transpose/grad
    dq_ref = dq_ref.to(dtype).reshape(total_q, nheads, hdim_q)
    dk_ref = dk_ref.to(dtype).reshape(total_k, nheads, hdim_q)
    dv_ref = dv_ref.to(dtype).reshape(total_k, nheads, hdim_v)
    
    # Compare results
    dq_diff = (dq - dq_ref).abs().max().item()
    dk_diff = (dk - dk_ref).abs().max().item()
    dv_diff = (dv - dv_ref).abs().max().item()
    
    # Tolerance based on dtype
    if dtype == torch.bfloat16:
        tol = 0.02  # bf16 has lower precision
    else:  # float16
        tol = 0.01
    
    passed = dq_diff <= tol and dk_diff <= tol and dv_diff <= tol
    
    status_q = '✓' if dq_diff <= tol else '✗'
    status_k = '✓' if dk_diff <= tol else '✗'
    status_v = '✓' if dv_diff <= tol else '✗'
    
    print(f"  dQ max diff: {dq_diff:.6f} {status_q}")
    print(f"  dK max diff: {dk_diff:.6f} {status_k}")
    print(f"  dV max diff: {dv_diff:.6f} {status_v}")
    print(f"  Result: {'PASSED' if passed else 'FAILED'}")
    
    # If failed, show mismatch details
    if not passed:
        for name, aiter_tensor, ref_tensor in [
            ("dQ", dq, dq_ref),
            ("dK", dk, dk_ref),
            ("dV", dv, dv_ref),
        ]:
            diff = (aiter_tensor - ref_tensor).abs()
            max_diff = diff.max().item()
            if max_diff > tol:
                max_idx = torch.unravel_index(torch.argmax(diff), diff.shape)
                coords = tuple(idx.item() for idx in max_idx)
                print(f"  {name} max diff at {coords}: aiter={aiter_tensor[max_idx].item():.6f}, ref={ref_tensor[max_idx].item():.6f}")
    
    return {
        "passed": passed,
        "dq_diff": dq_diff,
        "dk_diff": dk_diff,
        "dv_diff": dv_diff,
        "tol": tol,
        "co_name": co_name,
    }


def list_tests():
    """List all available tests from CSV."""
    configs = load_kernel_configs()
    
    print(f"\nFound {len(configs)} kernel configurations in CSV:")
    print("-" * 100)
    print(f"{'#':<4} {'dtype':<6} {'hdim_q':<8} {'hdim_v':<8} {'mask':<6} {'a32':<5} {'mode':<6} {'co_name'}")
    print("-" * 100)
    
    for i, config in enumerate(configs):
        mask_str = {0: "none", 1: "causal", 2: "cau_br"}.get(int(config['mask']), "?")
        a32_str = "a32" if int(config['atomic32']) else "a16"
        mode_str = "group" if int(config['mode']) else "normal"
        print(f"{i+1:<4} {config['dtype']:<6} {config['hdim_q']:<8} {config['hdim_v']:<8} "
              f"{mask_str:<6} {a32_str:<5} {mode_str:<6} {config['co_name']}")
    
    print("-" * 100)
    
    # Count by category
    mode0_count = sum(1 for c in configs if int(c['mode']) == 0)
    mode1_count = sum(1 for c in configs if int(c['mode']) == 1)
    print(f"\nNormal mode: {mode0_count}")
    print(f"Group mode (varlen API): {mode1_count}")
    print(f"Total: {mode0_count + mode1_count}")


def show_csv_mapping():
    """Show the CSV to kernel mapping."""
    configs = load_kernel_configs()
    hsa_path = get_hsa_path()
    
    print(f"\nHSA path: {hsa_path}")
    print(f"\nKernel mapping from fmha_bwd_dqdkdv.csv:")
    print("-" * 80)
    
    for config in configs:
        co_file = hsa_path / config['co_name']
        exists = "✓" if co_file.exists() else "✗"
        print(f"  {exists} {config['co_name']}")


def run_tests(kernel_filter: str = None, quick: bool = False):
    """Run all tests or filtered tests."""
    arch = get_device_arch()
    print(f"\nRunning on: {arch}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    configs = load_kernel_configs()
    
    if not configs:
        print("No kernel configurations found!")
        return 1
    
    results = {}
    passed = 0
    failed = 0
    skipped = 0
    
    for i, config in enumerate(configs):
        co_name = config['co_name']
        
        # Apply filter if specified
        if kernel_filter and kernel_filter not in co_name:
            continue
        
        # Get test parameters
        params = get_test_params_for_kernel(config)
        
        if params is None:
            print(f"\nSkipping {co_name} (unsupported configuration)")
            skipped += 1
            continue
        
        # For quick mode, use smaller sequences
        if quick:
            params['seqlen_k'] = min(params['seqlen_k'], 10000)
        
        # Create test name
        test_name = co_name.replace('.co', '').replace('bwd_', '')
        
        # Choose appropriate test function based on mode
        is_varlen = params.pop('is_varlen', False)
        test_func = run_mha_varlen_backward_test if is_varlen else run_mha_backward_test
        
        try:
            result = test_func(
                test_name=test_name,
                **params,
            )
            results[co_name] = result
            
            if result.get("passed", False):
                passed += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"\nTest {co_name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results[co_name] = {"passed": False, "error": str(e), "co_name": co_name}
            failed += 1
        
        # Clear GPU memory between tests
        torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total:   {passed + failed + skipped}")
    print(f"{'='*70}")
    
    if failed > 0:
        print("\nFailed tests:")
        for co_name, result in results.items():
            if not result.get("passed", False):
                error = result.get("error", "")
                if error:
                    print(f"  - {co_name}: {error}")
                else:
                    dq = result.get('dq_diff', None)
                    dk = result.get('dk_diff', None)
                    dv = result.get('dv_diff', None)
                    dq_str = f"{dq:.4f}" if isinstance(dq, (int, float)) else "?"
                    dk_str = f"{dk:.4f}" if isinstance(dk, (int, float)) else "?"
                    dv_str = f"{dv:.4f}" if isinstance(dv, (int, float)) else "?"
                    print(f"  - {co_name}: dQ={dq_str}, dK={dk_str}, dV={dv_str}")
        return 1
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Test 64-bit address fixes for FMHA backward kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_large_addr.py                      # Run all tests
  python test_large_addr.py --kernel hd128_bf16  # Filter by kernel name
  python test_large_addr.py --list               # List all kernels
  python test_large_addr.py --csv                # Show CSV mapping
  python test_large_addr.py --quick              # Quick mode (smaller sequences)
"""
    )
    parser.add_argument("--kernel", type=str, default=None,
                        help="Filter tests by kernel name (substring match)")
    parser.add_argument("--list", action="store_true",
                        help="List all available tests")
    parser.add_argument("--csv", action="store_true",
                        help="Show CSV to kernel mapping")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode with smaller sequences (for debugging)")
    args = parser.parse_args()
    
    if args.list:
        list_tests()
        return 0
    
    if args.csv:
        show_csv_mapping()
        return 0
    
    return run_tests(args.kernel, args.quick)


if __name__ == "__main__":
    sys.exit(main())
