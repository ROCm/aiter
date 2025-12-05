#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import run_perftest, checkAllclose, benchmark
import pandas as pd
import argparse

def torch_causal_conv1d_update_ref(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(-(width - 1), 0, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)

@benchmark()
def test_multiuser_streaming(dtype=torch.float32):
    """
    Scenario 1: Multi-User Real-time Streaming (matching C++ test_multiuser_streaming)
    - Batch: 4 users
    - Dim: 128
    - Width: 4
    - State Length: 16
    - SiLU: True
    - Steps: [1, 2, 3, 1, 2, 1] (variable-length chunks)
    - Mode: Non-Circular (cache_seqlens = None)
    """
    ret = {}
    
    # Configuration matching C++ Scenario 1
    batch = 4
    dim = 128
    width = 4
    state_len = 16
    use_silu = True
    step_seqlens = [1, 2, 3, 1, 2, 1]
    
    # Initialize weights and bias (shared across all steps)
    torch.manual_seed(42)
    weight = torch.randn(dim, width, dtype=dtype, device="cuda")
    bias = torch.randn(dim, dtype=dtype, device="cuda")
    
    # Initialize conv_state to zeros (persistent across steps)
    conv_state = torch.zeros(batch, dim, state_len, dtype=dtype, device="cuda")
    conv_state_ref = torch.zeros(batch, dim, state_len, dtype=dtype, device="cuda")
    
    # Non-circular mode: cache_seqlens = None
    cache_seqlens_empty = torch.empty(0, dtype=torch.int32, device="cuda")
    conv_state_indices_empty = torch.empty(0, dtype=torch.int32, device="cuda")
    
    activation = "silu"
    total_tokens = 0
    total_us = 0
    max_err = 0.0
    all_passed = True
    
    # Simulate streaming inference with multiple steps
    for step_idx, seqlen in enumerate(step_seqlens):
        # Generate new input for this step
        x = torch.randn(batch, dim, seqlen, dtype=dtype, device="cuda")
        x_ref = x.clone()
        out = torch.empty(batch, dim, seqlen, dtype=dtype, device="cuda")
        
        # Compute reference using PyTorch
        ref = torch_causal_conv1d_update_ref(
            x_ref, conv_state_ref, weight, bias, activation=activation, cache_seqlens=None
        )
        
        # Run AIter implementation (Non-Circular mode)
        _, us_aiter = run_perftest(
            aiter.causal_conv1d_update,
            x,
            conv_state,
            weight,
            bias,
            out,
            use_silu,
            cache_seqlens_empty,
            conv_state_indices_empty,
        )
        
        # Check accuracy
        err = checkAllclose(ref, out, rtol=1e-3, atol=1e-3)
        max_err = max(max_err, err)
        
        if err > 0.01:
            all_passed = False
            
        total_tokens += seqlen
        total_us += us_aiter
    
    # Calculate aggregate metrics
    total_bytes = 0
    for seqlen in step_seqlens:
        bytes_read = batch * dim * seqlen * dtype.itemsize + batch * dim * state_len * dtype.itemsize + \
                     dim * width * dtype.itemsize + dim * dtype.itemsize
        bytes_write = batch * dim * seqlen * dtype.itemsize + batch * dim * state_len * dtype.itemsize
        total_bytes += bytes_read + bytes_write
    
    total_flops = sum(batch * dim * seqlen * width * 2 for seqlen in step_seqlens)
    
    ret["us"] = total_us / len(step_seqlens)
    ret["TB/s"] = total_bytes / total_us / 1e6
    ret["err"] = max_err
    ret["GFLOPS"] = total_flops / (total_us * 1e3)
    ret["total_tokens"] = total_tokens
    ret["num_steps"] = len(step_seqlens)
    ret["passed"] = all_passed
    ret["scenario"] = "scenario1_multiuser_streaming"
    
    return ret


@benchmark()
def test_batch_text_generation(dtype=torch.float32):
    """
    Scenario 2: Batch Text Generation (matching C++ test_batch_text_generation)
    - Batch: 8 sequences
    - Dim: 128
    - Width: 4
    - State Length: 16
    - SiLU: True
    - Steps: [1, 2, 3, 2, 1] (variable tokens per step)
    - Mode: Non-Circular (cache_seqlens = None)
    """
    ret = {}
    
    # Configuration matching C++ Scenario 2
    batch = 8
    dim = 128
    width = 4
    state_len = 16
    use_silu = True
    step_seqlens = [1, 2, 3, 2, 1]
    
    # Initialize weights and bias
    torch.manual_seed(42)
    weight = torch.randn(dim, width, dtype=dtype, device="cuda")
    bias = torch.randn(dim, dtype=dtype, device="cuda")
    
    # Initialize conv_state to zeros
    conv_state = torch.zeros(batch, dim, state_len, dtype=dtype, device="cuda")
    conv_state_ref = torch.zeros(batch, dim, state_len, dtype=dtype, device="cuda")
    
    # Non-circular mode
    cache_seqlens_empty = torch.empty(0, dtype=torch.int32, device="cuda")
    conv_state_indices_empty = torch.empty(0, dtype=torch.int32, device="cuda")
    
    activation = "silu"
    total_tokens = 0
    total_us = 0
    max_err = 0.0
    all_passed = True
    
    for step_idx, seqlen in enumerate(step_seqlens):
        x = torch.randn(batch, dim, seqlen, dtype=dtype, device="cuda")
        x_ref = x.clone()
        out = torch.empty(batch, dim, seqlen, dtype=dtype, device="cuda")
        
        ref = torch_causal_conv1d_update_ref(
            x_ref, conv_state_ref, weight, bias, activation=activation, cache_seqlens=None
        )
        
        _, us_aiter = run_perftest(
            aiter.causal_conv1d_update,
            x,
            conv_state,
            weight,
            bias,
            out,
            use_silu,
            cache_seqlens_empty,
            conv_state_indices_empty,
        )
        
        err = checkAllclose(ref, out, rtol=1e-3, atol=1e-3)
        max_err = max(max_err, err)
        
        if err > 0.01:
            all_passed = False
            
        total_tokens += seqlen
        total_us += us_aiter
    
    total_bytes = 0
    for seqlen in step_seqlens:
        bytes_read = batch * dim * seqlen * dtype.itemsize + batch * dim * state_len * dtype.itemsize + \
                     dim * width * dtype.itemsize + dim * dtype.itemsize
        bytes_write = batch * dim * seqlen * dtype.itemsize + batch * dim * state_len * dtype.itemsize
        total_bytes += bytes_read + bytes_write
    
    total_flops = sum(batch * dim * seqlen * width * 2 for seqlen in step_seqlens)
    
    ret["us"] = total_us / len(step_seqlens)
    ret["TB/s"] = total_bytes / total_us / 1e6
    ret["err"] = max_err
    ret["GFLOPS"] = total_flops / (total_us * 1e3)
    ret["total_tokens"] = total_tokens
    ret["num_steps"] = len(step_seqlens)
    ret["passed"] = all_passed
    ret["scenario"] = "scenario2_batch_text_generation"
    
    return ret


@benchmark()
def test_circular_buffer_performance(dtype=torch.float32):
    """
    Scenario 3: High-Performance Circular Buffer (matching C++ test_circular_buffer_performance)
    - Batch: 4 concurrent requests
    - Dim: 128
    - Width: 4
    - State Length: 16
    - SiLU: True
    - Steps: [1, 2, 1, 3, 1]
    - Mode: Circular Buffer (cache_seqlens provided, O(1) update)
    """
    ret = {}
    
    # Configuration matching C++ Scenario 3
    batch = 4
    dim = 128
    width = 4
    state_len = 16
    use_silu = True
    step_seqlens = [1, 2, 1, 3, 1]
    
    # Initialize weights and bias
    torch.manual_seed(42)
    weight = torch.randn(dim, width, dtype=dtype, device="cuda")
    bias = torch.randn(dim, dtype=dtype, device="cuda")
    
    # Initialize conv_state and cache_seqlens
    conv_state = torch.zeros(batch, dim, state_len, dtype=dtype, device="cuda")
    conv_state_ref = torch.zeros(batch, dim, state_len, dtype=dtype, device="cuda")
    cache_seqlens = torch.zeros(batch, dtype=torch.int32, device="cuda")
    cache_seqlens_ref = torch.zeros(batch, dtype=torch.int32, device="cuda")
    
    conv_state_indices_empty = torch.empty(0, dtype=torch.int32, device="cuda")
    
    activation = "silu"
    total_tokens = 0
    total_us = 0
    max_err = 0.0
    all_passed = True
    
    # Simulate streaming with circular buffer
    for step_idx, seqlen in enumerate(step_seqlens):
        x = torch.randn(batch, dim, seqlen, dtype=dtype, device="cuda")
        x_ref = x.clone()
        out = torch.empty(batch, dim, seqlen, dtype=dtype, device="cuda")
        
        # Compute reference with circular buffer
        ref = torch_causal_conv1d_update_ref(
            x_ref, conv_state_ref, weight, bias, activation=activation, cache_seqlens=cache_seqlens_ref
        )
        
        # Run AIter implementation with circular buffer
        _, us_aiter = run_perftest(
            aiter.causal_conv1d_update,
            x,
            conv_state,
            weight,
            bias,
            out,
            use_silu,
            cache_seqlens,
            conv_state_indices_empty,
        )
        
        err = checkAllclose(ref, out, rtol=1e-3, atol=1e-3)
        max_err = max(max_err, err)
        
        if err > 0.01:
            all_passed = False
        
        # Update cache_seqlens for next step
        cache_seqlens += seqlen
        cache_seqlens_ref += seqlen
        
        total_tokens += seqlen
        total_us += us_aiter
    
    total_bytes = 0
    for seqlen in step_seqlens:
        bytes_read = batch * dim * seqlen * dtype.itemsize + batch * dim * state_len * dtype.itemsize + \
                     dim * width * dtype.itemsize + dim * dtype.itemsize + batch * 4
        bytes_write = batch * dim * seqlen * dtype.itemsize + batch * dim * state_len * dtype.itemsize
        total_bytes += bytes_read + bytes_write
    
    total_flops = sum(batch * dim * seqlen * width * 2 for seqlen in step_seqlens)
    
    ret["us"] = total_us / len(step_seqlens)
    ret["TB/s"] = total_bytes / total_us / 1e6
    ret["err"] = max_err
    ret["GFLOPS"] = total_flops / (total_us * 1e3)
    ret["total_tokens"] = total_tokens
    ret["num_steps"] = len(step_seqlens)
    ret["passed"] = all_passed
    ret["scenario"] = "scenario3_circular_buffer"
    
    return ret


@benchmark()
def test_causal_conv1d_update_continuous_batching(batch, dim, seqlen, width, state_len, dtype, use_silu=False):
    """
    Test continuous batching mode with conv_state_indices.
    Note: This is a single-step test, not streaming.
    """
    ret = {}
    
    batch_state = batch + 2  # Physical state buffer larger than logical batch
    
    x = torch.randn(batch, dim, seqlen, dtype=dtype, device="cuda")
    conv_state = torch.randn(batch_state, dim, state_len, dtype=dtype, device="cuda")
    conv_state_ref = conv_state.clone()
    weight = torch.randn(dim, width, dtype=dtype, device="cuda")
    bias = torch.randn(dim, dtype=dtype, device="cuda")
    out = torch.empty(batch, dim, seqlen, dtype=dtype, device="cuda")
    
    # Create indices mapping logical batch to physical state
    # Include one padding token (negative index)
    conv_state_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
    if batch > 0:
        conv_state_indices[0] = -1  # Mark first sequence as padding
    
    # Compute reference - use first batch entries from state
    # (Reference doesn't support conv_state_indices directly)
    activation = None if not use_silu else "silu"
    conv_state_for_ref = conv_state_ref[:batch]
    ref = torch_causal_conv1d_update_ref(
        x, conv_state_for_ref, weight, bias, activation=activation, cache_seqlens=None
    )
    
    # Run AIter implementation
    cache_seqlens_empty = torch.empty(0, dtype=torch.int32, device="cuda")
    
    _, us_aiter = run_perftest(
        aiter.causal_conv1d_update,
        x,
        conv_state,
        weight,
        bias,
        out,
        use_silu,
        cache_seqlens_empty,
        conv_state_indices,
    )
    
    err = checkAllclose(ref, out, rtol=1e-3, atol=1e-3)
    
    bytes_read = x.nbytes + conv_state.nbytes + weight.nbytes + bias.nbytes + conv_state_indices.nbytes
    bytes_write = out.nbytes + conv_state.nbytes
    total_bytes = bytes_read + bytes_write
    
    ret["us"] = us_aiter
    ret["TB/s"] = total_bytes / us_aiter / 1e6
    ret["err"] = err
    ret["GFLOPS"] = (batch * dim * seqlen * width * 2) / (us_aiter * 1e3)
    
    return ret


@benchmark()
def test_causal_conv1d_update_no_bias(batch, dim, seqlen, width, state_len, dtype, use_silu=False):
    """Test without bias (single-step test)."""
    ret = {}
    
    x = torch.randn(batch, dim, seqlen, dtype=dtype, device="cuda")
    conv_state = torch.randn(batch, dim, state_len, dtype=dtype, device="cuda")
    conv_state_ref = conv_state.clone()
    weight = torch.randn(dim, width, dtype=dtype, device="cuda")
    bias = torch.empty(0, dtype=dtype, device="cuda")
    out = torch.empty(batch, dim, seqlen, dtype=dtype, device="cuda")
    
    activation = None if not use_silu else "silu"
    ref = torch_causal_conv1d_update_ref(
        x, conv_state_ref, weight, None, activation=activation, cache_seqlens=None
    )
    
    _, us_aiter = run_perftest(
        aiter.causal_conv1d_update,
        x,
        conv_state,
        weight,
        bias,
        out,
        use_silu,
        torch.empty(0, dtype=torch.int32, device="cuda"),
        torch.empty(0, dtype=torch.int32, device="cuda"),
    )
    
    err = checkAllclose(ref, out, rtol=1e-3, atol=1e-3)
    
    bytes_read = x.nbytes + conv_state.nbytes + weight.nbytes
    bytes_write = out.nbytes + conv_state.nbytes
    total_bytes = bytes_read + bytes_write
    
    ret["us"] = us_aiter
    ret["TB/s"] = total_bytes / us_aiter / 1e6
    ret["err"] = err
    ret["GFLOPS"] = (batch * dim * seqlen * width * 2) / (us_aiter * 1e3)
    
    return ret


# Test configurations matching C++ test_causal_conv1d_update.cpp scenarios

# Three main scenarios (matching C++ exactly)
# Scenario 1: test_multiuser_streaming - Non-Circular mode
# Scenario 2: test_batch_text_generation - Non-Circular mode  
# Scenario 3: test_circular_buffer_performance - Circular mode

# For optional single-step tests (continuous batching, no_bias)
l_dtype = ["float32"]
l_batch = [4, 8]
l_dim = [128]
l_seqlen = [1, 2, 3]
l_width = [4]
l_state_len = [16]
l_use_silu = [True]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Causal Conv1D Update test configuration (matching C++ test flow)",
)
parser.add_argument("--scenario", type=str, default="all",
                   choices=["all", "scenario1", "scenario2", "scenario3"],
                   help="Test scenario: scenario1/scenario2/scenario3/all")
parser.add_argument("--batch", type=int, default=None, help="Override batch size")
parser.add_argument("--dim", type=int, default=None, help="Override dim")
parser.add_argument("--seqlen", type=int, default=None, help="Override seqlen (for single-step tests)")
parser.add_argument("--width", type=int, default=None, help="Override width")
parser.add_argument("--state_len", type=int, default=None, help="Override state_len")
parser.add_argument("--dtype", type=str, default="float32", help="Data type")
parser.add_argument("--mode", type=str, default=None, 
                   choices=["continuous", "no_bias"],
                   help="Single-step test mode: continuous/no_bias")
parser.add_argument("--use_silu", action="store_true")
parser.add_argument("--no_silu", action="store_true")
parser.add_argument("--verbose", action="store_true")


def main():
    args = parser.parse_args()
    
    results = []
    
    print("=" * 80)
    print("Testing AIter Causal Conv1D Update Implementation")
    print("Matching C++ test_causal_conv1d_update.cpp flow")
    print("=" * 80)
    
    dtype = getattr(torch, args.dtype)
    
    # Run the three main scenarios (matching C++ test structure)
    scenarios = {
        "scenario1": {
            "name": "Scenario 1: Multi-User Real-time Streaming",
            "func": test_multiuser_streaming,
            "description": "4 users, Non-Circular mode, steps=[1,2,3,1,2,1]"
        },
        "scenario2": {
            "name": "Scenario 2: Batch Text Generation",
            "func": test_batch_text_generation,
            "description": "8 sequences, Non-Circular mode, steps=[1,2,3,2,1]"
        },
        "scenario3": {
            "name": "Scenario 3: High-Performance Circular Buffer",
            "func": test_circular_buffer_performance,
            "description": "4 concurrent requests, Circular mode, steps=[1,2,1,3,1]"
        }
    }
    
    # Determine which scenarios to run
    if args.scenario == "all":
        scenarios_to_run = ["scenario1", "scenario2", "scenario3"]
    else:
        scenarios_to_run = [args.scenario]
    
    # Run each scenario
    for scenario_key in scenarios_to_run:
        scenario = scenarios[scenario_key]
        
        print(f"\n{'='*80}")
        print(f"{scenario['name']}")
        print(f"  {scenario['description']}")
        print(f"{'='*80}")
        
        try:
            result = scenario["func"](dtype=dtype)
            results.append(result)
            
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            print(f"\n{status}")
            print(f"  Total tokens: {result['total_tokens']}")
            print(f"  Num steps: {result['num_steps']}")
            print(f"  Avg time per step: {result['us']:.2f} us")
            print(f"  Total bandwidth: {result['TB/s']:.2f} TB/s")
            print(f"  Total GFLOPS: {result['GFLOPS']:.2f}")
            print(f"  Max error: {result['err']:.2e}")
            
        except Exception as e:
            print(f"\n❌ Failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Optional: Run single-step tests for continuous batching and no_bias
    if args.mode:
        print(f"\n{'='*80}")
        print(f"Single-Step Tests: {args.mode}")
        print(f"{'='*80}")
        
        batches = [args.batch] if args.batch else l_batch
        dims = [args.dim] if args.dim else l_dim
        seqlens = [args.seqlen] if args.seqlen else l_seqlen
        widths = [args.width] if args.width else l_width
        state_lens = [args.state_len] if args.state_len else l_state_len
        
        if args.use_silu:
            use_silus = [True]
        elif args.no_silu:
            use_silus = [False]
        else:
            use_silus = l_use_silu
        
        for batch in batches:
            for dim in dims:
                for seqlen in seqlens:
                    for width in widths:
                        for state_len in state_lens:
                            if state_len < width - 1:
                                continue
                            for use_silu in use_silus:
                                test_name = f"{args.mode}_B{batch}_D{dim}_S{seqlen}_W{width}_ST{state_len}"
                                if use_silu:
                                    test_name += "_silu"
                                
                                print(f"\n{'-'*60}")
                                print(f"Test: {test_name}")
                                print(f"{'-'*60}")
                                
                                try:
                                    if args.mode == "continuous":
                                        result = test_causal_conv1d_update_continuous_batching(
                                            batch, dim, seqlen, width, state_len, dtype, use_silu
                                        )
                                    elif args.mode == "no_bias":
                                        result = test_causal_conv1d_update_no_bias(
                                            batch, dim, seqlen, width, state_len, dtype, use_silu
                                        )
                                    
                                    result["test"] = test_name
                                    result["mode"] = args.mode
                                    result["batch"] = batch
                                    result["dim"] = dim
                                    result["seqlen"] = seqlen
                                    result["width"] = width
                                    result["state_len"] = state_len
                                    result["use_silu"] = use_silu
                                    results.append(result)
                                    
                                    print(f"✅ Time: {result['us']:.2f} us")
                                    print(f"✅ Bandwidth: {result['TB/s']:.2f} TB/s")
                                    print(f"✅ GFLOPS: {result['GFLOPS']:.2f}")
                                    print(f"✅ Error: {result['err']:.2e}")
                                except Exception as e:
                                    print(f"❌ Failed: {e}")
                                    if args.verbose:
                                        import traceback
                                        traceback.print_exc()
    
    # Create summary
    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
        
        output_file = "causal_conv1d_update_test_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")
        
        print("\n" + "=" * 80)
        print("STATISTICS")
        print("=" * 80)
        print(f"Total tests: {len(results)}")
        if "passed" in df.columns:
            print(f"Tests passed: {df['passed'].sum()}/{len(results)}")
        if "err" in df.columns:
            print(f"Max error: {df['err'].max():.2e}")
            print(f"Tests with err < 1e-2: {(df['err'] < 1e-2).sum()}/{len(results)}")
        if "us" in df.columns:
            print(f"Average time: {df['us'].mean():.2f} us")
        if "TB/s" in df.columns:
            print(f"Average bandwidth: {df['TB/s'].mean():.2f} TB/s")
        if "GFLOPS" in df.columns:
            print(f"Average GFLOPS: {df['GFLOPS'].mean():.2f}")
    
    print("\n" + "=" * 80)
    print("Test flow matches C++ test_causal_conv1d_update.cpp")
    print("  ✓ Scenario 1: test_multiuser_streaming (Non-Circular)")
    print("  ✓ Scenario 2: test_batch_text_generation (Non-Circular)")
    print("  ✓ Scenario 3: test_circular_buffer_performance (Circular)")
    print("=" * 80)


if __name__ == "__main__":
    main()

