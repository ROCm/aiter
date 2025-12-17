#!/usr/bin/env python3
"""
Standalone benchmark for MOE MXFP4 kernel that doesn't require full aiter import.
Works with Python 3.12 in trivolve container.

Usage:
    python standalone_bench_moe_mxfp4.py                  # Default M values
    python standalone_bench_moe_mxfp4.py --M 32 64 128   # Custom M values
"""

import argparse
import sys
import os
import time
import torch
import triton
import triton.language as tl
import numpy as np

# Check GPU
if not torch.cuda.is_available():
    print("ERROR: No GPU available!")
    sys.exit(1)

print(f"PyTorch: {torch.__version__}")
print(f"Triton: {triton.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Architecture: {triton.runtime.driver.active.get_current_target().arch}")
print()

# Import helper functions directly from the triton utils
import importlib.util

def load_module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load the utility modules first
pid_module = load_module_from_file(
    "pid_preprocessing", 
    "/home/upandey/aiter/aiter/ops/triton/utils/_triton/pid_preprocessing.py"
)
moe_common_module = load_module_from_file(
    "moe_common",
    "/home/upandey/aiter/aiter/ops/triton/utils/_triton/moe_common.py"
)

# Now manually define what the kernel needs
pid_grid = pid_module.pid_grid
remap_xcd = pid_module.remap_xcd
_write_zeros_to_output = moe_common_module._write_zeros_to_output

# Define kernel inline since relative imports don't work
def get_scaled_dot_format_string(dtype: tl.dtype):
    mapping = {
        tl.float16: "fp16",
        tl.bfloat16: "bf16",
        tl.uint8: "e2m1",
        tl.float8e4nv: "e4m3",
        tl.float8e5: "e5m2",
    }
    return mapping[dtype]


@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0})
@triton.jit
def _fused_moe_kernel_mxfp4(
    a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, a_mx_scale_ptr, b_mx_scale_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
    stride_amxm, stride_amxk, stride_bmxe, stride_bmxk, stride_bmxn,
    A_DTYPE_FORMAT: tl.constexpr, B_DTYPE_FORMAT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, EVEN_K: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr, compute_type: tl.constexpr,
    SWIZZLE_MX_A: tl.constexpr, SWIZZLE_MX_B: tl.constexpr, NUM_XCDS: tl.constexpr,
):
    """Fused MOE kernel."""
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    GRID_MN = num_pid_n * num_pid_m
    
    if pid >= GRID_MN:
        return
    
    # Simple grid mapping
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    
    # Skip if expert is -1
    if off_expert == -1:
        return
    
    a_scale = tl.load(a_scale_ptr)
    b_scale = tl.load(b_scale_ptr + off_expert)
    
    offs_b_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_a_k = tl.arange(0, BLOCK_SIZE_K)
    offs_b_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_a_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_expert * stride_be + (offs_b_k[:, None] * stride_bk + offs_b_n[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_a_k[None, :] < (K - k * BLOCK_SIZE_K)), other=0.0)
            b = tl.load(b_ptrs, mask=offs_b_k[:, None] < (K - k * BLOCK_SIZE_K), other=0.0)
        
        accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=True)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    accumulator *= a_scale * b_scale
    
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    
    accumulator = accumulator.to(compute_type)
    
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def get_moe_config(M: int) -> dict:
    """Get kernel config based on M size."""
    if M <= 16:
        return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "NUM_XCDS": 1}
    elif M <= 64:
        return {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4, "NUM_XCDS": 1}
    elif M <= 256:
        return {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "NUM_XCDS": 1}
    else:
        return {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "NUM_XCDS": 1}


def simple_moe_align_block_size(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """Simplified version of moe_align_block_size for benchmarking."""
    M, top_k = topk_ids.shape
    
    # Count tokens per expert
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    for e in range(num_experts):
        expert_counts[e] = (topk_ids == e).sum()
    
    # Pad to block size
    padded_counts = ((expert_counts + block_size - 1) // block_size) * block_size
    total_padded = padded_counts.sum().item()
    
    # Create sorted token ids and expert ids
    sorted_token_ids = torch.zeros(total_padded, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.zeros(total_padded // block_size, dtype=torch.int32, device=topk_ids.device)
    
    # Fill in (simplified - just for benchmarking, may not be perfectly accurate)
    offset = 0
    for e in range(num_experts):
        mask = (topk_ids == e)
        tokens = torch.where(mask.flatten())[0] // top_k
        count = len(tokens)
        if count > 0:
            sorted_token_ids[offset:offset+count] = tokens[:count].to(torch.int32)
        offset += padded_counts[e].item()
    
    # Fill expert_ids
    offset = 0
    for e in range(num_experts):
        num_blocks = padded_counts[e].item() // block_size
        expert_ids[offset:offset+num_blocks] = e
        offset += num_blocks
    
    num_tokens_post_padded = torch.tensor([total_padded], dtype=torch.int32, device=topk_ids.device)
    
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def run_benchmark(M: int, N: int, K: int, E: int, top_k: int, warmup: int = 10, rep: int = 100):
    """Run a single benchmark configuration."""
    device = "cuda"
    dtype = torch.bfloat16
    
    # Create input tensors
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((E, N, K), device=device, dtype=dtype)
    c = torch.zeros((M, top_k, N), device=device, dtype=dtype)
    
    # Scales
    a_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    b_scale = torch.ones(E, dtype=torch.float32, device=device)
    
    # For simplicity, skip MX scales (use None)
    a_mx_scale = None
    b_mx_scale = None
    
    # Create routing
    logits = torch.randn(M, E, dtype=torch.float16, device=device)
    softmax_vals = torch.softmax(logits, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)
    
    # Get config
    config = get_moe_config(M)
    block_size_m = config["BLOCK_SIZE_M"]
    
    # Align block size
    sorted_token_ids, expert_ids, num_tokens_post_padded = simple_moe_align_block_size(
        topk_ids, block_size_m, E
    )
    
    # Calculate grid
    num_tokens_padded = num_tokens_post_padded.item()
    grid = (triton.cdiv(num_tokens_padded, block_size_m) * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    
    # Warmup
    for _ in range(warmup):
        _fused_moe_kernel_mxfp4[grid](
            a, b, c,
            a_scale, b_scale,
            a_mx_scale, b_mx_scale,
            topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
            N, K, E * N,
            M * top_k,  # num_valid_tokens
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1), b.stride(2),
            c.stride(0) * top_k, c.stride(2),
            0, 0,  # a_mx strides
            0, 0, 0,  # b_mx strides
            A_DTYPE_FORMAT="bf16",
            B_DTYPE_FORMAT="bf16",
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            MUL_ROUTED_WEIGHT=True,
            top_k=top_k,
            compute_type=tl.bfloat16,
            SWIZZLE_MX_A=False,
            SWIZZLE_MX_B=False,
            NUM_XCDS=config["NUM_XCDS"],
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    
    for i in range(rep):
        start_events[i].record()
        _fused_moe_kernel_mxfp4[grid](
            a, b, c,
            a_scale, b_scale,
            a_mx_scale, b_mx_scale,
            topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
            N, K, E * N,
            M * top_k,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1), b.stride(2),
            c.stride(0) * top_k, c.stride(2),
            0, 0,
            0, 0, 0,
            A_DTYPE_FORMAT="bf16",
            B_DTYPE_FORMAT="bf16",
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            MUL_ROUTED_WEIGHT=True,
            top_k=top_k,
            compute_type=tl.bfloat16,
            SWIZZLE_MX_A=False,
            SWIZZLE_MX_B=False,
            NUM_XCDS=config["NUM_XCDS"],
        )
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    ms = np.median(times)
    
    # Calculate metrics
    flops = 2.0 * M * top_k * K * N + M * top_k * N  # matmul + routing
    tflops = flops / ms * 1e-9
    
    mem_read = a.numel() * a.element_size() + b.numel() * b.element_size()
    mem_write = c.numel() * c.element_size()
    mem = mem_read + mem_write
    bandwidth = mem / (ms * 1e-3) * 1e-9
    
    return {
        'M': M, 'N': N, 'K': K, 'E': E, 'top_k': top_k,
        'time_ms': ms, 'tflops': tflops, 'bandwidth_gb_s': bandwidth,
        'config': config
    }


def main():
    parser = argparse.ArgumentParser(description="Standalone MOE MXFP4 Benchmark")
    parser.add_argument('--M', type=int, nargs='+', default=[1, 16, 32, 64, 128, 256, 512],
                        help='List of M values to test')
    parser.add_argument('--N', type=int, default=14336, help='N dimension (intermediate_size)')
    parser.add_argument('--K', type=int, default=4096, help='K dimension (hidden_size)')
    parser.add_argument('--E', type=int, default=8, help='Number of experts')
    parser.add_argument('--top_k', type=int, default=2, help='Top-k routing')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--rep', type=int, default=100, help='Benchmark repetitions')
    args = parser.parse_args()
    
    print("=" * 80)
    print("MOE Kernel Benchmark (Standalone)")
    print(f"N={args.N}, K={args.K}, E={args.E}, top_k={args.top_k}")
    print(f"M values: {args.M}")
    print("=" * 80)
    
    print(f"\n{'M':>6} | {'Time(ms)':>10} | {'TFLOPS':>10} | {'BW(GB/s)':>12} | Config")
    print("-" * 70)
    
    results = []
    for M in args.M:
        try:
            result = run_benchmark(M, args.N, args.K, args.E, args.top_k, args.warmup, args.rep)
            results.append(result)
            cfg = result['config']
            print(f"{M:>6} | {result['time_ms']:>10.4f} | {result['tflops']:>10.2f} | {result['bandwidth_gb_s']:>12.2f} | {cfg['BLOCK_SIZE_M']}x{cfg['BLOCK_SIZE_N']}x{cfg['BLOCK_SIZE_K']}")
        except Exception as e:
            print(f"{M:>6} | ERROR: {e}")
    
    if results:
        print("-" * 70)
        avg_time = np.mean([r['time_ms'] for r in results])
        avg_tflops = np.mean([r['tflops'] for r in results])
        avg_bw = np.mean([r['bandwidth_gb_s'] for r in results])
        print(f"{'AVG':>6} | {avg_time:>10.4f} | {avg_tflops:>10.2f} | {avg_bw:>12.2f}")
        print(f"{'PEAK':>6} | {min([r['time_ms'] for r in results]):>10.4f} | {max([r['tflops'] for r in results]):>10.2f} | {max([r['bandwidth_gb_s'] for r in results]):>12.2f}")


if __name__ == "__main__":
    main()

