#!/usr/bin/env python3
"""
Simple MOE benchmark for Triton kernel optimization.
Shows timing across multiple M values for kernel optimization comparison.
"""

import argparse
import torch
import triton
import triton.language as tl
import numpy as np

# Check GPU
if not torch.cuda.is_available():
    print("ERROR: No GPU available!")
    exit(1)

print(f"PyTorch: {torch.__version__}")
print(f"Triton: {triton.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Arch: {triton.runtime.driver.active.get_current_target().arch}")
print()

# ============================================================================
# Optimized MOE Kernel
# ============================================================================

@triton.jit
def _optimized_moe_kernel(
    a_ptr, b_ptr, c_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, num_valid_tokens,
    stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr, NUM_XCDS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    GRID_MN = num_pid_m * num_pid_n
    
    if pid >= GRID_MN:
        return
    
    # XCD remapping for better load balancing
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    
    # Grouped ordering for L2 cache reuse
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    
    off_expert = tl.load(expert_ids_ptr + pid_m)
    
    if off_expert < 0:
        return
    
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
    
    a_ptrs = a_ptr + (offs_token // top_k)[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + off_expert * stride_be + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    if EVEN_K:
        for k in range(0, K // BLOCK_SIZE_K):
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc=acc)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            acc = tl.dot(a, b, acc=acc)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc = acc * moe_weight[:, None]
    
    acc = acc.to(tl.bfloat16)
    
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

# ============================================================================
# Benchmark helpers
# ============================================================================

def moe_align_block_size(topk_ids, block_size, num_experts):
    M, top_k = topk_ids.shape
    device = topk_ids.device
    flat_topk = topk_ids.flatten()
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for e in range(num_experts):
        expert_counts[e] = (flat_topk == e).sum()
    padded_counts = ((expert_counts + block_size - 1) // block_size) * block_size
    total_padded = padded_counts.sum().item()
    sorted_token_ids = torch.full((total_padded,), M * top_k, dtype=torch.int32, device=device)
    expert_ids = torch.zeros(total_padded // block_size, dtype=torch.int32, device=device)
    offset = 0
    for e in range(num_experts):
        mask = (flat_topk == e)
        indices = mask.nonzero(as_tuple=False).flatten()
        count = indices.shape[0]
        if count > 0:
            sorted_token_ids[offset:offset + count] = indices.int()
        num_blocks = padded_counts[e].item() // block_size
        for b in range(num_blocks):
            expert_ids[offset // block_size + b] = e
        offset += padded_counts[e].item()
    num_tokens_post_padded = torch.tensor([total_padded], dtype=torch.int32, device=device)
    return sorted_token_ids, expert_ids, num_tokens_post_padded

def get_optimal_config(M):
    """Get optimal configuration based on M value."""
    if M <= 64:
        return {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}
    elif M <= 128:
        return {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}
    else:
        return {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 2}

def benchmark_moe_triton(M, N, K, E, top_k, warmup=25, rep=100):
    device = "cuda"
    dtype = torch.bfloat16
    
    config = get_optimal_config(M)
    
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((E, N, K), device=device, dtype=dtype)
    c = torch.zeros((M * top_k, N), device=device, dtype=dtype)
    
    routing_weights = torch.softmax(torch.randn(M, E, device=device, dtype=torch.float32), dim=1)
    topk_weights, topk_ids = torch.topk(routing_weights, k=top_k, dim=1)
    topk_weights = topk_weights.flatten().to(dtype)
    
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E)
    
    num_valid_tokens = M * top_k
    total_blocks_m = num_tokens_post_padded.item() // config["BLOCK_SIZE_M"]
    total_blocks_n = (N + config["BLOCK_SIZE_N"] - 1) // config["BLOCK_SIZE_N"]
    grid = (total_blocks_m * total_blocks_n,)
    EVEN_K = (K % config["BLOCK_SIZE_K"] == 0)
    
    # Warmup
    for _ in range(warmup):
        _optimized_moe_kernel[grid](a, b, c, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
            N, K, num_valid_tokens, a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2),
            c.stride(0), c.stride(1), BLOCK_SIZE_M=config["BLOCK_SIZE_M"], BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"], GROUP_SIZE_M=config["GROUP_SIZE_M"], MUL_ROUTED_WEIGHT=True,
            top_k=top_k, NUM_XCDS=8, EVEN_K=EVEN_K, num_warps=config["num_warps"], num_stages=config["num_stages"])
    torch.cuda.synchronize()
    
    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    
    for i in range(rep):
        start_events[i].record()
        _optimized_moe_kernel[grid](a, b, c, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
            N, K, num_valid_tokens, a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2),
            c.stride(0), c.stride(1), BLOCK_SIZE_M=config["BLOCK_SIZE_M"], BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"], GROUP_SIZE_M=config["GROUP_SIZE_M"], MUL_ROUTED_WEIGHT=True,
            top_k=top_k, NUM_XCDS=8, EVEN_K=EVEN_K, num_warps=config["num_warps"], num_stages=config["num_stages"])
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    ms = np.median(times)
    
    flops = 2.0 * M * top_k * K * N + M * top_k * N
    tflops = flops / ms * 1e-9
    
    mem_read = a.numel() * a.element_size() + b.numel() * b.element_size()
    mem_write = c.numel() * c.element_size()
    mem = mem_read + mem_write
    bandwidth = mem / (ms * 1e-3) * 1e-9
    
    return {'M': M, 'time_ms': ms, 'tflops': tflops, 'bandwidth_gb_s': bandwidth, 'config': config}

def main():
    parser = argparse.ArgumentParser(description="Simple MOE Benchmark")
    parser.add_argument('--M', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='List of M values (batch sizes) to test')
    parser.add_argument('--N', type=int, default=14336, help='N dimension (intermediate_size)')
    parser.add_argument('--K', type=int, default=4096, help='K dimension (hidden_size)')
    parser.add_argument('--E', type=int, default=8, help='Number of experts')
    parser.add_argument('--top_k', type=int, default=2, help='Top-k routing')
    args = parser.parse_args()
    
    print("=" * 90)
    print("MOE Triton Kernel Benchmark (OPTIMIZED)")
    print(f"Config: N={args.N}, K={args.K}, E={args.E}, top_k={args.top_k}")
    print("=" * 90)
    
    print(f"\n{'M':>6} | {'Time(ms)':>10} | {'TFLOPS':>10} | {'BW(GB/s)':>12} | Config")
    print("-" * 90)
    
    results = []
    for M in args.M:
        try:
            result = benchmark_moe_triton(M, args.N, args.K, args.E, args.top_k)
            results.append(result)
            cfg = result['config']
            cfg_str = f"{cfg['BLOCK_SIZE_M']}x{cfg['BLOCK_SIZE_N']}x{cfg['BLOCK_SIZE_K']} w{cfg['num_warps']} g{cfg['GROUP_SIZE_M']}"
            print(f"{M:>6} | {result['time_ms']:>10.4f} | {result['tflops']:>10.2f} | {result['bandwidth_gb_s']:>12.2f} | {cfg_str}")
        except Exception as e:
            print(f"{M:>6} | ERROR: {e}")
    
    if results:
        print("-" * 90)
        print(f"{'AVG':>6} | {np.mean([r['time_ms'] for r in results]):>10.4f} | {np.mean([r['tflops'] for r in results]):>10.2f} | {np.mean([r['bandwidth_gb_s'] for r in results]):>12.2f}")
        print(f"{'BEST':>6} | {min([r['time_ms'] for r in results]):>10.4f} | {max([r['tflops'] for r in results]):>10.2f} | {max([r['bandwidth_gb_s'] for r in results]):>12.2f}")

if __name__ == "__main__":
    main()


