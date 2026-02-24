import torch
import triton
import triton.language as tl

@triton.jit
def fused_rearrange_kernel(
    mixed_ptr, q_ptr, k_ptr, v_ptr,
    L, H, Dk, Dv,
    stride_ml, stride_md,
    stride_ql, stride_qh, stride_qd,
    stride_kl, stride_kh, stride_kd,
    stride_vl, stride_vh, stride_vd,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    # Program IDs
    pid_l = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Offsets
    offs_l = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    # Masks
    mask_l = offs_l < L
    mask_dk = offs_d < Dk
    mask_dv = offs_d < Dv

    # Calculate base offsets for Q, K, and V in the mixed tensor
    # Assuming mixed_qkv is packed as [Q, K, V] along the last dim
    offset_q_base = 0
    offset_k_base = H * Dk
    offset_v_base = 2 * H * Dk

    # Load and Store Query
    idx_m_q = (offs_l[:, None] * stride_ml + 
               (pid_h * Dk + offset_q_base + offs_d[None, :]))
    q = tl.load(mixed_ptr + idx_m_q, mask=mask_l[:, None] & mask_dk[None, :])
    idx_out_q = (offs_l[:, None] * stride_ql + pid_h * stride_qh + offs_d[None, :] * stride_qd)
    tl.store(q_ptr + idx_out_q, q, mask=mask_l[:, None] & mask_dk[None, :])

    # Load and Store Key
    idx_m_k = (offs_l[:, None] * stride_ml + 
               (pid_h * Dk + offset_k_base + offs_d[None, :]))
    k = tl.load(mixed_ptr + idx_m_k, mask=mask_l[:, None] & mask_dk[None, :])
    idx_out_k = (offs_l[:, None] * stride_kl + pid_h * stride_kh + offs_d[None, :] * stride_kd)
    tl.store(k_ptr + idx_out_k, k, mask=mask_l[:, None] & mask_dk[None, :])

    # Load and Store Value (Note: Dv might differ from Dk)
    idx_m_v = (offs_l[:, None] * stride_ml + 
               (pid_h * Dv + offset_v_base + offs_d[None, :]))
    v = tl.load(mixed_ptr + idx_m_v, mask=mask_l[:, None] & mask_dv[None, :])
    idx_out_v = (offs_l[:, None] * stride_vl + pid_h * stride_vh + offs_d[None, :] * stride_vd)
    tl.store(v_ptr + idx_out_v, v, mask=mask_l[:, None] & mask_dv[None, :])


def triton_rearrange_qkv(mixed_qkv, key_dim, value_dim, Dk, Dv):
    if mixed_qkv is None:
        return None, None, None

    L, total_dim = mixed_qkv.shape
    H = total_dim // (2 * Dk + Dv)

    # Pre-allocate contiguous output tensors
    # Shape: (1, L, H, D)
    q = torch.empty((1, L, H, Dk), device=mixed_qkv.device, dtype=mixed_qkv.dtype)
    k = torch.empty((1, L, H, Dk), device=mixed_qkv.device, dtype=mixed_qkv.dtype)
    v = torch.empty((1, L, H, Dv), device=mixed_qkv.device, dtype=mixed_qkv.dtype)

    BLOCK_SIZE_L=128
    grid = (triton.cdiv(L, BLOCK_SIZE_L), H)

    fused_rearrange_kernel[grid](
        mixed_qkv, q, k, v,
        L, H, Dk, Dv,
        mixed_qkv.stride(0), mixed_qkv.stride(1),
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        BLOCK_SIZE_D=triton.next_power_of_2(max(Dk, Dv))
    )

    return q, k, v



import sys
import torch
import triton
import math
from typing import Optional
from triton.testing import do_bench, perf_report, Benchmark

# Assuming the Triton kernel we wrote earlier is in a file named `fused_ops.py`
# from fused_ops import triton_rearrange_qkv 

## --- MOCK WRAPPER FOR BENCHMARKING (If not using the previous Triton response) ---
def pytorch_rearrange_qkv(mixed_qkv, key_dim, value_dim, head_k_dim, head_v_dim):
    # This is the original code you provided
    query, key, value = torch.split(
        mixed_qkv,
        [key_dim , key_dim , value_dim ],
        dim=-1,
    )
    # Reshape logic
    q = query.view(1, -1, (key_dim ) // head_k_dim, head_k_dim)
    k = key.view(1, -1, (key_dim ) // head_k_dim, head_k_dim)
    v = value.view(1, -1, (value_dim) // head_v_dim, head_v_dim)
    return q.contiguous(), k.contiguous(), v.contiguous()

## --- BENCHMARK FUNCTION ---

def bench_qkv_fn(L, H, Dk, Dv, provider):
    device = "cuda"
    dtype = torch.bfloat16
    
    # Calculate total dimension based on QKV (2*K + 1*V)
    # Matches the logic: [key_dim // tp, key_dim // tp, value_dim // tp]
    total_dim = (2 * H * Dk) + (H * Dv)
    mixed_qkv = torch.randn((L, total_dim), device=device, dtype=dtype)
    
    # Parameters for the original function
    key_dim = H * Dk 
    value_dim = H * Dv

    if provider == "pytorch":
        fn = lambda: pytorch_rearrange_qkv(mixed_qkv, key_dim, value_dim,  Dk, Dv)
    else:
        # Assuming you have implemented the triton_rearrange_qkv from the previous step
        # fn = lambda: triton_rearrange_qkv(mixed_qkv, L, H, Dk, Dv)
        fn = lambda: triton_rearrange_qkv(mixed_qkv, key_dim, value_dim,  Dk, Dv) # Placeholder for Triton implementation

    ms = do_bench(fn, warmup=25, rep=100)
    
    # Metric: GB/s (Memory Bandwidth)
    # Read: mixed_qkv; Write: q, k, v (totaling the same size)
    total_bytes = 2 * mixed_qkv.numel() * mixed_qkv.element_size()
    gbps = total_bytes / (ms * 1e-3) * 1e-9
    # return gbps
    return ms

## --- PERFORMANCE REPORT CONFIG ---

@perf_report(
    Benchmark(
        x_names=["L"],  # Sequence Length
        x_vals=[1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["pytorch", "triton"],
        line_names=["PyTorch (Split+View+Contig)", "Triton (Fused)"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="GB/s",
        plot_name="qkv-rearrange-performance",
        args={"H": 32, "Dk": 128, "Dv": 128,},
    )
)
def run_benchmark(L, H, Dk, Dv, provider):
    return bench_qkv_fn(L, H, Dk, Dv, provider)

if __name__ == "__main__":
    # You can extend this with argparse like your example
    run_benchmark.run(save_path=".", print_data=True)