"""Side-by-side V3.2 vs V4 chunked_gather bwd memory comparison on MI300X."""
import sys
sys.path.insert(0, "/workspace/aiter_main_7890e4b")

import torch

from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
    sparse_mla_fwd, sparse_mla_bwd,
)
from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention_v4 import (
    sparse_mla_fwd_v4, sparse_mla_bwd_v4,
)


def _mb(b):
    return b / 1024**2


def measure_v3(T, H, D_V, D_ROPE, TOPK):
    D_QK = D_V + D_ROPE
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    q = torch.randn(T, H, D_QK, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(T, 1, D_QK, dtype=torch.bfloat16, device="cuda")
    topk = torch.randint(0, T, (T, TOPK), dtype=torch.int32, device="cuda")
    do = torch.randn(T, H, D_V, dtype=torch.bfloat16, device="cuda")
    o, lse = sparse_mla_fwd(q, kv, topk, kv_lora_rank=D_V)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    dq, dkv = sparse_mla_bwd(q, kv, o, do, topk, lse, kv_lora_rank=D_V, method="chunked_gather")
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return _mb(peak - base)


def measure_v4(T, H, D_V, D_ROPE, TOPK, use_sink):
    D_QK = D_V + D_ROPE
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    q = torch.randn(T, H, D_QK, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(T, 1, D_QK, dtype=torch.bfloat16, device="cuda")
    topk = torch.randint(0, T, (T, TOPK), dtype=torch.int32, device="cuda")
    do = torch.randn(T, H, D_V, dtype=torch.bfloat16, device="cuda")
    sink = torch.randn(H, dtype=torch.float32, device="cuda") if use_sink else None
    o, lse = sparse_mla_fwd_v4(q, kv, topk, attn_sink=sink, kv_lora_rank=D_V)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    dq, dkv, ds = sparse_mla_bwd_v4(q, kv, o, do, topk, lse, attn_sink=sink, kv_lora_rank=D_V)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return _mb(peak - base)


def main():
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Arch:   {torch.cuda.get_device_properties(0).gcnArchName}")
    print()
    print("Apples-to-apples bwd peak extra memory (MB), same shape, V3.2 chunked_gather vs V4(chunked_gather)")
    print()
    print(f"  {'T':>6} {'H':>4} {'TOPK':>5}   {'V3.2':>10} {'V4 no-sink':>12} {'V4 sink':>10}   {'V4/V3.2':>9}   note")
    print("  " + "-" * 95)

    configs = [
        (4096,   64,  640,  "V4-Flash CSA"),
        (4096,   64, 1024,  "V3.2 typical at H=64"),
        (4096,  128, 1024,  "V3.2 standard config"),
        (4096,  128, 1152,  "V4-Pro CSA"),
        (4096,  128,  128,  "SWA-only"),
        (4096,  128,  160,  "HCA at T=4K"),
        (8192,  128, 1024,  "V3.2 at T=8K"),
        (8192,  128, 1152,  "V4-Pro CSA at T=8K"),
        (16384, 128, 1152,  "V4-Pro CSA at T=16K"),
    ]
    for T, H, TOPK, note in configs:
        v3 = measure_v3(T, H, 512, 64, TOPK)
        v4_ns = measure_v4(T, H, 512, 64, TOPK, use_sink=False)
        v4_s  = measure_v4(T, H, 512, 64, TOPK, use_sink=True)
        print(f"  {T:>6d} {H:>4d} {TOPK:>5d}   {v3:>7.1f} MB {v4_ns:>9.1f} MB {v4_s:>7.1f} MB   {v4_s/v3:>7.2f}x   {note}")


if __name__ == "__main__":
    main()
