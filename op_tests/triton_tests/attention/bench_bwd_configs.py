"""Benchmark backward kernel with fixed configs to compare spill-free vs spill-heavy."""
import torch
import triton
import triton.language as tl
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
    _sparse_mla_bwd_kernel,
    _sparse_mla_bwd_preprocess,
    sparse_mla_fwd,
)


def bench_config(bh, tk, nw, ns, q, kv, o, do, topk_indices, lse, kv_lora_rank, scale):
    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]

    dq = torch.empty_like(q)
    dkv = torch.zeros(total_tokens, d_qk, dtype=torch.float32, device=q.device)
    delta = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)
    q_t = q.transpose(1, 2).contiguous()
    do_t = do.transpose(1, 2).contiguous()

    # Preprocess
    BLOCK_H_PRE = min(64, num_heads)
    grid_pre = (total_tokens, triton.cdiv(num_heads, BLOCK_H_PRE))
    _sparse_mla_bwd_preprocess[grid_pre](
        O_ptr=o, dO_ptr=do, Delta_ptr=delta,
        stride_o_t=o.stride(0), stride_o_h=o.stride(1),
        num_heads=num_heads, D_V=kv_lora_rank, BLOCK_H=BLOCK_H_PRE,
    )

    actual_bh = min(bh, num_heads)
    grid = (total_tokens, triton.cdiv(num_heads, actual_bh))

    def run():
        dkv.zero_()
        _sparse_mla_bwd_kernel.fn[grid](
            q, kv, do, topk_indices, lse, delta, dq, dkv, q_t, do_t,
            q.stride(0), q.stride(1),
            kv.stride(0),
            do.stride(0), do.stride(1),
            dq.stride(0), dq.stride(1),
            dkv.stride(0),
            topk_indices.stride(0),
            q_t.stride(0),
            do_t.stride(0),
            scale, num_heads,
            TOPK=topk, BLOCK_H=actual_bh, TILE_K=tk,
            D_V=kv_lora_rank, D_ROPE=rope_rank,
            num_warps=nw, num_stages=ns,
        )

    # Warmup
    for _ in range(3):
        run()
    torch.cuda.synchronize()

    # Benchmark
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    reps = 50
    ev0.record()
    for _ in range(reps):
        run()
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / reps

    flops_bwd = total_tokens * num_heads * topk * (
        2 * d_qk + 2 * kv_lora_rank + 2 * d_qk + 2 * d_qk
    )
    tflops = flops_bwd / (ms * 1e-3) / 1e12
    return ms, tflops


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton: {triton.__version__}")

    batch, seq_len, num_heads = 1, 4096, 128
    kv_lora_rank, rope_rank, topk = 512, 64, 1024
    d_qk = kv_lora_rank + rope_rank
    total_tokens = batch * seq_len
    scale = 1.0 / (d_qk ** 0.5)

    torch.manual_seed(42)
    q = torch.randn(total_tokens, num_heads, d_qk, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(total_tokens, 1, d_qk, dtype=torch.bfloat16, device="cuda")
    topk_indices = torch.randint(0, total_tokens, (total_tokens, topk),
                                 dtype=torch.int32, device="cuda")
    o, lse = sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank, scale)
    do = torch.randn_like(o)

    configs = [
        # (BLOCK_H, TILE_K, num_warps, num_stages, label)
        (64, 16, 4, 2, "BH=64 w=4 s=2  (105 spills, autotune best)"),
        (64, 16, 4, 1, "BH=64 w=4 s=1  ( 96 spills)"),
        (64, 16, 8, 2, "BH=64 w=8 s=2  (180 spills)"),
        (64, 16, 8, 1, "BH=64 w=8 s=1  (202 spills)"),
        (32, 16, 4, 2, "BH=32 w=4 s=2  (  0 spills)"),
        (32, 16, 4, 1, "BH=32 w=4 s=1  (  0 spills)"),
        (32, 16, 8, 2, "BH=32 w=8 s=2  (101 spills)"),
        (32, 16, 8, 1, "BH=32 w=8 s=1  ( 97 spills)"),
        (16, 16, 4, 2, "BH=16 w=4 s=2  (  0 spills)"),
        (16, 16, 4, 1, "BH=16 w=4 s=1  (  0 spills)"),
        (16, 16, 8, 2, "BH=16 w=8 s=2  ( 50 spills)"),
        (16, 16, 8, 1, "BH=16 w=8 s=1  ( 45 spills)"),
    ]

    print(f"\nBackward benchmark: S{seq_len} H{num_heads} topk{topk} (TK=16 fixed)\n")
    print(f"  {'Config':<45s}  {'ms':>8s}  {'TFLOPS':>8s}")
    print(f"  {'-'*45}  {'-'*8}  {'-'*8}")

    for bh, tk, nw, ns, label in configs:
        try:
            ms, tflops = bench_config(bh, tk, nw, ns, q, kv, o, do, topk_indices,
                                       lse, kv_lora_rank, scale)
            print(f"  {label:<45s}  {ms:8.3f}  {tflops:8.1f}")
        except Exception as e:
            print(f"  {label:<45s}  ERROR: {e}")


if __name__ == "__main__":
    main()
