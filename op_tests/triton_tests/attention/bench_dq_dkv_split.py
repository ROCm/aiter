"""
Benchmark dQ and dKV kernels in isolation to find which is the bottleneck
in the split_intermediate backward method.

Usage:
  python op_tests/triton_tests/attention/bench_dq_dkv_split.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
    sparse_mla_fwd,
    _bwd_dq_store_intermediates,
    _bwd_dkv_hg_fused,
    _bwd_dkv_privatized,
    _bwd_dkv_xcd_local,
    _bwd_dkv_nonatomic_scatter,
    _bwd_dkv_reduce_copies,
    _sparse_mla_bwd_preprocess,
)
import triton


def bench(fn, reps=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


def run_config(seq_len=4096, num_heads=128, kv_lora_rank=512, rope_rank=64, topk=1024,
               num_copies=8, device="cuda", reps=50):
    T, H, DV, DR = seq_len, num_heads, kv_lora_rank, rope_rank
    D = DV + DR
    scale = D ** -0.5
    bh = 64
    num_hg = triton.cdiv(H, bh)

    q   = torch.randn(T, H, D,  dtype=torch.bfloat16, device=device)
    kv  = torch.randn(T, 1, D,  dtype=torch.bfloat16, device=device)
    do  = torch.randn(T, H, DV, dtype=torch.bfloat16, device=device)
    topk_idx = torch.randint(0, T, (T, topk), dtype=torch.int32, device=device)
    o, lse = sparse_mla_fwd(q, kv, topk_idx, DV, scale)

    dq    = torch.empty_like(q)
    dkv   = torch.zeros(T, D, dtype=torch.float32, device=device)
    delta = torch.empty(T, H, dtype=torch.float32, device=device)
    dS    = torch.zeros(T, H, topk, dtype=torch.bfloat16, device=device)
    P     = torch.zeros(T, H, topk, dtype=torch.bfloat16, device=device)
    q_t   = q.transpose(1, 2).contiguous()
    do_t  = do.transpose(1, 2).contiguous()

    # preprocess (delta)
    _sparse_mla_bwd_preprocess[(T, triton.cdiv(H, bh))](
        o, do, delta,
        o.stride(0), o.stride(1),
        H, DV, BLOCK_H=bh,
    )

    def run_dq():
        _bwd_dq_store_intermediates[(T, num_hg)](
            q, kv, do, topk_idx, lse, delta,
            dq, dS, P,
            q.stride(0), q.stride(1), kv.stride(0),
            do.stride(0), do.stride(1),
            dq.stride(0), dq.stride(1),
            topk_idx.stride(0),
            dS.stride(0), dS.stride(1),
            scale, H,
            TOPK=topk, BLOCK_H=bh, TILE_K=16,
            D_V=DV, D_ROPE=DR,
            num_warps=4, num_stages=2,
        )

    def run_dkv_hg_fused():
        _bwd_dkv_hg_fused[(T,)](
            q_t, do_t, dS, P, topk_idx, dkv,
            q_t.stride(0), do_t.stride(0),
            dS.stride(0), dS.stride(1),
            topk_idx.stride(0), dkv.stride(0),
            H,
            TOPK=topk, TILE_K=64, BLOCK_H=bh,
            NUM_HG=num_hg, D_V=DV, D_ROPE=DR,
            num_warps=4, num_stages=1,
        )

    stride_copies = T * D
    dkv_copies = torch.zeros(num_copies * stride_copies, dtype=torch.float32, device=device)
    total_elems = T * D
    reduce_block = 1024

    num_xcd, cus_per_xcd = 8, 38  # MI300X: 304 CUs, 38 per XCD
    dkv_xcd_copies = torch.zeros(num_xcd * stride_copies, dtype=torch.float32, device=device)

    def run_dkv_privatized():
        _bwd_dkv_privatized[(T,)](
            q_t, do_t, dS, P, topk_idx, dkv_copies,
            q_t.stride(0), do_t.stride(0),
            dS.stride(0), dS.stride(1),
            topk_idx.stride(0),
            stride_copies, D,
            H,
            TOPK=topk, TILE_K=64, BLOCK_H=bh,
            NUM_HG=num_hg, NUM_COPIES=num_copies,
            D_V=DV, D_ROPE=DR,
            num_warps=4, num_stages=1,
        )
        _bwd_dkv_reduce_copies[(triton.cdiv(total_elems, reduce_block),)](
            dkv_copies, dkv,
            stride_copies, total_elems,
            NUM_COPIES=num_copies, BLOCK=reduce_block,
        )

    def run_dkv_xcd_local():
        dkv_xcd_copies.zero_()
        _bwd_dkv_xcd_local[(T,)](
            q_t, do_t, dS, P, topk_idx, dkv_xcd_copies,
            q_t.stride(0), do_t.stride(0),
            dS.stride(0), dS.stride(1),
            topk_idx.stride(0), stride_copies, D,
            H,
            TOPK=topk, TILE_K=64, BLOCK_H=bh,
            NUM_HG=num_hg, NUM_XCD=num_xcd, CUS_PER_XCD=cus_per_xcd,
            D_V=DV, D_ROPE=DR,
            num_warps=4, num_stages=1,
        )
        _bwd_dkv_reduce_copies[(triton.cdiv(total_elems, reduce_block),)](
            dkv_xcd_copies, dkv,
            stride_copies, total_elems,
            NUM_COPIES=num_xcd, BLOCK=reduce_block,
        )

    def run_dkv_nonatomic():
        _bwd_dkv_nonatomic_scatter[(T,)](
            q_t, do_t, dS, P, topk_idx, dkv,
            q_t.stride(0), do_t.stride(0),
            dS.stride(0), dS.stride(1),
            topk_idx.stride(0), dkv.stride(0),
            H,
            TOPK=topk, TILE_K=64, BLOCK_H=bh,
            NUM_HG=num_hg, D_V=DV, D_ROPE=DR,
            num_warps=4, num_stages=1,
        )

    ms_dq           = bench(run_dq, reps)
    ms_dkv_fused    = bench(run_dkv_hg_fused, reps)
    ms_dkv_private  = bench(run_dkv_privatized, reps)
    ms_dkv_xcd      = bench(run_dkv_xcd_local, reps)
    ms_dkv_nonatomic = bench(run_dkv_nonatomic, reps)

    # Theoretical HBM traffic from kernel parameters (no L2 reuse assumed for Q/dO).
    # fetch: Q_lora_T + Q_rope_T + dO_T loaded num_tiles*num_hg times each per CTA;
    #        dS + P loaded num_tiles*num_hg times each (all unique); topk_idx num_tiles times.
    num_tiles = triton.cdiv(topk, 64)  # TILE_K=64
    fetch_bytes = T * (
        num_tiles * num_hg * (DV * bh * 2)       # Q_lora_T  [D_V x BLOCK_H] bf16
        + num_tiles * num_hg * (DR * bh * 2)     # Q_rope_T  [D_ROPE x BLOCK_H] bf16
        + num_tiles * num_hg * (DV * bh * 2)     # dO_T      [D_V x BLOCK_H] bf16
        + num_tiles * num_hg * (bh * 64 * 2)     # dS        [BLOCK_H x TILE_K] bf16
        + num_tiles * num_hg * (bh * 64 * 2)     # P         [BLOCK_H x TILE_K] bf16
        + num_tiles * (64 * 4)                    # topk_idx  [TILE_K] int32
    )
    # write: nonatomic = T*TOPK*D*4; atomic = 4x empirical (from rocprof: WRITE_SIZE = 4*nonatomic)
    write_nonatomic_bytes = T * topk * D * 4
    write_atomic_bytes    = 4 * write_nonatomic_bytes

    fetch_gb         = fetch_bytes         / 1e9
    write_nonatomic_gb = write_nonatomic_bytes / 1e9
    write_atomic_gb    = write_atomic_bytes    / 1e9

    bw_peak_GBs = 5300  # 5.3 TB/s
    t_bw_atomic    = (fetch_gb + write_atomic_gb)    / bw_peak_GBs * 1e3  # ms
    t_bw_nonatomic = (fetch_gb + write_nonatomic_gb) / bw_peak_GBs * 1e3  # ms

    label = f"S{seq_len}_H{num_heads}_topk{topk}"
    print(f"\n  {label}")
    print(f"  {'Kernel':<36s}  {'Time (ms)':>10s}  {'vs hg_fused':>12s}")
    print(f"  {'-'*64}")
    print(f"  {'dKV (hg_fused, atomic)':<36s}  {ms_dkv_fused:10.2f}  {'baseline':>12s}")
    print(f"  {'dKV (privatized x8, i%8)':<36s}  {ms_dkv_private:10.2f}  {ms_dkv_private/ms_dkv_fused:11.2f}x")
    print(f"  {'dKV (xcd_local x8, (i%304)//38)':<36s}  {ms_dkv_xcd:10.2f}  {ms_dkv_xcd/ms_dkv_fused:11.2f}x")
    print(f"  {'dKV (nonatomic, wrong results)':<36s}  {ms_dkv_nonatomic:10.2f}  {ms_dkv_nonatomic/ms_dkv_fused:11.2f}x")
    print(f"  {'-'*64}")
    print(f"  {'BW floor (atomic)':<36s}  {t_bw_atomic:10.2f}  "
          f"  ({fetch_gb:.0f}+{write_atomic_gb:.0f}={fetch_gb+write_atomic_gb:.0f} GB / 5300 GB/s)")
    print(f"  {'BW floor (nonatomic)':<36s}  {t_bw_nonatomic:10.2f}  "
          f"  ({fetch_gb:.0f}+{write_nonatomic_gb:.0f}={fetch_gb+write_nonatomic_gb:.0f} GB / 5300 GB/s)")
    print(f"  {'-'*64}")
    print(f"  Atomic overhead above nonatomic:  {ms_dkv_fused - ms_dkv_nonatomic:.2f} ms")
    print(f"  Nonatomic overhead above BW floor: {ms_dkv_nonatomic - t_bw_nonatomic:.2f} ms  "
          f"(random scatter latency)")
    print(f"  Atomic serialization stalls:       "
          f"{ms_dkv_fused - ms_dkv_nonatomic - (t_bw_atomic - t_bw_nonatomic):.2f} ms  (estimated)")
    print(f"\n  dQ benchmark:")
    print(f"  {'dQ (store_intermediates)':<36s}  {ms_dq:10.2f}")
    print(f"  {'dQ + dKV_fused (split_int total)':<36s}  {ms_dq + ms_dkv_fused:10.2f}")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\n" + "=" * 70)
    print("  dQ vs dKV kernel split (split_intermediate method)")
    print("=" * 70)

    configs = [
        (4096, 128, 512, 64, 1024),
        (4096, 128, 512, 64, 2048),
        (8192, 128, 512, 64, 1024),
    ]
    for cfg in configs:
        run_config(*cfg)


if __name__ == "__main__":
    main()
