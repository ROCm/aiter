"""
Minimal script that runs dKV kernels for rocprof hardware counter collection.

Profiles two kernels back-to-back:
  1. _bwd_dkv_hg_fused     — baseline (cross-XCD atomic adds)
  2. _bwd_dkv_xcd_local    — XCD-local routing ((i%304)//38 assignment)

Usage:
  rocprof -i op_tests/triton_tests/attention/rocprof_counters.txt \
          --timestamp on \
          -o /tmp/dkv_profile.csv \
          python op_tests/triton_tests/attention/rocprof_dkv_only.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
    sparse_mla_fwd,
    _bwd_dq_store_intermediates,
    _bwd_dkv_hg_fused,
    _bwd_dkv_xcd_local,
    _bwd_dkv_reduce_copies,
    _sparse_mla_bwd_preprocess,
)
import triton

T, H, DV, DR, TOPK = 4096, 128, 512, 64, 1024
D = DV + DR
bh = 64
num_hg = triton.cdiv(H, bh)
scale = D ** -0.5
device = "cuda"
num_xcd, cus_per_xcd = 8, 38  # MI300X
stride_copies = T * D
total_elems = T * D
reduce_block = 1024

torch.manual_seed(42)
q        = torch.randn(T, H, D,    dtype=torch.bfloat16, device=device)
kv       = torch.randn(T, 1, D,    dtype=torch.bfloat16, device=device)
do       = torch.randn(T, H, DV,   dtype=torch.bfloat16, device=device)
topk_idx = torch.randint(0, T, (T, TOPK), dtype=torch.int32, device=device)
o, lse   = sparse_mla_fwd(q, kv, topk_idx, DV, scale)

dq         = torch.empty_like(q)
dkv        = torch.zeros(T, D, dtype=torch.float32, device=device)
delta      = torch.empty(T, H, dtype=torch.float32, device=device)
dS         = torch.zeros(T, H, TOPK, dtype=torch.bfloat16, device=device)
P          = torch.zeros(T, H, TOPK, dtype=torch.bfloat16, device=device)
q_t        = q.transpose(1, 2).contiguous()
do_t       = do.transpose(1, 2).contiguous()
dkv_copies = torch.zeros(num_xcd * stride_copies, dtype=torch.float32, device=device)

# preprocess delta
_sparse_mla_bwd_preprocess[(T, triton.cdiv(H, bh))](
    o, do, delta, o.stride(0), o.stride(1), H, DV, BLOCK_H=bh,
)

# run dQ once to populate dS and P (inputs to dKV)
_bwd_dq_store_intermediates[(T, num_hg)](
    q, kv, do, topk_idx, lse, delta, dq, dS, P,
    q.stride(0), q.stride(1), kv.stride(0),
    do.stride(0), do.stride(1),
    dq.stride(0), dq.stride(1),
    topk_idx.stride(0), dS.stride(0), dS.stride(1),
    scale, H,
    TOPK=TOPK, BLOCK_H=bh, TILE_K=16,
    D_V=DV, D_ROPE=DR, num_warps=4, num_stages=2,
)
torch.cuda.synchronize()

# warmup dKV
for _ in range(3):
    dkv.zero_()
    _bwd_dkv_hg_fused[(T,)](
        q_t, do_t, dS, P, topk_idx, dkv,
        q_t.stride(0), do_t.stride(0),
        dS.stride(0), dS.stride(1),
        topk_idx.stride(0), dkv.stride(0),
        H, TOPK=TOPK, TILE_K=64, BLOCK_H=bh,
        NUM_HG=num_hg, D_V=DV, D_ROPE=DR,
        num_warps=4, num_stages=1,
    )
torch.cuda.synchronize()

# profiled run 1 — _bwd_dkv_hg_fused (baseline, cross-XCD atomics)
dkv.zero_()
_bwd_dkv_hg_fused[(T,)](
    q_t, do_t, dS, P, topk_idx, dkv,
    q_t.stride(0), do_t.stride(0),
    dS.stride(0), dS.stride(1),
    topk_idx.stride(0), dkv.stride(0),
    H, TOPK=TOPK, TILE_K=64, BLOCK_H=bh,
    NUM_HG=num_hg, D_V=DV, D_ROPE=DR,
    num_warps=4, num_stages=1,
)
torch.cuda.synchronize()
print("Run 1 (hg_fused) done.")

# warmup _bwd_dkv_xcd_local
for _ in range(3):
    dkv_copies.zero_()
    _bwd_dkv_xcd_local[(T,)](
        q_t, do_t, dS, P, topk_idx, dkv_copies,
        q_t.stride(0), do_t.stride(0),
        dS.stride(0), dS.stride(1),
        topk_idx.stride(0), stride_copies, D,
        H, TOPK=TOPK, TILE_K=64, BLOCK_H=bh,
        NUM_HG=num_hg, NUM_XCD=num_xcd, CUS_PER_XCD=cus_per_xcd,
        D_V=DV, D_ROPE=DR,
        num_warps=4, num_stages=1,
    )
torch.cuda.synchronize()

# profiled run 2 — _bwd_dkv_xcd_local (XCD-local, (i%304)//38 routing)
dkv_copies.zero_()
_bwd_dkv_xcd_local[(T,)](
    q_t, do_t, dS, P, topk_idx, dkv_copies,
    q_t.stride(0), do_t.stride(0),
    dS.stride(0), dS.stride(1),
    topk_idx.stride(0), stride_copies, D,
    H, TOPK=TOPK, TILE_K=64, BLOCK_H=bh,
    NUM_HG=num_hg, NUM_XCD=num_xcd, CUS_PER_XCD=cus_per_xcd,
    D_V=DV, D_ROPE=DR,
    num_warps=4, num_stages=1,
)
_bwd_dkv_reduce_copies[(triton.cdiv(total_elems, reduce_block),)](
    dkv_copies, dkv,
    stride_copies, total_elems,
    NUM_COPIES=num_xcd, BLOCK=reduce_block,
)
torch.cuda.synchronize()
print("Run 2 (xcd_local) done. Check /tmp/dkv_profile.csv for hardware counters.")
