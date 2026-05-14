"""
Sweep NUM_COPIES for privatized dKV to test whether intra-XCD atomic
serialization is the remaining bottleneck after XCD-local routing.

Two routing strategies tested at each copy count:
  "modulo"  -- copy = pid % NUM_COPIES  (not XCD-local, baseline reference)
  "xcd"     -- copy = (pid % 304) // 38  clamped to NUM_COPIES
               (XCD-local when NUM_COPIES == num_xcd=8)

Hypothesis being tested (H2):
  If xcd_privatized routing is correct and intra-XCD serialization is the
  bottleneck, then increasing copies beyond 8 (i.e., >1 copy per XCD) should
  further reduce atomic contention and improve timing.

  If timing is flat across all copy counts for "xcd" routing, either:
    (a) the routing doesn't actually achieve XCD locality (-> H1), or
    (b) something other than atomic serialization is the bottleneck.

Usage:
  python op_tests/triton_tests/attention/bench_copies_sweep.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
    sparse_mla_fwd,
    _bwd_dq_store_intermediates,
    _bwd_dkv_privatized,
    _bwd_dkv_reduce_copies,
    _sparse_mla_bwd_preprocess,
)


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


T, H, DV, DR, TOPK = 4096, 128, 512, 64, 1024
D = DV + DR
bh = 64
num_hg = triton.cdiv(H, bh)
scale = D ** -0.5
device = "cuda"
num_cus = 304
cus_per_xcd = 38
num_xcd = 8
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

_sparse_mla_bwd_preprocess[(T, triton.cdiv(H, bh))](
    o, do, delta, o.stride(0), o.stride(1), H, DV, BLOCK_H=bh,
)
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


def run_privatized(num_copies, routing="modulo", reps=50):
    """
    routing="modulo": copy = pid % num_copies  (not XCD-local)
    routing="xcd":    copy = (pid % 304) // 38 % num_copies  (XCD-local attempt)
    """
    stride_copies = T * D
    dkv_copies = torch.zeros(num_copies * stride_copies, dtype=torch.float32, device=device)
    total_elems = T * D

    if routing == "modulo":
        def fn():
            dkv_copies.zero_()
            _bwd_dkv_privatized[(T,)](
                q_t, do_t, dS, P, topk_idx, dkv_copies,
                q_t.stride(0), do_t.stride(0),
                dS.stride(0), dS.stride(1),
                topk_idx.stride(0), stride_copies, D,
                H,
                TOPK=TOPK, TILE_K=64, BLOCK_H=bh,
                NUM_HG=num_hg, NUM_COPIES=num_copies,
                D_V=DV, D_ROPE=DR,
                num_warps=4, num_stages=1,
            )
            _bwd_dkv_reduce_copies[(triton.cdiv(total_elems, reduce_block),)](
                dkv_copies, dkv, stride_copies, total_elems,
                NUM_COPIES=num_copies, BLOCK=reduce_block,
            )
    else:
        # XCD-local: use _bwd_dkv_xcd_local with num_copies copies
        from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
            _bwd_dkv_xcd_local,
        )
        # When num_copies > num_xcd, use num_copies as the total copy count
        # with (pid % 304) // (304 // num_copies) routing
        # For num_copies == 8: same as xcd_privatized
        # For num_copies > 8: multiple copies per XCD, further reduces serialization depth
        effective_cus_per_copy = max(1, num_cus // num_copies)
        def fn():
            dkv_copies.zero_()
            _bwd_dkv_xcd_local[(T,)](
                q_t, do_t, dS, P, topk_idx, dkv_copies,
                q_t.stride(0), do_t.stride(0),
                dS.stride(0), dS.stride(1),
                topk_idx.stride(0), stride_copies, D,
                H,
                TOPK=TOPK, TILE_K=64, BLOCK_H=bh,
                NUM_HG=num_hg, NUM_XCD=num_copies,
                CUS_PER_XCD=effective_cus_per_copy,
                D_V=DV, D_ROPE=DR,
                num_warps=4, num_stages=1,
            )
            _bwd_dkv_reduce_copies[(triton.cdiv(total_elems, reduce_block),)](
                dkv_copies, dkv, stride_copies, total_elems,
                NUM_COPIES=num_copies, BLOCK=reduce_block,
            )

    return bench(fn, reps)


print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Config: T={T} H={H} D={D} TOPK={TOPK}")
print(f"\nSweeping NUM_COPIES for two routing strategies:")
print(f"  modulo: copy = pid % N                    (NOT XCD-local, cross-XCD atomics remain)")
print(f"  xcd:    copy = (pid%304)//(304//N)         (XCD-local attempt, 1 copy per N/8 XCDs)")
print(f"\nIf H2 (intra-XCD serialization is bottleneck):")
print(f"  -> 'xcd' timing should improve as N increases beyond 8 (less serialization per copy)")
print(f"If H1 (routing is wrong):")
print(f"  -> 'xcd' timing ~ 'modulo' timing at same N (same cross-XCD pressure)")

copy_counts = [1, 2, 4, 8, 16, 32, 64]

print(f"\n{'Copies':>8s}  {'modulo (ms)':>12s}  {'xcd (ms)':>10s}  {'xcd/modulo':>10s}")
print("-" * 48)

for nc in copy_counts:
    try:
        ms_mod = run_privatized(nc, routing="modulo")
    except Exception as e:
        ms_mod = float("nan")

    try:
        ms_xcd = run_privatized(nc, routing="xcd")
    except Exception as e:
        ms_xcd = float("nan")

    ratio = ms_xcd / ms_mod if ms_mod > 0 else float("nan")
    print(f"{nc:8d}  {ms_mod:12.2f}  {ms_xcd:10.2f}  {ratio:10.3f}x")

print(f"\nInterpretation:")
print(f"  If 'xcd' at N=8 < 'modulo' at N=8: XCD-local routing IS helping (H2 supported)")
print(f"  If 'xcd' timing flat N=8..64:       intra-XCD serialization is NOT the bottleneck")
print(f"  If 'xcd' timing drops N=8..64:      intra-XCD serialization IS the bottleneck (H2)")
print(f"  If 'xcd' ~ 'modulo' at all N:       routing is wrong (H1)")
