#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Test the exact shape reported by colleague:
  b=248, hq=8, hk=1, sq=1, sk=131072, d=64, blk=64

Verifies:
  1. CK-UA direct correctness (5 trials)
  2. Whether the selector activates CK-UA for this shape
  3. Performance: CK-UA vs Triton 2D vs Triton 3D vs auto (selector)
"""
import torch
import math
import time

from aiter.ops.unified_attention import unified_attention_fwd
from aiter.ops.triton.attention.unified_attention import unified_attention as triton_ua
import aiter.ops.triton.attention.unified_attention as ua_mod
from aiter.ops.triton.utils.device_info import get_num_sms

device = "cuda"
dtype = torch.bfloat16

seqs = 248
nqh = 8
nkh = 1
hdim = 64
maxk = 131072
blk = 64
scale = 1.0 / math.sqrt(hdim)

print(f"Shape: seqs={seqs} hq={nqh} hk={nkh} hdim={hdim} maxk={maxk} blk={blk}")
print()

# ============================================================
# 1. Correctness: CK-UA direct vs Triton
# ============================================================
print("=" * 60)
print("1. Correctness: CK-UA direct vs Triton (5 trials)")
print("=" * 60)

all_pass = True
for trial in range(5):
    torch.manual_seed(42 + trial)
    needed = (maxk + blk - 1) // blk
    num_phys = needed * seqs
    q = torch.randn(seqs, nqh, hdim, dtype=dtype, device=device)
    k = torch.randn(num_phys, blk, nkh, hdim, dtype=dtype, device=device)
    v = torch.randn(num_phys, blk, nkh, hdim, dtype=dtype, device=device)
    cu = torch.arange(seqs + 1, dtype=torch.int32, device=device)
    sl = torch.full((seqs,), maxk, dtype=torch.int32, device=device)
    bt = torch.randint(0, num_phys, (seqs, needed), dtype=torch.int32, device=device)

    o_ck = torch.zeros_like(q)
    unified_attention_fwd(
        o_ck, q, k, v, bt, sl, cu,
        mask_type=2, scale_s=scale,
        scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
    )

    o_ref = torch.zeros_like(q)
    triton_ua(
        q=q, k=k, v=v, out=o_ref, cu_seqlens_q=cu,
        max_seqlen_q=1, seqused_k=sl, max_seqlen_k=maxk,
        softmax_scale=scale, causal=True, window_size=(-1, -1),
        block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None,
    )
    torch.cuda.synchronize()

    diff = (o_ck.float() - o_ref.float()).abs().max().item()
    mismatched = ((o_ck.float() - o_ref.float()).abs() > 0.015).sum().item()
    total = o_ck.numel()
    ok = diff < 0.05
    if not ok:
        all_pass = False
    print(
        f"  trial {trial}: maxdiff={diff:.6f}  mismatched={mismatched}/{total} "
        f"({100*mismatched/total:.1f}%)  {'PASS' if ok else 'FAIL'}"
    )
    del q, k, v, cu, sl, bt, o_ck, o_ref
    torch.cuda.empty_cache()

print(f"\nResult: {'ALL PASS' if all_pass else '*** FAILURES ***'}")

# ============================================================
# 2. Selector check
# ============================================================
print()
print("=" * 60)
print("2. Selector check")
print("=" * 60)

cu_count = get_num_sms()
triton_wgs = nkh * seqs
lo = cu_count * 4
hi = cu_count * 8
in_range = lo <= triton_wgs <= hi

print(f"  GPU CUs:           {cu_count}")
print(f"  triton_2d_wgs:     nkh * seqs = {nkh} * {seqs} = {triton_wgs}")
print(f"  Selector range:    [{lo}, {hi}]")
print(f"  In range?          {in_range}")
if in_range:
    print(f"  -> Selector WOULD activate CK-UA")
else:
    print(f"  -> Selector does NOT activate CK-UA (uses Triton)")
    print(f"  -> Using -use_ck bypasses the selector and forces CK-UA")

# ============================================================
# 3. Performance
# ============================================================
print()
print("=" * 60)
print("3. Performance comparison")
print("=" * 60)

torch.manual_seed(42)
needed = (maxk + blk - 1) // blk
num_phys = needed * seqs
q = torch.randn(seqs, nqh, hdim, dtype=dtype, device=device)
k = torch.randn(num_phys, blk, nkh, hdim, dtype=dtype, device=device)
v = torch.randn_like(k)
cu = torch.arange(seqs + 1, dtype=torch.int32, device=device)
sl = torch.full((seqs,), maxk, dtype=torch.int32, device=device)
bt = torch.randint(0, num_phys, (seqs, needed), dtype=torch.int32, device=device)
out = torch.zeros_like(q)

kw = dict(
    q=q, k=k, v=v, out=out, cu_seqlens_q=cu,
    max_seqlen_q=1, seqused_k=sl, max_seqlen_k=maxk,
    softmax_scale=scale, causal=True, window_size=(-1, -1),
    block_table=bt, softcap=0.0,
    q_descale=None, k_descale=None, v_descale=None,
    alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None,
)

warmup = 5
iters = 20


def timed(fn):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


ck_ms = timed(
    lambda: unified_attention_fwd(
        out, q, k, v, bt, sl, cu,
        mask_type=2, scale_s=scale,
        scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
    )
)

saved_ck = ua_mod._try_ck_unified_attention
ua_mod._try_ck_unified_attention = lambda *a, **kw: False
saved_2d = ua_mod.use_2d_kernel

ua_mod.use_2d_kernel = lambda *a, **kw: True
t2d_ms = timed(lambda: ua_mod.unified_attention(**kw))

ua_mod.use_2d_kernel = lambda *a, **kw: False
t3d_ms = timed(lambda: ua_mod.unified_attention(**kw))

ua_mod.use_2d_kernel = saved_2d
ua_mod._try_ck_unified_attention = saved_ck

auto_ms = timed(lambda: ua_mod.unified_attention(**kw))

best_triton = min(t2d_ms, t3d_ms)

print(f"  CK-UA direct:    {ck_ms:.4f} ms")
print(f"  Triton 2D:       {t2d_ms:.4f} ms")
print(f"  Triton 3D:       {t3d_ms:.4f} ms")
print(f"  Auto (selector): {auto_ms:.4f} ms")
print()
print(f"  CK-UA / best Triton = {ck_ms/best_triton:.3f}x", end="")
if ck_ms < best_triton:
    print(f" (CK-UA {(1-ck_ms/best_triton)*100:.0f}% faster)")
else:
    print(f" (Triton {(ck_ms/best_triton-1)*100:.0f}% faster)")
print(f"  Selector picks: {'CK-UA' if abs(auto_ms - ck_ms)/ck_ms < 0.1 else 'Triton'}")

del q, k, v, cu, sl, bt, out
