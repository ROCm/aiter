"""Standalone stage1-only perf comparison: asm sparse decode kernel vs ATOM's
triton split kernel (bf16).

- asm stage1   = the `..._sparse` .co kernel (fp8, its native dtype), timed alone.
- triton stage1 = ATOM `_paged_decode_split_kernel` (bf16), forced via kv_splits>1.

Only the stage1 GPU kernel device-time is compared (profiler filtered by name).
NOT bit-exact / not memory-fair: triton is bf16 while asm is fp8, and D=512 here
(MLA QK is really 576-wide, so triton QK is ~11% under-counted). Perf only.

Usage:
  ENABLE_CK=0 python op_tests/_stage1_bench.py            # default sweep
"""
import sys, os
sys.path.insert(0, "op_tests")
sys.path.insert(0, "/home/amd/feifei/ATOM")

import torch
import test_mla_v4_kargpreld as T
import aiter, aiter.mla
from aiter import dtypes
from aiter.test_common import run_perftest
from atom.model_ops.v4_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton as triton_decode,
)

Q = 1
SINK = True

# Mirror test_mla_v4_kargpreld.py sweep grids + active kernel variants.
_GQA_LIST = [64, 128]
_CTX_LENS = [256, 512, 1024]
_BATCH_SIZES = [64]
_SPLITS = [1]

# Perf iteration counts for run_perftest (mirrors test_mla_v4_kargpreld.py's
# _PERF usage). Kept as module constants so both bench fns share them.
_PERF = {"num_iters": 50, "num_warmup": 3}


def bench_asm_stage1(batch, ctx, split, gqa):
    inp = T._build_bf16_inputs(batch=batch, kv_seq_lens=ctx, q_seq_logical=Q,
                               seed=0, gqa_ratio=gqa, attn_sink=SINK)
    sm = 1.0 / (T._QUANT_D ** 0.5)
    qp, qr = T._native_to_2buff_for_asm(inp["q_bf16"])
    kvp, kvr = T._native_to_2buff_for_asm(inp["kv_bf16"])
    total_q = inp["q_bf16"].size(0)
    ns = inp["qo_indptr"].size(0) - 1
    nh = T.NUM_KV_HEADS * gqa
    dev = "cuda"
    ob = torch.empty((total_q, gqa, T.V_HEAD_DIM), dtype=dtypes.bf16, device=dev)
    sidx = torch.tensor([i * split for i in range(ns + 1)], dtype=torch.int32, device=dev)
    lb = torch.empty((total_q, split, nh, T.V_HEAD_DIM), dtype=dtypes.fp32, device=dev)
    eb = torch.empty((total_q, split, nh, 1), dtype=dtypes.fp32, device=dev)
    kw = dict(
        q=qp, qrope=qr.contiguous(), kv_buffer=kvp, kvrope=kvr.contiguous(), output=ob,
        qo_indptr=inp["qo_indptr"], kv_indptr=inp["kv_indptr"],
        kv_page_indices=inp["kv_page_indices"], kv_last_page_lens=inp["kv_last_page_lens"],
        split_indptr=sidx, max_seqlen_q=inp["max_seqlen_q"], sink=inp["sink"],
        sm_scale=sm, out_16_nosplit=0, num_kv_splits=split, logits=lb, attn_lse=eb,
    )
    _, us = run_perftest(
        aiter.mla.mla_decode_fwd_v4_nm,
        **kw,
        num_iters=_PERF["num_iters"],
        num_warmup=_PERF["num_warmup"],
    )
    return us


def bench_triton_stage1(batch, ctx, split, gqa):
    dev = "cuda"
    D = T.V_HEAD_DIM  # 512
    Tn = batch  # one decode token per sequence
    H = gqa
    q = torch.randn((Tn, H, D), dtype=torch.bfloat16, device=dev)
    total_pages = batch * ctx
    unified_kv = torch.randn((total_pages, D), dtype=torch.bfloat16, device=dev) * 0.1
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=dev)
    kv_indptr = torch.arange(0, (Tn + 1) * ctx, ctx, dtype=torch.int32, device=dev)
    attn_sink = torch.randn((H,), dtype=torch.float32, device=dev)
    sm = 1.0 / (D ** 0.5)
    _, us = run_perftest(
        triton_decode,
        q, unified_kv, kv_indices, kv_indptr, attn_sink, sm, kv_splits=split,
        num_iters=_PERF["num_iters"],
        num_warmup=_PERF["num_warmup"],
    )
    return us


import itertools

print(f"{'gqa':>4} {'batch':>6} {'ctx':>6} {'split':>6} | {'asm_stage1_us':>14} {'triton_stage1_us':>17} {'triton/asm':>11}")
print("-" * 80)
# gqa outermost so rows mirror the kargpreld summary-table grouping.
for gqa, batch, ctx, split in itertools.product(_GQA_LIST, _BATCH_SIZES, _CTX_LENS, _SPLITS):
    asm_us = bench_asm_stage1(batch, ctx, split, gqa)
    tri_us = bench_triton_stage1(batch, ctx, split, gqa)
    ratio = tri_us / asm_us if asm_us > 0 else float("nan")
    print(f"{gqa:>4} {batch:>6} {ctx:>6} {split:>6} | {asm_us:>14.2f} {tri_us:>17.2f} {ratio:>10.2f}x")
