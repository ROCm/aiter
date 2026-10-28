# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 KV-cache FlyDSL paged-attention decode vs a torch reference.

Shapes are MiniMax-M3 on TP4: one KV head per rank, 16 query heads per KV head,
head_dim 128, one query token per row, a sparse selection of at most
16 blocks x 128 tokens per row (the caller compacts it into a block table). The
K/V pages are single-region: ``[block_size x 64 B e2m1][block_size x 4 B e8m0]``.

Run:
    python op_tests/test_flydsl_pa_decode_mxfp4.py   (pytest checks, then the timing table)
    python op_tests/test_flydsl_pa_decode_mxfp4.py -r 1 64 -c 2048 -bs 128
    pytest op_tests/test_flydsl_pa_decode_mxfp4.py -q
"""

import argparse
import itertools
import sys

import pandas as pd
import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.pa_decode_mxfp4_kernel import DEFAULT_TILE_DEPTH
from aiter.ops.flydsl.kernels.pa_decode_plan import plan_pa_decode
from aiter.ops.flydsl.pa_decode import pa_decode
from aiter.ops.flydsl.pa_decode_mxfp4 import pa_decode_mxfp4
from aiter.ops.quant import per_1x32_f4_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

# v_mfma_scale_f32_16x16x128_f8f6f4 and ds_read_b64_tr_b16: CDNA4 only.
SUPPORTED_GFX = ["gfx950"]
pytestmark = pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX, reason="pa_decode_mxfp4 requires gfx950"
)

HEAD_DIM = 128
Q_MAX_FP8 = 448.0


# --------------------------------------------------------------------------
# MXFP4 pages
# --------------------------------------------------------------------------
def quantize_pages(x, block_size):
    """bf16 ``x [pages, kv_heads, block_size, 128]`` -> (pool, data, scale, deq).

    ``pool`` is the single-region byte page ``[pages, kv_heads, bs * 68]``
    (``[bs x 64 B data][bs x 4 B scale]``); ``data`` / ``scale`` are the two
    strided views the kernel takes; ``deq`` is the f32 dequantization
    (aiter's own ``mxfp4_to_f32`` / ``e8m0_to_f32``).
    """
    pages, heads, bs, _ = x.shape
    q, s = per_1x32_f4_quant(x.reshape(-1, HEAD_DIM))
    data = q.view(torch.uint8).reshape(pages, heads, bs, 64)
    scale = s.view(torch.uint8).reshape(pages, heads, bs, 4)
    pool = torch.empty(pages, heads, bs * 68, dtype=torch.uint8, device=x.device)
    pool[:, :, : bs * 64] = data.reshape(pages, heads, bs * 64)
    pool[:, :, bs * 64 :] = scale.reshape(pages, heads, bs * 4)
    f = fp4_utils.mxfp4_to_f32(data.reshape(-1, 64).clone())
    e = fp4_utils.e8m0_to_f32(scale.reshape(-1, 4)).repeat_interleave(32, dim=-1)
    deq = (f * e).reshape(pages, heads, bs, HEAD_DIM)
    return pool, *page_views(pool, bs), deq


def page_views(pool, bs):
    pages, heads, _ = pool.shape
    s0, s1 = pool.stride(0), pool.stride(1)
    data = torch.as_strided(pool, (pages, heads, bs, 64), (s0, s1, 64, 1))
    scale = torch.as_strided(
        pool, (pages, heads, bs, 4), (s0, s1, 4, 1), storage_offset=bs * 64
    )
    return data, scale


# --------------------------------------------------------------------------
# Case construction and the torch reference
# --------------------------------------------------------------------------
def make_case(ctx_lens, group, block_size, kv_heads=1, seed=0, extra_pages=3):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    ctx = torch.tensor(ctx_lens, dtype=torch.int32, device=dev)
    rows = len(ctx_lens)
    max_blocks = max(1, -(-max(ctx_lens) // block_size))
    pages_needed = rows * max_blocks
    pages = pages_needed + extra_pages
    # Random page ids; table entries past a row's context hold stale-but-valid ids.
    perm = torch.randperm(pages, device=dev, generator=g)
    bt = perm[:pages_needed].reshape(rows, max_blocks).to(torch.int32).contiguous()
    shape = (pages, kv_heads, block_size, HEAD_DIM)
    # Varying magnitudes per 32-block so a block<->scale mix-up cannot hide.
    mags = torch.tensor([1.0, 4.0, 0.25, 16.0], device=dev).repeat_interleave(32)
    k = (torch.randn(shape, device=dev, generator=g) * 2 * mags).to(torch.bfloat16)
    v = (torch.randn(shape, device=dev, generator=g) * mags).to(torch.bfloat16)
    q = (torch.randn(rows, kv_heads * group, HEAD_DIM, device=dev, generator=g) * 2).to(
        torch.bfloat16
    )
    return ctx, bt, k, v, q


def ref_attention(q_f32, k_deq, v_deq, bt, ctx, scale, block_size):
    """fp32 attention of every (row, q head) over its own paged tokens; P is
    rounded to bf16 before PV, as the kernel does."""
    rows, nq, _ = q_f32.shape
    kv_heads = k_deq.shape[1]
    group = nq // kv_heads
    out = torch.zeros(rows, nq, HEAD_DIM, dtype=torch.float32, device=q_f32.device)
    for r in range(rows):
        n = int(ctx[r])
        if n == 0:
            continue
        nb = -(-n // block_size)
        pg = bt[r, :nb].long()
        for h in range(kv_heads):
            kk = k_deq[pg, h].reshape(nb * block_size, HEAD_DIM)[:n]
            vv = v_deq[pg, h].reshape(nb * block_size, HEAD_DIM)[:n]
            qq = q_f32[r, h * group : (h + 1) * group]
            s = (qq @ kk.t()) * scale
            # the kernel rounds exp2(s - m) (max 1) to bf16 and divides by the
            # unrounded sum
            e = torch.exp(s - s.max(-1, keepdim=True).values)
            pn = e.to(torch.bfloat16).to(torch.float32)
            out[r, h * group : (h + 1) * group] = (pn @ vv) / e.sum(-1, keepdim=True)
    return out


def quantize_q_fp8(q_bf16, q_scale=1.0):
    """The writer's contract: round to the model dtype, divide by the per-tensor
    scale, RNE-saturate to e4m3. Returns (fp8, dequantized f32 incl. the scale)."""
    x = q_bf16.to(torch.float32) / q_scale
    q8 = x.clamp(-Q_MAX_FP8, Q_MAX_FP8).to(torch.float8_e4m3fn)
    return q8, q8.to(torch.float32) * q_scale


def rel_l2(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def run_mxfp4(
    q,
    kd,
    ks,
    vd,
    vs,
    bt,
    ctx,
    scale,
    *,
    q_scale=None,
    plan=None,
    tile_depth=DEFAULT_TILE_DEPTH,
):
    rows, nq, _ = q.shape
    out = torch.empty(rows, nq, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
    if plan is None:
        plan = plan_pa_decode(ctx, kd.shape[1])
    pa_decode_mxfp4(
        out,
        q,
        kd,
        ks,
        vd,
        vs,
        ctx,
        bt,
        scale,
        query_scale=q_scale,
        work_plan=plan,
        tile_depth=tile_depth,
    )
    return out


# --------------------------------------------------------------------------
# Correctness (pytest)
# --------------------------------------------------------------------------
CTX_SWEEP = [1, 15, 16, 17, 127, 128, 129, 255, 256, 257, 511, 2047, 2048]


@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize("group", [16, 8])
@pytest.mark.parametrize("qmode", ["bf16", "fp8"])
def test_pa_decode_mxfp4_ctx_sweep(block_size, group, qmode):
    ctx_lens = CTX_SWEEP
    ctx, bt, k, v, q = make_case(ctx_lens, group, block_size, seed=block_size + group)
    _, kd, ks, k_deq = quantize_pages(k, block_size)
    _, vd, vs, v_deq = quantize_pages(v, block_size)
    scale = HEAD_DIM**-0.5
    if qmode == "fp8":
        q_in, q_ref = quantize_q_fp8(q)
    else:
        q_in, q_ref = q, q.to(torch.float32)
    out = run_mxfp4(q_in, kd, ks, vd, vs, bt, ctx, scale)
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, scale, block_size)
    torch.cuda.synchronize()
    err = rel_l2(out.float(), ref)
    assert err < 3e-3, f"rel-L2 {err:.2e}"
    for r, n in enumerate(ctx_lens):
        e = rel_l2(out[r].float(), ref[r])
        assert e < 1e-2, f"ctx {n}: rel-L2 {e:.2e}"


@pytest.mark.parametrize("block_size", [16, 128])
def test_pa_decode_mxfp4_garbage_tail(block_size):
    """Rows past the context inside the last page, and table entries past the
    context, hold garbage (random bytes, e8m0 255 and 0): they must not leak."""
    ctx_lens = [1, 5, 13, 16 if block_size == 16 else 100]
    ctx, bt, k, v, q = make_case(ctx_lens, 16, block_size, seed=3)
    k_pool, kd, ks, k_deq = quantize_pages(k, block_size)
    v_pool, vd, vs, v_deq = quantize_pages(v, block_size)
    # garbage only where no sequence has a valid token: rebuild the pools with the
    # valid prefix kept and everything else scribbled
    for pool in (k_pool, v_pool):
        d, s = page_views(pool, block_size)
        for r, n in enumerate(ctx_lens):
            nb = -(-n // block_size)
            for b in range(nb):
                lo = n - b * block_size if b == nb - 1 else block_size
                pid = int(bt[r, b])
                g = torch.Generator(device=pool.device).manual_seed(11 + pid)
                for t in (d, s):
                    t[pid, :, lo:] = torch.randint(
                        0,
                        256,
                        t[pid, :, lo:].shape,
                        dtype=torch.uint8,
                        device=pool.device,
                        generator=g,
                    )
                if lo < block_size:
                    s[pid, :, lo, 0] = 255
                    s[pid, :, lo, 1] = 0
    scale = HEAD_DIM**-0.5
    q_in, q_ref = quantize_q_fp8(q)
    out = run_mxfp4(q_in, kd, ks, vd, vs, bt, ctx, scale)
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, scale, block_size)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert rel_l2(out.float(), ref) < 3e-3


def test_pa_decode_mxfp4_q_scale():
    ctx, bt, k, v, q = make_case([300, 2048, 64], 16, 128, seed=5)
    _, kd, ks, k_deq = quantize_pages(k, 128)
    _, vd, vs, v_deq = quantize_pages(v, 128)
    qs = 0.25
    q_in, q_ref = quantize_q_fp8(q, qs)
    q_scale = torch.tensor([qs], dtype=torch.float32, device=q.device)
    scale = HEAD_DIM**-0.5
    out = run_mxfp4(q_in, kd, ks, vd, vs, bt, ctx, scale, q_scale=q_scale)
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, scale, 128)
    torch.cuda.synchronize()
    assert rel_l2(out.float(), ref) < 3e-3


@pytest.mark.parametrize("kv_heads,group", [(2, 16), (4, 4), (1, 1), (1, 3)])
def test_pa_decode_mxfp4_heads(kv_heads, group):
    """Several KV heads (head stride in the views) and GQA groups that are not
    16 (padding heads alias head 0 and are never stored)."""
    ctx_lens = [1, 17, 300, 1025]
    ctx, bt, k, v, q = make_case(ctx_lens, group, 128, kv_heads=kv_heads, seed=9)
    _, kd, ks, k_deq = quantize_pages(k, 128)
    _, vd, vs, v_deq = quantize_pages(v, 128)
    scale = HEAD_DIM**-0.5
    q_in, q_ref = quantize_q_fp8(q)
    out = run_mxfp4(q_in, kd, ks, vd, vs, bt, ctx, scale)
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, scale, 128)
    torch.cuda.synchronize()
    assert rel_l2(out.float(), ref) < 3e-3


@pytest.mark.parametrize("max_partitions", [1, 2, 3, 8])
@pytest.mark.parametrize("tile_depth", [1, 2, 3, 4])
def test_pa_decode_mxfp4_task_shapes(max_partitions, tile_depth):
    """Tasks of 1..N plan tiles: waves with fewer tiles than the unroll depth
    (phantom tiles) and tasks that own several tiles per wave."""
    ctx_lens = [16, 255, 1000, 4097, 5000]
    ctx, bt, k, v, q = make_case(ctx_lens, 16, 16, seed=17)
    _, kd, ks, k_deq = quantize_pages(k, 16)
    _, vd, vs, v_deq = quantize_pages(v, 16)
    scale = HEAD_DIM**-0.5
    q_in, q_ref = quantize_q_fp8(q)
    plan = plan_pa_decode(ctx, 1, max_partitions=max_partitions)
    out = run_mxfp4(
        q_in, kd, ks, vd, vs, bt, ctx, scale, plan=plan, tile_depth=tile_depth
    )
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, scale, 16)
    torch.cuda.synchronize()
    assert rel_l2(out.float(), ref) < 3e-3


def test_pa_decode_mxfp4_default_depth_is_bitwise_depth4():
    """The tile pipeline depth changes scheduling only, never the arithmetic."""
    ctx, bt, k, v, q = make_case([1, 130, 777, 2048], 16, 128, seed=21)
    _, kd, ks, _ = quantize_pages(k, 128)
    _, vd, vs, _ = quantize_pages(v, 128)
    q_in, _ = quantize_q_fp8(q)
    scale = HEAD_DIM**-0.5
    a = run_mxfp4(q_in, kd, ks, vd, vs, bt, ctx, scale)
    b = run_mxfp4(q_in, kd, ks, vd, vs, bt, ctx, scale, tile_depth=4)
    torch.cuda.synchronize()
    assert torch.equal(a, b)


@pytest.mark.parametrize("rows", [1, 8, 64, 512])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_pa_decode_mxfp4_direct_output_is_bitwise_split_plus_reduce(rows, out_dtype):
    """One task per row (plan max_partitions == 1): the kernel writes the output
    itself. It must equal the split + reduce path bit for bit: the reduce of a
    single normalized partial only multiplies it by a weight within one f32 ulp of
    1.0, which cannot move a value that is already a bf16/f16 number."""
    ctx_lens = [(2048, 1, 129, 777, 1000, 17, 2048, 300)[i % 8] for i in range(rows)]
    ctx, bt, k, v, q = make_case(ctx_lens, 16, 128, seed=rows)
    _, kd, ks, k_deq = quantize_pages(k, 128)
    _, vd, vs, v_deq = quantize_pages(v, 128)
    q_in, q_ref = quantize_q_fp8(q)
    plan = plan_pa_decode(ctx, 1, max_partitions=1)
    assert plan.capacity == rows  # rows exactly equal to capacity
    outs = []
    for direct in (None, False, True):
        out = torch.full(
            (rows, 16, HEAD_DIM), float("nan"), dtype=out_dtype, device=q.device
        )
        pa_decode_mxfp4(
            out, q_in, kd, ks, vd, vs, ctx, bt, HEAD_DIM**-0.5,
            work_plan=plan, direct_output=direct,
        )  # fmt: skip
        outs.append(out)
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[2])  # None picks the direct path
    assert torch.equal(outs[1], outs[2])
    if rows <= 64:
        ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, HEAD_DIM**-0.5, 128)
        assert rel_l2(outs[2].float(), ref) < 3e-3


def test_pa_decode_mxfp4_direct_output_plan_boundaries():
    """capacity == rows only with max_partitions == 1 (or a budget that cannot
    split); a plan that splits keeps the reduce path and refuses a forced direct."""
    ctx_lens = [2048] * 4
    ctx, bt, k, v, q = make_case(ctx_lens, 16, 128, seed=4)
    _, kd, ks, _ = quantize_pages(k, 128)
    _, vd, vs, _ = quantize_pages(v, 128)
    q_in, _ = quantize_q_fp8(q)
    out = torch.empty_like(q)
    split_plan = plan_pa_decode(ctx, 1, max_partitions=4)
    assert split_plan.capacity > len(ctx_lens)
    with pytest.raises(ValueError, match="max_partitions == 1"):
        pa_decode_mxfp4(
            out, q_in, kd, ks, vd, vs, ctx, bt, 1.0,
            work_plan=split_plan, direct_output=True,
        )  # fmt: skip
    # default capacity == rows when rows >= 2 * CU, yet max_partitions stays CU:
    # not eligible for the direct path (an empty row could make a row split)
    big = plan_pa_decode(ctx, 1)
    a = run_mxfp4(q_in, kd, ks, vd, vs, bt, ctx, HEAD_DIM**-0.5, plan=big)
    b = run_mxfp4(
        q_in, kd, ks, vd, vs, bt, ctx, HEAD_DIM**-0.5,
        plan=plan_pa_decode(ctx, 1, max_partitions=1),
    )  # fmt: skip
    torch.cuda.synchronize()
    assert rel_l2(a.float(), b.float()) < 1e-2


def test_pa_decode_mxfp4_direct_output_strided_and_empty_rows():
    ctx_lens = [300, 0, 2048, 5]
    ctx, bt, k, v, q = make_case([max(n, 1) for n in ctx_lens], 16, 128, seed=8)
    ctx = torch.tensor(ctx_lens, dtype=torch.int32, device=q.device)
    _, kd, ks, k_deq = quantize_pages(k, 128)
    _, vd, vs, v_deq = quantize_pages(v, 128)
    q_in, q_ref = quantize_q_fp8(q)
    plan = plan_pa_decode(ctx, 1, max_partitions=1)
    # output rows padded to 2x: a strided view
    big = torch.full((4, 32, HEAD_DIM), 7.0, dtype=torch.bfloat16, device=q.device)
    out = big[:, :16]
    pa_decode_mxfp4(out, q_in, kd, ks, vd, vs, ctx, bt, HEAD_DIM**-0.5, work_plan=plan)
    torch.cuda.synchronize()
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, HEAD_DIM**-0.5, 128)
    for r in (0, 2, 3):
        assert rel_l2(out[r].float(), ref[r]) < 3e-3
    assert (big[:, 16:] == 7.0).all()  # nothing written outside the view
    # a context-0 row has no task: the direct path leaves it alone (documented)
    assert (out[1] == 7.0).all()
    # the reduce path zero-fills it
    out2 = torch.full_like(q, 7.0)
    pa_decode_mxfp4(
        out2, q_in, kd, ks, vd, vs, ctx, bt, HEAD_DIM**-0.5,
        work_plan=plan, direct_output=False,
    )  # fmt: skip
    assert (out2[1] == 0).all()


def test_pa_decode_mxfp4_direct_output_graph_replay():
    ctx_lens = [2048] * 8
    ctx, bt, k, v, q = make_case(ctx_lens, 16, 128, seed=12)
    _, kd, ks, k_deq = quantize_pages(k, 128)
    _, vd, vs, v_deq = quantize_pages(v, 128)
    q_in, q_ref = quantize_q_fp8(q)
    plan = plan_pa_decode(ctx, 1, max_partitions=1)
    out = torch.empty_like(q)

    def call():
        pa_decode_mxfp4(
            out, q_in, kd, ks, vd, vs, ctx, bt, HEAD_DIM**-0.5, work_plan=plan
        )

    call()  # compile outside capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    first = out.clone()
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, first)
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, HEAD_DIM**-0.5, 128)
    assert rel_l2(out.float(), ref) < 3e-3


def run_free(q, kd, ks, vd, vs, bt, ctx, scale, out_dtype=torch.bfloat16, **kw):
    out = torch.full(q.shape, 7.0, dtype=out_dtype, device=q.device)
    pa_decode_mxfp4(out, q, kd, ks, vd, vs, ctx, bt, scale, work_plan=None, **kw)
    return out


@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize("max_partitions", [None, 1, 2, 3, 8])
@pytest.mark.parametrize("qmode", ["bf16", "fp8"])
def test_pa_decode_mxfp4_plan_free_vs_reference(block_size, max_partitions, qmode):
    """No plan: every shape the planned path handles, with tasks of several tiles
    (max_partitions 3 on 5000 tokens), single tiles, and tails."""
    ctx_lens = [1, 16, 255, 256, 257, 1000, 2048, 4097, 5000]
    ctx, bt, k, v, q = make_case(ctx_lens, 16, block_size, seed=block_size + 3)
    _, kd, ks, k_deq = quantize_pages(k, block_size)
    _, vd, vs, v_deq = quantize_pages(v, block_size)
    scale = HEAD_DIM**-0.5
    q_in, q_ref = quantize_q_fp8(q) if qmode == "fp8" else (q, q.float())
    out = run_free(q_in, kd, ks, vd, vs, bt, ctx, scale, max_partitions=max_partitions)
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, scale, block_size)
    torch.cuda.synchronize()
    assert rel_l2(out.float(), ref) < 3e-3
    for r, n in enumerate(ctx_lens):
        assert rel_l2(out[r].float(), ref[r]) < 1e-2, f"ctx {n}"


@pytest.mark.parametrize("rows", [1, 8, 64, 512])
def test_pa_decode_mxfp4_plan_free_is_bitwise_the_planned_path(rows):
    """Where the plan and the plan-free split coincide (one 256-token tile per
    task: ctx 2048, 8 tasks; or one task per row) the outputs are the same bits."""
    ctx, bt, k, v, q = make_case([2048] * rows, 16, 128, seed=rows + 40)
    _, kd, ks, _ = quantize_pages(k, 128)
    _, vd, vs, _ = quantize_pages(v, 128)
    q_in, _ = quantize_q_fp8(q)
    scale = HEAD_DIM**-0.5
    for mp in (1, 8):
        plan = plan_pa_decode(ctx, 1, max_partitions=mp)
        if plan.capacity != rows * mp:  # the planner's task budget (2 * CUs) caps it
            continue
        planned = run_mxfp4(q_in, kd, ks, vd, vs, bt, ctx, scale, plan=plan)
        free = run_free(q_in, kd, ks, vd, vs, bt, ctx, scale, max_partitions=mp)
        torch.cuda.synchronize()
        assert torch.equal(planned, free), f"rows {rows} max_partitions {mp}"
    # the default split is the planner's budget: 8 tasks up to 64 rows, 1 from 512
    default = run_free(q_in, kd, ks, vd, vs, bt, ctx, scale)
    want = 1 if rows >= 512 else 8
    explicit = run_free(q_in, kd, ks, vd, vs, bt, ctx, scale, max_partitions=want)
    torch.cuda.synchronize()
    assert torch.equal(default, explicit)


@pytest.mark.parametrize("max_partitions", [None, 1, 4])
def test_pa_decode_mxfp4_plan_free_empty_rows(max_partitions):
    """A context-0 row has no work but is still written (zeros), with the reduce
    kernel (>1 partition) and without it (1 partition)."""
    ctx_lens = [300, 0, 2048, 0, 5]
    ctx, bt, k, v, q = make_case([max(n, 1) for n in ctx_lens], 16, 128, seed=8)
    ctx = torch.tensor(ctx_lens, dtype=torch.int32, device=q.device)
    _, kd, ks, k_deq = quantize_pages(k, 128)
    _, vd, vs, v_deq = quantize_pages(v, 128)
    q_in, q_ref = quantize_q_fp8(q)
    out = run_free(
        q_in, kd, ks, vd, vs, bt, ctx, HEAD_DIM**-0.5, max_partitions=max_partitions
    )
    torch.cuda.synchronize()
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, HEAD_DIM**-0.5, 128)
    for r, n in enumerate(ctx_lens):
        if n:
            assert rel_l2(out[r].float(), ref[r]) < 3e-3
        else:
            assert (out[r] == 0).all(), f"row {r}"


@pytest.mark.parametrize("rows", [255, 256, 257])
def test_pa_decode_mxfp4_plan_free_default_split_boundaries(rows):
    """The default split is max(1, 2 * 256 CUs // rows): 2 tasks at 255/256 rows, 1 at
    257. Both sides of the switch agree with each other and with the reference."""
    ctx, bt, k, v, q = make_case([600] * rows, 16, 128, seed=rows)
    _, kd, ks, k_deq = quantize_pages(k, 128)
    _, vd, vs, v_deq = quantize_pages(v, 128)
    q_in, q_ref = quantize_q_fp8(q)
    scale = HEAD_DIM**-0.5
    out = run_free(q_in, kd, ks, vd, vs, bt, ctx, scale)
    out1 = run_free(q_in, kd, ks, vd, vs, bt, ctx, scale, max_partitions=1)
    torch.cuda.synchronize()
    assert rel_l2(out.float(), out1.float()) < 1e-2
    ref = ref_attention(q_ref[:8], k_deq, v_deq, bt[:8], ctx[:8], scale, 128)
    assert rel_l2(out[:8].float(), ref) < 3e-3


@pytest.mark.parametrize("max_partitions", [1, 8])
def test_pa_decode_mxfp4_plan_free_graph_replay(max_partitions):
    ctx, bt, k, v, q = make_case([2048, 700, 2048, 33], 16, 128, seed=14)
    _, kd, ks, k_deq = quantize_pages(k, 128)
    _, vd, vs, v_deq = quantize_pages(v, 128)
    q_in, q_ref = quantize_q_fp8(q)
    out = torch.empty_like(q)

    def call():
        pa_decode_mxfp4(
            out, q_in, kd, ks, vd, vs, ctx, bt, HEAD_DIM**-0.5,
            max_partitions=max_partitions,
        )  # fmt: skip

    call()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    first = out.clone()
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, first)
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, HEAD_DIM**-0.5, 128)
    assert rel_l2(out.float(), ref) < 3e-3


def test_pa_decode_mxfp4_plan_free_argument_checks():
    ctx, bt, k, v, q = make_case([2048], 16, 16, seed=2, extra_pages=0)
    _, kd, ks, _ = quantize_pages(k, 16)
    _, vd, vs, _ = quantize_pages(v, 16)
    q_in, _ = quantize_q_fp8(q)
    out = torch.empty_like(q)
    with pytest.raises(ValueError, match="one task per sequence"):
        pa_decode_mxfp4(
            out, q_in, kd, ks, vd, vs, ctx, bt, 1.0,
            max_partitions=2, direct_output=True,
        )  # fmt: skip
    with pytest.raises(ValueError, match="plan-free"):
        pa_decode_mxfp4(
            out, q_in, kd, ks, vd, vs, ctx, bt, 1.0,
            work_plan=plan_pa_decode(ctx, 1), max_partitions=2,
        )  # fmt: skip
    ctx, bt, k, v, q = make_case([40000], 16, 16, seed=2, extra_pages=0)
    _, kd, ks, _ = quantize_pages(k, 16)
    _, vd, vs, _ = quantize_pages(v, 16)
    q_in, _ = quantize_q_fp8(q)
    with pytest.raises(NotImplementedError, match="max_partitions"):
        run_free(q_in, kd, ks, vd, vs, bt, ctx, 1.0, max_partitions=1)


def test_pa_decode_mxfp4_rejects_oversized_task():
    ctx, bt, k, v, q = make_case([40000], 16, 16, seed=2, extra_pages=0)
    _, kd, ks, _ = quantize_pages(k, 16)
    _, vd, vs, _ = quantize_pages(v, 16)
    q_in, _ = quantize_q_fp8(q)
    plan = plan_pa_decode(ctx, 1, max_partitions=1)
    with pytest.raises(NotImplementedError, match="max_partitions"):
        run_mxfp4(q_in, kd, ks, vd, vs, bt, ctx, 1.0, plan=plan)


# --------------------------------------------------------------------------
# Timing: MXFP4 KV (this kernel) vs the FP8 KV pa_decode at the same shape
# --------------------------------------------------------------------------
def fp8_inputs(k, v, block_size):
    """Per-token fp8 K/V in pa_decode's shuffled layouts (x = 16)."""
    pages, heads, bs, d = k.shape
    out = []
    scales = []
    for x in (k, v):
        amax = x.abs().amax(dim=-1).to(torch.float32)
        sc = (amax / 448.0).clamp_min(1e-12)
        q8 = (x.to(torch.float32) / sc[..., None]).to(torch.float8_e4m3fn)
        out.append(q8)
        scales.append(sc)
    k8, v8 = out
    key = k8.reshape(pages, heads, bs, d // 16, 16).permute(0, 1, 3, 2, 4).contiguous()
    val = v8.reshape(pages, heads, bs // 16, 16, d).permute(0, 1, 2, 4, 3).contiguous()
    return key, val, scales[0].contiguous(), scales[1].contiguous()


ATOL = RTOL = 1e-2
# K + V bytes per token: MXFP4 data + e8m0 scales; fp8 data + fp32 per-token scales.
MXFP4_TOKEN_BYTES = 2 * (HEAD_DIM // 2 + HEAD_DIM // 32)
FP8_TOKEN_BYTES = 2 * (HEAD_DIM + 4)


@benchmark()
def bench_pa_decode_mxfp4(rows, ctx_len, group, block_size):
    ctx, bt, k, v, q = make_case([ctx_len] * rows, group, block_size, seed=1)
    _, kd, ks, k_deq = quantize_pages(k, block_size)
    _, vd, vs, v_deq = quantize_pages(v, block_size)
    scale = HEAD_DIM**-0.5
    q8, q_ref8 = quantize_q_fp8(q)
    plan = plan_pa_decode(ctx, 1)
    # name -> (fn, fp32 reference or None, K+V bytes per token)
    candidates = {
        "mxfp4_qfp8": (
            lambda: run_mxfp4(q8, kd, ks, vd, vs, bt, ctx, scale, plan=plan),
            ref_attention(q_ref8, k_deq, v_deq, bt, ctx, scale, block_size),
            MXFP4_TOKEN_BYTES,
        ),
        "mxfp4_qbf16": (
            lambda: run_mxfp4(q, kd, ks, vd, vs, bt, ctx, scale, plan=plan),
            ref_attention(q.float(), k_deq, v_deq, bt, ctx, scale, block_size),
            MXFP4_TOKEN_BYTES,
        ),
    }
    if block_size in (16, 128):  # the page sizes pa_decode serves
        key, val, kscale, vscale = fp8_inputs(k, v, block_size)
        out8 = torch.empty_like(q)

        def run_fp8():
            pa_decode(
                out8, q, key, val, ctx, bt, scale, 1,
                key_scale=kscale, value_scale=vscale, work_plan=plan,
            )  # fmt: skip
            return out8

        # Timing baseline only, no reference: test_flydsl_pa_decode.py checks it, and
        # on this data it lands ~1e-1 (rel L2) off an fp32 attention over its own
        # dequantized K/V, far outside the MXFP4 tolerance below.
        candidates["fp8kv"] = (run_fp8, None, FP8_TOKEN_BYTES)

    flops = 4 * HEAD_DIM * q.shape[1] * rows * ctx_len  # QK + PV
    ret = {"gfx": get_gfx()}
    for name, (fn, ref, token_bytes) in candidates.items():
        out, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = rows * ctx_len * token_bytes / us / 1e6
        if ref is not None:
            ret[f"{name} err"] = checkAllclose(
                ref,
                out.float(),
                rtol=RTOL,
                atol=ATOL,
                msg=f"{name}: rows={rows} ctx={ctx_len} group={group} bs={block_size}",
            )
    return ret


bench_pa_decode_mxfp4.__test__ = False


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("pa_decode_mxfp4 unsupported on %s; skipping", get_gfx())
        return 0
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MXFP4 KV-cache paged-attention decode correctness and perf",
    )
    parser.add_argument(
        "-r",
        "--rows",
        type=int,
        nargs="*",
        default=[1, 8, 64, 512],
        help="""Decode rows (one query token each).
    e.g.: -r 1 4096""",
    )
    parser.add_argument(
        "-c",
        "--ctx",
        type=int,
        nargs="*",
        default=[2048],
        help="""Context length per row.
    e.g.: -c 2048""",
    )
    parser.add_argument(
        "-g",
        "--group",
        type=int,
        nargs="*",
        default=[16],
        help="""Query heads per KV head (at most 16).
    e.g.: -g 16 8""",
    )
    parser.add_argument(
        "-bs",
        "--block-size",
        type=int,
        nargs="*",
        default=[16, 128],
        help="""KV page size in tokens (16, 64 or 128; the fp8 baseline runs at 16 and 128).
    e.g.: -bs 128""",
    )
    args = parser.parse_args()

    aiter.logger.info("pa_decode_mxfp4: running correctness checks...")
    status = pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    if status != pytest.ExitCode.OK:
        aiter.logger.error(
            "pa_decode_mxfp4: checks failed (pytest exit %s); skipping perf sweep",
            int(status),
        )
        return 1

    df = []
    for rows, ctx_len, group, block_size in itertools.product(
        args.rows, args.ctx, args.group, args.block_size
    ):
        df.append(bench_pa_decode_mxfp4(rows, ctx_len, group, block_size))
    df = pd.DataFrame(df)
    aiter.logger.info(
        "pa_decode_mxfp4 summary (markdown):\n%s", df.to_markdown(index=False)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
