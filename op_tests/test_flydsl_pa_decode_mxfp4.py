# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 KV-cache FlyDSL paged-attention decode vs a torch reference.

Shapes are MiniMax-M3 on TP4: one KV head per rank, 16 query heads per KV head,
head_dim 128, one query token per row, a sparse selection of at most
16 blocks x 128 tokens per row (the caller compacts it into a block table). The
K/V pages are single-region: ``[block_size x 64 B e2m1][block_size x 4 B e8m0]``.

Run:
    python op_tests/test_flydsl_pa_decode_mxfp4.py   (correctness checks, then the timing table)
    python op_tests/test_flydsl_pa_decode_mxfp4.py -r 1 64 -c 2048 -bs 128
"""

import argparse
import itertools
import re
import sys

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.pa_decode_plan import plan_pa_decode
from aiter.ops.flydsl.pa_decode_mxfp4 import pa_decode_mxfp4
from aiter.ops.quant import per_1x32_f4_quant
from aiter.test_common import assertAllclose, benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

# v_mfma_scale_f32_16x16x128_f8f6f4 and ds_read_b64_tr_b16: CDNA4 only.
SUPPORTED_GFX = ["gfx950"]

HEAD_DIM = 128
Q_MAX_FP8 = 448.0
ATOL = RTOL = 1e-2
TOL_ERR_RATIO = 1e-2  # bf16 output vs the fp32 reference: a few outliers, never a row


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


def quantize_q_fp8(q_bf16):
    """The writer's contract: round to the model dtype, RNE-saturate to e4m3 (scale
    1). Returns (fp8, dequantized f32)."""
    q8 = q_bf16.to(torch.float32).clamp(-Q_MAX_FP8, Q_MAX_FP8).to(torch.float8_e4m3fn)
    return q8, q8.to(torch.float32)


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
    plan=None,
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
        work_plan=plan,
    )
    return out


# --------------------------------------------------------------------------
# Correctness: plain asserts; a failure raises and the script exits non-zero
# --------------------------------------------------------------------------
def expect_error(exc, match, fn, *args, **kwargs):
    """``fn`` must raise ``exc`` whose message matches the regex ``match``."""
    try:
        fn(*args, **kwargs)
    except exc as e:
        assert re.search(match, str(e)), f"{exc.__name__} message: {e}"
    else:
        raise AssertionError(f"{exc.__name__} ({match!r}) was not raised")


def check_close(out, ref, msg=""):
    """assertAllclose against the torch reference: fails the script when more than
    TOL_ERR_RATIO of the elements are outside rtol/atol; returns that ratio."""
    return assertAllclose(
        ref.float(),
        out.float(),
        rtol=RTOL,
        atol=ATOL,
        tol_err_ratio=TOL_ERR_RATIO,
        msg=msg,
    )


CTX_SWEEP = [1, 15, 16, 17, 127, 128, 129, 255, 256, 257, 511, 2047, 2048]


def test_ctx_sweep(block_size, group):
    ctx_lens = CTX_SWEEP
    ctx, bt, k, v, q = make_case(ctx_lens, group, block_size, seed=block_size + group)
    _, kd, ks, k_deq = quantize_pages(k, block_size)
    _, vd, vs, v_deq = quantize_pages(v, block_size)
    scale = HEAD_DIM**-0.5
    q_in, q_ref = quantize_q_fp8(q)
    out = run_mxfp4(q_in, kd, ks, vd, vs, bt, ctx, scale)
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, scale, block_size)
    torch.cuda.synchronize()
    return {"err": check_close(out, ref)}


def test_garbage_tail(block_size):
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
    return {"err": check_close(out, ref)}


def test_heads(kv_heads, group):
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
    return {"err": check_close(out, ref)}


def test_task_shapes(max_partitions):
    """Tasks of 1..N plan tiles: waves with fewer tiles than the unroll depth
    (phantom tiles) and tasks that own several tiles per wave."""
    ctx_lens = [16, 255, 1000, 4097, 5000]
    ctx, bt, k, v, q = make_case(ctx_lens, 16, 16, seed=17)
    _, kd, ks, k_deq = quantize_pages(k, 16)
    _, vd, vs, v_deq = quantize_pages(v, 16)
    scale = HEAD_DIM**-0.5
    q_in, q_ref = quantize_q_fp8(q)
    plan = plan_pa_decode(ctx, 1, max_partitions=max_partitions)
    out = run_mxfp4(q_in, kd, ks, vd, vs, bt, ctx, scale, plan=plan)
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, scale, 16)
    torch.cuda.synchronize()
    return {"err": check_close(out, ref)}


def test_one_task_per_row(rows, out_dtype):
    """Plan max_partitions == 1: the kernel writes every output row itself, in
    bf16 and f16."""
    ctx_lens = [(2048, 1, 129, 777, 1000, 17, 2048, 300)[i % 8] for i in range(rows)]
    ctx, bt, k, v, q = make_case(ctx_lens, 16, 128, seed=rows)
    _, kd, ks, k_deq = quantize_pages(k, 128)
    _, vd, vs, v_deq = quantize_pages(v, 128)
    q_in, q_ref = quantize_q_fp8(q)
    plan = plan_pa_decode(ctx, 1, max_partitions=1)
    assert plan.capacity == rows  # rows exactly equal to capacity
    out = torch.full(
        (rows, 16, HEAD_DIM), float("nan"), dtype=out_dtype, device=q.device
    )
    pa_decode_mxfp4(out, q_in, kd, ks, vd, vs, ctx, bt, HEAD_DIM**-0.5, work_plan=plan)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()  # every row was written
    if rows > 64:
        return {}
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, HEAD_DIM**-0.5, 128)
    return {"err": check_close(out, ref)}


def test_default_plan_vs_one_task_per_row():
    """With rows >= 2 * CUs the default plan has capacity == rows yet splits: it is
    not eligible for the direct path, and agrees with the one-task plan."""
    ctx_lens = [2048] * 4
    ctx, bt, k, v, q = make_case(ctx_lens, 16, 128, seed=4)
    _, kd, ks, _ = quantize_pages(k, 128)
    _, vd, vs, _ = quantize_pages(v, 128)
    q_in, _ = quantize_q_fp8(q)
    a = run_mxfp4(
        q_in, kd, ks, vd, vs, bt, ctx, HEAD_DIM**-0.5, plan=plan_pa_decode(ctx, 1)
    )
    b = run_mxfp4(
        q_in, kd, ks, vd, vs, bt, ctx, HEAD_DIM**-0.5,
        plan=plan_pa_decode(ctx, 1, max_partitions=1),
    )  # fmt: skip
    torch.cuda.synchronize()
    return {"err": check_close(a, b)}


def test_direct_output_strided_and_empty_rows():
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
    err = max(check_close(out[r], ref[r]) for r in (0, 2, 3))
    assert (big[:, 16:] == 7.0).all()  # nothing written outside the view
    # a context-0 row has no task: the direct path leaves it alone (documented)
    assert (out[1] == 7.0).all()
    # the reduce path (a splitting plan) zero-fills it
    out2 = torch.full_like(q, 7.0)
    pa_decode_mxfp4(
        out2, q_in, kd, ks, vd, vs, ctx, bt, HEAD_DIM**-0.5,
        work_plan=plan_pa_decode(ctx, 1, max_partitions=2),
    )  # fmt: skip
    assert (out2[1] == 0).all()
    return {"err": err}


def replay_matches(call, out, ref):
    """Capture ``call`` into a CUDA graph, replay it over a zeroed ``out`` and check
    the replay reproduces the eager result and the reference."""
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
    return check_close(out, ref)


def test_direct_output_graph_replay():
    ctx_lens = [2048] * 8
    ctx, bt, k, v, q = make_case(ctx_lens, 16, 128, seed=12)
    _, kd, ks, k_deq = quantize_pages(k, 128)
    _, vd, vs, v_deq = quantize_pages(v, 128)
    q_in, q_ref = quantize_q_fp8(q)
    plan = plan_pa_decode(ctx, 1, max_partitions=1)
    out = torch.empty_like(q)
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, HEAD_DIM**-0.5, 128)

    def call():
        pa_decode_mxfp4(
            out, q_in, kd, ks, vd, vs, ctx, bt, HEAD_DIM**-0.5, work_plan=plan
        )

    return {"err": replay_matches(call, out, ref)}


def run_free(q, kd, ks, vd, vs, bt, ctx, scale, out_dtype=torch.bfloat16, **kw):
    out = torch.full(q.shape, 7.0, dtype=out_dtype, device=q.device)
    pa_decode_mxfp4(out, q, kd, ks, vd, vs, ctx, bt, scale, work_plan=None, **kw)
    return out


def test_plan_free_vs_reference(block_size):
    """No plan: one task per row, from single tiles to 20 tiles (5000 tokens),
    with tails."""
    ctx_lens = [1, 16, 255, 256, 257, 1000, 2048, 4097, 5000]
    ctx, bt, k, v, q = make_case(ctx_lens, 16, block_size, seed=block_size + 3)
    _, kd, ks, k_deq = quantize_pages(k, block_size)
    _, vd, vs, v_deq = quantize_pages(v, block_size)
    scale = HEAD_DIM**-0.5
    q_in, q_ref = quantize_q_fp8(q)
    out = run_free(q_in, kd, ks, vd, vs, bt, ctx, scale)
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, scale, block_size)
    torch.cuda.synchronize()
    return {"err": check_close(out, ref)}


def test_plan_free_is_bitwise_the_planned_path(rows):
    """A one-task plan and the plan-free form run the same task: the same bits."""
    ctx, bt, k, v, q = make_case([2048] * rows, 16, 128, seed=rows + 40)
    _, kd, ks, _ = quantize_pages(k, 128)
    _, vd, vs, _ = quantize_pages(v, 128)
    q_in, _ = quantize_q_fp8(q)
    scale = HEAD_DIM**-0.5
    plan = plan_pa_decode(ctx, 1, max_partitions=1)
    planned = run_mxfp4(q_in, kd, ks, vd, vs, bt, ctx, scale, plan=plan)
    free = run_free(q_in, kd, ks, vd, vs, bt, ctx, scale)
    torch.cuda.synchronize()
    assert torch.equal(planned, free)
    return {}


def test_plan_free_empty_rows():
    """A context-0 row has no work but is still written (zeros)."""
    ctx_lens = [300, 0, 2048, 0, 5]
    ctx, bt, k, v, q = make_case([max(n, 1) for n in ctx_lens], 16, 128, seed=8)
    ctx = torch.tensor(ctx_lens, dtype=torch.int32, device=q.device)
    _, kd, ks, k_deq = quantize_pages(k, 128)
    _, vd, vs, v_deq = quantize_pages(v, 128)
    q_in, q_ref = quantize_q_fp8(q)
    out = run_free(q_in, kd, ks, vd, vs, bt, ctx, HEAD_DIM**-0.5)
    torch.cuda.synchronize()
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, HEAD_DIM**-0.5, 128)
    err = 0.0
    for r, n in enumerate(ctx_lens):
        if n:
            err = max(err, check_close(out[r], ref[r]))
        else:
            assert (out[r] == 0).all(), f"row {r}"
    return {"err": err}


def test_plan_free_graph_replay():
    ctx, bt, k, v, q = make_case([2048, 700, 2048, 33], 16, 128, seed=14)
    _, kd, ks, k_deq = quantize_pages(k, 128)
    _, vd, vs, v_deq = quantize_pages(v, 128)
    q_in, q_ref = quantize_q_fp8(q)
    out = torch.empty_like(q)
    ref = ref_attention(q_ref, k_deq, v_deq, bt, ctx, HEAD_DIM**-0.5, 128)

    def call():
        pa_decode_mxfp4(out, q_in, kd, ks, vd, vs, ctx, bt, HEAD_DIM**-0.5)

    return {"err": replay_matches(call, out, ref)}


def test_plan_free_argument_checks():
    ctx, bt, k, v, q = make_case([2048], 16, 16, seed=2, extra_pages=0)
    _, kd, ks, _ = quantize_pages(k, 16)
    _, vd, vs, _ = quantize_pages(v, 16)
    q_in, _ = quantize_q_fp8(q)
    out = torch.empty_like(q)
    expect_error(
        NotImplementedError, "one task per sequence",
        pa_decode_mxfp4, out, q_in, kd, ks, vd, vs, ctx, bt, 1.0, max_partitions=2,
    )  # fmt: skip
    expect_error(
        ValueError, "plan-free",
        pa_decode_mxfp4, out, q_in, kd, ks, vd, vs, ctx, bt, 1.0,
        work_plan=plan_pa_decode(ctx, 1), max_partitions=2,
    )  # fmt: skip
    ctx, bt, k, v, q = make_case([40000], 16, 16, seed=2, extra_pages=0)
    _, kd, ks, _ = quantize_pages(k, 16)
    _, vd, vs, _ = quantize_pages(v, 16)
    q_in, _ = quantize_q_fp8(q)
    expect_error(
        NotImplementedError, "max_partitions",
        run_free, q_in, kd, ks, vd, vs, bt, ctx, 1.0,
    )  # fmt: skip
    return {}


def test_output_device_and_empty_batch():
    ctx, bt, k, v, q = make_case([16], 16, 16, seed=2)
    _, kd, ks, _ = quantize_pages(k, 16)
    _, vd, vs, _ = quantize_pages(v, 16)
    q_in, _ = quantize_q_fp8(q)
    expect_error(
        ValueError, "output must be on",
        pa_decode_mxfp4, q.cpu(), q_in, kd, ks, vd, vs, ctx, bt, 1.0, max_partitions=1,
    )  # fmt: skip
    # zero sequences: nothing to do, in particular no division by num_seqs
    empty = torch.empty(0, 16, HEAD_DIM, dtype=torch.bfloat16)
    pa_decode_mxfp4(
        empty, q_in[:0], kd, ks, vd, vs, ctx[:0], bt[:0], 1.0, work_plan=None
    )
    return {}


def test_rejects_oversized_task():
    ctx, bt, k, v, q = make_case([40000], 16, 16, seed=2, extra_pages=0)
    _, kd, ks, _ = quantize_pages(k, 16)
    _, vd, vs, _ = quantize_pages(v, 16)
    q_in, _ = quantize_q_fp8(q)
    plan = plan_pa_decode(ctx, 1, max_partitions=1)
    expect_error(
        NotImplementedError, "max_partitions",
        run_mxfp4, q_in, kd, ks, vd, vs, bt, ctx, 1.0, plan=plan,
    )  # fmt: skip
    return {}


def grid(**axes):
    return [dict(zip(axes, vals)) for vals in itertools.product(*axes.values())]


# (check, its cases): every case runs, in this order.
CHECKS = [
    (test_ctx_sweep, grid(block_size=[16, 64, 128], group=[16, 8])),
    (test_garbage_tail, grid(block_size=[16, 128])),
    (
        test_heads,
        [{"kv_heads": h, "group": g} for h, g in ((2, 16), (4, 4), (1, 1), (1, 3))],
    ),
    (test_task_shapes, grid(max_partitions=[1, 2, 3, 8])),
    (
        test_one_task_per_row,
        grid(rows=[1, 8, 64, 512], out_dtype=[torch.bfloat16, torch.float16]),
    ),
    (test_default_plan_vs_one_task_per_row, [{}]),
    (test_direct_output_strided_and_empty_rows, [{}]),
    (test_direct_output_graph_replay, [{}]),
    (test_plan_free_vs_reference, grid(block_size=[16, 64, 128])),
    (test_plan_free_is_bitwise_the_planned_path, grid(rows=[1, 8, 64, 512])),
    (test_plan_free_empty_rows, [{}]),
    (test_plan_free_graph_replay, [{}]),
    (test_plan_free_argument_checks, [{}]),
    (test_output_device_and_empty_batch, [{}]),
    (test_rejects_oversized_task, [{}]),
]


def run_checks():
    rows = []
    for fn, cases in CHECKS:
        for case in cases:
            ret = fn(**case)  # asserts inside; the mismatch ratio it measured, if any
            params = " ".join(f"{k}={v}" for k, v in case.items())
            rows.append({"check": fn.__name__[5:], "case": params, **ret})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Timing: pa_decode_mxfp4 (MXFP4 KV cache), the one candidate
# --------------------------------------------------------------------------
# K + V bytes per token: MXFP4 data + e8m0 scales.
TOKEN_BYTES = 2 * (HEAD_DIM // 2 + HEAD_DIM // 32)


@benchmark()
def bench_pa_decode_mxfp4(rows, ctx_len, group, block_size):
    ctx, bt, k, v, q = make_case([ctx_len] * rows, group, block_size, seed=1)
    _, kd, ks, k_deq = quantize_pages(k, block_size)
    _, vd, vs, v_deq = quantize_pages(v, block_size)
    scale = HEAD_DIM**-0.5
    q8, q_ref8 = quantize_q_fp8(q)
    plan = plan_pa_decode(ctx, 1)
    ref = ref_attention(q_ref8, k_deq, v_deq, bt, ctx, scale, block_size)
    out, us = run_perftest(
        lambda: run_mxfp4(q8, kd, ks, vd, vs, bt, ctx, scale, plan=plan)
    )
    flops = 4 * HEAD_DIM * q.shape[1] * rows * ctx_len  # QK + PV
    return {
        "gfx": get_gfx(),
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "TB/s": rows * ctx_len * TOKEN_BYTES / us / 1e6,
        "err": checkAllclose(
            ref,
            out.float(),
            rtol=RTOL,
            atol=ATOL,
            msg=f"rows={rows} ctx={ctx_len} group={group} bs={block_size}",
        ),
    }


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
        help="""KV page size in tokens.
    e.g.: -bs 128""",
    )
    args = parser.parse_args()

    aiter.logger.info("pa_decode_mxfp4: running correctness checks...")
    checks = run_checks()  # raises (non-zero exit) on the first mismatch
    aiter.logger.info(
        "pa_decode_mxfp4 correctness (%d cases passed):\n%s",
        len(checks),
        checks.to_markdown(index=False),
    )

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
