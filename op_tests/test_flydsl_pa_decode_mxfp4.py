# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 KV-cache FlyDSL paged-attention decode vs a torch reference.

Shapes are MiniMax-M3 on TP4: one KV head per rank, 16 query heads per KV head,
head_dim 128, one query token per row, a sparse selection of at most
16 blocks x 128 tokens per row (the caller compacts it into a block table). The
K/V pages are single-region: ``[block_size x 64 B e2m1][block_size x 4 B e8m0]``.

Run ``python op_tests/test_flydsl_pa_decode_mxfp4.py`` for the correctness
sweep + timing table, or ``pytest op_tests/test_flydsl_pa_decode_mxfp4.py``.
"""

import argparse
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.pa_decode_plan import plan_pa_decode
from aiter.ops.flydsl.pa_decode_mxfp4 import pa_decode_mxfp4
from aiter.ops.quant import per_1x32_f4_quant
from aiter.test_common import benchmark, run_perftest
from aiter.utility import fp4_utils

SUPPORTED_GFX = ["gfx950"]
HEAD_DIM = 128
Q_MAX_FP8 = 448.0


def _device():
    return torch.device("cuda")


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
    dev = _device()
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


def ref_attention(q_f32, k_deq, v_deq, bt, ctx, scale, block_size, p_bf16=True):
    """fp32 attention of every (row, q head) over its own paged tokens; P is
    rounded to bf16 before PV when ``p_bf16`` (what the kernel does)."""
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
            p = torch.softmax(s, dim=-1)
            l = p.sum(-1, keepdim=True)
            pn = p
            if p_bf16:
                # the kernel rounds exp2(s - m) (max 1) to bf16 and divides by the
                # unrounded sum
                e = torch.exp(s - s.max(-1, keepdim=True).values)
                l = e.sum(-1, keepdim=True)
                pn = e.to(torch.bfloat16).to(torch.float32)
            out[r, h * group : (h + 1) * group] = (pn @ vv) / l
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
    tile_depth=None,
    max_ctx=None,
):
    rows, nq, _ = q.shape
    out = torch.empty(rows, nq, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
    if plan is None:
        plan = plan_pa_decode(ctx, kd.shape[1])
    kwargs = {} if tile_depth is None else {"tile_depth": tile_depth}
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
        max_context_length=max_ctx,
        **kwargs,
    )
    return out


# --------------------------------------------------------------------------
# Correctness (pytest)
# --------------------------------------------------------------------------
CTX_SWEEP = [1, 15, 16, 17, 127, 128, 129, 255, 256, 257, 511, 2047, 2048]


def _skip_unsupported():
    if get_gfx() not in SUPPORTED_GFX:
        pytest.skip(f"pa_decode_mxfp4 supports {SUPPORTED_GFX}, got {get_gfx()}")


@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize("group", [16, 8])
@pytest.mark.parametrize("qmode", ["bf16", "fp8"])
def test_pa_decode_mxfp4_ctx_sweep(block_size, group, qmode):
    _skip_unsupported()
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
    _skip_unsupported()
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
    _skip_unsupported()
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
    _skip_unsupported()
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
    _skip_unsupported()
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
    _skip_unsupported()
    ctx, bt, k, v, q = make_case([1, 130, 777, 2048], 16, 128, seed=21)
    _, kd, ks, _ = quantize_pages(k, 128)
    _, vd, vs, _ = quantize_pages(v, 128)
    q_in, _ = quantize_q_fp8(q)
    scale = HEAD_DIM**-0.5
    a = run_mxfp4(q_in, kd, ks, vd, vs, bt, ctx, scale)
    b = run_mxfp4(q_in, kd, ks, vd, vs, bt, ctx, scale, tile_depth=4)
    torch.cuda.synchronize()
    assert torch.equal(a, b)


def test_pa_decode_mxfp4_rejects_oversized_task():
    _skip_unsupported()
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


@benchmark()
def bench_pa_decode_mxfp4(rows, ctx_len, group, block_size):
    ctx, bt, k, v, q = make_case([ctx_len] * rows, group, block_size, seed=1)
    _, kd, ks, k_deq = quantize_pages(k, block_size)
    _, vd, vs, v_deq = quantize_pages(v, block_size)
    scale = HEAD_DIM**-0.5
    q8, q_ref8 = quantize_q_fp8(q)
    ref = ref_attention(q_ref8, k_deq, v_deq, bt, ctx, scale, block_size)
    plan = plan_pa_decode(ctx, 1)
    candidates = {
        "mxfp4_qfp8": lambda: run_mxfp4(q8, kd, ks, vd, vs, bt, ctx, scale, plan=plan),
        "mxfp4_qbf16": lambda: run_mxfp4(q, kd, ks, vd, vs, bt, ctx, scale, plan=plan),
    }
    kv_bytes = {
        "mxfp4_qfp8": rows * ctx_len * (2 * (64 + 4)),
        "mxfp4_qbf16": rows * ctx_len * (2 * (64 + 4)),
    }
    refs = {
        "mxfp4_qfp8": ref,
        "mxfp4_qbf16": ref_attention(
            q.float(), k_deq, v_deq, bt, ctx, scale, block_size
        ),
    }
    if block_size in (16, 128) and group <= 16:
        from aiter.ops.flydsl.pa_decode import pa_decode

        key, val, kscale, vscale = fp8_inputs(k, v, block_size)
        out8 = torch.empty_like(q)
        plan8 = plan_pa_decode(ctx, 1)

        def run_fp8():
            pa_decode(
                out8,
                q,
                key,
                val,
                ctx,
                bt,
                scale,
                1,
                key_scale=kscale,
                value_scale=vscale,
                work_plan=plan8,
            )
            return out8

        candidates["fp8kv"] = run_fp8
        kv_bytes["fp8kv"] = rows * ctx_len * (2 * (128 + 4))
        # the fp8 cache holds the quantized K/V; compare to ITS own dequantization
        k8d = (
            key.to(torch.float32).permute(0, 1, 3, 2, 4).reshape(k.shape)
            * kscale[..., None]
        )
        v8d = (
            val.to(torch.float32).permute(0, 1, 2, 4, 3).reshape(v.shape)
            * vscale[..., None]
        )
        refs["fp8kv"] = ref_attention(q.float(), k8d, v8d, bt, ctx, scale, block_size)
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = rel_l2(out.float(), refs[name])
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = kv_bytes[name] / us / 1e6
        ret[f"{name} relL2"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("pa_decode_mxfp4 unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-r", "--rows", type=int, nargs="*", default=[1, 8, 32, 64, 128, 512, 4096]
    )
    parser.add_argument("-c", "--ctx", type=int, nargs="*", default=[2048])
    parser.add_argument("-g", "--group", type=int, nargs="*", default=[16])
    parser.add_argument("-bs", "--block-size", type=int, nargs="*", default=[16, 128])
    args = parser.parse_args()
    df = []
    for rows, ctx_len, group, bs in itertools.product(
        args.rows, args.ctx, args.group, args.block_size
    ):
        df.append(bench_pa_decode_mxfp4(rows, ctx_len, group, bs))
    df = pd.DataFrame(df)
    aiter.logger.info(
        "pa_decode_mxfp4 summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
