# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf for the fused MoE routing preamble.

``fused_moe_router_impl`` replaces the 4-kernel decode preamble in one launch:

    biased_grouped_topk -> moe_sorting -> fused_dynamic_mx_quant_moe_sort

The reference is that stock sequence, not a hand-written torch model, so a
mismatch means the fusion diverged from the path it is meant to replace.

topk ids and sorted rows are compared as multisets: any permutation of a given
expert's rows is a valid sort, and the fused kernel's ballot rank need not
match moe_sorting's atomic cursor. fp4 bytes and e8m0 scales must be exact.

This test can be run two ways:

1. pytest (correctness only):
   pytest op_tests/test_fused_moe_router.py -v

2. command line (correctness + perf summary table):
   python op_tests/test_fused_moe_router.py -m 1,64,128 -ek 320,8
"""

import argparse
import itertools
import os
import sys

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.fused_moe_router import (
    fused_moe_router_impl,
    get_fused_moe_router_workspace,
)
from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort
from aiter.ops.topk import biased_grouped_topk
from aiter.test_common import benchmark, run_perftest
from aiter.utility.dtypes import str2tuple

torch.set_default_device("cuda")

# The kernel is MXFP4 + gfx950 only; cols is fixed by the BlockSize*TD tiling
# the entry asserts (the cols == BlockSize * TD check in the entry).
COLS = 4096
GROUP_SIZE = 32
# Workspace sizing hint. Larger -m still works, it just allocates a
# second, bigger buffer.
WORKSPACE_MAX_TOKENS = 128
SUPPORTED = get_gfx() == "gfx950"

# Weights are renormalized by a sum reduced in a different order than the stock
# path, so they agree to fp32 rounding rather than bit-exactly.
W_TOL = 2e-6


def _skip_msg():
    return f"fused_moe_router requires gfx950, got {get_gfx()}"


def _make_mask(E, ep_rank, ep_size, vllm_shape=False):
    """Linear expert shard, matching vLLM's determine_expert_map.

    vllm_shape reproduces the real buffer, which is E+1 long with an
    always-masked sentinel in the trailing slot (expert_map_manager.py).
    """
    n = E + 1 if vllm_shape else E
    m = torch.zeros(n, dtype=torch.int32)
    per = E // ep_size
    m[ep_rank * per : (ep_rank + 1) * per] = 1
    return m


def _deswizzle(osc, nrows):
    """Invert mx_scale_shuffle_idx so scale rows can be compared by token."""
    scale_n = COLS // GROUP_SIZE
    scalen_pad = ((scale_n + 7) // 8) * 8
    x = torch.arange(nrows).view(-1, 1)
    y = torch.arange(scale_n).view(1, -1)
    idx = (
        (x // 32 * scalen_pad) * 32
        + (y // 8) * 256
        + (y % 4) * 64
        + (x % 16) * 4
        + (y % 8) // 4 * 2
        + (x % 32) // 16
    )
    return osc.reshape(-1).view(torch.uint8)[idx.view(-1)].view(nrows, -1)


def _inputs(M, E, bias_dtype, seed):
    torch.manual_seed(seed)
    g = torch.randn(M, E, dtype=dtypes.bf16)
    h = torch.randn(M, COLS, dtype=dtypes.bf16)
    # Draw the bias in bf16 and widen, so an fp32 bias holds bf16-representable
    # values. The stock wrapper only takes it in the gating dtype, so anything
    # needing the extra mantissa would make the two paths disagree on ties for
    # a reason that is not a kernel bug. What is under test here is that the
    # fp32 dispatch reads the wider layout correctly.
    b = torch.randn(E, dtype=dtypes.bf16).to(bias_dtype)
    return g, b, h


def _run_stock(g, b, h, M, E, topk, unit_size, need_renorm, rsf, mask):
    tw = torch.empty(M, topk, dtype=dtypes.fp32)
    ti = torch.empty(M, topk, dtype=torch.int32)
    # The stock wrapper coerces the bias to the gating dtype; the fused kernel
    # reads it as-is, so feed the reference the same values it would see.
    biased_grouped_topk(
        g,
        b.to(g.dtype),
        tw,
        ti,
        num_expert_group=1,
        topk_group=1,
        need_renorm=need_renorm,
        routed_scaling_factor=rsf,
    )
    sids, sw, seids, nv, moe_buf = moe_sorting(
        ti, tw, E, COLS, dtypes.bf16, block_size=unit_size, expert_mask=mask
    )
    o4, osc = fused_dynamic_mx_quant_moe_sort(
        h,
        sids,
        nv,
        token_num=M,
        topk=topk,
        block_size=unit_size,
        quant_dtype=dtypes.fp4x2,
        sorted_weights=sw,
    )
    return {
        "ti": ti,
        "tw": tw,
        "sids": sids,
        "sw": sw,
        "seids": seids,
        "nv": nv,
        "o4": o4,
        "osc": osc,
        "moe_buf": moe_buf,
    }


def _alloc(ref, M, topk):
    """Output buffers, poisoned so an unwritten slot cannot pass by luck."""
    return {
        "ti": torch.full((M, topk), -1, dtype=torch.int32),
        "tw": torch.zeros(M, topk, dtype=dtypes.fp32),
        "sids": torch.full_like(ref["sids"], -1),
        "sw": torch.zeros_like(ref["sw"]),
        "seids": torch.full_like(ref["seids"], -1),
        "nv": torch.zeros_like(ref["nv"]),
        "o4": torch.zeros(ref["o4"].shape, dtype=torch.uint8).view(ref["o4"].dtype),
        "osc": torch.zeros_like(ref["osc"]),
    }


def _call_fused(g, b, h, a, E, topk, unit_size, need_renorm, rsf, mask, moe_buf):
    fused_moe_router_impl(
        g,
        b,
        h,
        a["ti"],
        a["tw"],
        a["sids"],
        a["sw"],
        a["seids"],
        a["nv"],
        a["o4"],
        a["osc"],
        E,
        topk,
        unit_size,
        GROUP_SIZE,
        need_renorm,
        rsf,
        get_fused_moe_router_workspace(g.device, max(g.shape[0], WORKSPACE_MAX_TOKENS)),
        expert_mask=mask,
        moe_buf=moe_buf,
    )


def _compare(ref, got, M, topk, unit_size):
    """Return {metric: error count or magnitude}; all zero means pass."""
    errs = {}

    # topk: per-token set of (expert -> weight). Selection order within a token
    # is free, the weight attached to each expert is not.
    rset = {
        (r, int(e)): float(w)
        for r in range(M)
        for e, w in zip(ref["ti"][r].tolist(), ref["tw"][r].tolist())
    }
    gset = {
        (r, int(e)): float(w)
        for r in range(M)
        for e, w in zip(got["ti"][r].tolist(), got["tw"][r].tolist())
    }
    errs["topk_id_err"] = len(set(rset) ^ set(gset))
    errs["topk_w_err"] = (
        max((abs(rset[k] - gset[k]) for k in rset & gset.keys()), default=0.0)
        if rset
        else 0.0
    )

    nv_r, nv_g = int(ref["nv"][0]), int(got["nv"][0])
    errs["num_valid_err"] = int(nv_r != nv_g) + int(
        int(ref["nv"][1]) != int(got["nv"][1])
    )
    if nv_r != nv_g:
        # Everything downstream is indexed by num_valid; comparing past a
        # disagreement reports noise, so stop here.
        errs.update(sorted_id_err=-1, sorted_w_err=-1, fp4_err=-1, scale_err=-1)
        return errs

    if nv_r == 0:
        # Under EP a rank can own no expert any token routed to. Nothing is
        # written, and both sides agreeing on that is the whole check.
        errs.update(
            expert_id_err=0, sorted_id_err=0, sorted_w_err=0.0, fp4_err=0, scale_err=0
        )
        return errs

    nblk = nv_r // unit_size
    errs["expert_id_err"] = int(
        not torch.equal(ref["seids"][:nblk], got["seids"][:nblk])
    )

    rs, gs = ref["sids"][:nv_r].tolist(), got["sids"][:nv_r].tolist()
    rw, gw = ref["sw"][:nv_r].tolist(), got["sw"][:nv_r].tolist()
    bad_id, dw = 0, 0.0
    for blk in range(nblk):
        s = slice(blk * unit_size, (blk + 1) * unit_size)
        a, c = sorted(zip(rs[s], rw[s])), sorted(zip(gs[s], gw[s]))
        if [x[0] for x in a] != [x[0] for x in c]:
            bad_id += 1
        else:
            dw = max(dw, max((abs(x[1] - y[1]) for x, y in zip(a, c)), default=0.0))
    errs["sorted_id_err"] = bad_id
    errs["sorted_w_err"] = dw

    # Only rows the sort references are defined: under EP most tokens route to
    # no local expert and neither path writes their out_fp4 row.
    r4 = ref["o4"].view(torch.uint8).view(M, -1)
    g4 = got["o4"].view(torch.uint8).view(M, -1)
    live = torch.zeros(M, dtype=torch.bool)
    tok = ref["sids"][:nv_r] & 0xFFFFFF
    live[tok[tok < M].long()] = True
    errs["fp4_err"] = int((r4[live] != g4[live]).sum().item())

    # The swizzled scale buffer is only defined below num_valid, and which token
    # owns a row is permutation-dependent. Deswizzle both, then compare each
    # side's row against the scale its own sorted_ids says it should carry.
    sr = _deswizzle(ref["osc"], nv_r)
    sg = _deswizzle(got["osc"], nv_r)
    tok_r = (ref["sids"][:nv_r] & 0xFFFFFF).clamp(max=M)
    tok_g = (got["sids"][:nv_r] & 0xFFFFFF).clamp(max=M)
    want = torch.zeros(M + 1, COLS // GROUP_SIZE, dtype=torch.uint8)
    real = tok_r < M
    want[tok_r[real].long()] = sr[real]
    errs["scale_err"] = int((sg != want[tok_g.long()]).sum().item())
    return errs


def _run_case(
    M, E, topk, unit_size, bias_dtype, need_renorm, rsf, ep, use_moe_buf, seed=None
):
    """One config, correctness only. Returns the error dict."""
    mask = None
    if ep is not None:
        mask = _make_mask(E, ep[0], ep[1], vllm_shape=ep[2])
    g, b, h = _inputs(M, E, bias_dtype, M if seed is None else seed)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, need_renorm, rsf, mask)
    got = _alloc(ref, M, topk)
    moe_buf = None
    if use_moe_buf:
        # Poisoned: the kernel is supposed to zero it while routing runs.
        moe_buf = torch.full_like(ref["moe_buf"], 7.0)
    _call_fused(g, b, h, got, E, topk, unit_size, need_renorm, rsf, mask, moe_buf)
    torch.cuda.synchronize()
    errs = _compare(ref, got, M, topk, unit_size)
    errs["moe_buf_err"] = int((moe_buf != 0).sum().item()) if use_moe_buf else 0
    return errs, (g, b, h, ref, got, mask)


def _failed(errs):
    return any((v > W_TOL if k.endswith("_w_err") else v != 0) for k, v in errs.items())


def _pin_path(fused):
    """Pin the launch path. The host picks by token count otherwise, so a sweep
    over M would silently stop exercising the barrier above kSplitMinTokens.

    The env var is the crossover itself, so pinning means moving it out of
    reach on either side.
    """
    os.environ["AITER_MOE_ROUTING_SPLIT"] = "2147483647" if fused else "0"


def _unpin_path():
    os.environ.pop("AITER_MOE_ROUTING_SPLIT", None)


def _fill_pat(t, tag):
    """Fill with a pattern that is wrong in every interpretation.

    Zero and -1 are both plausible values for these buffers, so poisoning with
    them lets an unwritten slot pass by luck. 0x7f is a large positive int32, a
    huge float, and a nonzero e8m0 exponent byte.
    """
    t.view(torch.uint8).fill_(0x7F if tag % 2 == 0 else 0xA5)
    return t


def _alloc_poisoned(ref, M, topk, tag=0):
    """Output buffers filled with allocator-garbage rather than clean values."""
    a = _alloc(ref, M, topk)
    for k in ("ti", "tw", "sids", "sw", "seids", "nv", "o4", "osc"):
        _fill_pat(a[k], tag)
    # num_valid is read as a count; garbage is not a meaningful start state and
    # the kernel unconditionally overwrites it.
    a["nv"].zero_()
    return a


def _tail_errs(got, M, topk, unit_size, ctx):
    """Check the region past num_valid against the kernel's own contract.

    NOT against stock: stock leaves that region untouched, so it holds whatever
    the allocator handed it. The fused kernel deliberately does more, because
    the downstream stage-1 GEMM reads an id per block before it consults
    num_valid_ids. Contract: sorted_ids = pack_id(M, topk),
    sorted_weights = 0, sorted_expert_ids = 0.

    The boundary is the kernel's OWN num_valid, not stock's: under a gating tie
    the two elect different experts, which shifts per-expert padding and makes
    the two values legitimately differ. num_valid agreement is _compare's job.
    """
    fails = []
    nv = int(got["nv"][0])
    sentinel = (M & 0xFFFFFF) | ((topk & 0xFF) << 24)
    checks = (
        ("sorted_ids", got["sids"][nv:], sentinel),
        ("sorted_weights", got["sw"][nv:], 0.0),
        ("sorted_expert_ids", got["seids"][nv // unit_size :], 0),
    )
    for name, buf, want in checks:
        if buf.numel() == 0:
            continue
        n = int((buf != want).sum())
        if n:
            fails.append(
                f"{ctx}: {name} tail has {n}/{buf.numel()} slots "
                f"!= {want} (first: {buf[buf != want][0].item()})"
            )
    return fails


def _tie_free_bias(E):
    """A bias with E distinct values, so the top-k boundary is never a tie.

    With a zero or coarse bias the boundary score genuinely ties for some
    tokens, and the two paths then pick different experts of equal weight --
    both correct. Measured: 4 of 64 tokens tie at E=320. Breaking ties is what
    makes the rest of the input space assertable; ties get their own test.
    """
    return (torch.arange(E, dtype=torch.float32) * 1e-3).to(dtypes.bf16)


@benchmark()
def bench_fused_moe_router(
    M, E, topk, unit_size, bias_dtype, need_renorm, rsf, ep_label, use_moe_buf
):
    if not SUPPORTED:
        return {}
    ep = {
        "noEP": None,
        "EP_r0": (0, 4, False),
        "EP_r2": (2, 4, False),
        "EP_vllm": (2, 4, True),
    }[ep_label]
    errs, (g, b, h, _ref, got, mask) = _run_case(
        M, E, topk, unit_size, bias_dtype, need_renorm, rsf, ep, use_moe_buf
    )

    _, fused_us = run_perftest(
        _call_fused,
        g,
        b,
        h,
        got,
        E,
        topk,
        unit_size,
        need_renorm,
        rsf,
        mask,
        None,
        num_iters=10,
        num_warmup=2,
    )
    _, stock_us = run_perftest(
        _run_stock,
        g,
        b,
        h,
        M,
        E,
        topk,
        unit_size,
        need_renorm,
        rsf,
        mask,
        num_iters=10,
        num_warmup=2,
    )
    return {
        "fused_us": fused_us,
        "stock_us": stock_us,
        "uplift": stock_us / fused_us if fused_us else 0.0,
        **errs,
    }


@benchmark()
def bench_barrier_stress(M_list, E, topk, unit_size, iters):
    """Hammer the self-resetting grid barrier.

    The semaphore is a process-wide static that is zeroed once and then relies
    on the sense flip to rearm (see the workspace layout note in the entry). A
    flip bug does not show up in a single launch -- it poisons the *next* one. So: force the
    fused path at every M, queue many launches back to back with no host sync,
    and vary M between them so the barrier's participant count changes while
    the semaphore still carries the previous launch's state.
    """
    if not SUPPORTED:
        return {}
    prev = os.environ.get("AITER_MOE_ROUTING_SPLIT")
    # Crossover out of reach: force the barrier at every M.
    os.environ["AITER_MOE_ROUTING_SPLIT"] = "2147483647"
    try:
        # Precompute references, one per M, while nothing else is in flight.
        refs = {}
        for M in M_list:
            g, b, h = _inputs(M, E, dtypes.bf16, M)
            refs[M] = (
                g,
                b,
                h,
                _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None),
            )
        torch.cuda.synchronize()

        # Queue everything with no sync in between, each into its own buffers so
        # a late writer from a previous launch is visible rather than overwritten.
        outs = []
        for i in range(iters):
            M = M_list[i % len(M_list)]
            g, b, h, ref = refs[M]
            got = _alloc(ref, M, topk)
            _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None)
            outs.append((M, ref, got))
        torch.cuda.synchronize()

        bad = 0
        for M, ref, got in outs:
            if _failed(_compare(ref, got, M, topk, unit_size)):
                bad += 1
        return {"launches": len(outs), "stress_err": bad}
    finally:
        if prev is None:
            os.environ.pop("AITER_MOE_ROUTING_SPLIT", None)
        else:
            os.environ["AITER_MOE_ROUTING_SPLIT"] = prev


def _expect_raises(fn, what):
    try:
        fn()
    except (RuntimeError, AssertionError):
        return 0
    print(f"ERROR: {what} was accepted, expected a check to fire")
    return 1


def check_rejects_bad_shapes(E=320, topk=8, unit_size=16):
    """The entry must reject what it cannot serve instead of mis-routing."""
    if not SUPPORTED:
        return 0
    M = 16
    g, b, h = _inputs(M, E, dtypes.bf16, 0)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None)
    got = _alloc(ref, M, topk)
    bad = 0

    h_narrow = torch.randn(M, COLS // 2, dtype=dtypes.bf16)
    bad += _expect_raises(
        lambda: _call_fused(
            g, b, h_narrow, got, E, topk, unit_size, True, 1.0, None, None
        ),
        f"cols={COLS // 2}",
    )
    g_wide = torch.randn(M, 1024, dtype=dtypes.bf16)
    b_wide = torch.randn(1024, dtype=dtypes.bf16)
    bad += _expect_raises(
        lambda: _call_fused(
            g_wide, b_wide, h, got, 1024, topk, unit_size, True, 1.0, None, None
        ),
        "num_experts=1024",
    )
    bad += _expect_raises(
        lambda: _call_fused(g, b, h, got, E, topk, 24, True, 1.0, None, None),
        "unit_size=24 (not a power of two)",
    )
    bad += _expect_raises(
        lambda: _call_fused(
            g, b.to(torch.float16), h, got, E, topk, unit_size, True, 1.0, None, None
        ),
        "fp16 correction bias",
    )

    # Everything the entry reinterpret_casts: a wrong dtype or a strided view
    # would otherwise be read as bf16 at the wrong offsets.
    for name, gg, hh in (
        ("fp16 gating", g.to(torch.float16), h),
        ("fp16 hidden", g, h.to(torch.float16)),
        # [:, :COLS] of a 2*COLS-wide tensor: right shape, wrong row stride.
        ("non-contiguous gating", torch.randn(M, 2 * E, dtype=dtypes.bf16)[:, :E], h),
        (
            "non-contiguous hidden",
            g,
            torch.randn(M, 2 * COLS, dtype=dtypes.bf16)[:, :COLS],
        ),
    ):
        bad += _expect_raises(
            lambda gg=gg, hh=hh: _call_fused(
                gg, b, hh, got, E, topk, unit_size, True, 1.0, None, None
            ),
            name,
        )
    return bad


# ---------------------------------------------------------------------------
# pytest entry points (correctness only; the CLI below adds perf + sweeps)
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(not SUPPORTED, reason=_skip_msg())


def _assert_ok(errs, ctx):
    bad = {
        k: v for k, v in errs.items() if (v > W_TOL if k.endswith("_w_err") else v != 0)
    }
    assert not bad, f"{ctx}: {bad}"


# 103/104/105 straddle kSplitMinTokens, where the host switches from the fused
# barrier to a two-launch split. 23 is the phase-3 case where a truncated
# rows-per-block would leave a tail of rows owned by no block.
@pytest.mark.parametrize("M", [1, 8, 23, 32, 33, 64, 100, 103, 104, 105, 128])
@pytest.mark.parametrize("ep", [None, (0, 4, False), (2, 4, False), (2, 4, True)])
def test_tokens_and_ep(M, ep):
    errs, _ = _run_case(M, 320, 8, 16, dtypes.bf16, True, 1.0, ep, False)
    _assert_ok(errs, f"M={M} ep={ep}")


@pytest.mark.parametrize("unit_size", [16, 32, 64, 128])
@pytest.mark.parametrize("M", [1, 64, 128])
def test_unit_size(M, unit_size):
    errs, _ = _run_case(M, 320, 8, unit_size, dtypes.bf16, True, 1.0, None, False)
    _assert_ok(errs, f"M={M} unit_size={unit_size}")


@pytest.mark.parametrize("E,topk", [(256, 1), (256, 2), (320, 8), (512, 8)])
@pytest.mark.parametrize("M", [1, 64, 128])
def test_experts_and_topk(M, E, topk):
    errs, _ = _run_case(M, E, topk, 16, dtypes.bf16, True, 1.0, None, False)
    _assert_ok(errs, f"M={M} E={E} topk={topk}")


@pytest.mark.parametrize("need_renorm", [True, False])
@pytest.mark.parametrize("rsf", [1.0, 2.5])
def test_renorm_and_scaling(need_renorm, rsf):
    errs, _ = _run_case(64, 320, 8, 16, dtypes.bf16, need_renorm, rsf, None, False)
    _assert_ok(errs, f"need_renorm={need_renorm} rsf={rsf}")


# The entry dispatches on the real bias dtype rather than coercing it, because
# reading fp32 through the bf16 layout would silently mis-route.
@pytest.mark.parametrize("bias_dtype", [dtypes.bf16, dtypes.fp32])
def test_bias_dtype(bias_dtype):
    errs, _ = _run_case(64, 320, 8, 16, bias_dtype, True, 1.0, None, False)
    _assert_ok(errs, f"bias_dtype={bias_dtype}")


@pytest.mark.parametrize("M", [1, 64, 128])
def test_moe_buf_zero_fill(M):
    errs, _ = _run_case(M, 320, 8, 16, dtypes.bf16, True, 1.0, None, True)
    _assert_ok(errs, f"M={M} moe_buf")


def test_barrier_rearms():
    r = bench_barrier_stress([1, 33, 64, 100, 128], 320, 8, 16, 64)
    assert r.get("stress_err", 0) == 0, r


def test_bad_shapes_rejected():
    assert check_rejects_bad_shapes() == 0


# The override is a token count, so it must reject a value that only looks
# boolean: atoi("true") == 0 would silently mean "split at every M".
def test_split_override_rejects_non_numeric():
    M, E, topk, unit_size = 16, 320, 8, 16
    g, b, h = _inputs(M, E, dtypes.bf16, 0)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None)
    got = _alloc(ref, M, topk)
    prev = os.environ.get("AITER_MOE_ROUTING_SPLIT")
    os.environ["AITER_MOE_ROUTING_SPLIT"] = "true"
    try:
        with pytest.raises(RuntimeError, match="AITER_MOE_ROUTING_SPLIT"):
            _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None)
    finally:
        if prev is None:
            os.environ.pop("AITER_MOE_ROUTING_SPLIT", None)
        else:
            os.environ["AITER_MOE_ROUTING_SPLIT"] = prev


# An explicit out dtype overrides hidden_states.dtype, so a bf16 input can
# still resolve to fp16 and reach the bf16-only moe_buf clear.
def test_fp16_out_dtype_unsupported():
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe_router_supported

    M, E, topk = 16, 320, 8
    h = torch.randn(M, COLS, dtype=dtypes.bf16)
    w1 = torch.empty(E, 512, COLS // 2, dtype=dtypes.fp4x2)
    w2 = torch.empty(E, COLS, 128, dtype=dtypes.fp4x2)
    assert not fused_moe_router_supported(
        h,
        w1,
        w2,
        topk,
        quant_type=QuantType.per_1x32.value,
        activation=ActivationType.Silu.value,
        dtype=dtypes.fp16,
    )


# ---------------------------------------------------------------------------
# Path-pinned tests. The host picks fused vs split by token count, so these
# force both: they share device code and differ only in where the barrier is.
# ---------------------------------------------------------------------------


@pytest.fixture(params=[True, False], ids=["fused", "split"])
def path(request):
    _pin_path(request.param)
    yield "fused" if request.param else "split"
    _unpin_path()


# The op test's main sweep stops at num_valid. The downstream stage-1 GEMM
# launches over the whole buffer and reads an id per block before it checks
# num_valid_ids, so the tail is live: under vLLM's caching allocator that
# memory is a previous graph's activations, and a missed slot is a float bit
# pattern used as a token index. unit_size moves where the boundary lands
# relative to the grid stride; M moves how much tail there is.
@pytest.mark.parametrize("unit_size", [16, 32, 128])
@pytest.mark.parametrize("M", [1, 7, 23, 33, 64, 103, 128])
def test_tail_fill_exact(path, M, unit_size):
    _, (_g, _b, _h, _ref, got, _m) = _run_case(
        M, 320, 8, unit_size, dtypes.bf16, True, 1.0, None, False
    )
    fails = _tail_errs(got, M, 8, unit_size, f"{path} M={M} u={unit_size}")
    assert not fails, fails


# An all-zero mask makes num_valid 0 and every phase-3 slice empty; an all-ones
# mask makes the unit table maximal. Both are reachable under EP, and both are
# where an off-by-one in the local-id scan shows up rather than in the balanced
# shard test_tokens_and_ep covers.
def _extreme_masks(E):
    m = {
        "none_owned": torch.zeros(E, dtype=torch.int32),
        "all_owned": torch.ones(E, dtype=torch.int32),
    }
    for name, sl in (
        ("one_owned", slice(E // 2, E // 2 + 1)),
        ("alternating", slice(None, None, 2)),
        ("last_only", slice(E - 1, E)),
        ("first_only", slice(0, 1)),
    ):
        t = torch.zeros(E, dtype=torch.int32)
        t[sl] = 1
        m[name] = t
    # vLLM's real buffer is E+1 with an always-masked sentinel slot.
    sent = torch.zeros(E + 1, dtype=torch.int32)
    sent[: E // 4] = 1
    m["vllm_sentinel"] = sent
    return m


@pytest.mark.parametrize("mask_name", list(_extreme_masks(320)))
@pytest.mark.parametrize("M", [1, 33, 128])
def test_ep_mask_extremes(path, M, mask_name):
    E, topk, unit_size = 320, 8, 16
    mask = _extreme_masks(E)[mask_name]
    g, b, h = _inputs(M, E, dtypes.bf16, M)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, mask)
    got = _alloc_poisoned(ref, M, topk, M)
    _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, mask, None)
    torch.cuda.synchronize()
    ctx = f"{path} {mask_name} M={M}"
    _assert_ok(_compare(ref, got, M, topk, unit_size), ctx)
    assert not _tail_errs(got, M, topk, unit_size, ctx)


def _numerics_inputs(M, E):
    """Inputs on the quantiser's decision boundaries, not sampled from randn.

    All-zero hidden rows drive the e8m0 abs-max reduction to its zero case, and
    huge/tiny magnitudes drive it to the exponent clamps -- the places a scale
    byte is computed rather than copied.
    """
    torch.manual_seed(0)
    hn = torch.randn(M, COLS, dtype=dtypes.bf16)
    gr = torch.randn(M, E, dtype=dtypes.bf16)
    gz = torch.zeros(M, E, dtype=dtypes.bf16)
    bd = _tie_free_bias(E)

    # Every token routes to the same experts: one expert's count is M, the rest
    # are empty. Maximal skew for the phase-2 histogram.
    gs = torch.zeros(M, E, dtype=dtypes.bf16)
    gs[:, :8] = 10.0
    # One zero row among live rows: the zero-scale case must not contaminate
    # its neighbours in the swizzled scale tile.
    hm = torch.randn(M, COLS, dtype=dtypes.bf16)
    hm[::2] = 0
    # Per-group magnitude swing: adjacent groups land on far apart exponents,
    # which is what the shuffle has to keep straight.
    hg = torch.randn(M, COLS, dtype=dtypes.bf16) * (
        2.0
        ** torch.randint(-30, 30, (1, COLS // GROUP_SIZE))
        .repeat_interleave(GROUP_SIZE, 1)
        .to(dtypes.bf16)
    )
    # One group huge: the abs-max must not leak across the group boundary.
    hs = torch.randn(M, COLS, dtype=dtypes.bf16)
    hs[:, GROUP_SIZE : 2 * GROUP_SIZE] = 50000.0
    return {
        "gating_single_hot": (gs, bd, hn),
        # Gating carries no information, so selection is decided entirely by
        # the bias -- the value the fp32/bf16 dispatch reads differently.
        "bias_decides": (gz, bd, hn),
        "bias_fp32": (gz, bd.to(torch.float32), hn),
        "hidden_zero": (gr, bd, torch.zeros(M, COLS, dtype=dtypes.bf16)),
        "hidden_half_zero": (gr, bd, hm),
        "hidden_huge": (gr, bd, torch.full((M, COLS), 60000.0, dtype=dtypes.bf16)),
        "hidden_tiny": (gr, bd, torch.full((M, COLS), 1e-38, dtype=dtypes.bf16)),
        "hidden_group_swing": (gr, bd, hg),
        "hidden_one_group_huge": (gr, bd, hs),
    }


@pytest.mark.parametrize("case", list(_numerics_inputs(1, 320)))
def test_adversarial_numerics(path, case):
    M, E, topk, unit_size = 64, 320, 8, 16
    g, b, h = _numerics_inputs(M, E)[case]
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None)
    got = _alloc_poisoned(ref, M, topk, 3)
    _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None)
    torch.cuda.synchronize()
    _assert_ok(_compare(ref, got, M, topk, unit_size), f"{path} {case}")


def _tie_gatings(M, E, topk):
    # Exactly topk+1 experts tie for topk slots.
    gb = torch.full((M, E), -10.0, dtype=dtypes.bf16)
    gb[:, : topk + 1] = 3.0
    torch.manual_seed(0)
    return {
        # Every expert scores identically: maximal ambiguity.
        "all_equal": torch.zeros(M, E, dtype=dtypes.bf16),
        "all_ones": torch.ones(M, E, dtype=dtypes.bf16),
        "boundary_tie": gb,
        "randn": torch.randn(M, E, dtype=dtypes.bf16),
    }


# Deliberate top-k ties: which expert wins is a free choice and the two paths
# may differ. What is NOT free is the multiset of weights per token, the total
# routed row count, and the tail -- a tie must not drop or duplicate a row.
@pytest.mark.parametrize("case", ["all_equal", "all_ones", "boundary_tie", "randn"])
def test_gating_ties(path, case):
    M, E, topk, unit_size = 64, 320, 8, 16
    g = _tie_gatings(M, E, topk)[case]
    b = torch.zeros(E, dtype=dtypes.bf16)
    torch.manual_seed(0)
    h = torch.randn(M, COLS, dtype=dtypes.bf16)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None)
    got = _alloc_poisoned(ref, M, topk, 5)
    _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None)
    torch.cuda.synchronize()

    ctx = f"{path} {case}"
    nv = int(got["nv"][0])
    # num_valid itself is NOT invariant under ties: a different winning expert
    # changes that expert's row count and so how much its last unit is padded.
    # What is invariant is the number of REAL rows -- every (token, slot) pair
    # routes somewhere exactly once.
    sentinel = (M & 0xFFFFFF) | ((topk & 0xFF) << 24)
    live = got["sids"][:nv][got["sids"][:nv] != sentinel]
    assert live.numel() == M * topk, f"{ctx}: {live.numel()} real rows, nv={nv}"
    assert len(set(live.tolist())) == live.numel(), f"{ctx}: duplicate routed rows"

    rw = torch.sort(ref["tw"], dim=1).values
    gw = torch.sort(got["tw"], dim=1).values
    d = float((rw - gw).abs().max())
    assert d <= W_TOL, f"{ctx}: per-token weight multiset differs by {d:.2e}"

    ti = got["ti"]
    assert not int(((ti < 0) | (ti >= E)).sum()), f"{ctx}: topk_ids out of range"
    dup = [r for r in range(M) if len(set(ti[r].tolist())) != topk]
    assert not dup, f"{ctx}: tokens {dup[:4]} elected a duplicate expert"
    assert not _tail_errs(got, M, topk, unit_size, ctx)


# num_valid landing exactly on max_tokens leaves no tail to fill; valid_blocks
# landing exactly on gridDim leaves the last phase-3 slice empty. M spans every
# grid inflection: the GRID floor (16), num_cu, and the split crossover (104).
@pytest.mark.parametrize("unit_size,topk", [(16, 8), (128, 8), (16, 1), (64, 2)])
@pytest.mark.parametrize(
    "M", [1, 2, 15, 16, 17, 63, 65, 102, 104, 105, 127, 129, 160, 200, 255, 256]
)
def test_shape_boundaries(path, M, unit_size, topk):
    E = 320
    g, b, h = _inputs(M, E, dtypes.bf16, M)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None)
    got = _alloc_poisoned(ref, M, topk, M)
    try:
        _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        # The LDS budget check is a documented refusal, not a failure.
        if "shared memory" in str(exc) or "resident" in str(exc):
            pytest.skip(str(exc))
        raise
    ctx = f"{path} M={M} u={unit_size} topk={topk}"
    _assert_ok(_compare(ref, got, M, topk, unit_size), ctx)
    assert not _tail_errs(got, M, topk, unit_size, ctx)


# The two launch paths share device code and differ only in where the barrier
# is, so any divergence localises the bug to the barrier, not to the routing.
@pytest.mark.parametrize("M", [1, 16, 33, 64, 100, 103, 104, 128, 200])
def test_fused_split_agree(M):
    E, topk, unit_size = 320, 8, 16
    g, b, h = _inputs(M, E, dtypes.bf16, M)
    ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None)
    keys = ("ti", "tw", "sids", "sw", "seids", "nv", "o4", "osc")
    outs = {}
    try:
        for fused in (True, False):
            _pin_path(fused)
            got = _alloc_poisoned(ref, M, topk, M)
            _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None)
            torch.cuda.synchronize()
            outs[fused] = {k: got[k].clone() for k in keys}
    finally:
        _unpin_path()
    diff = {
        k: int((outs[True][k] != outs[False][k]).sum())
        for k in keys
        if not torch.equal(outs[True][k], outs[False][k])
    }
    assert not diff, f"M={M}: fused and split differ in {diff}"


# The workspace is allocated by host code that runs only at capture time, so
# its pointer is baked into the replayed launch. get_fused_moe_router_workspace
# retains the buffer for the life of the process; a caller that allocated its
# own and dropped it would leave the graph replaying through freed memory.
# Capture small then large on ONE stream (the order and the key vLLM uses) and
# require the small graph to keep replaying correctly.
def test_graph_replay_survives_workspace_growth():
    E, topk, unit_size = 320, 8, 16
    keys = ("ti", "tw", "sids", "sw", "seids", "nv", "o4", "osc")
    stream = torch.cuda.Stream()
    # One pool shared by every capture, as vLLM does: a private per-graph pool
    # is never handed out, so a freed block could not be reused and the test
    # would pass vacuously.
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for M in (16, 128):
        g, b, h = _inputs(M, E, dtypes.bf16, M)
        ref = _run_stock(g, b, h, M, E, topk, unit_size, True, 1.0, None)
        got = _alloc_poisoned(ref, M, topk, M)
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None)
        torch.cuda.current_stream().wait_stream(warm)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            _call_fused(g, b, h, got, E, topk, unit_size, True, 1.0, None, None)
        torch.cuda.synchronize()
        # Retain the inputs: a replay reads them, so letting them be freed
        # hands the replay recycled memory and makes it nondeterministic.
        graphs[M] = (graph, ref, got, (g, b, h))

    for M, (graph, ref, got, _keep) in graphs.items():
        for r in range(8):
            for k in keys:
                _fill_pat(got[k], r)
            got["nv"].zero_()
            graph.replay()
            torch.cuda.synchronize()
            ctx = f"M={M} replay {r} after growing to 128"
            _assert_ok(_compare(ref, got, M, topk, unit_size), ctx)
            assert not _tail_errs(got, M, topk, unit_size, ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-m",
        type=str2tuple,
        default=[1, 8, 23, 32, 33, 64, 100, 103, 104, 105, 128],
        help="""Token counts. 103/104/105 straddle the fused/split crossover.
    e.g.: -m 1,64,128""",
    )
    parser.add_argument(
        "-ek",
        "--expert_topk",
        type=str2tuple,
        nargs="*",
        default=[[256, 1], [256, 2], [320, 8], [512, 8]],
        help="""Expert count and topk pairs.
    e.g.: -ek 320,8""",
    )
    parser.add_argument(
        "-u",
        "--unit_size",
        type=str2tuple,
        default=[16, 32, 64, 128],
        help="""Sort block sizes (block_size_M), powers of two.
    e.g.: -u 16,32""",
    )
    parser.add_argument(
        "-s",
        "--stress_iters",
        type=int,
        default=128,
        help="""Back-to-back launches in the grid-barrier stress test.
    e.g.: -s 512""",
    )
    args = parser.parse_args()
    # str2tuple collapses a comma-less argument to a bare int.
    as_list = lambda v: list(v) if isinstance(v, (list, tuple)) else [v]
    args.m = as_list(args.m)
    args.unit_size = as_list(args.unit_size)
    args.expert_topk = [tuple(as_list(x)) for x in args.expert_topk]

    if not SUPPORTED:
        print(f"skip test_fused_moe_router: {_skip_msg()}")
        sys.exit(0)

    base_E, base_topk, base_u = 320, 8, 16
    sections = {}

    # Tokens x EP placement, at the base config.
    df = [
        bench_fused_moe_router(
            m, base_E, base_topk, base_u, dtypes.bf16, True, 1.0, ep, False
        )
        for ep, m in itertools.product(["noEP", "EP_r0", "EP_r2", "EP_vllm"], args.m)
    ]
    sections["tokens_ep"] = pd.DataFrame(df)

    # One axis at a time off the base config: the full product is far more
    # configs than the shard budget allows and adds little coverage.
    probe_m = [m for m in (1, 64, 128) if m in args.m] or args.m[:1]
    df = [
        bench_fused_moe_router(
            m, base_E, base_topk, u, dtypes.bf16, True, 1.0, "noEP", False
        )
        for u, m in itertools.product(args.unit_size, probe_m)
    ]
    sections["unit_size"] = pd.DataFrame(df)

    df = [
        bench_fused_moe_router(
            m, E, topk, base_u, dtypes.bf16, True, 1.0, "noEP", False
        )
        for (E, topk), m in itertools.product(args.expert_topk, probe_m)
    ]
    sections["experts_topk"] = pd.DataFrame(df)

    df = [
        bench_fused_moe_router(
            64, base_E, base_topk, base_u, bias_dtype, renorm, rsf, "noEP", moe_buf
        )
        for bias_dtype, renorm, rsf, moe_buf in [
            (dtypes.bf16, True, 1.0, True),
            (dtypes.fp32, True, 1.0, False),
            (dtypes.bf16, False, 1.0, False),
            (dtypes.bf16, True, 2.5, False),
        ]
    ]
    sections["options"] = pd.DataFrame(df)

    failed = []
    for name, frame in sections.items():
        aiter.logger.info(
            "fused_moe_router %s summary (markdown):\n%s",
            name,
            frame.to_markdown(index=False),
        )
        err_cols = [c for c in frame.columns if c.endswith("_err")]
        bad = frame[
            frame[err_cols]
            .apply(lambda c: c.abs() > (W_TOL if c.name.endswith("_w_err") else 0))
            .any(axis=1)
        ]
        if len(bad):
            print(f"\nERROR: {len(bad)} failing config(s) in {name}:")
            print(bad.to_string(index=False))
            failed.append(name)

    stress = bench_barrier_stress(
        [1, 33, 64, 100, 128], base_E, base_topk, base_u, args.stress_iters
    )
    aiter.logger.info(
        "fused_moe_router barrier stress: %d launches, %d bad",
        stress["launches"],
        stress["stress_err"],
    )
    if stress["stress_err"]:
        failed.append("barrier_stress")

    if check_rejects_bad_shapes():
        failed.append("bad_shapes")

    if failed:
        print(
            f"FAIL: section(s) with regressions: {', '.join(failed)}", file=sys.stderr
        )
        sys.exit(1)
    print("All fused_moe_router tests passed!")
