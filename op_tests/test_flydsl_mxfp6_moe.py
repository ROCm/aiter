# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MXFP6 grouped MoE (aiter.ops.flydsl.flydsl_mxfp6_moe), gfx950.

Correctness against torch references and timing, one markdown table per test function:
  test_mfma_contract  the scaled-MFMA operand contract per format pair (packing, K
                      order, E8M0 lanes) against float64
  test_quant_rows     the row quantizer against the torch recipe, bit-exact codes and
                      scales
  test_route          one-block and multi-block routers: placement invariants
  test_mxfp6_moe      the whole op against a torch reference with the same MX numerics
                      (element and per-row error), bitwise run-to-run determinism,
                      relative L2 against bf16
  test_moe_paths      test_mxfp6_moe on shapes whose kernels the default grid never
                      launches (padded rows and their copy kernel, A8W6 half tiles,
                      hidden 128, K tails of 2 and 3 steps); fails if the shape stops
                      reaching its path
  test_moe_modes      the whole op with the row-major and the tile-step-major h, under
                      torch.inference_mode() and torch.no_grad(), and with an x that
                      requires grad: each against the torch reference and bitwise equal
                      to the default call
  test_tuned_row      every row of the tuned config for this device: the op runs the
                      row's kernels, against the torch reference and bitwise equal to
                      the heuristic's kernels, with both timed in alternating rounds
  test_fused_moe      aiter.fused_moe on the prepared (tagged) weights: bitwise equal
                      to the op called directly, also into output= (a buffer, and
                      hidden_states itself), under torch.inference_mode(), with
                      the neutral spellings of the other arguments, and on a layer
                      prepared from weights that are MXFP6 already; both timed in
                      alternating rounds
No GPU needed (aiter's dry run):
  test_agpr_form      the 256x256 tiles keep their accumulators in AGPRs through
                      rocdl.experimental.sched_agpr and the LLVM option
                      amdgpu-mfma-vgpr-form. A FlyDSL that keeps both names while its
                      LLVM ignores them compiles a slow form with no error: fails
                      unless every 256x256 kernel has no v_accvgpr_* in its K loop and
                      no scratch
  test_h_layout       the contract of the stage wrappers and of the op for the 6-bit
                      h between the stages (nothing compiled): the layout each stage
                      picks, and the ValueErrors raised before a kernel is picked
  test_contract       the public entry: the calls it takes (casts of the routing
                      tensors, no tokens) and the ones it refuses before a kernel is
                      picked (dtype, shape, device, architecture, argument types);
                      the same through aiter.fused_moe: tagged weights reach the op,
                      every argument this path does not implement is refused, and so
                      are weights that are not one prepared layer (a copy without
                      the tag, tags that differ, scales missing or swapped); weight
                      preparation on the CPU and its refusals, from float weights
                      and from weights that are MXFP6 already (packed codes and
                      scales): both give the same layer
  test_tuned_dispatch the tuned config: a row is picked for exactly its device and
                      shape and the heuristic for every other call; a malformed row
                      and one whose kernels do not fit the layer are refused before a
                      kernel is picked
Only test_mxfp6_moe and test_tuned_row read the shipped tuned config
(aiter/configs/tuned_mxfp6_moe.csv); the others run on the heuristic's kernels.

    python3 op_tests/test_flydsl_mxfp6_moe.py
    python3 op_tests/test_flydsl_mxfp6_moe.py -s 8192,4608,64,4 -t 4096 32768 \
        --wfmt e2m3 --afmt e2m3 e4m3
"""

import argparse
import contextlib
import dataclasses
import functools
import inspect
import itertools
import math
import os
import re
import shutil
import statistics
import tempfile
from pathlib import Path
from unittest import mock

import flydsl.compiler as flyc
import flydsl.expr as fx
import pandas as pd
import torch
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.aot.flydsl.dry_run import dry_run
from aiter.fused_moe import fused_moe
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl import (
    Mxfp6MoeWeights,
    flydsl_mxfp6_moe,
    mxfp6_moe,
    prepare_mxfp6_moe_packed_weights,
    prepare_mxfp6_moe_weights,
)
from aiter.ops.flydsl import mxfp6_moe_kernels as mk
from aiter.ops.flydsl import mxfp6_moe_policy as policy
from aiter.ops.flydsl import mxfp6_moe_quant as q
from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    current_stream_arg,
    ptr_arg,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = list(mxfp6_moe.SUPPORTED_GFX)
DRY_RUN_CU = 256  # compute units of the dry run's device (MI355X)

# Whole op against the same-numerics torch reference. The two differ in how the
# GEMMs accumulate (order; fewer bits than fp32 inside the MFMA for E3M2 / E4M3
# operands, see test_mfma_contract), which moves a bf16 result by one ulp and now and
# then an h element or a block scale to the neighbouring MX code. Three bounds, with
# the values measured on gfx950 over 36 cases of 300 to 32768 tokens:
#   elements: rtol / atol with at most MOE_TOL_RATIO outside (largest measured: one
#       element of a 32768-token call). Blind to one wrong row of a few thousand.
#   largest relative L2 error of a row < MOE_ROW_TOL: 0.0195 measured; a row with one
#       expert wrong is off by 0.31-0.85, another token's row by 1.4. The reference
#       against itself with 1e-5 relative noise on its GEMM results has the same
#       tail (0.018), so the bound is not tighter.
#   median relative L2 error of the rows < MOE_ROW_MEDIAN_TOL: 0.0014 measured; an h
#       left unquantized gives 0.027-0.036, the other activation format 0.041, the
#       other weight format 0.12.
MOE_RTOL = MOE_ATOL = 2e-2
MOE_TOL_RATIO = 1e-3
MOE_ROW_TOL = 5e-2
MOE_ROW_MEDIAN_TOL = 5e-3

# run_perftest's default time is the GPU time of the kernels of a call. The op is
# several launches with host code between them, which is half of a small call's time
# (300 tokens, 1152 x 640: 29 us of kernels in a 65 us call), so every test here
# times the whole call with GPU events
TIMING = {"use_cuda_event": True}


# ---------------------------------------------------------------------------------
# torch references (not timed)
# ---------------------------------------------------------------------------------
def ref_quant_dequant_e2m3(x):
    """OCP MX E2M3: E8M0 scale 2^ceil(log2(amax / 7.5)) per 32, round-half-even onto
    the grid, clamp."""
    b = x.float().reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    amax = b.abs().amax(-1, keepdim=True)
    e = torch.ceil(torch.log2((amax / 7.5).clamp(min=2.0**-126)))
    scale = torch.exp2(torch.where(amax > 0, e, torch.full_like(e, -127.0)))
    u = (b / scale).abs().clamp(max=7.5)
    step = torch.exp2(torch.floor(torch.log2(u.clamp(min=1.0))) - 3)
    r = torch.round(u / step) * step
    return (torch.copysign(r.clamp(max=7.5), b) * scale).reshape(x.shape)


def _qd(x, fmt):
    return q.dequant_mx(*q.quant_mx(x, fmt), fmt)


def run_torch_mx(x, ids, w, w1, w2, wfmt, afmt):
    """MoE with the op's numerics: weights in wfmt, x and the SwiGLU output in afmt
    (dequantized per expert)."""
    inter = w2.shape[2]
    xq = _qd(x.float(), afmt).to(dtypes.bf16)
    out = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=dtypes.fp32)
    for e in range(w1.shape[0]):
        t, j = (ids == e).nonzero(as_tuple=True)
        if len(t):
            h = xq[t].float() @ _qd(w1[e].float(), wfmt).t()
            a = torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]
            y = _qd(a.to(dtypes.bf16).float(), afmt) @ _qd(w2[e].float(), wfmt).t()
            out.index_add_(0, t, y.to(dtypes.bf16).float() * w[t, j][:, None])
    return out.to(dtypes.bf16)


def run_torch_bf16(x, ids, w, w1, w2):
    """The unquantized bf16 MoE, for the accuracy column."""
    inter = w2.shape[2]
    out = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=dtypes.fp32)
    for e in range(w1.shape[0]):
        t, j = (ids == e).nonzero(as_tuple=True)
        if len(t):
            h = x[t] @ w1[e].t()
            y = (torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]) @ w2[e].t()
            out.index_add_(0, t, y.float() * w[t, j][:, None])
    return out.to(dtypes.bf16)


# ---------------------------------------------------------------------------------
# 1. scaled-MFMA operand contract: one wave, prepacked lane words, against float64
# ---------------------------------------------------------------------------------
_MFMA_FMT = {"e4m3": 0, "e2m3": 2, "e3m2": 3, "e2m1": 4}  # the instruction's codes
_MFMA_BITS = {"e4m3": 8, "e2m3": 6, "e3m2": 6, "e2m1": 4}


def _decode_table(fmt):
    """The value of every code of fmt, float64."""
    if fmt == "e4m3":
        codes = torch.arange(256, dtype=torch.uint8, device="cpu")
        return codes.view(torch.float8_e4m3fn).double()
    ebits, mbits = {"e2m3": (2, 3), "e3m2": (3, 2), "e2m1": (2, 1)}[fmt]
    bias = (1 << (ebits - 1)) - 1
    vals = []
    for c in range(1 << (1 + ebits + mbits)):
        s = -1.0 if c >> (ebits + mbits) else 1.0
        e, m = (c >> mbits) & ((1 << ebits) - 1), c & ((1 << mbits) - 1)
        frac = m / (1 << mbits)
        mag = frac * 2.0 ** (1 - bias) if e == 0 else (1 + frac) * 2.0 ** (e - bias)
        vals.append(s * mag)
    return torch.tensor(vals, dtype=torch.float64, device="cpu")


def _pack_lane_words(codes, bits):
    """[16, 128] codes -> [64 lanes, bits dwords]. FP6 / FP4 lane = K[32g, +32);
    FP8 lane = K[16g, +16) then K[64 + 16g, +16)."""
    out = torch.zeros(64, bits, dtype=torch.int64, device="cpu")
    for lane in range(64):
        row, g = lane % 16, lane // 16
        if bits == 8:
            ks = [*range(16 * g, 16 * g + 16), *range(64 + 16 * g, 80 + 16 * g)]
        else:
            ks = list(range(32 * g, 32 * g + 32))
        acc = 0
        for j, k in enumerate(ks):
            acc |= int(codes[row, k]) << (bits * j)
        for wd in range(bits):
            out[lane, wd] = (acc >> (32 * wd)) & 0xFFFFFFFF
    out[out >= 2**31] -= 2**32
    return out.to(torch.int32)


@functools.cache
def _get_compiled_mfma_probe(fmt_a, fmt_b):
    """One wave, one V_MFMA_SCALE_F32_16X16X128_F8F6F4 on prepacked lane words."""
    nwa, nwb = _MFMA_BITS[fmt_a], _MFMA_BITS[fmt_b]
    name = f"mxfp6_moe_test_mfma_{fmt_a}_{fmt_b}"

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def probe(
        A: fx.Pointer, B: fx.Pointer, SA: fx.Pointer, SB: fx.Pointer, C: fx.Pointer
    ):
        tid = fx.thread_idx.x
        i32pt = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=4
        )
        f32pt = fx.PointerType.get(
            T.f32, address_space=fx.AddressSpace.Global, alignment=4
        )
        a, b, sa, sb = (
            fx.inttoptr(i32pt, fx.Int64(fx.ptrtoint(p))) for p in (A, B, SA, SB)
        )
        c = fx.inttoptr(f32pt, fx.Int64(fx.ptrtoint(C)))
        av = fx.Vector.from_elements(
            [a[tid * fx.Int32(nwa) + fx.Int32(j)] for j in range_constexpr(nwa)],
            fx.Int32,
        )
        bv = fx.Vector.from_elements(
            [b[tid * fx.Int32(nwb) + fx.Int32(j)] for j in range_constexpr(nwb)],
            fx.Int32,
        )
        # Raw rocdl.mfma_scale_f32_16x16x128_f8f6f4: the probe pins the contract of
        # this instruction, on operand words packed by the test itself
        r = fx.Vector(
            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.vec(4, T.f32),
                [
                    av,
                    bv,
                    fx.Vector.filled(4, 0.0, fx.Float32),
                    _MFMA_FMT[fmt_a],
                    _MFMA_FMT[fmt_b],
                    0,
                    sa[tid],
                    0,
                    sb[tid],
                ],
            )
        )
        for i in range_constexpr(4):
            c[tid * fx.Int32(4) + fx.Int32(i)] = r[i]

    @flyc.jit
    def launch(
        A: fx.Pointer,
        B: fx.Pointer,
        SA: fx.Pointer,
        SB: fx.Pointer,
        C: fx.Pointer,
        stream: fx.Stream,
    ):
        probe(A, B, SA, SB, C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


@benchmark()
def test_mfma_contract(fmt_a, fmt_b):
    # one instruction: no timing
    gen = torch.Generator(device="cpu").manual_seed(0)
    codes = []
    for f in (fmt_a, fmt_b):
        ok = torch.nonzero(torch.isfinite(_decode_table(f))).flatten()
        pick = torch.randint(len(ok), (16, 128), generator=gen, device="cpu")
        codes.append(ok[pick])
    # a distinct E8M0 per (row, K block): a scale read from the wrong lane shows
    sa = torch.randint(120, 135, (16, 4), generator=gen, device="cpu")
    sb = torch.randint(120, 135, (16, 4), generator=gen, device="cpu")
    scale_a = torch.repeat_interleave(2.0 ** (sa.double() - 127), 32, 1)
    scale_b = torch.repeat_interleave(2.0 ** (sb.double() - 127), 32, 1)
    va = _decode_table(fmt_a)[codes[0]] * scale_a
    vb = _decode_table(fmt_b)[codes[1]] * scale_b
    ref = va @ vb.T
    lane = torch.arange(64, device="cpu")
    # NaN-filled: an accumulator lane the kernel does not write fails the comparison
    out = torch.full((64, 4), float("nan"), dtype=dtypes.fp32, device="cuda")
    args = [
        _pack_lane_words(codes[0], _MFMA_BITS[fmt_a]).cuda(),
        _pack_lane_words(codes[1], _MFMA_BITS[fmt_b]).cuda(),
        sa[lane % 16, lane // 16].int().cuda(),
        sb[lane % 16, lane // 16].int().cuda(),
        out,
    ]
    _run_compiled(
        _get_compiled_mfma_probe(fmt_a, fmt_b),
        *[ptr_arg(a) for a in args],
        current_stream_arg(out.device),
    )
    o = out.cpu().double()
    got = torch.empty(16, 16, dtype=torch.float64, device="cpu")
    for ln in range(64):
        for i in range(4):
            got[4 * (ln // 16) + i, ln % 16] = o[ln, i]
    # E2M3 / E2M1 products are exact; E3M2 / E4M3 see the MFMA's internal
    # accumulation below fp32
    err = checkAllclose(
        ref.float(),
        got.float(),
        rtol=1e-4,
        atol=1e-4 * ref.abs().max().item(),
        tol_err_ratio=0.0,
        msg=f"mfma {fmt_a} x {fmt_b}: ",
    )
    assert err == 0, f"{fmt_a} x {fmt_b}: {err:.2%} outside tolerance"
    rel = ((got - ref).norm() / ref.norm()).item()
    return {"gfx": get_gfx_runtime(), "rel_l2": rel, "err": err}


# ---------------------------------------------------------------------------------
# 2. MX row quantization kernel against the torch recipe (bit-exact)
# ---------------------------------------------------------------------------------
@benchmark()
def test_quant_rows(rows, k, fmt):
    g = torch.Generator(device="cuda").manual_seed(0)
    # values spanning 8 decades, an all-zero block and a tiny-scale block
    decades = torch.logspace(-4, 4, k, device="cuda")
    decades = decades[torch.randperm(k, generator=g, device="cuda")]
    x = torch.randn(rows, k, generator=g, device="cuda") * decades
    x = x.to(dtypes.bf16)
    x[min(5, rows - 1), :32] = 0
    x[min(6, rows - 1), 64:96] = 1e-30
    if fmt == "e2m3":
        ref = ref_quant_dequant_e2m3(x).to(dtypes.bf16).float()
        host = _qd(x, fmt).to(dtypes.bf16).float()
        err0 = checkAllclose(
            ref, host, rtol=0, atol=0, tol_err_ratio=0.0, msg="quant_mx vs recipe: "
        )
        assert err0 == 0, "the host quantizer differs from the E2M3 recipe"
    c, s = q.quant_mx(x, fmt)
    ref_codes, ref_scales = q.pack_rows(c, fmt), q.pack_a_scales(s)
    (codes, scales), us = run_perftest(mk.flydsl_mxfp6_moe_quant_rows, x, fmt, **TIMING)
    ks = k // 128
    err = checkAllclose(
        ref_codes.int(),
        codes.int(),
        rtol=0,
        atol=0,
        tol_err_ratio=0.0,
        msg=f"codes {fmt}: ",
    )
    # only the scale bytes of the row's K steps are written
    err_s = checkAllclose(
        q.unpack_a_scales(ref_scales, ks).int(),
        q.unpack_a_scales(scales, ks).int(),
        rtol=0,
        atol=0,
        tol_err_ratio=0.0,
        msg=f"scales {fmt}: ",
    )
    assert err == 0 and err_s == 0, f"{fmt}: codes {err:.2%}, scales {err_s:.2%}"
    # a quantizer moves bytes: bf16 in, codes and scale dwords out; no FLOP count
    nbytes = x.nbytes + codes.nbytes + scales.nbytes
    return {
        "gfx": get_gfx_runtime(),
        "flydsl us": us,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": max(err, err_s),
    }


# ---------------------------------------------------------------------------------
# 3. routers
# ---------------------------------------------------------------------------------
# entries up to which flydsl_mxfp6_moe_route runs the one-block router: each router
# is run on every case, also where the op would pick the other one
ROUTERS = {"one_block": 2**62, "multi_block": 0}


@benchmark()
def test_route(router, tokens, topk, experts, route_m):
    g = torch.Generator(device="cuda").manual_seed(0)
    # skewed expert load
    p = torch.softmax(torch.randn(experts, generator=g, device="cuda") * 2, 0)
    ids = torch.multinomial(
        p.expand(tokens, experts), topk, replacement=False, generator=g
    ).int()
    with mock.patch.object(mk, "ROUTE_ONE_BLOCK_MAX", ROUTERS[router]):
        (ridx, pos, texp, m), us = run_perftest(
            mk.flydsl_mxfp6_moe_route, ids, experts, route_m, **TIMING
        )
    nt = texp.numel() - 1
    count = torch.bincount(ids.flatten().long(), minlength=experts)
    ntile = (count + route_m - 1) // route_m
    tstart = torch.cumsum(ntile, 0) - ntile
    exp_texp = torch.full((nt,), -1, dtype=torch.int32, device="cuda")
    valid = torch.zeros(m, dtype=torch.bool, device="cuda")
    for e in range(experts):
        first, last = int(tstart[e]), int(tstart[e] + ntile[e])
        exp_texp[first:last] = e
        valid[first * route_m : last * route_m] = True
    used = torch.zeros(m, dtype=torch.bool, device="cuda")
    used[pos.flatten().long()] = True
    token = torch.arange(tokens, device="cuda")[:, None].expand(tokens, topk).int()
    checks = {
        "used tile count": int(texp[nt]) == int(ntile.sum()),
        "tile -> expert": torch.equal(texp[:nt], exp_texp),
        "row -> token": torch.equal(ridx[pos.long()], token),
        "slot in its expert's tiles": torch.equal(texp[(pos // route_m).long()], ids),
        "slots unique": pos.flatten().unique().numel() == tokens * topk,
        "pad slots -> row 0": bool((ridx[valid & ~used] == 0).all()),
    }
    bad = [k for k, v in checks.items() if not v]
    assert not bad, f"{router}: {bad}"
    # a router moves bytes: ids in, gather index, positions and tile map out
    nbytes = ids.nbytes + ridx.nbytes + pos.nbytes + texp.nbytes
    return {
        "gfx": get_gfx_runtime(),
        "flydsl us": us,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": float(len(bad)),
    }


# ---------------------------------------------------------------------------------
# 4. whole op
# ---------------------------------------------------------------------------------
def _layer_inputs(g, hidden, inter, experts):
    w1 = torch.randn(experts, 2 * inter, hidden, generator=g, device="cuda")
    w2 = torch.randn(experts, hidden, inter, generator=g, device="cuda")
    w1 = (w1 / math.sqrt(hidden)).to(dtypes.bf16)
    w2 = (w2 / math.sqrt(inter)).to(dtypes.bf16)
    return w1, w2


def _call_inputs(g, tokens, hidden, experts, topk):
    x = torch.randn(tokens, hidden, generator=g, device="cuda").to(dtypes.bf16)
    # random distinct experts per token (uneven per-expert load) and distinct routing
    # weights
    ids = torch.rand(tokens, experts, generator=g, device="cuda").argsort(-1)
    ids = ids[:, :topk].int().contiguous()
    w = torch.rand(tokens, topk, generator=g, device="cuda") + 0.5
    return x, ids, (w / w.sum(-1, keepdim=True)).float()


def _moe_inputs(tokens, hidden, inter, experts, topk):
    # the previous case's layer is released by now: give its memory back before this
    # case's weights are allocated
    torch.cuda.empty_cache()
    g = torch.Generator(device="cuda").manual_seed(0)
    layer = _layer_inputs(g, hidden, inter, experts)
    return (*layer, *_call_inputs(g, tokens, hidden, experts, topk))


def _moe_work(tokens, hidden, inter, experts, topk, afmt):
    """FLOPs and bytes of one call: stage 1 is a [T * topk, 2I, H] GEMM and stage 2 a
    [T * topk, H, I] one; the bytes are the 6-bit weights, x and y, and per routed row
    the quantized h (written and read) and the bf16 stage-2 row (written and read)."""
    flops = 6 * tokens * topk * hidden * inter
    abits = q.FORMATS[afmt][3]
    nbytes = (
        experts * 3 * hidden * inter * 6 // 8
        + tokens * hidden * 2 * 2
        + tokens * topk * (inter * abits // 8 + hidden * 2) * 2
    )
    return flops, nbytes


def _check_moe(out, ref, msg):
    """out against the same-numerics reference -> (fraction of elements outside
    rtol / atol, largest and median relative L2 error of a row), all asserted."""
    o, r = out.to(dtypes.fp32), ref.to(dtypes.fp32)
    err = checkAllclose(
        r, o, rtol=MOE_RTOL, atol=MOE_ATOL, tol_err_ratio=MOE_TOL_RATIO, msg=msg
    )
    assert err < MOE_TOL_RATIO, f"{msg}{err:.3%} of elements outside rtol / atol"
    rows = (o - r).norm(dim=1) / r.norm(dim=1).clamp(min=1e-30)
    row, med = rows.max().item(), rows.median().item()
    # NaN (an unwritten or corrupted row) fails these comparisons too
    assert row < MOE_ROW_TOL, f"{msg}a row is off by {row:.3%} (relative L2)"
    assert med < MOE_ROW_MEDIAN_TOL, f"{msg}the median row is off by {med:.3%}"
    return err, row, med


@benchmark()
def test_mxfp6_moe(tokens, hidden, inter, experts, topk, wfmt, afmt):
    w1, w2, x, ids, w = _moe_inputs(tokens, hidden, inter, experts, topk)
    st = prepare_mxfp6_moe_weights(w1, w2, a_fmt=afmt, w_fmt=wfmt)

    # a closure: run_perftest copies its tensor arguments to rotate them, and a
    # layer's weights are fixed between calls
    def run():
        return flydsl_mxfp6_moe(x, st, w, ids)

    out, us = run_perftest(run, num_iters=20, num_warmup=3, **TIMING)
    det = all(torch.equal(out, run()) for _ in range(3))
    assert det, "not bitwise deterministic run to run"
    ref = run_torch_mx(x, ids, w, w1, w2, wfmt, afmt)
    err, row, med = _check_moe(out, ref, f"mxfp6_moe {wfmt}/{afmt}: ")
    bf = run_torch_bf16(x, ids, w, w1, w2).float()
    rel = ((out.float() - bf).norm() / bf.norm()).item()
    flops, nbytes = _moe_work(tokens, hidden, inter, experts, topk, afmt)
    return {
        "gfx": get_gfx_runtime(),
        "flydsl us": us,
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": err,
        "max row rel_l2": row,
        "median row rel_l2": med,
        "deterministic": det,
        "rel_l2_vs_bf16": rel,
    }


# ---------------------------------------------------------------------------------
# 5. the GEMM kernel's own choices: the AGPR form (compiled, no GPU), the h layout
#    contract (host only, no GPU), rarely selected paths and call modes (GPU)
# ---------------------------------------------------------------------------------
_GEMM_ARGS = inspect.signature(mk._get_compiled_gemm)
_LAUNCHERS = (
    "_get_compiled_quant_rows",
    "_get_compiled_combine",
    "_get_compiled_route",
    "_get_compiled_route_mb",
    "_get_compiled_pad_rows",
    "_get_compiled_gemm",
)


def _clear_launchers():
    """Drop the cached launchers: one built in a dry run must not serve a run on the
    GPU."""
    for name in _LAUNCHERS:
        getattr(mk, name).cache_clear()


def _spy_gemm(calls, build=True):
    """Record the GEMM kernel each stage launch asks for (its compile arguments by
    name) in calls; build=False: nothing is compiled."""
    real = mk._get_compiled_gemm

    def spy(*args, **kwargs):
        calls.append(dict(_GEMM_ARGS.bind(*args, **kwargs).arguments))
        return real(*args, **kwargs) if build else None

    return mock.patch.object(mk, "_get_compiled_gemm", spy)


def _asked(calls):
    """(stage kernels, h layouts) of recorded stage launches, as the configs name
    them."""
    kernels = [
        policy.Mxfp6StageConfig(
            tile_m=32 * c["FM"],
            tile_n=32 * c["FN"],
            ring=c["NSTAGE"],
            group_m=c["GM"],
            nt_weights=bool(c["NT_WEIGHTS"]),
            pad_rows=bool(c["PAD_A"]),
            warm=c["WARM"],
            half_tiles=bool(c["HALF_TILES"]),
        )
        for c in calls
    ]
    return kernels, [c["H_TILE_STEP"] for c in calls]


def _names(config, afmt):
    """(stage kernels, h layouts) the stage launches of the op ask for with config."""
    step = int(q.FORMATS[afmt][3] == 6 and config.h_layout == "tile_step")
    return [config.stage1, config.stage2], [step, step]


@contextlib.contextmanager
def _tuned_table(rows=()):
    """The op with a tuned config of these rows only (column -> cell) in place of the
    shipped one; no rows: every call takes the heuristic's kernels."""
    columns = [*policy.TUNED_KEYS, *policy.TUNED_RESULTS]
    with tempfile.TemporaryDirectory(prefix="mxfp6_moe_tuned.") as tmp:
        path = os.path.join(tmp, "tuned_mxfp6_moe.csv")
        pd.DataFrame(list(rows), columns=columns).to_csv(path, index=False)
        try:
            with mock.patch.dict(os.environ, {"AITER_CONFIG_MXFP6_MOE": path}):
                policy.clear_tuned_configs()
                yield
        finally:
            policy.clear_tuned_configs()


@contextlib.contextmanager
def _host_only(calls):
    """The launch wrappers with nothing compiled and nothing launched; the GEMM kernel
    each stage launch asks for is recorded in calls. No tuned rows: the op picks the
    heuristic's kernels."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(_tuned_table())
        stack.enter_context(_spy_gemm(calls, build=False))
        for name in _LAUNCHERS[:-1]:
            stack.enter_context(mock.patch.object(mk, name, lambda *a: None))
        stack.enter_context(mock.patch.object(mk, "_run_compiled", lambda *a: None))
        yield


def _empty(*shape, dtype=torch.uint8):
    return torch.empty(*shape, dtype=dtype, device="cuda")


def _empty_layer(hidden, inter, experts, afmt, wfmt="e2m3"):
    """A prepared layer on empty buffers of prepare_mxfp6_moe_weights' shapes
    (FakeTensors inside aiter's dry run)."""
    bn1, bn2 = policy.stage_bn(2 * inter, afmt), policy.stage_bn(hidden, afmt)
    b1 = _empty(experts, 2 * inter // bn1, hidden // 128, bn1 // 16, 1536)
    s1 = _empty(experts, 2 * inter // 16, q.kq(hidden), 64, dtype=torch.int32)
    b2 = _empty(experts, hidden // bn2, inter // 128, bn2 // 16, 1536)
    s2 = _empty(experts, hidden // 16, q.kq(inter), 64, dtype=torch.int32)
    b1.mxfp6_fmt = b2.mxfp6_fmt = (afmt, wfmt)
    return Mxfp6MoeWeights(
        experts=experts,
        model_dim=hidden,
        inter_dim=inter,
        a_fmt=afmt,
        w_fmt=wfmt,
        w1_frags=b1,
        w1_scales=s1,
        w2_frags=b2,
        w2_scales=s2,
        slab_n1=bn1,
        slab_n2=bn2,
        w1_bytes=b1.nbytes + s1.nbytes,
        w2_bytes=b2.nbytes + s2.nbytes,
    )


def _config(st, tokens, topk, **kw):
    """The heuristic's kernels of a call on the prepared layer st, as the op picks
    them; kw: default_config's pad_rows / h_layout."""
    return policy.default_config(
        tokens=tokens,
        topk=topk,
        experts=st.experts,
        model_dim=st.model_dim,
        inter_dim=st.inter_dim,
        a_fmt=st.a_fmt,
        slab_n1=st.slab_n1,
        slab_n2=st.slab_n2,
        w1_bytes=st.w1_bytes,
        w2_bytes=st.w2_bytes,
        cu_num=get_cu_num(),
        **kw,
    )


def _stage_calls(st, tokens, topk, config):
    """Keyword arguments of the stage 1 and the stage 2 launch as the op makes them,
    on empty buffers of the op's shapes. h_layout is left out."""
    hidden, inter, afmt = st.model_dim, st.inter_dim, st.a_fmt
    bits = q.FORMATS[afmt][3]
    nt = policy.route_tiles(tokens * topk, st.experts, config.route_m)
    rows = nt * config.route_m
    h_rows = _empty(rows, inter * bits // 8)
    h_scales = _empty(rows, q.kq(inter), 4, dtype=torch.int32)
    shared = {
        "ridx": _empty(rows, dtype=torch.int32),
        "texp": _empty(nt + 1, dtype=torch.int32),
        "h_rows": h_rows,
        "h_scales": h_scales,
        "model_dim": hidden,
        "inter_dim": inter,
        "a_fmt": afmt,
        "w_fmt": st.w_fmt,
    }
    stage1 = {
        "x_rows": _empty(tokens, hidden * bits // 8),
        "x_scales": _empty(tokens, q.kq(hidden), 4, dtype=torch.int32),
        "w1_frags": st.w1_frags,
        "w1_scales": st.w1_scales,
        "config": config.stage1,
        **shared,
    }
    stage2 = {
        "w2_frags": st.w2_frags,
        "w2_scales": st.w2_scales,
        "out": _empty(rows, hidden, dtype=dtypes.bf16),
        "config": config.stage2,
        **shared,
    }
    return stage1, stage2


def _k_loop(lines):
    """The K loop of a kernel's final ISA: the instructions of its longest backward
    branch."""
    labels = {}
    for i, line in enumerate(lines):
        if m := re.match(r"^(\.LBB\w+):", line):
            labels[m.group(1)] = i
    best = None
    for j, line in enumerate(lines):
        m = re.match(r"\s+s_c?branch\w*\s+(\.LBB\w+)", line)
        if not m or labels.get(m.group(1), j) >= j:
            continue
        start = labels[m.group(1)]
        if best is None or j - start > best[1] - best[0]:
            best = (start, j)
    return lines[best[0] : best[1] + 1] if best else []


@benchmark()
def test_agpr_form(hidden, inter, experts, topk, tokens, afmt):
    # compile only: no torch reference, no timing. FlyDSL reads the dump settings at
    # each compile, and a cached kernel is not dumped
    dump = tempfile.mkdtemp(prefix="mxfp6_moe_agpr.")
    env = {
        "FLYDSL_DEBUG_DUMP_ASM": "1",
        "FLYDSL_RUNTIME_ENABLE_CACHE": "0",
        "FLYDSL_DUMP_DIR": dump,
    }
    _clear_launchers()
    try:
        with mock.patch.dict(os.environ, env), dry_run(SUPPORTED_GFX[0], DRY_RUN_CU):
            st = _empty_layer(hidden, inter, experts, afmt)
            config = _config(st, tokens, topk)
            stage1, stage2 = _stage_calls(st, tokens, topk, config)
            mk.flydsl_mxfp6_moe_stage1(**stage1, h_layout=config.h_layout)
            mk.flydsl_mxfp6_moe_stage2(**stage2, h_layout=config.h_layout)
        # the 256x256 tiles: 8 x 8 fragments of 16x16 per wave pair
        kernels = sorted(Path(dump).glob("mxfp6_moe_gemm_*_f8x8_*/*_final_isa.s"))
        copies, scratch, mfma = 0, 0, 0
        for f in kernels:
            text = f.read_text()
            body = [line.split(";")[0] for line in _k_loop(text.splitlines())]
            mfma += sum(1 for line in body if re.match(r"\s+v_mfma", line))
            copies += sum(1 for line in body if re.match(r"\s+v_accvgpr_", line))
            size = re.search(r"\.private_segment_fixed_size:\s+(\d+)", text)
            scratch = max(scratch, int(size.group(1)))
    finally:
        _clear_launchers()
        shutil.rmtree(dump, ignore_errors=True)
    ok = bool(kernels) and copies == 0 and scratch == 0 and mfma > 0
    assert ok, (
        f"256x256 kernels compiled to the slow form: {copies} v_accvgpr_* in their K "
        f"loops, {scratch} B scratch ({len(kernels)} kernels, {mfma} K-loop MFMAs): "
        "the AGPR pin or the amdgpu-mfma-vgpr-form option has no effect in this "
        "FlyDSL / LLVM"
    )
    return {
        "gfx": SUPPORTED_GFX[0],
        "256x256 kernels": len(kernels),
        "K-loop MFMA": mfma,
        "K-loop v_accvgpr_*": copies,
        "max scratch B": scratch,
        "fast form": ok,
    }


def _raises(error, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except error:
        return True
    return False


_refused = functools.partial(_raises, ValueError)


@benchmark()
def test_h_layout(hidden, inter, experts, topk, tokens, afmt):
    # host code only: the GEMM kernel each stage asks for is recorded and nothing is
    # compiled or launched, so no reference and no timing
    six = int(q.FORMATS[afmt][3] == 6)
    calls, got = [], {}

    def picked(fn):
        """The h layout of the kernels fn's stage launches ask for (1: tile-step)."""
        del calls[:]
        fn()
        return tuple(c["H_TILE_STEP"] for c in calls)

    with dry_run(SUPPORTED_GFX[0], DRY_RUN_CU), _host_only(calls):
        st = _empty_layer(hidden, inter, experts, afmt)
        config = _config(st, tokens, topk)
        rows_config = _config(st, tokens, topk, h_layout="rows")

        def stages(**kw):
            # fresh buffers per call, as the op allocates them
            stage1, stage2 = _stage_calls(st, tokens, topk, config)
            mk.flydsl_mxfp6_moe_stage1(**stage1, **kw)
            mk.flydsl_mxfp6_moe_stage2(**stage2, **kw)

        def op(cfg=None):
            x = _empty(tokens, hidden, dtype=dtypes.bf16)
            ids = _empty(tokens, topk, dtype=torch.int32)
            w = _empty(tokens, topk, dtype=dtypes.fp32)
            flydsl_mxfp6_moe(x, st, w, ids, config=cfg)

        # 6-bit: stage 1 writes the h tile-step-major and stage 2 reads it so, unless
        # both are told rows (as the row quantizer writes them); 8-bit: one layout
        got["stages default tile_step"] = picked(stages) == (six, six)
        got["stages rows"] = picked(lambda: stages(h_layout="rows")) == (0, 0)
        tile_step = picked(lambda: stages(h_layout="tile_step"))
        got["stages tile_step"] = tile_step == (six, six)
        # the op gives both stages the layout of its config
        got["op default tile_step"] = picked(op) == (six, six)
        got["op rows"] = picked(lambda: op(rows_config)) == (0, 0)
        # tensors without a version counter, as a serving engine makes them
        with torch.inference_mode():
            got["inference_mode"] = picked(op) == (six, six)

        # every refusal comes before a kernel is picked
        del calls[:]
        stage1, stage2 = _stage_calls(st, tokens, topk, config)
        s1, s2 = mk.flydsl_mxfp6_moe_stage1, mk.flydsl_mxfp6_moe_stage2
        m = config.route_m
        got["h_layout 'row' refused"] = _refused(s2, **stage2, h_layout="row")
        got["h_layout 'auto' refused"] = _refused(s2, **stage2, h_layout="auto")
        got["config h_layout refused"] = _refused(
            _config, st, tokens, topk, h_layout="row"
        )
        short = {**stage2, "out": stage2["out"][m:]}
        got["short out refused"] = _refused(s2, **short)
        short = {**stage1, "h_rows": stage1["h_rows"][m:]}
        got["short h_rows refused"] = _refused(s1, **short)
        short = {**stage1, "h_scales": stage1["h_scales"][m:]}
        got["short h_scales refused"] = _refused(s1, **short)
        # a tile narrower than the slabs the weights were prepared in
        narrow = dataclasses.replace(config.stage2, tile_n=st.slab_n2 // 2)
        got["tile across slabs refused"] = _refused(s2, **{**stage2, "config": narrow})
        odd = dataclasses.replace(config.stage2, tile_m=config.stage2.tile_m + 16)
        got["tile of 16 rows more refused"] = _refused(s2, **{**stage2, "config": odd})
        # 32-bit buffer offsets: the rows must stay below 4 GiB - 4 KiB
        big = {**stage1, "x_rows": _empty(2**32 - 4095)}
        got["rows of 4 GiB - 4 KiB + 1 refused"] = _refused(s1, **big)
        # both stages run on the routing tiles of the config
        other = dataclasses.replace(config, route_m=384 - m)
        got["stages on other routing tiles refused"] = _refused(op, other)
        got["no kernel picked by a refused call"] = not calls
        # ... and the largest rows are taken
        big = {**stage1, "x_rows": _empty(2**32 - 4096)}
        got["rows of 4 GiB - 4 KiB taken"] = not _refused(s1, **big) and len(calls) == 1
    bad = [k for k, v in got.items() if not v]
    assert not bad, f"h layout contract, {afmt}: {bad}"
    return {"gfx": SUPPORTED_GFX[0], "checks": len(got), "err": float(len(bad))}


_ROUTING = ("topk_weight", "topk_ids")
# fused_moe arguments this path does not implement, each off its default; the tensor
# arguments (_FUSED_TENSORS) are refused whatever they hold
_FUSED_UNSUPPORTED = {
    "activation": ActivationType.Gelu,
    "doweight_stage1": True,
    "block_size_M": 64,
    "dtype": dtypes.fp32,
    "hidden_pad": 1,
    "intermediate_pad": 1,
    "splitk": 2,
    "swiglu_limit": 7.0,
    "beta": 1.0,
    "linear_beta": 1.0,
    "gate_mode": "interleave",
    "shared_expert_id": 0,
    "stage2_scatter": object(),
    "quant_type_a": QuantType.per_1x32,
    "quant_dtype_a": dtypes.fp8,
    "quant_dtype_a2": dtypes.fp8,
    "herd_topk": 1,
    "deterministic": True,
}
_FUSED_TENSORS = (
    "expert_mask",
    "a1_scale",
    "a2_scale",
    "num_local_tokens",
    "bias1",
    "bias2",
    "shared_w1",
    "shared_w2",
    "shared_w1_scale",
    "shared_w2_scale",
    "w1_global_scale",
    "w2_global_scale",
)
# spellings of "nothing asked" that fused_moe takes besides the defaults, and the
# arguments no call of this path reads
_FUSED_NEUTRAL = (
    ("dtype", dtypes.bf16),
    ("swiglu_limit", 0),
    ("swiglu_limit", math.inf),
    ("block_size_M", 0),
    ("block_size_M", -1),
    ("splitk", 1),
    ("activation", ActivationType.Silu),
    ("gate_mode", "separated"),
    ("moe_sorting_dispatch_policy", 1),
    ("herd_routed_scaling_factor", 2.0),
    ("clamp_shared", False),
)


def _stored_mx(w1, w2, wfmt):
    """A layer's weights as an MXFP6 checkpoint stores them: (w1, w1_scale, w2,
    w2_scale) with the 6-bit codes packed four to three bytes and one E8M0 scale per
    32."""
    c1, s1 = q.quant_mx(w1, wfmt)
    c2, s2 = q.quant_mx(w2, wfmt)
    return q.pack6(c1), s1, q.pack6(c2), s2


def _stored_codes(packed):
    """The 6-bit codes of packed bytes, read as the stored layout is documented:
    element j of each four at bits [6j, 6j + 6) of three little-endian bytes."""
    b = packed.reshape(*packed.shape[:-1], -1, 3)
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    codes = (
        b0 & 63,
        ((b1 & 15) << 2) | (b0 >> 6),
        ((b2 & 3) << 4) | (b1 >> 4),
        b2 >> 2,
    )
    return torch.stack(codes, -1).reshape(*packed.shape[:-1], -1)


def _same_layer(a, b):
    """Two prepared layers hold the same fields, tensors bit for bit, tags too."""
    return all(
        (
            (
                torch.equal(x, y)
                and getattr(x, "mxfp6_fmt", None) == getattr(y, "mxfp6_fmt", None)
            )
            if isinstance(x, torch.Tensor)
            else x == y
        )
        for x, y in zip(dataclasses.astuple(a), dataclasses.astuple(b))
    )


def _fused_args(x, st, w, ids):
    """Keyword arguments of the aiter.fused_moe call on the prepared layer st."""
    return {
        "hidden_states": x,
        "w1": st.w1_frags,
        "w2": st.w2_frags,
        "topk_weight": w,
        "topk_ids": ids,
        "quant_type": QuantType.per_1x32,
        "w1_scale": st.w1_scales,
        "w2_scale": st.w2_scales,
    }


@benchmark()
def test_contract(hidden, inter, experts, topk, tokens, afmt):
    # host code only, as test_h_layout: no reference and no timing
    calls, got = [], {}
    with dry_run(SUPPORTED_GFX[0], DRY_RUN_CU), _host_only(calls):
        st = _empty_layer(hidden, inter, experts, afmt)
        good = {
            "hidden_states": _empty(tokens, hidden, dtype=dtypes.bf16),
            "weights": st,
            "topk_weight": _empty(tokens, topk, dtype=dtypes.fp32),
            "topk_ids": _empty(tokens, topk, dtype=torch.int32),
        }

        def call(**kw):
            return flydsl_mxfp6_moe(**{**good, **kw})

        def taken(**kw):
            """The call ran both stages and returned bf16 [tokens, hidden]."""
            del calls[:]
            y = call(**kw)
            return (
                len(calls) == 2
                and y.shape == (tokens, hidden)
                and y.dtype == dtypes.bf16
            )

        got["taken"] = taken()
        # int64 ids and any float weights are cast (bits: test_moe_modes)
        got["int64 ids taken"] = taken(topk_ids=_empty(tokens, topk, dtype=torch.int64))
        got["bf16 weights taken"] = taken(
            topk_weight=_empty(tokens, topk, dtype=dtypes.bf16)
        )
        got["config taken"] = taken(config=_config(st, tokens, topk))
        del calls[:]
        none = call(
            hidden_states=_empty(0, hidden, dtype=dtypes.bf16),
            topk_weight=_empty(0, topk, dtype=dtypes.fp32),
            topk_ids=_empty(0, topk, dtype=torch.int32),
        )
        got["no tokens: [0, H], nothing launched"] = (
            none.shape == (0, hidden) and none.dtype == dtypes.bf16 and not calls
        )

        # the kernels read raw pointers: every tensor outside the contract is refused
        refusals = {
            "fp32 hidden_states": {
                "hidden_states": _empty(tokens, hidden, dtype=dtypes.fp32)
            },
            "hidden_states of another width": {
                "hidden_states": _empty(tokens, hidden + 128, dtype=dtypes.bf16)
            },
            "1-D hidden_states": {"hidden_states": _empty(hidden, dtype=dtypes.bf16)},
            "strided hidden_states": {
                "hidden_states": _empty(tokens, 2 * hidden, dtype=dtypes.bf16)[:, ::2]
            },
            "hidden_states on the CPU": {
                "hidden_states": torch.empty(
                    tokens, hidden, dtype=dtypes.bf16, device="cpu"
                )
            },
            "swapped routing arguments": {
                "topk_weight": good["topk_ids"],
                "topk_ids": good["topk_weight"],
            },
            "int16 ids": {"topk_ids": _empty(tokens, topk, dtype=torch.int16)},
            "ids of another token count": {
                "topk_ids": _empty(tokens + 1, topk, dtype=torch.int32)
            },
            "1-D ids": {"topk_ids": _empty(tokens, dtype=torch.int32)},
            "integer weights": {"topk_weight": good["topk_ids"]},
            "weights of another top-k": {
                "topk_weight": _empty(tokens, topk + 1, dtype=dtypes.fp32)
            },
            "ids on the CPU": {
                "topk_ids": torch.empty(tokens, topk, dtype=torch.int32, device="cpu")
            },
        }
        for name, kw in refusals.items():
            got[f"{name} refused"] = _refused(call, **kw)
        fields = dataclasses.astuple(st)
        got["weights not a prepared layer refused"] = _raises(
            TypeError, call, weights=fields
        )
        stage = _config(st, tokens, topk).stage1
        got["config of one stage refused"] = _raises(TypeError, call, config=stage)
        with mock.patch.object(mxfp6_moe, "get_gfx_runtime", lambda: "gfx942"):
            got["gfx942 refused"] = _raises(RuntimeError, call)
        got["no kernel picked by a refused call"] = not calls

        # through aiter.fused_moe: the tag on the prepared weights sends the call here
        via = _fused_args(good["hidden_states"], st, *map(good.get, _ROUTING))

        def fused(**kw):
            return fused_moe(**{**via, **kw})

        def fused_taken(**kw):
            del calls[:]
            y = fused(**kw)
            return (
                len(calls) == 2
                and y.shape == (tokens, hidden)
                and y.dtype == dtypes.bf16
            )

        got["fused_moe taken"] = fused_taken()
        for name, value in _FUSED_NEUTRAL:
            got[f"fused_moe {name}={value} taken"] = fused_taken(**{name: value})
        buf = _empty(tokens, hidden, dtype=dtypes.bf16)
        del calls[:]
        got["fused_moe output= returned"] = fused(output=buf) is buf and len(calls) == 2
        del calls[:]

        # an argument this path does not implement is refused, not ignored; a new
        # argument of fused_moe fails here until it is in one of the three lists
        t = _empty(1)
        unsupported = dict(_FUSED_UNSUPPORTED, **dict.fromkeys(_FUSED_TENSORS, t))
        names = set(inspect.signature(fused_moe).parameters) - set(via)
        got["every other fused_moe argument is classified"] = names == set(
            unsupported
        ) | {"output", *mxfp6_moe._FUSED_MOE_UNUSED}
        for name, value in unsupported.items():
            got[f"fused_moe {name} refused"] = _raises(
                NotImplementedError, fused, **{name: value}
            )
        other = _empty_layer(hidden + 128, inter, experts, afmt)
        copy = st.w1_frags.clone()
        got["a copy drops the tag"] = not hasattr(copy, "mxfp6_fmt")
        retagged = st.w1_frags.clone()
        retagged.mxfp6_fmt = ("e2m1", "e2m3")
        refusals = {
            "quant_type No": {"quant_type": QuantType.No},
            "w1 copied without its tag": {"w1": copy},
            "w2 copied without its tag": {"w2": st.w2_frags.clone()},
            "an unknown tag": {"w1": retagged, "w2": retagged},
            "no w1_scale": {"w1_scale": None},
            "no w2_scale": {"w2_scale": None},
            "swapped scales": {"w1_scale": st.w2_scales, "w2_scale": st.w1_scales},
            "w1 of another layer": {"w1": other.w1_frags},
            "w1_scale on the CPU": {
                "w1_scale": torch.empty(
                    st.w1_scales.shape, dtype=torch.int32, device="cpu"
                )
            },
            "fp32 hidden_states": {
                "hidden_states": _empty(tokens, hidden, dtype=dtypes.fp32)
            },
        }
        for name, kw in refusals.items():
            got[f"fused_moe {name} refused"] = _refused(fused, **kw)
        for name, bad in {
            "another shape": _empty(tokens + 1, hidden, dtype=dtypes.bf16),
            "another dtype": _empty(tokens, hidden, dtype=dtypes.fp32),
        }.items():
            got[f"fused_moe output= of {name} refused"] = _raises(
                RuntimeError, fused, output=bad
            )
        got["no kernel picked by a refused fused_moe call"] = not calls

    # weight preparation is torch only: a layer of the smallest shape, on the CPU
    E, H, I = 2, 256, 128
    w1, w2 = torch.randn(E, 2 * I, H), torch.randn(E, H, I)
    cpu = prepare_mxfp6_moe_weights(w1, w2, a_fmt=afmt)
    bn1, bn2 = cpu.slab_n1, cpu.slab_n2
    got["prepared weights are tagged"] = (
        cpu.w1_frags.mxfp6_fmt == cpu.w2_frags.mxfp6_fmt == (afmt, "e2m3")
    )
    got["prepared on the CPU"] = (
        cpu.w1_frags.shape == (E, 2 * I // bn1, H // 128, bn1 // 16, 1536)
        and cpu.w1_scales.shape == (E, 2 * I // 16, q.kq(H), 64)
        and cpu.w2_frags.shape == (E, H // bn2, I // 128, bn2 // 16, 1536)
        and cpu.w2_scales.shape == (E, H // 16, q.kq(I), 64)
        and (cpu.experts, cpu.model_dim, cpu.inter_dim) == (E, H, I)
    )
    prep = prepare_mxfp6_moe_weights
    got["2-D w1 refused"] = _refused(prep, w1[0], w2)
    got["w1 without up rows refused"] = _refused(prep, w1[:, :I], w2)
    got["w2 of another expert count refused"] = _refused(prep, w1, w2[:1])
    got["hidden of 192 refused"] = _refused(prep, w1[:, :, :192], w2[:, :192])
    got["integer w1 refused"] = _refused(prep, w1.int(), w2)
    got["unknown a_fmt refused"] = _refused(prep, w1, w2, a_fmt="e2m1")
    got["e4m3 weights refused"] = _refused(prep, w1, w2, w_fmt="e4m3")

    # weights that are MXFP6 already: the stored bytes and scales, only preshuffled
    packed = prepare_mxfp6_moe_packed_weights
    p1, s1, p2, s2 = stored = _stored_mx(w1, w2, "e2m3")
    got["stored bytes hold the codes in the documented layout"] = torch.equal(
        _stored_codes(p1), q.quant_mx(w1, "e2m3")[0]
    )
    got["packed weights give the layer of the float weights"] = _same_layer(
        packed(*stored, a_fmt=afmt), cpu
    )
    got["packed e3m2 weights give the layer of the float weights"] = _same_layer(
        packed(*_stored_mx(w1, w2, "e3m2"), a_fmt=afmt, w_fmt="e3m2"),
        prep(w1, w2, a_fmt=afmt, w_fmt="e3m2"),
    )
    refusals = {
        "float w1": (w1, s1, p2, s2),
        "unpacked codes as w1": (_stored_codes(p1), s1, p2, s2),
        "swapped scales": (p1, s2, p2, s1),
        "int32 w1_scale": (p1, s1.int(), p2, s2),
        "w2 of another expert count": (p1, s1, p2[:1], s2[:1]),
        "an odd row count": (p1[:, :-1], s1[:, :-1], p2, s2),
        "hidden of 192": (p1[..., :144], s1[..., :6], p2[:, :192], s2[:, :192]),
        "2-D w1": (p1[0], s1, p2, s2),
    }
    for name, tensors in refusals.items():
        got[f"packed: {name} refused"] = _refused(packed, *tensors)
    got["packed: unknown a_fmt refused"] = _refused(packed, *stored, a_fmt="e2m1")
    got["packed: e4m3 weights refused"] = _refused(packed, *stored, w_fmt="e4m3")
    bad = [k for k, v in got.items() if not v]
    assert not bad, f"op contract, {afmt}: {bad}"
    return {"gfx": SUPPORTED_GFX[0], "checks": len(got), "err": float(len(bad))}


@benchmark()
def test_tuned_dispatch(hidden, inter, experts, topk, tokens, afmt):
    # host code only, as test_h_layout: no reference and no timing
    calls, got = [], {}
    six = q.FORMATS[afmt][3] == 6
    with dry_run(SUPPORTED_GFX[0], DRY_RUN_CU), _host_only(calls):
        st = _empty_layer(hidden, inter, experts, afmt)
        heuristic = _config(st, tokens, topk)
        # kernels that fit the layer and are not the heuristic's: the other routing
        # tile, and the other layout of a 6-bit h
        tuned = _config(
            st,
            tokens,
            topk,
            route_m=384 - heuristic.route_m,
            h_layout="rows" if six else heuristic.h_layout,
        )
        gfx, cu = get_gfx_runtime(), get_cu_num()

        def row(**cells):
            return {
                "gfx": gfx,
                "cu_num": cu,
                "token": tokens,
                "model_dim": hidden,
                "inter_dim": inter,
                "expert": experts,
                "topk": topk,
                **policy.TUNED_CALL,
                "q_dtype_a": policy.Q_DTYPE_NAMES[afmt],
                "q_dtype_w": policy.Q_DTYPE_NAMES[st.w_fmt],
                **policy.row_results(tuned),
                **cells,
            }

        def op(rows, t=tokens):
            """The kernels the op asks for with a tuned config of rows."""
            del calls[:]
            with _tuned_table(rows):
                x = _empty(t, hidden, dtype=dtypes.bf16)
                ids = _empty(t, topk, dtype=torch.int32)
                flydsl_mxfp6_moe(x, st, _empty(t, topk, dtype=dtypes.fp32), ids)
            return _asked(calls)

        want, base = _names(tuned, afmt), _names(heuristic, afmt)
        got["the row differs from the heuristic"] = want != base
        got["no rows: heuristic"] = op([]) == base
        got["row of the call: tuned"] = op([row()]) == want
        # a table rewritten through pandas holds integers as floats
        floats = {"token": f"{tokens}.0", "block_m": f"{tuned.route_m}.0"}
        got["float cells: tuned"] = op([row(**floats)]) == want
        # any other device or shape keeps the heuristic
        for column, cell in (
            ("gfx", "gfx942"),
            ("cu_num", cu // 2),
            ("token", tokens + 1),
            ("model_dim", 2 * hidden),
            ("inter_dim", 2 * inter),
            ("expert", experts + 1),
            ("topk", topk + 1),
            ("q_dtype_a", policy.Q_DTYPE_NAMES["e4m3" if six else "e2m3"]),
            ("q_dtype_w", policy.Q_DTYPE_NAMES["e3m2"]),
        ):
            got[f"row of another {column}: heuristic"] = (
                op([row(**{column: cell})]) == base
            )
        other = _config(st, tokens + 1, topk)
        got["other token count: its heuristic"] = op([row()], tokens + 1) == _names(
            other, afmt
        )
        both = [row(gfx="gfx942"), row(), row(token=tokens + 1)]
        got["its row among others: tuned"] = op(both) == want

        # every refusal comes before a kernel is picked (op clears calls)
        name1, name2 = (
            policy.row_results(tuned)[k] for k in ("kernelName1", "kernelName2")
        )
        narrow = dataclasses.replace(tuned.stage2, tile_n=st.slab_n2 // 2)
        for what, rows in {
            "another activation": [row(act_type="ActivationType.Gelu")],
            "another quant type": [row(q_type="QuantType.per_1x128")],
            "doweight_stage1": [row(doweight_stage1=1)],
            "an unknown format": [row(q_dtype_a="torch.float4_e2m1fn_x2")],
            "an unknown kernel": [row(kernelName1="moe_ck2stages_gemm1_256x32x64x128")],
            "stage kernels swapped": [row(kernelName1=name2, kernelName2=name1)],
            "an unknown h_layout": [row(h_layout="row")],
            "kernels off the routing tiles": [row(block_m=384 - tuned.route_m)],
            "a missing column": [{k: v for k, v in row().items() if k != "h_layout"}],
            "a tile across the weight slabs": [
                row(kernelName2=policy.stage_kernel_name(False, narrow))
            ],
        }.items():
            got[f"{what} refused"] = _refused(op, rows) and not calls
        # a repeated key is refused where aiter merges a family's tuned files
        twice = _raises(RuntimeError, op, [row(), row()])
        got["a repeated key refused"] = twice and not calls
        # a misfit row of another shape is never looked at
        got["misfit row of another shape: heuristic"] = (
            op(
                [
                    row(
                        token=tokens + 1,
                        kernelName2=policy.stage_kernel_name(False, narrow),
                    )
                ]
            )
            == base
        )
        # the names a row holds read back as the kernels
        for stage1, stage in ((True, tuned.stage1), (False, heuristic.stage2)):
            name = policy.stage_kernel_name(stage1, stage)
            ok = policy.stage_config_from_name(name) == (stage1, stage)
            got[f"{name} reads back"] = ok
    bad = [k for k, v in got.items() if not v]
    assert not bad, f"tuned dispatch, {afmt}: {bad}"
    return {"gfx": SUPPORTED_GFX[0], "checks": len(got), "err": float(len(bad))}


# what a whole-op case must reach, from the GEMM kernels its stage launches ask for
# and the launchers of the row-padding kernel it built
REACHES = {
    "padded_rows": lambda calls, pads: bool(pads) and any(c["PAD_A"] for c in calls),
    "half_tiles": lambda calls, pads: any(c["HALF_TILES"] for c in calls),
    "hidden128": lambda calls, pads: any(
        c["epi"] == "swiglu_q" and c["K"] == 128 for c in calls
    ),
}


@benchmark()
def test_moe_paths(hidden, inter, experts, topk, tokens, afmt, wfmt, reach):
    calls, pads = [], []
    real_pad = mk._get_compiled_pad_rows

    def spy_pad(*args):
        pads.append(args)
        return real_pad(*args)

    # the path is the heuristic's choice: no tuned rows
    with (
        _tuned_table(),
        _spy_gemm(calls),
        mock.patch.object(mk, "_get_compiled_pad_rows", spy_pad),
    ):
        ret = test_mxfp6_moe(tokens, hidden, inter, experts, topk, wfmt, afmt)
    ok = REACHES[reach](calls, pads)
    assert ok, (
        f"{hidden}x{inter} E{experts} top-{topk} T{tokens} {afmt} no longer reaches "
        f"{reach}: pick another shape"
    )
    tails = f"{hidden // 128 % 4}/{inter // 128 % 4}"
    ret.update({"K tail s1/s2": tails, "reached": ok})
    return ret


@benchmark()
def test_moe_modes(tokens, hidden, inter, experts, topk, wfmt, afmt):
    w1, w2, x, ids, w = _moe_inputs(tokens, hidden, inter, experts, topk)
    st = prepare_mxfp6_moe_weights(w1, w2, a_fmt=afmt, w_fmt=wfmt)
    ref = run_torch_mx(x, ids, w, w1, w2, wfmt, afmt)
    six = int(q.FORMATS[afmt][3] == 6)
    op = flydsl_mxfp6_moe
    ids64, w64 = ids.long(), w.double()

    def layout(h_layout):  # the heuristic's kernels for this layout of the h
        config = _config(st, tokens, topk, h_layout=h_layout)
        return lambda: op(x, st, w, ids, config=config)

    def inference_mode():
        # as a serving engine runs the model: no tensor made here has a version
        # counter
        with torch.inference_mode():
            return op(x.clone(), st, w, ids)

    def no_grad():
        with torch.no_grad():
            return op(x, st, w, ids)

    candidates = {
        "default": lambda: op(x, st, w, ids),
        "rows": layout("rows"),
        "tile_step": layout("tile_step"),
        "inference_mode": inference_mode,
        "no_grad": no_grad,
        "requires_grad": lambda: op(x.clone().requires_grad_(), st, w, ids),
        # int64 ids and fp64 routing weights, cast by the op
        "cast": lambda: op(x, st, w64, ids64),
    }
    flops, nbytes = _moe_work(tokens, hidden, inter, experts, topk, afmt)
    ret, outs, calls = {"gfx": get_gfx_runtime()}, {}, []
    for name, fn in candidates.items():
        del calls[:]
        # "default" is the heuristic's choice: no tuned rows
        with _tuned_table(), _spy_gemm(calls):
            out, us = run_perftest(fn, num_iters=20, num_warmup=3, **TIMING)
        # stage 2 must read the layout stage 1 wrote; "rows" / "tile_step" must run
        # the kernels of that layout
        picks = {c["H_TILE_STEP"] for c in calls}
        want = {"rows": 0, "tile_step": six}.get(name, six)
        assert picks == {want}, f"{name}: stage launches picked h layouts {picks}"
        outs[name] = out.detach()
        err, _, _ = _check_moe(outs[name], ref, f"mxfp6_moe {wfmt}/{afmt} {name}: ")
        assert torch.equal(
            outs[name], outs["default"]
        ), f"{name}: output bits differ from the default call's"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# one prepared layer: the rows of a layer follow each other in the tuned config
_TUNED_LAYER = {}


def _tuned_layer(hidden, inter, experts, wfmt, afmt):
    key = (hidden, inter, experts, wfmt, afmt)
    if key not in _TUNED_LAYER:
        _TUNED_LAYER.clear()
        torch.cuda.empty_cache()
        g = torch.Generator(device="cuda").manual_seed(0)
        w1, w2 = _layer_inputs(g, hidden, inter, experts)
        st = prepare_mxfp6_moe_weights(w1, w2, a_fmt=afmt, w_fmt=wfmt)
        _TUNED_LAYER[key] = (w1, w2, st)
    return _TUNED_LAYER[key]


def _tuned_rows():
    """The rows of the tuned config for this device: (token, hidden, inter, experts,
    topk, wfmt, afmt) -> the kernels the row names."""
    device = (get_gfx_runtime(), get_cu_num())
    table = policy.load_tuned_configs(AITER_CONFIGS.AITER_CONFIG_MXFP6_MOE_FILE)
    return {
        (t, h, i, e, k, wf, af): config
        for (*dev, t, h, i, e, k, af, wf), config in table.items()
        if tuple(dev) == device
    }


@benchmark()
def test_tuned_row(tokens, hidden, inter, experts, topk, wfmt, afmt):
    w1, w2, st = _tuned_layer(hidden, inter, experts, wfmt, afmt)
    g = torch.Generator(device="cuda").manual_seed(1)
    x, ids, w = _call_inputs(g, tokens, hidden, experts, topk)
    tuned = _tuned_rows()[tokens, hidden, inter, experts, topk, wfmt, afmt]
    heuristic = _config(st, tokens, topk)

    def run_tuned():  # as a caller runs the op: the row is picked by the op
        return flydsl_mxfp6_moe(x, st, w, ids)

    def run_heuristic():
        return flydsl_mxfp6_moe(x, st, w, ids, config=heuristic)

    calls = []
    with _spy_gemm(calls):
        out = run_tuned()
    assert _asked(calls) == _names(
        tuned, afmt
    ), f"the op asked for {_asked(calls)}, the tuned row names {tuned}"
    assert torch.equal(
        out, run_heuristic()
    ), "output bits differ from the heuristic's kernels'"
    ref = run_torch_mx(x, ids, w, w1, w2, wfmt, afmt)
    err, row, med = _check_moe(out, ref, f"mxfp6_moe tuned {wfmt}/{afmt}: ")
    # alternating rounds: the two are compared within this process
    us_t, us_h = [], []
    for _ in range(3):
        for fn, samples in ((run_tuned, us_t), (run_heuristic, us_h)):
            samples.append(run_perftest(fn, num_iters=20, num_warmup=3, **TIMING)[1])
    us = statistics.median(us_t)
    flops, nbytes = _moe_work(tokens, hidden, inter, experts, topk, afmt)
    return {
        "gfx": get_gfx_runtime(),
        "row": "heuristic's kernels" if tuned == heuristic else "tuned",
        "flydsl us": us,
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": err,
        "max row rel_l2": row,
        "median row rel_l2": med,
        "heuristic us": statistics.median(us_h),
        "heuristic / tuned": statistics.median(h / t for h, t in zip(us_h, us_t)),
    }


@benchmark()
def test_fused_moe(tokens, hidden, inter, experts, topk, wfmt, afmt):
    w1, w2, x, ids, w = _moe_inputs(tokens, hidden, inter, experts, topk)
    st = prepare_mxfp6_moe_weights(w1, w2, a_fmt=afmt, w_fmt=wfmt)
    via = _fused_args(x, st, w, ids)
    buf = torch.full_like(x, float("nan"))

    def direct():
        return flydsl_mxfp6_moe(x, st, w, ids)

    def fused(**kw):
        return fused_moe(**{**via, **kw})

    def inference_mode():
        with torch.inference_mode():
            return fused(hidden_states=x.clone())

    want = direct()
    err, row, med = _check_moe(
        want, run_torch_mx(x, ids, w, w1, w2, wfmt, afmt), f"mxfp6_moe {wfmt}/{afmt}: "
    )
    calls = []
    with _spy_gemm(calls):
        out = fused()
    assert len(calls) == 2, f"fused_moe launched {len(calls)} stage kernels"
    assert torch.equal(out, want), "fused_moe: output bits differ from the op's"
    assert fused(output=buf) is buf, "fused_moe did not return its output= buffer"
    assert torch.equal(buf, want), "fused_moe output=: bits differ from the op's"
    assert torch.equal(inference_mode(), want), "inference_mode: bits differ"
    # a layer prepared from the weights as an MXFP6 checkpoint stores them
    pk = prepare_mxfp6_moe_packed_weights(
        *_stored_mx(w1, w2, wfmt), a_fmt=afmt, w_fmt=wfmt
    )
    assert torch.equal(
        fused_moe(**_fused_args(x, pk, w, ids)), want
    ), "fused_moe on packed weights: output bits differ from the op's"
    del pk
    xc = x.clone()  # in place: the result is copied once the kernels are done
    assert fused(hidden_states=xc, output=xc) is xc, "output=x: not returned"
    assert torch.equal(xc, want), "fused_moe output=x: bits differ from the op's"
    for name, value in _FUSED_NEUTRAL:
        assert torch.equal(
            fused(**{name: value}), want
        ), f"fused_moe {name}={value}: output bits differ from the op's"
    # alternating rounds: the two are compared within this process
    us_d, us_f = [], []
    for _ in range(3):
        for fn, samples in ((direct, us_d), (fused, us_f)):
            samples.append(run_perftest(fn, num_iters=20, num_warmup=3, **TIMING)[1])
    us = statistics.median(us_f)
    flops, nbytes = _moe_work(tokens, hidden, inter, experts, topk, afmt)
    return {
        "gfx": get_gfx_runtime(),
        "fused_moe us": us,
        "fused_moe TFLOPS": flops / us / 1e6,
        "fused_moe TB/s": nbytes / us / 1e6,
        "fused_moe err": err,
        "max row rel_l2": row,
        "median row rel_l2": med,
        "direct us": statistics.median(us_d),
        "fused_moe / direct": statistics.median(f / d for f, d in zip(us_f, us_d)),
    }


def summarize(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL MXFP6 grouped MoE",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (2048, 1024, 8, 4),
            (6272, 2560, 64, 4),
            (2048, 1024, 160, 4),
            (1152, 640, 16, 8),
            (6144, 2048, 64, 4),
        ],
        help="hidden,inter,experts,topk (hidden 6272: N %% 256 != 0 tiles;\n"
        "E=160 at 4096 tokens: ~100 rows per expert -> 128-row tiles)",
    )
    parser.add_argument("-t", "--tokens", type=int, nargs="*", default=[300, 4096])
    parser.add_argument(
        "--wfmt", nargs="*", default=["e2m3", "e3m2"], help="weight format(s)"
    )
    parser.add_argument(
        "--afmt", nargs="*", default=["e2m3", "e4m3"], help="activation format(s)"
    )
    parser.add_argument(
        "--mfma",
        nargs="*",
        default=[
            "e2m3,e2m3",
            "e3m2,e3m2",
            "e4m3,e2m3",
            "e2m3,e3m2",
            "e4m3,e4m3",
            "e2m1,e2m1",
            "e4m3,e3m2",
            "e2m3,e4m3",
        ],
        help="A,B format pairs of the MFMA contract",
    )
    parser.add_argument(
        "--quant_k", type=int, nargs="*", default=[640, 2048, 10496], help="row width"
    )
    parser.add_argument(
        "--route",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (4096, 4, 64, 256),
            (300, 8, 160, 256),
            (32768, 8, 64, 256),
            (5000, 4, 192, 128),
        ],
        help="tokens,topk,experts,route_m",
    )
    parser.add_argument(
        "--agpr",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(8192, 4608, 64, 4, 4096), (6144, 2048, 64, 4, 4096)],
        help="hidden,inter,experts,topk,tokens of the AGPR form and the h layout\n"
        "contract (no GPU)",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=[
            "1408,3072,8,2,4096,e2m3,padded_rows",
            "10496,1024,8,2,1024,e4m3,half_tiles",
            "128,768,8,2,4096,e2m3,hidden128",
        ],
        help="hidden,inter,experts,topk,tokens,afmt,reach with reach one of\n"
        f"{', '.join(REACHES)} (defaults: K tails of 3 / 2 steps in stage 1,\n"
        "2 in stage 2)",
    )
    parser.add_argument(
        "--modes",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(2048, 1024, 8, 4, 4096), (1152, 640, 16, 8, 300)],
        help="hidden,inter,experts,topk,tokens of the h layouts, the autograd modes,\n"
        "the op contract and fused_moe (256-row and 128-row tiles)",
    )
    args = parser.parse_args()

    # no GPU needed: aiter's dry run
    summarize(
        "agpr form",
        [test_agpr_form(*c, af) for c, af in itertools.product(args.agpr, args.afmt)],
    )
    summarize(
        "h layout",
        [
            test_h_layout(*c, af)
            for c, af in itertools.product(args.agpr + args.modes, args.afmt)
        ],
    )
    summarize(
        "contract",
        [test_contract(*c, af) for c, af in itertools.product(args.modes, args.afmt)],
    )
    summarize(
        "tuned dispatch",
        [
            test_tuned_dispatch(*c, af)
            for c, af in itertools.product(args.agpr + args.modes, args.afmt)
        ],
    )
    gfx = get_gfx_runtime() if torch.cuda.is_available() else "no GPU"
    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning("mxfp6_moe unsupported on %s; skipping the GPU tests", gfx)
        return

    summarize("mfma contract", [test_mfma_contract(*p.split(",")) for p in args.mfma])
    summarize(
        "quant_rows",
        [
            test_quant_rows(77, k, f)
            for k, f in itertools.product(args.quant_k, q.FORMATS)
        ],
    )
    summarize(
        "route",
        [test_route(r, *c) for r, c in itertools.product(ROUTERS, args.route)],
    )
    summarize(
        "mxfp6_moe",
        [
            test_mxfp6_moe(t, h, i, e, k, wf, af)
            for (h, i, e, k), t, wf, af in itertools.product(
                args.shape, args.tokens, args.wfmt, args.afmt
            )
        ],
    )
    paths = [c.split(",") for c in args.paths]
    summarize(
        "mxfp6_moe paths",
        [test_moe_paths(*map(int, c[:5]), c[5], args.wfmt[0], c[6]) for c in paths],
    )
    summarize(
        "mxfp6_moe modes",
        [
            test_moe_modes(t, h, i, e, k, args.wfmt[0], af)
            for (h, i, e, k, t), af in itertools.product(args.modes, args.afmt)
        ],
    )
    summarize(
        "fused_moe",
        [
            test_fused_moe(t, h, i, e, k, args.wfmt[0], af)
            for (h, i, e, k, t), af in itertools.product(args.modes, args.afmt)
        ],
    )
    # every row of the tuned config for this device, a layer's rows together
    rows = sorted(_tuned_rows(), key=lambda r: (r[1:], r[0]))
    summarize("mxfp6_moe tuned rows", [test_tuned_row(*r) for r in rows])
    _TUNED_LAYER.clear()


if __name__ == "__main__":
    main()
