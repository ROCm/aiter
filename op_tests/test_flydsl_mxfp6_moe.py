# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL MXFP6 grouped MoE (aiter.ops.flydsl.mxfp6_moe), gfx950.

Correctness against torch references + perf, one markdown table per test function:
  test_mfma_contract  scaled-MFMA operand contract per format pair (packing, K order, E8M0 lanes)
  test_quant_rows     FlyDSL MX row quantization == torch recipe, bit-exact codes and scales
  test_route          one-block and multi-block routers: placement invariants
  test_mxfp6_moe      whole op vs a torch reference with identical MX numerics, bitwise
                      run-to-run determinism, rel L2 vs bf16
Tests of kernels/mxfp6_moe_gemm.py's own choices (they belong with that kernel: they fail on one without them):
  test_agpr_form      no GPU (aiter's dry run). The 256x256 tiles keep their accumulators in AGPRs through the
                      FlyDSL-unstable rocdl.experimental.sched_agpr and the LLVM option amdgpu-mfma-vgpr-form. A
                      FlyDSL that keeps both names while its LLVM ignores them compiles a slow form with no error:
                      fails unless every f8x8 kernel has 0 v_accvgpr_* in its K loop and 0 B of scratch
  test_h_layout       no GPU (dry run, nothing compiled). run_gg6e's contract for the 6-bit h between the stages:
                      the layout the two stage calls pick as forward makes them (h_layout="tile_step"), with no
                      keyword (rows), with h_layout="rows", with H_LAYOUT = "tile_step" and under inference_mode, and
                      the ValueErrors for bad keywords and a short output
  test_moe_paths      test_mxfp6_moe on shapes whose kernels its default grid never launches (padded rows and their
                      copy kernel, A8W6 half tiles, hidden 128, K tails of 2 and 3); fails if the shape stops
                      reaching its path
  test_moe_modes      whole op with the row-major h and with the tile-step-major h (both stage calls with that
                      h_layout), under torch.inference_mode() and torch.no_grad(), and with an x that requires
                      grad: each vs the torch reference and bitwise equal to the default call

    python3 op_tests/test_flydsl_mxfp6_moe.py
    python3 op_tests/test_flydsl_mxfp6_moe.py -s 8192,4608,64,4 -t 4096 32768 --wfmt e2m3 --afmt e2m3 e4m3
"""

import argparse
import importlib
import itertools
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import torch

import aiter
import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter import dtypes
from aiter.aot.flydsl.dry_run import dry_run
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import mxfp6_moe_quant as q
from aiter.ops.flydsl.kernels import mxfp6_moe_aux as ak
from aiter.ops.flydsl.kernels import mxfp6_moe_gemm as gemm
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
from aiter.test_common import benchmark, checkAllclose, run_perftest
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

SUPPORTED_GFX = ["gfx950"]  # V_MFMA_SCALE_F32_16X16X128_F8F6F4, 160 KiB LDS


# ------------------------------------------------------------------------------------------
# torch references (not timed)
# ------------------------------------------------------------------------------------------
def ref_quant_dequant_e2m3(x):
    """OCP MX E2M3: E8M0 scale 2^ceil(log2(amax / 7.5)) per 32, round-half-even onto the grid, clamp."""
    b = x.float().reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    amax = b.abs().amax(-1, keepdim=True)
    e = torch.ceil(torch.log2((amax / 7.5).clamp(min=2.0**-126)))
    scale = torch.exp2(torch.where(amax > 0, e, torch.full_like(e, -127.0)))
    u = (b / scale).abs().clamp(max=7.5)
    step = torch.exp2(torch.floor(torch.log2(u.clamp(min=1.0))) - 3)
    r = torch.round(u / step) * step
    return (torch.copysign(r.clamp(max=7.5), b) * scale).reshape(x.shape)


def qd(x, fmt):
    return q.dequant_mx(*q.quant_mx(x, fmt), fmt)


def run_torch_mx(x, ids, w, w1, w2, wfmt, afmt):
    """MoE with the op's numerics: weights in wfmt, x and the SwiGLU output in afmt (dequantized per expert)."""
    inter = w2.shape[2]
    xq = qd(x.float(), afmt).to(dtypes.bf16)
    out = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=dtypes.fp32)
    for e in range(w1.shape[0]):
        t, j = (ids == e).nonzero(as_tuple=True)
        if len(t):
            h = xq[t].float() @ qd(w1[e].float(), wfmt).t()
            a = (torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]).to(dtypes.bf16)
            y = qd(a.float(), afmt) @ qd(w2[e].float(), wfmt).t()
            out.index_add_(0, t, y.to(dtypes.bf16).float() * w[t, j][:, None])
    return out.to(dtypes.bf16)


def run_torch_bf16(x, ids, w, w1, w2):
    inter = w2.shape[2]
    out = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=dtypes.fp32)
    for e in range(w1.shape[0]):
        t, j = (ids == e).nonzero(as_tuple=True)
        if len(t):
            h = x[t] @ w1[e].t()
            y = (torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]) @ w2[e].t()
            out.index_add_(0, t, y.float() * w[t, j][:, None])
    return out.to(dtypes.bf16)


# ------------------------------------------------------------------------------------------
# 1. scaled-MFMA operand contract: one wave, prepacked lane words, vs float64
# ------------------------------------------------------------------------------------------
_FMT = {"e4m3": 0, "e2m3": 2, "e3m2": 3, "e2m1": 4}
_BITS = {"e4m3": 8, "e2m3": 6, "e3m2": 6, "e2m1": 4}
_PROBE = {}


def _decode_table(fmt):
    if fmt == "e4m3":
        return torch.arange(256, dtype=torch.uint8, device="cpu").view(torch.float8_e4m3fn).double()
    ebits, mbits = {"e2m3": (2, 3), "e3m2": (3, 2), "e2m1": (2, 1)}[fmt]
    bias = (1 << (ebits - 1)) - 1
    vals = []
    for c in range(1 << (1 + ebits + mbits)):
        s = -1.0 if c >> (ebits + mbits) else 1.0
        e, m = (c >> mbits) & ((1 << ebits) - 1), c & ((1 << mbits) - 1)
        mag = (m / (1 << mbits)) * 2.0 ** (1 - bias) if e == 0 else (1 + m / (1 << mbits)) * 2.0 ** (e - bias)
        vals.append(s * mag)
    return torch.tensor(vals, dtype=torch.float64, device="cpu")


def _pack_lane_words(codes, bits):
    """[16, 128] codes -> [64 lanes, bits dwords]: FP6/FP4 lane = K[32g, +32); FP8 lane = K[16g,+16) ++ K[64+16g,+16)."""
    out = torch.zeros(64, bits, dtype=torch.int64, device="cpu")
    for lane in range(64):
        row, g = lane % 16, lane // 16
        if bits == 8:
            ks = list(range(16 * g, 16 * g + 16)) + list(range(64 + 16 * g, 64 + 16 * g + 16))
        else:
            ks = list(range(32 * g, 32 * g + 32))
        acc = 0
        for j, k in enumerate(ks):
            acc |= int(codes[row, k]) << (bits * j)
        for wd in range(bits):
            out[lane, wd] = (acc >> (32 * wd)) & 0xFFFFFFFF
    out[out >= 2**31] -= 2**32
    return out.to(torch.int32)


def _probe(fa, fb):
    if (fa, fb) in _PROBE:
        return _PROBE[(fa, fb)]
    nwa, nwb = _BITS[fa], _BITS[fb]

    @flyc.kernel(name=f"mx6_test_mfma_{fa}_{fb}", known_block_size=[64, 1, 1])
    def k(A: fx.Pointer, B: fx.Pointer, SA: fx.Pointer, SB: fx.Pointer, C: fx.Pointer):
        tid = fx.Int32(gpu.thread_id("x"))
        i32 = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
        f32 = fx.PointerType.get(T.f32, address_space=fx.AddressSpace.Global, alignment=4)
        a = fx.inttoptr(i32, fx.Int64(fx.ptrtoint(A)))
        b = fx.inttoptr(i32, fx.Int64(fx.ptrtoint(B)))
        sa = fx.inttoptr(i32, fx.Int64(fx.ptrtoint(SA)))
        sb = fx.inttoptr(i32, fx.Int64(fx.ptrtoint(SB)))
        c = fx.inttoptr(f32, fx.Int64(fx.ptrtoint(C)))
        av = fx.Vector.from_elements([a[tid * fx.Int32(nwa) + fx.Int32(j)] for j in range_constexpr(nwa)], fx.Int32)
        bv = fx.Vector.from_elements([b[tid * fx.Int32(nwb) + fx.Int32(j)] for j in range_constexpr(nwb)], fx.Int32)
        # raw scaled MFMA: this probe pins the exact instruction contract the op relies on
        r = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32), [av, bv, fx.Vector.filled(4, 0.0, fx.Float32), _FMT[fa], _FMT[fb], 0, sa[tid], 0, sb[tid]]))
        for i in range_constexpr(4):
            c[tid * fx.Int32(4) + fx.Int32(i)] = r[i]

    @flyc.jit
    def launch(A: fx.Pointer, B: fx.Pointer, SA: fx.Pointer, SB: fx.Pointer, C: fx.Pointer, stream: fx.Stream):
        k(A, B, SA, SB, C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    _PROBE[(fa, fb)] = launch
    return launch


@benchmark()
def test_mfma_contract(fmt_a, fmt_b):
    gen = torch.Generator(device="cpu").manual_seed(0)
    codes = []
    for f in (fmt_a, fmt_b):
        tab = _decode_table(f)
        ok = torch.nonzero(torch.isfinite(tab)).flatten()
        codes.append(ok[torch.randint(len(ok), (16, 128), generator=gen, device="cpu")])
    sa = torch.randint(120, 135, (16, 4), generator=gen, device="cpu")  # distinct E8M0 per (row, K-block)
    sb = torch.randint(120, 135, (16, 4), generator=gen, device="cpu")
    va = _decode_table(fmt_a)[codes[0]] * torch.repeat_interleave(2.0 ** (sa.double() - 127), 32, 1)
    vb = _decode_table(fmt_b)[codes[1]] * torch.repeat_interleave(2.0 ** (sb.double() - 127), 32, 1)
    ref = va @ vb.T
    lane = torch.arange(64, device="cpu")
    out = torch.full((64, 4), float("nan"), dtype=dtypes.fp32, device="cuda")
    args = [_pack_lane_words(codes[0], _BITS[fmt_a]).cuda(), _pack_lane_words(codes[1], _BITS[fmt_b]).cuda(),
            sa[lane % 16, lane // 16].int().cuda(), sb[lane % 16, lane // 16].int().cuda(), out]
    flyc.compile(_probe(fmt_a, fmt_b), *[ptr_arg(a) for a in args], torch.cuda.current_stream())
    o = out.cpu().double()
    got = torch.empty(16, 16, dtype=torch.float64, device="cpu")
    for ln in range(64):
        for i in range(4):
            got[4 * (ln // 16) + i, ln % 16] = o[ln, i]
    # E2M3 / E2M1 products are exact; E3M2 / E4M3 see the MFMA's below-fp32 internal accumulation
    err = checkAllclose(ref.float(), got.float(), rtol=1e-4, atol=1e-4 * ref.abs().max().item(),
                        tol_err_ratio=0.0, msg=f"mfma {fmt_a} x {fmt_b}: ")
    assert err == 0, f"{fmt_a} x {fmt_b}: {err:.2%} outside tolerance"
    return {"gfx": get_gfx_runtime(), "rel_l2": ((got - ref).norm() / ref.norm()).item(), "err": err}


# ------------------------------------------------------------------------------------------
# 2. MX row quantization kernel vs the torch recipe (bit-exact)
# ------------------------------------------------------------------------------------------
@benchmark()
def test_quant_rows(rows, k, fmt):
    torch.manual_seed(0)
    # values spanning 8 decades, an all-zero block and a tiny-scale block
    x = (torch.randn(rows, k) * torch.logspace(-4, 4, k)[torch.randperm(k)]).to(dtypes.bf16)
    x[min(5, rows - 1), :32] = 0
    x[min(6, rows - 1), 64:96] = 1e-30
    if fmt == "e2m3":
        ref = ref_quant_dequant_e2m3(x).to(dtypes.bf16).float()
        mine = q.dequant_mx(*q.quant_mx(x, fmt), fmt).to(dtypes.bf16).float()
        err0 = checkAllclose(ref, mine, rtol=0, atol=0, tol_err_ratio=0.0, msg="quant_mx vs recipe: ")
        assert err0 == 0
    c, s = q.quant_mx(x, fmt)
    ref_codes, ref_scales = q.pack_rows(c, fmt), q.pack_a_scales(s)
    (codes, scales), us = run_perftest(ak.quant_rows, x, fmt)
    ks = k // 128
    err = checkAllclose(ref_codes.int(), codes.int(), rtol=0, atol=0, tol_err_ratio=0.0, msg=f"codes {fmt}: ")
    err_s = checkAllclose(q.unpack_a_scales(ref_scales, ks).int(), q.unpack_a_scales(scales, ks).int(),
                          rtol=0, atol=0, tol_err_ratio=0.0, msg=f"scales {fmt}: ")
    assert err == 0 and err_s == 0
    nbytes = rows * k * 2 + codes.numel() + scales.numel() * 4
    return {"gfx": get_gfx_runtime(), "us": us, "TB/s": nbytes / us / 1e6, "err": max(err, err_s)}


# ------------------------------------------------------------------------------------------
# 3. routers
# ------------------------------------------------------------------------------------------
@benchmark()
def test_route(router, tokens, topk, experts, bm):
    g = torch.Generator(device="cuda").manual_seed(0)
    p = torch.softmax(torch.randn(experts, generator=g, device="cuda") * 2, 0)  # skewed expert load
    ids = torch.multinomial(p.expand(tokens, experts), topk, replacement=False, generator=g).int()
    (ridx, pos, texp, m), us = run_perftest(getattr(ak, router), ids, experts, bm)
    nt = texp.numel() - 1
    ntile = (torch.bincount(ids.flatten().long(), minlength=experts) + bm - 1) // bm
    tstart = torch.cumsum(ntile, 0) - ntile
    exp_texp = torch.full((nt,), -1, dtype=torch.int32)
    valid = torch.zeros(m, dtype=torch.bool)
    for e in range(experts):
        exp_texp[int(tstart[e]):int(tstart[e] + ntile[e])] = e
        valid[int(tstart[e]) * bm:int(tstart[e] + ntile[e]) * bm] = True
    used = torch.zeros(m, dtype=torch.bool)
    used[pos.flatten().long()] = True
    checks = {
        "used tile count": int(texp[nt]) == int(ntile.sum()),
        "tile -> expert": torch.equal(texp[:nt], exp_texp),
        "row -> token": torch.equal(ridx[pos.long()], torch.arange(tokens)[:, None].expand(tokens, topk).int()),
        "slot in its expert's tiles": torch.equal(texp[(pos // bm).long()], ids),
        "slots unique": pos.flatten().unique().numel() == tokens * topk,
        "pad slots -> row 0": bool((ridx[valid & ~used] == 0).all()),
    }
    bad = [k for k, v in checks.items() if not v]
    assert not bad, f"{router}: {bad}"
    return {"gfx": get_gfx_runtime(), "us": us, "err": float(len(bad))}


# ------------------------------------------------------------------------------------------
# 4. whole op
# ------------------------------------------------------------------------------------------
def _module(wfmt, afmt):
    os.environ["MX6_WFMT"], os.environ["MX6_AFMT"] = wfmt, afmt
    from aiter.ops.flydsl import mxfp6_moe

    return importlib.reload(mxfp6_moe)


def _moe_inputs(tokens, hidden, inter, experts, topk):
    g = torch.Generator(device="cuda").manual_seed(0)
    layer = {"w1": (torch.randn(experts, 2 * inter, hidden, generator=g, device="cuda") / math.sqrt(hidden)).to(dtypes.bf16),
             "w2": (torch.randn(experts, hidden, inter, generator=g, device="cuda") / math.sqrt(inter)).to(dtypes.bf16)}
    x = torch.randn(tokens, hidden, generator=g, device="cuda").to(dtypes.bf16)
    # random distinct experts per token (uneven per-expert load) and distinct routing weights
    ids = torch.rand(tokens, experts, generator=g, device="cuda").argsort(-1)[:, :topk].int()
    w = torch.rand(tokens, topk, generator=g, device="cuda") + 0.5
    return layer, x, ids, (w / w.sum(-1, keepdim=True)).float()


@benchmark()
def test_mxfp6_moe(tokens, hidden, inter, experts, topk, wfmt, afmt):
    m = _module(wfmt, afmt)
    layer, x, ids, w = _moe_inputs(tokens, hidden, inter, experts, topk)
    st = m.run(None, layer)
    inp = {"x": x, "ids": ids, "w": w}
    out, us = run_perftest(m.run, inp, st, num_iters=20, num_warmup=3)
    det = all(torch.equal(out, m.run(inp, st)) for _ in range(3))
    assert det, "not bitwise deterministic run to run"
    ref = run_torch_mx(x, ids, w, layer["w1"], layer["w2"], wfmt, afmt)
    err = checkAllclose(ref.to(dtypes.fp32), out.to(dtypes.fp32), rtol=2e-2, atol=2e-2, tol_err_ratio=1e-3,
                        msg=f"mxfp6_moe {wfmt}/{afmt}: ")
    assert err < 1e-3, f"{err:.3%} of elements outside rtol/atol vs the same-numerics reference"
    bf = run_torch_bf16(x, ids, w, layer["w1"], layer["w2"]).float()
    flops = 6 * tokens * topk * hidden * inter
    abits = 8 if afmt == "e4m3" else 6
    nbytes = (experts * 3 * hidden * inter * 6 // 8 + tokens * hidden * 2 * 2
              + tokens * topk * (inter * abits // 8 + hidden * 2) * 2)
    rel = ((out.float() - bf).norm() / bf.norm()).item()
    del layer, st, out, ref, bf
    torch.cuda.empty_cache()
    return {"gfx": get_gfx_runtime(), "flydsl us": us, "flydsl TFLOPS": flops / us / 1e6,
            "flydsl TB/s": nbytes / us / 1e6, "flydsl err": err, "deterministic": det, "rel_l2_vs_bf16": rel}


# ------------------------------------------------------------------------------------------
# 5. mxfp6_moe_gemm's own choices: the AGPR form (compiled, no GPU), the h layout contract (host only, no GPU),
#    rarely selected paths and call modes (GPU)
# ------------------------------------------------------------------------------------------
def _stage_calls(hidden, inter, experts, topk, tokens, afmt, wfmt="e2m3"):
    """run_gg6e's arguments for both stages as mxfp6_moe.forward passes them, on empty buffers of the op's shapes
    (FakeTensors inside aiter's dry run) -> (args, kwargs) of stage 1 and of stage 2, and the tile rows."""
    m = _module(wfmt, afmt)
    bm = m.pick_bm(tokens * topk / experts)
    nt = (tokens * topk + bm - 1) // bm + experts
    M, bits = nt * bm, q.FORMATS[afmt][3]
    bn1, bn2 = m.stage_bn(2 * inter, afmt), m.stage_bn(hidden, afmt)

    def e(*shape, d=torch.uint8):
        return torch.empty(*shape, dtype=d, device="cuda")
    ridx, texp = e(M, d=torch.int32), e(nt + 1, d=torch.int32)
    ha, hs = e(M, inter * bits // 8), e(M, q.kq(inter), 4, d=torch.int32)
    s1 = ((e(tokens, hidden * bits // 8), e(tokens, q.kq(hidden), 4, d=torch.int32), ridx, texp,
           e(experts, 2 * inter // bn1, hidden // 128, bn1 // 16, 1536), e(experts, 2 * inter // 16, q.kq(hidden), 64, d=torch.int32), ha),
          {"N": 2 * inter, "K": hidden, "a_fmt": afmt, "b_fmt": wfmt, "epi": "swiglu_q", "gather": True, "gm": 8,
           "cfg": m.gemm_cfg(2 * inter, afmt, bm), "out_scales": hs, "h_layout": "tile_step"})
    s2 = ((ha, hs, ridx, texp, e(experts, hidden // bn2, inter // 128, bn2 // 16, 1536),
           e(experts, hidden // 16, q.kq(inter), 64, d=torch.int32), e(M, hidden, d=dtypes.bf16)),
          {"N": hidden, "K": inter, "a_fmt": afmt, "b_fmt": wfmt, "epi": "store", "gather": False, "gm": 2,
           "cfg": m.gemm_cfg(hidden, afmt, bm), "h_layout": "tile_step"})
    return s1, s2, bm


def _k_loop(lines):
    """The K loop of a kernel's final ISA: the instructions of its longest backward branch."""
    labels = {m.group(1): i for i, line in enumerate(lines) if (m := re.match(r"^(\.LBB\w+):", line))}
    best = None
    for j, line in enumerate(lines):
        m = re.match(r"\s+s_c?branch\w*\s+(\.LBB\w+)", line)
        if m and m.group(1) in labels and labels[m.group(1)] < j and (best is None or j - labels[m.group(1)] > best[1] - best[0]):
            best = (labels[m.group(1)], j)
    return lines[best[0]:best[1] + 1] if best else []


@benchmark()
def test_agpr_form(hidden, inter, experts, topk, tokens, afmt):
    # compile only: no torch reference, no timing. FlyDSL reads these at each compile; a cached kernel is not dumped
    dump = tempfile.mkdtemp(prefix="gg6e_agpr.")
    env = {"FLYDSL_DEBUG_DUMP_ASM": "1", "FLYDSL_RUNTIME_ENABLE_CACHE": "0", "FLYDSL_DUMP_DIR": dump}
    gemm.compile_gg6e.cache_clear()
    try:
        with mock.patch.dict(os.environ, env), dry_run(SUPPORTED_GFX[0], 256):
            s1, s2, _ = _stage_calls(hidden, inter, experts, topk, tokens, afmt)
            gemm.run_gg6e(*s1[0], **s1[1])
            gemm.run_gg6e(*s2[0], **s2[1])
        kernels = sorted(Path(dump).glob("gg6e_*_f8x8_*/*_final_isa.s"))
        copies, scratch, mfma = 0, 0, 0
        for f in kernels:
            text = f.read_text()
            body = [line.split(";")[0] for line in _k_loop(text.splitlines())]
            mfma += sum(1 for line in body if re.match(r"\s+v_mfma", line))
            copies += sum(1 for line in body if re.match(r"\s+v_accvgpr_", line))
            scratch = max(scratch, int(re.search(r"\.private_segment_fixed_size:\s+(\d+)", text).group(1)))
    finally:
        gemm.compile_gg6e.cache_clear()  # launchers built in the dry run must not serve a later run on the GPU
        shutil.rmtree(dump, ignore_errors=True)
    ok = bool(kernels) and copies == 0 and scratch == 0 and mfma > 0
    assert ok, (f"f8x8 kernels compiled to the slow form: {copies} v_accvgpr_* in their K loops, {scratch} B scratch "
                f"({len(kernels)} kernels, {mfma} K-loop MFMAs): the AGPR pin or the amdgpu-mfma-vgpr-form option has "
                "no effect in this FlyDSL / LLVM")
    return {"gfx": SUPPORTED_GFX[0], "f8x8 kernels": len(kernels), "K-loop MFMA": mfma, "K-loop v_accvgpr_*": copies,
            "max scratch B": scratch, "fast form": ok}


def _spy_gg6e(picks, real=None):
    """compile_gg6e replaced by a recorder of the h layout each call picks (1: tile-step-major); real: also compile."""
    def spy(*a, **kw):
        picks.append(int(kw.get("H_TILE_STEP", 0)))
        return real(*a, **kw) if real else None
    return mock.patch.object(gemm, "compile_gg6e", spy)


@benchmark()
def test_h_layout(hidden, inter, experts, topk, tokens, afmt):
    # host code only: compile_gg6e is recorded and nothing is compiled or launched, so no reference and no timing
    six = int(q.FORMATS[afmt][3] == 6)
    picks, got = [], {}

    def stages(**kw):  # forward's two calls on fresh buffers (kw replaces its keywords) -> the h layout each picked
        s1, s2, _ = _stage_calls(hidden, inter, experts, topk, tokens, afmt)
        del picks[:]
        gemm.run_gg6e(*s1[0], **{**s1[1], **kw})
        gemm.run_gg6e(*s2[0], **{**s2[1], **kw})
        return tuple(picks)

    def refused(*args, **kw):
        try:
            gemm.run_gg6e(*args, **kw)
        except ValueError:
            return True
        return False

    with dry_run(SUPPORTED_GFX[0], 256), _spy_gg6e(picks), mock.patch.object(gemm, "_run_compiled", lambda *a, **kw: None), \
            mock.patch.object(gemm, "compile_xpad", lambda *a, **kw: None, create=True):
        # 6-bit: as forward calls them, stage 1 writes the h tile-step-major and stage 2 reads it so; with no keyword
        # both use rows, as quant_rows writes them; 8-bit: never tile-step
        got["forward tile_step"] = stages() == (six, six)
        got["default rows"] = stages(h_layout=None) == (0, 0)
        got["h_layout rows"] = stages(h_layout="rows") == (0, 0)
        with mock.patch.object(gemm, "H_LAYOUT", "tile_step"):
            got["H_LAYOUT tile_step"] = stages(h_layout=None) == (six, six)
        with torch.inference_mode():  # tensors without a version counter, as a serving engine makes them
            got["inference_mode"] = stages() == (six, six)
        (a2, kw2), bm = _stage_calls(hidden, inter, experts, topk, tokens, afmt)[1:]
        del picks[:]
        for name, kw in (("epi", {"epi": "silu"}), ("h_layout", {"h_layout": "row"}), ("h_layout auto", {"h_layout": "auto"}),
                         ("ablate", {"ablate": "bar"})):
            got[f"bad {name} refused"] = refused(*a2, **{**kw2, **kw})
        got["short out refused"] = refused(*a2[:6], a2[6][bm:], **kw2)
        got["no kernel picked by a refused call"] = not picks
    bad = [k for k, v in got.items() if not v]
    assert not bad, f"run_gg6e's h layout contract, {afmt}: {bad}"
    return {"gfx": SUPPORTED_GFX[0], "checks": len(got), "err": float(len(bad))}


# what a whole-op case must reach, from compile_gg6e's positional arguments (N, K, a_fmt, b_fmt, epi, gather, FM, FN,
# WGM, WGN) of the two stage calls, the op's tile rows bm, and whether the row-padding copy kernel was built
REACHES = {
    "padded_rows": lambda calls, bm, padded: padded,
    "half_tiles": lambda calls, bm, padded: any(a[4] == "swiglu_q" and 16 * a[6] * a[8] < bm for a in calls),
    "hidden128": lambda calls, bm, padded: any(a[4] == "swiglu_q" and a[1] == 128 for a in calls),
}


@benchmark()
def test_moe_paths(hidden, inter, experts, topk, tokens, afmt, wfmt, reach):
    calls, padded = [], []
    real_gg6e, real_xpad = gemm.compile_gg6e, getattr(gemm, "compile_xpad", None)

    def spy_gg6e(*a, **kw):
        calls.append(a)
        return real_gg6e(*a, **kw)

    def spy_xpad(*a, **kw):
        padded.append(a)
        return real_xpad(*a, **kw)

    with mock.patch.object(gemm, "compile_gg6e", spy_gg6e), mock.patch.object(gemm, "compile_xpad", spy_xpad, create=True):
        ret = test_mxfp6_moe(tokens, hidden, inter, experts, topk, wfmt, afmt)
    ok = REACHES[reach](calls, _module(wfmt, afmt).pick_bm(tokens * topk / experts), bool(padded))
    assert ok, f"{hidden}x{inter} E{experts} top-{topk} T{tokens} {afmt} no longer reaches {reach}: pick another shape"
    ret.update({"K tail s1/s2": f"{hidden // 128 % 4}/{inter // 128 % 4}", "reached": ok})
    return ret


@benchmark()
def test_moe_modes(tokens, hidden, inter, experts, topk, wfmt, afmt):
    m = _module(wfmt, afmt)
    layer, x, ids, w = _moe_inputs(tokens, hidden, inter, experts, topk)
    st = m.run(None, layer)
    inp = {"x": x, "ids": ids, "w": w}
    ref = run_torch_mx(x, ids, w, layer["w1"], layer["w2"], wfmt, afmt)
    six, picks = int(q.FORMATS[afmt][3] == 6), []

    def layout(h_layout):  # both of forward's stage calls with this h_layout
        def run():
            with mock.patch.object(m, "run_gg6e", lambda *a, **kw: gemm.run_gg6e(*a, **{**kw, "h_layout": h_layout})):
                return m.run(inp, st)
        return run

    def inference_mode():  # as a serving engine runs the model: no tensor made here has a version counter
        with torch.inference_mode():
            return m.run({"x": x.clone(), "ids": ids, "w": w}, st)

    def no_grad():
        with torch.no_grad():
            return m.run(inp, st)

    candidates = {"default": lambda: m.run(inp, st), "rows": layout("rows"), "tile_step": layout("tile_step"),
                  "inference_mode": inference_mode, "no_grad": no_grad,
                  "requires_grad": lambda: m.run({"x": x.clone().requires_grad_(), "ids": ids, "w": w}, st)}
    flops = 6 * tokens * topk * hidden * inter
    nbytes = (experts * 3 * hidden * inter * 6 // 8 + tokens * hidden * 2 * 2
              + tokens * topk * (inter * (6 if six else 8) // 8 + hidden * 2) * 2)
    ret, outs = {"gfx": get_gfx_runtime()}, {}
    for name, fn in candidates.items():
        del picks[:]
        with _spy_gg6e(picks, gemm.compile_gg6e):
            out, us = run_perftest(fn, num_iters=5, num_warmup=1)
        # stage 2 must read the layout stage 1 wrote; "rows" / "tile_step" must run the kernels of that layout
        assert len(set(picks)) == 1 and picks[0] == {"rows": 0, "tile_step": six}.get(name, picks[0]),             f"{name}: stage calls picked h layouts {picks}"
        outs[name] = out.detach()
        err = checkAllclose(ref.to(dtypes.fp32), outs[name].to(dtypes.fp32), rtol=2e-2, atol=2e-2, tol_err_ratio=1e-3,
                            msg=f"mxfp6_moe {wfmt}/{afmt} {name}: ")
        assert err < 1e-3, f"{name}: {err:.3%} of elements outside rtol/atol vs the same-numerics reference"
        assert torch.equal(outs[name], outs["default"]), f"{name}: output bits differ from the default call's"
        ret.update({f"{name} us": us, f"{name} TFLOPS": flops / us / 1e6, f"{name} TB/s": nbytes / us / 1e6, f"{name} err": err})
    st.clear()  # the candidates' closures hold it
    del layer, ref, outs
    torch.cuda.empty_cache()
    return ret


def summarize(name, rows):
    aiter.logger.info("%s summary (markdown):\n%s", name, pd.DataFrame(rows).to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="FlyDSL MXFP6 grouped MoE")
    parser.add_argument("-s", "--shape", type=dtypes.str2tuple, nargs="*",
                        default=[(2048, 1024, 8, 4), (6272, 2560, 64, 4), (2048, 1024, 160, 4), (1152, 640, 16, 8),
                                 (6144, 2048, 64, 4)],
                        help="hidden,inter,experts,topk (hidden 6272: N %% 256 != 0 tiles; E=160 at 4096 tokens:\n"
                             "~100 rows/expert -> 128-row tiles)")
    parser.add_argument("-t", "--tokens", type=int, nargs="*", default=[300, 4096])
    parser.add_argument("--wfmt", nargs="*", default=["e2m3", "e3m2"], help="weight format(s)")
    parser.add_argument("--afmt", nargs="*", default=["e2m3", "e4m3"], help="activation format(s)")
    parser.add_argument("--mfma", nargs="*", default=["e2m3,e2m3", "e3m2,e3m2", "e4m3,e2m3", "e2m3,e3m2",
                                                      "e4m3,e4m3", "e2m1,e2m1", "e4m3,e3m2", "e2m3,e4m3"],
                        help="A,B format pairs for the MFMA contract")
    parser.add_argument("--quant_k", type=int, nargs="*", default=[640, 2048, 10496])
    parser.add_argument("--route", type=dtypes.str2tuple, nargs="*",
                        default=[(4096, 4, 64, 256), (300, 8, 160, 256), (32768, 8, 64, 256), (5000, 4, 192, 128)],
                        help="tokens,topk,experts,bm")
    parser.add_argument("--agpr", type=dtypes.str2tuple, nargs="*", default=[(8192, 4608, 64, 4, 4096), (6144, 2048, 64, 4, 4096)],
                        help="hidden,inter,experts,topk,tokens for the AGPR form and the h layout contract (no GPU)")
    parser.add_argument("--paths", nargs="*", default=["1408,3072,8,2,4096,e2m3,padded_rows", "10496,1024,8,2,1024,e4m3,half_tiles",
                                                       "128,768,8,2,4096,e2m3,hidden128"],
                        help=f"hidden,inter,experts,topk,tokens,afmt,reach with reach one of {', '.join(REACHES)}\n"
                             "(defaults: K tails of 3 / 2 steps in stage 1, 2 in stage 2)")
    parser.add_argument("--modes", type=dtypes.str2tuple, nargs="*", default=[(2048, 1024, 8, 4, 4096), (1152, 640, 16, 8, 300)],
                        help="hidden,inter,experts,topk,tokens for the h layouts and autograd modes (256-row, 128-row tiles)")
    args = parser.parse_args()

    # no GPU needed: aiter's dry run
    summarize("agpr form", [test_agpr_form(*c, af) for c, af in itertools.product(args.agpr, args.afmt)])
    summarize("h layout", [test_h_layout(*c, af) for c, af in itertools.product(args.agpr + args.modes, args.afmt)])
    gfx = get_gfx_runtime() if torch.cuda.is_available() else "no GPU"
    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning("mxfp6_moe unsupported on %s; skipping the GPU tests", gfx)
        return
    torch.set_default_device("cuda")

    summarize("mfma contract", [test_mfma_contract(*p.split(",")) for p in args.mfma])
    summarize("quant_rows", [test_quant_rows(77, k, f)
                             for k, f in itertools.product(args.quant_k, ["e2m3", "e3m2", "e4m3"])])
    summarize("route", [test_route(r, *c) for r, c in itertools.product(["route", "route_mb"], args.route)])
    summarize("mxfp6_moe", [test_mxfp6_moe(t, h, i, e, k, wf, af) for (h, i, e, k), t, wf, af
                            in itertools.product(args.shape, args.tokens, args.wfmt, args.afmt)])
    paths = [c.split(",") for c in args.paths]
    summarize("mxfp6_moe paths", [test_moe_paths(*map(int, c[:5]), c[5], args.wfmt[0], c[6]) for c in paths])
    summarize("mxfp6_moe modes", [test_moe_modes(t, h, i, e, k, args.wfmt[0], af) for (h, i, e, k, t), af
                                  in itertools.product(args.modes, args.afmt)])


if __name__ == "__main__":
    main()
