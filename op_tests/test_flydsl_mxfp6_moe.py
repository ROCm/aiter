# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL MXFP6 grouped MoE (aiter.ops.flydsl.mxfp6_moe), gfx950.

Correctness against torch references + perf, one markdown table per test function:
  test_mfma_contract  scaled-MFMA operand contract per format pair (packing, K order, E8M0 lanes)
  test_quant_rows     FlyDSL MX row quantization == torch recipe, bit-exact codes and scales
  test_route          one-block and multi-block routers: placement invariants
  test_mxfp6_moe      whole op vs a torch reference with identical MX numerics, bitwise
                      run-to-run determinism, rel L2 vs bf16

    python3 op_tests/test_flydsl_mxfp6_moe.py
    python3 op_tests/test_flydsl_mxfp6_moe.py -s 8192,4608,64,4 -t 4096 32768 --wfmt e2m3 --afmt e2m3 e4m3
"""

import argparse
import importlib
import itertools
import math
import os

import pandas as pd
import torch

import aiter
import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import mxfp6_moe_quant as q
from aiter.ops.flydsl.kernels import mxfp6_moe_aux as ak
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


@benchmark()
def test_mxfp6_moe(tokens, hidden, inter, experts, topk, wfmt, afmt):
    m = _module(wfmt, afmt)
    g = torch.Generator(device="cuda").manual_seed(0)
    layer = {"w1": (torch.randn(experts, 2 * inter, hidden, generator=g, device="cuda") / math.sqrt(hidden)).to(dtypes.bf16),
             "w2": (torch.randn(experts, hidden, inter, generator=g, device="cuda") / math.sqrt(inter)).to(dtypes.bf16)}
    x = torch.randn(tokens, hidden, generator=g, device="cuda").to(dtypes.bf16)
    # random distinct experts per token (uneven per-expert load) and distinct routing weights
    ids = torch.rand(tokens, experts, generator=g, device="cuda").argsort(-1)[:, :topk].int()
    w = torch.rand(tokens, topk, generator=g, device="cuda") + 0.5
    w = (w / w.sum(-1, keepdim=True)).float()
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


def summarize(name, rows):
    aiter.logger.info("%s summary (markdown):\n%s", name, pd.DataFrame(rows).to_markdown(index=False))


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("mxfp6_moe unsupported on %s; skipping", get_gfx_runtime())
        return
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
    args = parser.parse_args()
    torch.set_default_device("cuda")

    summarize("mfma contract", [test_mfma_contract(*p.split(",")) for p in args.mfma])
    summarize("quant_rows", [test_quant_rows(77, k, f)
                             for k, f in itertools.product(args.quant_k, ["e2m3", "e3m2", "e4m3"])])
    summarize("route", [test_route(r, *c) for r, c in itertools.product(["route", "route_mb"], args.route)])
    summarize("mxfp6_moe", [test_mxfp6_moe(t, h, i, e, k, wf, af) for (h, i, e, k), t, wf, af
                            in itertools.product(args.shape, args.tokens, args.wfmt, args.afmt)])


if __name__ == "__main__":
    main()
