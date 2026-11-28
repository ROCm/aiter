# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bytes and timing of the MiniMax-M3.1 TP1 norm kernels (aiter.ops.m3_tp1_prefill).

norm_mx's bytes are compared with the unfused aiter chain on the same inputs:
moe_reduction, ``x * 2 + s`` in torch, add+RMSNorm with its MXFP8 (e8m0) epilogue
(fused_qk_rmsnorm_group_quant), per_1x32_f4_quant_hip and shuffle_scale (w4; pad
rows stay unwritten). mx8_quant's follow the rule in torch (``mx8_ref``). Outputs
start as garbage. ``subnormal`` puts bf16-subnormal values into the route rows and
``s_add``; its reference then flushes fp32 denormals as aiter's build does.

    python3 op_tests/test_m3_tp1_norm.py [--quick] [--only norm mx8 args perf]
"""

import argparse
import sys
from itertools import product

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import m3_tp1_prefill as K
from aiter.ops.flydsl.moe_kernels import _run_moe_reduction
from aiter.ops.quant import per_1x32_f4_quant_hip
from aiter.ops.shuffle import shuffle_scale
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

H, EPS = 6144, 1e-6  # MiniMax-M3.1 hidden size
NAMES = ["INPUT_MX8", "FOLD_MX8", "RED4_FOLD_MX8", "POST_BF16_MX8"]
NAMES += ["POST_BF16_MX8_MX4", "FOLD_H", "RED4_FOLD_H"]
ADD3, OUT_BF16, MX8, MX4, RED4, HOUT = 1, 2, 4, 8, 16, 32


def gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def acts(m, n, seed, mode="randn"):
    """bf16 [m, n]: randn, massive-activation channels, or 10 decades across
    the 32-groups with some all-zero groups."""
    g = gen(seed)
    x = torch.randn(m, n, generator=g)
    if mode == "massive":
        x[:, ::97] *= 400.0
    elif mode == "wide":
        e = torch.exp(torch.randn(m, n // 32, 1, generator=g) * 4)
        x = (x.view(m, n // 32, 32) * e).view(m, n)
        x[::7, :64] = 0
    return x.to(torch.bfloat16)


def subnormals(shape, g):
    mag = torch.randint(1, 128, shape, generator=g).float() * 2.0**-133
    return (mag * (torch.randint(0, 2, shape, generator=g) * 2 - 1)).to(torch.bfloat16)


def mx8_ref(x):
    """The MXFP8 rule on bf16 [rows, k]: e4m3 bytes and e8m0 codes [rows, k / 32].
    amax skips NaN; the clamp is fmax / fmin, so a NaN quantizes to -448."""
    rows, k = x.shape
    xf = x.float().view(rows, k // 32, 32)
    a = xf.abs()
    amax = torch.where(torch.isnan(a), 0.0, a).amax(-1).clamp_min(1e-10)
    bits = (amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)).view(torch.int32)
    code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    recip = ((254 - code).clamp_min(0) << 23).view(torch.float32)
    lim = torch.tensor(448.0)
    y = torch.fmin(torch.fmax(xf * recip[..., None], -lim), lim)
    return y.to(torch.float8_e4m3fn).view(torch.uint8).view(rows, k), code.to(
        torch.uint8
    )


def w4_ref(codes):
    """aiter's shuffle_scale of row-major codes; the pad rows up to 256 keep the
    outputs' 0xA5 fill, as the kernels do not write them."""
    m, g = codes.shape
    pad = torch.full(((m + 255) // 256 * 256, g), 0xA5, dtype=torch.uint8)
    pad[:m] = codes.view(torch.uint8)
    return shuffle_scale(pad)


def ndiff(a, b):
    return int(
        (a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum()
    )


def add_rmsnorm_ref(h, res, w, mx8=False):
    """aiter's Gemma add+RMSNorm: normed bf16, res_out = h + res and, with mx8, its
    MXFP8 epilogue (e4m3, e8m0 codes per 32)."""
    out, res_out = torch.empty_like(h), torch.empty_like(h)
    q8 = torch.empty(h.shape, dtype=dtypes.fp8) if mx8 else None
    s8 = torch.empty(h.shape[0], H // 32, dtype=torch.uint8) if mx8 else None
    aiter.fused_qk_rmsnorm_group_quant(
        q_out_quantized=q8, q_out_scale=s8, q=h, q_weight=w, q_epsilon=EPS,
        q_out_unquantized=out, q_res_out=res_out, q_residual=res, group_size=32,
        gemma_norm=True,
    )  # fmt: skip
    return out, res_out, q8, s8


def fold_ref(flags, x, s, flush):
    """The norm input: aiter moe_reduction of the route rows (RED4), then
    ``r * 2 + s`` (ADD3); ``flush`` emulates the fp32 denormal flush."""
    if not flush:
        r = x
        if flags & RED4:
            r = torch.empty(x.shape[0], H, dtype=torch.bfloat16)
            _run_moe_reduction(x, r, x.shape[0], 4, H)
        return r * 2.0 + s if flags & ADD3 else r

    def ftz(t):
        return torch.where(t.abs() < 2.0**-126, torch.zeros_like(t).copysign(t), t)

    acc = torch.zeros(x.shape[0], H)
    for k in range(4):
        acc = ftz(acc + ftz(x[:, k].float()))
    h = ftz(acc.to(torch.bfloat16).float() * 2.0) + ftz(s.float())
    return ftz(h).to(torch.bfloat16)


def make_case(flags, m, mode, w4, seed=0):
    """Inputs, the kernel's argument list (outputs pre-filled with garbage) and
    the reference outputs."""
    g = gen(seed + 1)
    am = "randn" if mode == "subnormal" else mode
    x = acts(m, H, seed, am)
    if flags & RED4:
        x = (acts(4 * m, H, seed + 11, am).float() * 0.25).to(torch.bfloat16)
        x = x.view(m, 4, H)
    s = acts(m, H, seed + 7, am) if flags & ADD3 else None
    res = acts(m, H, seed + 100, am)
    w = (torch.randn(H, generator=g) * 0.5).to(torch.bfloat16)
    if mode == "subnormal":
        pick = torch.rand(x.shape, generator=g) < 0.3
        x = torch.where(pick, subnormals(x.shape, g), x)
        x[: m // 4] = subnormals((m // 4, 4, H), g)  # four subnormal route values
        pick = torch.rand(s.shape, generator=g) < 0.3
        s = torch.where(pick, subnormals(s.shape, g), s)
    rows = (m + 255) // 256 * 256 if w4 else m
    out = {
        "res_out": torch.full((m, H), float("nan"), dtype=torch.bfloat16),
        "out_bf16": torch.full((m, H), float("nan"), dtype=torch.bfloat16),
        "q8": torch.full((m, H), 0x5A, dtype=torch.uint8).view(dtypes.fp8),
        "s8": torch.full((rows, H // 32), 0xA5, dtype=torch.uint8),
        "q4": torch.full((m, H // 2), 0x5A, dtype=torch.uint8),
        "s4": torch.full((m, H // 32), 0xA5, dtype=torch.uint8),
    }
    hout = bool(flags & HOUT)
    args = [flags, x, s, None if hout else res, None if hout else w]
    args += [None if hout else out["res_out"]]
    args += [out["out_bf16"] if flags & (OUT_BF16 | HOUT) else None]
    args += [out[k] if flags & MX8 else None for k in ("q8", "s8")]
    args += [out[k] if flags & MX4 else None for k in ("q4", "s4")] + [EPS, m, H, w4]
    h = fold_ref(flags, x, s, mode == "subnormal")
    if hout:
        return args, out, {"out_bf16": h}, h
    normed, res_out, q8, s8 = add_rmsnorm_ref(h, res, w, mx8=True)
    ref = {"res_out": res_out, "q8": q8, "s8": w4_ref(s8) if w4 else s8}
    if flags & OUT_BF16:
        ref["out_bf16"] = normed
    if flags & MX4:
        q4, s4 = per_1x32_f4_quant_hip(normed, shuffle=False)
        ref["q4"], ref["s4"] = q4.view(torch.uint8), s4.view(torch.uint8)
    return args, out, ref, h


@benchmark()
def test_norm(name, m, mode, w4):
    flags = getattr(K, "NORM_" + name)
    args, out, ref, h = make_case(flags, m, mode, w4)
    K.norm_mx(*args)
    d = {k: ndiff(out[k], r) for k, r in ref.items()}
    assert all(v == 0 for v in d.values()), f"{name} m={m} {mode} w4={w4}: {d}"
    ret = {"diff bytes": sum(d.values()), "outputs": ",".join(d)}
    if mode == "subnormal":
        ieee = ndiff(h, fold_ref(flags, args[1], args[2], False))
        assert ieee > 0, "the case no longer reaches the flushed values"
        ret["norm input values the IEEE chain changes"] = ieee
    return ret


def mx8_inputs(m, k, mode, seed=0):
    if mode != "special":
        return acts(m, k, seed, mode)
    x = acts(m, k, seed)
    x[0::5, 3] = float("nan")  # a NaN in a group of finite values
    x[1::5, 40], x[2::5, 70] = float("inf"), float("-inf")
    x[3::5] = subnormals((k,), gen(seed)).abs()  # all-subnormal rows
    x[4::5, 96:128] = -subnormals((32,), gen(seed)).abs()  # one subnormal group
    return x


@benchmark()
def test_mx8_quant(m, k, mode, w4):
    x = mx8_inputs(m, k, mode)
    q_ref, s_ref = mx8_ref(x)
    q = torch.full((m, k), 0x5A, dtype=torch.uint8).view(dtypes.fp8)
    rows = (m + 255) // 256 * 256 if w4 else m
    s = torch.full((rows, k // 32), 0xA5, dtype=torch.uint8)
    K.mx8_quant(x, q, s, w4)
    d = (ndiff(q, q_ref), ndiff(s, w4_ref(s_ref) if w4 else s_ref))
    assert d == (0, 0), f"mx8_quant m={m} k={k} {mode} w4={w4}: {d}"
    return {"diff bytes": sum(d)}


def expect_error(fn, args, match):
    try:
        fn(*args)
        torch.cuda.synchronize()
    except RuntimeError as e:
        assert match in str(e), f"unexpected error: {e}"
        return
    raise AssertionError(f"{fn.__name__} accepted invalid arguments ({match})")


def test_arg_checks():
    args = make_case(K.NORM_FOLD_MX8, 64, "randn", False)[0]
    expect_error(K.norm_mx, [ADD3 | OUT_BF16] + args[1:], "unsupported flags")
    expect_error(K.norm_mx, args[:2] + [None] + args[3:], "s_add must be")
    expect_error(K.norm_mx, args[:-2] + [H // 2, False], "n must be in")
    expect_error(K.norm_mx, args[:-1] + [True], "s8 must be")  # unpadded rows
    x, q = args[1], args[7]
    expect_error(K.mx8_quant, [x[:, :48], q, q, False], "contiguous")
    small = torch.empty(64, H // 64, dtype=torch.uint8)
    expect_error(K.mx8_quant, [x, q, small, False], "s must be")
    aiter.logger.info("m3_tp1 norm argument checks passed")


@benchmark()
def perf_norm(name, m):
    """The fused kernel, and aiter's add+RMSNorm alone (bf16 out) on the same rows."""
    flags = getattr(K, "NORM_" + name)
    args = make_case(flags, m, "randn", bool(flags & MX8))[0]
    ret = {"us": run_perftest(K.norm_mx, *args, num_iters=50, num_warmup=5)[1]}
    if not flags & HOUT:
        h = args[1].sum(1).to(torch.bfloat16) if flags & RED4 else args[1]
        ret["add+RMSNorm alone us"] = run_perftest(add_rmsnorm_ref, h, *args[3:5])[1]
    return ret


@benchmark()
def perf_mx8_quant(m, k):
    x, q = acts(m, k, 0), torch.empty((m, k), dtype=dtypes.fp8)
    s = torch.empty(((m + 255) // 256 * 256, k // 32), dtype=torch.uint8)
    us = run_perftest(K.mx8_quant, x, q, s, True, num_iters=50, num_warmup=5)[1]
    return {"us": us, "TB/s": m * k * (2 + 1 + 1 / 32) / us / 1e6}


parser = argparse.ArgumentParser(description="MiniMax-M3.1 TP1 norm kernels")
parser.add_argument("--only", nargs="*", choices=["norm", "mx8", "args", "perf"])
parser.add_argument("--quick", action="store_true", help="fewer and smaller cases")
cli = parser.parse_args()
if get_gfx_runtime() != "gfx950":
    aiter.logger.info("m3_tp1 norm kernels target gfx950; skipping")
    sys.exit(0)
part = set(cli.only or ["norm", "mx8", "args", "perf"])
ms = [1, 37, 3001, 8192] if cli.quick else [1, 37, 1000, 3001, 8192, 16384]
modes = ["randn"] if cli.quick else ["randn", "massive", "wide"]
tables = {}
if "args" in part:
    test_arg_checks()
if "norm" in part:
    rows = []
    for name in NAMES:
        w4s = (False, True) if getattr(K, "NORM_" + name) & MX8 else (False,)
        rows += [test_norm(name, m, md, w4) for w4 in w4s for m in ms for md in modes]
        if "RED4" in name:  # fp32 denormal flush of the route rows and s_add
            rows += [test_norm(name, m, "subnormal", False) for m in (37, 3001)]
    tables["norm_mx"] = pd.DataFrame(rows)
if "mx8" in part:
    mx8_ms = [1, 37, 4097] if cli.quick else [1, 37, 1000, 4097, 8192]
    mx8_modes = ["randn", "special"] + ([] if cli.quick else ["wide"])
    cases = product(mx8_ms, (3072, 6144, 8192, 12288), mx8_modes, (False, True))
    tables["mx8_quant"] = pd.DataFrame(test_mx8_quant(*c) for c in cases)
if "perf" in part:
    big = (8192, 16384)
    tables["norm_mx timing"] = pd.DataFrame(perf_norm(n, m) for m in big for n in NAMES)
    mk = ((8192, 3072), (8192, 6144), (8192, 8192), (16384, 6144))
    tables["mx8_quant timing"] = pd.DataFrame(perf_mx8_quant(m, k) for m, k in mk)
for title, df in tables.items():
    aiter.logger.info(f"{title}:\n{df.to_markdown(index=False)}")
