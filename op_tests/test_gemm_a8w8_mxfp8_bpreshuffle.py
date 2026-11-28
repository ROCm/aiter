# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP8 (1x32 e8m0 scales) bpreshuffle GEMM.

Runs every tuned (M, N, K) of this arch by default (an untuned gfx950 falls
back to a default shape list), or the shapes given by -mnk, through
gemm_a8w8_mxfp8_bpreshuffle and checks it against an fp32 reference.

Both gfx950 and gfx1250 read row-major x_scale [M, K/32]. gfx950 reads
row-major w_scale [N, K/32]; gfx1250 reads 32x4 shuffled w_scale, as laid out
by shuffle_mxfp8_bpreshuffle_scale (whose layout test_w_scale_layout checks).

gfx950 also checks the OPUS MXFP8 1x32 backends: every kid of a8w8_mxfp8
(plain NT / NN / TT / TN) and of a8w8_mxfp8_bpreshuffle (NT, shuffled B)
(split-K kids at S = 2 / 4), the public route of a libtype "opus" tuned row
(injected, not read from the production CSVs), outputs and operands of 4 GiB
and more, K that is not a multiple of 256 (or, through the API, of 32), the
shapes the launchers must reject and the gemm_a8w8_mxfp8_opus API.
"""

import argparse
import itertools
from unittest import mock

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops import shuffle
from aiter.ops.gemm_op_a8w8 import _get_mxfp8_bpreshuffle_config
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale, shuffle_weight
from aiter.test_common import benchmark, checkAllclose, perftest, run_perftest

# Shapes for a gfx950 without tuned rows: DeepSeek-V3 dense projections, ragged M included.
DEFAULT_MNK_GFX950 = [
    (m, n, k)
    for n, k in ((2112, 7168), (7168, 2048), (24576, 1536), (7168, 16384))
    for m in (1, 16, 37, 64, 300, 1024, 4096)
]

# gfx950, untuned K that 128 does not divide: no mxpsh tile runs it, so the
# entry falls back to the batched GEMM's heuristic (DeepSeek-V4.1's
# shared-expert down projection at TP8).
RAGGED_K_MNK_GFX950 = [(6, 5120, 288), (2048, 5120, 288)]


def run_torch(x, x_scale, weight, w_scale, w_block=(128, 128)):
    """fp32 reference from row-major 1x32 A-scales and w_block B-scales."""
    m, k = x.shape
    n = weight.shape[0]
    x_scale = torch.exp2(x_scale.view(torch.uint8).float() - 127)
    x = (x.float().view(m, k // 32, 32) * x_scale[..., None]).view(m, k)
    w_scale = torch.exp2(w_scale.view(torch.uint8).float() - 127)
    w_scale = w_scale.repeat_interleave(w_block[0], 0)[:n]
    w_scale = w_scale.repeat_interleave(w_block[1], 1)
    return x @ (weight.float() * w_scale).T


@perftest()
def run_gemm(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_mxfp8_bpreshuffle(x, weight, x_scale, w_scale, dtype)


@benchmark()
def test_gemm(m, n, k, dtype=dtypes.bf16):
    x = torch.randn(m, k, dtype=dtypes.bf16)
    weight = (torch.randn(n, k) * 0.02).to(dtypes.fp8)
    w_scale = torch.randint(122, 127, (n, k // 32), dtype=torch.uint8)
    w_scale = w_scale.view(dtypes.fp8_e8m0)
    xq, x_scale = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
    )
    ref = run_torch(xq, x_scale, weight, w_scale, (1, 32))
    out, us = run_gemm(
        xq,
        shuffle_weight(weight, layout=(16, 16)),
        x_scale,
        shuffle_mxfp8_bpreshuffle_scale(w_scale),
        dtype,
    )
    err = checkAllclose(ref, out.float(), msg=f"M={m} N={n} K={k}: ")
    libtype, kernel_name, splitk = _get_mxfp8_bpreshuffle_config(m, n, k)
    return {
        "libtype": libtype,
        "splitK": splitk,
        "us": round(us, 2),
        "TFLOPS": round(2 * m * n * k / us / 1e6, 1),
        "TB/s": round((m * k + n * k + m * n * 2) / us / 1e6, 2),
        "err": err,
        "kernelName": kernel_name,
    }


def test_w_scale_layout(n=64, kb=16):
    """[N, K/32] e8m0: gfx1250 family at n32k4 offsets, gfx950 unchanged; bad shapes raise."""
    w_scale = torch.randint(0, 255, (n, kb), dtype=torch.uint8, device="cpu")
    r = torch.arange(n, device="cpu")[:, None]
    c = torch.arange(kb, device="cpu")[None, :]
    n32k4 = (r // 32 * (kb // 4) + c // 4) * 128 + r % 32 * 4 + c % 4
    for gfx, offset in (
        ("gfx1250", n32k4),
        ("gfx1250-strict", n32k4),
        ("gfx950", None),
    ):
        with mock.patch.object(shuffle, "get_gfx_runtime", return_value=gfx):
            out = shuffle_mxfp8_bpreshuffle_scale(w_scale.view(dtypes.fp8_e8m0))
            for bad in ((48, 16), (64, 6)) if offset is not None else ():
                try:
                    shuffle_mxfp8_bpreshuffle_scale(torch.zeros(bad, device="cpu"))
                    raise AssertionError(f"{gfx}: [N, K/32]={bad} must be refused")
                except ValueError:
                    pass
        assert out.shape == w_scale.shape and out.dtype == dtypes.fp8_e8m0, gfx
        flat = out.view(torch.uint8).flatten()
        got = flat.view(n, kb) if offset is None else flat[offset]
        assert torch.equal(got, w_scale), f"{gfx}: wrong w_scale layout"
    aiter.logger.info("shuffle_mxfp8_bpreshuffle_scale layout: OK")


# ---- OPUS a8w8_mxfp8 / a8w8_mxfp8_bpreshuffle (gfx950) ------------------------
# Ragged M / N tails, an N below every tile width, and a tile-aligned shape (the
# only one the "nob" kid runs); M % 16 and N % 16 so every layout is legal.
# K % 256 != 0 (the "np" / persistent kids): a partial last K tile (288), an odd
# number of whole ones (1152) and less than one (96).
DEFAULT_OPUS_MNK = [
    (304, 400, 768),
    (1008, 4128, 2048),
    (16, 128, 256),
    (2048, 1536, 7168),
    (304, 400, 288),
    (1008, 528, 1152),
    (48, 272, 96),
]
# A 4 GiB C: a 32-bit buffer range over all of C wraps here, so these need the
# kernel's per-tile 64-bit C base (and fault or drop rows without it).
DEFAULT_OPUS_LARGE_MNK = [(32768, 65536, 256)]
# A / B over 4 GiB (tile-aligned, so the "nob" kid runs them too): 32-bit
# offsets from the start of the operand wrap here, so these need the kernel's
# 64-bit base per tile (row-major operand) or per K tile (stored [K, M] / [K, N]).
DEFAULT_OPUS_LARGE_AB_MNK = [(1050624, 256, 4096), (256, 1050624, 4096)]
OPUS_LAYOUTS = ["NT", "NN", "TT", "TN"]
PLAIN, BPRESHUFFLE = "a8w8_mxfp8", "a8w8_mxfp8_bpreshuffle"


def opus_registry(family=PLAIN):
    from csrc.opus_gemm.opus_gemm_common import get_registry

    return get_registry(family, get_gfx(), get_cu_num())


def opus_inputs(m, n, k, layout, family=PLAIN):
    """Quantized operands as the layout stores them: A [M, K] or the [M, K] view
    of A stored [K, M]; B [N, K] or the [N, K] view of B stored [K, N]
    (a8w8_mxfp8_bpreshuffle: NT, B the shuffle_weight of [N, K])."""

    def quant(rows):
        # Row blocks: a multi-GiB operand's bf16 source does not have to fit.
        parts = [
            per_group_quant_hip(
                torch.randn(min(8192, rows - r), k, dtype=dtypes.bf16),
                quant_dtype=dtypes.fp8,
                group_size=32,
                scale_type=dtypes.fp8_e8m0,
            )
            for r in range(0, rows, 8192)
        ]
        return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])

    xq, x_scale = quant(m)
    wq, w_scale = quant(n)
    a = xq if layout[0] == "N" else xq.T.contiguous().T
    if family == BPRESHUFFLE:
        b = shuffle_weight(wq, layout=(16, 16))
    else:
        b = wq if layout[1] == "T" else wq.T.contiguous().T
    return xq, x_scale, wq, w_scale, a, b


def run_opus(a, b, y, x_scale, w_scale, kid, split_k, family=PLAIN):
    from aiter.ops.opus import opus_gemm

    return opus_gemm(
        a,
        b,
        y,
        kid=kid,
        family=family,
        layout="bpreshuffle" if family == BPRESHUFFLE else "plain",
        x_scale=x_scale,
        w_scale=w_scale,
        split_k=split_k,
    )


def opus_err(xq, x_scale, wq, w_scale, y):
    """Mismatch fraction against the fp32 reference, row block by row block
    (a 4 GiB output's reference does not fit at once); the worst block counts."""
    m, n = y.shape
    rows = max(1, (1 << 27) // n)
    return max(
        checkAllclose(
            run_torch(xq[r : r + rows], x_scale[r : r + rows], wq, w_scale, (1, 32)),
            y[r : r + rows].float(),
            printLog=False,
        )
        for r in range(0, m, rows)
    )


@benchmark()
def test_opus(m, n, k, family, layout, kid, split_k):
    xq, x_scale, wq, w_scale, a, b = opus_inputs(m, n, k, layout, family)
    # NaN sentinel: an element the kernel leaves unwritten fails the comparison.
    y = torch.full((m, n), float("nan"), dtype=dtypes.bf16)
    run_opus(a, b, y, x_scale, w_scale, kid, split_k, family)
    err = opus_err(xq, x_scale, wq, w_scale, y)
    # No rotated input copies for a multi-GiB output (each would be a full C).
    rotate = 1 if 2 * m * n >= 1 << 31 else 0
    _, us = run_perftest(
        run_opus,
        a,
        b,
        y,
        x_scale,
        w_scale,
        kid,
        split_k,
        family,
        num_rotate_args=rotate,
    )
    assert (
        err == 0
    ), f"OPUS {family} kid {kid} {layout} M={m} N={n} K={k} S={split_k}: err {err}"
    return {
        "gfx": get_gfx(),
        "kernel": opus_registry(family)[kid].name,
        "us": round(us, 2),
        "TFLOPS": round(2 * m * n * k / us / 1e6, 1),
        "TB/s": round((m * k + n * k + m * n * 2) / us / 1e6, 2),
        "err": err,
    }


@benchmark()
def test_opus_large_operand(m, n, k, family, layout, kid):
    """An A or B over 4 GiB. The reference covers the rows (columns) where an
    addressing error shows: both ends of the operand and the rows on either
    side of each 2 GiB multiple of its byte offset."""
    xq, x_scale, wq, w_scale, a, b = opus_inputs(m, n, k, layout, family)
    y = torch.full((m, n), float("nan"), dtype=dtypes.bf16)
    run_opus(a, b, y, x_scale, w_scale, kid, 0, family)
    torch.cuda.synchronize()
    assert not bool(y.isnan().any()), f"{family} kid {kid} {layout}: unwritten C"

    def sample(rows):
        edges = [0, rows] + [(g << 31) // k for g in range(1, rows * k >> 31)]
        picked = torch.cat(
            [torch.arange(max(0, e - 256), min(rows, e + 256)) for e in edges]
        )
        return picked.unique()

    rows, cols = sample(m), sample(n)
    err = checkAllclose(
        run_torch(xq[rows], x_scale[rows], wq[cols], w_scale[cols], (1, 32)),
        y[rows][:, cols].float(),
        printLog=False,
    )
    assert err == 0, f"OPUS {family} kid {kid} {layout} M={m} N={n} K={k}: err {err}"
    return {
        "kernel": opus_registry(family)[kid].name,
        "layout": layout,
        "A GiB": round(m * k / 2**30, 2),
        "B GiB": round(n * k / 2**30, 2),
        "err": err,
    }


@benchmark()
def test_opus_route(m, n, k, kid, split_k):
    """gemm_a8w8_mxfp8_bpreshuffle on an injected libtype "opus" row runs that
    a8w8_mxfp8_bpreshuffle kid."""
    import aiter.ops.opus as opus_pkg
    from aiter.ops import gemm_op_a8w8

    name = opus_registry(BPRESHUFFLE)[kid].name
    xq, x_scale, wq, w_scale, _, b = opus_inputs(m, n, k, "NT", BPRESHUFFLE)
    real_config, real_opus_gemm = (
        gemm_op_a8w8._get_mxfp8_bpreshuffle_config,
        opus_pkg.opus_gemm,
    )
    launched = []

    def spy(*args, **kwargs):
        launched.append((kwargs["kid"], kwargs["split_k"]))
        return real_opus_gemm(*args, **kwargs)

    gemm_op_a8w8._get_mxfp8_bpreshuffle_config = lambda *_: ("opus", name, split_k)
    opus_pkg.opus_gemm = spy
    try:
        out, us = run_gemm(xq, b, x_scale, w_scale)
    finally:
        gemm_op_a8w8._get_mxfp8_bpreshuffle_config = real_config
        opus_pkg.opus_gemm = real_opus_gemm
    assert launched and set(launched) == {(kid, split_k)}, launched
    err = opus_err(xq, x_scale, wq, w_scale, out)
    assert err == 0, f"public route to {name}: err {err}"
    return {"kernel": name, "us": round(us, 2), "err": err}


def check_opus_route_fallbacks():
    """A libtype "opus" row beyond its own dense shape: a ragged M under a
    padded-M "nob" row (its tile's "np" kid), a strided out and a strided XQ."""
    import aiter.ops.opus as opus_pkg
    from aiter.ops import gemm_op_a8w8

    kids = opus_registry(BPRESHUFFLE)
    nob, inst = next((k, i) for k, i in kids.items() if i.co_variant == "nob")
    np_kid = next(
        k
        for k, i in kids.items()
        if i.co_variant == "np" and (i.B_M, i.B_N) == (inst.B_M, inst.B_N)
    )
    n, k = 3 * inst.B_N, 768
    real_config, real_opus_gemm = (
        gemm_op_a8w8._get_mxfp8_bpreshuffle_config,
        opus_pkg.opus_gemm,
    )
    launched = []

    def spy(*args, **kwargs):
        launched.append(kwargs["kid"])
        return real_opus_gemm(*args, **kwargs)

    rows = []
    gemm_op_a8w8._get_mxfp8_bpreshuffle_config = lambda *_: ("opus", inst.name, 1)
    opus_pkg.opus_gemm = spy
    try:
        for case, m, want in (
            ("ragged M under a nob row", 2 * inst.B_M + 1, np_kid),
            ("strided out", 2 * inst.B_M, nob),
            ("strided XQ", 2 * inst.B_M, nob),
        ):
            xq, x_scale, wq, w_scale, _, b = opus_inputs(m, n, k, "NT", BPRESHUFFLE)
            a = xq
            if case == "strided XQ":
                a = torch.empty(m, k + 64, dtype=xq.dtype)[:, :k]
                a.copy_(xq)
            # NaN sentinel: the gap of a strided out must stay untouched.
            base = torch.full((m, n + 16), float("nan"), dtype=dtypes.bf16)
            out = base[:, :n] if case == "strided out" else base[:, :n].contiguous()
            del launched[:]
            got = aiter.gemm_a8w8_mxfp8_bpreshuffle(a, b, x_scale, w_scale, out=out)
            err = opus_err(xq, x_scale, wq, w_scale, out)
            assert got is out and launched == [want], (case, launched, want)
            assert err == 0, f"{case}: err {err}"
            assert bool(base[:, n:].isnan().all()), f"{case}: wrote outside out"
            rows.append({"case": case, "M": m, "kid": launched[0], "err": err})
    finally:
        gemm_op_a8w8._get_mxfp8_bpreshuffle_config = real_config
        opus_pkg.opus_gemm = real_opus_gemm
    return rows


def check_opus_rejects():
    """Inputs the launchers must refuse (a wrong result here is silent)."""
    kids = opus_registry()
    sk = next(kid for kid, inst in kids.items() if inst.co_variant == "sk")
    nob = next(kid for kid, inst in kids.items() if inst.co_variant == "nob")
    np_kid = next(kid for kid, inst in kids.items() if inst.co_variant == "np")
    xq, x_scale, _, w_scale, _, b = opus_inputs(304, 416, 768, "NT")
    padded = torch.empty(304, 768 + 64, dtype=xq.dtype)[:, :768]
    padded.copy_(xq)
    # A stored [K, M] needs M % 16; 312 % 16 = 8.
    _, x_scale_t, _, w_scale_t, a_t, b_t = opus_inputs(312, 416, 768, "TT")
    # The bpreshuffle family has the NT image only.
    _, _, _, _, a_nn, b_nn = opus_inputs(304, 416, 768, "NN")
    cases = {
        "non-dense XQ strides": (padded, b, x_scale, w_scale, PLAIN, np_kid, 0),
        "A stored [K, M] with M % 16": (
            a_t,
            b_t,
            x_scale_t,
            w_scale_t,
            PLAIN,
            np_kid,
            0,
        ),
        "nob kid on a ragged M": (xq, b, x_scale, w_scale, PLAIN, nob, 0),
        "split_k on a non-split-K kid": (xq, b, x_scale, w_scale, PLAIN, np_kid, 2),
        "K not a multiple of split_k * 256": (xq, b, x_scale, w_scale, PLAIN, sk, 8),
        "bpreshuffle with B stored [K, N]": (
            a_nn,
            b_nn,
            x_scale,
            w_scale,
            BPRESHUFFLE,
            np_kid,
            0,
        ),
    }
    rows = []
    for case, (a, bb, xs, ws, family, kid, split_k) in cases.items():
        y = torch.empty(a.shape[0], bb.shape[0], dtype=dtypes.bf16)
        try:
            run_opus(a, bb, y, xs, ws, kid, split_k, family)
            torch.cuda.synchronize()
            rejected = False
        except (RuntimeError, ValueError):
            rejected = True
        assert rejected, f"OPUS accepted: {case}"
        rows.append({"case": case, "kid": kid, "rejected": rejected})
    # A split-K workspace past 4 GiB: the planner must not choose it.
    from aiter.ops.opus.gemm_op_a8w8 import a8w8_mxfp8_kid_runs

    over = not a8w8_mxfp8_kid_runs(kids[sk], 8192, 65536, 1536, 2)
    assert over, "split-K workspace over 4 GiB accepted"
    rows.append({"case": "split-K workspace > 4 GiB", "kid": sk, "rejected": over})
    # A short M on a very wide N: the stores of the tile rows past M would wrap
    # back into C, so planner and launcher bound a whole tile of rows.
    inst = kids[np_kid]
    wide_n = -(-(1 << 30) // (inst.B_M - 1) // 16) * 16
    wide = not a8w8_mxfp8_kid_runs(inst, 16, wide_n, 256)
    try:
        run_opus(
            torch.empty(16, 256, dtype=xq.dtype),
            torch.empty(wide_n, 256, dtype=xq.dtype),
            torch.empty(16, wide_n, dtype=dtypes.bf16),
            torch.empty(16, 8, dtype=x_scale.dtype),
            torch.empty(wide_n, 8, dtype=w_scale.dtype),
            np_kid,
            0,
        )
        torch.cuda.synchronize()
        wide = False
    except (RuntimeError, ValueError):
        pass
    assert wide, f"short M on N={wide_n} accepted (tile rows past M wrap into C)"
    rows.append({"case": f"M=16 on N={wide_n}", "kid": np_kid, "rejected": wide})
    return rows


@benchmark()
def test_opus_api(m, n, k, layout):
    """gemm_a8w8_mxfp8_opus on plain NT / NN / TT / TN storage, kid picked by shape."""
    from aiter.ops.opus import gemm_a8w8_mxfp8_opus
    from aiter.ops.opus.gemm_op_a8w8 import _select_a8w8_mxfp8_kid

    xq, x_scale, wq, w_scale, a, b = opus_inputs(m, n, k, layout)
    y, us = run_perftest(gemm_a8w8_mxfp8_opus, a, b, x_scale, w_scale)
    err = opus_err(xq, x_scale, wq, w_scale, y)
    assert err == 0, f"API {layout} M={m} N={n} K={k}: err {err}"
    kid = _select_a8w8_mxfp8_kid(layout, m, n, k, get_gfx(), get_cu_num())
    return {"kernel": opus_registry()[kid].name, "us": round(us, 2), "err": err}


def check_opus_api():
    """gemm_a8w8_mxfp8_opus: explicit kernelId + out, unsupported storage and
    shapes rejected."""
    from aiter.ops.opus import gemm_a8w8_mxfp8_opus

    kids = opus_registry()
    sk = next(kid for kid, inst in kids.items() if inst.co_variant == "sk")
    rows = []
    # A split-K kid by kernelId (it picks its split) into the caller's buffer.
    xq, x_scale, wq, w_scale, a, b = opus_inputs(304, 416, 1024, "TN")
    out = torch.full((304, 416), float("nan"), dtype=dtypes.bf16)
    y = gemm_a8w8_mxfp8_opus(a, b, x_scale, w_scale, kernelId=sk, out=out)
    err = opus_err(xq, x_scale, wq, w_scale, y)
    assert (
        y.data_ptr() == out.data_ptr() and err == 0
    ), f"kernelId {sk} + out: err {err}"
    rows.append({"case": "kernelId (split-K) + out", "kid": sk, "ok": True})
    # K % 32 != 0: ceil(K / 32) scale groups, the last one over K % 32 elements
    # (the API zero-pads the operands to whole groups).
    m, n, k = 304, 416, 1025
    groups = -(-k // 32)

    def ragged_k(rows):
        x = torch.zeros(rows, groups * 32, dtype=dtypes.bf16)
        x[:, :k] = torch.randn(rows, k, dtype=dtypes.bf16)
        return per_group_quant_hip(
            x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
        )

    (xq_k, x_scale_k), (wq_k, w_scale_k) = ragged_k(m), ragged_k(n)
    ref = run_torch(xq_k, x_scale_k, wq_k, w_scale_k, (1, 32))
    for layout in OPUS_LAYOUTS:
        a_k, b_k = xq_k[:, :k].contiguous(), wq_k[:, :k].contiguous()
        y = gemm_a8w8_mxfp8_opus(
            a_k if layout[0] == "N" else a_k.T.contiguous().T,
            b_k if layout[1] == "T" else b_k.T.contiguous().T,
            x_scale_k,
            w_scale_k,
        )
        err = checkAllclose(ref, y.float(), printLog=False)
        assert err == 0, f"API {layout} K={k}: err {err}"
        rows.append({"case": f"K={k} {layout}", "kid": None, "ok": True})
    padded = torch.empty(416, 1024 + 64, dtype=wq.dtype)[:, :1024]
    padded.copy_(wq)
    _, x_scale_tt, _, w_scale_tt, a_tt, b_tt = opus_inputs(312, 416, 768, "TT")
    cases = {
        "non-dense WQ": (xq, padded, x_scale, w_scale, "needs dense"),
        # A stored [K, M] needs M % 16; 312 % 16 = 8.
        "TT with M % 16": (a_tt, b_tt, x_scale_tt, w_scale_tt, "no TT MXFP8"),
        # ceil(1025 / 32) = 33 scale groups, not K // 32 = 32.
        "K=1025 with K // 32 scale groups": (
            xq_k[:, :k].contiguous(),
            wq_k[:, :k].contiguous(),
            x_scale_k[:, : k // 32].contiguous(),
            w_scale_k,
            "x_scale / w_scale must be",
        ),
    }
    for case, (aa, bb, xs, ws, match) in cases.items():
        try:
            gemm_a8w8_mxfp8_opus(aa, bb, xs, ws)
            rejected = False
        except ValueError as e:
            rejected = match in str(e)
        assert rejected, f"API accepted: {case}"
        rows.append({"case": case, "kid": None, "ok": rejected})
    return rows


def opus_cases(mnks, family, layouts):
    """(m, n, k, family, layout, kid, split_k) for every kid of ``family`` that
    runs the shape."""
    from aiter.ops.opus.gemm_op_a8w8 import a8w8_mxfp8_kid_runs

    for (m, n, k), layout, (kid, inst) in itertools.product(
        mnks, layouts, sorted(opus_registry(family).items())
    ):
        for split_k in (2, 4) if inst.co_variant == "sk" else (0,):
            if a8w8_mxfp8_kid_runs(inst, m, n, k, split_k, layout):
                yield m, n, k, family, layout, kid, split_k


def tuned_shapes():
    table = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_MXFP8_BPRESHUFFLE_FILE)
    table = table[(table["gfx"] == get_gfx()) & (table["cu_num"] == get_cu_num())]
    shapes = list(table[["M", "N", "K"]].itertuples(index=False, name=None))
    if not shapes and get_gfx() == "gfx950":
        return DEFAULT_MNK_GFX950  # not tuned on this machine yet
    return shapes


def summarize(title, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", title, df.to_markdown(index=False))


def main():
    torch.set_default_device("cuda")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=None,
        help="""M,N,K shapes; default: every tuned shape of this arch.
        e.g.: -mnk 512,7168,16384 1000,65536,1536""",
    )
    parser.add_argument(
        "--opus_mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_OPUS_MNK,
        help="gfx950 OPUS kid sweep shapes (M % 16, N % 16 keep every layout legal)",
    )
    parser.add_argument(
        "--opus_large_mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_OPUS_LARGE_MNK,
        help="gfx950 OPUS shapes with a C of 4 GiB or more",
    )
    parser.add_argument(
        "--opus_large_ab_mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_OPUS_LARGE_AB_MNK,
        help="gfx950 OPUS shapes with an A or B of 4 GiB or more",
    )
    parser.add_argument(
        "-l",
        "--layout",
        nargs="*",
        default=OPUS_LAYOUTS,
        choices=OPUS_LAYOUTS,
        help="a8w8_mxfp8 operand layouts (first letter A, second B; T = A stored [K, M], N = B stored [K, N])",
    )
    args = parser.parse_args()

    test_w_scale_layout()
    if get_gfx() not in ("gfx950", "gfx1250"):
        aiter.logger.warning(
            "gemm_a8w8_mxfp8_bpreshuffle unsupported on %s; skipping", get_gfx()
        )
        return
    shapes = args.mnk or tuned_shapes()
    if args.mnk is None and get_gfx() == "gfx950":
        shapes = shapes + RAGGED_K_MNK_GFX950
    summarize("gemm_a8w8_mxfp8_bpreshuffle", [test_gemm(*mnk) for mnk in shapes])
    if get_gfx() != "gfx950":
        aiter.logger.warning(
            "%s: OPUS MXFP8 1x32 is gfx950 only; skipping it", get_gfx()
        )
        return
    torch.manual_seed(0)
    summarize(
        "OPUS a8w8_mxfp8 / a8w8_mxfp8_bpreshuffle kids",
        [
            test_opus(*c)
            for cases in (
                opus_cases(args.opus_mnk, PLAIN, args.layout),
                opus_cases(args.opus_mnk, BPRESHUFFLE, ["NT"]),
            )
            for c in cases
        ],
    )
    # The tile-aligned and the ragged-N kid of each kind: plain TN (both operands
    # transposed) and the bpreshuffle NT.
    large = [
        c
        for cases in (
            opus_cases(args.opus_large_mnk, PLAIN, ["TN"]),
            opus_cases(args.opus_large_mnk, BPRESHUFFLE, ["NT"]),
        )
        for c in cases
        if opus_registry(c[3])[c[5]].co_variant != "sk"
    ]
    summarize("OPUS MXFP8 1x32 large outputs", [test_opus(*c) for c in large])
    # Both operands row-major (plain NT, bpreshuffle NT) and both stored
    # transposed (plain TN), on the boundary-free and the first boundary kid.
    large_ab = [
        c[:6]
        for cases in (
            opus_cases(args.opus_large_ab_mnk, PLAIN, ["NT", "TN"]),
            opus_cases(args.opus_large_ab_mnk, BPRESHUFFLE, ["NT"]),
        )
        for c in cases
        if opus_registry(c[3])[c[5]].co_variant in ("nob", "np")
        and (opus_registry(c[3])[c[5]].B_M, opus_registry(c[3])[c[5]].B_N) == (256, 256)
    ]
    summarize(
        "OPUS MXFP8 1x32 large operands",
        [test_opus_large_operand(*c) for c in large_ab],
    )
    kids = opus_registry(BPRESHUFFLE)
    # Each row must be one the kid runs (K % (split_k * 256) for split-K):
    # the injected config skips the tuned-row check the real lookup makes.
    route = [
        (
            2048,
            1536,
            7168,
            next(k for k, i in kids.items() if i.co_variant == "nob"),
            0,
        ),
        (304, 416, 768, next(k for k, i in kids.items() if i.co_variant == "np"), 0),
        (304, 416, 1024, next(k for k, i in kids.items() if i.co_variant == "sk"), 2),
    ]
    summarize(
        "gemm_a8w8_mxfp8_bpreshuffle OPUS route", [test_opus_route(*r) for r in route]
    )
    summarize(
        "gemm_a8w8_mxfp8_bpreshuffle OPUS route fallbacks",
        check_opus_route_fallbacks(),
    )
    summarize("OPUS MXFP8 1x32 rejects", check_opus_rejects())
    summarize(
        "gemm_a8w8_mxfp8_opus",
        [
            test_opus_api(*mnk, layout)
            for mnk in args.opus_mnk
            for layout in args.layout
        ],
    )
    summarize("gemm_a8w8_mxfp8_opus checks", check_opus_api())


if __name__ == "__main__":
    main()
