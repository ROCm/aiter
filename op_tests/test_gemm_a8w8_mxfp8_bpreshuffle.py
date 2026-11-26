# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP8 (1x32 e8m0 scales) bpreshuffle GEMM.

Runs every tuned (M, N, K) of this arch by default (an untuned gfx950 falls
back to a default shape list), or the shapes given by -mnk, through
gemm_a8w8_mxfp8_bpreshuffle and checks it against an fp32 reference.

Both gfx950 and gfx1250 read row-major x_scale [M, K/32]. gfx950 reads
row-major w_scale [N, K/32]; gfx1250 reads 32x4 shuffled w_scale, as laid out
by shuffle_mxfp8_bpreshuffle_scale (whose layout test_w_scale_layout checks).
"""

import argparse
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


# gfx1250 decode kernel checks: ragged, one-tile and multi-tile M.
DECODE_MNK = [(1, 6144, 2048), (8, 6144, 2048), (33, 6144, 2048), (128, 6144, 2048)]
DECODE_MNK += [(8, 2304, 6144)]


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


def test_gfx1250_bmm_row_dispatch(m=16, n=32, k=128):
    """gfx1250: a batched, streaming or decode row runs the B = 1 batched runner;
    preshuffled A scales, and fp16 output on a decode row, take the direct kernel
    instead. CPU only (mocked launches)."""
    from aiter.ops import gemm_op_a8w8 as op
    from aiter.ops.flydsl import batched_gemm_a8w8 as bmm
    from aiter.ops.flydsl import mxfp8_bpreshuffle_gemm_gfx1250 as direct

    names = (
        "flydsl_bmm_mxfp8_32_wmma_t16x32x128_mw1_nw2_nb2_sk1_cm1_cn1",
        "flydsl_mxfp8_32_bpreshuffle_stream_t16x16_wn1_wk4_sk1_d4",
        "flydsl_mxfp8_32_bpreshuffle_decode_t16_nb2_ks1_kw1_wn1_d1",
    )
    x, w = (torch.empty((r, k), dtype=torch.uint8, device="cpu") for r in (m, n))
    xs, ws = (
        torch.empty((r, k // 32), dtype=torch.uint8, device="cpu") for r in (m, n)
    )
    calls = []

    def run_bmm(xq, wq, x_scale, w_scale, out, kernel_name=None, **_kwargs):
        calls.append(("bmm", tuple(xq.shape), kernel_name))
        return out

    def run_direct(*args, **kwargs):
        calls.append(("direct", kwargs["a_scale_preshuffled"], args[5]))
        return args[4]

    def row(name):
        return mock.patch.object(
            op, "_get_mxfp8_bpreshuffle_config", return_value=("flydsl", name, 1)
        )

    gemm = op.gemm_a8w8_mxfp8_bpreshuffle.__wrapped__
    with (
        mock.patch.object(op, "get_gfx", return_value="gfx1250"),
        mock.patch.object(op, "_mxfp8_32_fallback_kernel_name", return_value="dk"),
        mock.patch.object(bmm, "run_bmm_a8w8_mxfp8", run_bmm),
        mock.patch.object(
            direct, "run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250", run_direct
        ),
    ):
        for name in names:
            with row(name):
                assert gemm(x, w, xs, ws, dtype=dtypes.bf16).shape == (m, n)
        with row(names[0]):
            gemm(x, w, xs, ws, dtype=dtypes.bf16, a_scale_preshuffled=True)
        with row(names[2]):
            gemm(x, w, xs, ws, dtype=dtypes.fp16)
    assert calls == [("bmm", (m, 1, k), name) for name in names] + [
        ("direct", True, "dk_aspre"),
        ("direct", False, "dk"),
    ], calls
    aiter.logger.info("gfx1250 B=1 runner row dispatch: OK")


@benchmark()
def test_gfx1250_decode_kernels(m, n, k, dtype=dtypes.bf16):
    """Every decode kernel the gfx1250 tuner offers for the shape, launched as a
    tuned row launches it (the B = 1 batched runner), against the fp32
    reference; rows past M must stay unwritten."""
    from aiter.ops.flydsl.batched_gemm_a8w8 import run_bmm_a8w8_mxfp8
    from aiter.ops.flydsl.mxfp8_decode_gemm_gfx1250 import (
        mxfp8_32_decode_candidates,
    )

    names = mxfp8_32_decode_candidates(m, n, k)
    assert names, f"no decode kernel for M={m} N={n} K={k}"
    x = torch.randn(m, k, dtype=dtypes.bf16)
    weight = (torch.randn(n, k) * 0.02).to(dtypes.fp8)
    w_scale = torch.randint(122, 127, (n, k // 32), dtype=torch.uint8)
    w_scale = w_scale.view(dtypes.fp8_e8m0)
    xq, x_scale = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
    )
    ref = run_torch(xq, x_scale, weight, w_scale, (1, 32))
    w_sh = shuffle_weight(weight, layout=(16, 16)).view(1, n, k)
    ws_sh = shuffle_mxfp8_bpreshuffle_scale(w_scale).view(1, n, k // 32)
    best = (float("inf"), None)
    max_err = 0.0
    for name in names:
        buf = torch.full((m + 7, n), -3.0, dtype=dtype)
        _, us = run_perftest(
            run_bmm_a8w8_mxfp8,
            xq.view(m, 1, k),
            w_sh,
            x_scale.view(m, 1, k // 32),
            ws_sh,
            buf[:m].view(m, 1, n),
            kernel_name=name,
        )
        assert torch.all(buf[m:] == -3.0), f"{name}: rows past M written"
        err = checkAllclose(ref, buf[:m].float(), msg=f"{name} M={m}: ")
        assert err <= 0.05, f"{name} M={m}: {err:.2%} of outputs mismatch"
        max_err = max(max_err, err)
        best = min(best, (us, name))
    return {
        "candidates": len(names),
        "best us": round(best[0], 2),
        "max err": max_err,
        "best kernelName": best[1],
    }


def tuned_shapes():
    table = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_MXFP8_BPRESHUFFLE_FILE)
    table = table[(table["gfx"] == get_gfx()) & (table["cu_num"] == get_cu_num())]
    shapes = list(table[["M", "N", "K"]].itertuples(index=False, name=None))
    if not shapes and get_gfx() == "gfx950":
        return DEFAULT_MNK_GFX950  # not tuned on this machine yet
    return shapes


if __name__ == "__main__":
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
    args = parser.parse_args()

    test_w_scale_layout()
    test_gfx1250_bmm_row_dispatch()
    if get_gfx() not in ("gfx950", "gfx1250"):
        print(f"skip: gfx950 / gfx1250 only, got {get_gfx()}")
    else:
        shapes = args.mnk or tuned_shapes()
        if args.mnk is None and get_gfx() == "gfx950":
            shapes = shapes + RAGGED_K_MNK_GFX950
        df = pd.DataFrame([test_gemm(*mnk) for mnk in shapes])
        aiter.logger.info(f"gemm_a8w8_mxfp8_bpreshuffle summary:\n{df.to_markdown()}")
    if get_gfx() == "gfx1250":
        from aiter.ops.flydsl.mxfp8_decode_gemm_gfx1250 import (
            mxfp8_32_decode_candidates,
        )

        decode = [s for s in args.mnk or DECODE_MNK if mxfp8_32_decode_candidates(*s)]
        df = pd.DataFrame([test_gfx1250_decode_kernels(*mnk) for mnk in decode])
        aiter.logger.info(f"gfx1250 decode kernels:\n{df.to_markdown()}")
