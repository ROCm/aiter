# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Ragged MXFP4 (a4w4) grouped GEMM and wgrad on gfx950.

Group sizes are UNEQUAL and live on the device: each block reads its own bounds
from ``group_end_offsets`` and the host never syncs to learn them. The two
directions differ in which axis the groups cut:

    flydsl_grouped_gemm_a4w4_ragged   out[group_g] = X[group_g] @ W[g]^T
                                      groups partition the OUTPUT ROWS
    flydsl_grouped_wgrad_a4w4_ragged  grad_W[g]    = go[group_g]^T @ ia[group_g]
                                      groups partition the CONTRACTION

Operands are built with aiter's own per-1x32 quant at ``shuffle=False``: these
kernels take the plain cast output, with no preshuffled weight and no shuffled
scale plane.

``--groups`` is the ragged axis of the sweep, and its non-``even`` modes are not
padding -- they are the shapes under which a ragged MXFP4 kernel silently drops
work or emits NaN, and each needs its own geometry to reproduce at all:

    even     equal groups; the perf shape
    skewed   unequal groups, one much hotter than the rest
    empty    a zero-length group next to a hot one
    slack    M runs past offs[-1]; those rows belong to no group and must stay
             exactly zero and finite

The reference is a per-group f32 matmul over the SAME dequantized fp4 bits, so
what it measures is the kernel's arithmetic, not the cast's rounding.
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import (
    flydsl_grouped_gemm_a4w4_ragged,
    flydsl_grouped_wgrad_a4w4_ragged,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
SCALE_GROUP_SIZE = 32
SEED = 0
# The wgrad reads its scale plane as whole i32 (4 e8m0 each), so a token row must
# be a whole number of dwords.
WGRAD_ROW_ALIGN = 128
GROUP_MODES = ["even", "skewed", "empty", "slack"]


def group_sizes(mode, e, m):
    """(sizes, slack) for one ragged mode. ``sum(sizes) + slack == m``.

    Every mode keeps each group a multiple of WGRAD_ROW_ALIGN so the same shapes
    drive both kernels.
    """
    unit = WGRAD_ROW_ALIGN
    assert m % (e * unit) == 0, f"m={m} must be a multiple of e*{unit}"
    n_units = m // unit
    if mode == "even":
        return [m // e] * e, 0
    if mode == "skewed":
        # One group takes half the tokens, the rest split what is left.
        hot = (n_units // 2) * unit
        rest = m - hot
        per = (rest // (e - 1) // unit) * unit if e > 1 else 0
        sizes = [hot] + [per] * (e - 1)
        return sizes, m - sum(sizes)
    if mode == "empty":
        # A zero-length group immediately before the hottest one.
        sizes, _ = group_sizes("skewed", e, m)
        sizes = [0] + sizes[:-1] if e > 1 else sizes
        return sizes, m - sum(sizes)
    if mode == "slack":
        # Capacity padding: M runs past the last group's end.
        sizes, _ = group_sizes("even", e, m)
        sizes[-1] -= unit
        return sizes, m - sum(sizes)
    raise ValueError(f"unknown group mode {mode!r}")


def quant_mxfp4(x):
    """Per-1x32 MXFP4 cast along the last axis, no shuffle, as uint8 views."""
    quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    packed, scale = quant(x.contiguous(), shuffle=False)
    return packed.view(torch.uint8), scale.view(torch.uint8)


def dequant_mxfp4(packed, scale):
    """Undo the cast to f32. Mirrors op_tests/test_gemm_a4w4.py's reference."""
    x = fp4_utils.mxfp4_to_f32(packed)
    s = fp4_utils.e8m0_to_f32(scale).repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    return x * s[:, : x.shape[1]]


def run_torch_grouped_gemm(x_q, x_s, w_q, w_s, offs, e, n, k, dtype):
    """Reference only: per-group f32 matmul over the same dequantized bits.

    Rows covered by no group stay zero, which is what the kernel must produce for
    the ``slack`` mode's trailing capacity padding.
    """
    x_d = dequant_mxfp4(x_q, x_s)
    w_d = dequant_mxfp4(
        w_q.reshape(e * n, k // 2), w_s.reshape(e * n, k // SCALE_GROUP_SIZE)
    ).reshape(e, n, k)
    out = torch.zeros(x_q.shape[0], n, dtype=dtypes.fp32)
    start = 0
    for g in range(e):
        end = int(offs[g])
        if end > start:
            out[start:end] = x_d[start:end] @ w_d[g].T
        start = end
    return out.to(dtype)


def run_torch_wgrad(go_t, go_s, ia_t, ia_s, offs, e, n, k, dtype):
    """Reference only. An empty group contributes nothing, so its plane is zero."""
    go_d, ia_d = dequant_mxfp4(go_t, go_s), dequant_mxfp4(ia_t, ia_s)
    out = torch.zeros(e, n, k, dtype=dtypes.fp32)
    start = 0
    for g in range(e):
        end = int(offs[g])
        if end > start:
            out[g] = go_d[:, start:end] @ ia_d[:, start:end].T
        start = end
    return out.to(dtype)


@benchmark()
def test_grouped_gemm(e, m, n, k, dtype, groups):
    sizes, slack = group_sizes(groups, e, m)
    offs = torch.tensor([sum(sizes[: i + 1]) for i in range(e)], dtype=dtypes.i32)
    torch.manual_seed(SEED)
    x = torch.randn(m, k, dtype=dtype)
    w = torch.randn(e, n, k, dtype=dtype)
    x_q, x_s = quant_mxfp4(x)
    w_q, w_s = quant_mxfp4(w.reshape(e * n, k))
    w_q = w_q.reshape(e, n, k // 2)
    w_s = w_s.reshape(e, n, k // SCALE_GROUP_SIZE)
    # The MoE caller owns the destination, so pass one rather than letting the
    # kernel allocate and zero-fill its own.
    out = torch.zeros(m, n, dtype=dtype)

    ref = run_torch_grouped_gemm(x_q, x_s, w_q, w_s, offs, e, n, k, dtype)

    candidates = {
        "flydsl": lambda: flydsl_grouped_gemm_a4w4_ragged(
            x_q, w_q, x_s, w_s, offs, out=out
        ),
    }

    tokens = sum(sizes)  # slack rows are not work
    flops = 2 * tokens * n * k
    nbytes = (
        m * k // 2  # x, 2 fp4 per byte
        + e * n * k // 2  # w
        + (m * k + e * n * k) // SCALE_GROUP_SIZE  # e8m0 scale planes
        + m * n * out.element_size()  # out
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        got, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            got.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: ragged mxfp4 grouped gemm [{groups}]",
        )
        if slack:
            # Rows past offs[-1] belong to no group. Uninitialised memory here is
            # the failure this mode exists to catch, so it is an exact check.
            tail = got[int(offs[-1]) :]
            assert torch.equal(
                tail, torch.zeros_like(tail)
            ), f"{name}: {tail.shape[0]} slack rows must stay exactly zero"
            assert torch.isfinite(tail).all(), f"{name}: non-finite slack rows"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_grouped_wgrad(e, m, n, k, dtype, groups):
    sizes, slack = group_sizes(groups, e, m)
    offs = torch.tensor([sum(sizes[: i + 1]) for i in range(e)], dtype=dtypes.i32)
    torch.manual_seed(SEED)
    # Both operands are contracted over tokens, so both are cast along the token
    # axis -- i.e. the plain per-1x32 cast applied to the transpose, which is how
    # the backward pass produces them.
    go = torch.randn(m, n, dtype=dtype)
    ia = torch.randn(m, k, dtype=dtype)
    go_t, go_s = quant_mxfp4(go.transpose(0, 1))
    ia_t, ia_s = quant_mxfp4(ia.transpose(0, 1))

    ref = run_torch_wgrad(go_t, go_s, ia_t, ia_s, offs, e, n, k, dtype)

    candidates = {
        "flydsl": lambda: flydsl_grouped_wgrad_a4w4_ragged(
            go_t, go_s, ia_t, ia_s, offs
        ),
    }

    tokens = sum(sizes)
    flops = 2 * tokens * n * k
    nbytes = (
        m * n // 2  # go_t
        + m * k // 2  # ia_t
        + (m * n + m * k) // SCALE_GROUP_SIZE  # scale planes
        + e * n * k * torch.finfo(dtype).bits // 8  # grad_W
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        got, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            got.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: ragged mxfp4 wgrad [{groups}]",
        )
        for g, size in enumerate(sizes):
            if size == 0:
                # An expert that routed no token has no gradient. Anything other
                # than exact zero here is uninitialised memory reaching the
                # optimizer, which no tolerance would catch.
                assert torch.equal(
                    got[g], torch.zeros_like(got[g])
                ), f"{name}: empty group g{g} must produce an exactly-zero plane"
        assert torch.isfinite(got).all(), f"{name}: non-finite wgrad output"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "ragged mxfp4 grouped gemm/wgrad needs %s (CDNA4 scaled MFMA); "
            "unsupported on %s; skipping",
            SUPPORTED_GFX,
            get_gfx(),
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.d_dtypes["bf16"]],
        choices=[dtypes.d_dtypes["bf16"]],
        metavar="{bf16}",
        help="output data type.",
    )
    parser.add_argument(
        "-e",
        "--experts",
        type=int,
        nargs="*",
        default=[8],
        help="number of groups (experts).",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(32768, 2048, 1408), (32768, 1408, 2048)],
        help="m,n,k -- m is total tokens across all groups.",
    )
    parser.add_argument(
        "-g",
        "--groups",
        type=str,
        nargs="*",
        default=GROUP_MODES,
        choices=GROUP_MODES,
        help="ragged group-size distribution.",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        rows = []
        for groups, e, (m, n, k) in itertools.product(
            args.groups, args.experts, args.mnk
        ):
            rows.append(test_grouped_gemm(e, m, n, k, dtype, groups))
        aiter.logger.info(
            "ragged mxfp4 grouped gemm (fwd/dgrad) summary:\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )

        rows = []
        for groups, e, (m, n, k) in itertools.product(
            args.groups, args.experts, args.mnk
        ):
            rows.append(test_grouped_wgrad(e, m, n, k, dtype, groups))
        aiter.logger.info(
            "ragged mxfp4 grouped wgrad summary:\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
