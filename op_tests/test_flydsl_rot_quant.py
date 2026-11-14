# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes, gemm_a4w4
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_rot_quant, flydsl_rot_quant_moe_sort
from aiter.ops.quant import per_1x32_f4_quant, per_1x32_f4_quant_hip
from aiter.ops.shuffle import shuffle_scale, shuffle_weight
from aiter.test_common import assertAllclose, benchmark, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
# gemm_a4w4 does not match the dequantized product below N=256 (rel. err 0.68
# at N=128, whatever quantized the activation), and for K not a multiple of 256
# it reads past K, so its result depends on the padding. The gemm check stays
# within N >= 256 and K % 256 == 0.
GEMM_MIN_N = 256


def run_torch(x, rs, shuffle):
    """Rotate with the kernel's FWHT stage order in fp32, round to bf16, then
    the shared per-1x32 MXFP4 torch reference."""
    m, k = x.shape
    v = x.float().view(m, k // rs, rs)
    h = 1
    while h < rs:
        v = v.view(m, k // rs, rs // (2 * h), 2, h)
        a, b = v[..., 0, :], v[..., 1, :]
        v = torch.stack([a + b, a - b], dim=-2)
        h *= 2
    xr = (v.reshape(m, k) * (1.0 / math.sqrt(rs))).to(dtypes.bf16)
    xq, xs = per_1x32_f4_quant(xr, quant_dtype=dtypes.fp4x2, shuffle=shuffle)
    return xq.view(torch.uint8), xs.view(torch.uint8)


def run_torch_moe_sort(x, rs, sorted_ids, num_valid_ids, topk):
    """run_torch, with each valid route's scales copied to its sorted row and
    swizzled as mxfp4_moe_sort_fwd lays them out. A route is
    valid when its token is below M and its slot below topk: a sort with
    num_local_tokens pads with topk << 24 | count. Also returns the valid rows
    and their tokens."""
    m, k = x.shape
    xq, xs = run_torch(x, rs, False)
    ids = sorted_ids[: int(num_valid_ids[0])].long()
    token, slot = ids & 0xFFFFFF, (ids >> 24) & 0xFF
    rows = torch.nonzero((token < m) & (slot < topk)).flatten()
    token = token[rows]
    sorted_rows = (sorted_ids.shape[0] + 31) // 32 * 32
    cols = (k // 32 + 7) // 8 * 8
    sorted_xs = torch.zeros(((sorted_rows + 255) // 256 * 256, cols), dtype=torch.uint8)
    sorted_xs[rows, : k // 32] = xs[token]
    sorted_xs = shuffle_scale(sorted_xs).view(-1, cols)[:sorted_rows]
    return xq, sorted_xs, rows, token


def scale_mask(shape, rows, cols, shuffle):
    """True on the scales of `rows` x first `cols` columns, in either layout."""
    mask = torch.zeros(((shape[0] + 255) // 256 * 256, shape[1]), dtype=torch.uint8)
    mask[rows, :cols] = 1
    if shuffle:
        mask = shuffle_scale(mask).view(-1, shape[1])
    return mask[: shape[0]] != 0


def dequant(xq, xs):
    vals = fp4_utils.mxfp4_to_f32(xq)
    return vals * fp4_utils.e8m0_to_f32(xs.repeat_interleave(32, dim=1))


ZERO_ROW, TINY_ROW, NAN_ROW = 1, 3, 5


def make_input(m, k):
    x = torch.randn((m, k), dtype=dtypes.bf16)
    # Outlier channels in every other row drive groups to the top of the e2m1
    # range, where the e8m0 rounding and the saturation decide the result.
    x[::2, ::37] *= 64
    # Rows 1, 3 and 5 when present: a zero row (a padded token), a row far
    # below the HIP amax floor, and a NaN whose groups must get e8m0 NaN (255).
    # e2m1 has no NaN, so the bytes packed for a NaN group are arbitrary and
    # only its scale is compared.
    x[ZERO_ROW : ZERO_ROW + 1] = 0
    x[TINY_ROW : TINY_ROW + 1] *= 1e-30
    x[NAN_ROW : NAN_ROW + 1, 7] = float("nan")
    return x


@benchmark()
def test_rot_quant(m, k, rs, shuffle):
    x = make_input(m, k)
    rows = torch.arange(m)
    ref_q, ref_s = run_torch(x, rs, shuffle)

    # (op, fp4 rows, scale rows, scale columns compared). The padding columns
    # of the CK layout are compared too, the padding rows are uninitialized.
    candidates = {
        "flydsl": (
            lambda: flydsl_rot_quant(x, rotation_size=rs, shuffle=shuffle),
            rows != NAN_ROW,
            rows,
            ref_s.shape[1],
        )
    }
    if rs == 32:
        # #6734 fuses the same rotation. It floors amax (scale 0x5C for zero
        # groups), leaves the padding columns unwritten, and its NaN group
        # bytes are arbitrary, so those rows and columns are left out.
        hip_rows = (rows != ZERO_ROW) & (rows != TINY_ROW) & (rows != NAN_ROW)
        candidates["hip"] = (
            lambda: per_1x32_f4_quant_hip(x, shuffle=shuffle, hadamard_rotate=True),
            hip_rows,
            rows[hip_rows],
            k // 32,
        )

    flops = m * k * int(math.log2(rs))  # butterfly adds
    nbytes = m * k * x.element_size() + m * k // 2 + m * k // 32
    ret = {"gfx": get_gfx()}
    for name, (fn, q_rows, s_rows, s_cols) in candidates.items():
        (xq, xs), us = run_perftest(fn)
        # Packed bytes: any difference is a bug, so compare exactly.
        err = assertAllclose(
            ref_q.to(dtypes.fp32),
            xq.view(torch.uint8).to(dtypes.fp32),
            rtol=0,
            atol=0,
            tol_err_ratio=0,
            mask=q_rows.unsqueeze(1),
            msg=f"{name}: fp4",
        )
        assertAllclose(
            ref_s.to(dtypes.fp32),
            xs.view(torch.uint8).to(dtypes.fp32),
            rtol=0,
            atol=0,
            tol_err_ratio=0,
            mask=scale_mask(ref_s.shape, s_rows, s_cols, shuffle),
            msg=f"{name}: e8m0",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_rot_quant_gemm_a4w4(m, n, k, rs):
    """The model call: rotate and quantize the activation into the CK scale
    layout, then gemm_a4w4. The reference multiplies the dequantized bytes, so
    a scale-layout mismatch shows up as a wrong product."""
    x = torch.randn((m, k), dtype=dtypes.bf16)
    w = torch.randn((n, k), dtype=dtypes.bf16)
    wq, ws = per_1x32_f4_quant_hip(w, shuffle=False)
    wq_shuffled = shuffle_weight(wq, layout=(16, 16))
    ws_shuffled = shuffle_scale(ws)
    ref_q, ref_s = run_torch(x, rs, False)
    ref = dequant(ref_q, ref_s) @ dequant(wq, ws.view(torch.uint8)).T

    def run():
        xq, xs = flydsl_rot_quant(x, rotation_size=rs, shuffle=True)
        return gemm_a4w4(xq, wq_shuffled, xs, ws_shuffled, dtype=dtypes.bf16)

    out, us = run_perftest(run)
    err = assertAllclose(
        ref.to(dtypes.fp32), out.to(dtypes.fp32), msg="flydsl + gemm_a4w4"
    )
    nbytes = m * k * x.element_size() + n * k // 2 + n * k // 32 + m * n * 2
    return {
        "gfx": get_gfx(),
        "flydsl us": us,
        "flydsl TFLOPS": 2 * m * n * k / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": err,
    }


@benchmark()
def test_rot_quant_moe_sort(m, k, rs, experts, topk, block_m, counted):
    """The MoE stage-1 input: scales sorted by moe_sorting's sorted_ids. Only
    the valid routes' rows and the first K // 32 columns are defined, so only
    they are compared. `counted` sorts only the first half of the tokens through
    num_local_tokens, whose padding slots name a real token."""
    x = make_input(m, k)
    ids = torch.rand((m, experts)).topk(topk, dim=1).indices.to(dtypes.i32)
    local = torch.tensor([max(m // 2, 1)], dtype=dtypes.i32) if counted else None
    sorted_ids, _, _, num_valid_ids, _ = moe_sorting(
        ids,
        torch.rand((m, topk)),
        experts,
        k,
        dtypes.bf16,
        block_m,
        num_local_tokens=local,
    )
    ref_q, ref_s, live_rows, live_tokens = run_torch_moe_sort(
        x, rs, sorted_ids, num_valid_ids, topk
    )
    (xq, xs), us = run_perftest(
        flydsl_rot_quant_moe_sort,
        x,
        sorted_ids,
        num_valid_ids,
        m,
        topk,
        block_m,
        rotation_size=rs,
    )
    # Tokens without a valid route may be left unwritten.
    q_rows = torch.zeros(m, dtype=torch.bool)
    q_rows[live_tokens] = True
    q_rows[NAN_ROW : NAN_ROW + 1] = False
    err = assertAllclose(
        ref_q.to(dtypes.fp32),
        xq.view(torch.uint8).to(dtypes.fp32),
        rtol=0,
        atol=0,
        tol_err_ratio=0,
        mask=q_rows.unsqueeze(1),
        msg="flydsl: fp4",
    )
    assertAllclose(
        ref_s.to(dtypes.fp32),
        xs.view(torch.uint8).to(dtypes.fp32),
        rtol=0,
        atol=0,
        tol_err_ratio=0,
        mask=scale_mask(ref_s.shape, live_rows, k // 32, True),
        msg="flydsl: sorted e8m0",
    )
    nbytes = m * k * x.element_size() + m * k // 2 + m * topk * k // 32
    return {
        "gfx": get_gfx(),
        "flydsl us": us,
        "flydsl TFLOPS": m * k * int(math.log2(rs)) / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": err,
    }


def test_rot_quant_rejects_bad_input():
    """Each of these would otherwise be read as contiguous bf16 and quantized
    to plausible-looking garbage."""
    x = torch.randn((64, 4096), dtype=dtypes.bf16)
    cases = {
        "fp16": lambda: flydsl_rot_quant(x.half()),
        "fp32": lambda: flydsl_rot_quant(x.float()),
        "non-contiguous": lambda: flydsl_rot_quant(x[:, ::2]),
        "3D": lambda: flydsl_rot_quant(x.view(2, 32, 4096)),
        "K % RS": lambda: flydsl_rot_quant(x[:, :96].contiguous()),
        "rotation_size=16": lambda: flydsl_rot_quant(x, rotation_size=16),
        # A stage-2 input has M * topk rows.
        "token_num": lambda: flydsl_rot_quant_moe_sort(
            x,
            torch.zeros(1, dtype=dtypes.i32),
            torch.ones(1, dtype=dtypes.i32),
            8,
            8,
            32,
        ),
    }
    for name, fn in cases.items():
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError(f"{name}: expected ValueError")


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("flydsl_rot_quant unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        nargs="*",
        default=[1, 32, 300, 1024, 4096, 16384],
        help="""M (tokens).
    e.g.: -m 32""",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        nargs="*",
        default=[2880, 4096, 7168],
        help="""K (hidden size); skipped where RS does not divide it.
    e.g.: -k 4096""",
    )
    parser.add_argument(
        "-r",
        "--rs",
        type=int,
        nargs="*",
        default=[32, 64, 128],
        help="""Rotation size.
    e.g.: -r 64""",
    )
    parser.add_argument(
        "--shuffle",
        type=dtypes.str2bool,
        nargs="*",
        default=[False, True],
        help="""Write scales in the CK layout of shuffle_scale.
    e.g.: --shuffle true""",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        nargs="*",
        default=[256, 4096],
        help=f"""N of the gemm_a4w4 check, at least {GEMM_MIN_N}.
    e.g.: -n 4096""",
    )
    parser.add_argument(
        "-e",
        "--experts",
        type=int,
        nargs="*",
        default=[128],
        help="""Experts of the MoE check.
    e.g.: -e 128""",
    )
    parser.add_argument(
        "-t",
        "--topk",
        type=int,
        nargs="*",
        default=[8],
        help="""Routing top-k of the MoE check.
    e.g.: -t 8""",
    )
    parser.add_argument(
        "--block_m",
        type=int,
        nargs="*",
        default=[16, 32, 64],
        help="""moe_sorting block size of the MoE check.
    e.g.: --block_m 32""",
    )
    parser.add_argument(
        "--counted",
        type=dtypes.str2bool,
        nargs="*",
        default=[False, True],
        help="""Sort with num_local_tokens (half of M) in the MoE check.
    e.g.: --counted true""",
    )
    args = parser.parse_args()

    df = [
        test_rot_quant(m, k, rs, shuffle)
        for rs, shuffle, k, m in itertools.product(
            args.rs, args.shuffle, args.k, args.m
        )
        if k % rs == 0
    ]
    aiter.logger.info(
        "rot_quant summary (markdown):\n%s", pd.DataFrame(df).to_markdown(index=False)
    )

    df = [
        test_rot_quant_gemm_a4w4(m, n, k, rs)
        for rs, k, n, m in itertools.product(args.rs, args.k, args.n, args.m)
        if n >= GEMM_MIN_N and k % 256 == 0
    ]
    aiter.logger.info(
        "rot_quant + gemm_a4w4 summary (markdown):\n%s",
        pd.DataFrame(df).to_markdown(index=False),
    )

    df = [
        test_rot_quant_moe_sort(m, k, rs, experts, topk, block_m, counted)
        for rs, k, experts, topk, block_m, counted, m in itertools.product(
            args.rs, args.k, args.experts, args.topk, args.block_m, args.counted, args.m
        )
        if k % rs == 0 and topk <= experts
        # Block size and counted sorts change only the grid bound and the route
        # filter, which small M covers.
        and (m <= 1024 or (block_m, counted) == (32, False))
    ]
    aiter.logger.info(
        "rot_quant_moe_sort summary (markdown):\n%s",
        pd.DataFrame(df).to_markdown(index=False),
    )

    test_rot_quant_rejects_bad_input()


if __name__ == "__main__":
    main()
