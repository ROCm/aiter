# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import quant_mxfp4_hip
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight, shuffle_weight_a16w4
from aiter.test_common import benchmark

torch.set_default_device("cuda")


def even_round_scale(max_abs: torch.Tensor) -> torch.Tensor:
    max_abs_f32 = max_abs.to(torch.float32).clone()
    zero_mask = max_abs_f32 == 0

    as_int = max_abs_f32.view(torch.int32)
    as_int.add_(0x200000)
    as_int.bitwise_and_(0x7F800000)
    max_abs_f32 = as_int.view(torch.float32)

    f32_min_normal = 2.0 ** (-126)
    max_abs_f32.masked_fill_(zero_mask, f32_min_normal)

    max_abs_f32.log2_()
    max_abs_f32.floor_()
    max_abs_f32.sub_(2)
    max_abs_f32.clamp_(min=-127, max=127)
    max_abs_f32.exp2_()
    return max_abs_f32


def fp32_to_e2m1_rne(val: torch.Tensor) -> torch.Tensor:
    """E2M1 quantization with RNE (matches gfx950 HW builtin)."""
    qx = val.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    s = qx & 0x80000000
    qx = qx ^ s

    abs_f = qx.to(torch.int32).view(torch.float32)
    sat = abs_f >= 6.0
    denorm = (~sat) & (abs_f < 1.0)
    normal = ~(sat | denorm)

    DENORM_CONST = 149 << 23
    d = abs_f + torch.tensor(DENORM_CONST, dtype=torch.int32, device=val.device).view(
        torch.float32
    )
    d = (d.view(torch.int32).to(torch.int64) & 0xFFFFFFFF) - DENORM_CONST

    mant_odd = (qx >> 22) & 1
    VAL_TO_ADD = ((1 - 127) << 23) + (1 << 21) - 1
    n = (qx + (VAL_TO_ADD & 0xFFFFFFFF) + mant_odd) >> 22

    e2m1 = torch.full_like(qx, 7)
    e2m1 = torch.where(normal, n, e2m1)
    e2m1 = torch.where(denorm, d, e2m1)
    e2m1 = e2m1 | (s >> 28)
    return e2m1.to(torch.uint8)


def fp32_to_e2m1_rha(val: torch.Tensor) -> torch.Tensor:
    """E2M1 quantization with round-half-away (matches non-gfx950 SW fallback)."""
    a = val.abs()
    dev = val.device
    mag = torch.zeros_like(val, dtype=torch.uint8)
    mag = torch.where(a >= 0.25, torch.tensor(1, dtype=torch.uint8, device=dev), mag)
    mag = torch.where(a >= 0.75, torch.tensor(2, dtype=torch.uint8, device=dev), mag)
    mag = torch.where(a >= 1.25, torch.tensor(3, dtype=torch.uint8, device=dev), mag)
    mag = torch.where(a >= 1.75, torch.tensor(4, dtype=torch.uint8, device=dev), mag)
    mag = torch.where(a >= 2.5, torch.tensor(5, dtype=torch.uint8, device=dev), mag)
    mag = torch.where(a >= 3.5, torch.tensor(6, dtype=torch.uint8, device=dev), mag)
    mag = torch.where(a >= 5.0, torch.tensor(7, dtype=torch.uint8, device=dev), mag)
    sign = torch.where(
        val < 0,
        torch.tensor(8, dtype=torch.uint8, device=dev),
        torch.tensor(0, dtype=torch.uint8, device=dev),
    )
    return sign | mag


fp32_to_e2m1 = fp32_to_e2m1_rne if get_gfx() == "gfx950" else fp32_to_e2m1_rha


def ref_quant_mxfp4_even_round(inp: torch.Tensor, group_size: int = 32):
    inp_f32 = inp.float()
    rows, cols = inp_f32.shape
    n_groups = cols // group_size

    inp_grouped = inp_f32.reshape(rows, n_groups, group_size)
    group_max = inp_grouped.abs().amax(dim=-1)
    dq_scale = even_round_scale(group_max)

    q_scale = torch.where(dq_scale == 0, torch.zeros_like(dq_scale), 1.0 / dq_scale)
    scaled = inp_grouped * q_scale.unsqueeze(-1)

    nibbles = fp32_to_e2m1(scaled)
    nibbles = nibbles.reshape(rows, cols)
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)

    scale_e8m0 = ((dq_scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)

    return packed, scale_e8m0


def _fp4_scale_shuffle_id(scaleN_pad, x, y):
    return (
        (x // 32 * scaleN_pad) * 32
        + (y // 8) * 256
        + (y % 4) * 64
        + (x % 16) * 4
        + (y % 8) // 4 * 2
        + (x % 32) // 16
    )


@benchmark()
def test_no_shuffle(m, n, float_dtype):
    torch.manual_seed(42)
    inp = torch.randn((m, n), dtype=float_dtype, device="cuda")

    packed_hip, scale_hip = quant_mxfp4_hip(inp, group_size=32)
    py_packed, py_scale = ref_quant_mxfp4_even_round(inp.cpu(), group_size=32)

    scale_hip_u8 = scale_hip.view(torch.uint8).cpu()
    assert torch.equal(scale_hip_u8, py_scale), f"scale mismatch ({m},{n})"

    packed_hip_u8 = packed_hip.view(torch.uint8).cpu()
    assert torch.equal(packed_hip_u8, py_packed), f"packed mismatch ({m},{n})"

    return {"result": "PASS"}


@benchmark()
def test_e8m0_shuffle(m, n, float_dtype):
    rows, cols = m, n
    if rows % 16 != 0:
        return {"result": "SKIP"}
    K_pk = cols // 2
    if K_pk % 32 != 0:
        return {"result": "SKIP"}

    torch.manual_seed(42)
    inp = torch.randn((m, n), dtype=float_dtype, device="cuda")

    packed_out, scale_out = quant_mxfp4_hip(
        inp, group_size=32, e8m0_shuffle=True, shuffle_weight=True
    )
    packed_ref, scale_ref = quant_mxfp4_hip(inp, group_size=32)
    expected_w = shuffle_weight(packed_ref)

    scaleN = cols // 32
    scaleN_pad = ((scaleN + 7) // 8) * 8

    packed_out_u8 = packed_out.view(torch.uint8).cpu()
    expected_w_u8 = expected_w.view(torch.uint8).cpu()
    assert torch.equal(packed_out_u8, expected_w_u8), f"e8m0 weight mismatch ({m},{n})"

    scale_ref_u8 = scale_ref.view(torch.uint8).flatten().cpu()
    scale_out_u8 = scale_out.view(torch.uint8).flatten().cpu()
    for row in range(rows):
        for g in range(scaleN):
            si = _fp4_scale_shuffle_id(scaleN_pad, row, g)
            li = row * scaleN + g
            assert (
                scale_out_u8[si].item() == scale_ref_u8[li].item()
            ), f"Scale shuffle mismatch at row={row}, group={g}"

    return {"result": "PASS"}


@benchmark()
def test_a16w4_shuffle(m, n, float_dtype, gate_up):
    rows, cols = m, n
    scaleN = cols // 32
    if rows % 32 != 0 or scaleN % 8 != 0:
        return {"result": "SKIP"}
    K_pk = cols // 2
    if K_pk % 64 != 0:
        return {"result": "SKIP"}

    torch.manual_seed(42)
    inp = torch.randn((m, n), dtype=float_dtype, device="cuda")

    packed_out, scale_out = quant_mxfp4_hip(
        inp, group_size=32, a16w4_shuffle=True, gate_up=gate_up, shuffle_weight=True
    )
    packed_ref, scale_ref = quant_mxfp4_hip(inp, group_size=32)
    expected_w = shuffle_weight_a16w4(
        packed_ref.view(torch.uint8).unsqueeze(0), NLane=16, gate_up=gate_up
    ).squeeze(0)
    expected_s = shuffle_scale_a16w4(
        scale_ref.view(torch.uint8).reshape(rows, scaleN),
        experts_cnt=1,
        gate_up=gate_up,
    )

    packed_out_u8 = packed_out.view(torch.uint8).cpu()
    expected_w_u8 = expected_w.view(torch.uint8).cpu()
    assert torch.equal(
        packed_out_u8, expected_w_u8
    ), f"a16w4 weight mismatch (gate_up={gate_up})"

    scale_out_u8 = scale_out.view(torch.uint8).cpu()
    expected_s_u8 = expected_s.view(torch.uint8).cpu()
    assert torch.equal(
        scale_out_u8, expected_s_u8
    ), f"a16w4 scale mismatch (gate_up={gate_up})"

    return {"result": "PASS"}


@benchmark()
def test_edge_values(float_dtype):
    rows, cols = 32, 64

    inp_zero = torch.zeros(rows, cols, dtype=float_dtype, device="cuda")
    packed, scale = quant_mxfp4_hip(inp_zero, group_size=32)
    assert packed.view(torch.uint8).sum() == 0, "zero input failed"

    inp_large = torch.full((rows, cols), 1e4, dtype=float_dtype, device="cuda")
    packed, scale = quant_mxfp4_hip(inp_large, group_size=32)
    assert packed.view(torch.uint8).max() > 0, "large input failed"

    inp_tiny = torch.full((rows, cols), 1e-10, dtype=float_dtype, device="cuda")
    packed, scale = quant_mxfp4_hip(inp_tiny, group_size=32)

    inp_neg = torch.full((rows, cols), -3.0, dtype=float_dtype, device="cuda")
    packed, scale = quant_mxfp4_hip(inp_neg, group_size=32)
    py_packed, _ = ref_quant_mxfp4_even_round(inp_neg.cpu(), group_size=32)
    assert torch.equal(packed.view(torch.uint8).cpu(), py_packed), "neg input failed"

    return {"result": "PASS"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--e8m0_shuffle", action="store_true")
    parser.add_argument("--a16w4_shuffle", action="store_true")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--all", action="store_true", default=True)
    args = parser.parse_args()

    run_all = args.all and not any(
        [args.no_shuffle, args.e8m0_shuffle, args.a16w4_shuffle, args.edge]
    )

    no_shuffle_shapes = [
        (4096, 128),
        (4096, 256),
        (4096, 1024),
        (1, 32),
        (3, 128),
        (125, 64),
        (4097, 256),
    ]
    e8m0_shapes = [
        (4096, 128),
        (4096, 256),
        (4096, 1024),
        (16, 64),
        (48, 64),
        (32, 192),
        (80, 320),
        (256, 96),
    ]
    a16w4_shapes = [
        (4096, 256),
        (4096, 1024),
        (32, 256),
        (64, 512),
        (96, 256),
    ]
    float_dtypes = [torch.bfloat16, torch.float16]

    df = []

    if args.no_shuffle or run_all:
        for (m, n), dt in itertools.product(no_shuffle_shapes, float_dtypes):
            df.append(test_no_shuffle(m, n, dt))

    if args.e8m0_shuffle or run_all:
        for (m, n), dt in itertools.product(e8m0_shapes, float_dtypes):
            df.append(test_e8m0_shuffle(m, n, dt))

    if args.a16w4_shuffle or run_all:
        for (m, n), dt, gu in itertools.product(
            a16w4_shapes, float_dtypes, [False, True]
        ):
            df.append(test_a16w4_shuffle(m, n, dt, gu))

    if args.edge or run_all:
        for dt in float_dtypes:
            test_edge_values(dt)
        aiter.logger.info("test_edge_values: PASS")

    df = pd.DataFrame(df)
    if "gate_up" in df.columns:
        df["gate_up"] = df["gate_up"].fillna(0).astype(int)
    aiter.logger.info("quant_mxfp4 summary:\n%s", df.to_markdown(index=False))
