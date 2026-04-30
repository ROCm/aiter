# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.quant import quant_mxfp4_even_round_hip
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight, shuffle_weight_a16w4


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


def fp32_to_e2m1(val: torch.Tensor) -> torch.Tensor:
    a = val.abs()
    dev = val.device
    mag = torch.zeros_like(val, dtype=torch.uint8)
    mag = torch.where(a >= 0.25, torch.tensor(1, dtype=torch.uint8, device=dev), mag)
    mag = torch.where(a >= 0.75, torch.tensor(2, dtype=torch.uint8, device=dev), mag)
    mag = torch.where(a >= 1.25, torch.tensor(3, dtype=torch.uint8, device=dev), mag)
    mag = torch.where(a >= 1.75, torch.tensor(4, dtype=torch.uint8, device=dev), mag)
    mag = torch.where(a >= 2.5,  torch.tensor(5, dtype=torch.uint8, device=dev), mag)
    mag = torch.where(a >= 3.5,  torch.tensor(6, dtype=torch.uint8, device=dev), mag)
    mag = torch.where(a >= 5.0,  torch.tensor(7, dtype=torch.uint8, device=dev), mag)
    sign = torch.where(val < 0, torch.tensor(8, dtype=torch.uint8, device=dev),
                       torch.tensor(0, dtype=torch.uint8, device=dev))
    return sign | mag


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


@pytest.mark.parametrize("shape", [
    (4096, 128), (4096, 256), (4096, 1024),
    (1, 32), (3, 128), (125, 64), (4097, 256),
])
@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
def test_no_shuffle(shape, float_dtype):
    torch.manual_seed(42)
    inp = torch.randn(shape, dtype=float_dtype, device="cuda")

    packed_hip, scale_hip = quant_mxfp4_even_round_hip(inp, group_size=32)

    py_packed, py_scale = ref_quant_mxfp4_even_round(inp.cpu(), group_size=32)

    scale_hip_u8 = scale_hip.view(torch.uint8).cpu()
    assert torch.equal(scale_hip_u8, py_scale), (
        f"Scale mismatch: {(scale_hip_u8 != py_scale).sum()} elements differ"
    )

    packed_hip_u8 = packed_hip.view(torch.uint8).cpu()
    diff = (packed_hip_u8.to(torch.int16) - py_packed.to(torch.int16)).abs()
    bad = (diff > 0) & (diff != 1) & (diff != 16) & (diff != 17)
    assert not bad.any(), (
        f"Packed diff too large for shape {shape}: max={diff.max().item()}"
    )


@pytest.mark.parametrize("shape", [
    (4096, 128), (4096, 256), (4096, 1024),
    (16, 64), (48, 64),
    (32, 192), (80, 320), (256, 96),
])
@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
def test_e8m0_shuffle(shape, float_dtype):
    rows, cols = shape
    if rows % 16 != 0:
        pytest.skip(f"e8m0 shuffle_weight requires rows%16==0 (got {rows})")
    K_pk = cols // 2
    if K_pk % 32 != 0:
        pytest.skip(f"e8m0 shuffle_weight requires K_pk%32==0 (got K_pk={K_pk})")

    torch.manual_seed(42)
    inp = torch.randn(shape, dtype=float_dtype, device="cuda")

    packed_out, scale_out = quant_mxfp4_even_round_hip(
        inp, group_size=32, e8m0_shuffle=True, shuffle_weight=True
    )
    packed_ref, scale_ref = quant_mxfp4_even_round_hip(inp, group_size=32)
    expected_w = shuffle_weight(packed_ref)

    scaleN = cols // 32
    scaleN_pad = ((scaleN + 7) // 8) * 8

    packed_out_u8 = packed_out.view(torch.uint8).cpu()
    expected_w_u8 = expected_w.view(torch.uint8).cpu()
    assert torch.equal(packed_out_u8, expected_w_u8), (
        f"E8M0 weight mismatch: {(packed_out_u8 != expected_w_u8).sum()} differ"
    )

    scale_ref_u8 = scale_ref.view(torch.uint8).flatten().cpu()
    scale_out_u8 = scale_out.view(torch.uint8).flatten().cpu()

    def _fp4_scale_shuffle_id(scaleN_pad, x, y):
        return ((x // 32 * scaleN_pad) * 32 + (y // 8) * 256
                + (y % 4) * 64 + (x % 16) * 4 + (y % 8) // 4 * 2 + (x % 32) // 16)

    for row in range(rows):
        for g in range(scaleN):
            si = _fp4_scale_shuffle_id(scaleN_pad, row, g)
            li = row * scaleN + g
            assert scale_out_u8[si].item() == scale_ref_u8[li].item(), (
                f"Scale shuffle mismatch at row={row}, group={g}"
            )


@pytest.mark.parametrize("shape", [
    (4096, 256), (4096, 1024),
    (32, 256), (64, 512), (96, 256),
])
@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("gate_up", [False, True])
def test_a16w4_shuffle(shape, float_dtype, gate_up):
    rows, cols = shape
    scaleN = cols // 32
    if rows % 32 != 0 or scaleN % 8 != 0:
        pytest.skip(f"a16w4 requires rows%32==0 and scaleN%8==0 (got {rows}x{cols})")
    K_pk = cols // 2
    if K_pk % 64 != 0:
        pytest.skip(f"a16w4 shuffle_weight requires K_pk%64==0 (got K_pk={K_pk})")

    torch.manual_seed(42)
    inp = torch.randn(shape, dtype=float_dtype, device="cuda")

    packed_out, scale_out = quant_mxfp4_even_round_hip(
        inp, group_size=32, a16w4_shuffle=True, gate_up=gate_up, shuffle_weight=True
    )
    packed_ref, scale_ref = quant_mxfp4_even_round_hip(inp, group_size=32)
    expected_w = shuffle_weight_a16w4(
        packed_ref.view(torch.uint8).unsqueeze(0), NLane=16, gate_up=gate_up
    ).squeeze(0)
    expected_s = shuffle_scale_a16w4(
        scale_ref.view(torch.uint8).reshape(rows, scaleN),
        experts_cnt=1, gate_up=gate_up,
    )

    packed_out_u8 = packed_out.view(torch.uint8).cpu()
    expected_w_u8 = expected_w.view(torch.uint8).cpu()
    assert torch.equal(packed_out_u8, expected_w_u8), (
        f"A16W4 weight shuffle mismatch (gate_up={gate_up}): "
        f"{(packed_out_u8 != expected_w_u8).sum()} bytes differ"
    )

    scale_out_u8 = scale_out.view(torch.uint8).cpu()
    expected_s_u8 = expected_s.view(torch.uint8).cpu()
    assert torch.equal(scale_out_u8, expected_s_u8), (
        f"A16W4 scale shuffle mismatch (gate_up={gate_up}): "
        f"{(scale_out_u8 != expected_s_u8).sum()} bytes differ"
    )


@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
def test_edge_values(float_dtype):
    rows, cols = 32, 64

    inp_zero = torch.zeros(rows, cols, dtype=float_dtype, device="cuda")
    packed, scale = quant_mxfp4_even_round_hip(inp_zero, group_size=32)
    assert packed.view(torch.uint8).sum() == 0

    inp_large = torch.full((rows, cols), 1e4, dtype=float_dtype, device="cuda")
    packed, scale = quant_mxfp4_even_round_hip(inp_large, group_size=32)
    assert packed.view(torch.uint8).max() > 0

    inp_tiny = torch.full((rows, cols), 1e-10, dtype=float_dtype, device="cuda")
    packed, scale = quant_mxfp4_even_round_hip(inp_tiny, group_size=32)

    inp_neg = torch.full((rows, cols), -3.0, dtype=float_dtype, device="cuda")
    packed, scale = quant_mxfp4_even_round_hip(inp_neg, group_size=32)
    py_packed, _ = ref_quant_mxfp4_even_round(inp_neg.cpu(), group_size=32)
    assert torch.equal(packed.view(torch.uint8).cpu(), py_packed)


def test_single_group():
    inp = torch.randn(1, 32, dtype=torch.bfloat16, device="cuda")
    packed, scale = quant_mxfp4_even_round_hip(inp, group_size=32)
    assert packed.view(torch.uint8).shape == (1, 16)
    assert scale.view(torch.uint8).shape == (1, 1)
    py_packed, py_scale = ref_quant_mxfp4_even_round(inp.cpu(), group_size=32)
    assert torch.equal(scale.view(torch.uint8).cpu(), py_scale)


@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
def test_e2m1_boundary_values(float_dtype):
    boundary_vals = [
        0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
        2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, -6.0,
        0.0, 0.24, 0.26, 0.74, 0.76, 1.24, 1.26, 1.74,
        1.76, 2.49, 2.51, 3.49, 3.51, 4.99, 5.01, -0.25,
    ]
    inp = torch.tensor([boundary_vals], dtype=float_dtype, device="cuda")
    packed, scale = quant_mxfp4_even_round_hip(inp, group_size=32)
    py_packed, py_scale = ref_quant_mxfp4_even_round(
        inp.cpu(), group_size=32
    )
    assert torch.equal(scale.view(torch.uint8).cpu(), py_scale)
    packed_u8 = packed.view(torch.uint8).cpu()
    diff = (packed_u8.to(torch.int16) - py_packed.to(torch.int16)).abs()
    bad = (diff > 0) & (diff != 1) & (diff != 16) & (diff != 17)
    assert not bad.any(), f"max diff={diff.max().item()}"


def test_e8m0_shuffle_scale_only():
    torch.manual_seed(42)
    inp = torch.randn(4096, 256, dtype=torch.bfloat16, device="cuda")
    packed_shuf, scale_shuf = quant_mxfp4_even_round_hip(
        inp, group_size=32, e8m0_shuffle=True, shuffle_weight=False
    )
    packed_ref, scale_ref = quant_mxfp4_even_round_hip(inp, group_size=32)
    packed_shuf_u8 = packed_shuf.view(torch.uint8).cpu()
    packed_ref_u8 = packed_ref.view(torch.uint8).cpu()
    assert torch.equal(packed_shuf_u8, packed_ref_u8)
