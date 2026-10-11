# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness oracle for fused heterogeneous MoE (FHMoE).

The generic MXFP4 case is deliberately small. Dedicated MXFP8 tests use the
production HY4 TP8 dimensions (6144x256, E=257) and allocate several GiB.
Set AITER_HETERO_MOE_DSV4=1 to run the generic path with exact
DeepSeek-V4-Pro TP8 dimensions, or AITER_HETERO_MOE_HY4=1 to run that sweep
with HY4 MXFP8 routed weights, unclamped shared semantics, and tuned dispatch.
Set AITER_HETERO_MOE_FULL_SWEEP=1 to cover M=1,4,8,16,24,32,40,64 and
AITER_HETERO_MOE_STRESS_REPEATS=500 to stress forced-reduce and default
stage-2 execution and bound atomic run-to-run variation where applicable.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.flydsl.moe_kernels import runtime_swiglu_limit
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import (
    shuffle_scale,
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
from aiter.ops.triton.quant import dynamic_mxfp8_quant
from aiter.utility import fp4_utils
from aiter.utility.mx_types import MxDtypeInt


@dataclass(frozen=True)
class _Profile:
    hidden: int
    inter: int
    logical_inter: int
    experts: int
    routed_topk: int

    @property
    def shared_id(self) -> int:
        return self.experts - 1

    @property
    def intermediate_pad(self) -> int:
        return self.inter - self.logical_inter


@dataclass
class _Weights:
    routed_w1: torch.Tensor
    routed_w2: torch.Tensor
    routed_s1: torch.Tensor
    routed_s2: torch.Tensor
    raw_routed_w1: torch.Tensor
    raw_routed_w2: torch.Tensor
    raw_routed_s1: torch.Tensor
    raw_routed_s2: torch.Tensor
    routed_expert_ids: tuple[int, ...]
    shared_w1: torch.Tensor
    shared_w2: torch.Tensor
    shared_s1: torch.Tensor
    shared_s2: torch.Tensor
    native_w1: torch.Tensor
    native_w2: torch.Tensor
    native_s1: torch.Tensor
    native_s2: torch.Tensor
    raw_native_w1: torch.Tensor
    raw_native_w2: torch.Tensor
    raw_native_s1: torch.Tensor
    raw_native_s2: torch.Tensor


def _profile() -> _Profile:
    if _use_hy4_profile():
        return _Profile(6144, 256, 256, 257, 8)
    if os.environ.get("AITER_HETERO_MOE_DSV4", "0") == "1":
        return _Profile(7168, 384, 384, 385, 6)
    return _Profile(256, 128, 128, 9, 2)


def _use_hy4_profile() -> bool:
    return os.environ.get("AITER_HETERO_MOE_HY4", "0") == "1"


def _mxfp8_profile() -> _Profile:
    """The production contract for MXFP8 routed and FP8 shared experts."""
    return _Profile(6144, 256, 256, 257, 8)


def _m_values() -> list[int]:
    if os.environ.get("AITER_HETERO_MOE_FULL_SWEEP", "0") == "1":
        return [1, 4, 8, 16, 24, 32, 40, 64]
    return [4]


def _swiglu_limit(profile: _Profile) -> float:
    return 10.0


def _shared_swiglu_limit(swiglu_limit: float, *, clamp_shared: bool) -> float:
    return swiglu_limit if clamp_shared else float("inf")


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = actual.float() - expected.float()
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return float(torch.linalg.vector_norm(delta) / denominator)


def _cosine_distance(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.double().flatten()
    expected = expected.double().flatten()
    denominator = (actual.square() + expected.square()).sum().clamp_min(1e-24)
    return float(1 - 2 * (actual * expected).sum() / denominator)


def _expand_128x128_scale(
    block_scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Losslessly expand checkpoint 128x128 scales to row-by-32 scales."""
    scale_u8 = block_scale.view(torch.uint8)
    expanded = scale_u8.repeat_interleave(128, 0).repeat_interleave(4, 1)
    return expanded[:rows, : cols // 32].contiguous().view(dtypes.fp8_e8m0)


def _shuffle_w1(
    weight: torch.Tensor,
    scale: torch.Tensor,
    interleave: bool = True,
) -> tuple[torch.Tensor, ...]:
    raw_scale = scale.view(-1, scale.shape[-1])
    if interleave:
        shuffled_weight = shuffle_weight_a16w4(weight, 16, True)
        shuffled_scale = shuffle_scale_a16w4(raw_scale, weight.shape[0], True)
    else:
        shuffled_weight = shuffle_weight(weight, layout=(16, 16))
        shuffled_scale = shuffle_scale(raw_scale)
    return shuffled_weight, shuffled_scale


def _shuffle_w2(
    weight: torch.Tensor,
    scale: torch.Tensor,
    interleave: bool = True,
) -> tuple[torch.Tensor, ...]:
    raw_scale = scale.view(-1, scale.shape[-1])
    if interleave:
        shuffled_weight = shuffle_weight_a16w4(weight, 16, False)
        shuffled_scale = shuffle_scale_a16w4(raw_scale, weight.shape[0], False)
    else:
        shuffled_weight = shuffle_weight(weight, layout=(16, 16))
        shuffled_scale = shuffle_scale(raw_scale)
    return shuffled_weight, shuffled_scale


def _make_shared_weights(
    profile: _Profile,
    device: torch.device,
    interleave: bool = True,
) -> tuple:
    generator = torch.Generator(device=device).manual_seed(17)
    h = profile.hidden
    logical_i = profile.logical_inter
    padded_i = profile.inter

    gate = (torch.randn((logical_i, h), generator=generator, device=device) * 0.04).to(
        dtypes.fp8
    )
    up = (torch.randn((logical_i, h), generator=generator, device=device) * 0.04).to(
        dtypes.fp8
    )
    down = (torch.randn((h, logical_i), generator=generator, device=device) * 0.04).to(
        dtypes.fp8
    )
    native_w1 = torch.cat((gate, up), dim=0).unsqueeze(0)
    native_w2 = down.unsqueeze(0)

    s1_rows = 2 * logical_i // 128
    s1_cols = h // 128
    s2_rows = h // 128
    s2_cols = logical_i // 128
    native_block_s1 = (
        126
        + torch.arange(s1_rows * s1_cols, device=device, dtype=torch.uint8).view(
            s1_rows, s1_cols
        )
        % 2
    ).view(dtypes.fp8_e8m0)
    native_block_s2 = (
        126
        + torch.arange(s2_rows * s2_cols, device=device, dtype=torch.uint8).view(
            s2_rows, s2_cols
        )
        % 2
    ).view(dtypes.fp8_e8m0)
    native_s1 = _expand_128x128_scale(native_block_s1, 2 * logical_i, h)
    native_s2 = _expand_128x128_scale(native_block_s2, h, logical_i)

    padded_w1 = torch.zeros((1, 2 * padded_i, h), dtype=dtypes.fp8, device=device)
    padded_w1[:, :logical_i] = gate
    padded_w1[:, padded_i : padded_i + logical_i] = up
    padded_w2 = torch.zeros((1, h, padded_i), dtype=dtypes.fp8, device=device)
    padded_w2[:, :, :logical_i] = down

    padded_s1 = torch.full(
        (1, 2 * padded_i, h // 32),
        0x7F,
        dtype=torch.uint8,
        device=device,
    ).view(dtypes.fp8_e8m0)
    padded_s1[:, :logical_i] = native_s1[:logical_i]
    padded_s1[:, padded_i : padded_i + logical_i] = native_s1[logical_i:]
    padded_s2 = torch.full(
        (1, h, padded_i // 32),
        0x7F,
        dtype=torch.uint8,
        device=device,
    ).view(dtypes.fp8_e8m0)
    padded_s2[:, :, : logical_i // 32] = native_s2

    shared_w1, shared_s1 = _shuffle_w1(padded_w1, padded_s1, interleave)
    shared_w2, shared_s2 = _shuffle_w2(padded_w2, padded_s2, interleave)
    native_w1_shuf, native_s1_shuf = _shuffle_w1(
        native_w1, native_s1.unsqueeze(0), interleave
    )
    native_w2_shuf, native_s2_shuf = _shuffle_w2(
        native_w2, native_s2.unsqueeze(0), interleave
    )
    return (
        shared_w1,
        shared_w2,
        shared_s1,
        shared_s2,
        native_w1_shuf,
        native_w2_shuf,
        native_s1_shuf,
        native_s2_shuf,
        native_w1,
        native_w2,
        native_s1,
        native_s2,
    )


def _make_routed_weights(
    profile: _Profile,
    device: torch.device,
    interleave: bool = True,
) -> tuple:
    h = profile.hidden
    i = profile.inter
    e = profile.experts
    generator = torch.Generator(device=device).manual_seed(29)

    w1_u8 = torch.zeros((e, 2 * i, h // 2), dtype=torch.uint8, device=device)
    w2_u8 = torch.zeros((e, h, i // 2), dtype=torch.uint8, device=device)
    s1_u8 = torch.full((e, 2 * i, h // 32), 0x7F, dtype=torch.uint8, device=device)
    s2_u8 = torch.full((e, h, i // 32), 0x7F, dtype=torch.uint8, device=device)

    for expert_id in range(profile.routed_topk):
        scale = 0.02 + 0.005 * expert_id
        dense_w1 = (
            torch.randn((1, 2 * i, h), generator=generator, device=device) * scale
        ).to(dtypes.bf16)
        dense_w2 = (
            torch.randn((1, h, i), generator=generator, device=device) * scale
        ).to(dtypes.bf16)
        if profile.intermediate_pad:
            dense_w1[:, profile.logical_inter : i] = 0
            dense_w1[:, i + profile.logical_inter :] = 0
            dense_w2[:, :, profile.logical_inter :] = 0
        quant_w1, quant_s1 = per_1x32_f4_quant(dense_w1)
        quant_w2, quant_s2 = per_1x32_f4_quant(dense_w2)
        w1_u8[expert_id].copy_(quant_w1[0].view(torch.uint8))
        w2_u8[expert_id].copy_(quant_w2[0].view(torch.uint8))
        s1_u8[expert_id].copy_(quant_s1.view(torch.uint8))
        s2_u8[expert_id].copy_(quant_s2.view(torch.uint8))

    w1 = w1_u8.view(dtypes.fp4x2)
    w2 = w2_u8.view(dtypes.fp4x2)
    s1 = s1_u8.view(dtypes.fp8_e8m0)
    s2 = s2_u8.view(dtypes.fp8_e8m0)
    routed_expert_ids = tuple(range(profile.routed_topk))
    raw_w1 = w1[: profile.routed_topk].clone()
    raw_w2 = w2[: profile.routed_topk].clone()
    raw_s1 = s1[: profile.routed_topk].clone()
    raw_s2 = s2[: profile.routed_topk].clone()
    return (
        *_shuffle_w1(w1, s1, interleave),
        *_shuffle_w2(w2, s2, interleave),
        raw_w1,
        raw_w2,
        raw_s1,
        raw_s2,
        routed_expert_ids,
    )


def _make_mxfp8_routed_weights(
    profile: _Profile,
    device: torch.device,
    interleave: bool = True,
) -> tuple:
    h = profile.hidden
    i = profile.inter
    e = profile.experts
    generator = torch.Generator(device=device).manual_seed(29)

    w1 = torch.zeros((e, 2 * i, h), dtype=dtypes.fp8, device=device)
    w2 = torch.zeros((e, h, i), dtype=dtypes.fp8, device=device)
    s1_u8 = torch.full((e, 2 * i, h // 32), 0x7F, dtype=torch.uint8, device=device)
    s2_u8 = torch.full((e, h, i // 32), 0x7F, dtype=torch.uint8, device=device)

    routed_expert_ids = (
        (0, 1, 7, 8, 127, 128, 254, 255)
        if (profile.experts, profile.routed_topk) == (257, 8)
        else tuple(range(profile.routed_topk))
    )
    for slot, expert_id in enumerate(routed_expert_ids):
        scale = 0.02 + 0.005 * slot
        dense_w1 = (
            torch.randn((1, 2 * i, h), generator=generator, device=device) * scale
        ).to(dtypes.bf16)
        dense_w2 = (
            torch.randn((1, h, i), generator=generator, device=device) * scale
        ).to(dtypes.bf16)
        if profile.intermediate_pad:
            dense_w1[:, profile.logical_inter : i] = 0
            dense_w1[:, i + profile.logical_inter :] = 0
            dense_w2[:, :, profile.logical_inter :] = 0
        quant_w1, quant_s1 = dynamic_mxfp8_quant(dense_w1, quant_dtype=dtypes.fp8)
        quant_w2, quant_s2 = dynamic_mxfp8_quant(dense_w2, quant_dtype=dtypes.fp8)
        w1[expert_id].copy_(quant_w1[0])
        w2[expert_id].copy_(quant_w2[0])
        s1_u8[expert_id].copy_(quant_s1[0])
        s2_u8[expert_id].copy_(quant_s2[0])

    s1 = s1_u8.view(dtypes.fp8_e8m0)
    s2 = s2_u8.view(dtypes.fp8_e8m0)
    active = torch.tensor(routed_expert_ids, dtype=torch.long, device=device)
    raw_w1 = w1.index_select(0, active).clone()
    raw_w2 = w2.index_select(0, active).clone()
    raw_s1 = s1.index_select(0, active).clone()
    raw_s2 = s2.index_select(0, active).clone()
    return (
        *_shuffle_w1(w1, s1, interleave),
        *_shuffle_w2(w2, s2, interleave),
        raw_w1,
        raw_w2,
        raw_s1,
        raw_s2,
        routed_expert_ids,
    )


def _build_weights(
    profile: _Profile, interleave: bool, *, routed_mxfp8: bool = False
) -> _Weights:
    if get_gfx() != "gfx950":
        pytest.skip("heterogeneous MXFP4/FP8 or MXFP8/FP8 MoE requires gfx950")
    if "shared_w1" not in inspect.signature(fused_moe).parameters:
        pytest.fail("AITER fused_moe is missing the heterogeneous shared arguments")

    device = torch.device("cuda")
    (
        routed_w1,
        routed_s1,
        routed_w2,
        routed_s2,
        raw_routed_w1,
        raw_routed_w2,
        raw_routed_s1,
        raw_routed_s2,
        routed_expert_ids,
    ) = (
        _make_mxfp8_routed_weights(profile, device, interleave)
        if routed_mxfp8
        else _make_routed_weights(profile, device, interleave)
    )
    (
        shared_w1,
        shared_w2,
        shared_s1,
        shared_s2,
        native_w1,
        native_w2,
        native_s1,
        native_s2,
        raw_native_w1,
        raw_native_w2,
        raw_native_s1,
        raw_native_s2,
    ) = _make_shared_weights(profile, device, interleave)
    return _Weights(
        routed_w1,
        routed_w2,
        routed_s1,
        routed_s2,
        raw_routed_w1,
        raw_routed_w2,
        raw_routed_s1,
        raw_routed_s2,
        routed_expert_ids,
        shared_w1,
        shared_w2,
        shared_s1,
        shared_s2,
        native_w1,
        native_w2,
        native_s1,
        native_s2,
        raw_native_w1,
        raw_native_w2,
        raw_native_s1,
        raw_native_s2,
    )


@pytest.fixture(scope="module")
def weights() -> _Weights:
    return _build_weights(
        _profile(),
        interleave=True,
        routed_mxfp8=_use_hy4_profile(),
    )


@pytest.fixture(scope="module")
def generic_weights() -> _Weights:
    return _build_weights(_Profile(256, 128, 128, 9, 2), interleave=True)


@pytest.fixture(scope="module")
def a4_weights() -> _Weights:
    profile = _Profile(256, 128, 128, 9, 2)
    return _build_weights(profile, interleave=False)


@pytest.fixture(scope="module")
def mxfp8_weights() -> _Weights:
    return _build_weights(_mxfp8_profile(), interleave=True, routed_mxfp8=True)


@pytest.fixture(scope="module")
def dsv4_weights() -> _Weights:
    return _build_weights(_Profile(7168, 384, 384, 385, 6), interleave=True)


def _route_inputs(
    profile: _Profile,
    m: int,
    device: torch.device,
    routed_expert_ids: tuple[int, ...] | None = None,
) -> tuple:
    hidden_generator = torch.Generator(device=device).manual_seed(41 + m)
    hidden = torch.randn(
        (m, profile.hidden),
        generator=hidden_generator,
        dtype=dtypes.bf16,
        device=device,
    )
    row = torch.arange(m, device=device, dtype=dtypes.i32).unsqueeze(1)
    slot = torch.arange(profile.routed_topk, device=device, dtype=dtypes.i32).unsqueeze(
        0
    )
    active_ids = torch.tensor(
        routed_expert_ids or tuple(range(profile.routed_topk)),
        device=device,
        dtype=dtypes.i32,
    )
    assert active_ids.numel() == profile.routed_topk
    routed_ids = active_ids[(row + slot) % profile.routed_topk]
    raw_weights = torch.arange(
        1,
        profile.routed_topk + 1,
        device=device,
        dtype=dtypes.fp32,
    ).repeat(m, 1)
    routed_weights = raw_weights / raw_weights.sum(dim=1, keepdim=True) * 2.5
    shared_ids = torch.full((m, 1), profile.shared_id, device=device, dtype=dtypes.i32)
    shared_weights = torch.ones((m, 1), device=device, dtype=dtypes.fp32)
    return (
        hidden,
        routed_weights,
        routed_ids,
        torch.cat((routed_weights, shared_weights), dim=1),
        torch.cat((routed_ids, shared_ids), dim=1),
    )


def _common_kwargs(
    profile: _Profile,
    w1_scale,
    w2_scale,
    gate_mode: GateMode = GateMode.INTERLEAVE,
) -> dict:
    return {
        "activation": aiter.ActivationType.Silu,
        "quant_type": aiter.QuantType.per_1x32,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "intermediate_pad": profile.intermediate_pad,
        "swiglu_limit": _swiglu_limit(profile),
        "gate_mode": gate_mode.value,
    }


def _mark_shuffled(tensor: torch.Tensor) -> torch.Tensor:
    tensor.is_shuffled = True
    return tensor


def _run_composed_oracle(
    profile: _Profile,
    weights: _Weights,
    m: int,
    *,
    clamp_shared: bool,
    swiglu_limit: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    hidden, routed_weight, routed_ids, all_weight, all_ids = _route_inputs(
        profile,
        m,
        weights.routed_w1.device,
        weights.routed_expert_ids,
    )
    hetero_kwargs = _common_kwargs(profile, weights.routed_s1, weights.routed_s2)
    hetero_kwargs["swiglu_limit"] = swiglu_limit
    hetero_kwargs.update(
        shared_w1=weights.shared_w1,
        shared_w2=weights.shared_w2,
        shared_w1_scale=weights.shared_s1,
        shared_w2_scale=weights.shared_s2,
        shared_expert_id=profile.shared_id,
        clamp_shared=clamp_shared,
    )
    actual = fused_moe(
        hidden,
        weights.routed_w1,
        weights.routed_w2,
        all_weight,
        all_ids,
        **hetero_kwargs,
    )

    routed_e = profile.experts - 1
    routed_w1 = _mark_shuffled(weights.routed_w1[:routed_e])
    routed_w2 = _mark_shuffled(weights.routed_w2[:routed_e])
    routed_s1 = weights.routed_s1[: routed_e * 2 * profile.inter]
    routed_s2 = weights.routed_s2[: routed_e * profile.hidden]
    routed_kwargs = _common_kwargs(profile, routed_s1, routed_s2)
    routed_kwargs["swiglu_limit"] = swiglu_limit
    routed_out = fused_moe(
        hidden,
        routed_w1,
        routed_w2,
        routed_weight,
        routed_ids,
        **routed_kwargs,
    )

    shared_ids = torch.zeros((m, 1), dtype=dtypes.i32, device=hidden.device)
    shared_weight = torch.ones((m, 1), dtype=dtypes.fp32, device=hidden.device)
    native_profile = _Profile(
        profile.hidden,
        profile.logical_inter,
        profile.logical_inter,
        1,
        1,
    )
    shared_kwargs = _common_kwargs(native_profile, weights.native_s1, weights.native_s2)
    shared_kwargs["swiglu_limit"] = _shared_swiglu_limit(
        swiglu_limit, clamp_shared=clamp_shared
    )
    shared_out = fused_moe(
        hidden,
        weights.native_w1,
        weights.native_w2,
        shared_weight,
        shared_ids,
        **shared_kwargs,
    )
    return (
        actual,
        routed_out + shared_out,
        {
            "hidden": hidden,
            "routed_weight": routed_weight,
            "routed_ids": routed_ids,
            "all_weight": all_weight,
            "all_ids": all_ids,
            "hetero_kwargs": hetero_kwargs,
            "routed_out": routed_out,
            "shared_out": shared_out,
            "clamp_shared": clamp_shared,
            "swiglu_limit": swiglu_limit,
        },
    )


def _fp8_group_quant_dequant(
    x: torch.Tensor, group_size: int, *, fused_intermediate: bool = False
) -> torch.Tensor:
    shape = x.shape
    blocks = x.float().view(-1, group_size)
    amax = blocks.abs().amax(dim=1)
    scale = (
        fp4_utils.f32_to_fused_moe_mxfp8_scale(amax)
        if fused_intermediate
        else fp4_utils.f32_to_mx_e8m0_scale(amax, dtype=MxDtypeInt.FP8_E4M3)
    )
    scale_f32 = fp4_utils.e8m0_to_f32(scale).view(-1, 1)
    quant = (blocks / scale_f32).to(torch.float8_e4m3fn)
    return (quant.float() * scale_f32).view(shape)


def _fp4_group_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    quant, scale = per_1x32_f4_quant(x)
    scale_f32 = fp4_utils.e8m0_to_f32(scale).repeat_interleave(32, dim=-1)
    return fp4_utils.mxfp4_to_f32(quant) * scale_f32


def _activation_quant_dequant(
    x: torch.Tensor, quantization: str, *, fused_intermediate: bool = False
) -> torch.Tensor:
    if quantization == "fp4":
        return _fp4_group_quant_dequant(x)
    if quantization == "fp8":
        return _fp8_group_quant_dequant(x, 32, fused_intermediate=fused_intermediate)
    raise ValueError(f"Unsupported activation quantization: {quantization}")


def _dequant_fp8_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale_f32 = fp4_utils.e8m0_to_f32(scale)
    scale_f32 = scale_f32.repeat_interleave(32, dim=-1)
    return weight.float() * scale_f32


def _dequant_fp4_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale_f32 = fp4_utils.e8m0_to_f32(scale)
    scale_f32 = scale_f32.repeat_interleave(32, dim=-1)
    return fp4_utils.mxfp4_to_f32(weight) * scale_f32


def _torch_routed_reference(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_ids: torch.Tensor,
    weights: _Weights,
    profile: _Profile,
    activation_quantization: str = "fp8",
    swiglu_limit: float = 10.0,
) -> torch.Tensor:
    """Dequantized FP32 reference for routed MXFP4 or MXFP8 experts."""
    dequant_weight = (
        _dequant_fp8_weight
        if weights.raw_routed_w1.dtype == dtypes.fp8
        else _dequant_fp4_weight
    )
    w1 = dequant_weight(weights.raw_routed_w1, weights.raw_routed_s1)
    w2 = dequant_weight(weights.raw_routed_w2, weights.raw_routed_s2)
    x = _activation_quant_dequant(hidden.float(), activation_quantization)
    expanded_x = x[:, None, :].expand(-1, profile.routed_topk, -1)
    slot_out = torch.zeros(
        (*routed_ids.shape, profile.hidden),
        dtype=dtypes.fp32,
        device=hidden.device,
    )
    activation_limit = runtime_swiglu_limit(swiglu_limit, "silu")

    for weight_index, expert_id in enumerate(weights.routed_expert_ids):
        mask = routed_ids == expert_id
        if not mask.any():
            continue
        gate_up = F.linear(expanded_x[mask], w1[weight_index])
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=activation_limit)
        up = up.clamp(min=-activation_limit, max=activation_limit)
        inter = F.silu(gate) * up
        inter = _activation_quant_dequant(
            inter,
            activation_quantization,
            fused_intermediate=activation_quantization == "fp8",
        )
        slot_out[mask] = F.linear(inter, w2[weight_index])

    return (slot_out * routed_weight[..., None]).sum(dim=1)


def _torch_shared_reference(
    hidden: torch.Tensor,
    weights: _Weights,
    group_size: int | str | None,
    swiglu_limit: float,
) -> torch.Tensor:
    w1 = _dequant_fp8_weight(weights.raw_native_w1, weights.raw_native_s1)
    w2 = _dequant_fp8_weight(weights.raw_native_w2, weights.raw_native_s2)
    x = hidden.float()
    if group_size == "fp4":
        x = _fp4_group_quant_dequant(x)
    elif group_size is not None:
        x = _fp8_group_quant_dequant(x, group_size)
    gate_up = F.linear(x, w1[0])
    if group_size is not None:
        gate_up = gate_up.to(dtypes.bf16).float()
    gate, up = gate_up.chunk(2, dim=-1)
    activation_limit = runtime_swiglu_limit(swiglu_limit, "silu")
    gate = gate.clamp(max=activation_limit)
    up = up.clamp(min=-activation_limit, max=activation_limit)
    inter = F.silu(gate) * up
    if group_size == "fp4":
        inter = _fp4_group_quant_dequant(inter)
    elif group_size is not None:
        inter = inter.to(dtypes.bf16).float()
        inter = _fp8_group_quant_dequant(inter, group_size, fused_intermediate=True)
    return F.linear(inter, w2[0])


def _torch_heterogeneous_reference(
    profile: _Profile,
    weights: _Weights,
    context: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    routed = _torch_routed_reference(
        context["hidden"],
        context["routed_weight"],
        context["routed_ids"],
        weights,
        profile,
        swiglu_limit=context["swiglu_limit"],
    )
    shared = _torch_shared_reference(
        context["hidden"],
        weights,
        32,
        _shared_swiglu_limit(
            context["swiglu_limit"], clamp_shared=context["clamp_shared"]
        ),
    )
    return routed + shared, routed, shared


@pytest.mark.parametrize("m", _m_values())
def test_heterogeneous_moe_matches_precision_oracles(
    monkeypatch: pytest.MonkeyPatch,
    weights: _Weights,
    m: int,
):
    profile = _profile()
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")
    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "1")
    forced, unfused, context = _run_composed_oracle(
        profile,
        weights,
        m,
        clamp_shared=not _use_hy4_profile(),
    )
    stress_repeats = int(os.environ.get("AITER_HETERO_MOE_STRESS_REPEATS", "1"))
    assert stress_repeats > 0
    for repeat in range(stress_repeats):
        forced_repeat = fused_moe(
            context["hidden"],
            weights.routed_w1,
            weights.routed_w2,
            context["all_weight"],
            context["all_ids"],
            **context["hetero_kwargs"],
        )
        assert torch.equal(
            forced, forced_repeat
        ), f"forced-reduce output changed on repeat {repeat + 1}"

    # A caller-provided buffer is the tensor returned, and holds the result.
    out_buf = torch.full_like(forced, -7.0)
    forced_into_buf = fused_moe(
        context["hidden"],
        weights.routed_w1,
        weights.routed_w2,
        context["all_weight"],
        context["all_ids"],
        **context["hetero_kwargs"],
        output=out_buf,
    )
    assert forced_into_buf is out_buf, "output buffer was not the tensor returned"
    assert torch.equal(forced, out_buf), "output buffer holds a different result"

    high_precision, routed_high, shared_high = _torch_heterogeneous_reference(
        profile, weights, context
    )

    assert torch.isfinite(forced).all()
    forced_error = _rel_l2(forced, high_precision)
    unfused_error = _rel_l2(unfused, high_precision)
    forced_cosine = _cosine_distance(forced, high_precision)
    assert forced_error <= 4e-2, f"forced-reduce FP32 error: {forced_error:.3e}"
    assert forced_error <= unfused_error + 4e-3, (
        "heterogeneous error regressed against separately scheduled routed and "
        f"native-shared kernels: {forced_error=:.3e}, {unfused_error=:.3e}"
    )
    assert (
        forced_cosine <= 1e-3
    ), f"forced-reduce FP32 cosine distance: {forced_cosine:.3e}"

    half_shared_weight = context["all_weight"].clone()
    half_shared_weight[:, profile.routed_topk] = 0.5
    half_shared = fused_moe(
        context["hidden"],
        weights.routed_w1,
        weights.routed_w2,
        half_shared_weight,
        context["all_ids"],
        **context["hetero_kwargs"],
    )
    observed_shared = 2 * (forced.float() - half_shared.float())
    shared_error = _rel_l2(observed_shared, shared_high)
    assert (
        shared_error <= 4e-2
    ), f"same-schedule native-shared contribution error: {shared_error:.3e}"

    half_routed_weight = context["all_weight"].clone()
    half_routed_weight[:, : profile.routed_topk] *= 0.5
    half_routed = fused_moe(
        context["hidden"],
        weights.routed_w1,
        weights.routed_w2,
        half_routed_weight,
        context["all_ids"],
        **context["hetero_kwargs"],
    )
    observed_routed = 2 * (forced.float() - half_routed.float())
    routed_error = _rel_l2(observed_routed, routed_high)
    assert (
        routed_error <= 4e-2
    ), f"same-schedule routed contribution error: {routed_error:.3e}"

    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "0")
    atomic_errors = []
    atomic_variations = []
    atomic_first = None
    for _ in range(max(3, stress_repeats)):
        atomic = fused_moe(
            context["hidden"],
            weights.routed_w1,
            weights.routed_w2,
            context["all_weight"],
            context["all_ids"],
            **context["hetero_kwargs"],
        )
        assert torch.isfinite(atomic).all()
        atomic_errors.append(_rel_l2(atomic, forced))
        if atomic_first is None:
            atomic_first = atomic
        else:
            atomic_variations.append(_rel_l2(atomic, atomic_first))
    assert (
        max(atomic_errors) <= 8e-3
    ), f"atomic output escaped the forced-reduce envelope: {atomic_errors}"
    assert max(atomic_variations, default=0.0) <= 8e-3, (
        "default atomic run-to-run variation escaped its precision envelope: "
        f"{atomic_variations}"
    )
    assert atomic_first is not None
    atomic_error = _rel_l2(atomic_first, high_precision)
    atomic_cosine = _cosine_distance(atomic_first, high_precision)
    assert atomic_error <= 4e-2, f"default atomic FP32 error: {atomic_error:.3e}"
    assert atomic_error <= unfused_error + 4e-3, (
        "default atomic error regressed against separately scheduled routed and "
        f"native-shared kernels: {atomic_error=:.3e}, {unfused_error=:.3e}"
    )
    assert (
        atomic_cosine <= 1e-3
    ), f"default atomic FP32 cosine distance: {atomic_cosine:.3e}"

    shared_limit = _shared_swiglu_limit(
        context["swiglu_limit"], clamp_shared=context["clamp_shared"]
    )
    golden = _torch_shared_reference(context["hidden"], weights, None, shared_limit)
    per_32 = _torch_shared_reference(context["hidden"], weights, 32, shared_limit)
    per_128 = _torch_shared_reference(context["hidden"], weights, 128, shared_limit)
    error_32 = _rel_l2(per_32, golden)
    error_128 = _rel_l2(per_128, golden)
    assert error_32 <= error_128 + 5e-4, (
        "per-32 heterogeneous activation quantization regressed against native "
        f"per-128 FP8: {error_32=:.3e}, {error_128=:.3e}"
    )


def test_mxfp8_routed_fp8_shared_heterogeneous_path(
    monkeypatch: pytest.MonkeyPatch,
    mxfp8_weights: _Weights,
):
    """Exercise HY4 MXFP8 routed and FP8 shared experts across the ID range."""
    profile = _mxfp8_profile()
    assert mxfp8_weights.routed_expert_ids == (0, 1, 7, 8, 127, 128, 254, 255)
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")
    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "1")

    fused, composed, context = _run_composed_oracle(
        profile, mxfp8_weights, 4, clamp_shared=False
    )
    high_precision, _, _ = _torch_heterogeneous_reference(
        profile, mxfp8_weights, context
    )

    assert torch.isfinite(fused).all()
    fused_error = _rel_l2(fused, high_precision)
    composed_error = _rel_l2(composed, high_precision)
    cosine = _cosine_distance(fused, high_precision)
    assert fused_error <= 4e-2, f"MXFP8 FHMoE FP32 error: {fused_error:.3e}"
    # The fused and composed paths can select different tiles and reduction
    # orders. Keep a comparative guard, but allow normal MXFP8 accumulation
    # variance while the independent FP32 and cosine limits enforce accuracy.
    assert fused_error <= composed_error + 2e-2, (
        "MXFP8 FHMoE error regressed against separately scheduled routed and "
        f"shared kernels: {fused_error=:.3e}, {composed_error=:.3e}"
    )
    assert cosine <= 1e-3, f"MXFP8 FHMoE cosine distance: {cosine:.3e}"

    # The final routed row is a dummy placeholder for the separately supplied
    # shared expert. Corrupt it to prove shared tiles never read routed storage.
    dummy_w1 = mxfp8_weights.routed_w1[-1].clone()
    dummy_w2 = mxfp8_weights.routed_w2[-1].clone()
    try:
        mxfp8_weights.routed_w1[-1].fill_(1)
        mxfp8_weights.routed_w2[-1].fill_(-1)
        corrupted_dummy_output = fused_moe(
            context["hidden"],
            mxfp8_weights.routed_w1,
            mxfp8_weights.routed_w2,
            context["all_weight"],
            context["all_ids"],
            **context["hetero_kwargs"],
        )
    finally:
        mxfp8_weights.routed_w1[-1].copy_(dummy_w1)
        mxfp8_weights.routed_w2[-1].copy_(dummy_w2)
    assert torch.equal(fused, corrupted_dummy_output)


def test_mxfp8_hy4_shared_silu_is_unclamped(
    monkeypatch: pytest.MonkeyPatch,
    mxfp8_weights: _Weights,
):
    profile = _mxfp8_profile()
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")
    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "1")
    fused, composed, context = _run_composed_oracle(
        profile,
        mxfp8_weights,
        4,
        clamp_shared=False,
        swiglu_limit=0.25,
    )

    half_shared_weight = context["all_weight"].clone()
    half_shared_weight[:, profile.routed_topk] = 0.5
    half_shared = fused_moe(
        context["hidden"],
        mxfp8_weights.routed_w1,
        mxfp8_weights.routed_w2,
        half_shared_weight,
        context["all_ids"],
        **context["hetero_kwargs"],
    )
    observed_shared = 2 * (fused.float() - half_shared.float())
    unclamped = _torch_shared_reference(
        context["hidden"], mxfp8_weights, 32, float("inf")
    )
    clamped = _torch_shared_reference(context["hidden"], mxfp8_weights, 32, 0.25)

    assert _rel_l2(fused, composed) <= 4e-2
    assert _rel_l2(unclamped, clamped) >= 1e-2
    assert _rel_l2(observed_shared, unclamped) < _rel_l2(observed_shared, clamped)


def test_mxfp8_hy4_compatible_geometry_supports_explicit_shared_clamp(
    monkeypatch: pytest.MonkeyPatch,
    mxfp8_weights: _Weights,
):
    profile = _mxfp8_profile()
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")
    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "1")
    fused, _, context = _run_composed_oracle(
        profile,
        mxfp8_weights,
        4,
        clamp_shared=True,
        swiglu_limit=0.25,
    )
    expected, _, _ = _torch_heterogeneous_reference(profile, mxfp8_weights, context)

    assert torch.isfinite(fused).all()
    assert _rel_l2(fused, expected) <= 4e-2


def test_generic_shared_silu_is_clamped(
    monkeypatch: pytest.MonkeyPatch,
    generic_weights: _Weights,
):
    profile = _Profile(256, 128, 128, 9, 2)
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")
    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "1")
    fused, composed, context = _run_composed_oracle(
        profile,
        generic_weights,
        4,
        clamp_shared=True,
        swiglu_limit=0.25,
    )

    half_shared_weight = context["all_weight"].clone()
    half_shared_weight[:, profile.routed_topk] = 0.5
    half_shared = fused_moe(
        context["hidden"],
        generic_weights.routed_w1,
        generic_weights.routed_w2,
        half_shared_weight,
        context["all_ids"],
        **context["hetero_kwargs"],
    )
    observed_shared = 2 * (fused.float() - half_shared.float())
    clamped = _torch_shared_reference(context["hidden"], generic_weights, 32, 0.25)
    unclamped = _torch_shared_reference(
        context["hidden"], generic_weights, 32, float("inf")
    )

    assert _rel_l2(fused, composed) <= 4e-2
    assert _rel_l2(clamped, unclamped) >= 1e-2
    assert _rel_l2(observed_shared, clamped) < _rel_l2(observed_shared, unclamped)


@pytest.mark.parametrize(
    ("m", "stage1_fragment", "stage2_fragment"),
    (
        (1, "_kw2", "_atomic"),
        (2, "_kw2", "_atomic"),
        (4, "_t32", "_atomic"),
        (8, "_xcd4", "_atomic"),
        (16, "_t32x128", "_atomic"),
        (32, "_t32x64", "_t32x128x256_atomic"),
        (64, "_t32x128", "_atomic"),
        (128, "_w2", "_atomic"),
        (256, "_t32x64", "_atomic"),
        (512, "_t32x64", "_atomic"),
        (1024, "_t64", "_t64x128x256_atomic"),
        (2048, "_t128", "_atomic_sbm128"),
        (4096, "_t64", "_reduce"),
        (8192, "_t128", "_reduce"),
        (16384, "_t128", "_reduce"),
        (32768, "_t128", "_reduce"),
        (38836, "_t128", "_reduce"),
    ),
)
def test_mxfp8_production_decode_and_prefill_match_sampled_fp32(
    monkeypatch: pytest.MonkeyPatch,
    mxfp8_weights: _Weights,
    m: int,
    stage1_fragment: str,
    stage2_fragment: str,
):
    """Exercise every tuned HY4 bucket and its representative runtime shapes."""
    import importlib

    fused_moe_module = importlib.import_module("aiter.fused_moe")
    from aiter.jit.core import AITER_CONFIGS

    profile = _mxfp8_profile()
    config_path = Path(AITER_CONFIGS.AITER_CONFIG_FHMOE_FILE)
    metadata = fused_moe_module.get_2stage_cfgs(
        fused_moe_module.get_padded_M(m),
        profile.hidden,
        profile.inter,
        profile.experts,
        profile.routed_topk + 1,
        torch.bfloat16,
        dtypes.fp8,
        dtypes.fp8,
        aiter.QuantType.per_1x32,
        True,
        aiter.ActivationType.Silu,
        False,
        0,
        0,
        True,
        GateMode.INTERLEAVE,
        config_file=str(config_path),
    )
    assert stage1_fragment in metadata.stage1.keywords["kernelName"]
    assert stage2_fragment in metadata.stage2.keywords["kernelName"]

    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")
    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "0")
    hidden, routed_weight, routed_ids, all_weight, all_ids = _route_inputs(
        profile,
        m,
        mxfp8_weights.routed_w1.device,
        mxfp8_weights.routed_expert_ids,
    )
    kwargs = _common_kwargs(profile, mxfp8_weights.routed_s1, mxfp8_weights.routed_s2)
    kwargs.update(
        shared_w1=mxfp8_weights.shared_w1,
        shared_w2=mxfp8_weights.shared_w2,
        shared_w1_scale=mxfp8_weights.shared_s1,
        shared_w2_scale=mxfp8_weights.shared_s2,
        shared_expert_id=profile.shared_id,
        clamp_shared=False,
    )
    actual = fused_moe(
        hidden,
        mxfp8_weights.routed_w1,
        mxfp8_weights.routed_w2,
        all_weight,
        all_ids,
        **kwargs,
    )
    if stage2_fragment == "_atomic":
        out_buf = torch.full_like(actual, -7)
        buffered = fused_moe(
            hidden,
            mxfp8_weights.routed_w1,
            mxfp8_weights.routed_w2,
            all_weight,
            all_ids,
            output=out_buf,
            **kwargs,
        )
        assert buffered is out_buf
        assert torch.isfinite(buffered).all()
        assert _rel_l2(buffered, actual) <= 8e-3

    sample_rows = torch.tensor(
        sorted({0, m // 2, m - 1}), dtype=torch.long, device=hidden.device
    )
    sample_context = {
        "hidden": hidden.index_select(0, sample_rows),
        "routed_weight": routed_weight.index_select(0, sample_rows),
        "routed_ids": routed_ids.index_select(0, sample_rows),
        "clamp_shared": False,
        "swiglu_limit": _swiglu_limit(profile),
    }
    expected, _, _ = _torch_heterogeneous_reference(
        profile, mxfp8_weights, sample_context
    )
    actual_sample = actual.index_select(0, sample_rows)

    assert torch.isfinite(actual_sample).all()
    assert _rel_l2(actual_sample, expected) <= 4e-2
    assert _cosine_distance(actual_sample, expected) <= 1e-3


def test_dsv4_production_reduce_matches_fp32(
    monkeypatch: pytest.MonkeyPatch,
    dsv4_weights: _Weights,
):
    profile = _Profile(7168, 384, 384, 385, 6)
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")
    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "0")
    fused, composed, context = _run_composed_oracle(
        profile, dsv4_weights, 4, clamp_shared=True
    )
    expected, _, _ = _torch_heterogeneous_reference(profile, dsv4_weights, context)

    assert torch.isfinite(fused).all()
    assert _rel_l2(fused, expected) <= 5e-2
    assert _rel_l2(fused, expected) <= _rel_l2(composed, expected) + 1e-2


def test_mxfp8_shared_k_offsets_cover_multiple_tiles(
    monkeypatch: pytest.MonkeyPatch,
    mxfp8_weights: _Weights,
):
    """Exercise multiple K tiles in both stages with the production HY4 shape.

    M remains small intentionally: stage-1 H=6144 and stage-2 I=256, rather
    than token count, are what force shared-weight loads beyond the first tile.
    """
    profile = _mxfp8_profile()
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")
    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "1")

    fused, composed, _ = _run_composed_oracle(
        profile, mxfp8_weights, 4, clamp_shared=False
    )

    assert torch.isfinite(fused).all()
    assert _rel_l2(fused, composed) <= 4e-2


def test_mxfp8_heterogeneous_path_requires_interleaved_gate(
    mxfp8_weights: _Weights,
):
    profile = _mxfp8_profile()
    hidden, _, _, all_weight, all_ids = _route_inputs(
        profile,
        4,
        mxfp8_weights.routed_w1.device,
        mxfp8_weights.routed_expert_ids,
    )
    kwargs = _common_kwargs(
        profile,
        mxfp8_weights.routed_s1,
        mxfp8_weights.routed_s2,
        gate_mode=GateMode.SEPARATED,
    )
    kwargs.update(
        shared_w1=mxfp8_weights.shared_w1,
        shared_w2=mxfp8_weights.shared_w2,
        shared_w1_scale=mxfp8_weights.shared_s1,
        shared_w2_scale=mxfp8_weights.shared_s2,
        shared_expert_id=profile.shared_id,
        clamp_shared=False,
    )

    with pytest.raises(ValueError, match="require interleaved gate/up layout"):
        fused_moe(
            hidden,
            mxfp8_weights.routed_w1,
            mxfp8_weights.routed_w2,
            all_weight,
            all_ids,
            **kwargs,
        )


def test_mxfp8_heterogeneous_path_supports_non_hy4_contract():
    profile = _Profile(256, 128, 128, 9, 2)
    weights = _build_weights(profile, interleave=True, routed_mxfp8=True)
    fused, composed, _ = _run_composed_oracle(profile, weights, 4, clamp_shared=True)

    assert torch.isfinite(fused).all()
    assert _rel_l2(fused, composed) <= 4e-2


def test_zero_swiglu_limit_references_are_unclamped(generic_weights: _Weights):
    profile = _Profile(256, 128, 128, 9, 2)
    hidden, routed_weight, routed_ids, _, _ = _route_inputs(
        profile,
        4,
        generic_weights.routed_w1.device,
        generic_weights.routed_expert_ids,
    )
    routed_zero = _torch_routed_reference(
        hidden,
        routed_weight,
        routed_ids,
        generic_weights,
        profile,
        swiglu_limit=0,
    )
    routed_unclamped = _torch_routed_reference(
        hidden,
        routed_weight,
        routed_ids,
        generic_weights,
        profile,
        swiglu_limit=float("inf"),
    )
    shared_zero = _torch_shared_reference(hidden, generic_weights, 32, 0)
    shared_unclamped = _torch_shared_reference(
        hidden, generic_weights, 32, float("inf")
    )

    torch.testing.assert_close(routed_zero, routed_unclamped)
    torch.testing.assert_close(shared_zero, shared_unclamped)


def test_no_shared_explicit_defaults_preserve_old_api_output(
    monkeypatch: pytest.MonkeyPatch,
    weights: _Weights,
):
    profile = _profile()
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")
    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "1")
    hidden, routed_weight, routed_ids, _, _ = _route_inputs(
        profile, 4, weights.routed_w1.device
    )
    routed_e = profile.experts - 1
    routed_w1 = _mark_shuffled(weights.routed_w1[:routed_e])
    routed_w2 = _mark_shuffled(weights.routed_w2[:routed_e])
    routed_s1 = weights.routed_s1[: routed_e * 2 * profile.inter]
    routed_s2 = weights.routed_s2[: routed_e * profile.hidden]
    kwargs = _common_kwargs(profile, routed_s1, routed_s2)

    omitted = fused_moe(
        hidden,
        routed_w1,
        routed_w2,
        routed_weight,
        routed_ids,
        **kwargs,
    )
    explicit = fused_moe(
        hidden,
        routed_w1,
        routed_w2,
        routed_weight,
        routed_ids,
        shared_w1=None,
        shared_w2=None,
        shared_w1_scale=None,
        shared_w2_scale=None,
        shared_expert_id=-1,
        **kwargs,
    )
    assert torch.equal(omitted, explicit)


def test_heterogeneous_moe_uses_a_separate_custom_op_schema():
    from aiter.fhmoe import fhmoe_
    from aiter.ops.flydsl.fhmoe import (
        flydsl_fhmoe_stage1,
        flydsl_fhmoe_stage2,
    )
    from aiter.ops.flydsl.kernels.fhmoe import (
        compile_mixed_fhmoe_gemm1,
        compile_mixed_fhmoe_gemm2,
    )
    from aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage import (
        compile_mixed_moe_gemm1,
        compile_mixed_moe_gemm2,
    )
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    assert callable(fhmoe_)
    legacy_schema = torch.ops.aiter.fused_moe_.default._schema
    fhmoe_schema = torch.ops.aiter.fhmoe_.default._schema
    fhmoe_fields = {
        "shared_w1",
        "shared_w2",
        "shared_w1_scale",
        "shared_w2_scale",
        "shared_expert_id",
        "clamp_shared",
    }
    legacy_schema_fields = {argument.name for argument in legacy_schema.arguments}
    fhmoe_schema_fields = {argument.name for argument in fhmoe_schema.arguments}

    assert fhmoe_fields.isdisjoint(legacy_schema_fields)
    assert fhmoe_fields <= fhmoe_schema_fields

    ordinary_apis = (
        flydsl_moe_stage1,
        flydsl_moe_stage2,
        compile_mixed_moe_gemm1,
        compile_mixed_moe_gemm2,
    )
    assert all(
        fhmoe_fields.isdisjoint(inspect.signature(api).parameters)
        for api in ordinary_apis
    )
    assert {"situ_beta", "situ_linear_beta", "k_batch_intra_block"} <= set(
        inspect.signature(flydsl_moe_stage1).parameters
    )
    assert {"shared_w1", "shared_w1_scale", "shared_expert_id"} <= set(
        inspect.signature(flydsl_fhmoe_stage1).parameters
    )
    assert {"shared_w2", "shared_w2_scale", "shared_expert_id"} <= set(
        inspect.signature(flydsl_fhmoe_stage2).parameters
    )
    assert all(
        "shared_expert_id" in inspect.signature(api).parameters
        for api in (compile_mixed_fhmoe_gemm1, compile_mixed_fhmoe_gemm2)
    )
    assert all(
        "v2_output_layout" in inspect.signature(api).parameters
        for api in (flydsl_fhmoe_stage1, compile_mixed_fhmoe_gemm1)
    )
    assert all(
        "clamp_shared" in inspect.signature(api).parameters
        for api in (flydsl_fhmoe_stage1, compile_mixed_fhmoe_gemm1)
    )
    fhmoe_apis = (
        flydsl_fhmoe_stage1,
        flydsl_fhmoe_stage2,
        compile_mixed_fhmoe_gemm1,
        compile_mixed_fhmoe_gemm2,
    )
    assert all("xcd_swizzle" in inspect.signature(api).parameters for api in fhmoe_apis)


def test_fhmoe_runtime_compile_bridge_forwards_xcd(monkeypatch: pytest.MonkeyPatch):
    from aiter.ops.flydsl import fhmoe

    tensor = torch.empty(0)
    compile_calls = []

    def invoke_compiler(**kwargs):
        compile_kwargs = {"xcd_swizzle": kwargs["xcd_swizzle"]}
        if "v2_output_layout" in kwargs:
            compile_kwargs["v2_output_layout"] = kwargs["v2_output_layout"]
        return kwargs["_compile_kernel"](**compile_kwargs)

    def compile_stage1(**kwargs):
        compile_calls.append((1, kwargs))
        return 1

    def compile_stage2(**kwargs):
        compile_calls.append((2, kwargs))
        return 2

    monkeypatch.setattr(fhmoe, "_flydsl_moe_stage1_impl", invoke_compiler)
    monkeypatch.setattr(fhmoe, "_flydsl_moe_stage2_impl", invoke_compiler)
    monkeypatch.setattr(fhmoe, "compile_flydsl_fhmoe_stage1", compile_stage1)
    monkeypatch.setattr(fhmoe, "compile_flydsl_fhmoe_stage2", compile_stage2)

    assert (
        fhmoe.flydsl_fhmoe_stage1(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            shared_w1=tensor,
            shared_w1_scale=tensor,
            shared_expert_id=8,
            clamp_shared=False,
            xcd_swizzle=4,
            v2_output_layout=True,
        )
        == 1
    )
    assert (
        fhmoe.flydsl_fhmoe_stage2(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            shared_w2=tensor,
            shared_w2_scale=tensor,
            shared_expert_id=8,
            xcd_swizzle=4,
        )
        == 2
    )
    assert compile_calls == [
        (
            1,
            {
                "shared_expert_id": 8,
                "clamp_shared": False,
                "xcd_swizzle": 4,
                "v2_output_layout": True,
            },
        ),
        (2, {"shared_expert_id": 8, "xcd_swizzle": 4}),
    ]


def test_fhmoe_rejects_unclamped_splitk_with_finite_limit():
    from aiter.ops.flydsl.fhmoe import flydsl_fhmoe_stage1

    tensor = torch.empty(0)
    with pytest.raises(
        NotImplementedError,
        match="cannot clamp routed SiLU while leaving the shared expert unclamped",
    ):
        flydsl_fhmoe_stage1(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            out_dtype="fp8",
            act="silu",
            k_batch=2,
            gate_mode="interleave",
            swiglu_limit=10.0,
            shared_w1=tensor,
            shared_w1_scale=tensor,
            shared_expert_id=8,
            clamp_shared=False,
        )


@pytest.mark.parametrize("act", ("swiglu", "situv2"))
def test_fhmoe_rejects_unclamped_shared_expert_for_non_silu(act: str):
    from aiter.ops.flydsl.fhmoe import (
        compile_flydsl_fhmoe_stage1,
        flydsl_fhmoe_stage1,
    )

    error = "clamp_shared=False is supported only with act='silu'"
    with pytest.raises(ValueError, match=error):
        compile_flydsl_fhmoe_stage1(
            model_dim=256,
            inter_dim=128,
            experts=9,
            topk=3,
            tile_m=32,
            tile_n=128,
            tile_k=256,
            doweight_stage1=False,
            a_dtype="fp8",
            b_dtype="fp4",
            out_dtype="fp8",
            act=act,
            shared_expert_id=8,
            clamp_shared=False,
        )

    tensor = torch.empty(0)
    with pytest.raises(ValueError, match=error):
        flydsl_fhmoe_stage1(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            act=act,
            shared_w1=tensor,
            shared_w1_scale=tensor,
            shared_expert_id=8,
            clamp_shared=False,
        )


def test_low_level_fhmoe_rejects_separated_mxfp8():
    from aiter.ops.flydsl.fhmoe import (
        compile_flydsl_fhmoe_stage1,
        flydsl_fhmoe_stage1,
    )

    with pytest.raises(ValueError, match="interleaved gate/up"):
        compile_flydsl_fhmoe_stage1(
            model_dim=256,
            inter_dim=128,
            experts=9,
            topk=3,
            tile_m=32,
            tile_n=128,
            tile_k=256,
            doweight_stage1=False,
            a_dtype="fp8",
            b_dtype="fp8",
            out_dtype="fp8",
            gate_mode="separated",
            shared_expert_id=8,
        )

    tensor = torch.empty(0)
    with pytest.raises(ValueError, match="interleaved gate/up"):
        flydsl_fhmoe_stage1(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            a_dtype="fp8",
            b_dtype="fp8",
            out_dtype="fp8",
            gate_mode="separated",
            shared_w1=tensor,
            shared_w1_scale=tensor,
            shared_expert_id=8,
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("stage2_scatter", object()),
        ("quant_type_a", aiter.QuantType.per_1x32),
        ("quant_dtype_a", dtypes.fp8),
        ("quant_dtype_a2", dtypes.fp8),
        ("a1_scale", torch.ones(1)),
        ("a2_scale", torch.ones(1)),
    ),
)
def test_fhmoe_rejects_dropped_public_controls(argument: str, value):
    tensor = torch.empty(0)
    kwargs = {
        "shared_w1": tensor,
        "shared_w2": tensor,
        "shared_w1_scale": tensor,
        "shared_w2_scale": tensor,
        "shared_expert_id": 0,
        argument: value,
    }
    with pytest.raises(NotImplementedError, match=argument):
        fused_moe(tensor, tensor, tensor, tensor, tensor, **kwargs)


@pytest.mark.parametrize(
    ("gate_mode", "quant_dtype_a"),
    (
        (GateMode.SEPARATED.value, dtypes.fp4x2),
        (GateMode.INTERLEAVE.value, dtypes.fp8),
    ),
)
def test_fhmoe_accepts_matching_public_activation_dtype(
    monkeypatch: pytest.MonkeyPatch,
    gate_mode: str,
    quant_dtype_a: torch.dtype,
):
    import aiter.fhmoe as fhmoe_module

    captured = {}

    def fake_fhmoe(**kwargs):
        captured.update(kwargs)
        return torch.empty(0)

    monkeypatch.setattr(fhmoe_module, "_fhmoe", fake_fhmoe)
    tensor = torch.empty(0)
    tensor.is_shuffled = True

    fused_moe(
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        gate_mode=gate_mode,
        shared_w1=tensor,
        shared_w2=tensor,
        shared_w1_scale=tensor,
        shared_w2_scale=tensor,
        shared_expert_id=0,
        quant_dtype_a=quant_dtype_a,
    )

    assert captured["gate_mode"] == gate_mode
    assert captured["clamp_shared"] is True


def test_hy4_semantics_require_explicit_unclamped_request():
    from aiter.fhmoe import _use_hy4_mxfp8_fhmoe_contract

    geometry = {
        "model_dim": 6144,
        "inter_dim": 256,
        "experts": 257,
        "topk": 9,
        "routed_mxfp8": True,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "gate_interleaved": True,
        "doweight_stage1": False,
        "shared_expert_id": 256,
    }

    assert _use_hy4_mxfp8_fhmoe_contract(
        clamp_shared=True,
        **geometry,
    )
    assert _use_hy4_mxfp8_fhmoe_contract(
        clamp_shared=False,
        **geometry,
    )

    with pytest.raises(ValueError, match="HY4-compatible MXFP8"):
        _use_hy4_mxfp8_fhmoe_contract(
            clamp_shared=False,
            **(geometry | {"model_dim": 4096}),
        )


def test_clamp_shared_preserves_existing_positional_argument_order():
    import aiter.fhmoe as fhmoe_module

    parameters = list(inspect.signature(fused_moe).parameters)
    assert parameters[-4:] == [
        "quant_type_a",
        "quant_dtype_a",
        "quant_dtype_a2",
        "clamp_shared",
    ]
    assert list(inspect.signature(fhmoe_module.fhmoe_fake).parameters)[-2:] == [
        "output",
        "clamp_shared",
    ]


def test_unclamped_request_requires_shared_expert_dispatch():
    with pytest.raises(ValueError, match="requires FHMoE"):
        fused_moe(None, None, None, None, None, clamp_shared=False)


def test_flydsl_stage2_classification_covers_plain_and_fhmoe_wrappers():
    import functools
    import importlib

    fhmoe_module = importlib.import_module("aiter.fhmoe")
    fused_moe_module = importlib.import_module("aiter.fused_moe")

    assert fused_moe_module._is_flydsl_stage2(fused_moe_module._flydsl_stage2_wrapper)
    assert fused_moe_module._is_flydsl_stage2(fhmoe_module._flydsl_fhmoe_stage2_wrapper)
    assert not fused_moe_module._is_flydsl_stage2(lambda: None)

    fp8_stage2 = functools.partial(
        fused_moe_module._flydsl_stage2_wrapper,
        kernelName="flydsl_moe2_afp8_wfp8_bf16_t32x128x128_atomic",
    )
    fhmoe_fp8_stage2 = functools.partial(
        fhmoe_module._flydsl_fhmoe_stage2_wrapper,
        kernelName="flydsl_moe2_afp8_wfp8_bf16_t32x128x128_atomic",
    )
    a16w4_stage2 = functools.partial(
        fused_moe_module._flydsl_stage2_wrapper,
        kernelName="flydsl_moe2_abf16_wfp4_bf16_t32x128x128_atomic",
    )
    a16wi4_stage2 = functools.partial(
        fused_moe_module._flydsl_stage2_wrapper,
        kernelName="flydsl_moe2_abf16_wint4_bf16_t16x128x128_reduce",
    )

    assert fused_moe_module._stage2_honors_forced_reduce(fp8_stage2)
    assert fused_moe_module._stage2_honors_forced_reduce(fhmoe_fp8_stage2)
    assert not fused_moe_module._stage2_honors_forced_reduce(a16w4_stage2)
    assert not fused_moe_module._stage2_honors_forced_reduce(a16wi4_stage2)
    unknown_stage2 = functools.partial(
        fused_moe_module._flydsl_stage2_wrapper,
        kernelName="flydsl_moe2_unknown",
    )
    with pytest.raises(ValueError, match="requires a recognized FlyDSL stage2"):
        fused_moe_module._stage2_honors_forced_reduce(unknown_stage2)


def test_fhmoe_sort_block_override_must_match_metadata(
    monkeypatch: pytest.MonkeyPatch,
    mxfp8_weights: _Weights,
):
    profile = _mxfp8_profile()
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")
    hidden, _, _, all_weight, all_ids = _route_inputs(
        profile, 1, mxfp8_weights.routed_w1.device, mxfp8_weights.routed_expert_ids
    )
    kwargs = _common_kwargs(profile, mxfp8_weights.routed_s1, mxfp8_weights.routed_s2)
    kwargs.update(
        shared_w1=mxfp8_weights.shared_w1,
        shared_w2=mxfp8_weights.shared_w2,
        shared_w1_scale=mxfp8_weights.shared_s1,
        shared_w2_scale=mxfp8_weights.shared_s2,
        shared_expert_id=profile.shared_id,
        clamp_shared=False,
    )

    matched = fused_moe(
        hidden,
        mxfp8_weights.routed_w1,
        mxfp8_weights.routed_w2,
        all_weight,
        all_ids,
        block_size_M=32,
        **kwargs,
    )
    assert torch.isfinite(matched).all()
    with pytest.raises(ValueError, match="does not match"):
        fused_moe(
            hidden,
            mxfp8_weights.routed_w1,
            mxfp8_weights.routed_w2,
            all_weight,
            all_ids,
            block_size_M=64,
            **kwargs,
        )


def test_fhmoe_rejects_unmarked_weight_layout():
    tensor = torch.empty(0)
    with pytest.raises(ValueError, match="is_shuffled=True"):
        fused_moe(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            shared_w1=tensor,
            shared_w2=tensor,
            shared_w1_scale=tensor,
            shared_w2_scale=tensor,
            shared_expert_id=0,
        )


def _mock_gfx950_fhmoe_metadata(monkeypatch: pytest.MonkeyPatch):
    import importlib

    fused_moe_module = importlib.import_module("aiter.fused_moe")
    monkeypatch.setattr(fused_moe_module, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fused_moe_module, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(dtypes, "fp8", torch.float8_e4m3fn)
    fused_moe_module.get_2stage_cfgs.cache_clear()
    fused_moe_module.cfg_2stages_by_file.clear()
    return fused_moe_module


def _mock_dsv4_i384_fhmoe_metadata(monkeypatch: pytest.MonkeyPatch):
    fused_moe_module = _mock_gfx950_fhmoe_metadata(monkeypatch)
    monkeypatch.delenv("AITER_BYPASS_TUNE_CONFIG", raising=False)
    return fused_moe_module


def test_gfx950_metadata_mock_normalizes_fp8_alias(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(dtypes, "fp8", torch.float8_e4m3fnuz)
    _mock_gfx950_fhmoe_metadata(monkeypatch)
    assert dtypes.fp8 == torch.float8_e4m3fn


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({}, True),
        ({"token_num": 2048}, True),
        ({"token_num": 0}, False),
        ({"token_num": 2049}, False),
        ({"model_dim": 7169}, False),
        ({"inter_dim": 512}, False),
        ({"experts": 384}, False),
        ({"topk": 6}, False),
        ({"hidden_pad": 128}, False),
        ({"intermediate_pad": 128}, False),
        ({"gate_mode": GateMode.SEPARATED}, False),
        ({"doweight_stage1": True}, False),
        ({"routed_mxfp4": False}, False),
    ),
)
def test_dsv4_i384_fhmoe_config_scope(
    monkeypatch: pytest.MonkeyPatch, overrides, expected
):
    from aiter.fhmoe import _uses_dsv4_fhmoe_config

    _mock_dsv4_i384_fhmoe_metadata(monkeypatch)
    args = {
        "token_num": 1,
        "model_dim": 7168,
        "inter_dim": 384,
        "experts": 385,
        "topk": 7,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "gate_mode": GateMode.INTERLEAVE,
        "doweight_stage1": False,
        "routed_mxfp4": True,
    }
    args.update(overrides)

    assert _uses_dsv4_fhmoe_config(**args) is expected


@pytest.mark.parametrize(
    ("max_tokens", "expected"),
    (
        (0, False),
        (1, True),
        (1536, True),
        (2048, True),
        (2049, False),
        (True, False),
    ),
)
def test_dsv4_i384_fhmoe_capability(
    monkeypatch: pytest.MonkeyPatch, max_tokens, expected
):
    from aiter.fhmoe import supports_dsv4_i384_fhmoe

    _mock_dsv4_i384_fhmoe_metadata(monkeypatch)
    assert supports_dsv4_i384_fhmoe(max_tokens) is expected


def test_dsv4_i384_fhmoe_capability_follows_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import csv
    import importlib

    fhmoe = importlib.import_module("aiter.fhmoe")
    _mock_dsv4_i384_fhmoe_metadata(monkeypatch)
    config_path = Path(__file__).resolve().parents[1] / "aiter/configs/tuned_fhmoe.csv"
    with config_path.open(newline="") as config:
        reader = csv.DictReader(config)
        rows = list(reader)
        assert reader.fieldnames is not None
        fieldnames = reader.fieldnames

    dsv4_rows = [row for row in rows if row["inter_dim"] == "384"]
    row_4096 = dict(dsv4_rows[-1])
    row_4096["token"] = "4096"
    complete_path = tmp_path / "complete.csv"
    with complete_path.open("w", newline="") as config:
        writer = csv.DictWriter(config, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([*rows, row_4096])

    monkeypatch.setattr(
        fhmoe, "_dsv4_i384_fhmoe_config_file", lambda: str(complete_path)
    )
    assert fhmoe.supports_dsv4_i384_fhmoe(2049)
    assert fhmoe.supports_dsv4_i384_fhmoe(4096)

    gap_path = tmp_path / "gap.csv"
    with gap_path.open("w", newline="") as config:
        writer = csv.DictWriter(config, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([row for row in [*rows, row_4096] if row["token"] != "16"])
    monkeypatch.setattr(fhmoe, "_dsv4_i384_fhmoe_config_file", lambda: str(gap_path))
    assert not fhmoe.supports_dsv4_i384_fhmoe(2048)
    assert not fhmoe.supports_dsv4_i384_fhmoe(4096)

    duplicate_path = tmp_path / "duplicate.csv"
    with duplicate_path.open("w", newline="") as config:
        writer = csv.DictWriter(config, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([*rows, dsv4_rows[-1]])
    monkeypatch.setattr(
        fhmoe, "_dsv4_i384_fhmoe_config_file", lambda: str(duplicate_path)
    )
    assert not fhmoe.supports_dsv4_i384_fhmoe(2048)

    invalid_path = tmp_path / "invalid.csv"
    invalid_rows = [dict(row) for row in rows]
    invalid_rows[4]["kernelName1"] = "flydsl_moe1_invalid"
    with invalid_path.open("w", newline="") as config:
        writer = csv.DictWriter(config, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(invalid_rows)
    monkeypatch.setattr(
        fhmoe, "_dsv4_i384_fhmoe_config_file", lambda: str(invalid_path)
    )
    assert not fhmoe.supports_dsv4_i384_fhmoe(2048)

    mismatched_path = tmp_path / "mismatched_block.csv"
    mismatched_rows = [dict(row) for row in rows]
    mismatched_row = next(
        row
        for row in mismatched_rows
        if row["inter_dim"] == "384" and row["token"] == "1"
    )
    mismatched_row["block_m"] = "64" if mismatched_row["block_m"] != "64" else "32"
    with mismatched_path.open("w", newline="") as config:
        writer = csv.DictWriter(config, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mismatched_rows)
    monkeypatch.setattr(
        fhmoe,
        "_dsv4_i384_fhmoe_config_file",
        lambda: str(mismatched_path),
    )
    assert not fhmoe.supports_dsv4_i384_fhmoe(1)


@pytest.mark.parametrize(
    "incompatible_kind",
    ("wfp8", "f16", "invalid_sbm", "invalid_kwave"),
)
def test_dsv4_i384_fhmoe_capability_rejects_incompatible_kernels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    incompatible_kind: str,
):
    import csv
    import importlib

    fhmoe = importlib.import_module("aiter.fhmoe")
    _mock_dsv4_i384_fhmoe_metadata(monkeypatch)
    source = Path(__file__).resolve().parents[1] / "aiter/configs/tuned_fhmoe.csv"
    with source.open(newline="") as config:
        reader = csv.DictReader(config)
        rows = list(reader)
        assert reader.fieldnames is not None
        fieldnames = reader.fieldnames

    dsv4 = next(
        row for row in rows if row["inter_dim"] == "384" and row["token"] == "1"
    )
    hy4 = next(row for row in rows if row["inter_dim"] == "256" and row["token"] == "1")
    if incompatible_kind == "wfp8":
        dsv4["kernelName1"] = hy4["kernelName1"]
        dsv4["kernelName2"] = hy4["kernelName2"]
    elif incompatible_kind == "f16":
        dsv4["kernelName2"] = dsv4["kernelName2"].replace("_bf16_", "_f16_")
    elif incompatible_kind == "invalid_sbm":
        dsv4["kernelName2"] = "flydsl_moe2_afp8_wfp4_bf16_t64x128x128_atomic_sbm32"
    else:
        dsv4["kernelName1"] = dsv4["kernelName1"].replace("_kw4_fp8", "_kw3_fp8")

    incompatible = tmp_path / f"{incompatible_kind}.csv"
    with incompatible.open("w", newline="") as config:
        writer = csv.DictWriter(config, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(
        fhmoe,
        "_dsv4_i384_fhmoe_config_file",
        lambda: str(incompatible),
    )
    fhmoe._supports_dsv4_i384_fhmoe_config.cache_clear()

    assert not fhmoe.supports_dsv4_i384_fhmoe(1)


def test_dsv4_i384_fhmoe_capability_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import importlib

    fhmoe = importlib.import_module("aiter.fhmoe")
    _mock_dsv4_i384_fhmoe_metadata(monkeypatch)
    monkeypatch.setattr(
        fhmoe,
        "_dsv4_i384_fhmoe_config_file",
        lambda: str(tmp_path / "missing.csv"),
    )
    assert not fhmoe.supports_dsv4_i384_fhmoe(1)


@pytest.mark.parametrize("bypass", ("1", "2", "invalid"))
def test_dsv4_i384_fhmoe_capability_rejects_config_bypass(
    monkeypatch: pytest.MonkeyPatch, bypass: str
):
    import importlib

    fhmoe = importlib.import_module("aiter.fhmoe")
    config_path = Path(__file__).resolve().parents[1] / "aiter/configs/tuned_fhmoe.csv"
    monkeypatch.setattr(fhmoe, "_dsv4_i384_fhmoe_config_file", lambda: str(config_path))
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", bypass)
    assert not fhmoe.supports_dsv4_i384_fhmoe(1)


@pytest.mark.parametrize(
    ("num_tokens", "expected_stage1", "expected_stage2"),
    (
        (
            1,
            "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_kw4_fp8",
            "flydsl_moe2_afp8_wfp4_bf16_t32x256x128_atomic",
        ),
        (
            4,
            "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_kw2",
            "flydsl_moe2_afp8_wfp4_bf16_t32x128x128_reduce_persist",
        ),
        (
            8,
            "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w3_gui",
            "flydsl_moe2_afp8_wfp4_bf16_t32x256x128_reduce_bnt2_persist",
        ),
        (
            16,
            "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w2_gui_fp8",
            "flydsl_moe2_afp8_wfp4_bf16_t32x128x128_atomic_bnt2",
        ),
        (
            512,
            "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8",
            "flydsl_moe2_afp8_wfp4_bf16_t32x256x128_atomic_bnt2_persist",
        ),
        (
            1536,
            "flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w3_bnt0_gui",
            "flydsl_moe2_afp8_wfp4_bf16_t64x128x128_atomic",
        ),
        (
            2048,
            "flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w3_bnt0_gui",
            "flydsl_moe2_afp8_wfp4_bf16_t64x128x128_atomic",
        ),
    ),
)
def test_dsv4_i384_fhmoe_uses_dedicated_config(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    expected_stage1: str,
    expected_stage2: str,
):
    fused_moe_module = _mock_gfx950_fhmoe_metadata(monkeypatch)

    config_path = Path(__file__).resolve().parents[1] / "aiter/configs/tuned_fhmoe.csv"

    metadata = fused_moe_module.get_2stage_cfgs(
        fused_moe_module.get_padded_M(num_tokens),
        7168,
        384,
        385,
        7,
        torch.bfloat16,
        dtypes.fp8,
        dtypes.fp4x2,
        aiter.QuantType.per_1x32,
        True,
        aiter.ActivationType.Silu,
        False,
        0,
        0,
        True,
        GateMode.INTERLEAVE,
        config_file=str(config_path),
    )

    assert metadata.stage1.keywords["kernelName"] == expected_stage1
    assert metadata.stage2.keywords["kernelName"] == expected_stage2


@pytest.mark.parametrize(
    ("num_tokens", "expected_block_m"),
    (
        (1, 32),
        (512, 32),
        (4096, 64),
        (16384, 128),
        (32768, 128),
        (38836, 128),
    ),
)
def test_hy4_mxfp8_fhmoe_uses_dedicated_config(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    expected_block_m: int,
):
    fused_moe_module = _mock_gfx950_fhmoe_metadata(monkeypatch)
    config_path = Path(__file__).resolve().parents[1] / "aiter/configs/tuned_fhmoe.csv"

    metadata = fused_moe_module.get_2stage_cfgs(
        fused_moe_module.get_padded_M(num_tokens),
        6144,
        256,
        257,
        9,
        torch.bfloat16,
        dtypes.fp8,
        dtypes.fp8,
        aiter.QuantType.per_1x32,
        True,
        aiter.ActivationType.Silu,
        False,
        0,
        0,
        True,
        GateMode.INTERLEAVE,
        config_file=str(config_path),
    )

    assert metadata.stage1.keywords["kernelName"].startswith("flydsl_moe1_afp8_wfp8_")
    assert metadata.stage2.keywords["kernelName"].startswith("flydsl_moe2_afp8_wfp8_")
    assert "_layout_" not in metadata.stage2.keywords["kernelName"]
    assert metadata.fuse_quant == "fp8"
    assert metadata.block_m == expected_block_m


def test_hy4_mxfp8_fhmoe_capability(monkeypatch: pytest.MonkeyPatch):
    import importlib

    fhmoe = importlib.import_module("aiter.fhmoe")
    _mock_gfx950_fhmoe_metadata(monkeypatch)
    config_path = Path(__file__).resolve().parents[1] / "aiter/configs/tuned_fhmoe.csv"
    monkeypatch.setattr(fhmoe, "_fhmoe_config_file", lambda: str(config_path))
    fhmoe._supports_hy4_mxfp8_fhmoe_config.cache_clear()

    assert fhmoe.supports_hy4_mxfp8_fhmoe(38836)
    assert not fhmoe.supports_hy4_mxfp8_fhmoe(38837)
    assert not fhmoe.supports_hy4_mxfp8_fhmoe(1 << 24)
    assert not fhmoe.supports_hy4_mxfp8_fhmoe(0)


def test_hy4_route_output_limit_fits_uint32_byte_extent():
    from aiter.fhmoe_contract import HY4_FHMOE_MAX_TOKENS

    bytes_per_token = 9 * 6144 * 2
    assert HY4_FHMOE_MAX_TOKENS == 38836
    assert HY4_FHMOE_MAX_TOKENS * bytes_per_token <= (1 << 32) - 1
    assert (HY4_FHMOE_MAX_TOKENS + 1) * bytes_per_token > (1 << 32) - 1


def test_global_moe_large_m_uses_32768_floor_tier_until_131072():
    from aiter.fused_moe import get_padded_M

    assert get_padded_M(32768) == 32768
    assert get_padded_M(32769) == 32768
    assert get_padded_M(65536) == 32768
    assert get_padded_M(131071) == 32768
    assert get_padded_M(131072) == 131072
    assert get_padded_M(131073) == 131072


def test_hy4_mxfp8_fhmoe_capability_fails_on_missing_bucket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import csv
    import importlib

    fhmoe = importlib.import_module("aiter.fhmoe")
    _mock_gfx950_fhmoe_metadata(monkeypatch)
    source = Path(__file__).resolve().parents[1] / "aiter/configs/tuned_fhmoe.csv"
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        rows = list(reader)
        fields = reader.fieldnames
    missing = tmp_path / "missing_hy4_m4096.csv"
    with missing.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            row
            for row in rows
            if not (row["model_dim"] == "6144" and row["token"] == "4096")
        )
    monkeypatch.setattr(fhmoe, "_fhmoe_config_file", lambda: str(missing))
    fhmoe._supports_hy4_mxfp8_fhmoe_config.cache_clear()

    assert not fhmoe.supports_hy4_mxfp8_fhmoe(4096)


@pytest.mark.parametrize(
    "incompatible_kind",
    (
        "wfp4",
        "f16",
        "invalid_sbm",
        "invalid_kwave",
    ),
)
def test_hy4_mxfp8_fhmoe_capability_rejects_incompatible_kernels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, incompatible_kind: str
):
    import csv
    import importlib

    fhmoe = importlib.import_module("aiter.fhmoe")
    _mock_gfx950_fhmoe_metadata(monkeypatch)
    source = Path(__file__).resolve().parents[1] / "aiter/configs/tuned_fhmoe.csv"
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        rows = list(reader)
        fields = reader.fieldnames
    dsv4 = next(
        row for row in rows if row["model_dim"] == "7168" and row["token"] == "1"
    )
    incompatible = tmp_path / f"hy4_{incompatible_kind}_kernels.csv"
    with incompatible.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            updated = dict(row)
            if row["model_dim"] == "6144" and row["token"] == "1":
                if incompatible_kind == "wfp4":
                    updated["kernelName1"] = dsv4["kernelName1"]
                    updated["kernelName2"] = dsv4["kernelName2"]
                elif incompatible_kind == "f16":
                    updated["kernelName2"] = updated["kernelName2"].replace(
                        "_bf16_", "_f16_"
                    )
                elif incompatible_kind == "invalid_sbm":
                    updated["kernelName2"] = (
                        "flydsl_moe2_afp8_wfp8_bf16_t64x128x128_atomic_sbm32"
                    )
                elif incompatible_kind == "invalid_kwave":
                    updated["kernelName1"] = updated["kernelName1"].replace(
                        "_kw2_fp8", "_kw5_fp8"
                    )
            writer.writerow(updated)

    monkeypatch.setattr(fhmoe, "_fhmoe_config_file", lambda: str(incompatible))
    fhmoe._supports_hy4_mxfp8_fhmoe_config.cache_clear()

    assert not fhmoe.supports_hy4_mxfp8_fhmoe(1)


def test_hy4_mxfp8_fhmoe_capability_rejects_config_bypass(
    monkeypatch: pytest.MonkeyPatch,
):
    import importlib

    fhmoe = importlib.import_module("aiter.fhmoe")
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    assert not fhmoe.supports_hy4_mxfp8_fhmoe(1)


@pytest.mark.parametrize("num_tokens", (768, 2049))
def test_dsv4_i384_fhmoe_config_requires_exact_bucket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
):
    import csv

    fused_moe_module = _mock_gfx950_fhmoe_metadata(monkeypatch)
    source_path = Path(__file__).resolve().parents[1] / "aiter/configs/tuned_fhmoe.csv"
    with source_path.open(newline="") as config:
        reader = csv.DictReader(config)
        rows = list(reader)
        assert reader.fieldnames is not None
        fieldnames = reader.fieldnames

    missing_token = str(fused_moe_module.get_padded_M(num_tokens))
    config_path = tmp_path / f"missing_{missing_token}.csv"
    with config_path.open("w", newline="") as config:
        writer = csv.DictWriter(config, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for row in rows if row["token"] != missing_token)

    monkeypatch.setenv("AITER_ONLINE_TUNE", "1")

    with pytest.raises(NotImplementedError, match="requires an exact tuned config"):
        fused_moe_module.get_2stage_cfgs(
            fused_moe_module.get_padded_M(num_tokens),
            7168,
            384,
            385,
            7,
            torch.bfloat16,
            dtypes.fp8,
            dtypes.fp4x2,
            aiter.QuantType.per_1x32,
            True,
            aiter.ActivationType.Silu,
            False,
            0,
            0,
            True,
            GateMode.INTERLEAVE,
            config_file=str(config_path),
        )


def test_dsv4_i384_fhmoe_config_has_true_shapes():
    import csv

    config_path = Path(__file__).resolve().parents[1] / "aiter/configs/tuned_fhmoe.csv"
    ordinary_path = (
        Path(__file__).resolve().parents[1]
        / "aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv"
    )
    with config_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    with ordinary_path.open(newline="") as f:
        ordinary_rows = list(csv.DictReader(f))

    dsv4_rows = [row for row in rows if int(row["inter_dim"]) == 384]
    assert {int(row["token"]) for row in dsv4_rows} == {
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
    }
    assert all(int(row["shared_expert_id"]) == 384 for row in dsv4_rows)
    assert all(int(row["hidden_pad"]) == 0 for row in dsv4_rows)
    assert all(int(row["intermediate_pad"]) == 0 for row in dsv4_rows)
    assert all(row["gate_mode"] == "GateMode.INTERLEAVE" for row in dsv4_rows)
    assert all(row["kernelName1"].startswith("flydsl_") for row in dsv4_rows)
    assert all(row["kernelName2"].startswith("flydsl_") for row in dsv4_rows)

    ordinary_m16 = next(
        row
        for row in ordinary_rows
        if int(row["token"]) == 16
        and int(row["inter_dim"]) == 384
        and int(row["expert"]) == 385
        and int(row["topk"]) == 7
    )
    assert ordinary_m16["kernelName2"].startswith("opus_")


def test_fhmoe_aot_manifest_covers_native_i384():
    from aiter.aot.flydsl.moe import parse_csv
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    ordinary_path = (
        Path(__file__).resolve().parents[1]
        / "aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv"
    )
    config_path = Path(__file__).resolve().parents[1] / "aiter/configs/tuned_fhmoe.csv"
    ordinary_jobs = parse_csv(str(ordinary_path))
    ordinary_variant_jobs = parse_csv(
        str(ordinary_path),
        include_forced_reduce=True,
        include_stage2_fp8=True,
    )
    base_jobs = parse_csv(str(config_path))
    assert not any(job.get("forced_reduce_variant") for job in base_jobs)
    assert not any(job.get("stage2_fp8_variant") for job in base_jobs)
    forced_only_jobs = parse_csv(str(config_path), include_forced_reduce=True)
    assert any(job.get("forced_reduce_variant") for job in forced_only_jobs)
    assert not any(job.get("stage2_fp8_variant") for job in forced_only_jobs)
    fp8_only_jobs = parse_csv(str(config_path), include_stage2_fp8=True)
    assert not any(job.get("forced_reduce_variant") for job in fp8_only_jobs)
    assert any(job.get("stage2_fp8_variant") for job in fp8_only_jobs)
    all_variant_jobs = parse_csv(
        str(config_path),
        include_forced_reduce=True,
        include_stage2_fp8=True,
    )
    forced_reduce_jobs = [
        job for job in all_variant_jobs if job.get("forced_reduce_variant")
    ]
    fp8_reduce_jobs = [job for job in all_variant_jobs if job.get("stage2_fp8_variant")]
    dedicated_jobs = [job for job in base_jobs if job["inter_dim"] == 384]
    hy4_jobs = [job for job in base_jobs if job["inter_dim"] == 256]
    ordinary_fhmoe_jobs = [
        job for job in ordinary_jobs if job.get("shared_expert_id", -1) >= 0
    ]
    ordinary_forced_reduce = [
        job for job in ordinary_variant_jobs if job.get("forced_reduce_variant")
    ]
    ordinary_fp8_reduce = [
        job for job in ordinary_variant_jobs if job.get("stage2_fp8_variant")
    ]

    assert not ordinary_fhmoe_jobs
    assert ordinary_forced_reduce
    assert ordinary_fp8_reduce
    assert all(job.get("shared_expert_id", -1) < 0 for job in ordinary_forced_reduce)
    assert all(job.get("shared_expert_id", -1) < 0 for job in ordinary_fp8_reduce)
    assert all(job["b_dtype"] in ("fp4", "fp8") for job in ordinary_fp8_reduce)
    assert forced_reduce_jobs
    assert all(job["stage"] == 2 for job in forced_reduce_jobs)
    assert all(job["mode"] == "reduce" for job in forced_reduce_jobs)
    atomic_jobs = [
        job
        for job in base_jobs
        if job["stage"] == 2 and job.get("mode", "atomic") != "reduce"
    ]
    identity = lambda job: (
        job["model_dim"],
        job["inter_dim"],
        job["token_num"],
        job["kernel_name"],
    )
    assert {identity(job) for job in forced_reduce_jobs} == {
        identity(job) for job in atomic_jobs
    }
    assert len(forced_reduce_jobs) == len(atomic_jobs)
    reduce_jobs = [
        job
        for job in [*base_jobs, *forced_reduce_jobs]
        if job["stage"] == 2
        and job.get("mode") == "reduce"
        and not job.get("stage2_fp8_variant")
    ]
    assert {identity(job) for job in fp8_reduce_jobs} == {
        identity(job) for job in reduce_jobs
    }
    assert len(dedicated_jobs) == 24
    assert all(job["inter_dim"] == 384 for job in dedicated_jobs)
    assert all(job["shared_expert_id"] == 384 for job in dedicated_jobs)
    assert all(job["b_dtype"] == "fp4" for job in dedicated_jobs)
    assert all(not job.get("enable_bias", False) for job in dedicated_jobs)
    assert {job["token_num"] for job in dedicated_jobs} == {
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
    }
    assert len(hy4_jobs) == 48
    assert all(job["shared_expert_id"] == 256 for job in hy4_jobs)
    assert all(job["b_dtype"] == "fp8" for job in hy4_jobs)
    hy4_stage1_jobs = [job for job in hy4_jobs if job["stage"] == 1]
    assert len(hy4_stage1_jobs) == 32
    assert {job["clamp_shared"] for job in hy4_stage1_jobs} == {True, False}
    assert all("clamp_shared" not in job for job in hy4_jobs if job["stage"] == 2)
    assert {job["token_num"] for job in hy4_jobs} == {
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
    }
    for job in all_variant_jobs:
        params = get_flydsl_kernel_params(job["kernel_name"])
        assert params is not None
        assert job.get("xcd_swizzle", 0) == params.get("xcd_swizzle", 0)
        # HY4's FP8/FP8 rows are intentionally non-persistent. Restored DSV4
        # FP8/FP4 rows retain their legacy persist suffixes; the shared runtime
        # normalizes FP8-A grid scheduling in resolve_flydsl_grid_y_persist_m.
        if job["stage"] == 2 and job["a_dtype"] == "fp8" and job["b_dtype"] == "fp8":
            assert not job.get("persist", False)
        if job["stage"] == 1:
            assert job["inter_dim"] % job["tile_n"] == 0
        else:
            assert job["inter_dim"] % job["tile_k"] == 0

    m2048_names = {
        job["kernel_name"] for job in dedicated_jobs if job["token_num"] == 2048
    }
    assert m2048_names == {
        "flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w3_bnt0_gui",
        "flydsl_moe2_afp8_wfp4_bf16_t64x128x128_atomic",
    }


def test_package_aot_mirrors_moe_debug_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    from aiter.aot.flydsl import common

    config_path = Path(__file__).resolve().parents[1] / "aiter/configs/tuned_fhmoe.csv"
    monkeypatch.setattr(
        common,
        "collect_aot_jobs",
        lambda _paths, parser: parser(str(config_path)),
    )

    monkeypatch.delenv("AITER_FLYDSL_FORCE_REDUCE", raising=False)
    monkeypatch.delenv("AITER_FLYDSL_STAGE2_FP8", raising=False)
    base_jobs = common._collect_aot_jobs_for(common.OpKind.MOE)
    assert not any(job.get("forced_reduce_variant") for job in base_jobs)
    assert not any(job.get("stage2_fp8_variant") for job in base_jobs)

    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "1")
    forced_jobs = common._collect_aot_jobs_for(common.OpKind.MOE)
    assert any(job.get("forced_reduce_variant") for job in forced_jobs)
    assert not any(job.get("stage2_fp8_variant") for job in forced_jobs)

    monkeypatch.delenv("AITER_FLYDSL_FORCE_REDUCE")
    monkeypatch.setenv("AITER_FLYDSL_STAGE2_FP8", "1")
    fp8_jobs = common._collect_aot_jobs_for(common.OpKind.MOE)
    assert not any(job.get("forced_reduce_variant") for job in fp8_jobs)
    assert any(job.get("stage2_fp8_variant") for job in fp8_jobs)


def test_fhmoe_aot_import_is_safe_in_lightweight_mode():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["AITER_AOT_IMPORT"] = "1"
    code = """
import aiter
assert not hasattr(aiter, "ActivationType")
from aiter.fhmoe_contract import _is_hy4_mxfp8_fhmoe_contract
from aiter.aot.flydsl.fhmoe import _FHMoEAOTBackend
assert _is_hy4_mxfp8_fhmoe_contract(
    model_dim=6144,
    inter_dim=256,
    experts=257,
    topk=9,
    routed_mxfp8=True,
    hidden_pad=0,
    intermediate_pad=0,
    gate_interleaved=True,
    doweight_stage1=False,
    shared_expert_id=256,
)
assert _FHMoEAOTBackend(shared_expert_id=256).shared_expert_id == 256
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_fhmoe_aot_cache_is_retrievable_in_run_only_mode(tmp_path: Path):
    import subprocess
    import sys

    if get_gfx() != "gfx950":
        pytest.skip("FHMoE AOT cache retrieval requires gfx950")
    root = Path(__file__).resolve().parents[1]
    cache = tmp_path / "cache"
    env = os.environ.copy()
    env["FLYDSL_RUNTIME_CACHE_DIR"] = str(cache)
    compile_env = dict(env, AITER_AOT_IMPORT="1")
    compile_code = """
from aiter.aot.flydsl.moe import compile_one_config, parse_csv
from aiter.jit.core import AITER_CONFIGS
jobs = [
    job for job in parse_csv(
        AITER_CONFIGS.AITER_CONFIG_FHMOE_FILE,
        include_forced_reduce=True,
        include_stage2_fp8=True,
    )
    if job["model_dim"] == 6144 and job["token_num"] == 1
]
assert len(jobs) == 5
for job in jobs:
    assert compile_one_config(**job)["compile_time"] is not None
"""
    compiled = subprocess.run(
        [sys.executable, "-c", compile_code],
        cwd=root,
        env=compile_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr

    run_env = dict(env, FLYDSL_RUNTIME_RUN_ONLY="1")
    retrieve_code = """
from aiter.ops.flydsl.fhmoe import (
    compile_flydsl_fhmoe_stage1,
    compile_flydsl_fhmoe_stage2,
)
compile_flydsl_fhmoe_stage1(
    model_dim=6144, inter_dim=256, experts=257, topk=9,
    tile_m=32, tile_n=64, tile_k=256, doweight_stage1=False,
    a_dtype="fp8", b_dtype="fp8", out_dtype="fp8", act="silu",
    persist_m=1, use_async_copy=True, k_batch=1, waves_per_eu=4,
    b_nt=2, gate_mode="interleave", model_dim_pad=0, inter_dim_pad=0,
    enable_bias=False, a_scale_one=False, xcd_swizzle=0, k_wave=2,
    v2_output_layout=False, shared_expert_id=256, clamp_shared=False,
)
compile_flydsl_fhmoe_stage1(
    model_dim=6144, inter_dim=256, experts=257, topk=9,
    tile_m=32, tile_n=64, tile_k=256, doweight_stage1=False,
    a_dtype="fp8", b_dtype="fp8", out_dtype="fp8", act="silu",
    persist_m=1, use_async_copy=True, k_batch=1, waves_per_eu=4,
    b_nt=2, gate_mode="interleave", model_dim_pad=0, inter_dim_pad=0,
    enable_bias=False, a_scale_one=False, xcd_swizzle=0, k_wave=2,
    v2_output_layout=False, shared_expert_id=256, clamp_shared=True,
)
compile_flydsl_fhmoe_stage2(
    model_dim=6144, inter_dim=256, experts=257, topk=9,
    tile_m=32, tile_n=128, tile_k=128, doweight_stage2=True,
    a_dtype="fp8", b_dtype="fp8", out_dtype="bf16", accumulate=True,
    enable_bias=False, model_dim_pad=0, inter_dim_pad=0, persist_m=1,
    sort_block_m=0, waves_per_eu=None, use_async_copy=False,
    cu_num_mul=1, b_nt=0, xcd_swizzle=0, shared_expert_id=256,
    use_global_a=False,
)
compile_flydsl_fhmoe_stage2(
    model_dim=6144, inter_dim=256, experts=257, topk=9,
    tile_m=32, tile_n=128, tile_k=128, doweight_stage2=True,
    a_dtype="fp8", b_dtype="fp8", out_dtype="bf16", accumulate=False,
    enable_bias=False, model_dim_pad=0, inter_dim_pad=0, persist_m=1,
    sort_block_m=0, waves_per_eu=None, use_async_copy=False,
    cu_num_mul=1, b_nt=0, xcd_swizzle=0, shared_expert_id=256,
    use_global_a=False,
)
compile_flydsl_fhmoe_stage2(
    model_dim=6144, inter_dim=256, experts=257, topk=9,
    tile_m=32, tile_n=128, tile_k=128, doweight_stage2=True,
    a_dtype="fp8", b_dtype="fp8", out_dtype="fp8", accumulate=False,
    enable_bias=False, model_dim_pad=0, inter_dim_pad=0, persist_m=1,
    sort_block_m=0, waves_per_eu=None, use_async_copy=False,
    cu_num_mul=1, b_nt=0, xcd_swizzle=0, shared_expert_id=256,
    use_global_a=False,
)
"""
    retrieved = subprocess.run(
        [sys.executable, "-c", retrieve_code],
        cwd=root,
        env=run_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert retrieved.returncode == 0, retrieved.stderr


def test_fhmoe_aot_csv_preserves_padding(tmp_path: Path):
    import csv

    from aiter.aot.flydsl.moe import parse_csv

    source = Path(__file__).resolve().parents[1] / "aiter/configs/tuned_fhmoe.csv"
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        row = next(
            row for row in reader if row["model_dim"] == "6144" and row["token"] == "1"
        )
        fields = reader.fieldnames
    row["hidden_pad"] = "64"
    row["intermediate_pad"] = "128"
    padded = tmp_path / "padded_fhmoe.csv"
    with padded.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    jobs = parse_csv(str(padded))
    assert jobs
    assert all(job["model_dim_pad"] == 64 for job in jobs)
    assert all(job["inter_dim_pad"] == 128 for job in jobs)


@pytest.mark.parametrize("stage", (1, 2))
def test_fhmoe_aot_forwards_padding_to_compiler(
    monkeypatch: pytest.MonkeyPatch, stage: int
):
    from aiter.aot.flydsl import moe as aot_moe

    forwarded = {}

    class Backend:
        def build_stage1_args(self, *_args, **_kwargs):
            return ()

        def build_stage2_args(self, *_args, **_kwargs):
            return ()

        def compile_stage1(self, **kwargs):
            forwarded.update(kwargs)
            return object()

        def compile_stage2(self, **kwargs):
            forwarded.update(kwargs)
            return object()

    monkeypatch.setattr(aot_moe, "_run_compiled", lambda *_args: None)
    aot_moe._precompile_to_cache(
        stage=stage,
        model_dim=256,
        inter_dim=128,
        experts=9,
        topk=3,
        tile_m=32,
        tile_n=128,
        tile_k=128,
        a_dtype="fp8",
        b_dtype="fp8",
        out_dtype="bf16",
        gate_mode="interleave",
        mode="atomic",
        cu_num=256,
        token_num=1,
        block_m=32,
        model_dim_pad=64,
        inter_dim_pad=128,
        _aot_backend=Backend(),
    )

    assert forwarded["model_dim_pad"] == 64
    assert forwarded["inter_dim_pad"] == 128


def test_fhmoe_aot_precompile_keeps_native_i384(monkeypatch: pytest.MonkeyPatch):
    from aiter.aot.flydsl import fhmoe as aot_fhmoe
    from aiter.aot.flydsl import moe as aot_moe

    forwarded = {}

    def precompile(**kwargs):
        forwarded.update(kwargs)

    monkeypatch.setattr(aot_moe, "_precompile_to_cache", precompile)
    aot_fhmoe.precompile_fhmoe_to_cache(
        experts=385,
        shared_expert_id=384,
        cu_num=256,
        stage=2,
        model_dim=7168,
        inter_dim=384,
        topk=7,
    )

    assert forwarded["inter_dim"] == 384
    assert forwarded["_aot_backend"].shared_expert_id == 384


def test_fhmoe_aot_stage2_scale_size_uses_padded_inter_dim():
    from aiter.aot.flydsl.moe import _mx_w2_scale_numel

    assert _mx_w2_scale_numel(385, 7168, 384) == 385 * 7168 * (512 // 32)
    assert _mx_w2_scale_numel(257, 6144, 256) == 257 * 6144 * (256 // 32)


def test_fhmoe_aot_precompile_accepts_hy4_mxfp8(monkeypatch: pytest.MonkeyPatch):
    from aiter.aot.flydsl import fhmoe as aot_fhmoe
    from aiter.aot.flydsl import moe as aot_moe

    forwarded = {}

    def precompile(**kwargs):
        forwarded.update(kwargs)

    monkeypatch.setattr(aot_moe, "_precompile_to_cache", precompile)
    aot_fhmoe.precompile_fhmoe_to_cache(
        experts=257,
        shared_expert_id=256,
        a_dtype="fp8",
        b_dtype="fp8",
        cu_num=256,
        stage=1,
        model_dim=6144,
        inter_dim=256,
        topk=9,
        gate_mode="interleave",
    )

    assert forwarded["b_dtype"] == "fp8"
    assert forwarded["inter_dim"] == 256
    assert forwarded["_aot_backend"].shared_expert_id == 256


@pytest.mark.parametrize("cu_num", (0, 256))
def test_fhmoe_aot_precompile_accepts_hy4_mxfp8_stage2(
    monkeypatch: pytest.MonkeyPatch, cu_num: int
):
    from aiter.aot.flydsl import fhmoe as aot_fhmoe
    from aiter.aot.flydsl import moe as aot_moe

    forwarded = {}

    def precompile(**kwargs):
        forwarded.update(kwargs)

    monkeypatch.setattr(aot_moe, "_precompile_to_cache", precompile)
    aot_fhmoe.precompile_fhmoe_to_cache(
        experts=257,
        shared_expert_id=256,
        a_dtype="fp8",
        b_dtype="fp8",
        cu_num=cu_num,
        stage=2,
        model_dim=6144,
        inter_dim=256,
        topk=9,
    )

    assert forwarded["stage"] == 2
    assert forwarded["b_dtype"] == "fp8"
    assert forwarded["cu_num"] == 256


def test_fhmoe_aot_precompile_rejects_separated_mxfp8():
    from aiter.aot.flydsl import fhmoe as aot_fhmoe

    with pytest.raises(ValueError, match="interleaved gate/up layout"):
        aot_fhmoe.precompile_fhmoe_to_cache(
            experts=257,
            shared_expert_id=256,
            a_dtype="fp8",
            b_dtype="fp8",
            cu_num=256,
            stage=1,
            model_dim=6144,
            inter_dim=256,
            topk=9,
        )


def test_fhmoe_aot_precompile_rejects_unknown_cu_count():
    from aiter.aot.flydsl import fhmoe as aot_fhmoe

    with pytest.raises(ValueError, match="supports only gfx950"):
        aot_fhmoe.precompile_fhmoe_to_cache(
            experts=257,
            shared_expert_id=256,
            a_dtype="fp8",
            b_dtype="fp8",
            cu_num=999,
            stage=2,
            model_dim=6144,
            inter_dim=256,
            topk=9,
        )


def test_fhmoe_aot_precompile_rejects_invalid_stage():
    from aiter.aot.flydsl import fhmoe as aot_fhmoe

    with pytest.raises(ValueError, match="stage must be 1 or 2"):
        aot_fhmoe.precompile_fhmoe_to_cache(
            experts=257,
            shared_expert_id=256,
            cu_num=256,
            stage=0,
        )


def test_fhmoe_aot_clamp_default_and_explicit_unclamped_specialization(
    monkeypatch: pytest.MonkeyPatch,
):
    from aiter.aot.flydsl import fhmoe as aot_fhmoe
    from aiter.ops.flydsl import fhmoe as ops_fhmoe

    monkeypatch.setattr(
        ops_fhmoe,
        "compile_flydsl_fhmoe_stage1",
        lambda **kwargs: kwargs,
    )
    backend = aot_fhmoe._FHMoEAOTBackend(shared_expert_id=256)
    kwargs = {
        "model_dim": 6144,
        "inter_dim": 256,
        "experts": 257,
        "topk": 9,
        "b_dtype": "fp8",
        "model_dim_pad": 0,
        "inter_dim_pad": 0,
        "gate_mode": "interleave",
        "doweight_stage1": False,
    }

    assert backend.compile_stage1(**kwargs)["clamp_shared"] is True
    assert (
        backend.compile_stage1(**(kwargs | {"clamp_shared": False}))["clamp_shared"]
        is False
    )


@pytest.mark.parametrize("clamp_shared", (True, False))
def test_fhmoe_aot_precompile_forwards_clamp_shared(
    monkeypatch: pytest.MonkeyPatch,
    clamp_shared: bool,
):
    from aiter.aot.flydsl import fhmoe as aot_fhmoe
    from aiter.aot.flydsl import moe as aot_moe

    captured = {}
    monkeypatch.setattr(
        aot_fhmoe._FHMoEAOTBackend,
        "compile_stage1",
        lambda _self, **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(aot_moe, "_run_compiled", lambda *_args, **_kwargs: None)

    aot_moe._precompile_to_cache(
        stage=1,
        model_dim=6144,
        inter_dim=256,
        experts=257,
        topk=9,
        tile_m=32,
        tile_n=64,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp8",
        out_dtype="fp8",
        token_num=1,
        gate_mode="interleave",
        k_wave=2,
        shared_expert_id=256,
        clamp_shared=clamp_shared,
        _aot_backend=aot_fhmoe._FHMoEAOTBackend(shared_expert_id=256),
    )

    assert captured["clamp_shared"] is clamp_shared


def test_fhmoe_aot_precompile_accepts_non_hy4_mxfp8(
    monkeypatch: pytest.MonkeyPatch,
):
    from aiter.aot.flydsl import fhmoe as aot_fhmoe
    from aiter.aot.flydsl import moe as aot_moe

    forwarded = {}

    def precompile(**kwargs):
        forwarded.update(kwargs)

    monkeypatch.setattr(aot_moe, "_precompile_to_cache", precompile)
    aot_fhmoe.precompile_fhmoe_to_cache(
        experts=9,
        shared_expert_id=8,
        a_dtype="fp8",
        b_dtype="fp8",
        cu_num=256,
        stage=1,
        model_dim=256,
        inter_dim=128,
        topk=3,
        gate_mode="interleave",
    )

    assert forwarded["b_dtype"] == "fp8"
    assert forwarded["_aot_backend"].shared_expert_id == 8
    with pytest.raises(ValueError, match="HY4-compatible MXFP8"):
        aot_fhmoe.precompile_fhmoe_to_cache(
            experts=9,
            shared_expert_id=8,
            a_dtype="fp8",
            b_dtype="fp8",
            cu_num=256,
            stage=1,
            model_dim=256,
            inter_dim=128,
            topk=3,
            gate_mode="interleave",
            clamp_shared=False,
        )


def test_fhmoe_aot_keeps_dsv4_shape_mxfp8_generic(
    monkeypatch: pytest.MonkeyPatch,
):
    from aiter.aot.flydsl import fhmoe as aot_fhmoe
    from aiter.aot.flydsl import moe as aot_moe

    forwarded = {}

    def precompile(**kwargs):
        forwarded.update(kwargs)

    monkeypatch.setattr(aot_moe, "_precompile_to_cache", precompile)
    aot_fhmoe.precompile_fhmoe_to_cache(
        experts=385,
        shared_expert_id=384,
        a_dtype="fp8",
        b_dtype="fp8",
        cu_num=256,
        stage=1,
        model_dim=7168,
        inter_dim=384,
        topk=7,
        gate_mode="interleave",
    )

    assert forwarded["b_dtype"] == "fp8"
    assert forwarded["model_dim"] == 7168
    assert forwarded["_aot_backend"].shared_expert_id == 384


def test_fhmoe_aot_stage1_forwards_optional_swiglu_abi(
    monkeypatch: pytest.MonkeyPatch,
):
    from aiter.aot.flydsl import fhmoe as aot_fhmoe
    from aiter.ops.flydsl import fhmoe as ops_fhmoe

    tensor = torch.empty(0)
    forwarded = {}

    monkeypatch.setattr(aot_fhmoe, "_shared_weight", lambda *_: tensor)
    monkeypatch.setattr(aot_fhmoe, "_shared_scale", lambda *_: tensor)

    def build_args(*args, **kwargs):
        forwarded.update(kwargs)
        return args

    monkeypatch.setattr(ops_fhmoe, "_s1_args_fhmoe", build_args)

    result = aot_fhmoe._FHMoEAOTBackend(shared_expert_id=8).build_stage1_args(
        *((tensor,) * 10),
        1,
        2,
        3,
        4,
        "cpu",
        swiglu_limit=10.0,
        pass_swiglu_limit=False,
    )

    assert result
    assert forwarded["swiglu_limit"] == 10.0
    assert forwarded["pass_swiglu_limit"] is False


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("partial", "must be provided together"),
        ("scale_dtype", "scales must use FP8 E8M0"),
        ("scale_shape", "Expected preshuffled shared_w1_scale shape"),
    ),
)
def test_heterogeneous_moe_rejects_invalid_shared_contract(
    weights: _Weights,
    case: str,
    message: str,
):
    profile = _profile()
    hidden, _, _, all_weight, all_ids = _route_inputs(
        profile, 4, weights.routed_w1.device
    )
    kwargs = _common_kwargs(profile, weights.routed_s1, weights.routed_s2)
    if case == "partial":
        kwargs["shared_w1"] = weights.shared_w1
    else:
        kwargs.update(
            shared_w1=weights.shared_w1,
            shared_w2=weights.shared_w2,
            shared_w1_scale=weights.shared_s1,
            shared_w2_scale=weights.shared_s2,
            shared_expert_id=profile.shared_id,
        )
        if case == "scale_dtype":
            kwargs["shared_w1_scale"] = weights.shared_s1.float()
        else:
            kwargs["shared_w1_scale"] = weights.shared_s1[:-1].contiguous()

    with pytest.raises(ValueError, match=message):
        fused_moe(
            hidden,
            weights.routed_w1,
            weights.routed_w2,
            all_weight,
            all_ids,
            **kwargs,
        )


def test_a4w4_routed_fp8_shared_heterogeneous_path(
    monkeypatch: pytest.MonkeyPatch,
    a4_weights: _Weights,
):
    profile = _Profile(256, 128, 128, 9, 2)
    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "1")
    hidden, routed_weight, routed_ids, all_weight, all_ids = _route_inputs(
        profile, 4, a4_weights.routed_w1.device
    )
    kwargs = _common_kwargs(
        profile,
        a4_weights.routed_s1,
        a4_weights.routed_s2,
        gate_mode=GateMode.SEPARATED,
    )
    kwargs.update(
        shared_w1=a4_weights.shared_w1,
        shared_w2=a4_weights.shared_w2,
        shared_w1_scale=a4_weights.shared_s1,
        shared_w2_scale=a4_weights.shared_s2,
        shared_expert_id=profile.shared_id,
    )
    actual = fused_moe(
        hidden,
        a4_weights.routed_w1,
        a4_weights.routed_w2,
        all_weight,
        all_ids,
        **kwargs,
    )

    routed_high = _torch_routed_reference(
        hidden,
        routed_weight,
        routed_ids,
        a4_weights,
        profile,
        activation_quantization="fp4",
    )
    shared_high = _torch_shared_reference(
        hidden,
        a4_weights,
        "fp4",
        _shared_swiglu_limit(_swiglu_limit(profile), clamp_shared=True),
    )
    assert torch.isfinite(actual).all()
    error = _rel_l2(actual, routed_high + shared_high)
    assert error <= 5e-2, f"A4W4/FP8 heterogeneous FP32 error: {error:.3e}"
