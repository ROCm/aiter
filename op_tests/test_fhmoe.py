# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness oracle for fused heterogeneous MoE (FHMoE).

The default case is deliberately small. Set AITER_HETERO_MOE_DSV4=1 to use
the exact DeepSeek-V4-Pro TP8 dimensions; that profile allocates several GiB.
Set AITER_HETERO_MOE_FULL_SWEEP=1 to cover M=4,8,16,32,64 and
AITER_HETERO_MOE_STRESS_REPEATS=500 to stress both stage-2 epilogues and
bound the default atomic path's run-to-run variation.
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
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import (
    shuffle_scale,
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
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
    if os.environ.get("AITER_HETERO_MOE_DSV4", "0") == "1":
        return _Profile(7168, 512, 384, 385, 6)
    return _Profile(256, 128, 128, 9, 2)


def _m_values() -> list[int]:
    if os.environ.get("AITER_HETERO_MOE_FULL_SWEEP", "0") == "1":
        return [4, 8, 16, 32, 64]
    return [4]


def _swiglu_limit(profile: _Profile) -> float:
    return 10.0


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
    active = slice(0, profile.routed_topk)
    raw_w1 = w1[active].clone()
    raw_w2 = w2[active].clone()
    raw_s1 = s1[active].clone()
    raw_s2 = s2[active].clone()
    return (
        *_shuffle_w1(w1, s1, interleave),
        *_shuffle_w2(w2, s2, interleave),
        raw_w1,
        raw_w2,
        raw_s1,
        raw_s2,
    )


def _build_weights(profile: _Profile, interleave: bool) -> _Weights:
    if get_gfx() != "gfx950":
        pytest.skip("heterogeneous MXFP4/FP8 MoE requires gfx950")
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
    ) = _make_routed_weights(profile, device, interleave)
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
    return _build_weights(_profile(), interleave=True)


@pytest.fixture(scope="module")
def a4_weights() -> _Weights:
    profile = _Profile(256, 128, 128, 9, 2)
    return _build_weights(profile, interleave=False)


def _route_inputs(profile: _Profile, m: int, device: torch.device) -> tuple:
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
    routed_ids = (row + slot) % profile.routed_topk
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
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    hidden, routed_weight, routed_ids, all_weight, all_ids = _route_inputs(
        profile, m, weights.routed_w1.device
    )
    hetero_kwargs = _common_kwargs(profile, weights.routed_s1, weights.routed_s2)
    hetero_kwargs.update(
        shared_w1=weights.shared_w1,
        shared_w2=weights.shared_w2,
        shared_w1_scale=weights.shared_s1,
        shared_w2_scale=weights.shared_s2,
        shared_expert_id=profile.shared_id,
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
    routed_out = fused_moe(
        hidden,
        routed_w1,
        routed_w2,
        routed_weight,
        routed_ids,
        **_common_kwargs(profile, routed_s1, routed_s2),
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
    shared_out = fused_moe(
        hidden,
        weights.native_w1,
        weights.native_w2,
        shared_weight,
        shared_ids,
        **_common_kwargs(native_profile, weights.native_s1, weights.native_s2),
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
        },
    )


def _fp8_group_quant_dequant(x: torch.Tensor, group_size: int) -> torch.Tensor:
    shape = x.shape
    blocks = x.float().view(-1, group_size)
    scale = fp4_utils.f32_to_mx_e8m0_scale(
        blocks.abs().amax(dim=1), dtype=MxDtypeInt.FP8_E4M3
    )
    scale_f32 = fp4_utils.e8m0_to_f32(scale).view(-1, 1)
    quant = (blocks / scale_f32).to(dtypes.fp8)
    return (quant.float() * scale_f32).view(shape)


def _fp4_group_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    quant, scale = per_1x32_f4_quant(x)
    scale_f32 = fp4_utils.e8m0_to_f32(scale).repeat_interleave(32, dim=-1)
    return fp4_utils.mxfp4_to_f32(quant) * scale_f32


def _activation_quant_dequant(x: torch.Tensor, quantization: str) -> torch.Tensor:
    if quantization == "fp4":
        return _fp4_group_quant_dequant(x)
    if quantization == "fp8":
        return _fp8_group_quant_dequant(x, 32)
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
) -> torch.Tensor:
    """Dequantized FP32 reference for the routed A8W4 experts."""
    w1 = _dequant_fp4_weight(weights.raw_routed_w1, weights.raw_routed_s1)
    w2 = _dequant_fp4_weight(weights.raw_routed_w2, weights.raw_routed_s2)
    x = _activation_quant_dequant(hidden.float(), activation_quantization)
    expanded_x = x[:, None, :].expand(-1, profile.routed_topk, -1)
    slot_out = torch.zeros(
        (*routed_ids.shape, profile.hidden),
        dtype=dtypes.fp32,
        device=hidden.device,
    )

    for expert_id in range(profile.routed_topk):
        mask = routed_ids == expert_id
        if not mask.any():
            continue
        gate_up = F.linear(expanded_x[mask], w1[expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        limit = _swiglu_limit(profile)
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        inter = F.silu(gate) * up
        if activation_quantization == "fp8":
            inter = inter.to(dtypes.bf16).float()
        inter = _activation_quant_dequant(inter, activation_quantization)
        slot_out[mask] = F.linear(inter, w2[expert_id])

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
    gate = gate.clamp(max=swiglu_limit)
    up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    inter = F.silu(gate) * up
    if group_size == "fp4":
        inter = _fp4_group_quant_dequant(inter)
    elif group_size is not None:
        inter = inter.to(dtypes.bf16).float()
        inter = _fp8_group_quant_dequant(inter, group_size)
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
    )
    shared = _torch_shared_reference(
        context["hidden"], weights, 32, _swiglu_limit(profile)
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
    forced, unfused, context = _run_composed_oracle(profile, weights, m)
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

    limit = _swiglu_limit(profile)
    golden = _torch_shared_reference(context["hidden"], weights, None, limit)
    per_32 = _torch_shared_reference(context["hidden"], weights, 32, limit)
    per_128 = _torch_shared_reference(context["hidden"], weights, 128, limit)
    error_32 = _rel_l2(per_32, golden)
    error_128 = _rel_l2(per_128, golden)
    assert error_32 <= error_128 + 5e-4, (
        "per-32 heterogeneous activation quantization regressed against native "
        f"per-128 FP8: {error_32=:.3e}, {error_128=:.3e}"
    )


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
    from aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage import (
        compile_mixed_moe_gemm1,
        compile_mixed_moe_gemm2,
    )
    from aiter.ops.flydsl.kernels.fhmoe import (
        compile_mixed_fhmoe_gemm1,
        compile_mixed_fhmoe_gemm2,
    )
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
    from aiter.ops.flydsl.fhmoe import (
        flydsl_fhmoe_stage1,
        flydsl_fhmoe_stage2,
    )

    assert callable(fhmoe_)
    legacy_schema = torch.ops.aiter.fused_moe_.default._schema
    fhmoe_schema = torch.ops.aiter.fhmoe_.default._schema
    fhmoe_fields = {
        "shared_w1",
        "shared_w2",
        "shared_w1_scale",
        "shared_w2_scale",
        "shared_expert_id",
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
    fhmoe_apis = (
        flydsl_fhmoe_stage1,
        flydsl_fhmoe_stage2,
        compile_mixed_fhmoe_gemm1,
        compile_mixed_fhmoe_gemm2,
    )
    assert all(
        "xcd_swizzle" not in inspect.signature(api).parameters for api in fhmoe_apis
    )


def test_fhmoe_runtime_compile_bridge_uses_xcd0(monkeypatch: pytest.MonkeyPatch):
    from aiter.ops.flydsl import fhmoe

    tensor = torch.empty(0)
    compile_calls = []

    def invoke_compiler(**kwargs):
        return kwargs["_compile_kernel"](xcd_swizzle=kwargs["xcd_swizzle"])

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
        )
        == 2
    )
    assert compile_calls == [
        (1, {"shared_expert_id": 8}),
        (2, {"shared_expert_id": 8}),
    ]
    with pytest.raises(ValueError, match="do not support XCD swizzling"):
        fhmoe._compile_fhmoe_xcd0(lambda: None, xcd_swizzle=4)


def test_fhmoe_aot_jobs_disable_xcd_swizzling():
    from aiter.aot.flydsl.moe import parse_csv

    csv_path = (
        Path(__file__).resolve().parents[1]
        / "aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv"
    )
    jobs = parse_csv(str(csv_path))
    fhmoe_jobs = [job for job in jobs if job.get("shared_expert_id", -1) >= 0]

    assert len(fhmoe_jobs) == 32
    assert all(job.get("xcd_swizzle", 0) == 0 for job in fhmoe_jobs)
    assert all("_xcd" not in job["kernel_name"] for job in fhmoe_jobs)


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
        hidden, a4_weights, "fp4", _swiglu_limit(profile)
    )
    assert torch.isfinite(actual).all()
    error = _rel_l2(actual, routed_high + shared_high)
    assert error <= 5e-2, f"A4W4/FP8 heterogeneous FP32 error: {error:.3e}"
