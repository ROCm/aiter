# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe, fused_moe_multi_b
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import (
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
from aiter.utility.fp4_utils import e8m0_shuffle

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="requires a ROCm GPU",
)


@pytest.fixture(autouse=True)
def _force_deterministic_stage2_reduce(monkeypatch):
    # The atomic path depends on the selected sorting backend zeroing moe_buf.
    # Route-output + reduce is deterministic and directly exercises the multi-B
    # stage2 accumulate=False ABI, including no_combine.
    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "1")


def _quantize_and_shuffle(
    weight: torch.Tensor, *, model_dim: int, inter_dim: int, stage: int
) -> tuple[torch.Tensor, torch.Tensor]:
    quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    packed, scale = quant(weight, quant_dtype=dtypes.fp4x2)
    if stage == 1:
        packed = packed.view(weight.shape[0], 2 * inter_dim, model_dim // 2)
    else:
        packed = packed.view(weight.shape[0], model_dim, inter_dim // 2)
    packed = shuffle_weight(packed, layout=(16, 16))
    packed.is_shuffled = True
    scale = e8m0_shuffle(scale.reshape(-1, scale.shape[-1]))
    return packed, scale


def _quantize_and_shuffle_mixed(
    weight: torch.Tensor,
    *,
    experts: int,
    stage: int,
    weight_kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight_kind == "a8w4":
        quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
        packed, scale = quant(weight, quant_dtype=dtypes.fp4x2)
        packed = packed.view(*weight.shape[:-1], weight.shape[-1] // 2)
    elif weight_kind == "mxfp8":
        packed, scale = aiter.per_1x32_f8_scale_f8_quant(
            weight,
            quant_dtype=dtypes.fp8,
            scale_type=dtypes.fp8_e8m0,
        )
        packed = packed.view(weight.shape)
    else:
        raise ValueError(f"unknown weight_kind={weight_kind}")

    gate_up = stage == 1
    packed = shuffle_weight_a16w4(packed, 16, gate_up)
    packed.is_shuffled = True
    scale_2d = scale.reshape(-1, scale.shape[-1])
    if gate_up or weight_kind == "a8w4":
        scale = shuffle_scale_a16w4(scale_2d, experts, gate_up)
    else:
        scale = e8m0_shuffle(scale_2d)
    return packed, scale


def _quantize_fp8_blockscale(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    experts, rows, cols = weight.shape
    blocks = (
        weight.view(experts, rows // 128, 128, cols // 128, 128)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(experts, -1, 128 * 128)
    )
    quantized, scale = aiter.pertoken_quant(blocks, quant_dtype=dtypes.fp8)
    quantized = (
        quantized.view(experts, rows // 128, cols // 128, 128, 128)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view_as(weight)
    )
    scale = scale.view(experts, rows // 128, cols // 128)
    dequantized = (
        quantized.view(experts, rows // 128, 128, cols // 128, 128).float()
        * scale[:, :, None, :, None]
    )
    return (
        shuffle_weight(quantized, layout=(16, 16)),
        scale,
        dequantized.reshape_as(weight),
    )


def _qdq_fp8_per_1x128(value: torch.Tensor) -> torch.Tensor:
    rows = value.reshape(-1, 128)
    quantized, scale = aiter.pertoken_quant(rows, quant_dtype=dtypes.fp8)
    return (quantized.float() * scale.reshape(-1, 1)).reshape_as(value)


def _fp8_blockscale_torch_reference(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    hidden_qdq = _qdq_fp8_per_1x128(hidden)
    output = torch.zeros(
        (hidden.shape[0], topk_ids.shape[1], hidden.shape[1]),
        dtype=torch.float32,
        device=hidden.device,
    )
    for token in range(hidden.shape[0]):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            gate_up = torch.mv(w1[expert], hidden_qdq[token].float())
            gate, up = gate_up.chunk(2)
            intermediate = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            intermediate_qdq = _qdq_fp8_per_1x128(intermediate).float()
            output[token, slot] = (
                torch.mv(w2[expert], intermediate_qdq) * topk_weights[token, slot]
            )
    return output.to(torch.bfloat16)


def _boundary_routing(
    tokens: int, topk: int, partition_sizes: tuple[int, ...], device
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    for size in partition_sizes:
        offsets.append(offsets[-1] + size)
    ids = [0, offsets[-1] - 1]
    for boundary in offsets[1:-1]:
        ids.extend((boundary - 1, boundary))
    # Real top-k routing never repeats an expert within one token. Keep every
    # boundary first, then complete a deterministic expert permutation.
    ids = list(dict.fromkeys(ids))
    ids.extend(expert for expert in range(offsets[-1]) if expert not in ids)
    ids = torch.tensor(ids, dtype=torch.int32, device=device)
    repeats = (tokens * topk + ids.numel() - 1) // ids.numel()
    topk_ids = ids.repeat(repeats)[: tokens * topk].view(tokens, topk)
    weights = torch.rand(tokens, topk, dtype=torch.float32, device=device)
    return topk_ids, weights / weights.sum(dim=-1, keepdim=True)


def _run_case(
    partition_sizes: tuple[int, ...],
    tokens: int,
    *,
    with_bias: bool = False,
    no_combine: bool = False,
):
    if get_gfx() != "gfx950":
        pytest.skip("multi-B kernels are gfx950-only")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    experts = sum(partition_sizes)
    model_dim, inter_dim = 512, 256
    topk = min(8, experts)
    torch.manual_seed(20260802 + experts + tokens)

    hidden = torch.randn(tokens, model_dim, dtype=dtype, device=device)
    w1_raw = torch.randn(experts, 2 * inter_dim, model_dim, dtype=dtype, device=device)
    w2_raw = torch.randn(experts, model_dim, inter_dim, dtype=dtype, device=device)
    topk_ids, topk_weights = _boundary_routing(tokens, topk, partition_sizes, device)
    bias1 = (
        torch.randn(experts, 2 * inter_dim, dtype=torch.float32, device=device)
        if with_bias
        else None
    )
    bias2 = (
        torch.randn(experts, model_dim, dtype=torch.float32, device=device)
        if with_bias
        else None
    )

    full_w1, full_s1 = _quantize_and_shuffle(
        w1_raw, model_dim=model_dim, inter_dim=inter_dim, stage=1
    )
    full_w2, full_s2 = _quantize_and_shuffle(
        w2_raw, model_dim=model_dim, inter_dim=inter_dim, stage=2
    )
    expected = fused_moe(
        hidden,
        full_w1,
        full_w2,
        topk_weights,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.per_1x32,
        w1_scale=full_s1,
        w2_scale=full_s2,
        bias1=bias1,
        bias2=bias2,
    )

    w1_parts = []
    w2_parts = []
    s1_parts = []
    s2_parts = []
    for w1_raw_part, w2_raw_part in zip(
        w1_raw.split(partition_sizes), w2_raw.split(partition_sizes), strict=True
    ):
        w1_part, s1_part = _quantize_and_shuffle(
            w1_raw_part.contiguous(),
            model_dim=model_dim,
            inter_dim=inter_dim,
            stage=1,
        )
        w2_part, s2_part = _quantize_and_shuffle(
            w2_raw_part.contiguous(),
            model_dim=model_dim,
            inter_dim=inter_dim,
            stage=2,
        )
        w1_parts.append(w1_part)
        w2_parts.append(w2_part)
        s1_parts.append(s1_part)
        s2_parts.append(s2_part)

    assert torch.equal(torch.cat(w1_parts).view(torch.uint8), full_w1.view(torch.uint8))
    assert torch.equal(torch.cat(w2_parts).view(torch.uint8), full_w2.view(torch.uint8))
    assert torch.equal(torch.cat(s1_parts).view(torch.uint8), full_s1.view(torch.uint8))
    assert torch.equal(torch.cat(s2_parts).view(torch.uint8), full_s2.view(torch.uint8))

    actual = fused_moe_multi_b(
        hidden,
        w1_parts,
        w2_parts,
        topk_weights,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.per_1x32,
        w1_scale_partitions=s1_parts,
        w2_scale_partitions=s2_parts,
        bias1=bias1,
        bias2=bias2,
        no_combine=no_combine,
    )
    if no_combine:
        assert actual.shape == (tokens, topk, model_dim)
        actual = actual.sum(dim=1)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "partition_sizes",
    [
        (8, 8),
        (4, 4, 4, 4),
        (2, 2, 2, 2, 2, 2, 2, 2),
    ],
)
@pytest.mark.parametrize("tokens", [1, 17])
def test_multi_b_even_partitions_match_contiguous(monkeypatch, partition_sizes, tokens):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    _run_case(partition_sizes, tokens)


def test_multi_b_e257_uneven_boundaries_and_bias(monkeypatch):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    _run_case((64, 64, 64, 65), 17, with_bias=True)


def test_multi_b_no_combine_matches_contiguous(monkeypatch):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    _run_case((4, 4, 4, 4), 7, no_combine=True)


@pytest.mark.parametrize(
    ("weight_kind", "activation", "swiglu_limit"),
    [
        ("a8w4", aiter.ActivationType.Silu, 1.25),
        ("mxfp8", aiter.ActivationType.Silu, None),
    ],
)
def test_multi_b_mixed_backends_match_contiguous(
    monkeypatch, weight_kind, activation, swiglu_limit
):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")
    if get_gfx() != "gfx950":
        pytest.skip("multi-B kernels are gfx950-only")

    partition_sizes = (4, 4, 4, 4)
    experts = sum(partition_sizes)
    tokens, topk = 7, 4
    model_dim, inter_dim = 512, 256
    device = torch.device("cuda")
    torch.manual_seed(4217 if weight_kind == "a8w4" else 4218)
    hidden = torch.randn(tokens, model_dim, dtype=torch.bfloat16, device=device)
    w1_raw = torch.randn(
        experts, 2 * inter_dim, model_dim, dtype=torch.bfloat16, device=device
    )
    w2_raw = torch.randn(
        experts, model_dim, inter_dim, dtype=torch.bfloat16, device=device
    )
    topk_ids, topk_weights = _boundary_routing(tokens, topk, partition_sizes, device)

    full_w1, full_s1 = _quantize_and_shuffle_mixed(
        w1_raw, experts=experts, stage=1, weight_kind=weight_kind
    )
    full_w2, full_s2 = _quantize_and_shuffle_mixed(
        w2_raw, experts=experts, stage=2, weight_kind=weight_kind
    )
    common_kwargs = {
        "activation": activation,
        "quant_type": aiter.QuantType.per_1x32,
        "swiglu_limit": swiglu_limit,
        "gate_mode": GateMode.INTERLEAVE.value,
    }
    expected = fused_moe(
        hidden,
        full_w1,
        full_w2,
        topk_weights,
        topk_ids,
        w1_scale=full_s1,
        w2_scale=full_s2,
        **common_kwargs,
    )

    w1_parts = []
    w2_parts = []
    s1_parts = []
    s2_parts = []
    for size, w1_raw_part, w2_raw_part in zip(
        partition_sizes,
        w1_raw.split(partition_sizes),
        w2_raw.split(partition_sizes),
        strict=True,
    ):
        w1_part, s1_part = _quantize_and_shuffle_mixed(
            w1_raw_part.contiguous(),
            experts=size,
            stage=1,
            weight_kind=weight_kind,
        )
        w2_part, s2_part = _quantize_and_shuffle_mixed(
            w2_raw_part.contiguous(),
            experts=size,
            stage=2,
            weight_kind=weight_kind,
        )
        w1_parts.append(w1_part)
        w2_parts.append(w2_part)
        s1_parts.append(s1_part)
        s2_parts.append(s2_part)

    actual = fused_moe_multi_b(
        hidden,
        w1_parts,
        w2_parts,
        topk_weights,
        topk_ids,
        w1_scale_partitions=s1_parts,
        w2_scale_partitions=s2_parts,
        **common_kwargs,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def _run_fp8_blockscale_case(partition_sizes: tuple[int, ...], tokens: int):
    if get_gfx() != "gfx950":
        pytest.skip("conventional FP8 multi-B kernels are gfx950-only")

    device = torch.device("cuda")
    experts = sum(partition_sizes)
    model_dim = inter_dim = 256
    topk = min(8, experts)
    torch.manual_seed(8100 + experts * 11 + tokens)

    hidden = torch.randn(tokens, model_dim, dtype=torch.bfloat16, device=device)
    w1_raw = torch.randn(
        experts,
        2 * inter_dim,
        model_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    w2_raw = torch.randn(
        experts,
        model_dim,
        inter_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    topk_ids, topk_weights = _boundary_routing(tokens, topk, partition_sizes, device)

    full_w1, full_s1, dequant_w1 = _quantize_fp8_blockscale(w1_raw)
    full_w2, full_s2, dequant_w2 = _quantize_fp8_blockscale(w2_raw)
    expected = _fp8_blockscale_torch_reference(
        hidden,
        dequant_w1,
        dequant_w2,
        topk_weights,
        topk_ids,
    )

    # Materialize independent allocations, including uneven terminal
    # partitions, so every boundary must select a pointer-table entry.
    w1_parts = [part.clone() for part in full_w1.split(partition_sizes, dim=0)]
    w2_parts = [part.clone() for part in full_w2.split(partition_sizes, dim=0)]
    s1_parts = [part.clone() for part in full_s1.split(partition_sizes, dim=0)]
    s2_parts = [part.clone() for part in full_s2.split(partition_sizes, dim=0)]

    actual = fused_moe_multi_b(
        hidden,
        w1_parts,
        w2_parts,
        topk_weights,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.per_128x128,
        w1_scale_partitions=s1_parts,
        w2_scale_partitions=s2_parts,
        no_combine=True,
    )
    actual_f32 = actual.float().flatten()
    expected_f32 = expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32,
        expected_f32,
        dim=0,
    )
    relative_l2 = torch.linalg.vector_norm(actual_f32 - expected_f32) / torch.linalg.vector_norm(
        expected_f32
    )
    assert cosine > 0.9999, f"{cosine=}, {relative_l2=}"
    assert relative_l2 < 0.02, f"{cosine=}, {relative_l2=}"


@pytest.mark.parametrize(
    "partition_sizes",
    [
        (8, 8),
        (4, 4, 4, 4),
        (2, 2, 2, 2, 2, 2, 2, 2),
        (3, 5),
        (1, 2, 2, 3),
        (1, 1, 1, 1, 1, 1, 1, 2),
    ],
)
@pytest.mark.parametrize("tokens", [1, 17])
def test_multi_b_fp8_blockscale_matches_contiguous(
    monkeypatch, partition_sizes, tokens
):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    _run_fp8_blockscale_case(partition_sizes, tokens)


def test_multi_b_list_of_one_uses_single_path(monkeypatch):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    if get_gfx() != "gfx950":
        pytest.skip("test fixture targets gfx950")

    device = torch.device("cuda")
    model_dim, inter_dim = 512, 256
    experts, tokens, topk = 8, 3, 2
    hidden = torch.randn(tokens, model_dim, dtype=torch.bfloat16, device=device)
    w1_raw = torch.randn(
        experts,
        2 * inter_dim,
        model_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    w2_raw = torch.randn(
        experts, model_dim, inter_dim, dtype=torch.bfloat16, device=device
    )
    w1, s1 = _quantize_and_shuffle(
        w1_raw, model_dim=model_dim, inter_dim=inter_dim, stage=1
    )
    w2, s2 = _quantize_and_shuffle(
        w2_raw, model_dim=model_dim, inter_dim=inter_dim, stage=2
    )
    topk_ids, topk_weights = _boundary_routing(tokens, topk, (experts,), device)

    expected = fused_moe(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.per_1x32,
        w1_scale=s1,
        w2_scale=s2,
    )
    actual = fused_moe_multi_b(
        hidden,
        [w1],
        [w2],
        topk_weights,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.per_1x32,
        w1_scale_partitions=[s1],
        w2_scale_partitions=[s2],
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_multi_b_gfx1250_fails_without_launch(monkeypatch):
    monkeypatch.setattr("aiter.fused_moe.get_gfx", lambda: "gfx1250")
    model_dim = inter_dim = 64
    sizes = (1, 1)
    w1 = [
        torch.empty(size, 2 * inter_dim, model_dim // 2, dtype=torch.uint8)
        for size in sizes
    ]
    w2 = [
        torch.empty(size, model_dim, inter_dim // 2, dtype=torch.uint8)
        for size in sizes
    ]
    s1 = [torch.empty(size * 2 * inter_dim, 8, dtype=torch.uint8) for size in sizes]
    s2 = [torch.empty(size * model_dim, 8, dtype=torch.uint8) for size in sizes]
    hidden = torch.empty(1, model_dim)
    with pytest.raises(NotImplementedError, match="gfx950-only"):
        fused_moe_multi_b(
            hidden,
            w1,
            w2,
            torch.ones(1, 1),
            torch.zeros(1, 1, dtype=torch.int64),
            quant_type=aiter.QuantType.per_1x32,
            w1_scale_partitions=s1,
            w2_scale_partitions=s2,
        )


def test_multi_b_mx_layout_rejects_fp32_scales_explicitly():
    model_dim = inter_dim = 64
    sizes = (1, 1)
    w1 = [
        torch.empty(size, 2 * inter_dim, model_dim, dtype=torch.uint8) for size in sizes
    ]
    w2 = [torch.empty(size, model_dim, inter_dim, dtype=torch.uint8) for size in sizes]
    s1 = [torch.empty(size * 2 * inter_dim, 8, dtype=torch.float32) for size in sizes]
    s2 = [torch.empty(size * model_dim, 8, dtype=torch.float32) for size in sizes]
    with pytest.raises(TypeError, match="one-byte E8M0"):
        fused_moe_multi_b(
            torch.empty(1, model_dim),
            w1,
            w2,
            torch.ones(1, 1),
            torch.zeros(1, 1, dtype=torch.int64),
            quant_type=aiter.QuantType.per_1x32,
            w1_scale_partitions=s1,
            w2_scale_partitions=s2,
        )
