# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_MULTI_B_PATH = (
    Path(__file__).resolve().parents[1] / "aiter" / "ops" / "flydsl" / "multi_b.py"
)
_MULTI_B_SPEC = importlib.util.spec_from_file_location(
    "_aiter_multi_b_contract", _MULTI_B_PATH
)
assert _MULTI_B_SPEC is not None and _MULTI_B_SPEC.loader is not None
_MULTI_B_MODULE = importlib.util.module_from_spec(_MULTI_B_SPEC)
sys.modules[_MULTI_B_SPEC.name] = _MULTI_B_MODULE
_MULTI_B_SPEC.loader.exec_module(_MULTI_B_MODULE)

expert_partition_index = _MULTI_B_MODULE.expert_partition_index
partition_module_tag = _MULTI_B_MODULE.partition_module_tag
partition_offsets = _MULTI_B_MODULE.partition_offsets
validate_multi_b_partitions = _MULTI_B_MODULE.validate_multi_b_partitions


def _make_contract_parts(
    sizes: tuple[int, ...], model_dim: int = 64, inter_dim: int = 64
):
    w1 = [
        torch.empty((size, 2 * inter_dim, model_dim // 2), dtype=torch.uint8)
        for size in sizes
    ]
    w2 = [
        torch.empty((size, model_dim, inter_dim // 2), dtype=torch.uint8)
        for size in sizes
    ]
    scale1_cols = (model_dim // 32 + 7) // 8 * 8
    scale2_cols = (inter_dim // 32 + 7) // 8 * 8
    w1_scale = [
        torch.empty((size * 2 * inter_dim, scale1_cols), dtype=torch.uint8)
        for size in sizes
    ]
    w2_scale = [
        torch.empty((size * model_dim, scale2_cols), dtype=torch.uint8)
        for size in sizes
    ]
    return w1, w2, w1_scale, w2_scale


def _make_fp8_contract_parts(
    sizes: tuple[int, ...], model_dim: int = 256, inter_dim: int = 256
):
    fp8_dtype = torch.float8_e4m3fn
    w1 = [
        torch.empty((size, 2 * inter_dim, model_dim), dtype=fp8_dtype) for size in sizes
    ]
    w2 = [torch.empty((size, model_dim, inter_dim), dtype=fp8_dtype) for size in sizes]
    w1_scale = [
        torch.empty(
            (size, 2 * inter_dim // 128, model_dim // 128),
            dtype=torch.float32,
        )
        for size in sizes
    ]
    w2_scale = [
        torch.empty(
            (size, model_dim // 128, inter_dim // 128),
            dtype=torch.float32,
        )
        for size in sizes
    ]
    return w1, w2, w1_scale, w2_scale


@pytest.mark.parametrize(
    "sizes",
    [
        (4, 4),
        (2, 2, 2, 2),
        (1, 1, 1, 1, 1, 1, 1, 1),
        (64, 64, 64, 65),
    ],
)
def test_multi_b_contract_accepts_even_and_terminal_uneven_partitions(sizes):
    w1, w2, w1_scale, w2_scale = _make_contract_parts(sizes)
    spec = validate_multi_b_partitions(w1, w2, w1_scale, w2_scale)
    assert spec.partition_sizes == sizes
    assert spec.experts == sum(sizes)
    assert spec.model_dim == 64
    assert spec.inter_dim == 64
    assert spec.scale_layout == "mx_1x32"


@pytest.mark.parametrize(
    "sizes",
    [
        (3, 5),
        (1, 2, 2, 3),
        (1, 1, 1, 1, 1, 1, 1, 2),
    ],
)
def test_fp8_blockscale_contract_accepts_exact_rank3_scales(sizes):
    w1, w2, w1_scale, w2_scale = _make_fp8_contract_parts(sizes)
    spec = validate_multi_b_partitions(
        w1,
        w2,
        w1_scale,
        w2_scale,
        scale_layout="fp8_128x128",
    )
    assert spec.partition_sizes == sizes
    assert spec.experts == sum(sizes)
    assert spec.model_dim == 256
    assert spec.inter_dim == 256
    assert spec.packing_ratio == 1
    assert spec.scale_layout == "fp8_128x128"


@pytest.mark.parametrize(
    "sizes",
    [
        (4, 4),
        (2, 2, 2, 2),
        (1, 1, 1, 1, 1, 1, 1, 1),
        (64, 64, 64, 65),
    ],
)
def test_global_expert_ids_map_deterministically_across_every_boundary(sizes):
    offsets = partition_offsets(sizes)
    ids = {0, offsets[-1] - 1}
    for boundary in offsets[1:-1]:
        ids.update((boundary - 1, boundary))

    for expert_id in sorted(ids):
        partition, local_id = expert_partition_index(expert_id, sizes)
        assert expert_id == offsets[partition] + local_id
        assert 0 <= local_id < sizes[partition]


def test_partition_module_identity_includes_uneven_terminal_size():
    assert partition_module_tag((64, 64, 64, 65)) == "_mb64x64x64x65"
    assert partition_module_tag((64, 64, 64, 64)) != partition_module_tag(
        (64, 64, 64, 65)
    )


def test_multi_b_contract_rejects_scale_cardinality_and_shape_errors():
    sizes = (2, 2)
    w1, w2, w1_scale, w2_scale = _make_contract_parts(sizes)

    with pytest.raises(ValueError, match="cardinality"):
        validate_multi_b_partitions(w1, w2, w1_scale[:1], w2_scale)

    bad_shape = list(w2_scale)
    bad_shape[1] = torch.empty((127, 8), dtype=torch.uint8)
    with pytest.raises(ValueError, match=r"w2_scale_partitions\[1\].*shape"):
        validate_multi_b_partitions(w1, w2, w1_scale, bad_shape)

    fp32_scale = list(w1_scale)
    fp32_scale[0] = fp32_scale[0].float()
    with pytest.raises(TypeError, match="one-byte E8M0"):
        validate_multi_b_partitions(w1, w2, fp32_scale, w2_scale)


def test_fp8_blockscale_contract_rejects_wrong_scale_format_and_shape():
    sizes = (2, 3)
    w1, w2, w1_scale, w2_scale = _make_fp8_contract_parts(sizes)

    e8m0_scale = list(w1_scale)
    e8m0_scale[0] = torch.empty(e8m0_scale[0].shape, dtype=torch.uint8)
    with pytest.raises(TypeError, match="contiguous FP32.*fp8_128x128"):
        validate_multi_b_partitions(
            w1,
            w2,
            e8m0_scale,
            w2_scale,
            scale_layout="fp8_128x128",
        )

    flat_scale = list(w2_scale)
    flat_scale[1] = flat_scale[1].view(flat_scale[1].shape[0], -1)
    with pytest.raises(ValueError, match="exact rank-3 shape"):
        validate_multi_b_partitions(
            w1,
            w2,
            w1_scale,
            flat_scale,
            scale_layout="fp8_128x128",
        )


def test_fp8_blockscale_contract_rejects_non_fp8_payloads():
    sizes = (1, 1)
    w1, w2, w1_scale, w2_scale = _make_fp8_contract_parts(sizes)
    w1[1] = w1[1].view(torch.uint8)
    with pytest.raises(TypeError, match="weight dtype mismatch"):
        validate_multi_b_partitions(
            w1,
            w2,
            w1_scale,
            w2_scale,
            scale_layout="fp8_128x128",
        )


def test_multi_b_contract_rejects_unsupported_partition_count():
    w1, w2, w1_scale, w2_scale = _make_contract_parts((1, 1, 1))
    with pytest.raises(ValueError, match="2/4/8"):
        validate_multi_b_partitions(w1, w2, w1_scale, w2_scale)


def test_multi_b_contract_rejects_bad_bias_shape():
    sizes = (2, 3)
    w1, w2, w1_scale, w2_scale = _make_contract_parts(sizes)
    with pytest.raises(ValueError, match="indexed by global expert id"):
        validate_multi_b_partitions(
            w1,
            w2,
            w1_scale,
            w2_scale,
            bias1=torch.empty((4, 128), dtype=torch.float32),
        )


def _partitioned_torch_moe(
    hidden: torch.Tensor,
    w1_parts: tuple[torch.Tensor, ...],
    w2_parts: tuple[torch.Tensor, ...],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
) -> torch.Tensor:
    sizes = tuple(weight.shape[0] for weight in w1_parts)
    tokens, topk = topk_ids.shape
    output = torch.zeros((tokens, hidden.shape[1]), dtype=torch.float32)
    for token in range(tokens):
        for slot in range(topk):
            global_expert = int(topk_ids[token, slot])
            partition, local_expert = expert_partition_index(global_expert, sizes)
            gate_up = (
                w1_parts[partition][local_expert] @ hidden[token] + bias1[global_expert]
            )
            gate, up = gate_up.chunk(2)
            intermediate = F.silu(gate) * up
            route = (
                w2_parts[partition][local_expert] @ intermediate + bias2[global_expert]
            )
            output[token] += topk_weights[token, slot] * route
    return output


@pytest.mark.parametrize("tokens", [1, 7, 19])
@pytest.mark.parametrize("partitions", [2, 4, 8])
def test_partitioned_routing_and_global_bias_match_contiguous_torch_oracle(
    tokens, partitions
):
    torch.manual_seed(17 + tokens + partitions)
    experts, model_dim, inter_dim, topk = 8, 8, 6, 2
    sizes = (experts // partitions,) * partitions
    hidden = torch.randn(tokens, model_dim)
    w1 = torch.randn(experts, 2 * inter_dim, model_dim)
    w2 = torch.randn(experts, model_dim, inter_dim)
    bias1 = torch.arange(experts, dtype=torch.float32)[:, None].expand(
        experts, 2 * inter_dim
    )
    bias2 = -torch.arange(experts, dtype=torch.float32)[:, None].expand(
        experts, model_dim
    )

    boundary_ids = []
    for boundary in partition_offsets(sizes)[1:-1]:
        boundary_ids.extend((boundary - 1, boundary))
    ids = torch.tensor(boundary_ids or [0, experts - 1], dtype=torch.long)
    repeats = (tokens * topk + ids.numel() - 1) // ids.numel()
    topk_ids = ids.repeat(repeats)[: tokens * topk].view(tokens, topk)
    topk_weights = torch.rand(tokens, topk)

    w1_parts = tuple(w1.split(sizes))
    w2_parts = tuple(w2.split(sizes))
    got = _partitioned_torch_moe(
        hidden, w1_parts, w2_parts, topk_ids, topk_weights, bias1, bias2
    )

    gate_up = torch.einsum("tki,tkoi->tko", hidden[:, None, :], w1[topk_ids])
    gate_up = gate_up + bias1[topk_ids]
    gate, up = gate_up.chunk(2, dim=-1)
    intermediate = F.silu(gate) * up
    route = torch.einsum("tki,tkoi->tko", intermediate, w2[topk_ids])
    route = route + bias2[topk_ids]
    expected = (route * topk_weights[..., None]).sum(dim=1)
    torch.testing.assert_close(got, expected)
