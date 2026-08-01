# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only contract tests for the SharedEP MXFP4 adapter."""

import pytest
import torch

from aiter.ops.triton.moe.shared_ep_mxfp4 import (
    get_shared_ep_mxfp4_config,
    shared_ep_mxfp4_route_rows,
    validate_shared_ep_mxfp4_contract,
)

TOP_K = 2
BLOCK_M = 64
K = 128
NUM_EXPERTS = 2
CONFIG = {
    "BLOCK_SIZE_M": BLOCK_M,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": K,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
}


def _route_metadata() -> tuple[torch.Tensor, ...]:
    route_capacity = 6
    routes = torch.full((2 * BLOCK_M,), route_capacity, dtype=torch.int32)
    routes[:3] = torch.tensor([0, 3, 4], dtype=torch.int32)
    routes[BLOCK_M : BLOCK_M + 3] = torch.tensor([1, 2, 5], dtype=torch.int32)
    experts = torch.tensor([0, 1], dtype=torch.int32)
    num_padded = torch.tensor([2 * BLOCK_M], dtype=torch.int32)
    return routes, experts, num_padded


def _weight_and_scales(output_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.full((NUM_EXPERTS, output_dim, K // 2), 0x22, dtype=torch.uint8)
    scales = torch.empty((NUM_EXPERTS, output_dim, K // 32), dtype=torch.uint8)
    scales[0] = torch.tensor([127, 128, 126, 127], dtype=torch.uint8)
    scales[1] = torch.tensor([128, 127, 129, 126], dtype=torch.uint8)
    return weight, scales


def _dequant_mxfp4(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    packed = weight.view(torch.uint8)
    nibbles = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=torch.uint8)
    nibbles[..., ::2] = packed & 0xF
    nibbles[..., 1::2] = packed >> 4
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
    )
    values = lut[nibbles.long()]
    scale_values = torch.exp2(scales.view(torch.uint8).float() - 127.0)
    return values * scale_values.repeat_interleave(32, dim=-1)


def test_shared_ep_route_rows_are_canonical() -> None:
    route_ids = torch.tensor([5, 0, 3, 2], dtype=torch.int32)
    w13_input, w13_output = shared_ep_mxfp4_route_rows(
        route_ids, stage="w13", top_k=TOP_K
    )
    torch.testing.assert_close(w13_input, torch.tensor([2, 0, 1, 1], dtype=torch.int32))
    torch.testing.assert_close(w13_output, route_ids)

    w2_input, w2_output = shared_ep_mxfp4_route_rows(route_ids, stage="w2", top_k=TOP_K)
    torch.testing.assert_close(w2_input, route_ids)
    torch.testing.assert_close(w2_output, route_ids)


def test_shared_ep_w13_w2_contract_and_config_hook() -> None:
    routes, experts, num_padded = _route_metadata()
    w13_weight, w13_scales = _weight_and_scales(256)
    w13_profile = validate_shared_ep_mxfp4_contract(
        torch.ones((1, 3, K), dtype=torch.bfloat16),
        w13_weight,
        w13_scales,
        routes,
        experts,
        num_padded,
        stage="w13",
        top_k=TOP_K,
        route_block_size=BLOCK_M,
        out=torch.empty((6, 128), dtype=torch.bfloat16),
    )
    assert w13_profile.route_capacity == 6
    assert w13_profile.num_active_routes == 6

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if fp4_dtype is not None and e8m0_dtype is not None:
        tagged_profile = validate_shared_ep_mxfp4_contract(
            torch.ones((1, 3, K), dtype=torch.bfloat16),
            w13_weight.view(fp4_dtype),
            w13_scales.view(e8m0_dtype),
            routes,
            experts,
            num_padded,
            stage="w13",
            top_k=TOP_K,
            route_block_size=BLOCK_M,
        )
        assert tagged_profile.route_capacity == 6

    seen = []

    def config_hook(profile):
        seen.append(profile)
        return CONFIG

    selected = get_shared_ep_mxfp4_config(w13_profile, config_hook=config_hook)
    assert selected["BLOCK_SIZE_M"] == BLOCK_M
    assert seen == [w13_profile]

    w2_weight, w2_scales = _weight_and_scales(128)
    w2_profile = validate_shared_ep_mxfp4_contract(
        torch.ones((6, K), dtype=torch.bfloat16),
        w2_weight,
        w2_scales,
        routes,
        experts,
        num_padded,
        stage="w2",
        top_k=TOP_K,
        route_block_size=BLOCK_M,
        route_weights=torch.arange(6, dtype=torch.float32).view(3, TOP_K),
        out=torch.empty((6, 128), dtype=torch.bfloat16),
    )
    assert w2_profile.num_owner_rows == 3
    assert w2_profile.route_capacity == 6


def test_dsv4_pro_production_shapes_validate_without_materialization() -> None:
    device = torch.device("meta")
    owners = 8
    tokens_per_owner = 32
    top_k = 6
    hidden_size = 7168
    intermediate_size = 3072
    num_local_experts = 48
    block_m = 128
    owner_rows = owners * tokens_per_owner
    route_capacity = owner_rows * top_k
    sorted_capacity = route_capacity + num_local_experts * (block_m - 1)
    route_blocks = (sorted_capacity + block_m - 1) // block_m

    routes = torch.empty(sorted_capacity, dtype=torch.int32, device=device)
    experts = torch.empty(route_blocks, dtype=torch.int32, device=device)
    num_padded = torch.empty(1, dtype=torch.int32, device=device)

    w13_profile = validate_shared_ep_mxfp4_contract(
        torch.empty(
            (owners, tokens_per_owner, hidden_size), dtype=torch.bfloat16, device=device
        ),
        torch.empty(
            (num_local_experts, 2 * intermediate_size, hidden_size // 2),
            dtype=torch.uint8,
            device=device,
        ),
        torch.empty(
            (num_local_experts, 2 * intermediate_size, hidden_size // 32),
            dtype=torch.uint8,
            device=device,
        ),
        routes,
        experts,
        num_padded,
        stage="w13",
        top_k=top_k,
        route_block_size=block_m,
        out=torch.empty(
            (route_capacity, intermediate_size),
            dtype=torch.bfloat16,
            device=device,
        ),
        check_route_values=False,
    )
    assert w13_profile.num_owner_rows == owner_rows
    assert w13_profile.route_capacity == route_capacity
    assert w13_profile.num_padded_routes == sorted_capacity
    assert w13_profile.input_dim == hidden_size
    assert w13_profile.output_dim == intermediate_size

    w2_profile = validate_shared_ep_mxfp4_contract(
        torch.empty(
            (route_capacity, intermediate_size),
            dtype=torch.bfloat16,
            device=device,
        ),
        torch.empty(
            (num_local_experts, hidden_size, intermediate_size // 2),
            dtype=torch.uint8,
            device=device,
        ),
        torch.empty(
            (num_local_experts, hidden_size, intermediate_size // 32),
            dtype=torch.uint8,
            device=device,
        ),
        routes,
        experts,
        num_padded,
        stage="w2",
        top_k=top_k,
        route_block_size=block_m,
        route_weights=torch.empty(
            (owners, tokens_per_owner, top_k),
            dtype=torch.float32,
            device=device,
        ),
        out=torch.empty(
            (route_capacity, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        ),
        check_route_values=False,
    )
    assert w2_profile.num_owner_rows == owner_rows
    assert w2_profile.route_capacity == route_capacity
    assert w2_profile.input_dim == intermediate_size
    assert w2_profile.output_dim == hidden_size


def test_shared_ep_contract_fails_closed() -> None:
    routes, experts, num_padded = _route_metadata()
    weight, scales = _weight_and_scales(256)
    activations = torch.ones((3, K), dtype=torch.bfloat16)
    common = {
        "activations": activations,
        "weight": weight,
        "weight_scales": scales,
        "sorted_route_ids": routes,
        "expert_ids": experts,
        "num_tokens_post_padded": num_padded,
        "stage": "w13",
        "top_k": TOP_K,
        "route_block_size": BLOCK_M,
    }

    with pytest.raises(ValueError, match="canonical unswizzled"):
        validate_shared_ep_mxfp4_contract(
            **common,
            weight_layout="swizzled",
        )

    with pytest.raises(ValueError, match="unswizzled E8M0"):
        validate_shared_ep_mxfp4_contract(
            **common,
            scale_layout="swizzled",
        )

    duplicate_routes = routes.clone()
    duplicate_routes[1] = duplicate_routes[0]
    with pytest.raises(ValueError, match="unique"):
        validate_shared_ep_mxfp4_contract(
            **{**common, "sorted_route_ids": duplicate_routes}
        )

    with pytest.raises(TypeError, match="E8M0"):
        validate_shared_ep_mxfp4_contract(**{**common, "weight_scales": scales.float()})

    with pytest.raises(ValueError, match="one E8M0 byte"):
        validate_shared_ep_mxfp4_contract(
            **{**common, "weight_scales": scales[..., :-1].contiguous()}
        )

    with pytest.raises(ValueError, match="route capacity"):
        validate_shared_ep_mxfp4_contract(
            **common,
            out=torch.empty((5, 128), dtype=torch.bfloat16),
        )

    with pytest.raises(ValueError, match="contiguous route-major"):
        validate_shared_ep_mxfp4_contract(
            **common,
            out=torch.empty((6, 256), dtype=torch.bfloat16)[:, ::2],
        )


def test_shared_ep_fp4_reference_uses_e8m0_per_group() -> None:
    weight, scales = _weight_and_scales(1)
    dequant = _dequant_mxfp4(weight, scales)
    torch.testing.assert_close(dequant[0, 0, ::32], torch.tensor([1.0, 2.0, 0.5, 1.0]))
    torch.testing.assert_close(dequant[1, 0, ::32], torch.tensor([2.0, 1.0, 4.0, 0.5]))
