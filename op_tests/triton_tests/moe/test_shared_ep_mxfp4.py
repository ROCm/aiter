# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.moe.shared_ep_mxfp4 import (
    get_shared_ep_mxfp4_config,
    shared_ep_mxfp4_route_rows,
    shared_ep_mxfp4_w2,
    shared_ep_mxfp4_w13,
    validate_shared_ep_mxfp4_contract,
)

_TOP_K = 2
_BLOCK_M = 64
_K = 128
_NUM_EXPERTS = 2
_ROUTE_EXPERT = {0: 0, 3: 0, 4: 0, 1: 1, 2: 1, 5: 1}
_CONFIG = {
    "BLOCK_SIZE_M": _BLOCK_M,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": _K,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16,
    "kpack": 1,
}


def _route_metadata(device: torch.device) -> tuple[torch.Tensor, ...]:
    route_capacity = 6
    sorted_route_ids = torch.full(
        (2 * _BLOCK_M,), route_capacity, dtype=torch.int32, device=device
    )
    sorted_route_ids[:3] = torch.tensor([0, 3, 4], dtype=torch.int32, device=device)
    sorted_route_ids[_BLOCK_M : _BLOCK_M + 3] = torch.tensor(
        [1, 2, 5], dtype=torch.int32, device=device
    )
    expert_ids = torch.tensor([0, 1], dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.tensor(
        [2 * _BLOCK_M], dtype=torch.int32, device=device
    )
    return sorted_route_ids, expert_ids, num_tokens_post_padded


def _weight_and_scales(
    output_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    # Packed nibble 0x2 is E2M1 +1.0.  Distinct E8M0 exponents exercise
    # non-unit scales in every 32-wide logical K group.
    weight = torch.full(
        (_NUM_EXPERTS, output_dim, _K // 2),
        0x22,
        dtype=torch.uint8,
        device=device,
    )
    scales = torch.empty(
        (_NUM_EXPERTS, output_dim, _K // 32),
        dtype=torch.uint8,
        device=device,
    )
    scales[0] = torch.tensor([127, 128, 126, 127], dtype=torch.uint8, device=device)
    scales[1] = torch.tensor([128, 127, 129, 126], dtype=torch.uint8, device=device)
    return weight, scales


def _dequant_mxfp4(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    packed = weight.view(torch.uint8)
    nibbles = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.uint8,
        device=packed.device,
    )
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
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    values = lut[nibbles.long()]
    scale_values = torch.exp2(scales.view(torch.uint8).float() - 127.0)
    return values * scale_values.repeat_interleave(32, dim=-1)


def _require_gfx950() -> None:
    if not torch.cuda.is_available():
        pytest.skip("ROCm device is not available")
    try:
        from aiter.ops.triton.utils._triton import arch_info

        arch = str(arch_info.get_arch()).split(":", 1)[0]
    except (AttributeError, ImportError, RuntimeError) as exc:  # pragma: no cover
        pytest.skip(f"unable to query ROCm architecture: {exc}")
    if arch != "gfx950":
        pytest.skip(f"SharedEP MXFP4 GPU test requires gfx950, got {arch}")


def test_route_indexing_contract() -> None:
    route_ids = torch.tensor([5, 0, 3, 2], dtype=torch.int32)

    w13_input, w13_output = shared_ep_mxfp4_route_rows(route_ids, stage="w13", top_k=2)
    torch.testing.assert_close(w13_input, torch.tensor([2, 0, 1, 1], dtype=torch.int32))
    torch.testing.assert_close(w13_output, route_ids)

    w2_input, w2_output = shared_ep_mxfp4_route_rows(route_ids, stage="w2", top_k=2)
    torch.testing.assert_close(w2_input, route_ids)
    torch.testing.assert_close(w2_output, route_ids)


def test_cpu_contract_w13_w2_and_config_hook() -> None:
    device = torch.device("cpu")
    routes, experts, num_padded = _route_metadata(device)
    w13_weight, w13_scales = _weight_and_scales(256, device)
    activations = torch.ones((3, _K), dtype=torch.bfloat16)
    w13_out = torch.empty((6, 128), dtype=torch.bfloat16)

    w13_profile = validate_shared_ep_mxfp4_contract(
        activations,
        w13_weight,
        w13_scales,
        routes,
        experts,
        num_padded,
        stage="w13",
        top_k=_TOP_K,
        route_block_size=_BLOCK_M,
        out=w13_out,
    )
    assert w13_profile.route_capacity == 6
    assert w13_profile.num_active_routes == 6
    assert w13_profile.output_dim == 128

    seen = []

    def config_hook(profile):
        seen.append(profile)
        return _CONFIG

    assert (
        get_shared_ep_mxfp4_config(w13_profile, config_hook=config_hook)["BLOCK_SIZE_M"]
        == _BLOCK_M
    )
    assert seen == [w13_profile]

    w2_weight, w2_scales = _weight_and_scales(128, device)
    intermediate = torch.ones((6, _K), dtype=torch.bfloat16)
    route_weights = torch.arange(6, dtype=torch.float32).view(3, _TOP_K)
    w2_out = torch.empty((6, 128), dtype=torch.bfloat16)
    w2_profile = validate_shared_ep_mxfp4_contract(
        intermediate,
        w2_weight,
        w2_scales,
        routes,
        experts,
        num_padded,
        stage="w2",
        top_k=_TOP_K,
        route_block_size=_BLOCK_M,
        route_weights=route_weights,
        out=w2_out,
    )
    assert w2_profile.num_owner_rows == 3
    assert w2_profile.route_capacity == 6


def test_cpu_contract_rejects_unsafe_layout_routes_and_scales() -> None:
    device = torch.device("cpu")
    routes, experts, num_padded = _route_metadata(device)
    weight, scales = _weight_and_scales(256, device)
    activations = torch.ones((3, _K), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="canonical unswizzled"):
        validate_shared_ep_mxfp4_contract(
            activations,
            weight,
            scales,
            routes,
            experts,
            num_padded,
            stage="w13",
            top_k=_TOP_K,
            route_block_size=_BLOCK_M,
            weight_layout="swizzled",
        )

    with pytest.raises(ValueError, match="unswizzled E8M0"):
        validate_shared_ep_mxfp4_contract(
            activations,
            weight,
            scales,
            routes,
            experts,
            num_padded,
            stage="w13",
            top_k=_TOP_K,
            route_block_size=_BLOCK_M,
            scale_layout="swizzled",
        )

    duplicate_routes = routes.clone()
    duplicate_routes[1] = duplicate_routes[0]
    with pytest.raises(ValueError, match="unique"):
        validate_shared_ep_mxfp4_contract(
            activations,
            weight,
            scales,
            duplicate_routes,
            experts,
            num_padded,
            stage="w13",
            top_k=_TOP_K,
            route_block_size=_BLOCK_M,
        )

    with pytest.raises(TypeError, match="E8M0"):
        validate_shared_ep_mxfp4_contract(
            activations,
            weight,
            scales.float(),
            routes,
            experts,
            num_padded,
            stage="w13",
            top_k=_TOP_K,
            route_block_size=_BLOCK_M,
        )

    with pytest.raises(ValueError, match="one E8M0 byte"):
        validate_shared_ep_mxfp4_contract(
            activations,
            weight,
            scales[..., :-1].contiguous(),
            routes,
            experts,
            num_padded,
            stage="w13",
            top_k=_TOP_K,
            route_block_size=_BLOCK_M,
        )


def test_fp4_e8m0_reference_has_per_group_scales() -> None:
    weight, scales = _weight_and_scales(1, torch.device("cpu"))
    dequant = _dequant_mxfp4(weight, scales)
    torch.testing.assert_close(
        dequant[0, 0, ::32],
        torch.tensor([1.0, 2.0, 0.5, 1.0]),
    )
    torch.testing.assert_close(
        dequant[1, 0, ::32],
        torch.tensor([2.0, 1.0, 4.0, 0.5]),
    )


def test_gfx950_shared_ep_w13_w2_reference() -> None:
    _require_gfx950()
    device = torch.device("cuda")
    routes, experts, num_padded = _route_metadata(device)

    activations = (
        torch.tensor(
            [1.0 / 64, 1.0 / 32, 1.0 / 16],
            dtype=torch.bfloat16,
            device=device,
        )[:, None]
        .expand(3, _K)
        .contiguous()
    )
    w13_weight, w13_scales = _weight_and_scales(256, device)
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", torch.uint8)
    w13_out = shared_ep_mxfp4_w13(
        activations,
        w13_weight.view(fp4_dtype),
        w13_scales.view(e8m0_dtype),
        routes,
        experts,
        num_padded,
        top_k=_TOP_K,
        route_block_size=_BLOCK_M,
        config=_CONFIG,
        swiglu_limit=0.05,
    )

    dense_w13 = _dequant_mxfp4(w13_weight, w13_scales)
    w13_ref = torch.empty_like(w13_out)
    for route_id, expert_id in _ROUTE_EXPERT.items():
        projected = torch.mv(
            dense_w13[expert_id], activations[route_id // _TOP_K].float()
        ).to(torch.bfloat16)
        gate, up = projected.float().chunk(2)
        gate = gate.clamp(max=0.05)
        up = up.clamp(min=-0.05, max=0.05)
        w13_ref[route_id] = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    torch.testing.assert_close(w13_out, w13_ref, rtol=5e-2, atol=2e-1)

    intermediate = (
        torch.tensor(
            [1.0 / 64, 1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2],
            dtype=torch.bfloat16,
            device=device,
        )[:, None]
        .expand(6, _K)
        .contiguous()
    )
    w2_weight, w2_scales = _weight_and_scales(128, device)
    route_weights = torch.tensor(
        [[0.10, 0.20], [0.30, 0.40], [0.50, 0.60]],
        dtype=torch.float32,
        device=device,
    )
    w2_out = shared_ep_mxfp4_w2(
        intermediate,
        w2_weight,
        w2_scales,
        route_weights,
        routes,
        experts,
        num_padded,
        top_k=_TOP_K,
        route_block_size=_BLOCK_M,
        config=_CONFIG,
    )

    dense_w2 = _dequant_mxfp4(w2_weight, w2_scales)
    w2_ref = torch.empty_like(w2_out)
    flat_route_weights = route_weights.view(-1)
    for route_id, expert_id in _ROUTE_EXPERT.items():
        projected = torch.mv(dense_w2[expert_id], intermediate[route_id].float())
        w2_ref[route_id] = (projected * flat_route_weights[route_id]).to(torch.bfloat16)
    torch.testing.assert_close(w2_out, w2_ref, rtol=5e-2, atol=2e-1)


def test_gfx950_dsv4_pro_shape_graph_replay_and_invalid_route_guard() -> None:
    _require_gfx950()
    device = torch.device("cuda")
    hidden_size = 7168
    intermediate_size = 3072
    top_k = 6
    owner_rows = 32
    route_capacity = owner_rows * top_k
    block_m = 128
    padded_routes = 2 * block_m
    config = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 4,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    }

    routes = torch.full(
        (padded_routes,),
        route_capacity,
        dtype=torch.int32,
        device=device,
    )
    routes[:route_capacity] = torch.arange(
        route_capacity,
        dtype=torch.int32,
        device=device,
    )
    experts = torch.zeros(
        (padded_routes // block_m,),
        dtype=torch.int32,
        device=device,
    )
    num_padded = torch.tensor([padded_routes], dtype=torch.int32, device=device)
    activations = torch.full(
        (owner_rows, hidden_size),
        1.0 / 4096,
        dtype=torch.bfloat16,
        device=device,
    )
    w13_weight = torch.full(
        (1, 2 * intermediate_size, hidden_size // 2),
        0x22,
        dtype=torch.uint8,
        device=device,
    )
    w13_scales = torch.full(
        (1, 2 * intermediate_size, hidden_size // 32),
        127,
        dtype=torch.uint8,
        device=device,
    )
    w2_weight = torch.full(
        (1, hidden_size, intermediate_size // 2),
        0x22,
        dtype=torch.uint8,
        device=device,
    )
    w2_scales = torch.full(
        (1, hidden_size, intermediate_size // 32),
        127,
        dtype=torch.uint8,
        device=device,
    )
    route_weights = (
        torch.arange(1, top_k + 1, dtype=torch.float32, device=device)
        .div_(top_k)
        .expand(owner_rows, -1)
        .contiguous()
    )

    eager_w13 = torch.empty(
        (route_capacity, intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )
    assert (
        shared_ep_mxfp4_w13(
            activations,
            w13_weight,
            w13_scales,
            routes,
            experts,
            num_padded,
            top_k=top_k,
            route_block_size=block_m,
            out=eager_w13,
            config=config,
            swiglu_limit=0.05,
        ).data_ptr()
        == eager_w13.data_ptr()
    )
    projected = torch.tensor(
        hidden_size / 4096,
        dtype=torch.bfloat16,
        device=device,
    ).float()
    expected_w13 = (
        torch.nn.functional.silu(projected.clamp(max=0.05))
        * projected.clamp(min=-0.05, max=0.05)
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        eager_w13,
        torch.ones_like(eager_w13) * expected_w13,
        rtol=5e-2,
        atol=5e-2,
    )

    eager_w2 = torch.empty(
        (route_capacity, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    assert (
        shared_ep_mxfp4_w2(
            eager_w13,
            w2_weight,
            w2_scales,
            route_weights,
            routes,
            experts,
            num_padded,
            top_k=top_k,
            route_block_size=block_m,
            out=eager_w2,
            config=config,
        ).data_ptr()
        == eager_w2.data_ptr()
    )
    expected_w2_rows = (
        eager_w13[:, 0].float() * intermediate_size * route_weights.view(-1)
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        eager_w2,
        expected_w2_rows[:, None].expand_as(eager_w2),
        # W13 and W2 each quantize BF16 activations to E2M1. The production
        # multi-tile path compounds the representable E2M1 rounding error.
        rtol=1.6e-1,
        atol=6e-1,
    )

    graph_w13 = torch.empty_like(eager_w13)
    graph_w2 = torch.empty_like(eager_w2)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        shared_ep_mxfp4_w13(
            activations,
            w13_weight,
            w13_scales,
            routes,
            experts,
            num_padded,
            top_k=top_k,
            route_block_size=block_m,
            out=graph_w13,
            config=config,
            swiglu_limit=0.05,
            check_route_values=False,
        )
        shared_ep_mxfp4_w2(
            graph_w13,
            w2_weight,
            w2_scales,
            route_weights,
            routes,
            experts,
            num_padded,
            top_k=top_k,
            route_block_size=block_m,
            out=graph_w2,
            config=config,
            check_route_values=False,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_w13, eager_w13, rtol=0, atol=0)
    torch.testing.assert_close(graph_w2, eager_w2, rtol=0, atol=0)

    graph_w13.fill_(7)
    graph_w2.fill_(7)
    num_padded.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(graph_w13 == 7)
    assert torch.all(graph_w2 == 7)

    routes[0] = -1
    num_padded.fill_(padded_routes)
    guarded_w13 = torch.full_like(eager_w13, 7)
    shared_ep_mxfp4_w13(
        activations,
        w13_weight,
        w13_scales,
        routes,
        experts,
        num_padded,
        top_k=top_k,
        route_block_size=block_m,
        out=guarded_w13,
        config=config,
        swiglu_limit=0.05,
        check_route_values=False,
    )
    torch.cuda.synchronize()
    assert torch.all(guarded_w13[0] == 7)
    torch.testing.assert_close(guarded_w13[1:], eager_w13[1:], rtol=0, atol=0)
