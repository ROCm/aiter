# SPDX-License-Identifier: MIT
from __future__ import annotations

import torch

from aiter.ops.flydsl.kernels.megamoe_tile import (
    CopyTransport,
    HierMegaMoETileConfig,
    LogicalTopology,
    K3DispatchWireLayout,
    K3PartialWireLayout,
    build_route_plan,
    apply_gate_up,
    dense_moe_reference,
    hierarchical_moe_reference,
    hidden_fraction,
    normalize_activation,
    pack_dispatch_records,
    unpack_dispatch_records,
    timeline_overlap_ratio,
)


def test_production_and_stub_topologies():
    prod = HierMegaMoETileConfig.production_ep16(rank=9)
    assert prod.num_logical_nodes == 2
    assert prod.experts_per_rank == 56
    assert LogicalTopology(16, 8).proxy_rank(0, 9) == 1
    mapped = LogicalTopology(16, 8, tuple(range(100, 116)))
    assert mapped.proxy_pe(0, 9) == 101

    stub = HierMegaMoETileConfig.single_host_ep8_stub(rank=6)
    assert stub.num_logical_nodes == 2
    assert stub.experts_per_rank == 112
    assert LogicalTopology(8, 4).proxy_rank(0, 6) == 2


def test_activation_field_normalization_and_validation():
    assert normalize_activation("SiLU") == "silu"
    assert normalize_activation("Swi_GLU") == "swiglu"
    assert normalize_activation("SiTU-V2") == "situv2"
    assert normalize_activation("situ") == "situv2"
    assert HierMegaMoETileConfig.production_ep16(
        rank=0, activation="SiTUv2"
    ).activation == "situv2"

    import pytest

    with pytest.raises(ValueError, match="unsupported activation"):
        normalize_activation("siluv2")
    with pytest.raises(ValueError, match="swiglu_limit"):
        HierMegaMoETileConfig.production_ep16(
            rank=0, activation="swiglu", swiglu_limit=0.0
        )
    with pytest.raises(ValueError, match="does not apply"):
        HierMegaMoETileConfig.production_ep16(
            rank=0, activation="situv2", swiglu_limit=7.0
        )


def test_activation_oracle_matches_aiter_contract():
    gate = torch.tensor([-9.0, -1.0, 2.0, 11.0])
    up = torch.tensor([-12.0, 0.5, 3.0, 13.0])

    silu = apply_gate_up(gate, up, "silu")
    torch.testing.assert_close(silu, torch.nn.functional.silu(gate) * up)

    limit = 7.0
    gc = gate.clamp(max=limit)
    uc = up.clamp(min=-limit, max=limit)
    swiglu = apply_gate_up(gate, up, "swiglu")
    swiglu_expected = gc * torch.sigmoid(1.702 * gc) * (uc + 1.0)
    torch.testing.assert_close(swiglu, swiglu_expected)

    beta, linear_beta = 4.0, 25.0
    situ = apply_gate_up(gate, up, "situv2")
    situ_expected = (
        beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    ) * (linear_beta * torch.tanh(up / linear_beta))
    torch.testing.assert_close(situ, situ_expected)
    assert not torch.equal(silu, swiglu)
    assert not torch.equal(silu, situ)


def test_legacy_fused_moe_seam_rejects_ambiguous_swiglu_quant_path():
    import pytest

    from aiter.ops.flydsl.kernels.megamoe_tile.compute import a4w4_local

    empty = torch.empty(0)
    with pytest.raises(NotImplementedError, match="cannot guarantee A4W4"):
        a4w4_local(
            empty,
            empty,
            empty,
            empty,
            empty,
            w1_scale=empty,
            w2_scale=empty,
            activation="swiglu",
        )


def test_persistent_workspace_rejects_unreachable_work_shards():
    import pytest

    from aiter.ops.flydsl.kernels.megamoe_tile import PersistentH1Workspace

    with pytest.raises(ValueError, match="worker_blocks must be >= work_shards"):
        PersistentH1Workspace.allocate(
            work_shards=8, worker_blocks=4, device="cpu"
        )


def test_route_plan_node_deduplicates_but_preserves_routes():
    topology = LogicalTopology(8, 4)
    # E=16 => two experts/rank. Token 0 has two routes on each logical node.
    ids = torch.tensor([[0, 3, 8, 11], [2, 4, 5, 7]], dtype=torch.int64)
    plan = build_route_plan(ids, num_experts=16, topology=topology)
    assert plan.node_mask.tolist() == [[True, True], [True, False]]
    assert plan.node_route_expected.tolist() == [[2, 2], [4, 0]]
    assert plan.destination_rank.tolist()[0] == [0, 1, 4, 5]


def test_copy_transport_data_before_signal():
    src = torch.arange(64, dtype=torch.uint8)
    dst = torch.zeros_like(src)
    signal = torch.zeros(1, dtype=torch.int64)
    transport = CopyTransport()
    transport.put_signal(dst, src, 37, signal, 7)
    torch.testing.assert_close(dst[:37], src[:37], rtol=0, atol=0)
    assert signal.item() == 7
    assert transport.stats.bytes == 37


def test_hierarchical_reference_matches_dense_for_two_logical_nodes():
    torch.manual_seed(7)
    m, d, inter, experts, topk = 5, 16, 8, 16, 4
    x = torch.randn(m, d)
    w1 = torch.randn(experts, 2 * inter, d) * 0.1
    w2 = torch.randn(experts, d, inter) * 0.1
    ids = torch.stack([torch.randperm(experts)[:topk] for _ in range(m)])
    weights = torch.rand(m, topk)
    weights /= weights.sum(dim=1, keepdim=True)

    dense = dense_moe_reference(x, ids, weights, w1, w2)
    hier = hierarchical_moe_reference(
        x,
        ids,
        weights,
        w1,
        w2,
        ep_world_size=8,
        logical_gpus_per_node=4,
    )
    torch.testing.assert_close(hier, dense, rtol=1e-6, atol=1e-6)


def test_hierarchical_reference_matches_dense_for_all_activations():
    torch.manual_seed(71)
    m, d, inter, experts, topk = 2, 8, 4, 8, 2
    x = torch.randn(m, d)
    w1 = torch.randn(experts, 2 * inter, d) * 0.1
    w2 = torch.randn(experts, d, inter) * 0.1
    ids = torch.tensor([[0, 6], [2, 7]])
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]])

    for activation in ("silu", "swiglu", "situv2"):
        dense = dense_moe_reference(
            x, ids, weights, w1, w2, activation=activation
        )
        hier = hierarchical_moe_reference(
            x,
            ids,
            weights,
            w1,
            w2,
            ep_world_size=8,
            logical_gpus_per_node=4,
            activation=activation,
        )
        torch.testing.assert_close(hier, dense, rtol=1e-6, atol=1e-6)


def test_k3_wire_record_is_2048_bytes_and_roundtrips():
    layout = K3DispatchWireLayout()
    assert layout.record_bytes == 2048
    assert layout.records_per_64k == 32
    assert K3PartialWireLayout().record_bytes == 7424
    assert K3PartialWireLayout().records_per_64k == 8
    rows = 3
    hidden = torch.arange(rows * layout.hidden_bytes, dtype=torch.int32).remainder(251).to(torch.uint8).view(rows, -1)
    scales = torch.arange(rows * layout.scale_bytes, dtype=torch.int32).remainder(253).to(torch.uint8).view(rows, -1)
    ids = torch.arange(rows * 16, dtype=torch.int32).view(rows, 16)
    weights = torch.arange(rows * 16, dtype=torch.float32).view(rows, 16) / 100
    source = torch.tensor([3, 7, 11], dtype=torch.int32)
    masks = torch.tensor([0x00FF, 0xFF00, 0xA55A], dtype=torch.int64)
    record = pack_dispatch_records(hidden, scales, ids, weights, source, masks)
    fields = unpack_dispatch_records(record)
    torch.testing.assert_close(fields["hidden_fp4_u8"], hidden, rtol=0, atol=0)
    torch.testing.assert_close(fields["scales_e8m0_u8"], scales, rtol=0, atol=0)
    torch.testing.assert_close(fields["topk_ids"], ids, rtol=0, atol=0)
    torch.testing.assert_close(fields["topk_weights"], weights, rtol=0, atol=0)
    torch.testing.assert_close(fields["source_flat_token"], source, rtol=0, atol=0)
    torch.testing.assert_close(fields["destination_route_mask"], masks, rtol=0, atol=0)


def test_overlap_metrics():
    assert hidden_fraction(4.0, 10.0, 10.0) == 1.0
    assert hidden_fraction(4.0, 10.0, 12.0) == 0.5
    ratio = timeline_overlap_ratio([(0.0, 4.0)], [(2.0, 8.0)])
    assert ratio == 0.5
