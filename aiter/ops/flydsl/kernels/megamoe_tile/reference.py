# SPDX-License-Identifier: MIT
from __future__ import annotations

import torch

from .activation import apply_gate_up, normalize_activation
from .topology import LogicalTopology, build_route_plan


def situ_v2(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    """Kimi-K3 SiTUv2 activation used by the accuracy oracle."""

    return (
        float(beta) * torch.tanh(gate / float(beta)) * torch.sigmoid(gate)
    ) * (float(linear_beta) * torch.tanh(up / float(linear_beta)))


def silu_gate(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return apply_gate_up(gate, up, "silu")


def swiglu_gate(
    gate: torch.Tensor, up: torch.Tensor, *, limit: float = 7.0
) -> torch.Tensor:
    return apply_gate_up(gate, up, "swiglu", swiglu_limit=limit)


def _expert_route(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    expert: int,
    *,
    activation: str,
    swiglu_limit: float | None,
    beta: float,
    linear_beta: float,
) -> torch.Tensor:
    gu = torch.mv(w1[expert].float(), x.float())
    gate, up = gu.chunk(2, dim=0)
    mid = apply_gate_up(
        gate,
        up,
        activation,
        swiglu_limit=swiglu_limit,
        situ_beta=beta,
        situ_linear_beta=linear_beta,
    )
    return torch.mv(w2[expert].float(), mid)


def dense_moe_reference(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    """Unpartitioned routing-weighted FP32 reference."""

    activation = normalize_activation(activation)
    m, topk = topk_ids.shape
    out = torch.zeros((m, w2.shape[1]), dtype=torch.float32, device=x.device)
    for token in range(m):
        for slot in range(topk):
            expert = int(topk_ids[token, slot])
            out[token] += float(topk_weights[token, slot]) * _expert_route(
                x[token],
                w1,
                w2,
                expert,
                activation=activation,
                swiglu_limit=swiglu_limit,
                beta=beta,
                linear_beta=linear_beta,
            )
    return out


def hierarchical_moe_reference(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    ep_world_size: int,
    logical_gpus_per_node: int,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    """Two-level dispatch/combine oracle with a copy-based RDMA stub.

    A token is copied once per destination node, expanded back into expert
    routes inside that node, reduced to one node partial, and finally reduced at
    the source.  This intentionally mirrors the future H1/H2 state machine.
    """

    activation = normalize_activation(activation)
    topology = LogicalTopology(ep_world_size, logical_gpus_per_node)
    plan = build_route_plan(topk_ids, num_experts=w1.shape[0], topology=topology)
    m, topk = topk_ids.shape
    d = int(w2.shape[1])
    node_partial = torch.zeros(
        (m, topology.num_nodes, d), dtype=torch.float32, device=x.device
    )

    # The clone is the host reference equivalent of one token/node wire record.
    wire_records: dict[tuple[int, int], torch.Tensor] = {}
    for token in range(m):
        for node in range(topology.num_nodes):
            if bool(plan.node_mask[token, node]):
                wire_records[(token, node)] = x[token].clone()

    for token in range(m):
        for slot in range(topk):
            expert = int(topk_ids[token, slot])
            node = int(plan.destination_node[token, slot])
            route_out = _expert_route(
                wire_records[(token, node)],
                w1,
                w2,
                expert,
                activation=activation,
                swiglu_limit=swiglu_limit,
                beta=beta,
                linear_beta=linear_beta,
            )
            node_partial[token, node] += float(topk_weights[token, slot]) * route_out

    return node_partial.sum(dim=1)
