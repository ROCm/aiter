# SPDX-License-Identifier: MIT
"""Allocation and AITER boundary for the experimental MiniMax M3 hybrid."""

import torch
import aiter
from aiter.fused_moe import fused_moe_2stages

from .front import front
from .tail import tail
from .pipeline import Shape, launch
from .hip_ipc import DevicePointer, PeerMemory

SORT_BLOCK = 16
HIDDEN = 6144
EXPERTS = 129
TOPK = 5


def _require_expert_abi(config):
    first = (getattr(config.stage1, "keywords", None) or {}).get("kernelName", "")
    second = (getattr(config.stage2, "keywords", None) or {}).get("kernelName", "")
    if (
        not first.startswith("flydsl_moe1_abf16_wfp4_")
        or not second.startswith("flydsl_moe2_abf16_wfp4_")
        or "atomic" not in second
        or config.block_m != SORT_BLOCK
        or config.run_1stage
        or config.prequant
    ):
        raise ValueError(
            "M3 fused metadata requires A16W4 BM16 inline-activation two-stage MoE"
        )
    return config


def allocate_workspace(tokens, device):
    """Call once per decode width before graph capture; no sorting launch."""
    Shape(tokens)
    rows = tokens * TOPK
    # AITER's upper bound is at most one BM16 block per route. The shared
    # expert is repeated, so actual padded rows are usually much smaller.
    padded = rows * SORT_BLOCK
    return {
        "HMid": torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device),
        "XNorm": torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device),
        "Scores": torch.empty((tokens, 128), dtype=torch.int32, device=device),
        "Sigmoids": torch.empty((tokens, 128), dtype=torch.float32, device=device),
        "RouteIDs": torch.empty((tokens, TOPK), dtype=torch.int32, device=device),
        "RouteWeights": torch.empty((tokens, TOPK), dtype=torch.float32, device=device),
        "SortIDs": torch.empty(padded, dtype=torch.int32, device=device),
        "SortWeights": torch.empty(padded, dtype=torch.float32, device=device),
        "SortExperts": torch.empty(
            padded // SORT_BLOCK, dtype=torch.int32, device=device
        ),
        "SortCounts": torch.empty(2, dtype=torch.int32, device=device),
        "SortM": torch.empty(padded, dtype=torch.int32, device=device),
        "SortReverse": torch.empty(rows, dtype=torch.int32, device=device),
        "SortOut": torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device),
        "Out": torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device),
        "Ready": torch.zeros(tokens * 8, dtype=torch.uint64, device=device),
        "FrontEpochs": torch.zeros(256, dtype=torch.int32, device=device),
        "TailEpochs": torch.zeros(256, dtype=torch.int32, device=device),
        "FrontErrors": torch.zeros(1, dtype=torch.int32, device=device),
        "TailErrors": torch.zeros(256 * 64, dtype=torch.int32, device=device),
    }


def allocate_peer(tokens, rank, cpu_group):
    """Persistent two-slot TP4 IPC region; allocate before graph capture."""
    Shape(tokens)
    if not 0 <= rank < 4:
        raise ValueError("Invalid TP rank")
    pairs = tokens * HIDDEN // 2
    return PeerMemory(2 * 4 * pairs * 8, rank, 4, group=cpu_group)


def expert_weights(w13, s13, w2, s2):
    """Use only the ATOM TP4 packed/shuffled 129-expert A16W4 layout."""
    if w13.shape[0] != EXPERTS or w2.shape[0] != EXPERTS:
        raise ValueError("Expected 128 routed experts plus shared expert")
    w13 = w13.view(torch.float4_e2m1fn_x2)
    w2 = w2.view(torch.float4_e2m1fn_x2)
    w13.is_shuffled = w2.is_shuffled = True
    return (w13, s13.view(torch.float8_e8m0fnu), w2, s2.view(torch.float8_e8m0fnu))


def run_prepared_experts(x, weights, state, swiglu_limit=7.0):
    w13, s13, w2, s2 = weights
    if x.shape != (state["RouteIDs"].shape[0], HIDDEN):
        raise ValueError("Invalid M3 normalized activation shape")
    return fused_moe_2stages(
        x,
        w13,
        w2,
        TOPK,
        state["SortIDs"],
        state["SortWeights"],
        state["SortExperts"],
        state["SortCounts"],
        state["SortOut"],
        True,
        SORT_BLOCK,
        activation=aiter.ActivationType.Swiglu,
        quant_type=aiter.QuantType.per_1x32,
        q_dtype_a=torch.bfloat16,
        q_dtype_w=torch.float4_e2m1fn_x2,
        w1_scale=s13,
        w2_scale=s2,
        swiglu_limit=swiglu_limit,
        gate_mode="separated",
        topk_ids=state["RouteIDs"],
        topk_weights=state["RouteWeights"],
        routing_num_experts=EXPERTS,
        _metadata_transform=_require_expert_abi,
    )


def launch_region(
    attn_reduced,
    residual,
    gain,
    gate,
    bias,
    weights,
    state,
    peer,
    rank,
    tokens,
    shared_weight,
    route_scale=2.0,
    eps=1.0e-6,
):
    """Enqueue front -> selected AITER experts -> direct peer-reduce tail.

    `peer` is a live TP4 PeerMemory instance with 2*4*(B*6144/2) uint64
    messages. All ranks must launch the same width/epoch on one stream.
    """
    shape = Shape(tokens)
    if not 0 <= rank < shape.tp:
        raise ValueError("Invalid TP rank")
    if attn_reduced.shape != (tokens, HIDDEN) or residual.shape != (tokens, HIDDEN):
        raise ValueError("Expected Bx6144 attention partial and residual")
    if gate.shape != (128, HIDDEN) or bias.numel() != 128 or gain.numel() != HIDDEN:
        raise ValueError("Unexpected router/gain shape")
    if peer.addresses.numel() != 4:
        raise ValueError("Expected four IPC peer pointers")

    def front_call():
        front[(shape.grid,)](
            attn_reduced,
            residual,
            gain,
            gate,
            bias,
            state["HMid"],
            state["XNorm"],
            state["Scores"],
            state["Sigmoids"],
            state["RouteIDs"],
            state["RouteWeights"],
            state["SortIDs"],
            state["SortWeights"],
            state["SortExperts"],
            state["SortCounts"],
            state["SortM"],
            state["SortReverse"],
            state["SortOut"],
            state["Ready"],
            state["FrontEpochs"],
            state["FrontErrors"],
            B=tokens,
            EPS=eps,
            ROUTE_SCALE=route_scale,
            SHARED_WEIGHT=shared_weight,
            num_warps=8,
            enable_fp_fusion=False,
        )

    def experts_call():
        return run_prepared_experts(state["XNorm"], weights, state)

    def tail_call(routed):
        tail[(shape.grid,)](
            routed,
            peer.addresses,
            DevicePointer(peer.ptr.value),
            state["Out"],
            state["TailEpochs"],
            state["TailErrors"],
            B=tokens,
            RANK=rank,
            num_warps=8,
        )

    launch(
        shape,
        front=front_call,
        experts=experts_call,
        tail=tail_call,
        front_args=(),
        expert_args=(),
        tail_args=(),
    )
    return state["Out"], state["HMid"]
