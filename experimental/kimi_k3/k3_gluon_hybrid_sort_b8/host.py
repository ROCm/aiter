"""Baseline AITER SiTUv2 experts between two same-stream Gluon launches."""

import torch
import aiter
from aiter.fused_moe import fused_moe, fused_moe_2stages, _adaptive_moe_sort
from .pipeline import require_expert_config


def expert_weights(weights):
    result = dict(
        w1=weights["WUp"].view(torch.float4_e2m1fn_x2),
        w2=weights["WDown"].view(torch.float4_e2m1fn_x2),
        w1_scale=weights["SUp"].view(torch.float8_e8m0fnu),
        w2_scale=weights["SDown"].view(torch.float8_e8m0fnu),
    )
    result["w1"].is_shuffled = result["w2"].is_shuffled = True
    return result


def run_experts(latent, weights, ids, route_weights, output=None, precision="a4w4"):
    assert precision in ("a4w4", "a16w4")
    dtype = torch.float4_e2m1fn_x2 if precision == "a4w4" else torch.bfloat16
    return fused_moe(
        latent,
        weights["w1"],
        weights["w2"],
        route_weights,
        ids,
        activation=aiter.ActivationType.Situv2,
        quant_type=aiter.QuantType.per_1x32,
        w1_scale=weights["w1_scale"],
        w2_scale=weights["w2_scale"],
        beta=4.0,
        linear_beta=25.0,
        gate_mode="separated",
        quant_dtype_a=dtype,
        quant_dtype_a2=dtype,
        output=output,
    )


def prepare_metadata(ids, route_weights):
    """Allocate and initialize once, outside graph capture or timed execution."""
    assert ids.shape == (8, 16) and route_weights.shape == (8, 16)
    parts = _adaptive_moe_sort(
        ids, route_weights, 896, 16, 16, 3584, atomic=True, emit_aux=True
    )
    return dict(
        zip(
            (
                "SortIDs",
                "SortWeights",
                "SortExperts",
                "SortCounts",
                "SortOut",
                "SortM",
                "SortReverse",
            ),
            parts,
        )
    )


def run_prepared_experts(latent, weights, ids, route_weights, metadata):
    """The same AITER A4W4 expert kernels, with routes prepared by the front."""
    assert latent.shape == (8, 3584)
    return fused_moe_2stages(
        latent,
        weights["w1"],
        weights["w2"],
        16,
        metadata["SortIDs"],
        metadata["SortWeights"],
        metadata["SortExperts"],
        metadata["SortCounts"],
        metadata["SortOut"],
        True,
        16,
        activation=aiter.ActivationType.Situv2,
        quant_type=aiter.QuantType.per_1x32,
        q_dtype_a=torch.float4_e2m1fn_x2,
        q_dtype_w=torch.float4_e2m1fn_x2,
        w1_scale=weights["w1_scale"],
        w2_scale=weights["w2_scale"],
        beta=4.0,
        linear_beta=25.0,
        gate_mode="separated",
        topk_ids=ids,
        topk_weights=route_weights,
        m_indices=metadata["SortM"],
        reverse_sorted=metadata["SortReverse"],
        routing_num_experts=896,
        _metadata_transform=require_expert_config,
    )
