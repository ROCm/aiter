"""Baseline AITER SiTUv2 experts between two same-stream Gluon launches."""

import torch
import aiter
from aiter.fused_moe import fused_moe


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
