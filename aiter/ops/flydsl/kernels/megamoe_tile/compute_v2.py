# SPDX-License-Identifier: MIT
"""Unfused A4W4 EP baseline used to validate the future H1/H2 kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .activation import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    apply_gate_up,
    normalize_activation,
    validate_activation_parameters,
)
from .markers import roctx_range


@dataclass
class PreparedA4W4Weights:
    w1: torch.Tensor
    w1_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    local_experts: int
    model_dim: int
    inter_dim: int
    w1_dequant: torch.Tensor | None
    w2_dequant: torch.Tensor | None


def _dequant_per_1x32(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

    values = mxfp4_to_f32(q)
    logical_k = q.shape[-1] * 2
    scale = scale.view(*q.shape[:-1], logical_k // 32)
    scales = e8m0_to_f32(scale).repeat_interleave(32, dim=-1)
    return values.float() * scales.float()


@torch.no_grad()
def prepare_local_a4w4_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    keep_dequant_reference: bool = True,
) -> PreparedA4W4Weights:
    """Quantize local experts to the current AITER A4W4 native layouts.

    ``keep_dequant_reference`` preserves the historical behavior used by the
    dense correctness reference.  Operator benchmarks which only consume the
    native packed weights can disable it to avoid retaining two very large
    FP32 tensors (about 14.8 GiB/rank for E56/H7168/I3072).
    """

    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.ops.shuffle import (
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility.fp4_utils import e8m0_shuffle

    if w1_bf16.ndim != 3 or w2_bf16.ndim != 3:
        raise ValueError("w1/w2 must be [E,2I,H] and [E,H,I]")
    e, two_i, h = w1_bf16.shape
    e2, h2, inter = w2_bf16.shape
    if e != e2 or h != h2 or two_i != 2 * inter:
        raise ValueError("inconsistent W1/W2 expert dimensions")

    w1q, w1s = per_1x32_f4_quant(w1_bf16, shuffle=False)
    w2q, w2s = per_1x32_f4_quant(w2_bf16, shuffle=False)
    w1_ref = _dequant_per_1x32(w1q, w1s) if keep_dequant_reference else None
    w2_ref = _dequant_per_1x32(w2q, w2s) if keep_dequant_reference else None
    # A4W4 separated GGUU layout.  Do not use gate_up interleave here.
    w1q = shuffle_weight(w1q, layout=(16, 16))
    w1s = e8m0_shuffle(w1s)
    w2q = shuffle_weight_a16w4(w2q, 16, False)
    w2s = shuffle_scale_a16w4(w2s, e, False)
    return PreparedA4W4Weights(
        w1q, w1s, w2q, w2s, e, h, inter, w1_ref, w2_ref
    )


@torch.no_grad()
def a4w4_dense_reference(
    x_bf16: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: PreparedA4W4Weights,
    *,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    situ_beta: float = DEFAULT_SITUV2_BETA,
    situ_linear_beta: float = DEFAULT_SITUV2_LINEAR_BETA,
) -> torch.Tensor:
    """Reference with A4 input, A4 weights and Stage1 A4 requantization."""

    from aiter.ops.quant import per_1x32_f4_quant

    activation = normalize_activation(activation)
    if weights.w1_dequant is None or weights.w2_dequant is None:
        raise ValueError(
            "dense reference requires weights prepared with "
            "keep_dequant_reference=True"
        )
    validate_activation_parameters(
        activation=activation,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    xq, xs = per_1x32_f4_quant(x_bf16, shuffle=False)
    x = _dequant_per_1x32(xq, xs)
    m, topk = topk_ids.shape
    out = torch.zeros((m, weights.model_dim), dtype=torch.float32, device=x.device)
    for token in range(m):
        for slot in range(topk):
            expert = int(topk_ids[token, slot])
            gu = torch.mv(weights.w1_dequant[expert], x[token])
            gate, up = gu.chunk(2, dim=0)
            mid = apply_gate_up(
                gate,
                up,
                activation,
                swiglu_limit=swiglu_limit,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )
            midq, mids = per_1x32_f4_quant(mid.unsqueeze(0), shuffle=False)
            mid_deq = _dequant_per_1x32(midq, mids)[0]
            route = torch.mv(weights.w2_dequant[expert], mid_deq)
            out[token] += float(topk_weights[token, slot]) * route
    return out


@torch.no_grad()
def run_local_ep_a4w4(
    x_bf16: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: PreparedA4W4Weights,
    *,
    global_experts: int,
    local_expert_mask: torch.Tensor,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    situ_beta: float = DEFAULT_SITUV2_BETA,
    situ_linear_beta: float = DEFAULT_SITUV2_LINEAR_BETA,
    block_m: int = 32,
    stage1_tile_n: int = 64,
    stage1_tile_k: int = 256,
    stage2_tile_n: int = 128,
    stage2_tile_k: int = 128,
) -> torch.Tensor:
    """Run the existing unfused local EP pipeline for one activation variant.

    The result is this rank's weighted partial.  The caller combines partials
    across EP ranks.  It is the accuracy baseline for the new fused H1/H2 path.
    """

    from aiter.fused_moe import moe_sorting
    from aiter.ops.flydsl import flydsl_moe_stage1
    from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.utility.fp4_utils import moe_mxfp4_sort

    activation = normalize_activation(activation)
    validate_activation_parameters(
        activation=activation,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    m, h = x_bf16.shape
    topk = int(topk_ids.shape[1])
    if h != weights.model_dim:
        raise ValueError("activation model_dim does not match weights")
    if local_expert_mask.numel() != global_experts:
        raise ValueError("local_expert_mask must cover global experts")
    if int(local_expert_mask.sum()) != weights.local_experts:
        raise ValueError("local mask count does not match local weights")

    with roctx_range("MEGAMOE_TILE/stage1_sort_quant"):
        sorted_ids, sorted_weights, sorted_eids, nvalid, _ = moe_sorting(
            topk_ids,
            topk_weights,
            global_experts,
            h,
            torch.bfloat16,
            block_m,
            expert_mask=local_expert_mask,
            accumulate=False,
        )
        a1q, a1s = per_1x32_f4_quant(x_bf16, shuffle=False)
        a1ss = moe_mxfp4_sort(
            a1s.view(m, 1, h // 32), sorted_ids, nvalid, m, block_m
        )

    with roctx_range(f"MEGAMOE_TILE/stage1_gmm1_{activation}_quant"):
        inter_q, inter_s = flydsl_moe_stage1(
            a=a1q,
            w1=weights.w1,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_eids,
            num_valid_ids=nvalid,
            topk=topk,
            tile_m=block_m,
            tile_n=stage1_tile_n,
            tile_k=stage1_tile_k,
            a_dtype="fp4",
            b_dtype="fp4",
            out_dtype="fp4",
            act=activation,
            situ_beta=float(situ_beta),
            situ_linear_beta=float(situ_linear_beta),
            swiglu_limit=(
                None if swiglu_limit is None else float(swiglu_limit)
            ),
            w1_scale=weights.w1_scale,
            a1_scale=a1ss,
            sorted_weights=None,
            use_async_copy=True,
            waves_per_eu=3,
            b_nt=0,
            gate_mode="separated",
            k_wave=1,
            v2_output_layout=True,
        )

    out = torch.zeros((m, h), dtype=torch.bfloat16, device=x_bf16.device)
    with roctx_range("MEGAMOE_TILE/stage2_gmm2_local_combine"):
        mxfp4_moe_gemm2(
            inter_sorted_quant=inter_q.view(torch.uint8),
            inter_sorted_shuffled_scale=inter_s.view(torch.uint8),
            w2_u8=weights.w2.view(torch.uint8),
            w2_scale_u8=weights.w2_scale.view(torch.uint8),
            sorted_expert_ids=sorted_eids,
            cumsum_tensor=nvalid,
            sorted_token_ids=sorted_ids,
            sorted_weights=sorted_weights,
            out=out,
            M_logical=m,
            max_sorted=inter_q.shape[0],
            NE=weights.local_experts,
            D_HIDDEN=h,
            D_INTER=weights.inter_dim,
            topk=topk,
            BM=block_m,
            BN=stage2_tile_n,
            BK=stage2_tile_k,
            use_nt=False,
            a_dtype="fp4",
            epilog="atomic",
            SBM=block_m,
            persist=False,
            out_dtype="bf16",
            HIDDEN_MAX=h,
            INTER_MAX=weights.inter_dim,
        )
    return out


def run_local_ep_a4w4_silu(*args, **kwargs) -> torch.Tensor:
    """Backward-compatible spelling for the original SiLU-only entry point."""

    requested = normalize_activation(kwargs.pop("activation", "silu"))
    if requested != "silu":
        raise ValueError(
            "run_local_ep_a4w4_silu only accepts activation='silu'; "
            "use run_local_ep_a4w4 for other activations"
        )
    return run_local_ep_a4w4(*args, activation="silu", **kwargs)
