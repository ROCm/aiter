# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# SonicMoE: Pure-Triton grouped GEMM MoE with full autograd support.

import math
import os
from enum import Enum

import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton._triton_kernels.moe.activations import (
    _glu_bwd_kernel,
    _glu_fwd_kernel,
    _pointwise_act_bwd_kernel,
    _pointwise_act_fwd_kernel,
)
from aiter.ops.triton._triton_kernels.moe.moe_routing.bitmatrix_sonicmoe import (
    _sonicmoe_bitmatrix_metadata_compute_stage1,
    _sonicmoe_bitmatrix_metadata_compute_stage2,
)
from aiter.ops.triton._triton_kernels.moe.moe_routing.routing_sonicmoe import (
    _sonicmoe_compute_col_partial_sum_kernel,
    _sonicmoe_general_compute_col_partial_sum_kernel,
    _sonicmoe_general_metadata_compute_stage2,
    _sonicmoe_token_offset_searchsorted_kernel,
)
from aiter.ops.triton._triton_kernels.moe.sonicmoe.backward import (
    db1_kernel,
    db2_and_ds_kernel,
)
from aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm import (
    _grouped_gemm_dw_kernel,
    _grouped_gemm_kernel,
)
from aiter.ops.triton._triton_kernels.moe.sonicmoe.token_gather import (
    token_gather_sum_kernel,
)
from aiter.ops.triton._triton_kernels.moe.sonicmoe.topk_softmax import (
    _softmax_over_topk_bwd_kernel,
    _topk_over_softmax_bwd_kernel,
)
from aiter.ops.triton.utils.sonicmoe_config_utils import (
    get_grouped_gemm_dw_config,
    get_grouped_gemm_fwd_config,
    get_sonicmoe_kernel_config,
    get_token_gather_config,
    split_launch_config,
)

LIBRARY_NAME = "aiter_sonicmoe"


def _local_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is not None and hasattr(tensor, "to_local"):
        return tensor.to_local()
    return tensor


def _use_qwen3_tuned_configs() -> bool:
    return os.environ.get("SONIC_MOE_USE_QWEN3_TUNED_GEMM", "0") == "1"


def grouped_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    A_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    A_is_transposed: bool = False,
    B_is_transposed: bool = False,
    A_scale: torch.Tensor | None = None,
    B_scale: torch.Tensor | None = None,
    block_size: int = 128,
    out_dtype: torch.dtype | None = None,
):
    """Run grouped GEMM, optionally with 1x128 activation and 128x128 weight scales."""
    if (A_scale is None) != (B_scale is None):
        raise ValueError("A_scale and B_scale must be provided together")
    if A_scale is not None and block_size != 128:
        raise ValueError("Sonic blockwise FP8 requires block_size=128")
    if A_is_transposed:
        if B_is_transposed:
            raise ValueError("a grouped wgrad does not support a transposed B")
        if bias is not None:
            raise ValueError("bias is invalid for a grouped wgrad")
        if scatter_idx is not None:
            raise ValueError("scatter_idx is invalid for a grouped wgrad")

    local_out = _local_tensor(out)
    local_b = _local_tensor(B)
    triton_b = local_b.transpose(1, 2) if B_is_transposed else local_b
    local_b_scale = _local_tensor(B_scale)
    triton_b_scale = (
        local_b_scale.transpose(1, 2)
        if B_is_transposed and local_b_scale is not None
        else local_b_scale
    )
    result = _grouped_gemm_triton(
        _local_tensor(A),
        triton_b,
        _local_tensor(cu_seqlens),
        local_out,
        _local_tensor(bias),
        _local_tensor(A_idx),
        _local_tensor(scatter_idx),
        A_is_transposed,
        _local_tensor(A_scale),
        triton_b_scale,
        block_size,
        out_dtype,
    )
    return out if out is not None else result


def _grouped_gemm_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    A_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    A_is_transposed: bool = False,
    A_scale: torch.Tensor | None = None,
    B_scale: torch.Tensor | None = None,
    block_size: int = 128,
    out_dtype: torch.dtype | None = None,
):
    if A_is_transposed and B.dim() == 2:
        return _grouped_gemm_dw(
            A, B, cu_seqlens, out, A_idx, A_scale, B_scale, block_size, out_dtype
        )

    E = B.shape[0]
    K_dim = B.shape[1]
    N = B.shape[2]

    TK = A.shape[0] if A_idx is None else A_idx.numel()

    if out is None:
        out = torch.empty(
            TK,
            N,
            dtype=out_dtype if out_dtype is not None else A.dtype,
            device=A.device,
        )

    blockwise_fp8 = A_scale is not None
    if blockwise_fp8:
        if B_scale is None:
            raise ValueError("B_scale is required when A_scale is provided")
        expected_a_scale = (A.shape[0], triton.cdiv(K_dim, block_size))
        expected_b_scale = (
            E,
            triton.cdiv(K_dim, block_size),
            triton.cdiv(N, block_size),
        )
        if tuple(A_scale.shape) != expected_a_scale:
            raise ValueError(
                f"A_scale must have shape {expected_a_scale}, got {tuple(A_scale.shape)}"
            )
        if tuple(B_scale.shape) != expected_b_scale:
            raise ValueError(
                f"B_scale must have shape {expected_b_scale}, got {tuple(B_scale.shape)}"
            )

    def grid(META):
        max_m_blocks = triton.cdiv(TK, META["BLOCK_M"]) + E - 1
        return (max_m_blocks * triton.cdiv(N, META["BLOCK_N"]),)

    launch_args = (
        A,
        B,
        A_scale if A_scale is not None else A,
        B_scale if B_scale is not None else B,
        out,
        cu_seqlens,
        bias if bias is not None else A,
        A_idx if A_idx is not None else cu_seqlens,
        scatter_idx if scatter_idx is not None else cu_seqlens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        A_scale.stride(0) if A_scale is not None else 0,
        A_scale.stride(1) if A_scale is not None else 0,
        B_scale.stride(0) if B_scale is not None else 0,
        B_scale.stride(1) if B_scale is not None else 0,
        B_scale.stride(2) if B_scale is not None else 0,
        out.stride(0),
        out.stride(1),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
    )
    launch_meta = {
        "N": N,
        "K": K_dim,
        "E": E,
        "SCALE_BLOCK_SIZE": block_size,
        "BLOCKWISE_FP8": blockwise_fp8,
        "HAS_BIAS": (bias is not None),
        "HAS_GATHER_IDX": (A_idx is not None),
        "HAS_SCATTER_IDX": (scatter_idx is not None),
    }
    fwd_cfg = get_grouped_gemm_fwd_config(
        N, K_dim, E, A_idx is not None, _use_qwen3_tuned_configs()
    )
    constexprs, launch = split_launch_config(fwd_cfg)
    _grouped_gemm_kernel[grid](*launch_args, **launch_meta, **constexprs, **launch)
    return out


def _grouped_gemm_dw(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None,
    A_idx: torch.Tensor | None,
    A_scale: torch.Tensor | None = None,
    B_scale: torch.Tensor | None = None,
    block_size: int = 128,
    out_dtype: torch.dtype | None = None,
):
    K_dim = A.shape[1]
    N = B.shape[1]
    E = cu_seqlens.shape[0] - 1

    if out is None:
        out = torch.empty(
            E,
            K_dim,
            N,
            dtype=out_dtype if out_dtype is not None else A.dtype,
            device=A.device,
        )

    blockwise_fp8 = A_scale is not None
    if blockwise_fp8:
        if B_scale is None:
            raise ValueError("B_scale is required when A_scale is provided")
        expert_rows = cu_seqlens[1:] - cu_seqlens[:-1]
        expected_scale_rows = (
            torch.div(
                expert_rows + block_size - 1,
                block_size,
                rounding_mode="floor",
            )
            .sum()
            .item()
        )
        if (
            A_scale.dim() != 2
            or A_scale.shape[0] != expected_scale_rows
            or A_scale.shape[1] != K_dim
        ):
            raise ValueError(
                f"A_scale must have shape [{expected_scale_rows}, {K_dim}], "
                f"got {tuple(A_scale.shape)}"
            )
        if (
            B_scale.dim() != 2
            or B_scale.shape[0] != expected_scale_rows
            or B_scale.shape[1] != N
        ):
            raise ValueError(
                f"B_scale must have shape [{expected_scale_rows}, {N}], "
                f"got {tuple(B_scale.shape)}"
            )
        if A_idx is not None:
            raise ValueError("blockwise FP8 grouped wgrad does not support A_idx")

    def grid(META):
        num_k_blocks = triton.cdiv(K_dim, META["BLOCK_K"])
        num_n_blocks = triton.cdiv(N, META["BLOCK_N"])
        return (E * num_k_blocks * num_n_blocks,)

    launch_args = (
        A,
        B,
        A_scale if A_scale is not None else A,
        B_scale if B_scale is not None else B,
        out,
        cu_seqlens,
        A_idx if A_idx is not None else cu_seqlens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        A_scale.stride(0) if A_scale is not None else 0,
        A_scale.stride(1) if A_scale is not None else 0,
        B_scale.stride(0) if B_scale is not None else 0,
        B_scale.stride(1) if B_scale is not None else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    launch_meta = {
        "N": N,
        "K": K_dim,
        "E": E,
        "SCALE_BLOCK_SIZE": block_size,
        "BLOCKWISE_FP8": blockwise_fp8,
        "HAS_GATHER_IDX": A_idx is not None,
    }
    dw_cfg = get_grouped_gemm_dw_config(
        N, K_dim, E, A_idx is not None, _use_qwen3_tuned_configs()
    )
    constexprs, launch = split_launch_config(dw_cfg)
    _grouped_gemm_dw_kernel[grid](*launch_args, **launch_meta, **constexprs, **launch)
    return out


def token_gather_and_sum_varlen_K_triton(
    x: torch.Tensor,  # (Mtotal, H)
    w: torch.Tensor | None,  # (Mtotal,)
    out: torch.Tensor,  # (T, H)
    M_perm: torch.Tensor,  # (Mtotal,) int32
    M_offset: torch.Tensor,  # (T+1,)   int32, variable K per token
    T: int,
    MAX_K: int,  # maximum K across all tokens
    H: int,
    is_varlen_K: bool,
):
    """Gather and reduce a variable number of weighted rows per token."""
    common = (x, w, M_perm, M_offset, out)
    kwargs = {
        "T": T,
        "H": H,
        "MAX_K": MAX_K,
        "stride_xM": x.stride(0),
        "stride_xH": x.stride(1),
        "stride_outT": out.stride(0),
        "stride_outH": out.stride(1),
        "w_is_None": (w is None),
        "is_varlen_K": is_varlen_K,
    }
    constexprs, launch = split_launch_config(get_token_gather_config(H))
    token_gather_sum_kernel[(T,)](*common, **kwargs, **constexprs, **launch)


_GLU_ACT_MAP = {"swiglu": 0, "geglu": 1, "reglu": 2}
_POINTWISE_ACT_MAP = {"gelu_tanh_approx": 3, "relu": 4, "silu": 5, "relu_sq": 6}


def _launch_config(TK, I):
    config = get_sonicmoe_kernel_config("activation_kernel")
    block_i = min(triton.next_power_of_2(I), config.pop("BLOCK_I_MAX"))
    block_m = config["BLOCK_M"]
    grid = (triton.cdiv(TK, block_m), triton.cdiv(I, block_i))
    constexprs, launch = split_launch_config(config)
    constexprs["BLOCK_I"] = block_i
    return grid, constexprs, launch


def activation_fwd(
    h: torch.Tensor, I: int, activation_type: str, concat_layout: bool = False
) -> torch.Tensor:
    TK = h.shape[0]

    if activation_type in _GLU_ACT_MAP:
        a = torch.empty(TK, I, dtype=h.dtype, device=h.device)
        grid, constexprs, launch = _launch_config(TK, I)
        _glu_fwd_kernel[grid](
            h,
            a,
            TK,
            I,
            h.stride(0),
            h.stride(1),
            a.stride(0),
            a.stride(1),
            CONCAT_LAYOUT=concat_layout,
            ACT_TYPE=_GLU_ACT_MAP[activation_type],
            **constexprs,
            **launch,
        )
        return a
    elif activation_type in _POINTWISE_ACT_MAP:
        a = torch.empty(TK, I, dtype=h.dtype, device=h.device)
        grid, constexprs, launch = _launch_config(TK, I)
        _pointwise_act_fwd_kernel[grid](
            h,
            a,
            TK,
            I,
            h.stride(0),
            h.stride(1),
            a.stride(0),
            a.stride(1),
            ACT_TYPE=_POINTWISE_ACT_MAP[activation_type],
            **constexprs,
            **launch,
        )
        return a
    else:
        raise NotImplementedError(f"activation_type={activation_type}")


def activation_bwd(
    h: torch.Tensor,
    da: torch.Tensor,
    I: int,
    activation_type: str,
    concat_layout: bool = False,
) -> torch.Tensor:
    TK = h.shape[0]

    if activation_type in _GLU_ACT_MAP:
        dh = torch.empty_like(h)
        grid, constexprs, launch = _launch_config(TK, I)
        _glu_bwd_kernel[grid](
            h,
            dh,
            da,
            TK,
            I,
            h.stride(0),
            h.stride(1),
            dh.stride(0),
            dh.stride(1),
            da.stride(0),
            da.stride(1),
            CONCAT_LAYOUT=concat_layout,
            ACT_TYPE=_GLU_ACT_MAP[activation_type],
            **constexprs,
            **launch,
        )
        return dh
    elif activation_type in _POINTWISE_ACT_MAP:
        dh = torch.empty_like(h)
        grid, constexprs, launch = _launch_config(TK, I)
        _pointwise_act_bwd_kernel[grid](
            h,
            dh,
            da,
            TK,
            I,
            h.stride(0),
            h.stride(1),
            dh.stride(0),
            dh.stride(1),
            da.stride(0),
            da.stride(1),
            ACT_TYPE=_POINTWISE_ACT_MAP[activation_type],
            **constexprs,
            **launch,
        )
        return dh
    else:
        raise NotImplementedError(f"activation_type={activation_type}")


@torch.library.custom_op(
    "triton_kernels::TC_topk_router_metadata",
    mutates_args={
        "expert_frequency",
        "expert_frequency_offset",
        "x_gather_idx",
        "s_scatter_idx",
        "s_reverse_scatter_idx",
    },
)
def TC_topk_router_metadata_triton(
    topk_router_indices: torch.Tensor,
    E: int,
    expert_frequency: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
) -> None:
    T, K = topk_router_indices.size()
    TK = T * K
    device = topk_router_indices.device
    E_POW2 = triton.next_power_of_2(E)
    K_POW2 = triton.next_power_of_2(K)
    config = get_sonicmoe_kernel_config("topk_routing")
    TOKENS_PER_BLOCK = config["ENTRIES_PER_TILE"] // K_POW2
    n_tiles = triton.cdiv(T, TOKENS_PER_BLOCK)

    # Transposed storage avoids cross-CTA histogram writes.
    col_partial_sum_trans = torch.empty(E, n_tiles, dtype=torch.int32, device=device)
    _sonicmoe_compute_col_partial_sum_kernel[(n_tiles,)](
        topk_router_indices,
        col_partial_sum_trans,
        T,
        E,
        n_tiles,
        TOKENS_PER_TILE=TOKENS_PER_BLOCK,
        K_POW2=K_POW2,
        K=K,
        E_POW2=E_POW2,
    )

    expert_frequency.copy_(col_partial_sum_trans.sum(dim=1, dtype=torch.int32))
    col_partial_sum = col_partial_sum_trans.T  # [n_tiles, E]

    _sonicmoe_bitmatrix_metadata_compute_stage1[(E + 2,)](
        expert_frequency,
        expert_frequency_offset,
        E,
        col_partial_sum,
        n_tiles,
        TK,
        BLOCK_M=config["PREFIX_BLOCK_M"],
        BLOCK_N=E_POW2,
    )

    _sonicmoe_bitmatrix_metadata_compute_stage2[(n_tiles,)](
        s_scatter_idx,
        s_reverse_scatter_idx,
        x_gather_idx,
        topk_router_indices,
        T,
        col_partial_sum,
        n_tiles,
        expert_frequency_offset[:E],
        K_POW2=K_POW2,
        TOKENS_PER_BLOCK=TOKENS_PER_BLOCK,
        K=K,
    )


@torch.library.custom_op(
    "triton_kernels::general_routing_router_metadata",
    mutates_args={
        "expert_frequency",
        "expert_frequency_offset",
        "x_gather_idx",
        "s_scatter_idx",
        "s_reverse_scatter_idx",
        "num_activated_expert_per_token_offset",
    },
)
def general_routing_router_metadata_triton(
    sorted_selected_T: torch.Tensor,
    selected_E: torch.Tensor,
    T: int,
    E: int,
    expert_frequency: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
    num_activated_expert_per_token_offset: torch.Tensor,
) -> None:
    TK = selected_E.size(0)
    device = selected_E.device
    E_POW2 = triton.next_power_of_2(E)
    config = get_sonicmoe_kernel_config("general_routing")
    BLOCK_SIZE = config["BLOCK_SIZE"]
    n_tiles = triton.cdiv(TK, BLOCK_SIZE)

    col_partial_sum_trans = torch.empty(E, n_tiles, dtype=torch.int32, device=device)
    _sonicmoe_general_compute_col_partial_sum_kernel[(n_tiles,)](
        selected_E,
        col_partial_sum_trans,
        TK,
        E,
        n_tiles,
        BLOCK_SIZE=BLOCK_SIZE,
        E_POW2=E_POW2,
    )

    expert_frequency.copy_(col_partial_sum_trans.sum(dim=1, dtype=torch.int32))
    col_partial_sum = col_partial_sum_trans.T  # [n_tiles, E], strides (1, n_tiles)

    _sonicmoe_bitmatrix_metadata_compute_stage1[(E + 2,)](
        expert_frequency,
        expert_frequency_offset,
        E,
        col_partial_sum,
        n_tiles,
        TK,
        BLOCK_M=config["PREFIX_BLOCK_M"],
        BLOCK_N=E_POW2,
    )

    _sonicmoe_general_metadata_compute_stage2[(n_tiles,)](
        s_scatter_idx,
        s_reverse_scatter_idx,
        x_gather_idx,
        selected_E,
        sorted_selected_T,
        TK,
        col_partial_sum,
        n_tiles,
        expert_frequency_offset[:E],
        BLOCK_SIZE=BLOCK_SIZE,
    )

    N_ITERS = max(1, math.ceil(math.log2(TK + 1)))
    TOKEN_BLOCK = config["TOKEN_SEARCH_BLOCK"]
    n_token_blocks = triton.cdiv(T + 1, TOKEN_BLOCK)
    _sonicmoe_token_offset_searchsorted_kernel[(n_token_blocks,)](
        sorted_selected_T,
        num_activated_expert_per_token_offset,
        T,
        TK,
        BLOCK_SIZE=TOKEN_BLOCK,
        N_ITERS=N_ITERS,
    )


@torch.library.custom_op(f"{LIBRARY_NAME}::_router_forward_rocm", mutates_args={"o"})
def _router_forward(
    y: torch.Tensor,
    o: torch.Tensor,
    topk_scores: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
    num_activated_expert_per_token_offset: torch.Tensor,
    varlen_K_max: int,
    H: int,
    is_varlen_K: bool,
) -> None:
    token_gather_and_sum_varlen_K_triton(
        y,
        topk_scores,
        o,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        o.size(0),
        varlen_K_max,
        H,
        is_varlen_K,
    )


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_softmax_topk_fwd_rocm",
    mutates_args={"topk_router_score", "topk_router_indices"},
)
def _topk_softmax_fwd(
    router_logits: torch.Tensor,
    topk_router_score: torch.Tensor,
    topk_router_indices: torch.Tensor,
    E: int,
    K: int,
    is_softmax_over_topk: bool,
    norm_topk_probs: bool,
) -> None:
    if is_softmax_over_topk:
        topk_results = router_logits.topk(K, dim=-1)
        vals = topk_results.values.softmax(dim=-1, dtype=torch.float32)
        topk_router_score.copy_(vals.to(topk_router_score.dtype))
        topk_router_indices.copy_(topk_results.indices.to(topk_router_indices.dtype))
    else:
        probs = router_logits.softmax(dim=-1, dtype=torch.float32)
        topk_results = probs.topk(K, dim=-1)
        vals = topk_results.values
        if norm_topk_probs:
            vals = vals / vals.sum(dim=-1, keepdim=True)
        topk_router_score.copy_(vals.to(topk_router_score.dtype))
        topk_router_indices.copy_(topk_results.indices.to(topk_router_indices.dtype))


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_topk_softmax_bwd_rocm", mutates_args={"dlogits_full"}
)
def _topk_softmax_bwd(
    router_logits: torch.Tensor,
    dlogits_full: torch.Tensor,
    dlogits: torch.Tensor | None,
    dtopk_score: torch.Tensor,
    topk_router_score: torch.Tensor,
    topk_router_indices: torch.Tensor,
    E: int,
    K: int,
    is_softmax_over_topk: bool = True,
    norm_topk_probs: bool = False,
) -> None:
    T = dtopk_score.shape[0]

    if is_softmax_over_topk:
        _softmax_over_topk_bwd_kernel[T,](
            dlogits,
            dlogits_full,
            topk_router_score,
            dtopk_score,
            topk_router_indices,
            dlogits_full.stride(0),
            dlogits_full.stride(1),
            topk_router_score.stride(0),
            topk_router_score.stride(1),
            dtopk_score.stride(0),
            dtopk_score.stride(1),
            topk_router_indices.stride(0),
            topk_router_indices.stride(1),
            K,
            triton.next_power_of_2(K),
            (dlogits is None),
        )
    else:
        _topk_over_softmax_bwd_kernel[T,](
            router_logits,
            dlogits_full,
            dtopk_score,
            topk_router_indices,
            topk_router_score,
            router_logits.stride(0),
            router_logits.stride(1),
            dlogits_full.stride(0),
            dlogits_full.stride(1),
            dtopk_score.stride(0),
            dtopk_score.stride(1),
            topk_router_indices.stride(0),
            topk_router_indices.stride(1),
            topk_router_score.stride(0),
            topk_router_score.stride(1),
            E,
            K,
            triton.next_power_of_2(E),
            triton.next_power_of_2(K),
            norm_topk_probs,
        )


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_up_projection_backward_act_rocm",
    mutates_args={"dx_expanded", "db1"},
)
def _up_projection_backward_act(
    w1: torch.Tensor,
    dx_expanded: torch.Tensor,
    dh: torch.Tensor,
    db1: torch.Tensor | None,
    expert_frequency_offset: torch.Tensor,
    is_glu_activation: bool,
    concat_layout: bool = False,
    grouped_weight_layout: bool = False,
) -> None:
    if grouped_weight_layout:
        E, _, I_full = w1.size()
        gemm_w1 = w1
    else:
        I_full, _, E = w1.size()
        gemm_w1 = w1.permute(2, 0, 1)
    I = I_full // 2 if is_glu_activation else I_full

    grouped_gemm(
        dh,
        gemm_w1,
        expert_frequency_offset,
        out=dx_expanded,
        B_is_transposed=grouped_weight_layout,
    )

    if db1 is not None:
        db1_cfg = get_sonicmoe_kernel_config("db1_kernel")
        constexprs, launch = split_launch_config(db1_cfg)
        db1_kernel[(E,)](
            dh,
            db1,
            expert_frequency_offset,
            (2 * I if is_glu_activation else I),
            E,
            CONCAT_LAYOUT=concat_layout and is_glu_activation,
            **constexprs,
            **launch,
        )


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_down_projection_backward_act_rocm",
    mutates_args={"dh", "ds", "db2", "a_prime"},
)
def _down_projection_backward_act(
    dout: torch.Tensor,
    h: torch.Tensor,
    w2: torch.Tensor,
    dh: torch.Tensor,
    ds: torch.Tensor,
    b2: torch.Tensor | None,
    db2: torch.Tensor | None,
    a_prime: torch.Tensor,
    topk_scores: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    activation_type: str,
    grouped_weight_layout: bool = False,
    concat_layout: bool = False,
) -> None:
    if grouped_weight_layout:
        E, I, H = w2.size()
        gemm_w2 = w2
    else:
        H, I, E = w2.size()
        gemm_w2 = w2.permute(2, 0, 1)
    TK = x_gather_idx.size(0)
    s = topk_scores[s_scatter_idx]

    # Compute u = dout @ w2.T once. The router gradient reuses this GEMM:
    # dot(dout, a @ w2) == dot(a, dout @ w2.T), while da = score * u.
    dout_gathered = dout[x_gather_idx]
    dh_unscaled = torch.empty(TK, I, dtype=dh.dtype, device=dh.device)
    grouped_gemm(
        dout_gathered,
        gemm_w2,
        expert_frequency_offset,
        out=dh_unscaled,
        B_is_transposed=grouped_weight_layout,
    )

    a_prime_val = activation_fwd(h, I, activation_type, concat_layout)
    a_prime.copy_(a_prime_val)
    ds_scattered = (a_prime_val.float() * dh_unscaled.float()).sum(dim=-1)

    dh_raw = dh_unscaled * s.unsqueeze(-1)
    dh_act = activation_bwd(h, dh_raw, I, activation_type, concat_layout)
    dh.copy_(dh_act)

    if db2 is None:
        ds[s_scatter_idx] = ds_scattered
    else:
        old_ds_partial = torch.empty(
            TK, 1, device=ds_scattered.device, dtype=ds_scattered.dtype
        )
        old_ds_partial[s_scatter_idx, 0] = ds_scattered

        db2_cfg = get_sonicmoe_kernel_config("db2_and_ds_kernel")
        block_h_max = db2_cfg.pop("BLOCK_H_MAX")
        BLOCK_H = min(triton.next_power_of_2(H), block_h_max)
        NUM_H_BLOCKS = triton.cdiv(H, BLOCK_H)
        new_ds_partial = torch.empty(
            TK, NUM_H_BLOCKS, dtype=torch.float32, device=ds.device
        )

        constexprs, launch = split_launch_config(db2_cfg)
        db2_and_ds_kernel[(E, NUM_H_BLOCKS)](
            dout,
            topk_scores,
            new_ds_partial,
            old_ds_partial,
            b2,
            db2,
            x_gather_idx,
            s_scatter_idx,
            expert_frequency_offset,
            H,
            E,
            1,
            BLOCK_H=BLOCK_H,
            **constexprs,
            **launch,
        )

        if NUM_H_BLOCKS == 1:
            ds.copy_(new_ds_partial.view(-1).to(dtype=ds.dtype))
        else:
            ds.copy_(new_ds_partial.sum(dim=-1, dtype=ds.dtype))


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_token_broadcast_backward_rocm", mutates_args={"dx_reduced"}
)
def _token_broadcast_backward(
    dx_reduced: torch.Tensor,
    dx_expanded: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
    num_activated_expert_per_token_offset: torch.Tensor | None,
    varlen_K_max: int,
    H: int,
    is_varlen_K: bool,
) -> None:
    if num_activated_expert_per_token_offset is None:
        assert not is_varlen_K
    token_gather_and_sum_varlen_K_triton(
        dx_expanded,
        None,
        dx_reduced,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        dx_reduced.size(0),
        varlen_K_max,
        H,
        is_varlen_K,
    )


class ActivationType(Enum):
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    REGLU = "reglu"
    RELU_SQ = "relu_sq"
    RELU = "relu"
    GELU = "gelu_tanh_approx"
    SILU = "silu"


def is_glu(activation_type: ActivationType):
    return activation_type in (
        ActivationType.SWIGLU,
        ActivationType.REGLU,
        ActivationType.GEGLU,
    )


SonicMoEActivationType = ActivationType
sonicmoe_is_glu = is_glu

__all__ = [
    "ActivationType",
    "SonicMoEActivationType",
    "is_glu",
    "moe_TC_softmax_topk_layer",
    "moe_general_routing_inputs",
    "moe_pre_routed_inputs",
    "sonicmoe_is_glu",
]


class TC_Softmax_Topk_Router_Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        router_logits: torch.Tensor,
        E: int,
        K: int,
        is_softmax_over_topk: bool,
        norm_topk_probs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = router_logits.size(0)

        topk_router_score = torch.empty(
            T, K, dtype=torch.float32, device=router_logits.device
        )
        topk_router_indices = torch.empty(
            T, K, dtype=torch.int32, device=router_logits.device
        )

        _topk_softmax_fwd(
            router_logits,
            topk_router_score,
            topk_router_indices,
            E,
            K,
            is_softmax_over_topk=is_softmax_over_topk,
            norm_topk_probs=norm_topk_probs,
        )

        ctx.save_for_backward(topk_router_score, topk_router_indices, router_logits)
        ctx.E = E
        ctx.dtype = router_logits.dtype
        ctx.is_softmax_over_topk = is_softmax_over_topk
        ctx.norm_topk_probs = norm_topk_probs

        return topk_router_score, topk_router_indices

    @staticmethod
    def backward(ctx, dtopk_score: torch.Tensor, _: torch.Tensor):
        T, K = dtopk_score.size()
        E = ctx.E
        topk_router_score, topk_router_indices, router_logits = ctx.saved_tensors
        dlogits = torch.zeros(
            T, ctx.E, dtype=ctx.dtype, device=topk_router_score.device
        )

        _topk_softmax_bwd(
            router_logits,
            dlogits,
            None,
            dtopk_score,
            topk_router_score,
            topk_router_indices,
            E,
            K,
            is_softmax_over_topk=ctx.is_softmax_over_topk,
            norm_topk_probs=ctx.norm_topk_probs,
        )

        return dlogits, None, None, None, None


class _UpProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        expert_frequency_offset: torch.Tensor,
        total_expert_freq: int,
        K: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        num_activated_expert_per_token_offset: torch.Tensor,
        is_each_token_has_variable_activated_experts: bool,
        activation_type: ActivationType,
        is_inference_mode_enabled: bool,
        concat_layout: bool = False,
        grouped_weight_layout: bool = False,
        inputs_are_pre_routed: bool = False,
    ) -> torch.Tensor:
        T, H = x.shape
        E = expert_frequency_offset.numel() - 1
        if grouped_weight_layout:
            E_w, H_w, I_full = w1.shape
            if E_w != E or H_w != H:
                raise ValueError(
                    f"Grouped w1 must be [E={E}, H={H}, I], got {tuple(w1.shape)}"
                )
            gemm_w1 = w1
        else:
            I_full, H_w, E_w = w1.shape
            if E_w != E or H_w != H:
                raise ValueError(
                    f"Legacy w1 must be [I, H={H}, E={E}], got {tuple(w1.shape)}"
                )
            gemm_w1 = w1.permute(2, 1, 0)
        is_glu_activation = is_glu(activation_type)
        I = I_full // 2 if is_glu_activation else I_full
        TK = total_expert_freq

        h = torch.empty(TK, I_full, dtype=x.dtype, device=x.device)
        grouped_gemm(
            x,
            gemm_w1,  # (E, H, I_full)
            expert_frequency_offset,
            out=h,
            bias=b1,
            A_idx=None if inputs_are_pre_routed else x_gather_idx,
        )

        a = activation_fwd(h, I, activation_type.value, concat_layout)

        h_save = h if not is_inference_mode_enabled else None

        ctx.T = T
        ctx.TK = TK
        ctx.E = E
        ctx.K = K
        ctx.H = H
        ctx.I = I
        ctx.is_each_token_has_variable_activated_experts = (
            is_each_token_has_variable_activated_experts
        )
        ctx.is_glu_activation = is_glu_activation
        ctx.concat_layout = concat_layout and is_glu_activation
        ctx.grouped_weight_layout = grouped_weight_layout
        ctx.inputs_are_pre_routed = inputs_are_pre_routed

        ctx.save_for_backward(
            x,
            w1,
            b1,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        )

        ctx.mark_non_differentiable(a)
        ctx.set_materialize_grads(False)

        return a, h_save

    @staticmethod
    def backward(ctx, _: None, dh: torch.Tensor):
        T = ctx.T
        TK = ctx.TK
        E = ctx.E
        K = ctx.K
        H = ctx.H
        is_glu_activation = ctx.is_glu_activation
        is_each_token_has_variable_activated_experts = (
            ctx.is_each_token_has_variable_activated_experts
        )
        concat_layout = ctx.concat_layout

        (
            x,
            w1,
            b1,
            expert_frequency_offset,
            x_gather_idx,
            _s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        ) = ctx.saved_tensors

        dx_expanded = torch.empty(TK, H, dtype=dh.dtype, device=dh.device)
        dw1 = torch.empty_like(w1)
        db1 = None if b1 is None else torch.empty_like(b1)

        _up_projection_backward_act(
            w1=_local_tensor(w1),
            dx_expanded=dx_expanded,
            dh=dh,
            db1=_local_tensor(db1),
            expert_frequency_offset=expert_frequency_offset,
            is_glu_activation=is_glu_activation,
            concat_layout=concat_layout,
            grouped_weight_layout=ctx.grouped_weight_layout,
        )

        grouped_gemm(
            x,
            dh,
            expert_frequency_offset,
            out=dw1 if ctx.grouped_weight_layout else dw1.permute(2, 1, 0),
            A_idx=None if ctx.inputs_are_pre_routed else x_gather_idx,
            A_is_transposed=True,
        )

        if ctx.inputs_are_pre_routed:
            dx_reduced = dx_expanded
        else:
            dx_reduced = torch.empty(T, H, dtype=dh.dtype, device=dh.device)
            _token_broadcast_backward(
                dx_reduced=dx_reduced,
                dx_expanded=dx_expanded,
                s_reverse_scatter_idx=s_reverse_scatter_idx,
                num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
                varlen_K_max=(E if is_each_token_has_variable_activated_experts else K),
                H=H,
                is_varlen_K=is_each_token_has_variable_activated_experts,
            )

        return dx_reduced, dw1, db1, *[None] * 13


class _DownProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        h: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        topk_scores: torch.Tensor,
        expert_frequency_offset: torch.Tensor,
        T: int,
        K: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        num_activated_expert_per_token_offset: torch.Tensor,
        is_varlen_K: bool,
        activation_type: ActivationType,
        grouped_weight_layout: bool,
        concat_layout: bool,
    ) -> torch.Tensor:
        TK = a.size(0)
        E = expert_frequency_offset.numel() - 1
        if grouped_weight_layout:
            E_w, I, H = w2.shape
            if E_w != E or I != a.size(1):
                raise ValueError(
                    f"Grouped w2 must be [E={E}, I={a.size(1)}, H], "
                    f"got {tuple(w2.shape)}"
                )
            gemm_w2 = w2
        else:
            H, I, E_w = w2.shape
            if E_w != E or I != a.size(1):
                raise ValueError(
                    f"Legacy w2 must be [H, I={a.size(1)}, E={E}], "
                    f"got {tuple(w2.shape)}"
                )
            gemm_w2 = w2.permute(2, 1, 0)

        y = torch.empty(TK, H, dtype=a.dtype, device=a.device)
        grouped_gemm(a, gemm_w2, expert_frequency_offset, out=y, bias=b2)

        o = torch.empty(T, H, device=a.device, dtype=a.dtype)
        topk_scores_flat = topk_scores.view(-1)

        _router_forward(
            y=y,
            o=o,
            topk_scores=topk_scores_flat,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
            varlen_K_max=(E if is_varlen_K else K),
            H=H,
            is_varlen_K=is_varlen_K,
        )

        ctx.T = T
        ctx.K = K
        ctx.is_varlen_K = is_varlen_K
        ctx.activation_type = activation_type
        ctx.grouped_weight_layout = grouped_weight_layout
        ctx.concat_layout = concat_layout

        ctx.save_for_backward(
            h,
            w2,
            b2,
            topk_scores_flat,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
        )

        return o

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        T = ctx.T
        K = ctx.K
        is_varlen_K = ctx.is_varlen_K
        activation_type = ctx.activation_type

        (
            h,
            w2,
            b2,
            topk_scores,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
        ) = ctx.saved_tensors

        dw2 = torch.empty_like(w2)
        db2 = None if b2 is None else torch.empty_like(b2)
        dh = torch.empty_like(h)

        I = w2.size(1)
        TK = x_gather_idx.size(0)

        a_prime = torch.empty(TK, I, dtype=h.dtype, device=h.device)
        ds = torch.empty_like(topk_scores)

        _down_projection_backward_act(
            dout=dout,
            h=h,
            w2=_local_tensor(w2),
            dh=dh,
            ds=ds,
            b2=_local_tensor(b2),
            db2=_local_tensor(db2),
            a_prime=a_prime,
            topk_scores=topk_scores,
            expert_frequency_offset=expert_frequency_offset,
            x_gather_idx=x_gather_idx,
            s_scatter_idx=s_scatter_idx,
            activation_type=activation_type.value,
            grouped_weight_layout=ctx.grouped_weight_layout,
            concat_layout=ctx.concat_layout,
        )

        s = topk_scores[s_scatter_idx]
        dout_gathered = dout[x_gather_idx]
        dy = dout_gathered * s.unsqueeze(-1)

        grouped_gemm(
            a_prime,
            dy,
            expert_frequency_offset,
            out=dw2 if ctx.grouped_weight_layout else dw2.permute(2, 1, 0),
            A_is_transposed=True,
        )

        if not is_varlen_K:
            ds = ds.view(T, K)

        return None, dh, dw2, db2, ds, *[None] * 11


def moe_TC_softmax_topk_layer(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    K: int,
    _stream_id: int,
    activation_type: ActivationType | str = ActivationType.SWIGLU,
    is_inference_mode_enabled: bool = False,
    is_softmax_over_topk: bool = True,
    norm_topk_probs: bool = False,
    concat_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run SonicMoE from dense router logits and select ``K`` experts per token.

    Weights use the legacy layouts ``w1=[2I,H,E]`` and ``w2=[H,I,E]``.
    Biases must either both be present or both be omitted. Returns the output,
    dense router logits, and per-expert assignment counts.
    """
    if (b1 is None) != (b2 is None):
        raise ValueError("b1 and b2 must either both be provided or both be omitted")
    E = router_w.size(0)
    router_logits = F.linear(x, router_w)
    topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(
        router_logits, E, K, is_softmax_over_topk, norm_topk_probs
    )

    T, K = topk_indices.size()
    TK = T * K
    device = topk_indices.device

    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    TC_topk_router_metadata_triton(
        topk_indices,
        E,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
    )

    if type(activation_type) == str:
        activation_type = ActivationType(activation_type)

    a, h = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        TK,
        K,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,
        activation_type,
        is_inference_mode_enabled,
        concat_layout,
        False,
        False,
    )

    o = _DownProjection.apply(
        a,
        h,
        w2,
        b2,
        topk_scores,
        expert_frequency_offset,
        T,
        K,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,
        activation_type,
        False,
        concat_layout,
    )

    return o, router_logits, expert_frequency


def moe_general_routing_inputs(
    x: torch.Tensor,
    router_scores: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    E: int,
    _stream_id: int,
    activation_type: ActivationType,
    is_inference_mode_enabled: bool = False,
    concat_layout: bool = False,
    grouped_weight_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run SonicMoE from explicit token/expert assignments.

    ``router_scores``, ``token_indices``, and ``expert_indices`` must be
    one-dimensional tensors with the same length. ``token_indices`` must be
    sorted in nondecreasing order and contain values in ``[0, x.size(0))``;
    ``expert_indices`` must contain values in ``[0, E)``. Set
    ``grouped_weight_layout`` for ``w1=[E,H,2I]`` and ``w2=[E,I,H]``.
    """
    if (b1 is None) != (b2 is None):
        raise ValueError("b1 and b2 must either both be provided or both be omitted")

    T = x.size(0)
    TK = router_scores.size(0)
    device = router_scores.device
    if any(
        tensor.dim() != 1 for tensor in (router_scores, token_indices, expert_indices)
    ):
        raise ValueError("routing scores and indices must be one-dimensional")
    if token_indices.numel() != TK or expert_indices.numel() != TK:
        raise ValueError("routing scores and indices must have the same length")
    if TK and (
        torch.any(token_indices[1:] < token_indices[:-1])
        or torch.any(token_indices < 0)
        or torch.any(token_indices >= T)
    ):
        raise ValueError("token_indices must be sorted and in [0, x.size(0))")
    if TK and (torch.any(expert_indices < 0) or torch.any(expert_indices >= E)):
        raise ValueError("expert_indices must be in [0, E)")

    if router_scores.dtype != torch.float32:
        router_scores = router_scores.float()

    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    num_activated_expert_per_token_offset = torch.empty(
        T + 1, dtype=torch.int32, device=device
    )

    general_routing_router_metadata_triton(
        token_indices,
        expert_indices,
        T,
        E,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    )

    a, h = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        TK,
        None,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        True,
        activation_type,
        is_inference_mode_enabled,
        concat_layout,
        grouped_weight_layout,
        False,
    )

    o = _DownProjection.apply(
        a,
        h,
        w2,
        b2,
        router_scores,
        expert_frequency_offset,
        T,
        None,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        True,
        activation_type,
        grouped_weight_layout,
        concat_layout,
    )

    return o, expert_frequency


def moe_pre_routed_inputs(
    x: torch.Tensor,
    router_scores: torch.Tensor,
    expert_frequency: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    _stream_id: int,
    activation_type: ActivationType,
    is_inference_mode_enabled: bool = False,
    concat_layout: bool = False,
    grouped_weight_layout: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run SonicMoE on expert-major tokens from an all-to-all dispatcher.

    Tokens must already be contiguous by expert and ``expert_frequency`` must
    contain one count per expert whose sum equals ``x.size(0)``. One router
    score is required per token. Grouped weights use ``w1=[E,H,2I]`` and
    ``w2=[E,I,H]``; set ``grouped_weight_layout=False`` for legacy layouts.
    """
    T = x.size(0)
    if router_scores.numel() != T:
        raise ValueError(
            f"Expected one router score per pre-routed token ({T}), "
            f"got {router_scores.numel()}"
        )
    if router_scores.dtype != torch.float32:
        router_scores = router_scores.float()
    if expert_frequency.dim() != 1:
        raise ValueError("expert_frequency must be one-dimensional")
    if expert_frequency.sum().item() != T:
        raise ValueError("expert_frequency must sum to the number of input tokens")

    expert_frequency = expert_frequency.to(device=x.device, dtype=torch.int32)
    expert_frequency_offset = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=x.device),
            expert_frequency.cumsum(dim=0, dtype=torch.int32),
        )
    )
    identity = torch.arange(T, dtype=torch.int32, device=x.device)

    a, h = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        T,
        1,
        identity,
        identity,
        identity,
        None,
        False,
        activation_type,
        is_inference_mode_enabled,
        concat_layout,
        grouped_weight_layout,
        True,
    )

    o = _DownProjection.apply(
        a,
        h,
        w2,
        b2,
        router_scores.reshape(T, 1),
        expert_frequency_offset,
        T,
        1,
        identity,
        identity,
        identity,
        None,
        False,
        activation_type,
        grouped_weight_layout,
        concat_layout,
    )
    return o, expert_frequency
