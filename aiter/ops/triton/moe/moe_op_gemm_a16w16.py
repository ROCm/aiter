# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE GEMM with 16-bit activations and weights.

This module owns the existing public ``moe_gemm_a16w16`` wrapper and extends
it with a config-driven Triton backend surface. The validated public large-M
enablement still lives in the gfx1201-specific fused-MoE follow-up path.
"""

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_op_gemm_a16w16 import (
    _moe_gemm_a16w16 as _moe_gemm_a16w16_kernel_triton,
)
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
from aiter.ops.triton.moe.moe_routing.topk_align import (
    align_topk_ids,
    get_topk_align_config,
)
from aiter.ops.triton.moe.reduce import reduce_grouped
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.moe_config_utils import get_moe_dispatch

_LOGGER = AiterTritonLogger()

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)


def is_gluon_available():
    """True when the gluon backend supports the current GPU architecture.

    The match is exact. A substring match would also accept gfx1250-strict,
    which this op does not tune, test or benchmark.
    """
    try:
        return get_arch() in _GLUON_SUPPORTED_ARCHS
    except Exception:  # noqa: BLE001
        return False


def is_triton_a16w16_available():
    """True when the current GPU arch ships Triton A16W16 MoE configs."""
    try:
        return bool(get_moe_dispatch("MOE-GEMM-A16W16", get_arch(), "triton"))
    except Exception:  # noqa: BLE001
        return False


def _load_gluon_backend():
    from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a16w16 import (
        _moe_gemm_a16w16 as _moe_gemm_a16w16_gluon,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a16w16 import (
        get_moe_a16w16_layouts,
    )

    return _moe_gemm_a16w16_gluon, get_moe_a16w16_layouts


def can_overflow_int32(tensor: torch.Tensor):
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32


def should_upcast_indices(*args):
    return any(tensor is not None and can_overflow_int32(tensor) for tensor in args)


def m2bucket(m):
    if m <= 8:
        return "tiny"
    if m <= 32:
        return "small"
    if m <= 128:
        return "medium"
    if m <= 256:
        return "medium2"
    if m <= 512:
        return "large"
    return "xlarge"


def allocate_output(
    M,
    N,
    out_dtype,
    reduction_n_matmul,
    reduction_n_reduction,
    routing_data,
    scatter_indx,
    split_k,
    device,
):
    if routing_data.n_expts_act == 1 or scatter_indx is None:
        y_rows = M
    else:
        # compressed number of rows
        y_rows = scatter_indx.shape[0] // routing_data.n_expts_act
    matmul_shape = (split_k, M, N // reduction_n_matmul)
    final_shape = (y_rows, N // reduction_n_matmul // reduction_n_reduction)
    matmul_output = torch.empty(matmul_shape, device=device, dtype=out_dtype)
    if scatter_indx is not None or split_k > 1:
        final_output = torch.empty(final_shape, device=device, dtype=out_dtype)
    else:
        final_output = None
    return matmul_output, final_output


def get_kernel_config_gluon(m, n, k, routing_data, apply_swiglu=False):
    """Tuned gfx1250 Gluon config for one call."""
    block_m = routing_data.block_m

    tuned = get_moe_dispatch("A16W16", get_arch(), "gluon")
    suffix = "_swiglu" if apply_swiglu else ""
    for key in (
        f"bm{block_m}{suffix}_n{n}_k{k}_{m2bucket(m)}",
        f"bm{block_m}{suffix}_any",
        f"bm{block_m}_any",
    ):
        if key in tuned:
            break
    cfg = tuned[key]

    block_k = cfg["block_k"]
    # The steady-state loop runs num_k_iter - (num_buffers - 1) times, so the
    # pipeline cannot be deeper than the number of K tiles.
    num_buffers = min(cfg["num_buffers"], triton.cdiv(k, block_k))

    config = dict(cfg)
    # block_m comes from routing, and the pipeline cannot be deeper than K.
    config["block_m"] = block_m
    config["num_buffers"] = num_buffers
    return config


def _get_kernel_config_triton(block_m: int, n: int, k: int):
    key = f"bm{block_m}_n{n}_k{k}"
    dispatch = get_moe_dispatch("MOE-GEMM-A16W16", get_arch(), "triton")
    if key not in dispatch:
        raise NotImplementedError(f"MOE-GEMM-A16W16 config is not available for {key}")
    config = dict(dispatch[key])
    w_cache_modifier = (
        config["w_cache_modifier"]
        if block_m <= config["w_cache_modifier_block_m_leq"]
        else None
    )
    return {
        "block_m": block_m,
        "block_n": config["block_n"],
        "block_k": config["block_k"],
        "group_m": config["group_m"],
        "xcd_swizzle": config["xcd_swizzle"],
        "w_cache_modifier": w_cache_modifier,
        "split_k": 1,
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }


def build_routing_from_topk(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    block_m: int = 32,
):
    return align_topk_ids(
        topk_ids, topk_weights, num_experts=num_experts, block_m=block_m
    )


def _moe_gemm_a16w16_triton(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    routing_data: RoutingData,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    out_dtype=torch.bfloat16,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    swiglu_add_residual=True,
    bias=None,
):
    assert is_triton_a16w16_available(), (
        "Triton moe_gemm_a16w16 requires a MOE-GEMM-A16W16 config for the "
        f"running arch, got {get_arch()!r}"
    )
    assert bias is None, "Triton moe_gemm_a16w16 does not support bias"
    if apply_swiglu:
        assert alpha == 1.0, "Triton a16w16 only supports alpha=1.0"
        assert limit == 1.0, "Triton a16w16 only supports limit=1.0"
        assert (
            swiglu_add_residual
        ), "Triton a16w16 requires swiglu_add_residual=True"

    assert x.dtype in (torch.bfloat16, torch.float16)
    assert w.dtype in (torch.bfloat16, torch.float16)
    assert x.is_contiguous()
    assert w.is_contiguous()

    num_rows = x.shape[-2] if gather_indx is None else gather_indx.shape[0]
    num_experts, n, k = w.shape
    assert num_experts == routing_data.n_expts_tot
    assert x.shape[-1] == k
    cfg = _get_kernel_config_triton(routing_data.block_m, n, k)

    reduction_n_matmul = 2 if apply_swiglu else 1
    reduction_n_reduction = 1
    y, y_final = allocate_output(
        num_rows,
        n,
        out_dtype,
        reduction_n_matmul,
        reduction_n_reduction,
        routing_data,
        scatter_indx,
        cfg["split_k"],
        x.device,
    )

    expt_data = routing_data.expt_data
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map

    out_n = n // reduction_n_matmul
    grid_m = routing_data.n_blocks(num_rows, cfg["block_m"])
    grid_n = triton.cdiv(out_n, cfg["block_n"])
    grid = grid_m * grid_n

    _moe_gemm_a16w16_kernel_triton[(grid,)](
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        x,
        x.stride(0),
        x.stride(1),
        w,
        w.stride(0),
        w.stride(1),
        w.stride(2),
        gammas,
        n,
        k,
        gather_indx,
        expt_hist,
        expt_token_offs_raw,
        expt_hist_sum,
        expt_block_pid_map,
        grid_m,
        grid_n,
        apply_swiglu,
        routing_data.n_expts_act,
        cfg["block_m"],
        cfg["block_n"],
        cfg["block_k"],
        cfg["group_m"],
        cfg["xcd_swizzle"],
        k % cfg["block_k"] == 0,
        k % cfg["block_k"],
        cfg["w_cache_modifier"],
        False,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )

    group_indx = (
        None
        if scatter_indx is None
        else scatter_indx.view(-1, routing_data.n_expts_act)
    )
    return reduce_grouped(
        y,
        group_indx,
        y_final,
        False,
        alpha,
        limit,
        reduction_n_reduction,
        out_dtype=out_dtype,
        swiglu_add_residual=swiglu_add_residual,
    )


def moe_gemm_a16w16(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales=None,  # API compatibility with lower-precision MoE ops; must be None
    w_scales=None,  # API compatibility with lower-precision MoE ops; must be None
    x_static_scale=None,  # API compatibility; must be None
    quant_static_scale=None,  # API compatibility; must be None
    bias=None,
    routing_data: RoutingData | None = None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    out_dtype=torch.bfloat16,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    swiglu_add_residual=True,
    unpadded_N=None,
    unpadded_K=None,
    backend: str | None = None,
):
    """
    Computes MoE GEMM with 16-bit activations and 16-bit weights.

    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])
    """
    arch = get_arch()
    backend_name = None if backend is None else backend.lower()
    use_gluon = backend_name in (None, "gluon") and is_gluon_available()
    use_triton_a16w16 = backend_name in (None, "triton") and is_triton_a16w16_available()

    assert use_gluon or use_triton_a16w16, (
        "moe_gemm_a16w16 is implemented only for gfx1250/gluon or "
        f"config-backed Triton, got arch={arch!r}, backend={backend!r}"
    )

    assert x_scales is None, "x_scales must be none"
    assert w_scales is None, "w_scales must be none"
    assert x_static_scale is None, "x_static_scale must be none"
    assert quant_static_scale is None, "quant_static_scale must be none"
    assert x.dtype.itemsize == 2, f"x must be 16-bit, got {x.dtype}"
    assert w.dtype.itemsize == 2, f"w must be 16-bit, got {w.dtype}"
    assert out_dtype.itemsize == 2, f"out_dtype must be 16-bit, got {out_dtype}"
    assert w.stride(-1) == 1, "`w` must be contiguous along its last dim"

    _LOGGER.info(
        "MOE_GEMM_A16W16: x=%s w=%s bias=%s apply_swiglu=%s",
        x.shape,
        w.shape,
        None if bias is None else bias.shape,
        apply_swiglu,
    )

    if use_triton_a16w16:
        assert unpadded_N is None, "Triton moe_gemm_a16w16 has no unpadded_N"
        assert unpadded_K is None, "Triton moe_gemm_a16w16 has no unpadded_K"
        return _moe_gemm_a16w16_triton(
            x,
            w,
            routing_data=routing_data,
            gather_indx=gather_indx,
            scatter_indx=scatter_indx,
            gammas=gammas,
            out_dtype=out_dtype,
            apply_swiglu=apply_swiglu,
            alpha=alpha,
            limit=limit,
            swiglu_add_residual=swiglu_add_residual,
            bias=bias,
        )

    _moe_gemm_a16w16_gluon, get_moe_a16w16_layouts = _load_gluon_backend()

    # determine shapes
    M = x.shape[-2] if gather_indx is None else gather_indx.shape[0]
    K, N = x.shape[-1], w.shape[-1]
    block_m = routing_data.block_m
    if unpadded_N and block_m == 16:
        N = unpadded_N
    if unpadded_K and block_m == 16:
        K = unpadded_K

    config = get_kernel_config_gluon(M, N, K, routing_data, apply_swiglu)

    if apply_swiglu:
        apply_swiglu_matmul = True
        reduction_n_matmul = 2
        apply_swiglu_reduction = False
        reduction_n_reduction = 1
    else:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
        apply_swiglu_reduction = False
        reduction_n_reduction = 1

    y, y_final = allocate_output(
        M,
        N,
        out_dtype,
        reduction_n_matmul,
        reduction_n_reduction,
        routing_data,
        scatter_indx,
        config["split_k"],
        x.device,
    )
    stride_bias = None if bias is None else bias.stride(0)

    expt_data = routing_data.expt_data
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map

    grid_m = routing_data.n_blocks(M, config["block_m"])
    grid_n = triton.cdiv(N, config["block_n"])
    grid = grid_m * grid_n * config["split_k"]

    _moe_gemm_a16w16_gluon[(grid,)](
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        x,
        x.stride(0),
        x.stride(1),
        w,
        w.stride(0),  # stride_w_e
        w.stride(1),  # stride_w_k
        w.stride(2),  # stride_w_n
        bias,
        stride_bias,
        gammas,
        M,
        N,
        K,
        gather_indx,
        expt_hist,
        expt_token_offs_raw,
        expt_hist_sum,
        expt_block_pid_map,
        grid_m,
        grid_n,
        apply_swiglu_matmul,
        alpha,
        limit,
        reduction_n_matmul,
        swiglu_add_residual,
        routing_data.n_expts_act,
        config["block_m"],
        config["block_n"],
        config["block_k"],
        config["group_m"],
        XCD_SWIZZLE=config["xcd_swizzle"],
        NUM_BUFFERS=config["num_buffers"],
        NUM_SUBTILES=config["num_subtiles"],
        EVEN_K=K % config["block_k"] == 0,
        **get_moe_a16w16_layouts(
            config["num_warps"],
            config["block_m"],
            config["block_n"],
            config["block_k"],
            config["block_n"] // reduction_n_matmul,
            config["num_ctas"],
            (
                16
                if (gather_indx is not None and gather_indx.dtype == torch.uint16)
                else 8
            ),
            config["cluster_split"],
        ),
        num_warps=config["num_warps"],
        UPCAST_INDICES=should_upcast_indices(x, w, y),
        waves_per_eu=config["waves_per_eu"],
        num_ctas=config["num_ctas"],
    )

    group_indx = (
        None
        if scatter_indx is None
        else scatter_indx.view(-1, routing_data.n_expts_act)
    )
    y_final = reduce_grouped(
        y,
        group_indx,
        y_final,
        apply_swiglu_reduction,
        alpha,
        limit,
        reduction_n_reduction,
        out_dtype=out_dtype,
        swiglu_add_residual=swiglu_add_residual,
    )

    return y_final


def fused_moe_bf16_g1u1_gfx1201_large_m(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
):
    """gfx1201 BF16 G1U1 selected-large-M path built on current MoE ops."""
    assert w1.is_contiguous()
    assert w2.is_contiguous()
    experts = w1.shape[0]
    route_cfg = get_topk_align_config()
    routing_data, gather_indx, scatter_indx, sorted_weights = build_routing_from_topk(
        topk_ids, topk_weight, num_experts=experts, block_m=route_cfg["block_m"]
    )
    stage1 = moe_gemm_a16w16(
        hidden_states.contiguous(),
        w1,
        routing_data=routing_data,
        gather_indx=gather_indx,
        apply_swiglu=True,
        out_dtype=torch.bfloat16,
        backend="triton",
    )
    ret = moe_gemm_a16w16(
        stage1.contiguous(),
        w2,
        routing_data=routing_data,
        scatter_indx=scatter_indx,
        gammas=sorted_weights,
        out_dtype=torch.bfloat16,
        backend="triton",
    )
    if output is None:
        return ret
    return output.copy_(ret)
