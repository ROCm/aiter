# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_details/_matmul.py

import itertools
from functools import lru_cache

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a16w16 import (
    _moe_gemm_a16w16 as _moe_gemm_a16w16_gluon,
)
from aiter.ops.triton._triton_kernels.moe.moe_op_gemm_a16w16 import (
    _moe_gemm_a16w16 as _moe_gemm_a16w16_triton,
)
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
from aiter.ops.triton.moe.reduce import reduce_grouped
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)


@lru_cache(maxsize=1)
def _warn_gluon_fallback_once():
    _LOGGER.warning(
        "Gluon was explicitly requested for moe_gemm_a16w16 but is not supported "
        "on this GPU; using Triton."
    )


def _is_gluon_available():
    try:
        return any(supported in get_arch() for supported in _GLUON_SUPPORTED_ARCHS)
    except Exception:  # noqa: BLE001
        return False


def can_overflow_int32(tensor: torch.Tensor):
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32


def should_upcast_indices(*args):
    return any(tensor is not None and can_overflow_int32(tensor) for tensor in args)


def allocate_output(
    M,
    N,
    out_dtype,
    reduction_n_matmul,
    reduction_n_reduction,
    routing_data,
    gather_indx,
    scatter_indx,
    block_m,
    split_k,
    device,
):
    if routing_data.n_expts_act == 1 or scatter_indx is None:
        y_rows = M
    else:
        y_rows = scatter_indx.shape[0] // routing_data.n_expts_act
    matmul_shape = (split_k, M, N // reduction_n_matmul)
    final_shape = (y_rows, N // reduction_n_matmul // reduction_n_reduction)
    matmul_output = torch.empty(matmul_shape, device=device, dtype=out_dtype)
    if scatter_indx is not None or split_k > 1:
        final_output = torch.empty(final_shape, device=device, dtype=out_dtype)
    else:
        final_output = None
    return matmul_output, final_output


def get_kernel_config_triton(m, n, k, routing_data):
    block_m = routing_data.block_m
    group_m = 4
    num_xcds = 8
    xcd_swizzle = num_xcds
    w_cache_modifier = ".cg" if block_m <= 32 else None
    split_k = 1
    num_stages = 1

    if block_m == 16:
        block_n = 128
        block_k = 128
        num_warps = 4

        grid_m = routing_data.n_blocks(m, block_m)
        grid_n = triton.cdiv(n, block_n)
        grid = grid_m * grid_n * split_k
        while block_n >= 64 and grid < 256:
            block_n = block_n // 2
            grid_m = routing_data.n_blocks(m, block_m)
            grid_n = triton.cdiv(n, block_n)
            grid = grid_m * grid_n * split_k

    elif block_m == 32:
        if n <= 1024:
            block_n = 128
            block_k = 128
            num_warps = 4
        else:
            block_n = 256
            block_k = 128
            num_warps = 8

    else:
        block_n = 256
        block_k = 128
        num_warps = 8

    return {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "group_m": group_m,
        "xcd_swizzle": xcd_swizzle,
        "w_cache_modifier": w_cache_modifier,
        "split_k": split_k,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    }


def get_kernel_config_gluon_gfx1250(m, n, k, routing_data):
    block_m = routing_data.block_m
    group_m = 4
    xcd_swizzle = 1
    w_cache_modifier = ".cg" if block_m <= 32 else None
    num_stages = 2
    split_k = 1

    block_n = 128
    num_warps = 4

    if block_m == 16 or block_m == 32:
        block_k = 128
    else:
        block_k = 64

    return {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "group_m": group_m,
        "xcd_swizzle": xcd_swizzle,
        "w_cache_modifier": w_cache_modifier,
        "split_k": split_k,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    }


def moe_gemm_a16w16(
    x: torch.Tensor,
    w: torch.Tensor,
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

    No quantization or dequantization is applied.

    Args:
        x (torch.Tensor): Activations with shape (num_tokens, K). Must be 16-bit.
        w (torch.Tensor): Weights with shape (E, K, N). Must be 16-bit.
        bias (torch.Tensor): Optional bias with shape (E, N).
        routing_data: Generated by the routing kernels.
        gather_indx: Generated along with routing data by routing kernels.
        scatter_indx: Generated along with routing data by routing kernels.
        gammas (torch.Tensor): Optional per-token gamma scales.
        out_dtype (dtype): Output dtype.
        apply_swiglu (bool): Whether to apply SwiGLU activation.
        alpha (float): SwiGLU alpha parameter.
        limit (float): SwiGLU clamp limit.
        swiglu_add_residual (bool): Whether to add SwiGLU residual.
        unpadded_N (int): Unpadded N dimension (used when block_m == 16).
        unpadded_K (int): Unpadded K dimension (used when block_m == 16).
        backend (Optional[str]): "triton", "gluon", or None (auto-detect).

    Returns:
        torch.Tensor: Output with shape (num_tokens, N) or (num_tokens, N//2) with SwiGLU.
    """
    if backend in (None, "gluon"):
        if _is_gluon_available():
            backend = "gluon"
        else:
            if backend == "gluon":
                _warn_gluon_fallback_once()
            backend = "triton"

    backend = backend.lower()
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"

    _LOGGER.info(
        "MOE_GEMM_A16W16: x=%s w=%s backend=%s",
        x.shape,
        w.shape,
        backend,
    )

    assert x.dtype.itemsize == 2, "x must be a 16-bit tensor"
    assert w.dtype.itemsize == 2, "w must be a 16-bit tensor"

    # determine shapes
    M = x.shape[-2] if gather_indx is None else gather_indx.shape[0]
    K, N = x.shape[-1], w.shape[-1]
    block_m = routing_data.block_m
    if unpadded_N and block_m == 16:
        N = unpadded_N
    if unpadded_K and block_m == 16:
        K = unpadded_K

    if backend == "gluon":
        config = get_kernel_config_gluon_gfx1250(M, N, K, routing_data)
    else:
        config = get_kernel_config_triton(M, N, K, routing_data)

    if apply_swiglu and config["split_k"] > 1:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
        apply_swiglu_reduction = True
        reduction_n_reduction = 2
    elif apply_swiglu:
        apply_swiglu_matmul = True
        reduction_n_matmul = 2
        apply_swiglu_reduction = False
        reduction_n_reduction = 1
    else:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
        apply_swiglu_reduction = False
        reduction_n_reduction = 1

    # allocate output memory
    y, y_final = allocate_output(
        M,
        N,
        out_dtype,
        reduction_n_matmul,
        reduction_n_reduction,
        routing_data,
        gather_indx,
        scatter_indx,
        config["block_m"],
        config["split_k"],
        x.device,
    )

    stride_bias = None if bias is None else bias.stride(0)

    # moe metadata
    expt_data = routing_data.expt_data
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map

    # spmd grid
    grid_m = routing_data.n_blocks(M, config["block_m"])
    grid_n = triton.cdiv(N, config["block_n"])
    grid = grid_m * grid_n * config["split_k"]

    if backend == "gluon":
        _moe_gemm_a16w16_gluon[(grid,)](
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
            NUM_BUFFERS=2,
            SPLIT_K=config["split_k"],
            EVEN_K=K % config["block_k"] == 0,
            W_CACHE_MODIFIER=config["w_cache_modifier"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
            UPCAST_INDICES=should_upcast_indices(x, w, y),
            waves_per_eu=config["waves_per_eu"],
            matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
            kpack=config["kpack"],
        )
    else:  # triton
        _moe_gemm_a16w16_triton[(grid,)](
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
            bias,
            stride_bias,
            gammas,
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
            SPLIT_K=config["split_k"],
            EVEN_K=K % config["block_k"] == 0,
            MASK_K_LIMIT=K % config["block_k"],
            W_CACHE_MODIFIER=config["w_cache_modifier"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
            UPCAST_INDICES=should_upcast_indices(x, w, y),
            waves_per_eu=config["waves_per_eu"],
            matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
            kpack=config["kpack"],
        )

    # grouped reduction
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


# -----------------------------------------------------------------------------
# Reference Implementation
# -----------------------------------------------------------------------------


def swiglu_torch(a, alpha, limit, add_residual=True):
    a_gelu = a[..., ::2]
    if limit is not None:
        a_gelu = a_gelu.clamp(max=limit)
    a_linear = a[..., 1::2]
    if limit is not None:
        a_linear = a_linear.clamp(min=-limit, max=limit)

    out_gelu = a_gelu * torch.sigmoid(alpha * a_gelu)
    if add_residual:
        out = out_gelu * (a_linear + 1)
    else:
        out = out_gelu * a_linear

    return out


def moe_gemm_torch(
    x,
    w,
    bias,
    routing_data: RoutingData = None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    add_residual=True,
):
    assert x.dtype.itemsize > 1
    assert w.dtype.itemsize > 1
    if bias is not None and bias.ndim == 1:
        bias = bias.view(1, *bias.shape)
    if w.ndim == 2:
        w = w.view(1, *w.shape)
    n_expts_act = routing_data.n_expts_act

    if routing_data.n_expts_tot > 1:
        sizes = routing_data.expt_hist
        off = torch.zeros(sizes.shape[0] + 1, dtype=torch.int32, device=x.device)
        off[1:] = torch.cumsum(sizes, 0)
        offs = list(itertools.pairwise(off))
    else:
        offs = [[0, x.shape[0]] for _ in range(w.shape[0])]

    n_rows = x.shape[0] if gather_indx is None else gather_indx.shape[0]
    n_cols = w.shape[-1] // 2 if apply_swiglu else w.shape[-1]
    y = torch.zeros((n_rows, n_cols), device=x.device, dtype=x.dtype)
    for i, (lo, hi) in enumerate(offs):
        if gather_indx is None:
            idx = torch.arange(lo, hi, device=x.device)
        else:
            gather_indx = gather_indx.to(torch.int32)
            idx = gather_indx[lo:hi] // n_expts_act
        out = torch.matmul(x[idx, :].float(), w[i].float())
        if bias is not None:
            out += bias[i, :]
        if apply_swiglu:
            out = swiglu_torch(out, alpha, limit, add_residual)
        if gammas is not None:
            out *= gammas[lo:hi, None]
        y[lo:hi, :] = out
    if scatter_indx is None:
        return y

    scatter_indx = scatter_indx.to(torch.int32)
    n_rows = y.shape[0] // n_expts_act
    out = torch.zeros((n_rows, y.shape[-1]), dtype=torch.float32, device=x.device)
    src_idx = scatter_indx.view(-1, n_expts_act)
    for i in range(n_rows):
        out[i, :] = y[src_idx[i], :].float().sum(0)

    return out
