# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs.py

from dataclasses import dataclass
import itertools
import sys
import torch
import triton
from enum import Enum, auto
import math
from triton_kernels.routing import GatherIndx, RoutingData, ScatterIndx
from aiter.ops.triton._moe_op_gemm_a8w4 import _moe_gemm_a8w4, _reduce_grouped, _downcast_to_static_fp8
import os
from aiter.utility.triton.triton_metadata_redirect import AOTMetadataContext
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH

# -----------------------------------------------------------------------------
#                    Matrix Multiplication + Outer Gather/Scatter
# -----------------------------------------------------------------------------


def can_overflow_int32(tensor: torch.Tensor):
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32


def should_upcast_indices(*args):
    return any(tensor is not None and can_overflow_int32(tensor) for tensor in args)


def allocate_output(x, w, out_dtype, reduction_n_matmul, reduction_n_reduction, routing_data, gather_indx, scatter_indx, split_k):
    # ---- output ------
    N = w.shape[-2] * 16
    # by default - M is number of rows in the activations
    M = x.shape[-2]
    # if the activations are gathered, then M is number of gather indices
    if gather_indx is not None:
        M = gather_indx.src_indx.shape[0]
    # final output
    if routing_data.n_expts_act == 1 or scatter_indx is None:
        y_rows = M
    else:
        y_rows = scatter_indx.src_indx.shape[0] // routing_data.n_expts_act # compressed number of rows
    matmul_shape = (split_k, M, N // reduction_n_matmul)
    final_shape = (y_rows, N // reduction_n_matmul // reduction_n_reduction)
    matmul_output = torch.zeros(matmul_shape, device=x.device, dtype=out_dtype)
    if scatter_indx or split_k > 1:
        final_output = torch.empty(final_shape, device=x.device, dtype=out_dtype)
    else:
        final_output = None
    return matmul_output, final_output


def get_kernel_config(
    m,
    n,
    k,
    routing_data
):
    # tokens per expert
    if routing_data is None:
        tokens_per_expt = m
    elif routing_data.expected_tokens_per_expt is None:
        tokens_per_expt = max(1, m // routing_data.n_expts_tot)
    else:
        tokens_per_expt = routing_data.expected_tokens_per_expt

    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
    if block_m == 64:
        block_m = 32
    group_m = 4
    num_xcds = 8
    xcd_swizzle = num_xcds
    w_cache_modifier = ".cg" if block_m <= 32 else None
    num_stages = 2

    split_k = 1
    if block_m == 16:
        block_n = 128
        block_k = 256
        num_warps = 4
    else:
        # for scale preshuffling
        block_n = 512
        block_k = 256
        num_warps = 8

    ret = {
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
        "kpack": 1
    }
    return ret


def swizzle_scales(data):
    NON_K_PRESHUFFLE_BLOCK_SIZE = 32
    block_shape = data.shape
    SCALE_K = block_shape[-2]
    N = block_shape[-1]
    data = data.transpose(-1, -2)
    data = data.view(-1, N // NON_K_PRESHUFFLE_BLOCK_SIZE, 2, 16, SCALE_K // 8, 2, 4, 1)
    data = data.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
    E = block_shape[0]
    data = data.reshape(E, N // 32, SCALE_K * 32)
    return data.transpose(-1, -2)


def downcast_to_static_fp8(x: torch.Tensor, scale: torch.Tensor):
    M, N = x.shape
    y = torch.empty((M, N), dtype=torch.float8_e4m3fn, device="cuda")

    BLOCK_M = min(triton.next_power_of_2(M), 128)
    if M <= 4096:
        BLOCK_N = 32
    else:
        BLOCK_N = 64
    grid_m = triton.cdiv(x.shape[0], BLOCK_M)
    grid_n = triton.cdiv(x.shape[1], BLOCK_N)

    _downcast_to_static_fp8[(grid_m, grid_n)](x, x.stride(0), x.stride(1),
                                            y, y.stride(0), y.stride(1),
                                            scale,
                                            M, N, BLOCK_M, BLOCK_N,
                                            num_warps=8)

    return y


def reduce_grouped(x: torch.Tensor, indx: torch.Tensor, out: torch.Tensor,
                   apply_swiglu = False, alpha = 1.0, limit = 1.0, reduction_n = 1,
                   out_dtype: bool = None):
    """
    In-place grouped row reduction.

    Arguments
    - x: Tensor[AnyFloat] of shape [(num_groups * K), N]
    - indx: Tensor[Int] of shape [num_groups, K]

    Description
    For each group g in [0, num_groups), this routine sums the K rows of `x`
    specified by `indx[g, :]` and overwrites the row corresponding to the first
    valid (non-negative) index with the per-group sum. Accumulation is performed
    in float32 for numerical stability, and the result is written back in the
    dtype of `x`.

    Behavior and edge cases
    - Invalid (-1) entries are skipped during accumulation and do not generate
      memory traffic. If a group has no valid entries, nothing is written for
      that group.
    - Reduction is performed tile-by-tile along the N dimension within a single
      kernel launch (persistent along N) to minimize launch overhead.

    Performance notes
    - Memory traffic per group is approximately (valid_rows_read + 1) * N * sizeof(x),
      plus index reads. With no invalid entries, this becomes (K + 1) reads/writes
      of length N per group.

    Returns
    - The input tensor `x` (modified in place).
    """
    if indx is None and x.shape[0] == 1:
        return x.squeeze(0)
    if indx is not None:
        num_groups = indx.shape[0]
    else:
        num_groups = x.shape[-2]
    K = 1 if indx is None else indx.shape[1]
    out_dtype = x.dtype if out_dtype is None else out_dtype
    assert x.shape[-1] % reduction_n == 0
    BLOCK_N = 512
    _reduce_grouped[(num_groups, )](
        x, x.stride(0), x.stride(1), x.stride(2),  #
        out, out.stride(0), out.stride(1),  #
        indx,  #
        x.shape[0], x.shape[-1],  #
        apply_swiglu, alpha, limit, reduction_n,
        BLOCK_N=BLOCK_N, K=K,  #
        num_warps=1,  #
    )
    return out

# -----------------------------------------------------------------------------
# Triton Implementation
# -----------------------------------------------------------------------------

def moe_gemm_a8w4(x, w, x_scales, w_scales, 
               x_static_scale = None, quant_static_scale = None,
               bias = None,
               routing_data: RoutingData | None = None,
               gather_indx: GatherIndx | None = None,
               scatter_indx: ScatterIndx | None = None,
               gammas = None,
               swizzle_mx_scale = None,
               out_dtype = torch.bfloat16,
               apply_swiglu = False,
               alpha = 1.0,
               limit = 1.0
               ):
    """
    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])
    """
    assert w.stride(-1) == 1, "`w` must be column-major when it has data-type mxfp"
    x_has_mx = x_scales is not None
    if x_has_mx: assert x.stride(-1) == 1, "'x' must be row-major when it has data-type mxfp"
    if x_has_mx:
        stride_x_mx_m = x_scales.stride(0)
        stride_x_mx_k = x_scales.stride(1)
    else:
        stride_x_mx_m = 0
        stride_x_mx_k = 0
    # determine shapes
    M = x.shape[-2] if gather_indx is None else gather_indx.src_indx.shape[0]
    K, N = x.shape[-1], w.shape[-2] * 16
    # compute optimization flags
    config = get_kernel_config(M, N, K, routing_data)
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
    y, y_final = allocate_output(x, w, out_dtype, reduction_n_matmul, reduction_n_reduction, routing_data, gather_indx, scatter_indx, config["split_k"])
    stride_bias = None if bias is None else bias.stride(0)
    # moe metadata
    expt_data = routing_data.expt_data
    block_m = config["block_m"]
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[block_m][-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map[block_m]
    # spmd grid
    if expt_block_pid_map is not None:
        grid_m = routing_data.n_blocks(M, config["block_m"])
    else:
        grid_m = triton.cdiv(M, config["block_m"])
    if N == 6144:
        N = 5760
    elif N == 3072:
        N = 2880
    grid_n = triton.cdiv(N, config["block_n"])
    grid = grid_m * grid_n * config["split_k"]

    if gather_indx:
        layer = "moe1"
    else:
        layer = "moe2"
    metadata_pth = f"{AITER_TRITON_CONFIGS_PATH}/moe/aot/{layer}_blockm{block_m}"
    if os.path.exists(metadata_pth):
        with AOTMetadataContext(_moe_gemm_a8w4.fn.__name__, f"{metadata_pth}"):
            _moe_gemm_a8w4[(grid,)](
                y, y.stride(0), y.stride(1), y.stride(2),
                x, x.stride(0), x.stride(1),
                x_scales, stride_x_mx_m, stride_x_mx_k,
                w, w.stride(0), w.stride(1), w.stride(2),
                w_scales, w_scales.stride(0), w_scales.stride(1), w_scales.stride(2),
                x_static_scale, quant_static_scale,
                bias, stride_bias,
                gammas,
                N, K,
                None if gather_indx is None else gather_indx.src_indx,
                expt_hist, expt_token_offs_raw, expt_hist_sum, expt_block_pid_map,
                grid_m, grid_n,
                apply_swiglu_matmul, alpha, limit, reduction_n_matmul,
                routing_data.n_expts_act,
                config["block_m"],
                config["block_n"],
                config["block_k"],
                config["group_m"],
                XCD_SWIZZLE=config["xcd_swizzle"],
                SWIZZLE_MX_SCALE=swizzle_mx_scale,
                SPLIT_K=config["split_k"],
                EVEN_K=K % config["block_k"] == 0,
                W_CACHE_MODIFIER=config["w_cache_modifier"],
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
                UPCAST_INDICES=should_upcast_indices(x, w, y),
                waves_per_eu=config["waves_per_eu"],
                matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
                kpack=config["kpack"])
    # Build grouped reduction inputs in a uniform way
    group_indx = None if scatter_indx is None else scatter_indx.src_indx.view(-1, routing_data.n_expts_act)
    y_final = reduce_grouped(
        y,
        group_indx,
        y_final,
        apply_swiglu_reduction, alpha, limit, reduction_n_reduction,
        out_dtype=out_dtype,
    )
    return y_final

# -----------------------------------------------------------------------------
# Reference Implementation
# -----------------------------------------------------------------------------

def swiglu_torch(a, alpha, limit):
    a_gelu = a[..., ::2]
    if limit is not None:
        a_gelu = a_gelu.clamp(max=limit)
    a_linear = a[..., 1::2]
    if limit is not None:
        a_linear = a_linear.clamp(min=-limit, max=limit)

    out_gelu = a_gelu * torch.sigmoid(alpha * a_gelu)
    out = out_gelu * (a_linear + 1)
    return out


def moe_gemm_torch(x, w, bias,
                 routing_data: RoutingData = None,
                 gather_indx: GatherIndx = None,
                 scatter_indx: ScatterIndx = None,
                 gammas = None,
                 apply_swiglu = False,
                 alpha = 1.0,
                 limit = 1.0
                 ):
    is_input_batched = x.ndim == 3
    assert x.dtype.itemsize > 1
    assert w.dtype.itemsize > 1
    if is_input_batched:
        assert gather_indx is None, "gather not supported in batched mode"
        assert scatter_indx is None, "scatter not supported in batched mode"
        assert routing_data is None, "routing not supported in batched mode"
        assert w.ndim == 3 and w.shape[0] == x.shape[0]
    if bias is not None and bias.ndim == 1:
        bias = bias.view(1, *bias.shape)
    if w.ndim == 2:
        w = w.view(1, *w.shape)
    if x.ndim == 2:
        x = x.view(1, *x.shape)
    if routing_data is None:
        routing_data = RoutingData(None, None, w.shape[0], 1)
    n_expts_act = routing_data.n_expts_act
    # memory offsets
    if routing_data.n_expts_tot > 1 and not is_input_batched:
        sizes = routing_data.expt_hist
        off = torch.zeros(sizes.shape[0] + 1, dtype=torch.int32)
        off[1:] = torch.cumsum(sizes, 0)
        offs = list(itertools.pairwise(off))
    else:
        offs = [[0, x.shape[1]] for _ in range(w.shape[0])]
    # compute
    n_rows = x.shape[1] if gather_indx is None else gather_indx.dst_indx.shape[0]
    n_cols = w.shape[-1] // 2 if apply_swiglu else w.shape[-1]
    y = torch.zeros((x.shape[0], n_rows, n_cols), device=x.device, dtype=x.dtype)
    for i, (lo, hi) in enumerate(offs):
        if gather_indx is None:
            idx = torch.arange(lo, hi, device=x.device)
        else:
            idx = gather_indx.src_indx[lo:hi] // n_expts_act
        batch = i if is_input_batched else 0
        out = torch.matmul(x[batch, idx, :].float(), w[i].float())
        if bias is not None:
            out += bias[i, :]
        if apply_swiglu:
            out = swiglu_torch(out, alpha, limit)
        if gammas is not None:
            out *= gammas[lo:hi, None]
        y[batch, lo:hi, :] = out
    if not is_input_batched:
        y = y.view(y.shape[1], y.shape[2])
    if scatter_indx is None:
        return y
    # accumulate output from all experts
    n_rows = y.shape[0] // n_expts_act
    out = torch.zeros((n_rows, y.shape[-1]), dtype=torch.float32, device=x.device)
    for i, (lo, hi) in enumerate(offs):
        dst_idx = scatter_indx.dst_indx[lo:hi] // n_expts_act
        msk = dst_idx != -1
        out[dst_idx[msk], :] += y[lo:hi, :][msk, :].float()
    return out
