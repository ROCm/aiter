# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs.py

import itertools
import torch
import triton
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton._triton_kernels.moe.moe_op_gemm_int8_smoothquant import (
    _moe_gemm_int8_smoothquant,
    _reduce_grouped,
)

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


def preshuffle_weights(w: torch.Tensor) -> torch.Tensor:
    """
    Preshuffle int8 weight from (E, K, N) to MFMA-friendly layout (E, N//16, K*16).

    Matches the shuffle_weight pattern from aiter.ops.shuffle for INT8:
      layout=(16, 16), BK=32, K_lane=16, BN=16

    The transformation:
      1. Transpose to (E, N, K)
      2. View as (E, N//16, 16, K//32, 2, 16) - decompose into MFMA tile blocks
      3. Permute to (E, N//16, K//32, 2, 16, 16) - reorder for register layout
      4. View as (E, N//16, K*16) - flatten K dimension

    Args:
        w: int8 weight tensor of shape (E, K, N) where
           - E = number of experts
           - K = input dimension (must be divisible by 32)
           - N = output dimension (must be divisible by 16)

    Returns:
        Preshuffled weight tensor of shape (E, K * 16, N // 16)
    """
    assert w.dtype == torch.int8, f"Expected int8 weights, got {w.dtype}"
    assert w.ndim == 3, f"Expected 3D weight tensor (E, K, N), got {w.ndim}D"
    E, K, N = w.shape
    assert K % 32 == 0, f"K ({K}) must be divisible by 32 for MFMA preshuffling"
    assert N % 16 == 0, f"N ({N}) must be divisible by 16 for MFMA preshuffling"

    # Transpose to (E, N, K)
    w = w.transpose(1, 2)

    # Preshuffle
    w = w.view(E, N // 16, 16, K // 32, 2, 16)
    w = w.permute(0, 1, 3, 4, 2, 5).contiguous()

    # Reshape to (E, N // 16, K * 16)
    w = w.view(E, N // 16, K * 16)

    # Transpose back to (E, K, N)
    w = w.transpose(1, 2)

    return w


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
    # final output
    if routing_data.n_expts_act == 1 or scatter_indx is None:
        y_rows = M
    else:
        y_rows = (
            scatter_indx.shape[0] // routing_data.n_expts_act
        )  # compressed number of rows
    matmul_shape = (split_k, M, N // reduction_n_matmul)
    final_shape = (y_rows, N // reduction_n_matmul // reduction_n_reduction)
    matmul_output = torch.empty(matmul_shape, device=device, dtype=out_dtype)
    if scatter_indx is not None or split_k > 1:
        final_output = torch.empty(final_shape, device=device, dtype=out_dtype)
    else:
        final_output = None
    return matmul_output, final_output


def get_kernel_config(m, n, k, routing_data):
    block_m = routing_data.block_m
    group_m = 1
    w_cache_modifier = ".cg" if block_m <= 32 else None
    split_k = 1
    num_cus = get_num_sms()

    if block_m == 16:
        block_n = 64
        block_k = 256
        num_warps = 4
        num_stages = 2
        kpack = 2

        grid_m = routing_data.n_blocks(m, block_m)
        grid_n = triton.cdiv(n, block_n)
        grid = grid_m * grid_n * split_k
        while block_n >= 64 and grid < num_cus:
            block_n = block_n // 2
            grid_m = routing_data.n_blocks(m, block_m)
            grid_n = triton.cdiv(n, block_n)
            grid = grid_m * grid_n * split_k
    else:
        block_n = 128
        block_k = 128
        num_warps = 8
        num_stages = 2
        kpack = 1

    ret = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "group_m": group_m,
        "w_cache_modifier": w_cache_modifier,
        "split_k": split_k,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": kpack,
    }
    return ret


def reduce_grouped(
    x: torch.Tensor,
    indx: torch.Tensor,
    out: torch.Tensor,
    alpha=1.0,
    limit=1.0,
    reduction_n=1,
    apply_activation: bool = False,
    out_dtype: torch.dtype = None,
    add_residual: bool = False,
):
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
    num_blocks = triton.cdiv(x.shape[-1], BLOCK_N)

    _reduce_grouped[(num_blocks, num_groups)](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out,
        out.stride(0),
        out.stride(1),
        indx,
        x.shape[0],
        x.shape[-1],
        alpha,
        limit,
        reduction_n,
        BLOCK_N=BLOCK_N,
        EVEN_N=(x.shape[-1] % BLOCK_N == 0),
        K=K,
        num_warps=2,
        ADD_RESIDUAL=add_residual,
        APPLY_ACTIVATION=apply_activation,
    )
    return out


# -----------------------------------------------------------------------------
# Triton Implementation
# -----------------------------------------------------------------------------


def moe_gemm_int8_smoothquant(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor = None,
    routing_data: RoutingData = None,
    gather_indx: torch.Tensor = None,
    scatter_indx: torch.Tensor = None,
    gammas: torch.Tensor = None,
    preshuffled: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
    apply_activation: bool = False,
    add_residual: bool = False,
    alpha: float = 1.0,
    limit: float = 1.0,
):
    """
    Performs MoE matrix multiplication with int8 quantized inputs:
    Y = (X @ W) * x_scale * w_scale

    Gated Activation (apply_activation=True):
        Input W must have shape [E, K, 2N] (double-width for gating)
        Output shape: [M, N] (dimension reduced by half)

        Then applies gated activation:
            silu(x[:N], alpha) * x[N:]

    Args:
        add_residual: If True, adds 1 to the linear component
            silu(x[:N], alpha) * (x[N:] + 1)
        apply_activation: If False, no activation applied (alpha set to 0 internally)
    """
    assert x.dtype == torch.int8, f"Expected int8 activations, got {x.dtype}"
    assert w.dtype == torch.int8, f"Expected int8 weights, got {w.dtype}"
    assert x_scale.dtype == torch.float32, f"Expected fp32 x_scale, got {x_scale.dtype}"
    assert w_scale.dtype == torch.float32, f"Expected fp32 w_scale, got {w_scale.dtype}"

    # determine shapes
    M = x.shape[-2] if gather_indx is None else gather_indx.shape[0]
    K, N = x.shape[-1], w.shape[-1]
    if preshuffled:
        N *= 16
    # compute optimization flags
    config = get_kernel_config(M, N, K, routing_data)
    if apply_activation:
        if config["split_k"] > 1:
            reduction_n_matmul = 1
            reduction_n_reduction = 2
        else:
            reduction_n_matmul = 2
            reduction_n_reduction = 1
    else:
        reduction_n_matmul = 1
        reduction_n_reduction = 1
        alpha = 0
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
    # launch kernel
    _moe_gemm_int8_smoothquant[(grid,)](
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        x,
        x.stride(0),
        x.stride(1),
        x_scale,
        x_scale.stride(0) if x_scale.ndim > 0 else 0,
        w,
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w_scale,
        w_scale.stride(0),
        w_scale.stride(1) if w_scale.ndim > 1 else 0,
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
        alpha,
        limit,
        reduction_n_matmul,
        (alpha != 0) and (config["split_k"] == 1),  # APPLY_ACTIVATION
        add_residual,
        routing_data.n_expts_act,
        config["block_m"],
        config["block_n"],
        config["block_k"],
        config["group_m"],
        PRESHUFFLED=preshuffled,
        EVEN_K=K % config["block_k"] == 0,
        MASK_K_LIMIT=K % config["block_k"],
        SPLIT_K=config["split_k"],
        W_CACHE_MODIFIER=config["w_cache_modifier"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        UPCAST_INDICES=should_upcast_indices(x, w, y),
        waves_per_eu=config["waves_per_eu"],
        matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
        kpack=config["kpack"],
    )
    # Build grouped reduction inputs in a uniform way
    group_indx = (
        None
        if scatter_indx is None
        else scatter_indx.view(-1, routing_data.n_expts_act)
    )
    y_final = reduce_grouped(
        y,
        group_indx,
        y_final,
        alpha,
        limit,
        reduction_n_reduction,
        apply_activation=(alpha != 0)
        and (config["split_k"] > 1),  # apply activation if split_k > 1
        out_dtype=out_dtype,
        add_residual=add_residual,
    )

    return y_final


# -----------------------------------------------------------------------------
# Reference Implementation
# -----------------------------------------------------------------------------


def swiglu_torch(a, alpha, limit, add_residual=False):
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


def moe_gemm_smoothquant_torch(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor = None,
    routing_data: RoutingData = None,
    gather_indx: torch.Tensor = None,
    scatter_indx: torch.Tensor = None,
    gammas: torch.Tensor = None,
    apply_activation: bool = False,
    add_residual: bool = False,
    alpha: float = 1.0,
    limit: float = 1.0,
):
    if bias is not None and bias.ndim == 1:
        bias = bias.view(1, *bias.shape)
    if w.ndim == 2:
        w = w.view(1, *w.shape)

    n_expts_act = routing_data.n_expts_act
    # memory offsets
    if routing_data.n_expts_tot > 1:
        sizes = routing_data.expt_hist
        off = torch.zeros(sizes.shape[0] + 1, dtype=torch.int32, device=x.device)
        off[1:] = torch.cumsum(sizes, 0)
        offs = list(itertools.pairwise(off))
    else:
        offs = [[0, x.shape[0]] for _ in range(w.shape[0])]
    # compute
    n_rows = x.shape[0] if gather_indx is None else gather_indx.shape[0]
    n_cols = w.shape[-1] // 2 if apply_activation else w.shape[-1]
    y = torch.zeros((n_rows, n_cols), device=x.device, dtype=torch.float32)
    for i, (lo, hi) in enumerate(offs):
        if gather_indx is None:
            idx = torch.arange(lo, hi, device=x.device)
        else:
            idx = gather_indx[lo:hi] // n_expts_act
        out = (
            torch.matmul(x[idx, :].float(), w[i].float())
            * x_scale[idx, None]
            * w_scale[i, None, :]
        )
        if bias is not None:
            out = out + bias[i, :]
        if apply_activation:
            out = swiglu_torch(out, alpha, limit, add_residual)
        if gammas is not None:
            out = out * gammas[lo:hi, None]
        y[lo:hi, :] = out
    if scatter_indx is None:
        return y
    # accumulate output from all experts
    n_rows_out = y.shape[0] // n_expts_act
    out = torch.zeros((n_rows_out, y.shape[-1]), dtype=torch.float32, device=x.device)
    src_idx = scatter_indx.view(-1, n_expts_act)
    for i in range(n_rows_out):
        out[i, :] = y[src_idx[i], :].sum(0)

    return out
