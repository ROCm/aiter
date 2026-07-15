# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_details/_matmul.py

import itertools
import os
import torch
import triton
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
from aiter.ops.triton._triton_kernels.moe.moe_op_gemm_a16w4 import (
    _moe_gemm_a16w4 as _moe_gemm_a16w4_triton,
)
from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a16w4 import (
    _moe_gemm_a16w4 as _moe_gemm_a16w4_gluon,
)
from aiter.ops.triton.moe.reduce import reduce_grouped
from aiter.ops.triton.utils._triton.arch_info import get_arch

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


def _env_config_int(prefix: str, name: str, k: int, default: int) -> int:
    value = os.environ.get(f"{prefix}_{name}_K{k}", os.environ.get(f"{prefix}_{name}"))
    if value is None:
        return default
    return int(value)


def allocate_output(
    x,
    w,
    out_dtype,
    reduction_n_matmul,
    reduction_n_reduction,
    routing_data,
    gather_indx,
    scatter_indx,
    block_m,
    split_k,
):
    # ---- output ------
    N = w.shape[-1]
    # by default - M is number of rows in the activations
    M = x.shape[-2]
    # if the activations are gathered, then M is number of gather indices
    if gather_indx is not None:
        M = gather_indx.shape[0]
    # final output
    if routing_data.n_expts_act == 1 or scatter_indx is None:
        y_rows = M
    else:
        y_rows = (
            scatter_indx.shape[0] // routing_data.n_expts_act
        )  # compressed number of rows
    matmul_shape = (split_k, M, N // reduction_n_matmul)
    final_shape = (y_rows, N // reduction_n_matmul // reduction_n_reduction)
    matmul_output = torch.empty(matmul_shape, device=x.device, dtype=out_dtype)
    if scatter_indx is not None or split_k > 1:
        final_output = torch.empty(final_shape, device=x.device, dtype=out_dtype)
    else:
        final_output = None
    return matmul_output, final_output


def get_kernel_config(m, n, k, routing_data):
    block_m = routing_data.block_m
    group_m = 4
    num_xcds = 8
    xcd_swizzle = num_xcds
    w_cache_modifier = ".cg" if block_m <= 32 else None
    num_stages = 1
    split_k = 1
    block_k = 256
    waves_per_eu = 0
    kpack = 1

    if block_m == 16:
        block_n = 128
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
            num_warps = 4
        elif n <= 4096:
            block_n = 256
            num_warps = 8
        else:
            block_n = 512
            num_warps = 8

    else:
        # Large block_m (dense prefill grouped MoE, e.g. block_m == 128).
        # Tuned on gfx1250 for the MiniMax-M3 routed GEMMs (N=6144, K in
        # {6144, 3072}, M ~= routed prefill tokens) via rocprofv3 kernel-trace
        # DURATION sweeps. BLOCK_N=256 with a K-dependent BLOCK_K beats the old
        # flat BLOCK_N=512 / BLOCK_K=256 by ~1.4x end-to-end on the fused path.
        num_warps = 8
        block_n = 256
        if k >= 4096:
            # deep-K projection (gate/up, K == hidden): larger K tile + more
            # waves hides the extra K iterations.
            block_k = 512
            waves_per_eu = 2
            group_m = 4
            kpack = 1
        else:
            # shallow-K projection (down, K == intermediate): square K tile,
            # kpack=2 packs two mxfp4 K-slices per MFMA for better throughput.
            block_k = 256
            waves_per_eu = 0
            group_m = 1
            kpack = 2

    ret = {
        "block_m": block_m,
        "block_n": _env_config_int("AITER_MOE_A16W4_TRITON", "BLOCK_N", k, block_n),
        "block_k": _env_config_int("AITER_MOE_A16W4_TRITON", "BLOCK_K", k, block_k),
        "num_warps": _env_config_int(
            "AITER_MOE_A16W4_TRITON", "NUM_WARPS", k, num_warps
        ),
        "num_stages": _env_config_int(
            "AITER_MOE_A16W4_TRITON", "NUM_STAGES", k, num_stages
        ),
        "group_m": _env_config_int("AITER_MOE_A16W4_TRITON", "GROUP_M", k, group_m),
        "xcd_swizzle": _env_config_int(
            "AITER_MOE_A16W4_TRITON", "XCD_SWIZZLE", k, xcd_swizzle
        ),
        "w_cache_modifier": w_cache_modifier,
        "split_k": split_k,
        "waves_per_eu": _env_config_int(
            "AITER_MOE_A16W4_TRITON", "WAVES_PER_EU", k, waves_per_eu
        ),
        "matrix_instr_nonkdim": _env_config_int(
            "AITER_MOE_A16W4_TRITON", "MATRIX_INSTR_NONKDIM", k, 16
        ),
        "kpack": _env_config_int("AITER_MOE_A16W4_TRITON", "KPACK", k, kpack),
    }
    return ret


def get_kernel_config_gluon(m, n, k, routing_data):
    block_m = routing_data.block_m
    group_m = 4
    xcd_swizzle = 1
    w_cache_modifier = ".cg" if block_m <= 32 else None
    num_stages = 2
    split_k = 1
    block_k = 512
    num_buffers = 1
    waves_per_eu = 0

    if block_m == 16:
        block_n = 128
        num_warps = 4
    elif block_m == 32:
        block_n = 128
        num_warps = 4
    else:
        # Large-block prefill (block_m=128) on gfx1250, re-tuned for the
        # MiniMax-M3 A16W4 routed GEMMs (N=6144, K in {6144, 3072}, M = routed
        # prefill tokens in [~6.5k, 131k]) via per-shape CUDA-event sweeps of
        # each _moe_gemm_a16w4 launch in isolation. A 256-wide N tile with a
        # 512-deep K tile, 4 warps, 2 pipeline stages and a single buffer beat
        # every other candidate (incl. the tuned Triton path) consistently
        # across M -- the grid is >=7968 CTAs so the GPU is saturated and the
        # optimum is essentially M-independent. ~18% faster per GEMM than tuned
        # Triton, ~15% end-to-end on the fused path. xcd_swizzle is the only
        # projection-dependent knob: gate/up (deep K=6144) prefers 2, down
        # (K=3072) prefers 8.
        block_n = 256
        block_k = 512
        num_warps = 4
        num_stages = 2
        waves_per_eu = 0
        group_m = 4
        num_buffers = 1
        xcd_swizzle = 2 if k >= 4096 else 8

    return {
        "block_m": block_m,
        "block_n": _env_config_int("AITER_MOE_A16W4_GLUON", "BLOCK_N", k, block_n),
        "block_k": _env_config_int("AITER_MOE_A16W4_GLUON", "BLOCK_K", k, block_k),
        "num_warps": _env_config_int(
            "AITER_MOE_A16W4_GLUON", "NUM_WARPS", k, num_warps
        ),
        "num_stages": _env_config_int(
            "AITER_MOE_A16W4_GLUON", "NUM_STAGES", k, num_stages
        ),
        "group_m": _env_config_int("AITER_MOE_A16W4_GLUON", "GROUP_M", k, group_m),
        "xcd_swizzle": _env_config_int(
            "AITER_MOE_A16W4_GLUON", "XCD_SWIZZLE", k, xcd_swizzle
        ),
        "w_cache_modifier": w_cache_modifier,
        "split_k": split_k,
        "waves_per_eu": _env_config_int(
            "AITER_MOE_A16W4_GLUON", "WAVES_PER_EU", k, waves_per_eu
        ),
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
        "num_buffers": _env_config_int(
            "AITER_MOE_A16W4_GLUON", "NUM_BUFFERS", k, num_buffers
        ),
    }


def _selected_backend() -> str:
    backend = os.environ.get("AITER_MOE_A16W4_BACKEND")
    if backend is None:
        # Default to "auto": gfx1250 large-block (block_m>=64) prefill GEMMs run
        # on the tuned Gluon path (see get_kernel_config_gluon), while small
        # block_m decode-style shapes stay on Triton. Set AITER_MOE_A16W4_BACKEND
        # explicitly to override.
        backend = "gluon" if os.environ.get("AITER_MOE_A16W4_GLUON") == "1" else "auto"
    backend = backend.lower()
    if backend not in {"triton", "gluon", "auto"}:
        raise ValueError(f"unknown AITER_MOE_A16W4_BACKEND={backend!r}")
    return backend


# -----------------------------------------------------------------------------
# Triton Implementation
# -----------------------------------------------------------------------------


def moe_gemm_a16w4(
    x,
    w,
    x_scales,  # This argument is for API compatibility with lower-precision data types. For a16, this should be set to None
    w_scales,
    x_static_scale=None,  # This argument is for API compatibility with lower-precision data types. For a16, this should be set to None
    quant_static_scale=None,  # This argument is for API compatibility with lower-precision data types. For a16, this should be set to None
    bias=None,
    routing_data: RoutingData | None = None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    swizzle_mx_scale=None,
    out_dtype=torch.bfloat16,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    swiglu_add_residual=True,
    unpadded_N=None,
    unpadded_K=None,
):
    """
    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])
    """
    assert w.stride(-2) == 1, "`w` must be column-major when it has data-type mxfp"
    assert x_scales is None, "x_scales must be none"
    assert x_static_scale is None, "x_static_scale must be none"
    assert quant_static_scale is None, "quant_static_scale must be none"

    # determine shapes
    M = x.shape[-2] if gather_indx is None else gather_indx.shape[0]
    K, N = x.shape[-1], w.shape[-1]
    block_m = routing_data.block_m
    if unpadded_N and block_m == 16:
        N = unpadded_N
    if unpadded_K and block_m == 16:
        K = unpadded_K

    # compute optimization flags
    backend = _selected_backend()
    # "auto" (the default) only routes to the tuned Gluon path where it has been
    # validated numerically correct: gfx1250 large-block (block_m>=64) prefill
    # with no MX-scale preshuffling. The Gluon path has a pre-existing accuracy
    # bug with swizzled scales (produces inf under swiglu), so swizzled shapes
    # stay on Triton unless Gluon is requested explicitly.
    use_gluon = get_arch() == "gfx1250" and (
        backend == "gluon"
        or (backend == "auto" and block_m >= 64 and swizzle_mx_scale is None)
    )
    config = (
        get_kernel_config_gluon(M, N, K, routing_data)
        if use_gluon
        else get_kernel_config(M, N, K, routing_data)
    )
    if os.environ.get("AITER_MOE_A16W4_DEBUG_PRINT"):
        print(
            f"moe_gemm_a16w4 backend={backend} use_gluon={use_gluon} "
            f"M={M} N={N} K={K} swiglu={int(bool(apply_swiglu))} config={config}",
            flush=True,
        )
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
        x,
        w,
        out_dtype,
        reduction_n_matmul,
        reduction_n_reduction,
        routing_data,
        gather_indx,
        scatter_indx,
        config["block_m"],
        config["split_k"],
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
    if use_gluon:
        w_scales_kernel = w_scales.transpose(1, 2)
        _moe_gemm_a16w4_gluon[(grid,)](
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
            w_scales_kernel,
            w_scales_kernel.stride(0),
            w_scales_kernel.stride(1),
            w_scales_kernel.stride(2),
            bias,
            stride_bias,
            gammas,
            x.shape[-2],
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
            NUM_BUFFERS=max(
                1, min(config["num_buffers"], triton.cdiv(K, config["block_k"]))
            ),
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
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
    else:
        _moe_gemm_a16w4_triton[(grid,)](
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
            w_scales,
            w_scales.stride(0),
            w_scales.stride(1),
            w_scales.stride(2),
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
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
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

    # memory offsets
    if routing_data.n_expts_tot > 1:
        sizes = routing_data.expt_hist
        off = torch.zeros(sizes.shape[0] + 1, dtype=torch.int32)
        off[1:] = torch.cumsum(sizes, 0)
        offs = list(itertools.pairwise(off))
    else:
        offs = [[0, x.shape[0]] for _ in range(w.shape[0])]

    # compute
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

    # accumulate output from all experts
    scatter_indx = scatter_indx.to(torch.int32)
    n_rows = y.shape[0] // n_expts_act
    out = torch.zeros((n_rows, y.shape[-1]), dtype=torch.float32, device=x.device)
    src_idx = scatter_indx.view(-1, n_expts_act)
    for i in range(n_rows):
        out[i, :] = y[src_idx[i], :].float().sum(0)

    return out
