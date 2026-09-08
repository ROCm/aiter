# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs.py

import itertools

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a4w4 import (
    _moe_gemm_a4w4_decode,
    _moe_gemm_a4w4_prefill,
    get_moe_a4w4_layouts_decode,
    get_moe_a4w4_layouts_prefill,
)
from aiter.ops.triton._triton_kernels.moe.moe_op_gemm_a4w4 import (
    _moe_gemm_a4w4,
    _mxfp4_quant_kernel,
)
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
from aiter.ops.triton.moe.reduce import (
    EpCombineScatter,
    reduce_grouped,
    scatter_grouped,
    validate_reduce_out,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.gemm_config_utils import pick_gemm_num_stages
from aiter.ops.triton.utils.moe_config_utils import get_moe_dispatch

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
    preshuffle_weights,
    y_out=None,
    skip_final=False,
    skip_matmul=False,
    out_mx_quant=False,
):
    # ---- output ------
    N = w.shape[-1]
    if preshuffle_weights:
        N = N * 16
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
    if out_mx_quant:
        # MXFP4 emit: the epilogue writes packed e2m1, two values per byte, so
        # the row is half as wide and uint8 rather than out_dtype.
        matmul_shape = (split_k, M, N // reduction_n_matmul // 2)
        matmul_dtype = torch.uint8
    else:
        matmul_shape = (split_k, M, N // reduction_n_matmul)
        matmul_dtype = out_dtype
    final_shape = (y_rows, N // reduction_n_matmul // reduction_n_reduction)
    if skip_matmul:
        # The epilogue scatters straight into a caller-owned window, so nothing
        # ever reads this buffer -- and at (M x hidden) bf16 it is tens of MB
        # per layer, allocated and dirtied for nothing.
        matmul_output = None
    else:
        matmul_output = torch.empty(matmul_shape, device=x.device, dtype=matmul_dtype)
    if skip_final:
        # The rows are delivered elsewhere (expert-parallel scatter), so a
        # reduced output would only be allocated to be thrown away -- at
        # (tokens x hidden) bf16 that is tens of MB per layer. Mirrors a8w4.
        #
        assert y_out is None, "y_out names a reduced output; skip_final has none"
        final_output = None
    elif scatter_indx is not None or split_k > 1:
        final_output = validate_reduce_out(y_out, final_shape, out_dtype, x.device)
    else:
        # No reduction runs: reduce_grouped early-returns the matmul buffer
        # itself (indx is None and split_k == 1), so a caller-provided buffer
        # would be silently dropped. Say so instead of writing nowhere.
        assert y_out is None, (
            "y_out was provided but this call has no grouped reduction "
            "(scatter_indx is None and split_k == 1), so nothing would write "
            "into it -- the result comes straight out of the matmul buffer."
        )
        final_output = None
    return matmul_output, final_output


def get_kernel_config_triton(m, n, k, routing_data):
    block_m = routing_data.block_m
    group_m = 4
    num_xcds = 1
    xcd_swizzle = num_xcds
    w_cache_modifier = ".cg" if block_m <= 32 else None
    arch = get_arch()

    split_k = 1
    if block_m == 16:
        block_n = 128
        block_k = 256
        num_warps = 4
        grid_m = routing_data.n_blocks(m, block_m)
        grid_n = triton.cdiv(n, block_n)
        grid = grid_m * grid_n * split_k
        while block_n >= 64 and grid < 256:
            block_n = block_n // 2
            grid_m = routing_data.n_blocks(m, block_m)
            grid_n = triton.cdiv(n, block_n)
            grid = grid_m * grid_n * split_k
    else:
        # for scale preshuffling
        block_n = 512
        block_k = 256
        num_warps = 4
    num_stages = pick_gemm_num_stages(
        arch, block_m, block_n, block_k, 4, 4, use_async_padding=True
    )

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
        "kpack": 1,
    }
    return ret


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


def get_kernel_config_gluon(m, n, k, routing_data):
    block_m = routing_data.block_m
    num_xcds = 1

    arch = get_arch()
    bucket = m2bucket(m)
    tuned = get_moe_dispatch("A4W4", arch, "gluon")
    key = f"bm{block_m}_n{n}_k{k}_{bucket}"
    if key not in tuned:
        key = f"bm{block_m}_any"
    cfg = tuned[key]
    block_n, block_k, num_buffers, num_warps = (
        cfg["block_n"],
        cfg["block_k"],
        cfg["num_buffers"],
        cfg["num_warps"],
    )

    num_buffers = min(num_buffers, triton.cdiv(k, block_k))

    ret = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "xcd_swizzle": num_xcds,
        "num_buffers": num_buffers,
        "waves_per_eu": 0,
        "num_ctas": 1,
    }
    return ret


# -----------------------------------------------------------------------------
# Triton Implementation
# -----------------------------------------------------------------------------


# This is fixed by spec for MXFP4. Do not tune this.
MXFP4_QUANT_BLOCK_SIZE = 32


def mxfp4_quant(
    x: torch.Tensor,
    block_size_m: int = 16,
    block_size_n: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D tensor `x` of shape [M, K] (bf16/fp16/fp32) to MXFP4 (E2M1) format
    quantized along the K dimension.

    Returns:
    - A packed MXFP4 tensor `x_fp4` of shape [M, N // 2] (stored as uint8), where
        each byte stores two 4-bit values.
    - A block-scale tensor `x_scale` of shape [M, N / 32], where each entry
        corresponds to one MXFP4 quantization block of 32 elements along the K dimension.
    """
    M, N = x.shape
    assert N % MXFP4_QUANT_BLOCK_SIZE == 0
    assert block_size_n % MXFP4_QUANT_BLOCK_SIZE == 0

    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    x_scale = torch.empty(
        (M, N // MXFP4_QUANT_BLOCK_SIZE), dtype=torch.uint8, device=x.device
    )

    grid = (
        triton.cdiv(M, block_size_m),
        triton.cdiv(N, block_size_n),
    )

    _mxfp4_quant_kernel[grid](
        x,
        x_fp4,
        x_scale,
        x.stride(0),
        x.stride(1),
        x_fp4.stride(0),
        x_fp4.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        M,
        N,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        EVEN_M_N=(M % block_size_m == 0) and (N % block_size_n == 0),
    )

    return x_fp4, x_scale


def moe_gemm_a4w4(
    x,
    w,
    x_scales,
    w_scales,
    bias=None,
    routing_data: RoutingData | None = None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    swizzle_mx_scale=None,
    preshuffle_weights=False,
    out_dtype=torch.bfloat16,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    swiglu_add_residual=True,
    unpadded_N=None,
    unpadded_K=None,
    backend=None,
    # Per-gate validity, same layout as scatter_indx. Default None == every gate
    # slot is live, which holds whenever routing() produced the indices. Pass a
    # mask when only some of a token's n_expts_act slots are computed here --
    # expert parallelism, where the other slots belong to another rank and are
    # never written, so the reduce must not sum them. Mirrors moe_gemm_a8w4.
    gate_valid=None,
    # Destination for the grouped reduction's result, instead of a freshly
    # allocated buffer. May be a slice of a taller tensor -- the reduce writes
    # through `out`'s strides -- which lets a caller whose consumer wants more
    # rows than this GEMM produces skip a full-width copy. Requires a grouped
    # reduction to actually run, i.e. scatter_indx is not None or split_k > 1.
    # Mirrors moe_gemm_a8w4.
    y_out=None,
    # Deliver GEMM2's un-reduced rows into an EP combine staging window instead
    # of reducing them locally. Same contract as moe_gemm_a8w4's `ep_scatter`.
    # Folded into the gluon epilogue when `ep_scatter.fused`; otherwise (and
    # always on the triton backend) it falls through to the standalone
    # `scatter_grouped` pass, which writes the same bytes.
    ep_scatter: EpCombineScatter | None = None,
    # Emit MXFP4 from the epilogue instead of bf16: returns
    # (packed e2m1 [M, N_out//2] uint8, e8m0 scales [M, N_out//32] uint8), the
    # exact pair `mxfp4_quant` would have produced, so GEMM2 consumes it
    # directly and that separate launch over the intermediate disappears.
    # GEMM1-style only -- requires split_k == 1 and no scatter_indx.
    out_mx_quant: bool = False,
):
    """
    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])
    """
    if backend is None:
        backend = "gluon" if get_arch() == "gfx1250" else "triton"
    assert backend in ("triton", "gluon"), f"Invalid backend: {backend}"
    if backend == "gluon":
        assert (
            get_arch() == "gfx1250"
        ), f"Gluon backend requires gfx1250, got {get_arch()}"
    use_gluon = backend == "gluon"
    if preshuffle_weights:
        assert (
            use_gluon
        ), "preshuffled weights are only supported by the gluon (gfx1250) kernel"

    assert w.stride(-2) == 1, "`w` must be column-major when it has data-type mxfp"
    assert x.stride(-1) == 1, "'x' must be row-major when it has data-type mxfp"

    # determine shapes
    num_tokens = x.shape[-2]
    M = x.shape[-2] if gather_indx is None else gather_indx.shape[0]
    K, N = x.shape[-1] * 2, w.shape[-1]
    if preshuffle_weights:
        N = N * 16

    if out_mx_quant:
        assert use_gluon, (
            "out_mx_quant is implemented in the gfx1250 gluon epilogue only; the "
            "triton fallback would silently write bf16"
        )
        assert scatter_indx is None, (
            "out_mx_quant is GEMM1-style (no scatter): scatter+combine would need "
            "an mxfp4-aware reduce_grouped"
        )
    if ep_scatter is not None:
        assert scatter_indx is not None, (
            "ep_scatter needs the scatter indices' row order: dst_row is indexed "
            "by sorted row, which only exists once the gates are sorted"
        )
        assert not out_mx_quant, "ep_scatter delivers bf16 rows, not MXFP4"
        assert (
            not apply_swiglu
        ), "ep_scatter is a GEMM2-side delivery; the activation belongs to GEMM1"

    block_m = routing_data.block_m
    if unpadded_N and block_m == 16:
        N = unpadded_N
    if unpadded_K and block_m == 16:
        K = unpadded_K

    # compute optimization flags
    if use_gluon:
        config = get_kernel_config_gluon(M, N, K, routing_data)
        split_k = 1
    else:
        config = get_kernel_config_triton(M, N, K, routing_data)
        split_k = config["split_k"]

    x_scales_tdm = False
    if use_gluon:
        mx_scale_block_k = config["block_k"] // MXFP4_QUANT_BLOCK_SIZE
        ASYNC_COPY_MIN_SCALE_WIDTH = 4
        x_scales_tdm = (
            mx_scale_block_k < ASYNC_COPY_MIN_SCALE_WIDTH
            or K % config["block_k"] != 0
            or x_scales.stride(0) % 16 != 0
        )

    if apply_swiglu and split_k > 1:
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

    if out_mx_quant:
        assert split_k == 1, "out_mx_quant requires split_k == 1"

    # Fold the EP scatter into the GEMM epilogue when the kernel we are about to
    # launch has one. Both gluon kernels do; the triton fallback has no gfx1250
    # epilogue and drops through to the standalone `scatter_grouped`, which
    # writes the same bytes.
    fused_ep_scatter = (
        ep_scatter is not None
        and ep_scatter.fused
        and use_gluon
        # split-k partials must be summed before a row can be delivered, and the
        # epilogue sees only its own partial.
        and split_k == 1
    )

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
        split_k,
        preshuffle_weights,
        y_out=y_out,
        skip_final=ep_scatter is not None,
        # The epilogue writes straight into the staging window, so the
        # (M x hidden) matmul buffer is never read. Skip allocating it.
        skip_matmul=fused_ep_scatter,
        out_mx_quant=out_mx_quant,
    )
    if fused_ep_scatter:
        # `Y` and its strides now name the staging window; the kernel indexes it
        # by dst_row instead of by sorted row, so no `start_m` bias applies.
        y_ptr = ep_scatter.out
        stride_y_m = ep_scatter.out.stride(0)
        stride_y_n = ep_scatter.out.stride(1)
        dst_row = ep_scatter.dst_row
    else:
        y_ptr = y
        stride_y_m = y.stride(1)
        stride_y_n = y.stride(2)
        dst_row = None

    # Companion e8m0 scale buffer for the MXFP4 emit path. Plain row-major, the
    # layout the consuming GEMM reads activation scales in (SWIZZLE_MX_SCALE
    # applies to w_scales only), and the same shape `mxfp4_quant` returns.
    if out_mx_quant:
        n_out = N // reduction_n_matmul  # post-swiglu width, pre-pack
        assert n_out % 32 == 0, "out_mx_quant requires N_out % 32 == 0"
        y_scale = torch.empty(
            (y.shape[-2], n_out // 32), dtype=torch.uint8, device=x.device
        )
        stride_y_mx_m = y_scale.stride(0)
        stride_y_mx_n = y_scale.stride(1)
    else:
        y_scale = None
        stride_y_mx_m = 0
        stride_y_mx_n = 0

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
    grid = grid_m * grid_n * split_k

    # launch kernel
    if use_gluon and block_m == 16:
        layouts = get_moe_a4w4_layouts_decode(
            BLOCK_M=config["block_m"],
            BLOCK_N=config["block_n"],
            BLOCK_K=config["block_k"],
            num_warps=config["num_warps"],
            ACTIVATION_REDUCTION_N=reduction_n_matmul,
            PRESHUFFLE_WEIGHTS=preshuffle_weights,
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            GatherIndx=gather_indx,
            X_SCALES_TDM=x_scales_tdm,
        )
        # launch gluon kernel
        _moe_gemm_a4w4_decode[(grid,)](
            y_ptr,
            stride_y_m,
            stride_y_n,
            x,
            x.stride(0),
            x.stride(1),
            x_scales,
            x_scales.stride(0),
            x_scales.stride(1),
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
            num_tokens,
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
            XCD_SWIZZLE=config["xcd_swizzle"],
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            PRESHUFFLE_WEIGHTS=preshuffle_weights,
            NUM_BUFFERS=config["num_buffers"],
            UPCAST_INDICES=should_upcast_indices(x, w, y_ptr),
            X_SCALES_TDM=x_scales_tdm,
            CLAMP_BOUNDS=K % config["block_k"] != 0,
            **layouts,
            YMxScale=y_scale,
            stride_y_mx_m=stride_y_mx_m,
            stride_y_mx_n=stride_y_mx_n,
            HAS_MX_OUT=out_mx_quant,
            DstRow=dst_row,
            EP_SCATTER=fused_ep_scatter,
            Y_ROWS=(ep_scatter.out.shape[0] if fused_ep_scatter else 0),
            num_warps=config["num_warps"],
        )
    elif use_gluon:
        # layouts
        layouts = get_moe_a4w4_layouts_prefill(
            BLOCK_M=config["block_m"],
            BLOCK_N=config["block_n"],
            BLOCK_K=config["block_k"],
            num_warps=config["num_warps"],
            num_ctas=config["num_ctas"],
            ACTIVATION_REDUCTION_N=reduction_n_matmul,
            PRESHUFFLE_WEIGHTS=preshuffle_weights,
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            GatherIndx=gather_indx,
            X_SCALES_TDM=x_scales_tdm,
        )
        clamp_bounds = (K % config["block_k"] != 0) or (
            triton.cdiv(K, config["block_k"]) < config["num_buffers"]
        )
        # launch gluon kernel
        _moe_gemm_a4w4_prefill[(grid,)](
            y_ptr,
            stride_y_m,
            stride_y_n,
            x,
            x.stride(0),
            x.stride(1),
            x_scales,
            x_scales.stride(0),
            x_scales.stride(1),
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
            num_tokens,
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
            XCD_SWIZZLE=config["xcd_swizzle"],
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            PRESHUFFLE_WEIGHTS=preshuffle_weights,
            NUM_BUFFERS=config["num_buffers"],
            UPCAST_INDICES=should_upcast_indices(x, w, y_ptr),
            X_SCALES_TDM=x_scales_tdm,
            CLAMP_BOUNDS=clamp_bounds,
            **layouts,
            YMxScale=y_scale,
            stride_y_mx_m=stride_y_mx_m,
            stride_y_mx_n=stride_y_mx_n,
            HAS_MX_OUT=out_mx_quant,
            DstRow=dst_row,
            EP_SCATTER=fused_ep_scatter,
            Y_ROWS=(ep_scatter.out.shape[0] if fused_ep_scatter else 0),
            num_ctas=config["num_ctas"],
            num_warps=config["num_warps"],
        )
    else:
        # launch triton kernel
        _moe_gemm_a4w4[(grid,)](
            y,
            y.stride(0),
            y.stride(1),
            y.stride(2),
            x,
            x.stride(0),
            x.stride(1),
            x_scales,
            x_scales.stride(0),
            x_scales.stride(1),
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
            SPLIT_K=split_k,
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
    group_valid = (
        None
        if (gate_valid is None or scatter_indx is None)
        else gate_valid.view(-1, routing_data.n_expts_act)
    )
    # Expert-parallel combine: hand the rows to the staging window instead of
    # reducing them. Returns the window view, which is NOT a per-token output --
    # the caller's combine produces that once every rank has delivered.
    # MXFP4 emit path: scatter_indx is None and split_k == 1, so we bypass
    # reduce_grouped and return (packed e2m1, e8m0 scales) directly -- the same
    # pair, in the same layout, that `mxfp4_quant` returns.
    if out_mx_quant:
        return y.squeeze(0), y_scale
    if ep_scatter is not None:
        if fused_ep_scatter:
            # The epilogue already placed every row in the window.
            return ep_scatter.out
        return scatter_grouped(y, ep_scatter.dst_row, ep_scatter.out)
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
        indx_valid=group_valid,
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
