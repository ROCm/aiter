# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs.py

import itertools

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a8w4 import (
    _moe_gemm_a8w4_decode as _moe_gemm_a8w4_decode_gluon,
)
from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a8w4 import (
    _moe_gemm_a8w4_decode_persistent as _moe_gemm_a8w4_decode_persistent_gluon,
)
from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a8w4 import (
    _moe_gemm_a8w4_prefill as _moe_gemm_a8w4_prefill_gluon,
)
from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a8w4 import (
    get_moe_a8w4_layouts,
)
from aiter.ops.triton._triton_kernels.moe.moe_op_gemm_a8w4 import (
    _moe_gemm_a8w4 as _moe_gemm_a8w4_triton,
)
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
from aiter.ops.triton.moe.reduce import (
    EpCombineScatter,
    reduce_grouped,
    scatter_grouped,
    validate_reduce_out,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.gemm_config_utils import pick_gemm_num_stages
from aiter.ops.triton.utils.moe_config_utils import get_moe_dispatch


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
    y_out=None,
    skip_final=False,
    skip_matmul=False,
):
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
    if skip_matmul:
        # The epilogue scatters straight into a caller-owned window, so nothing
        # ever reads this buffer -- and at (M x hidden) bf16 it is tens of MB per
        # layer, allocated and dirtied for nothing.
        matmul_output = None
    else:
        matmul_output = torch.empty(matmul_shape, device=device, dtype=out_dtype)
    if skip_final:
        # The rows are delivered elsewhere (expert-parallel scatter), so a
        # reduced output would only be allocated to be thrown away -- and at
        # (tokens x hidden) bf16 that is tens of MB per layer.
        assert y_out is None, "y_out names a reduced output; skip_final has none"
        final_output = None
    elif scatter_indx is not None or split_k > 1:
        final_output = validate_reduce_out(y_out, final_shape, out_dtype, device)
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


def get_kernel_config_triton(m, n, k, routing_data, swizzle_mx_scale=None):
    block_m = routing_data.block_m
    group_m = 4
    num_xcds = 8
    xcd_swizzle = num_xcds
    w_cache_modifier = ".cg" if block_m <= 32 else None
    arch = get_arch()
    split_k = 1

    # Tuned dispatch: per-(block_m, N, K) winners from a sweep tuner.
    # Entries carry BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages, … but
    # omit BLOCK_SIZE_M because block_m is the dispatch key, not a tunable
    # (routing decides block_m for the layer).
    tuned = get_moe_dispatch("A8W4", arch, "triton").get(f"bm{block_m}_n{n}_k{k}")
    if tuned is not None:
        return {
            "block_m": block_m,
            "block_n": tuned["BLOCK_SIZE_N"],
            "block_k": tuned["BLOCK_SIZE_K"],
            "num_warps": tuned["num_warps"],
            "num_stages": tuned["num_stages"],
            "group_m": group_m,
            "xcd_swizzle": xcd_swizzle,
            "w_cache_modifier": w_cache_modifier,
            "split_k": split_k,
            "waves_per_eu": tuned.get("waves_per_eu", 0),
            "matrix_instr_nonkdim": tuned.get("matrix_instr_nonkdim", 16),
            "kpack": tuned.get("kpack", 1),
        }

    # Fallback for shapes not in the tuned dispatch JSON.
    # Look for a tuned entry with the same (N, K) but any block_m — the tile
    # geometry and num_stages from that entry are a better starting point than
    # a generic default, and avoid regressing to num_stages=1 on gfx950.
    # Under CDNA4 swizzle, skip BLOCK_K<256 entries since unswizzle can't compile them.
    dispatch = get_moe_dispatch("A8W4", arch, "triton")
    proxy = next(
        (
            v
            for bm in (16, 32, 64, 128)
            if (v := dispatch.get(f"bm{bm}_n{n}_k{k}")) is not None
            and (swizzle_mx_scale != "CDNA4_SCALE" or v.get("BLOCK_SIZE_K", 0) >= 256)
        ),
        None,
    )
    if proxy is not None:
        return {
            "block_m": block_m,
            "block_n": proxy["BLOCK_SIZE_N"],
            "block_k": proxy["BLOCK_SIZE_K"],
            "num_warps": proxy["num_warps"],
            "num_stages": proxy["num_stages"],
            "group_m": group_m,
            "xcd_swizzle": xcd_swizzle,
            "w_cache_modifier": w_cache_modifier,
            "split_k": split_k,
            "waves_per_eu": proxy.get("waves_per_eu", 0),
            "matrix_instr_nonkdim": proxy.get("matrix_instr_nonkdim", 16),
            "kpack": proxy.get("kpack", 1),
        }

    # Last-resort: original shape-based heuristic, gated to gfx942 which has no
    # tuned JSON. Other arches fall back to a conservative safe default.
    block_k = 256
    num_stages = 2

    if arch == "gfx942":
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
            else:
                block_n = 256
                num_warps = 8
        else:
            block_n = 128
            num_warps = 4 if block_m == 128 else 8
    elif arch == "gfx950":
        if block_m == 16:
            block_n = 128
            num_warps = 4

            grid_m = routing_data.n_blocks(m, block_m)
            grid_n = triton.cdiv(n, block_n)
            grid = grid_m * grid_n * split_k
            # Floor at 64 (was 32): out_mx_quant=True with apply_swiglu requires
            # OUT_BLOCK_N = BLOCK_N // 2 >= 32. Loop boundary changed to keep
            # block_n >= 64 for both MX and non-MX paths.
            while block_n >= 128 and grid < get_num_sms():
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
                num_warps = 4
            else:
                block_n = 512
                num_warps = 4

        elif block_m == 64:
            # V4-Flash prefill-tuned (rocprof brute force v2): for block_m=64,
            # (bn=128, nw=4, ns=1) gives 2-4x speedup over the previous bn=512/nw=8
            # default on all four V4-Flash prefill shapes.
            block_n = 128
            num_warps = 4
            num_stages = 1

        else:
            # Cap by N: BN=512 wasted compute on small-N shapes (e.g. N=256
            # → 50% pad, grid_n=1). Tuned shapes bypass this via JSON.
            block_n = min(triton.next_power_of_2(n), 256)
            # routing caps block_m at 128; nw=4 wins ~2x at block_m=128 on gpt-oss
            # shapes (MI355X) but regresses ~7% at block_m=64, so 64 stays at 8.
            num_warps = 4 if block_m == 128 else 8

        # bits_a=8 (fp8), bits_b=4 (mxfp4). Picks ns=2 when the tile fits in LDS,
        # else falls back to ns=1. The previous hardcoded ns=1 silently regressed
        # gpt-oss W4A8 MoE shapes by 30-40% vs the JSON-tuned ns=2 winners; the
        # block_m==64 branch keeps its rocprof-tuned ns=1 override.
        if block_m != 64:
            num_stages = pick_gemm_num_stages(
                arch, block_m, block_n, block_k, 8, 4, use_async_padding=True
            )
    else:
        block_n = 128
        num_warps = 4

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


def get_gluon_a8w4_ctas_per_cga(m):
    num_ctas = 1
    if num_ctas == 1 or get_arch() != "gfx1250":
        return [1, 1]
    # Decode: shard the cluster along N only.
    if m < 1024:
        return [1, 1]
    # Prefill: shard along both M and N.
    if num_ctas == 4:
        return [2, 2]
    if num_ctas == 8:
        return [2, 4]
    if num_ctas == 16:
        return [4, 4]
    return [1, num_ctas]


def get_kernel_config_gluon(m, n, k, routing_data, out_mx_quant=False):
    ctas_per_cga = get_gluon_a8w4_ctas_per_cga(m)
    num_ctas = ctas_per_cga[0] * ctas_per_cga[1]
    block_m = routing_data.block_m
    num_xcds = 1
    w_cache_modifier = ".cg" if block_m <= 32 else None
    split_k = 1

    bucket = m2bucket(m)
    tuned = get_moe_dispatch("A8W4", get_arch(), "gluon")
    key = f"bm{block_m}_n{n}_k{k}_{bucket}"
    if key not in tuned:
        key = f"bm{block_m}_any"
    cfg = tuned[key]
    block_n, block_k, num_buffers, num_warps, persistent_iters = (
        cfg["block_n"],
        cfg["block_k"],
        cfg["num_buffers"],
        cfg["num_warps"],
        cfg["persistent_iters"],
    )

    num_buffers = min(num_buffers, triton.cdiv(k, block_k))
    block_m *= ctas_per_cga[0]
    block_n *= ctas_per_cga[1]

    ret = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_buffers": num_buffers,
        "xcd_swizzle": num_xcds,
        "split_k": split_k,
        "w_cache_modifier": w_cache_modifier,
        "waves_per_eu": 0,
        "persistent_iters": persistent_iters,
        "num_ctas": num_ctas,
        "ctas_per_cga": ctas_per_cga,
    }
    return ret


# -----------------------------------------------------------------------------
# Triton Implementation
# -----------------------------------------------------------------------------


def moe_gemm_a8w4(
    x,
    w,
    x_scales,
    w_scales,
    x_static_scale=None,
    quant_static_scale=None,
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
    preshuffled=False,
    unpadded_N=None,
    unpadded_K=None,
    # Idea 1: emit (fp8 e4m3, ue8m0 per-1×32 scale) directly from the GEMM
    # write-back. When out_mx_quant=True, returns (y_fp8, y_scale_ue8m0).
    # Requires SPLIT_K==1 and no scatter_indx (GEMM1-style).
    out_mx_quant: bool = False,
    # External residual to fold into reduce_grouped writeback (saves the
    # standalone routed+shared elementwise add).
    residual=None,
    backend=None,
    # Per-gate validity, same layout as scatter_indx. Default None == every gate
    # slot is live, which holds whenever routing() produced the indices. Pass a
    # mask when only some of a token's n_expts_act slots are computed here --
    # expert parallelism, where the other slots belong to another rank and are
    # never written, so the reduce must not sum them.
    gate_valid=None,
    # Destination for the grouped reduction's result, instead of a freshly
    # allocated buffer. May be a slice of a taller tensor -- the reduce writes
    # through `out`'s strides -- which lets a caller whose consumer wants more
    # rows than this GEMM produces skip a full-width copy. Requires a grouped
    # reduction to actually run, i.e. scatter_indx is not None or split_k > 1.
    y_out=None,
    # Expert-parallel combine: deliver the un-reduced rows to a combine staging
    # window instead of reducing them here (see EpCombineScatter). The peers'
    # rows for a token are missing at this point, so there is nothing to reduce;
    # whoever owns the staging window sums them once every rank has delivered.
    # Mutually exclusive with y_out, which names a reduced output.
    ep_scatter: EpCombineScatter | None = None,
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
    if preshuffled:
        assert (
            use_gluon
        ), "preshuffled weights are only supported by the gluon (gfx1250) kernel"
    assert w.stride(-2) == 1, "`w` must be column-major when it has data-type mxfp"
    x_has_mx = x_scales is not None
    if x_has_mx:
        assert x.stride(-1) == 1, "'x' must be row-major when it has data-type mxfp"
    if x_has_mx:
        stride_x_mx_m = x_scales.stride(0)
        stride_x_mx_k = x_scales.stride(1)
    else:
        stride_x_mx_m = 0
        stride_x_mx_k = 0
    # determine shapes
    num_tokens = x.shape[-2]
    M = num_tokens if gather_indx is None else gather_indx.shape[0]
    K, N = x.shape[-1], w.shape[-1]
    if preshuffled:
        # preshuffle layout is (E, K_packed*16, N//16); w.shape[-1] = N//16
        N = w.shape[-1] * 16
    # Output buffer must be sized to the PADDED N: the kernel writes full
    # block_n columns per tile (grid_n * block_n cols total), which can exceed
    # unpadded_N when block_n doesn't divide it evenly → OOB on the y buffer.
    padded_N = N
    block_m = routing_data.block_m
    if unpadded_N and block_m == 16:
        N = unpadded_N
    if unpadded_K and block_m == 16:
        K = unpadded_K
    if use_gluon:
        w = w.transpose(1, 2)
        w_scales = w_scales.transpose(1, 2)
    # compute optimization flags
    if use_gluon:
        config = get_kernel_config_gluon(M, N, K, routing_data, out_mx_quant)
    else:
        config = get_kernel_config_triton(M, N, K, routing_data, swizzle_mx_scale)
    # CDNA4 swizzle requires BLOCK_K % 256 == 0; some tuned small-K entries
    # pick BK<256 for utilization. Clamp only when swizzle is requested so
    # StridedLayout callers keep their tuned BK<256.
    if swizzle_mx_scale == "CDNA4_SCALE" and config["block_k"] < 256:
        config["block_k"] = 256
    # Fall back to TDM if the scale width is below the gfx1250 direct-to-LDS
    # floor (32 bits = 4 uint8 scales) or K is uneven
    X_SCALE_TDM = False
    if use_gluon and x_has_mx:
        mx_scale_block_k = config["block_k"] // 32
        ASYNC_COPY_MIN_SCALE_WIDTH = 8
        X_SCALE_TDM = (
            mx_scale_block_k < ASYNC_COPY_MIN_SCALE_WIDTH or K % config["block_k"] != 0
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
    # allocate output memory. With out_mx_quant=True, the kernel writes fp8 e4m3
    # into y; otherwise the requested out_dtype (bf16).
    if out_mx_quant:
        assert config["split_k"] == 1, "out_mx_quant requires split_k == 1"
        assert scatter_indx is None, (
            "out_mx_quant currently only supported for GEMM1-style (no scatter); "
            "scatter+combine would need fp8-aware reduce_grouped"
        )
        out_dtype = torch.float8_e4m3fn
    else:
        out_dtype = out_dtype  # noqa: PLW0127
    if ep_scatter is not None:
        assert scatter_indx is not None, (
            "ep_scatter needs the scatter indices' row order: dst_row is indexed "
            "by sorted row, which only exists once the gates are sorted"
        )
        assert not out_mx_quant, "ep_scatter delivers bf16 rows, not MXFP8"
        assert (
            not apply_swiglu
        ), "ep_scatter is a GEMM2-side delivery; the activation belongs to GEMM1"
        # `residual` is folded in by reduce_grouped, which does not run here --
        # so accepting both would drop the residual silently. It cannot simply
        # move into the scatter either: the residual is per TOKEN, and the rows
        # leaving here are per (token, expert), so adding it to each would count
        # it once per expert. It belongs after the combine.
        assert residual is None, (
            "ep_scatter cannot apply `residual`: it is per-token, but this path "
            "emits per-(token, expert) rows and never reduces them. Fold the "
            "residual into the combine's output instead."
        )
    # Fold the EP scatter into the GEMM epilogue when the kernel we are about to
    # launch has one. Only the two non-persistent gluon kernels do: the persistent
    # decode kernel writes back inside its N-tile loop through a rolling
    # descriptor, and the triton kernel has no gfx1250 epilogue at all. Both fall
    # through to the standalone `_scatter_grouped`, which writes the same bytes.
    fused_ep_scatter = (
        ep_scatter is not None
        and ep_scatter.fused
        and use_gluon
        and config["persistent_iters"] <= 1
        # split-k partials must be summed before a row can be delivered, and the
        # epilogue sees only its own partial.
        and config["split_k"] == 1
    )
    y, y_final = allocate_output(
        M,
        padded_N,
        out_dtype,
        reduction_n_matmul,
        reduction_n_reduction,
        routing_data,
        gather_indx,
        scatter_indx,
        config["block_m"],
        config["split_k"],
        x.device,
        y_out=y_out,
        skip_final=ep_scatter is not None,
        # The epilogue writes straight into the staging window, so the
        # (M x hidden) matmul buffer is never read. Skip allocating it.
        skip_matmul=fused_ep_scatter,
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
    # Companion ue8m0 scale buffer for the MXFP8 emit path.
    if out_mx_quant:
        n_out = padded_N // reduction_n_matmul  # post-swiglu width
        assert n_out % 32 == 0, "out_mx_quant requires N_out % 32 == 0"
        m_out = y.shape[-2]
        y_scale = torch.empty((m_out, n_out // 32), dtype=torch.uint8, device=x.device)
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
    # pid grid
    grid_m = routing_data.n_blocks(M, config["block_m"])
    grid_n = triton.cdiv(N, config["block_n"])
    if use_gluon and config["persistent_iters"] > 1:
        num_blocks_n = grid_n
        grid_n = triton.cdiv(num_blocks_n, config["persistent_iters"])
    grid = grid_m * grid_n * config["split_k"]
    # launch kernel
    if use_gluon and config["persistent_iters"] > 1:
        _moe_gemm_a8w4_decode_persistent_gluon[(grid,)](
            y,
            y.stride(1),
            y.stride(2),
            x,
            x.stride(0),
            x.stride(1),
            x_scales,
            stride_x_mx_m,
            stride_x_mx_k,
            w,
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w_scales,
            w_scales.stride(0),
            w_scales.stride(1),
            w_scales.stride(2),
            x_static_scale,
            quant_static_scale,
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
            num_blocks_n,
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
            NUM_BUFFERS=config["num_buffers"],
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            X_SCALE_TDM=X_SCALE_TDM,
            PRESHUFFLED=preshuffled,
            CLAMP_BOUNDS=K % config["block_k"] != 0,
            N_ITERS=config["persistent_iters"],
            num_warps=config["num_warps"],
            UPCAST_INDICES=should_upcast_indices(x, w, y_ptr),
            waves_per_eu=config["waves_per_eu"],
            YMxScale=y_scale,
            stride_y_mx_m=stride_y_mx_m,
            stride_y_mx_n=stride_y_mx_n,
            HAS_MX_OUT=out_mx_quant,
        )
    elif use_gluon and block_m == 16:
        _moe_gemm_a8w4_decode_gluon[(grid,)](
            y_ptr,
            stride_y_m,
            stride_y_n,
            x,
            x.stride(0),
            x.stride(1),
            x_scales,
            stride_x_mx_m,
            stride_x_mx_k,
            w,
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w_scales,
            w_scales.stride(0),
            w_scales.stride(1),
            w_scales.stride(2),
            x_static_scale,
            quant_static_scale,
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
            NUM_BUFFERS=config["num_buffers"],
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            X_SCALE_TDM=X_SCALE_TDM,
            PRESHUFFLED=preshuffled,
            CLAMP_BOUNDS=K % config["block_k"] != 0,
            num_warps=config["num_warps"],
            UPCAST_INDICES=should_upcast_indices(x, w, y_ptr),
            waves_per_eu=config["waves_per_eu"],
            YMxScale=y_scale,
            stride_y_mx_m=stride_y_mx_m,
            stride_y_mx_n=stride_y_mx_n,
            HAS_MX_OUT=out_mx_quant,
            DstRow=dst_row,
            EP_SCATTER=fused_ep_scatter,
            Y_ROWS=(ep_scatter.out.shape[0] if fused_ep_scatter else 0),
        )
    elif use_gluon:
        layouts = get_moe_a8w4_layouts(
            num_warps=config["num_warps"],
            BLOCK_M=config["block_m"],
            BLOCK_N=config["block_n"],
            BLOCK_K=config["block_k"],
            ctas_per_cga=config["ctas_per_cga"],
            ACTIVATION_REDUCTION_N=reduction_n_matmul,
            PRESHUFFLED=preshuffled,
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            is_x_microscaled=x_scales is not None,
            has_quant_static_scale=quant_static_scale is not None,
            apply_swiglu=apply_swiglu_matmul,
            GatherIndx=gather_indx,
            X_SCALE_TDM=X_SCALE_TDM,
            out_mx_quant=out_mx_quant,
            is_prefill=M >= 1024,
        )
        _moe_gemm_a8w4_prefill_gluon[(grid,)](
            y_ptr,
            stride_y_m,
            stride_y_n,
            x,
            x.stride(0),
            x.stride(1),
            x_scales,
            stride_x_mx_m,
            stride_x_mx_k,
            w,
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w_scales,
            w_scales.stride(0),
            w_scales.stride(1),
            w_scales.stride(2),
            x_static_scale,
            quant_static_scale,
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
            NUM_BUFFERS=config["num_buffers"],
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            PRESHUFFLED=preshuffled,
            X_SCALE_TDM=X_SCALE_TDM,
            CLAMP_BOUNDS=K % config["block_k"] != 0,
            num_warps=config["num_warps"],
            num_ctas=config["num_ctas"],
            UPCAST_INDICES=should_upcast_indices(x, w, y_ptr),
            waves_per_eu=config["waves_per_eu"],
            YMxScale=y_scale,
            stride_y_mx_m=stride_y_mx_m,
            stride_y_mx_n=stride_y_mx_n,
            HAS_MX_OUT=out_mx_quant,
            DstRow=dst_row,
            EP_SCATTER=fused_ep_scatter,
            Y_ROWS=(ep_scatter.out.shape[0] if fused_ep_scatter else 0),
            **layouts,
        )
    else:
        _moe_gemm_a8w4_triton[(grid,)](
            y,
            y.stride(0),
            y.stride(1),
            y.stride(2),
            x,
            x.stride(0),
            x.stride(1),
            x_scales,
            stride_x_mx_m,
            stride_x_mx_k,
            w,
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w_scales,
            w_scales.stride(0),
            w_scales.stride(1),
            w_scales.stride(2),
            x_static_scale,
            quant_static_scale,
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
            UPCAST_INDICES=should_upcast_indices(x, w, y_ptr),
            waves_per_eu=config["waves_per_eu"],
            matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
            kpack=config["kpack"],
            YMxScale=y_scale,
            stride_y_mx_m=stride_y_mx_m,
            stride_y_mx_n=stride_y_mx_n,
            HAS_MX_OUT=out_mx_quant,
        )

    # MXFP8 emit path: scatter_indx is None and split_k==1, so we bypass
    # reduce_grouped and return (fp8 values, ue8m0 scales) directly.
    if out_mx_quant:
        return y.squeeze(0), y_scale
    # Expert-parallel combine: hand the rows to the staging window instead of
    # reducing them. Returns the window view, which is not a per-token output --
    # the caller's combine produces that once every rank has delivered.
    if ep_scatter is not None:
        if fused_ep_scatter:
            # The epilogue already placed every row in the window.
            return ep_scatter.out
        return scatter_grouped(y, ep_scatter.dst_row, ep_scatter.out)
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
    # Step 9: external residual fold-in is now wired into reduce_grouped.
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
        residual=residual,
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
