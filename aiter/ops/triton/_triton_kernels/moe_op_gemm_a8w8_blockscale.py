# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py

import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid
from aiter.ops.triton._triton_kernels.quant_moe import _compute_static_fp8_quant


def matmul_launch_metadata(grid, kernel, args):
    ret = dict()
    M, N, K = None, args["N"], args["K"]
    Y, X, W = args["Y"], args["X"], args["W"]
    hist = args["ExptHist"]
    if hist is not None:
        n_rows = int(hist.float().mean())
        n_tokens = float(hist.sum())
        n_w_bytes = (W.numel() * W.element_size() // hist.numel()) * (hist > 0).sum()
    else:
        n_tokens = None
        n_w_bytes = W.numel() * W.element_size()
    repr = lambda s, x: f"{s}={x}" if x is not None else f"E_{len(hist)}({s})={n_rows}"
    nbits = X.dtype.itemsize * 8
    ret["name"] = f"{kernel.name} [{repr('M', M)}, {repr('N', N)}, {repr('K', K)}]"
    if args["B"] is not None:
        ret["name"] += "_bias"
    if args["APPLY_SWIGLU"]:
        ret["name"] += "_swiglu"
    if args["Quant_static_scale"] is not None:
        ret["name"] += "_quant"

    fM = n_tokens
    fK = K if K is not None else n_tokens
    ret[f"flops{nbits}"] = 2.0 * fM * N * fK

    gindx = args.get("GatherIndx", None)
    # sindx = args.get("WriteBackIndx", None)
    n_x_bytes = X.numel() * X.element_size()
    n_y_bytes = Y.numel() * Y.element_size()
    if hist is not None:
        assert n_tokens is not None
        n_expts_act = args["N_EXPTS_ACT"]

        if gindx is not None:
            # recreate inverse GatherIndx.
            dst = torch.full_like(gindx, -1)
            idx = torch.arange(len(gindx), device=gindx.device, dtype=torch.int32)
            mask = gindx != -1
            dst[gindx[mask]] = idx[mask]
            n_read_rows = (dst.view((-1, n_expts_act)) != -1).any(dim=1).sum()
        else:
            n_read_rows = n_tokens
        n_x_bytes = n_read_rows * X.shape[-1] * X.element_size()
        n_y_bytes = n_tokens * Y.shape[-1] * Y.element_size()
    ret["bytes"] = int(n_x_bytes + n_y_bytes + n_w_bytes)

    return ret


@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def _swiglu(input, alpha, limit):
    gelu, linear = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    gelu = gelu.to(tl.float32)
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + tl.exp2(-1.44269504089 * alpha * gelu))
    return tl.fma(s, linear, s)  # (s * (linear + 1))


@triton.jit
def _reduce_grouped(
    X,
    stride_xb: tl.uint64,
    stride_xm: tl.uint64,
    stride_xn,  #
    Out,
    stride_om: tl.uint64,
    stride_on,  # output tensor
    InIndx,
    B,
    N,  #
    # fused activation function
    APPLY_SWIGLU: tl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):

    pid_t = tl.program_id(1)
    pid_n = tl.program_id(0)

    BLOCK_N_OUT: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    start = pid_t * K
    # load indices into a tuple
    if InIndx is None:
        indxs = (pid_t,)
    else:
        indxs = ()
        for i in tl.static_range(0, K):
            indxs = indxs + (tl.load(InIndx + start + i),)
    XPtrs = X + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) * stride_xn
    OutPtrs = Out + (pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT)) * stride_on

    acc = tl.zeros([BLOCK_N_OUT], dtype=tl.float32)
    x_n_mask = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) < N
    # accumulate contributions for this tile
    for i in tl.static_range(0, K):
        curr = tl.zeros([BLOCK_N], dtype=tl.float32)
        # iterate over split_k partial values
        for b in tl.range(0, B):
            x_row_ptr = XPtrs + indxs[i] * stride_xm + b * stride_xb
            if EVEN_N:
                vals = tl.load(x_row_ptr)
            else:
                vals = tl.load(x_row_ptr, mask=x_n_mask, other=0.0)
            vals = vals.to(tl.float32)
            curr += vals

        # apply nonlinearity to split-k output
        if APPLY_SWIGLU:
            curr = _swiglu(curr[None, :], alpha, limit)
        curr = tl.reshape(curr, [curr.shape[-1]])
        # update final accumulator
        acc += curr
    # Compute per-32-col MXFP scales for this tile if requested
    Nrem = N // ACTIVATION_REDUCTION_N

    # write-back for this tile
    out_ptr = OutPtrs + pid_t * stride_om
    if EVEN_N:
        tl.store(out_ptr, acc)
    else:
        out_n_mask = pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT) < Nrem
        tl.store(out_ptr, acc, mask=out_n_mask)


@triton.jit(launch_metadata=matmul_launch_metadata)
def _moe_gemm_a8w8_blockscale(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    XBlockScale,  # [M, K_blocks] or [M_blocks, K_blocks]
    stride_x_bs_m,
    stride_x_bs_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WBlockScale,  # [K_blocks, N_blocks]
    stride_w_bs_e,
    stride_w_bs_k,
    stride_w_bs_n,
    X_static_scale,
    W_static_scale,
    Quant_static_scale,
    B,
    stride_b_e,  # Bias
    Gammas,
    N,
    K,  # shapes
    # expt data
    GatherIndx,
    ExptHist,
    ExptOffs,
    ExptOffsSum,
    ExptData,
    # true grid size
    grid_m,
    grid_n,
    # fused activation function
    APPLY_SWIGLU: tl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: tl.constexpr,
    # MoE config
    N_EXPTS_ACT: tl.constexpr,
    # optimization config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_M: tl.constexpr,
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    MASK_K_LIMIT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    W_CACHE_MODIFIER: tl.constexpr,
    UPCAST_INDICES: tl.constexpr = False,
    # Use per-row or 2D blockscale on X
    PER_ROW_X_SCALE: tl.constexpr = False,
):
    """
    Computes the 8 bit matmul C = A x B using the block-scale quantization approach.

    Key parameters:
    - X: Matrix X with shape (M, K).
    - E: Matrix E with shape (E, K, N).
    - Y: Matrix C with shape (E, M, N).
    - x_scale: Scale tensor for A with shape (M // group_m, K // group_k) or (M, K // group_k)
    - w_scale: Scale tensor for B with shape (K // group_k, N // group_n)
    - PER_ROW_X_SCALE: Determines whether we use per-row or 2D blockscale on X

    For this kernel implementation, GROUP_K must equal BLOCK_K.
    """

    tl.assume(stride_y_k >= 0)
    tl.assume(stride_y_m >= 0)
    tl.assume(stride_y_n >= 0)
    tl.assume(stride_x_m >= 0)
    tl.assume(stride_x_k >= 0)
    tl.assume(stride_w_e >= 0)
    tl.assume(stride_w_k >= 0)
    tl.assume(stride_w_n >= 0)
    if stride_x_bs_m is not None:
        tl.assume(stride_x_bs_m >= 0)
    if stride_x_bs_k is not None:
        tl.assume(stride_x_bs_k >= 0)
    if stride_w_bs_e is not None:
        tl.assume(stride_w_bs_e >= 0)
    if stride_w_bs_k is not None:
        tl.assume(stride_w_bs_k >= 0)
    if stride_w_bs_n is not None:
        tl.assume(stride_w_bs_n >= 0)
    if B is not None:
        tl.assume(stride_b_e >= 0)
    tl.assume(grid_m >= 0)
    tl.assume(grid_n >= 0)
    tl.static_assert(GROUP_K == BLOCK_K, "This kernel assumes one K-block per tile")

    is_x_blockscale: tl.constexpr = XBlockScale is not None
    is_w_blockscale: tl.constexpr = WBlockScale is not None

    OUT_BLOCK_N: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = tl.program_id(0)
    padding_m: tl.constexpr = 0

    index_type: tl.constexpr = tl.int64 if UPCAST_INDICES else tl.int32

    unpadded_m = grid_m - padding_m
    tl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    pid_emnk = pid
    # pid_e = pid_emnk // (unpadded_m * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, GROUP_SIZE_M)
    # For split-k, advance to the output k slice
    if SPLIT_K > 1:
        Y += pid_k.to(index_type) * stride_y_k
    # unpack expert data
    expt_data = tl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = tl.load(ExptHist + expt_id)
    start_m = tl.load(ExptOffs + expt_id)
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m = start_m.to(index_type)
    pid_n, pid_k = pid_n.to(index_type), pid_k.to(index_type)

    # A pointers
    splitk_block_size = tl.cdiv(K, SPLIT_K)
    offs_k_scale = (pid_k * splitk_block_size) // GROUP_K
    offs_k = tl.arange(0, BLOCK_K)
    offs_k_split = pid_k * splitk_block_size + offs_k

    offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
    if GatherIndx is None:
        offs_x_m = start_m + offs_x_m
    else:
        GatherIndx += start_m
        # no needs to bounds-check here because `offs_x_m` wraps around M dim
        offs_x_m = tl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
    XPtrs = (
        X
        + offs_x_m.to(index_type)[:, None] * stride_x_m
        + offs_k_split.to(index_type)[None, :] * stride_x_k
    )

    # B pointers
    offs_w_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_w_n = tl.max_contiguous(
        tl.multiple_of(offs_w_n % N, BLOCK_N),
        BLOCK_N,
    )
    W += expt_id * stride_w_e
    WPtrs = W + (
        offs_k_split.to(index_type)[:, None] * stride_w_k
        + offs_w_n.to(index_type)[None, :] * stride_w_n
    )

    if is_x_blockscale:
        if PER_ROW_X_SCALE:
            # XScale: [M, K_blocks]
            XScalePtrs = (
                XBlockScale
                + offs_x_m.to(index_type) * stride_x_bs_m
                + offs_k_scale * stride_x_bs_k
            )
        else:
            # XScale: [M_blocks, K_blocks]
            offs_x_scale_m = offs_x_m // GROUP_M
            XScalePtrs = (
                XBlockScale
                + offs_x_scale_m.to(index_type) * stride_x_bs_m
                + offs_k_scale * stride_x_bs_k
            )

    if is_w_blockscale:
        WBlockScale += expt_id * stride_w_bs_e
        offs_w_scale_n = offs_w_n // GROUP_N
        # WBlockScale: [K_blocks, N_blocks]
        WScalePtrs = (
            WBlockScale + offs_k_scale * stride_w_bs_k + offs_w_scale_n * stride_w_bs_n
        )

    offs_ks_step = BLOCK_K // GROUP_K
    num_k_iter = tl.cdiv(splitk_block_size, BLOCK_K)
    # compute output
    x_scale = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    w_scale = tl.full((BLOCK_N,), 1.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
        if EVEN_K:
            x = tl.load(XPtrs)
            w = tl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)
        else:
            x = tl.load(XPtrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            w = tl.load(
                WPtrs,
                mask=offs_k[:, None] < K - k * BLOCK_K,
                other=0.0,
                cache_modifier=W_CACHE_MODIFIER,
            )

        if is_x_blockscale:
            x_scale = tl.load(XScalePtrs)
            XScalePtrs += offs_ks_step * stride_x_bs_k

        if is_w_blockscale:
            w_scale = tl.load(WScalePtrs)
            WScalePtrs += offs_ks_step * stride_w_bs_k

        scale_matrix = x_scale[:, None] * w_scale[None, :]
        acc += tl.dot(x, w, input_precision="ieee") * scale_matrix
        XPtrs += BLOCK_K * stride_x_k
        WPtrs += BLOCK_K * stride_w_k

    # scalar fp8 scale
    if X_static_scale is not None:
        acc = acc * tl.load(X_static_scale)
    if W_static_scale is not None:
        acc = acc * tl.load(W_static_scale)
    # bias
    offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_y_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_y_n < N
    if B is not None:
        BPtrs = B + expt_id * stride_b_e + offs_y_n
        if pid_k == 0:
            bias = tl.load(BPtrs, mask=mask_n, other=0, cache_modifier=W_CACHE_MODIFIER)
        else:
            bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
        acc = acc + bias[None, :]
    if APPLY_SWIGLU and SPLIT_K == 1:
        out = _swiglu(acc, alpha, limit)
        tl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
        offs_y_n = OUT_BLOCK_N * pid_n + tl.arange(0, OUT_BLOCK_N)
        mask_n = offs_y_n < yN
    else:
        tl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )
        out = acc
    if Gammas is not None:
        gammas = tl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        out *= gammas[:, None]
    # quant
    if Quant_static_scale is not None:
        out = _compute_static_fp8_quant(out, tl.load(Quant_static_scale))
    # write-back
    Y += start_m * stride_y_m
    offs_y_m = offs_m
    YPtrs = (
        Y
        + offs_y_m.to(index_type)[:, None] * stride_y_m
        + offs_y_n.to(index_type)[None, :] * stride_y_n
    )
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(YPtrs, out, mask=mask)
