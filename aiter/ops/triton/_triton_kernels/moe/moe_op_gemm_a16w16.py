# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py

import torch
import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import pid_grid
from aiter.ops.triton._triton_kernels.moe.activations import _swiglu
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


def matmul_launch_metadata(grid, kernel, args):
    ret = {}
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

    def repr(s, x):
        return f"{s}={x}" if x is not None else f"E_{len(hist)}({s})={n_rows}"

    nbits = X.dtype.itemsize * 8
    ret["name"] = f"{kernel.name} [{repr('M', M)}, {repr('N', N)}, {repr('K', K)}]"
    gindx = args.get("GatherIndx", None)
    if gindx is not None:
        gindx = gindx.to(torch.int32)
        ret["name"] += "_layer1"
    else:
        ret["name"] += "_layer2"
    if args["B"] is not None:
        ret["name"] += "_bias"
    if args["APPLY_SWIGLU"]:
        ret["name"] += "_swiglu"

    fM = n_tokens
    fK = K if K is not None else n_tokens
    ret[f"flops{nbits}"] = 2.0 * fM * N * fK

    n_x_bytes = X.numel() * X.element_size()
    n_y_bytes = Y.numel() * Y.element_size()
    if hist is not None:
        assert n_tokens is not None
        n_expts_act = args["N_EXPTS_ACT"]

        if gindx is not None:
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
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE
    new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
    return new_pid


_moe_gemm_a16w16_repr = make_kernel_repr(
    "_moe_gemm_a16w16",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_M",
        "SPLIT_K",
        "XCD_SWIZZLE",
        "EVEN_K",
        "W_CACHE_MODIFIER",
        "N_EXPTS_ACT",
        "APPLY_SWIGLU",
        "SWIGLU_ADD_RESIDUAL",
    ],
)


@triton.jit(repr=_moe_gemm_a16w16_repr, launch_metadata=matmul_launch_metadata)
def _moe_gemm_a16w16(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    B,
    stride_b_e,
    Gammas,
    N,
    K,
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
    SWIGLU_ADD_RESIDUAL: tl.constexpr,
    # MoE config
    N_EXPTS_ACT: tl.constexpr,
    # optimization config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    XCD_SWIZZLE: tl.constexpr,
    EVEN_K: tl.constexpr,
    MASK_K_LIMIT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    W_CACHE_MODIFIER: tl.constexpr,
    UPCAST_INDICES: tl.constexpr = False,
):
    tl.assume(stride_y_k >= 0)
    tl.assume(stride_y_m >= 0)
    tl.assume(stride_y_n >= 0)
    tl.assume(stride_x_m >= 0)
    tl.assume(stride_x_k >= 0)
    tl.assume(stride_w_e >= 0)
    tl.assume(stride_w_k >= 0)
    tl.assume(stride_w_n >= 0)
    if B is not None:
        tl.assume(stride_b_e >= 0)
    tl.assume(grid_m >= 0)
    tl.assume(grid_n >= 0)

    OUT_BLOCK_N: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = tl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        padding_m = grid_m - tl.load(ExptOffsSum)
    else:
        padding_m: tl.constexpr = 0

    index_type: tl.constexpr = tl.int64 if UPCAST_INDICES else tl.int32

    unpadded_m = grid_m - padding_m
    tl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    pid_emnk = pid
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, total_actual_tiles, XCD_SWIZZLE)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, GROUP_M)
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

    # X pointers
    offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
    if GatherIndx is None:
        X += start_m * stride_x_m
    else:
        GatherIndx += start_m
        offs_x_m = tl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
    offs_x_k = BLOCK_K * pid_k + tl.arange(0, BLOCK_K)
    XPtrs = (
        X
        + offs_x_m.to(index_type)[:, None] * stride_x_m
        + offs_x_k.to(index_type)[None, :] * stride_x_k
    )

    # W pointers (full 16-bit weights, no packing)
    offs_w_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_w_n = tl.max_contiguous(
        tl.multiple_of(offs_w_n % N, BLOCK_N),
        BLOCK_N,
    )
    offs_w_k = BLOCK_K * pid_k + tl.arange(0, BLOCK_K)
    W += expt_id * stride_w_e
    WPtrs = W + (
        offs_w_k.to(index_type)[:, None] * stride_w_k
        + offs_w_n.to(index_type)[None, :] * stride_w_n
    )

    num_k_iter = tl.cdiv(K, BLOCK_K * SPLIT_K)
    if not EVEN_K:
        num_k_iter -= 1

    # main accumulation loop
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(num_k_iter):
        x = tl.load(XPtrs)
        w = tl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)
        acc = tl.dot(x, w, acc=acc, input_precision="ieee")
        XPtrs += (BLOCK_K * SPLIT_K) * stride_x_k
        WPtrs += (BLOCK_K * SPLIT_K) * stride_w_k

    if not EVEN_K:
        mask_x_k = offs_x_k < MASK_K_LIMIT
        mask_w_k = offs_w_k < MASK_K_LIMIT
        x = tl.load(XPtrs, mask=mask_x_k[None, :], other=0.0)
        w = tl.load(
            WPtrs, mask=mask_w_k[:, None], other=0.0, cache_modifier=W_CACHE_MODIFIER
        )
        acc = tl.dot(x, w, acc=acc, input_precision="ieee")

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
        out = _swiglu(acc, alpha, limit, ADD_RESIDUAL=SWIGLU_ADD_RESIDUAL)
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
