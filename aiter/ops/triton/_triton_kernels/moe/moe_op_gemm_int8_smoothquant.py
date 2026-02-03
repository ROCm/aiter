# Int8 SmoothQuant MoE GEMM Kernel
# Adapted from moe_op_gemm_a8w8.py for int8 quantization with per-token scaling

import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid


def matmul_launch_metadata(grid, kernel, args):
    """Provide launch metadata for profiling (flops, bytes)."""
    ret = dict()
    N, K = args["N"], args["K"]
    Y, X, W = args["Y"], args["X"], args["W"]
    hist = args["ExptHist"]
    
    if hist is not None:
        n_tokens = float(hist.sum())
        n_w_bytes = (W.numel() * W.element_size() // hist.numel()) * (hist > 0).sum()
    else:
        n_tokens = X.shape[0]
        n_w_bytes = W.numel() * W.element_size()

    # For int8 GEMM, use flops8 (8-bit operations)
    nbits = 8
    ret["name"] = f"{kernel.name} [M={n_tokens}, N={N}, K={K}]"
    
    gindx = args.get("GatherIndx", None)
    if gindx is not None:
        ret["name"] += "_layer1"
    else:
        ret["name"] += "_layer2"
    if args["B"] is not None:
        ret["name"] += "_bias"
    if args["APPLY_SWIGLU"]:
        ret["name"] += "_swiglu"
    if args["APPLY_SILU"]:
        ret["name"] += "_silu"

    fM = n_tokens
    fK = K if K is not None else n_tokens
    ret[f"flops{nbits}"] = 2.0 * fM * N * fK

    # Bytes: input activations + weights + output
    gindx = args.get("GatherIndx", None)
    n_x_bytes = X.numel() * X.element_size()
    n_y_bytes = Y.numel() * Y.element_size()
    if hist is not None:
        n_expts_act = args["N_EXPTS_ACT"]
        if gindx is not None:
            # Recreate inverse GatherIndx
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
    """
    Swizzle the program id for better XCD locality.
    """
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE
    new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
    return new_pid


@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def _swiglu(input, alpha, limit):
    """SwiGLU activation: splits input in half and applies gated activation."""
    gelu, linear = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    gelu = gelu.to(tl.float32)
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + tl.exp2(-1.44269504089 * alpha * gelu))
    return tl.fma(s, linear, s)  # s * (linear + 1)


@triton.jit
def _silu(input):
    """SiLU activation: x * sigmoid(x)"""
    x = input.to(tl.float32)
    return x * tl.sigmoid(x)


@triton.jit(launch_metadata=matmul_launch_metadata)
def _moe_gemm_int8_smoothquant(
    # Output tensor
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    # Input activation tensor (int8)
    X,
    stride_x_m,
    stride_x_k,
    # Activation scale (fp32, per token)
    XScale,
    stride_x_scale,
    # Weight tensor (int8)
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    # Weight scale (fp32, per output channel per expert)
    WScale,
    stride_w_scale_e,
    stride_w_scale_n,
    # Optional bias
    B,
    stride_b_e,
    # Gammas for gating
    Gammas,
    # Shapes
    N,
    K,
    # Expert data
    GatherIndx,
    ExptHist,
    ExptOffs,
    ExptOffsSum,
    ExptData,
    # Grid dimensions
    grid_m,
    grid_n,
    # Activation options
    APPLY_SWIGLU: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: tl.constexpr,
    # MoE config
    N_EXPTS_ACT: tl.constexpr,
    # Block sizes
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
    """
    Int8 MoE GEMM with SmoothQuant support.
    
    Performs: Y = (X @ W) * x_scale * w_scale
    
    Where:
    - X is int8 activations [M, K]
    - W is int8 weights [E, K, N] 
    - x_scale is fp32 per-token scale [M]
    - w_scale is fp32 per-output-channel scale [E, N]
    
    The matmul is done in int32 accumulator, then converted to fp32
    and scaled by both x_scale and w_scale.
    """
    # Assume positive strides for compiler hints
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

    # Swizzle program ids
    pid_emnk = pid
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, total_actual_tiles, XCD_SWIZZLE)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, GROUP_M)
    
    # For split-k, advance to the output k slice
    if SPLIT_K > 1:
        Y += pid_k.to(index_type) * stride_y_k
    
    # Unpack expert data
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

    # X pointers (int8 activations)
    offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
    if GatherIndx is None:
        X += start_m * stride_x_m
        XScale += start_m * stride_x_scale
    else:
        GatherIndx += start_m
        offs_x_m = tl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
    offs_x_k = BLOCK_K * pid_k + tl.arange(0, BLOCK_K)
    XPtrs = (
        X
        + offs_x_m.to(index_type)[:, None] * stride_x_m
        + offs_x_k.to(index_type)[None, :] * stride_x_k
    )

    # W pointers (int8 weights)
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

    # Compute matmul in int32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(num_k_iter):
        x = tl.load(XPtrs)  # int8
        w = tl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)  # int8
        
        # Int8 x Int8 -> Int32 dot product
        acc = tl.dot(x, w, acc, out_dtype=tl.int32)

        XPtrs += (BLOCK_K * SPLIT_K) * stride_x_k
        WPtrs += (BLOCK_K * SPLIT_K) * stride_w_k

    # Handle non-even K
    if not EVEN_K:
        mask_x_k = offs_x_k < MASK_K_LIMIT
        mask_w_k = offs_w_k < MASK_K_LIMIT

        x = tl.load(XPtrs, mask=mask_x_k[None, :], other=0)
        w = tl.load(
            WPtrs, mask=mask_w_k[:, None], other=0, cache_modifier=W_CACHE_MODIFIER
        )
        
        acc = tl.dot(x, w, acc, out_dtype=tl.int32)

    # Convert int32 accumulator to fp32
    acc_fp32 = acc.to(tl.float32)

    # Apply per-token activation scale
    offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    if GatherIndx is None:
        x_scale_ptrs = XScale + offs_x_m * stride_x_scale
    else:
        # For gathered inputs, load scale from original positions
        x_scale_ptrs = XScale + offs_x_m * stride_x_scale
    x_scale = tl.load(x_scale_ptrs, mask=mask_m, other=1.0)
    acc_fp32 = acc_fp32 * x_scale[:, None]

    # Apply per-channel weight scale
    offs_y_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
    mask_n = offs_y_n < N
    w_scale_ptrs = WScale + expt_id * stride_w_scale_e + offs_y_n * stride_w_scale_n
    w_scale = tl.load(w_scale_ptrs, mask=mask_n, other=1.0)
    acc_fp32 = acc_fp32 * w_scale[None, :]

    # Apply bias if provided
    if B is not None:
        BPtrs = B + expt_id * stride_b_e + offs_y_n
        if pid_k == 0:
            bias = tl.load(BPtrs, mask=mask_n, other=0, cache_modifier=W_CACHE_MODIFIER)
        else:
            bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
        acc_fp32 = acc_fp32 + bias[None, :]

    # Apply activation function
    if APPLY_SWIGLU and SPLIT_K == 1:
        out = _swiglu(acc_fp32, alpha, limit)
        tl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
        offs_y_n = OUT_BLOCK_N * pid_n + tl.arange(0, OUT_BLOCK_N)
        mask_n = offs_y_n < yN
    elif APPLY_SILU and SPLIT_K == 1:
        out = _silu(acc_fp32)
    else:
        tl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )
        out = acc_fp32

    # Apply gammas if provided
    if Gammas is not None:
        gammas = tl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        out *= gammas[:, None]

    # Write back
    Y += start_m * stride_y_m
    offs_y_m = offs_m
    YPtrs = (
        Y
        + offs_y_m.to(index_type)[:, None] * stride_y_m
        + offs_y_n.to(index_type)[None, :] * stride_y_n
    )
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(YPtrs, out, mask=mask)


@triton.jit
def _reduce_grouped_int8(
    X,
    stride_xb: tl.uint64,
    stride_xm: tl.uint64,
    stride_xn,
    Out,
    stride_om: tl.uint64,
    stride_on,
    InIndx,
    B,
    N,
    # Fused activation function
    APPLY_SWIGLU: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """
    Grouped row reduction for MoE with optional activation and dtype conversion.
    """
    pid_t = tl.program_id(1)
    pid_n = tl.program_id(0)

    BLOCK_N_OUT: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    start = pid_t * K
    
    # Load indices
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
    
    # Accumulate contributions
    for i in tl.static_range(0, K):
        curr = tl.zeros([BLOCK_N], dtype=tl.float32)
        for b in tl.range(0, B):
            x_row_ptr = XPtrs + indxs[i] * stride_xm + b * stride_xb
            if EVEN_N:
                vals = tl.load(x_row_ptr)
            else:
                vals = tl.load(x_row_ptr, mask=x_n_mask, other=0.0)
            vals = vals.to(tl.float32)
            curr += vals

        # Apply activation after split-k reduction
        if APPLY_SWIGLU:
            curr = _swiglu(curr[None, :], alpha, limit)
        elif APPLY_SILU:
            curr = _silu(curr[None, :])
        curr = tl.reshape(curr, [curr.shape[-1]])
        acc += curr

    Nrem = N // ACTIVATION_REDUCTION_N

    # Convert to output dtype and write back
    if OUT_DTYPE == tl.bfloat16:
        out_vals = acc.to(tl.bfloat16)
    else:
        out_vals = acc
    
    out_ptr = OutPtrs + pid_t * stride_om
    if EVEN_N:
        tl.store(out_ptr, out_vals)
    else:
        out_n_mask = pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT) < Nrem
        tl.store(out_ptr, out_vals, mask=out_n_mask)
