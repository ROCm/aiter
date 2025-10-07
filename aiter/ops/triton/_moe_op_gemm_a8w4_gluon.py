# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


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
            mask = (gindx != -1)
            dst[gindx[mask]] = idx[mask]
            n_read_rows = (dst.view((-1, n_expts_act)) != -1).any(dim=1).sum()
        else:
            n_read_rows = n_tokens
        n_x_bytes = n_read_rows * X.shape[-1] * X.element_size()
        n_y_bytes = n_tokens * Y.shape[-1] * Y.element_size()
    ret["bytes"] = int(n_x_bytes + n_y_bytes + n_w_bytes)

    return ret


# TODO: using aiter swizzle instead can lead to perf degradation in rare cases
@gluon.jit
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: gl.constexpr):
    """
    Swizzle the program id based on integer XCD_SWIZZLE.
    This is useful for reording how blocks are ordered. A scheduler may, for example,
    assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2.
    This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment
    becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to
    the same hardware unit.
    """
    # Number of pids per group in the new arrangement
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE

    # Compute current current and local pid within the group
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE

    # Calculate new pid based on the new grouping
    new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
    return new_pid


@gluon.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: gl.constexpr = 1):
    """
    Maps 1D pid to 2D grid coords (pid_m, pid_n).

    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - GROUP_SIZE_M: tl.constexpr: default is 1
    """
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@gluon.jit
def unswizzle_mx_scale_cdna4(x, BLOCK_N: gl.constexpr, MX_SCALE_BLOCK_K: gl.constexpr,
                             N_PRESHUFFLE_FACTOR: gl.constexpr = 32):
    x = x.reshape(BLOCK_N // N_PRESHUFFLE_FACTOR, MX_SCALE_BLOCK_K // 8, 4, 16, 2, 2, 1)
    x = x.permute(0, 5, 3, 1, 4, 2, 6)
    x = x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)
    return x


@gluon.jit
def clip(x, limit, clip_lower: gl.constexpr):
    res = gl.minimum(x, limit)
    if clip_lower:
        res = gl.maximum(-limit, res)
    return res


@gluon.jit
def _swiglu(input, alpha, limit):
    gelu, linear = gl.split(gl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    gelu = gelu.to(gl.float32)
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(gl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + gl.exp2(-1.44269504089 * alpha * gelu))
    return gl.fma(s, linear, s)  # (s * (linear + 1))


@gluon.jit
def _compute_quant(tensor, scale):
    tensor = tensor.to(gl.float32)
    tensor = tensor / scale
    tensor = tensor.to(gl.float8e4nv)
    return tensor


@triton.jit
def _downcast_to_static_fp8(x_ptr, stride_x_m, stride_x_n,
                      y_ptr, stride_y_m, stride_y_n,
                      scale_ptr,
                      M, N,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):

    x_dtype: tl.constexpr = x_ptr.dtype.element_ty
    tl.static_assert((x_dtype == tl.bfloat16) or (x_dtype == tl.float16) or (x_dtype == tl.float32), f"{x_dtype=} must be bfloat16 or float16 or float32")

    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    x_ptr += start_m * stride_x_m + start_n * stride_x_n
    y_ptr += start_m * stride_y_m + start_n * stride_y_n

    offs_m = tl.arange(0, BLOCK_M)[None, :].to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)[:, None].to(tl.int64)

    mask_m = start_m + offs_m < M
    mask_n = start_n + offs_n < N
    mask_xy = mask_m & mask_n

    offs_x = offs_m * stride_x_m + offs_n * stride_x_n
    offs_y = offs_m * stride_y_m + offs_n * stride_y_n

    x = tl.load(x_ptr + offs_x, mask=mask_xy)

    y = _compute_quant(x, tl.load(scale_ptr))

    tl.store(y_ptr + offs_y, y, mask=mask_xy)


@triton.jit
def _reduce_grouped(X, stride_xb: tl.uint64, stride_xm: tl.uint64, stride_xn,  #
                    Out, stride_om: tl.uint64, stride_on,  # output tensor
                    InIndx, B, N,  #
                    # fused activation function
                    APPLY_SWIGLU: tl.constexpr, alpha, limit, ACTIVATION_REDUCTION_N: tl.constexpr,
                    K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_t = tl.program_id(0)
    BLOCK_N_OUT: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    # persistent along N: single program on N, iterate tiles of size BLOCK_N
    start = pid_t * K
    # load indices into a tuple
    if InIndx is None:
        indxs = (pid_t, )
    else:
        indxs = ()
        for i in tl.static_range(0, K):
            indxs = indxs + (tl.load(InIndx + start + i), )
    # determine first valid topk row
    fi = indxs[(K - 1)]
    for i in tl.static_range(K - 2, -1, -1):
        fi = tl.where(indxs[i] != -1, indxs[i], fi)
    # record overwritten row index (may be -1 if none)
    XPtrs = X + tl.arange(0, BLOCK_N) * stride_xn
    OutPtrs = Out + tl.arange(0, BLOCK_N_OUT) * stride_on
    for n_curr in tl.range(0, N, BLOCK_N, num_stages=4):
        acc = tl.zeros([BLOCK_N_OUT], dtype=tl.float32)
        x_n_mask = tl.arange(0, BLOCK_N) < N - n_curr
        # accumulate contributions for this tile
        for i in tl.static_range(0, K):
            curr = tl.zeros([BLOCK_N], dtype=tl.float32)
            # iterate over split_k partial values
            for b in tl.range(0, B):
                is_valid = indxs[i] != -1
                x_row_ptr = XPtrs + indxs[i] * stride_xm + b * stride_xb
                vals = tl.load(x_row_ptr, mask=x_n_mask & is_valid, other=0.0)
                vals = vals.to(tl.float32)
                curr += vals
            # apply nonlinearity to split-k output
            if APPLY_SWIGLU:
                curr = _swiglu(curr[None, :], alpha, limit)
            curr = tl.reshape(curr, [curr.shape[-1]])
            # update final accumulator
            acc += curr
        # Compute per-32-col MXFP scales for this tile if requested
        Nrem = (N - n_curr) // ACTIVATION_REDUCTION_N
        out_n_mask = tl.arange(0, BLOCK_N_OUT) < Nrem
        # write-back for this tile
        out_ptr = OutPtrs + pid_t * stride_om
        tl.store(out_ptr, acc, mask=out_n_mask)
        XPtrs += BLOCK_N * stride_xn
        OutPtrs += BLOCK_N_OUT * stride_on


@gluon.jit(launch_metadata=matmul_launch_metadata)
def _moe_gemm_a8w4_gluon(
             Y, stride_y_k, stride_y_m, stride_y_n,
             X, stride_x_m, stride_x_k,
             XMxScale, stride_x_mx_m, stride_x_mx_k,
             W, stride_w_e, stride_w_k, stride_w_n,
             WMxScale, stride_w_mx_e, stride_w_mx_k, stride_w_mx_n,
             X_static_scale, Quant_static_scale,
             B, stride_b_e, # Bias
             Gammas,
             N, K, # shapes
             # expt data
             GatherIndx,
             ExptHist, ExptOffs, ExptOffsSum, ExptData,
             # true grid size
             grid_m, grid_n,
             # fused activation function
             APPLY_SWIGLU: gl.constexpr, alpha, limit, ACTIVATION_REDUCTION_N: gl.constexpr,
             # MoE config
             N_EXPTS_ACT: gl.constexpr,
             # optimization config
             BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
             GROUP_M: gl.constexpr, XCD_SWIZZLE: gl.constexpr,
             # One of ["CDNA4", None]
             SWIZZLE_MX_SCALE: gl.constexpr,
             EVEN_K: gl.constexpr, SPLIT_K: gl.constexpr,
             W_CACHE_MODIFIER: gl.constexpr,
             UPCAST_INDICES: gl.constexpr = False):


    blocked: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 16], threads_per_warp=[4, 16], warps_per_cta=[4, 1], order=[1, 0])
    blocked1: gl.constexpr = gl.BlockedLayout(size_per_thread = [1, 4], threads_per_warp = [1, 64], warps_per_cta = [4, 1], order = [1, 0])
    blocked2: gl.constexpr = gl.BlockedLayout(size_per_thread = [16, 1], threads_per_warp = [8, 8], warps_per_cta = [1, 4], order = [0, 1])
    blocked3: gl.constexpr = gl.BlockedLayout(size_per_thread = [1, 4], threads_per_warp = [4, 16], warps_per_cta = [4, 1], order = [1, 0])
    linear1: gl.constexpr = gl.DistributedLinearLayout(reg_bases = [[0, 4], [0, 0]], lane_bases = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp_bases = [[0, 0], [0, 0]], block_bases = [], shape=[16, 8])
    #linear1: gl.constexpr = gl.DistributedLinearLayout(reg_bases = [[0, 4], [16, 0]], lane_bases = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp_bases = [[0, 0], [0, 0]], block_bases = [], shape=[32, 8])
    linear2: gl.constexpr = gl.DistributedLinearLayout(reg_bases = [[0, 2], [0, 1]], lane_bases = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], warp_bases = [[1, 0], [2, 0]], block_bases = [], shape=[4, 256])
    shared: gl.constexpr =  gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=16, order=[1, 0])
    shared1: gl.constexpr =  gl.SwizzledSharedLayout(vec=16, per_phase=2, max_phase=8, order=[0, 1])
    shared2: gl.constexpr =  gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 4], tiles_per_warp=[2, 2])
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=16)
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=16)

    MX_PACK_DIVISOR: gl.constexpr = 32
    w_type: gl.constexpr = W.dtype.element_ty
    gl.static_assert(w_type == gl.uint8, "mx_weight_ptr must be uint8")
    gl.static_assert(WMxScale.dtype.element_ty == gl.uint8, "mx_scale_ptr must be uint8")
    gl.static_assert(BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR")

    OUT_BLOCK_N: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = gl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        # Determine how much padding there is on the expert data. This allows us to
        # know the true grid size and avoid processing padding tiles.
        padding_m = grid_m - gl.load(ExptOffsSum)
    else:
        padding_m: gl.constexpr = 0

    index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32

    unpadded_m = grid_m - padding_m
    total_actual_tiles = unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    # swizzle program ids
    pid_emnk = pid
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, total_actual_tiles, XCD_SWIZZLE)
    #pid_e = pid_emnk // (unpadded_m * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, GROUP_M)
    # For split-k, advance to the output k slice
    if SPLIT_K > 1:
        Y += pid_k.to(index_type) * stride_y_k
    # unpack expert data
    expt_data = gl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = gl.load(ExptHist + expt_id)
    start_m = gl.load(ExptOffs + expt_id)
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m = start_m.to(index_type)
    pid_n, pid_k = pid_n.to(index_type), pid_k.to(index_type)

    # A pointers
    offs_x_m = (BLOCK_M * block_id + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked))) % M
    #offs_x_m = offs_x_m % M
    if GatherIndx is None:
        X += start_m * stride_x_m
    else:
        GatherIndx += start_m
        # no needs to bounds-check here because `offs_x_m` wraps around M dim
        offs_x_m = gl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
    offs_x_k = BLOCK_K * pid_k + gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked))
    offs_x = offs_x_m.to(index_type)[:, None] * stride_x_m + offs_x_k.to(index_type)[None, :] * stride_x_k
    XPtrs = X #+ offs_x_m.to(index_type)[:, None] * stride_x_m + offs_x_k.to(index_type)[None, :] * stride_x_k

    W_K_DIVISOR: gl.constexpr = 2
    W_N_DIVISOR: gl.constexpr = 1
    PACKED_BLOCK_K_W: gl.constexpr = BLOCK_K // W_K_DIVISOR
    PACKED_BLOCK_N_W: gl.constexpr = BLOCK_N // W_N_DIVISOR
    MX_SCALE_BLOCK_K: gl.constexpr = BLOCK_K // MX_PACK_DIVISOR

    WMxScale += expt_id * stride_w_mx_e
    if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
        gl.static_assert(stride_w_mx_k is not None)
        gl.static_assert(stride_w_mx_n is not None)
        NON_K_PRESHUFFLE_BLOCK_SIZE: gl.constexpr = 32
        PACKED_MX_BLOCK: gl.constexpr = MX_SCALE_BLOCK_K * NON_K_PRESHUFFLE_BLOCK_SIZE
        SCALE_BLOCK_N: gl.constexpr = BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE
    else:
        PACKED_MX_BLOCK: gl.constexpr = MX_SCALE_BLOCK_K
        SCALE_BLOCK_N: gl.constexpr = BLOCK_N
    offs_w_n_scale = (pid_n * SCALE_BLOCK_N + gl.arange(0, SCALE_BLOCK_N, layout=gl.SliceLayout(1, blocked1))) % N
    # K dimension must be the last dimension for the scales
    offs_w_k_scale = PACKED_MX_BLOCK * pid_k + gl.arange(0, PACKED_MX_BLOCK, layout=gl.SliceLayout(0, blocked1))
    offs_w_scale = offs_w_k_scale.to(index_type)[None, :] * stride_w_mx_k + offs_w_n_scale.to(index_type)[:, None] * stride_w_mx_n
    WMxScalePtrs = WMxScale #+ offs_w_k_scale.to(index_type)[None, :] * stride_w_mx_k + offs_w_n_scale.to(index_type)[:, None] * stride_w_mx_n

    # B pointers
    offs_w_n = pid_n * PACKED_BLOCK_N_W + gl.arange(0, PACKED_BLOCK_N_W, layout=gl.SliceLayout(0, blocked2))
    offs_w_n = offs_w_n % (N // W_N_DIVISOR)
    offs_w_k = PACKED_BLOCK_K_W * pid_k + gl.arange(0, PACKED_BLOCK_K_W, layout=gl.SliceLayout(1, blocked2))
    W += expt_id * stride_w_e
    offs_w = offs_w_k.to(index_type)[:, None] * stride_w_k + offs_w_n.to(index_type)[None, :] * stride_w_n
    WPtrs = W  #+ (offs_w_k.to(index_type)[:, None] * stride_w_k + offs_w_n.to(index_type)[None, :] * stride_w_n)

    smem_x = gl.allocate_shared_memory(X.dtype.element_ty, [BLOCK_M, BLOCK_K], shared)
    smem_w = gl.allocate_shared_memory(W.dtype.element_ty, [PACKED_BLOCK_K_W, PACKED_BLOCK_N_W], shared1)
    smem_w_scales = gl.allocate_shared_memory(WMxScale.dtype.element_ty, [SCALE_BLOCK_N, PACKED_MX_BLOCK], shared2)
   
    x = gl.amd.cdna4.buffer_load(XPtrs, offsets=offs_x)
    w = gl.amd.cdna4.buffer_load(WPtrs, offsets=offs_w, cache=W_CACHE_MODIFIER)
    x_scales_dot = gl.full((BLOCK_M, MX_SCALE_BLOCK_K), 127, dtype=gl.uint8, layout=linear1)
    w_scales = gl.amd.cdna4.buffer_load(WMxScalePtrs, offsets=offs_w_scale)

    smem_x.store(x) 
    smem_w.store(w)
    smem_w_scales.store(w_scales)

    # compute output
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout)
    for k in range(K, BLOCK_K * pid_k + BLOCK_K * SPLIT_K, -(BLOCK_K * SPLIT_K)):
        offs_w_scale += (PACKED_MX_BLOCK * SPLIT_K) * stride_w_mx_k
        offs_x += (BLOCK_K * SPLIT_K) * stride_x_k
        offs_w += (PACKED_BLOCK_K_W * SPLIT_K) * stride_w_k

        x = gl.amd.cdna4.buffer_load(XPtrs, offsets=offs_x)
        x_dot = smem_x.load(layout=dot_a_layout)
        w = gl.amd.cdna4.buffer_load(WPtrs, offsets=offs_w, cache=W_CACHE_MODIFIER)
        w_dot = smem_w.load(layout=dot_b_layout)

        w_scales = gl.amd.cdna4.buffer_load(WMxScalePtrs, offsets=offs_w_scale)
        w_scales_dot = smem_w_scales.load(layout=linear2)
        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            w_scales_dot = unswizzle_mx_scale_cdna4(w_scales_dot, BLOCK_N, MX_SCALE_BLOCK_K)

        acc = gl.amd.cdna4.mfma_scaled(x_dot, x_scales_dot, "e4m3", w_dot, w_scales_dot, "e2m1", acc=acc)

        smem_x.store(x) 
        smem_w.store(w)
        smem_w_scales.store(w_scales)
    
    x_dot = smem_x.load(layout=dot_a_layout)
    w_dot = smem_w.load(layout=dot_b_layout)  
    w_scales_dot = smem_w_scales.load(layout=linear2)
    if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
        w_scales_dot = unswizzle_mx_scale_cdna4(w_scales_dot, BLOCK_N, MX_SCALE_BLOCK_K)

    acc = gl.amd.cdna4.mfma_scaled(x_dot, x_scales_dot, "e4m3", w_dot, w_scales_dot, "e2m1", acc=acc)

    # scalar fp8 scale
    if X_static_scale is not None:
        acc = acc * gl.load(X_static_scale)
    # bias
    offs_m = BLOCK_M * block_id + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked3))
    mask_m = offs_m < M
    if B is not None:
        BPtrs = B + expt_id * stride_b_e 
        offs_bias = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
        bias = gl.amd.cdna4.buffer_load(BPtrs, offsets=offs_bias)
        acc = acc + bias[None, :]
    if APPLY_SWIGLU and SPLIT_K == 1:
        out = _swiglu(acc, alpha, limit)
        gl.static_assert(out.shape[1] == OUT_BLOCK_N, f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})")
        offs_y_n = OUT_BLOCK_N * pid_n + gl.arange(0, OUT_BLOCK_N, layout=gl.SliceLayout(0, blocked3))
    else:
        gl.static_assert(ACTIVATION_REDUCTION_N == 1, "Activation reduction must be 1 if no activation fn is provided")
        offs_y_n = BLOCK_N * pid_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked3))
        out = acc
    mask_n = offs_y_n < yN
    if Gammas is not None:
        offs_gamma = BLOCK_M * block_id + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout))
        mask_gamma = offs_gamma < M
        gammas = gl.load(Gammas + start_m + offs_gamma, mask=mask_gamma, other=0.0)
        out *= gammas[:, None]
    # quant
    if Quant_static_scale is not None:
        out = _compute_quant(out, gl.load(Quant_static_scale))
    else:
        out = out.to(gl.float8e4nv)
    # write-back
    Y += start_m * stride_y_m
    offs_y_m = offs_m
    YPtrs = Y 
    offs_y = offs_y_m.to(index_type)[:, None] * stride_y_m + offs_y_n.to(index_type)[None, :] * stride_y_n
    mask = mask_m[:, None] & mask_n[None, :]
    out = gl.convert_layout(out, layout=blocked3)
    gl.amd.cdna4.buffer_store(stored_value=out, ptr=YPtrs, offsets=offs_y, mask=mask)
