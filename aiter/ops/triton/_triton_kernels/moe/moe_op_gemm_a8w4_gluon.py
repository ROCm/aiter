# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.language.core import _aggregate as aggregate
import triton.experimental.gluon.language as gl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid
from aiter.ops.triton._triton_kernels.moe.quant_moe import _compute_static_fp8_quant


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

    def repr(s, x):
        return f"{s}={x}" if x is not None else f"E_{len(hist)}({s})={n_rows}"

    nbits = X.dtype.itemsize * 8
    ret["name"] = f"{kernel.name} [{repr('M', M)}, {repr('N', N)}, {repr('K', K)}]"
    gindx = args.get("GatherIndx", None)
    # sindx = args.get("WriteBackIndx", None)
    if gindx is not None:
        ret["name"] += "_layer1"
    else:
        ret["name"] += "_layer2"
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


@gluon.jit
def unswizzle_mx_scale(
    x,
    BLOCK_N: tl.constexpr,
    MX_SCALE_BLOCK_K: tl.constexpr,
    PRESHUFFLE_FACTOR: tl.constexpr = 32,
):
    x = x.reshape(BLOCK_N // N_PRESHUFFLE_FACTOR, MX_SCALE_BLOCK_K // 8, 4, 16, 2, 2, 1)
    x = x.permute(0, 5, 3, 1, 4, 2, 6)
    x = x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)

    b_scale_buffer_slice = b_scale_buffer_slice.reshape((
                cfg.BLOCK_N_PRESHUFFLED,  #
                BLOCK_K_SCALE // cfg.SCALE_KWIDTH,  #
                cfg.PRESHUFFLE_FACTOR // 4,  #
                4,  #
                cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_N, BLOCK_K_SCALE))
    return x


@gluon.jit
def clip(x, limit, clip_lower: tl.constexpr):
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
    linear = linear.to(tl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + gl.exp2(-1.44269504089 * alpha * gelu))
    return gl.fma(s, linear, s)  # (s * (linear + 1))


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


@gluon.jit(launch_metadata=matmul_launch_metadata)
def _moe_gemm_a8w4(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    XMxScale,
    stride_x_mx_m,
    stride_x_mx_k,
    W,
    stride_w_e,
    stride_w_n,
    stride_w_k,
    WMxScale,
    stride_w_mx_e,
    stride_w_mx_n,
    stride_w_mx_k,
    X_static_scale,
    Quant_static_scale,
    B,
    stride_b_e,  # Bias
    Gammas,
    num_tokens,
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
    APPLY_SWIGLU: gl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: gl.constexpr,
    # MoE config
    N_EXPTS_ACT: gl.constexpr,
    # optimization config
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    # One of ["GFX1250", None]
    SWIZZLE_MX_SCALE: gl.constexpr,
    EVEN_K: gl.constexpr,
    MASK_K_LIMIT: gl.constexpr,
    SPLIT_K: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
    UPCAST_INDICES: gl.constexpr = False,
):

    NUM_BUFFERS: gl.constexpr = 2
    MAX_NUM_INDICES: gl.constexpr = 16
    NUM_GATHER_LOADS: gl.constexpr = BLOCK_M // MAX_NUM_INDICES

    is_x_microscaled: gl.constexpr = XMxScale is not None
    MX_PACK_DIVISOR: gl.constexpr = 32
    w_type: gl.constexpr = W.dtype.element_ty
    gl.static_assert(w_type == gl.uint8, "mx_weight_ptr must be uint8 or fp8")
    gl.static_assert(
        WMxScale.dtype.element_ty == gl.uint8, "mx_scale_ptr must be uint8"
    )
    gl.static_assert(
        BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR"
    )
    x_type: gl.constexpr = X.dtype.element_ty
    if is_x_microscaled:
        gl.static_assert(x_type == gl.float8e4nv, "mx_act_ptr must be float8e4nv")
        gl.static_assert(
            XMxScale.dtype.element_ty == gl.uint8, "mx_scale_ptr must be uint8"
        )

    OUT_BLOCK_N: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = gl.program_id(0)
    if ExptOffsSum is not None:
        # Determine how much padding there is on the expert data. This allows us to
        # know the true grid size and avoid processing padding tiles.
        padding_m = grid_m - gl.load(ExptOffsSum)
    else:
        padding_m: tl.constexpr = 0

    index_type: tl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32

    unpadded_m = grid_m - padding_m
    total_actual_tiles = unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    # swizzle program ids
    pid_emnk = pid
    # pid_e = pid_emnk // (unpadded_m * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, GROUP_M)
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
    off_x_m = BLOCK_M * block_id 
    if GatherIndx is None:
        X += start_m * stride_x_m
    else:
        IDX_BASE_LAYOUT: gl.constexpr = gl.BlockedLayout([BLOCK_M, 1], [1, 32], [1, 4], [1, 0])
        IDX_LAYOUT: gl.constexpr = gl.SliceLayout(1, IDX_BASE_LAYOUT)
        offs_x_m = BLOCK_M * block_id + gl.arange(0, BLOCK_M, layout=IDX_LAYOUT)
        GatherIndx += start_m
        offs_x_m = gl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT

    W_K_DIVISOR: gl.constexpr = 2
    W_N_DIVISOR: gl.constexpr = 1
    PACKED_BLOCK_K_W: gl.constexpr = BLOCK_K // W_K_DIVISOR
    PACKED_BLOCK_N_W: gl.constexpr = BLOCK_N // W_N_DIVISOR
    MX_SCALE_BLOCK_K: gl.constexpr = BLOCK_K // MX_PACK_DIVISOR

    WMxScale += expt_id * stride_w_mx_e
    if SWIZZLE_MX_SCALE == "GFX1250_SCALE":
        gl.static_assert(stride_w_mx_k is not None)
        gl.static_assert(stride_w_mx_n is not None)
        PRESHUFFLE_FACTOR: gl.constexpr = 128
        PACKED_MX_BLOCK: gl.constexpr = MX_SCALE_BLOCK_K * PRESHUFFLE_FACTOR
        SCALE_BLOCK_N: gl.constexpr = BLOCK_N // PRESHUFFLE_FACTOR
        SCALE_KWIDTH: gl.constexpr = 4 if MX_SCALE_BLOCK_K >= 4 else MX_SCALE_BLOCK_K
    else:
        PRESHUFFLE_FACTOR: gl.constexpr = 1
        PACKED_MX_BLOCK: gl.constexpr = MX_SCALE_BLOCK_K
        SCALE_BLOCK_N: gl.constexpr = BLOCK_N
    off_w_n_scale = pid_n * SCALE_BLOCK_N

    # B pointers
    off_w_n = pid_n * PACKED_BLOCK_N_W
    W += expt_id * stride_w_e

    if is_x_microscaled:
        if GatherIndx is None:
            XMxScale += start_m * stride_x_mx_m
        offs_x_k_scale = MX_SCALE_BLOCK_K * pid_k + tl.arange(0, MX_SCALE_BLOCK_K)
        XMxScalePtrs = (
            XMxScale
            + offs_x_m.to(index_type)[:, None] * stride_x_mx_m
            + offs_x_k_scale.to(index_type)[None, :] * stride_x_mx_k
        )

    SHARED_LAYOUT_X: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 16]], [BLOCK_M, BLOCK_K], [1, 0])
    SHARED_LAYOUT_W: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[PACKED_BLOCK_K_W, 16]], [BLOCK_N, PACKED_BLOCK_K_W], [1, 0])
    SHARED_LAYOUT_W_SCALES: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [SCALE_BLOCK_N, PACKED_MX_BLOCK], [1, 0])

    if GatherIndx is None:
        x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=X, shape=(M, K),
                                                            strides=(stride_x_m, stride_x_k), block_shape=(BLOCK_M, BLOCK_K),
                                                            layout=SHARED_LAYOUT_X)
    else:
        x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=X, shape=(num_tokens, K),
                                                            strides=(stride_x_m, stride_x_k), block_shape=(BLOCK_M, BLOCK_K),
                                                            layout=SHARED_LAYOUT_X)
    w_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=W, shape=(N, K // W_K_DIVISOR),
                                                             strides=(stride_w_n, stride_w_k,),
                                                             block_shape=(BLOCK_N, PACKED_BLOCK_K_W,), layout=SHARED_LAYOUT_W)
    w_scales_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=WMxScale, shape=(N // PRESHUFFLE_FACTOR, tl.cdiv(K, MX_PACK_DIVISOR) * PRESHUFFLE_FACTOR),
                                                             strides=(stride_w_mx_n, stride_w_mx_k),
                                                             block_shape=(SCALE_BLOCK_N, PACKED_MX_BLOCK), layout=SHARED_LAYOUT_W_SCALES)

    if SWIZZLE_MX_SCALE == "GFX1250_SCALE":
        WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(3, transposed=True, warp_bases=[[0, 2], [1, 0]], reg_bases=[[0, 1]], instr_shape=[16, 16, 128])
        WMMA_LAYOUT_PACKED: gl.constexpr = gl.amd.AMDWMMALayout(3, transposed=True, warp_bases=[[0, 2], [1, 0]], reg_bases=[[0, 1]], instr_shape=[16, 16, 64])
    else:
        WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(3, transposed=True, warp_bases=[[0, 1], [1, 0]], reg_bases=[], instr_shape=[16, 16, 128])
        WMMA_LAYOUT_PACKED: gl.constexpr = gl.amd.AMDWMMALayout(3, transposed=True, warp_bases=[[0, 1], [1, 0]], reg_bases=[], instr_shape=[16, 16, 64])
    DOT_LAYOUT_X: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, k_width=16)
    DOT_LAYOUT_W: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT_PACKED, k_width=16)
    DOT_LAYOUT_W_SCALES: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(DOT_LAYOUT_W, [BLOCK_N, MX_SCALE_BLOCK_K])

    x_buffer = gl.allocate_shared_memory(x_desc.dtype, shape=[NUM_BUFFERS] + x_desc.block_shape, layout=x_desc.layout)
    w_buffer = gl.allocate_shared_memory(w_desc.dtype, shape=[NUM_BUFFERS] + w_desc.block_shape, layout=w_desc.layout)
    w_scales_buffer = gl.allocate_shared_memory(w_scales_desc.dtype, shape=[NUM_BUFFERS] + w_scales_desc.block_shape, layout=w_scales_desc.layout)

    producer = 0
    consumer = 0
    for _ in gl.static_range(NUM_BUFFERS - 1):
        if GatherIndx is None:
            gl.amd.gfx1250.tdm.async_load(x_desc, [off_x_m, producer * BLOCK_K], x_buffer.index(producer % NUM_BUFFERS))
        else:
            gl.amd.gfx1250.tdm.async_gather(x_desc, offs_x_m, producer * BLOCK_K, x_buffer.index(producer % NUM_BUFFERS)) 
        gl.amd.gfx1250.tdm.async_load(w_desc, [off_w_n, producer * PACKED_BLOCK_K_W], w_buffer.index(producer % NUM_BUFFERS))
        gl.amd.gfx1250.tdm.async_load(w_scales_desc, [off_w_n_scale, producer * PACKED_MX_BLOCK], w_scales_buffer.index(producer % NUM_BUFFERS))
        producer += 1

    # compute output
    num_k_iter = tl.cdiv(K, BLOCK_K)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    for k in range(num_k_iter - (NUM_BUFFERS - 1)):
        if GatherIndx is None:
            gl.amd.gfx1250.tdm.async_load(x_desc, [off_x_m, producer * BLOCK_K], x_buffer.index(producer % NUM_BUFFERS))
        else:
            gl.amd.gfx1250.tdm.async_gather(x_desc, offs_x_m, producer * BLOCK_K, x_buffer.index(producer % NUM_BUFFERS)) 
        gl.amd.gfx1250.tdm.async_load(w_desc, [off_w_n, producer * PACKED_BLOCK_K_W], w_buffer.index(producer % NUM_BUFFERS))
        gl.amd.gfx1250.tdm.async_load(w_scales_desc, [off_w_n_scale, producer * PACKED_MX_BLOCK], w_scales_buffer.index(producer % NUM_BUFFERS))
        producer += 1

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 3)

        x = x_buffer.index(consumer % NUM_BUFFERS).load(layout=DOT_LAYOUT_X)
        w = w_buffer.index(consumer % NUM_BUFFERS).permute((1, 0)).load(layout=DOT_LAYOUT_W)
        w_scales_buffer_slice = w_scales_buffer.index(consumer % NUM_BUFFERS) #.load(layout=DOT_LAYOUT_W_SCALES)
        if SWIZZLE_MX_SCALE == "GFX1250_SCALE":
            w_scales_buffer_slice = w_scales_buffer_slice.reshape((
                SCALE_BLOCK_N, 
                MX_SCALE_BLOCK_K // SCALE_KWIDTH, 
                PRESHUFFLE_FACTOR // 4, 
                4, 
                SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((BLOCK_N, MX_SCALE_BLOCK_K))
        w_scales = w_scales_buffer_slice.load(layout=DOT_LAYOUT_W_SCALES)

        acc = gl.amd.gfx1250.wmma_scaled(x, 0, "e4m3", w, w_scales, "e2m1", acc)

        consumer += 1

    for k_ep in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - k_ep) * 3)

        x = x_buffer.index(consumer % NUM_BUFFERS).load(layout=DOT_LAYOUT_X)
        w = w_buffer.index(consumer % NUM_BUFFERS).permute((1, 0)).load(layout=DOT_LAYOUT_W)
        w_scales_buffer_slice = w_scales_buffer.index(consumer % NUM_BUFFERS)
        if SWIZZLE_MX_SCALE == "GFX1250_SCALE":
            w_scales_buffer_slice = w_scales_buffer_slice.reshape((
                SCALE_BLOCK_N, 
                MX_SCALE_BLOCK_K // SCALE_KWIDTH, 
                PRESHUFFLE_FACTOR // 4, 
                4, 
                SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((BLOCK_N, MX_SCALE_BLOCK_K))
        w_scales = w_scales_buffer_slice.load(layout=DOT_LAYOUT_W_SCALES)

        acc = gl.amd.gfx1250.wmma_scaled(x, 0, "e4m3", w, w_scales, "e2m1", acc)

    # scalar fp8 scale
    if X_static_scale is not None:
        acc = acc * gl.load(X_static_scale)
    # bias
    offs_m = BLOCK_M * block_id + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT))
    offs_y_n = BLOCK_N * pid_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))
    mask_m = offs_m < M
    mask_n = offs_y_n < N
    if B is not None:
        BPtrs = B + expt_id * stride_b_e #+ offs_y_n
        #bias = gl.load(BPtrs, mask=mask_n, other=0)
        bias = gl.amd.gfx1250.buffer_load(BPtrs, offs_y_n, mask=mask_n)
        acc = acc + bias[None, :]
    if APPLY_SWIGLU:
        out = _swiglu(acc, alpha, limit)
        tl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
        #STORE_LAYOUT: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 16], threads_per_warp=[4, 8], warps_per_cta=[4, 1], order=[1, 0])
        offs_m = BLOCK_M * block_id + gl.arange(0, BLOCK_M) #, layout=gl.SliceLayout(1, STORE_LAYOUT))
        offs_y_n = OUT_BLOCK_N * pid_n + gl.arange(0, OUT_BLOCK_N) #, layout=gl.SliceLayout(0, STORE_LAYOUT))
        mask_m = offs_m < M
        mask_n = offs_y_n < yN
    else:
        tl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )
        out = acc
    if Gammas is not None:
        gammas = gl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        out *= gammas[:, None]
    # quant
    if Quant_static_scale is not None:
        out = _compute_static_fp8_quant(out, gl.load(Quant_static_scale))
    # write-back
    Y += start_m * stride_y_m
    offs_y_m = offs_m
    #YPtrs = (
    #    Y
    #    + offs_y_m.to(index_type)[:, None] * stride_y_m
    #    + offs_y_n.to(index_type)[None, :] * stride_y_n
    #)
    offs_y = offs_y_m.to(index_type)[:, None] * stride_y_m + offs_y_n.to(index_type)[None, :] * stride_y_n
    mask = mask_m[:, None] & mask_n[None, :]
    if Quant_static_scale is None:
        out = out.to(tl.bfloat16)
    #if APPLY_SWIGLU:
    #    out = gl.convert_layout(out, layout=STORE_LAYOUT)
    #gl.store(YPtrs, out, mask=mask)
    gl.amd.gfx1250.buffer_store(out, Y, offs_y, mask=mask)