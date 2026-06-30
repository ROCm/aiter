import torch
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl

from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd, pid_grid
from aiter.ops.triton._triton_kernels.moe.activations import _swiglu

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


# TODO: using aiter swizzle instead can lead to perf degradation in rare cases
@gluon.jit
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
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
def unswizzle_mx_scale_cdna4(
    x,
    BLOCK_N: tl.constexpr,
    MX_SCALE_BLOCK_K: tl.constexpr,
    N_PRESHUFFLE_FACTOR: tl.constexpr = 32,
):
    x = x.reshape(BLOCK_N // N_PRESHUFFLE_FACTOR, MX_SCALE_BLOCK_K // 8, 4, 16, 2, 2, 1)
    x = x.permute(0, 5, 3, 1, 4, 2, 6)
    x = x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)
    return x

@gluon.jit(launch_metadata=matmul_launch_metadata)
def _moe_gemm_a16w4(
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
    WMxScale,  # E8M0 scale, pre-expanded by 32x along K -> shape (E, K, N) uint8
    stride_w_mx_e,
    stride_w_mx_k,
    stride_w_mx_n,
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
    ADD_RESIDUAL: gl.constexpr,
    # MoE config
    N_EXPTS_ACT: gl.constexpr,
    # optimization config
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    # Must be None: the kernel takes pre-expanded e8m0 scales (one byte per fp4 element).
    SWIZZLE_MX_SCALE: gl.constexpr,
    MASK_K_LIMIT: gl.constexpr,
    SPLIT_K: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
    num_warps: gl.constexpr,
    UPCAST_INDICES: gl.constexpr = False,
):
    #TODO add gl.assume

    MX_PACK_DIVISOR: gl.constexpr = 32
    w_type: gl.constexpr = W.dtype.element_ty
    #gl.static_assert(w_type == gl.uint8), "mx_weight_ptr must be uint8"
    #gl.static_assert(
    #    WMxScale.dtype.element_ty == gl.uint8, "mx_scale_ptr must be uint8"
    #)
    #gl.static_assert(
    #    BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR"
    #)
    OUT_BLOCK_N: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = gl.program_id(0)    
    index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32

    if XCD_SWIZZLE != 1:
        padding_m = grid_m - gl.load(ExptOffsSum)
        unpadded_m = grid_m - padding_m
        total_actual_tiles = unpadded_m * grid_n
        if padding_m > 0 and pid >= total_actual_tiles:
            return
        pid = remap_xcd(pid, total_actual_tiles, XCD_SWIZZLE)
    else:
        unpadded_m = grid_m

    pid_m, pid_n = pid_grid(pid, unpadded_m, grid_n, 1)

    expt_data = gl.load(ExptData + pid_m)
    if XCD_SWIZZLE == 1 and expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = gl.load(ExptHist + expt_id)
    start_m = gl.load(ExptOffs + expt_id)
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m = start_m.to(index_type)
    pid_n = pid_n.to(index_type)
    GLOBAL_X_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0]
    )
    GLOBAL_W_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[16, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0]
    )
    GLOBAL_WS_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0]
    )

    # X / gather offsets
    if GatherIndx is None:
        X += start_m * stride_x_m
        offs_x_m_scalar = BLOCK_M * block_id + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, GLOBAL_X_LAYOUT))
        # Wrap within this expert's [0, M) rows so blocks that extend past the
        # expert's token count don't read out of bounds (rows past M are masked
        # out at store time). Mirrors the `offs_x_m % M` guard in the Triton kernel.
        offs_x_m = offs_x_m_scalar % M
    else:
        if GatherIndx.dtype.element_ty == gl.uint16:
            IDX_LAYOUT: gl.constexpr = gl.SliceLayout(
                0, gl.BlockedLayout([1,16], [64,1], [1,num_warps], [0,1])
            )
            oob_idx = (num_tokens).to(gl.uint16)
        else:
            gl.static_assert(
                GatherIndx.dtype.element_ty == gl.int32,
                "Gather index datatype should be uint16 or int32", 
            )
            IDX_LAYOUT: gl.constexpr = gl.SliceLayout(
                0, gl.BlockedLayout([1,8], [64, 1], [1, num_warps], [0,1])
            )
            oob_idx = num_tokens
        
        offs_x_m = BLOCK_M * block_id + gl.arange(0, BLOCK_M, layout=IDX_LAYOUT)
        mask_idx = offs_x_m < M
        offs_x_m = offs_x_m % M
        GatherIndx += start_m
        offs_x_m = gl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
        offs_x_m = gl.where(mask_idx, offs_x_m, oob_idx)
        offs_x_m = gl.convert_layout(offs_x_m, gl.SliceLayout(1, GLOBAL_X_LAYOUT))
    offs_x_k = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, GLOBAL_X_LAYOUT))
    x_offsets = offs_x_m[:, None]*stride_x_m + offs_x_k[None, :]*stride_x_k
    XPtrs = (X + x_offsets)

    W_K_DIVISOR: gl.constexpr = 2  # fp4: two values packed per uint8 along K
    W_N_DIVISOR: gl.constexpr = 1
    PACKED_BLOCK_K_W: gl.constexpr = BLOCK_K // W_K_DIVISOR
    PACKED_BLOCK_N_W: gl.constexpr = BLOCK_N // W_N_DIVISOR
    MX_SCALE_BLOCK_K: gl.constexpr = BLOCK_K // MX_PACK_DIVISOR

    #W pointers
    # Wrap along N so a block extending past N (BLOCK_N need not divide N) does
    # not read out of bounds; the extra columns are masked out at store time.
    offs_w_n = (pid_n * PACKED_BLOCK_N_W + gl.arange(0, PACKED_BLOCK_N_W, gl.SliceLayout(1, GLOBAL_W_LAYOUT))) % (N // W_N_DIVISOR)
    offs_w_k = gl.arange(0, PACKED_BLOCK_K_W, gl.SliceLayout(0, GLOBAL_W_LAYOUT))
    W += expt_id * stride_w_e
    WPtrs = W + (
        offs_w_k[None, :] * stride_w_k +
        offs_w_n[:, None] * stride_w_n
    )

    #W scale pointers
    WMxScale += expt_id * stride_w_mx_e
    if SWIZZLE_MX_SCALE == "GFX1250_SCALE":
        gl.static_assert(stride_w_mx_k is not None)
        gl.static_assert(stride_w_mx_n is not None)
        PRESHUFFLE_FACTOR: gl.constexpr = 32
        PACKED_MX_BLOCK: gl.constexpr = MX_SCALE_BLOCK_K * PRESHUFFLE_FACTOR
        SCALE_BLOCK_N: gl.constexpr = BLOCK_N // PRESHUFFLE_FACTOR
        #SCALE_KWIDTH: gl.constexpr = 8
    else:
        PRESHUFFLE_FACTOR: gl.constexpr = 1
        PACKED_MX_BLOCK: gl.constexpr = MX_SCALE_BLOCK_K
        SCALE_BLOCK_N: gl.constexpr = BLOCK_N
    offs_w_n_scale = (pid_n * SCALE_BLOCK_N + gl.arange(0, SCALE_BLOCK_N, gl.SliceLayout(1, GLOBAL_WS_LAYOUT))) % N
    offs_w_k_scale = gl.arange(0, PACKED_MX_BLOCK, gl.SliceLayout(0, GLOBAL_WS_LAYOUT))
    WMxScalePtrs = (
        WMxScale +
        offs_w_k_scale[None, :] * stride_w_mx_k +
        offs_w_n_scale[:, None] * stride_w_mx_n 
    )

    #MFMA Layout - instruction [16,16,32]
    #Dot layout X and W
    MFMA_LAYOUT: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )

    DOT_LAYOUT_X: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MFMA_LAYOUT, k_width=8)
    DOT_LAYOUT_W: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA_LAYOUT, k_width=8)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=MFMA_LAYOUT)

    num_k_iter = tl.cdiv(K, BLOCK_K)
    for k in range(num_k_iter):
        # Mask against the true global K offset for this iteration so that a
        # full block is loaded whenever it lies entirely within K (including
        # the case where BLOCK_K evenly divides K), and only the trailing
        # partial block is masked.
        k_base = k * BLOCK_K
        #load x
        mask_x = (k_base + offs_x_k < K)[None, :]
        mask_w = (k_base // W_K_DIVISOR + offs_w_k < K // W_K_DIVISOR)[None, :]
        x = gl.load(XPtrs, mask=mask_x, other=0.0)
        #load w
        w = gl.load(WPtrs, mask=mask_w, other=0.0)
        #load w_scales
        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            w_scales = gl.load(WMxScalePtrs)
            w_scales = unswizzle_mx_scale_cdna4(w_scales, BLOCK_N, MX_SCALE_BLOCK_K)
        else:
            mask_w_k_scale = (k_base + offs_w_k_scale * MX_PACK_DIVISOR < K)[None, :]
            w_scales = gl.load(WMxScalePtrs, mask=mask_w_k_scale, other=0.0)
        #convert weight scales layout
        w_scales = (
            w_scales.reshape((BLOCK_N, MX_SCALE_BLOCK_K, 1))
            .broadcast_to((BLOCK_N, MX_SCALE_BLOCK_K, MX_PACK_DIVISOR))
            .reshape((BLOCK_N, MX_SCALE_BLOCK_K * MX_PACK_DIVISOR))
        )
        w_scales = gl.convert_layout(w_scales, GLOBAL_WS_LAYOUT)
        #upcast w to bf16
        w_bf16 = gl.amd.cdna4.scaled_upcast(w, w_scales, gl.bfloat16, axis=1)

        #convert weight layout
        x = gl.convert_layout(x, DOT_LAYOUT_X)
        w_kn = gl.convert_layout(w_bf16.trans(1,0), DOT_LAYOUT_W)

        #acc = mfma
        acc = gl.amd.cdna4.mfma(x, w_kn, acc)

        #advance K pointers for the next block
        XPtrs += BLOCK_K * stride_x_k
        WPtrs += PACKED_BLOCK_K_W * stride_w_k
        WMxScalePtrs += PACKED_MX_BLOCK * stride_w_mx_k

    #Store Y
    GLOBAL_STORE_LAYOUT_Y: gl.constexpr = MFMA_LAYOUT
    offs_y_m = BLOCK_M * block_id + gl.arange(0, BLOCK_M, gl.SliceLayout(1, GLOBAL_STORE_LAYOUT_Y))
    offs_y_n = BLOCK_N * pid_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, GLOBAL_STORE_LAYOUT_Y))
    mask_m = offs_y_m < M
    mask_n = offs_y_n < N
    #Bias
    if B is not None:
        BPtrs = B + expt_id * stride_b_e + offs_y_n
        bias = gl.load(BPtrs, mask=mask_n, other=0.0)
        out = acc + bias[None, :]
    else:
        out = acc
    Y += start_m * stride_y_m
    YPtrs = (
        Y +
        offs_y_m[:, None] * stride_y_m +
        offs_y_n[None, :] * stride_y_n 
    )
    mask = mask_m[:, None] * mask_n[None, :]
    gl.store(YPtrs, out, mask=mask)
    


    #TODO
    #Packed and  Unpacked layout
    #PACKED_LAYOUT: gl.constexpr = gl.BlockedLayout(
    #    size_per_thread=[0, 4],
    #    threads_per_warp=[15, 4],
    #    warps_per_cta=[num_warps, 0],
    #    order=[0,0]
    #)

    #UNPACKED_LAYOUT: gl.constexpr = gl.BlockedLayout(
    #    size_per_thread=[0, 8],
    #    threads_per_warp=[15, 4],
    #    warps_per_cta=[num_warps, 0],
    #    order=[0,0]
    #)


    #Shared Layout X
    #Shared Layout W
    #Shared Layout W scales
    #Shread Layout Y
    
    # x_buffer
    # w_buffer
    # ws_buffer

    #produce and consumer counters

    #Prologue
    #Load NUM_BUFFERS - 1
    #async load x
    #async load w
    #async load ws
    #increment producer 
    #commit group

    #Count iterations
    #accmulator

    #Loop
    #async load x
    #async load w
    #async load ws
    #increment producer 

    #async wait group
    #load from smem->reg

    #unswizzle scales
    #convert weight scales layout
    #update w to bf16
    #convert weight layout

    #acc = mfma
    #consumer += 1

    #Epilogue
    #async wait_group
    #load from smem->reg
    #unswizzle scales
    #convert weight scales layout
    #update w to bf16
    #convert weight layout

    #acc = mfma
    #consumer += 1

#Bias
#Swiglu
#Gammas
#Store y
     
    