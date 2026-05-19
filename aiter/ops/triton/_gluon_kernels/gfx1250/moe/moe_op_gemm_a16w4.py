import torch

# import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid
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

    gindx = args.get("GatherIndx", None)
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
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: gl.constexpr):
    """
    Swizzle the program id based on integer XCD_SWIZZLE.
    """
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE
    new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
    return new_pid


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
    gl.static_assert(
        SWIZZLE_MX_SCALE is None,
        "Gluon a16w4 path requires pre-expanded scales (SWIZZLE_MX_SCALE=None)",
    )

    gl.assume(stride_y_k >= 0)
    gl.assume(stride_y_m >= 0)
    gl.assume(stride_y_n >= 0)
    gl.assume(stride_x_m >= 0)
    gl.assume(stride_x_k >= 0)
    gl.assume(stride_w_e >= 0)
    gl.assume(stride_w_k >= 0)
    gl.assume(stride_w_n >= 0)
    gl.assume(stride_w_mx_e >= 0)
    gl.assume(stride_w_mx_k >= 0)
    gl.assume(stride_w_mx_n >= 0)
    if B is not None:
        gl.assume(stride_b_e >= 0)
    gl.assume(grid_m >= 0)
    gl.assume(grid_n >= 0)

    MX_PACK_DIVISOR: gl.constexpr = 32
    NUM_TDM_OPS: gl.constexpr = 3  # X, W (fp4 packed), W_scale (e8m0 expanded)
    w_type: gl.constexpr = W.dtype.element_ty
    gl.static_assert(w_type == gl.uint8, "mx_weight_ptr must be uint8")
    gl.static_assert(
        WMxScale.dtype.element_ty == gl.uint8, "mx_scale_ptr must be uint8"
    )
    gl.static_assert(
        BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR"
    )
    gl.static_assert(num_warps == 4 or num_warps == 8, "num_warps must be 4 or 8")

    OUT_BLOCK_N: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = gl.program_id(0)

    padding_m: gl.constexpr = 0
    index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32

    unpadded_m = grid_m - padding_m
    gl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    pid_emnk = pid
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

    # X / gather offsets
    offs_x_m_scalar = BLOCK_M * block_id
    if GatherIndx is None:
        X += start_m * stride_x_m
        offs_x_m = offs_x_m_scalar  # unused in non-gather path
    else:
        IDX_LAYOUT: gl.constexpr = gl.SliceLayout(
            0, gl.BlockedLayout([1, 8], [32, 1], [1, num_warps], [0, 1])
        )
        offs_x_m = BLOCK_M * block_id + gl.arange(0, BLOCK_M, layout=IDX_LAYOUT)
        GatherIndx += start_m
        # No need to bounds-check: `offs_x_m` wraps around M dim.
        offs_x_m = gl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT

    W_K_DIVISOR: gl.constexpr = 2  # fp4: two values packed per uint8 along K
    W_N_DIVISOR: gl.constexpr = 1
    PACKED_BLOCK_K_W: gl.constexpr = BLOCK_K // W_K_DIVISOR
    PACKED_BLOCK_N_W: gl.constexpr = BLOCK_N // W_N_DIVISOR

    off_w_n = pid_n * PACKED_BLOCK_N_W
    # WMxScale is the pre-expanded e8m0 tensor of shape (E, K, N); we read it as (N, K).
    off_w_n_scale = pid_n * BLOCK_N

    W += expt_id * stride_w_e
    WMxScale += expt_id * stride_w_mx_e

    # WMMA layout for plain bf16 x bf16: instr_shape [16, 16, 32], k_width=8.
    if num_warps == 4:
        WARP_BASES: gl.constexpr = [[0, 1], [1, 0]]
    else:
        WARP_BASES: gl.constexpr = [[0, 1], [0, 2], [1, 0]]
    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=WARP_BASES,
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    DOT_LAYOUT_X: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=WMMA_LAYOUT, k_width=8
    )
    DOT_LAYOUT_W: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=8
    )

    # Blocked layouts for fp4-packed W (BLOCK_N, BLOCK_K // 2) and its expanded e8m0
    # scale (BLOCK_N, BLOCK_K). size_per_thread along K doubles for the scale layout
    # so the unpack along K lines up element-for-element with the scale.
    # threads_per_warp = [8, 4] = 32 (wave32 on gfx1250).
    PACKED_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[8, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    UNPACKED_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    SHARED_LAYOUT_X: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 16]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    SHARED_LAYOUT_W: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[PACKED_BLOCK_K_W, 16]], [BLOCK_N, PACKED_BLOCK_K_W], [1, 0]
    )
    SHARED_LAYOUT_W_SCALES: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 16]], [BLOCK_N, BLOCK_K], [1, 0]
    )

    if GatherIndx is None:
        x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=X,
            shape=(M, K),
            strides=(stride_x_m, stride_x_k),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_LAYOUT_X,
        )
    else:
        x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=X,
            shape=(num_tokens, K),
            strides=(stride_x_m, stride_x_k),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_LAYOUT_X,
        )

    w_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=W,
        shape=(N, K // W_K_DIVISOR),
        strides=(stride_w_n, stride_w_k),
        block_shape=(BLOCK_N, PACKED_BLOCK_K_W),
        layout=SHARED_LAYOUT_W,
    )

    ws_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=WMxScale,
        shape=(N, K),
        strides=(stride_w_mx_n, stride_w_mx_k),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=SHARED_LAYOUT_W_SCALES,
    )

    x_buffer = gl.allocate_shared_memory(
        x_desc.dtype, shape=[NUM_BUFFERS] + x_desc.block_shape, layout=x_desc.layout
    )
    w_buffer = gl.allocate_shared_memory(
        w_desc.dtype, shape=[NUM_BUFFERS] + w_desc.block_shape, layout=w_desc.layout
    )
    ws_buffer = gl.allocate_shared_memory(
        ws_desc.dtype,
        shape=[NUM_BUFFERS] + ws_desc.block_shape,
        layout=ws_desc.layout,
    )

    producer = 0
    consumer = 0

    # Prologue: prime NUM_BUFFERS - 1 tile loads (X, W-packed, W-scale-expanded).
    for _ in gl.static_range(NUM_BUFFERS - 1):
        idx = producer % NUM_BUFFERS
        k_off = producer * BLOCK_K
        kp_off = producer * PACKED_BLOCK_K_W
        if GatherIndx is None:
            gl.amd.gfx1250.tdm.async_load(
                x_desc, [offs_x_m_scalar, k_off], x_buffer.index(idx)
            )
        else:
            gl.amd.gfx1250.tdm.async_gather(
                x_desc, offs_x_m, k_off, x_buffer.index(idx)
            )
        gl.amd.gfx1250.tdm.async_load(w_desc, [off_w_n, kp_off], w_buffer.index(idx))
        gl.amd.gfx1250.tdm.async_load(
            ws_desc, [off_w_n_scale, k_off], ws_buffer.index(idx)
        )
        producer += 1

    num_k_iter = tl.cdiv(K, BLOCK_K)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    # Steady state: each iteration issues 1 tile and consumes 1 tile.
    # NUM_BUFFERS - 1 tiles stay in flight.
    for _ in range(num_k_iter - (NUM_BUFFERS - 1)):
        # issue next tile
        idx = producer % NUM_BUFFERS
        k_off = producer * BLOCK_K
        kp_off = producer * PACKED_BLOCK_K_W
        if GatherIndx is None:
            gl.amd.gfx1250.tdm.async_load(
                x_desc, [offs_x_m_scalar, k_off], x_buffer.index(idx)
            )
        else:
            gl.amd.gfx1250.tdm.async_gather(
                x_desc, offs_x_m, k_off, x_buffer.index(idx)
            )
        gl.amd.gfx1250.tdm.async_load(w_desc, [off_w_n, kp_off], w_buffer.index(idx))
        gl.amd.gfx1250.tdm.async_load(
            ws_desc, [off_w_n_scale, k_off], ws_buffer.index(idx)
        )
        producer += 1

        # wait for the oldest in-flight tile, then consume it
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * NUM_TDM_OPS)
        c_idx = consumer % NUM_BUFFERS
        x_tile = x_buffer.index(c_idx).load(layout=DOT_LAYOUT_X)
        w_packed = w_buffer.index(c_idx).load(layout=PACKED_LAYOUT)
        w_scale = ws_buffer.index(c_idx).load(layout=UNPACKED_LAYOUT)
        # fp4 -> bf16 with per-32 e8m0 scale folded in. axis=1 means K doubles after unpack.
        w_bf16 = gl.amd.gfx1250.scaled_upcast(w_packed, w_scale, gl.bfloat16, axis=1)
        # (N, K) -> (K, N) for the B operand of WMMA, then move to the dot-operand layout.
        w_kn = gl.convert_layout(w_bf16.trans(1, 0), DOT_LAYOUT_W)
        acc = gl.amd.gfx1250.wmma(x_tile, w_kn, acc)
        consumer += 1

    # Epilogue: drain remaining NUM_BUFFERS - 1 tiles with a counting-down wait threshold.
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * NUM_TDM_OPS)
        c_idx = consumer % NUM_BUFFERS
        x_tile = x_buffer.index(c_idx).load(layout=DOT_LAYOUT_X)
        w_packed = w_buffer.index(c_idx).load(layout=PACKED_LAYOUT)
        w_scale = ws_buffer.index(c_idx).load(layout=UNPACKED_LAYOUT)
        w_bf16 = gl.amd.gfx1250.scaled_upcast(w_packed, w_scale, gl.bfloat16, axis=1)
        w_kn = gl.convert_layout(w_bf16.trans(1, 0), DOT_LAYOUT_W)
        acc = gl.amd.gfx1250.wmma(x_tile, w_kn, acc)
        consumer += 1

    # bias / activation / write-back (unchanged from prior version)
    offs_m = BLOCK_M * block_id + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT)
    )
    offs_y_n = BLOCK_N * pid_n + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
    )
    mask_m = offs_m < M
    mask_n = offs_y_n < N
    if B is not None:
        BPtrs = B + expt_id * stride_b_e
        bias = gl.amd.gfx1250.buffer_load(BPtrs, offs_y_n, mask=mask_n)
        acc = acc + bias[None, :]
    if APPLY_SWIGLU:
        out = _swiglu(acc, alpha, limit, ADD_RESIDUAL=ADD_RESIDUAL)
        tl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
        offs_m = BLOCK_M * block_id + gl.arange(0, BLOCK_M)
        offs_y_n = OUT_BLOCK_N * pid_n + gl.arange(0, OUT_BLOCK_N)
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

    Y += start_m * stride_y_m
    offs_y_m = offs_m
    offs_y = (
        offs_y_m.to(index_type)[:, None] * stride_y_m
        + offs_y_n.to(index_type)[None, :] * stride_y_n
    )
    mask = mask_m[:, None] & mask_n[None, :]
    gl.amd.gfx1250.buffer_store(out.to(Y.dtype.element_ty), Y, offs_y, mask=mask)
