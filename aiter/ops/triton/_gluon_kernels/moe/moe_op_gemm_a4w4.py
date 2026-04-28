import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
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
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: gl.constexpr = 1):
    """
    Maps 1D pid to 2D grid coords (pid_m, pid_n).

    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - GROUP_SIZE_M: default is 1
    """
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        gl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@gluon.jit(launch_metadata=matmul_launch_metadata)
def _moe_gemm_a4w4_gfx1250(
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
    stride_w_k,
    stride_w_n,
    WMxScale,
    stride_w_mx_e,
    stride_w_mx_k,
    stride_w_mx_n,
    # bias
    B,
    stride_b_e,
    Gammas,
    # shapes
    num_tokens,
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
    XCD_SWIZZLE: gl.constexpr,
    SPLIT_K: gl.constexpr,
    SWIZZLE_MX_SCALE: gl.constexpr, # "GFX1250_SCALE" | None
    NUM_BUFFERS: gl.constexpr,
    UPCAST_INDICES: gl.constexpr,
    # layouts
    WMMA_LAYOUT: gl.constexpr,
    WMMA_LAYOUT_PACKED: gl.constexpr,
    # triton configs
    NUM_WARPS: gl.constexpr,
):
    gl.assume(stride_y_k >= 0)
    gl.assume(stride_y_m >= 0)
    gl.assume(stride_y_n >= 0)
    gl.assume(stride_x_m >= 0)
    gl.assume(stride_x_k >= 0)
    gl.assume(stride_w_e >= 0)
    gl.assume(stride_w_k >= 0)
    gl.assume(stride_w_n >= 0)
    if stride_x_mx_m is not None:
        gl.assume(stride_x_mx_m >= 0)
    if stride_x_mx_k is not None:
        gl.assume(stride_x_mx_k >= 0)
    if stride_w_mx_e is not None:
        gl.assume(stride_w_mx_e >= 0)
    if stride_w_mx_k is not None:
        gl.assume(stride_w_mx_k >= 0)
    if stride_w_mx_n is not None:
        gl.assume(stride_w_mx_n >= 0)
    if B is not None:
        gl.assume(stride_b_e >= 0)
    gl.assume(grid_m >= 0)
    gl.assume(grid_n >= 0)

    MX_PACK_DIVISOR: gl.constexpr = 32
    gl.static_assert(
        BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR"
    )

    NUM_LOADS_IN_BATCH: gl.constexpr = 2
    gl.static_assert(NUM_BUFFERS >= 3, "NUM_BUFFERS must be at least 3")

    w_type: gl.constexpr = W.dtype.element_ty
    gl.static_assert(w_type == gl.uint8, "mx_weight_ptr must be uint8 or fp8")
    gl.static_assert(
        WMxScale.dtype.element_ty == gl.uint8, "mx_scale_ptr must be uint8"
    )
    x_type: gl.constexpr = X.dtype.element_ty
    gl.static_assert(x_type == gl.uint8, "mx_act_ptr must be uint8")
    gl.static_assert(
        XMxScale.dtype.element_ty == gl.uint8, "mx_scale_ptr must be uint8"
    )

    OUT_BLOCK_N: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    # get program id
    pid = gl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        # Determine how much padding there is on the expert data. This allows us to
        # know the true grid size and avoid processing padding tiles.
        padding_m = grid_m - gl.load(ExptOffsSum)
    else:
        padding_m: gl.constexpr = 0

    index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32

    # get unpadded grid size
    unpadded_m = grid_m - padding_m
    gl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    # swizzle program ids
    pid_mn = pid % (unpadded_m * grid_n)
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, 1)

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
    pid_n = pid_n.to(index_type)

    # get the packed block sizes
    # both A and B tensors are mxfp4
    #   2 MXFP4 elements are packed into 1 int8
    #   in the K dimension
    X_M_DIVISOR: gl.constexpr = 1
    X_K_DIVISOR: gl.constexpr = 2 # 2 MXFP4 elements packed into 1 byte
    W_K_DIVISOR: gl.constexpr = 2 # 2 MXFP4 elements packed into 1 byte
    W_N_DIVISOR: gl.constexpr = 1
    PACKED_BLOCK_M_X: gl.constexpr = BLOCK_M // X_M_DIVISOR
    PACKED_BLOCK_K_X: gl.constexpr = BLOCK_K // X_K_DIVISOR
    PACKED_BLOCK_K_W: gl.constexpr = BLOCK_K // W_K_DIVISOR
    PACKED_BLOCK_N_W: gl.constexpr = BLOCK_N // W_N_DIVISOR
    MX_SCALE_BLOCK_K: gl.constexpr = BLOCK_K // MX_PACK_DIVISOR # 32 elements share 1 scale element

    # wmma layouts
    DOT_LAYOUT_X: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=WMMA_LAYOUT_PACKED, k_width=16
    )
    DOT_LAYOUT_W: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT_PACKED, k_width=16
    )
    DOT_LAYOUT_X_SCALES: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        DOT_LAYOUT_X, [PACKED_BLOCK_M_X, MX_SCALE_BLOCK_K]
    )
    DOT_LAYOUT_W_SCALES: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        DOT_LAYOUT_W, [PACKED_BLOCK_N_W, MX_SCALE_BLOCK_K]
    )

    # A pointers
    offs_x_m = PACKED_BLOCK_M_X * block_id
    if GatherIndx is None:
        X += start_m * stride_x_m
    else:
        IDX_LAYOUT: gl.constexpr = gl.SliceLayout(
            0, gl.BlockedLayout([1, 8], [32, 1], [1, NUM_WARPS], [0, 1])
        )
        offs_x_m = PACKED_BLOCK_M_X * block_id + gl.arange(
            0, PACKED_BLOCK_M_X, layout=IDX_LAYOUT
        )
        GatherIndx += start_m
        offs_x_m = gl.amd.gfx1250.buffer_load(GatherIndx, offs_x_m) // N_EXPTS_ACT

    # B pointers
    offs_w_n = pid_n * PACKED_BLOCK_N_W
    W += expt_id * stride_w_e

    # A scale pointers
    if GatherIndx is None:
        XMxScale += start_m * stride_x_mx_m
        offs_x_m_scale = offs_x_m + gl.arange(0, PACKED_BLOCK_M_X, layout=gl.SliceLayout(1, DOT_LAYOUT_X_SCALES))
    else:
        offs_x_m_scale = gl.convert_layout(offs_x_m, gl.SliceLayout(1, DOT_LAYOUT_X_SCALES))
    offs_x_k_scale = gl.arange(0, MX_SCALE_BLOCK_K, layout=gl.SliceLayout(0, DOT_LAYOUT_X_SCALES))
    offs_x_scale = offs_x_m_scale.to(index_type)[:, None] * stride_x_mx_m + offs_x_k_scale.to(index_type)[None, :] * stride_x_mx_k

    # B scale pointers
    WMxScale += expt_id * stride_w_mx_e
    if SWIZZLE_MX_SCALE == "GFX1250_SCALE":
        gl.static_assert(stride_w_mx_k is not None)
        gl.static_assert(stride_w_mx_n is not None)
        SCALE_KWIDTH: gl.constexpr = 4 if MX_SCALE_BLOCK_K >= 4 else MX_SCALE_BLOCK_K
        PRESHUFFLE_FACTOR: gl.constexpr = 128
        PACKED_MX_BLOCK: gl.constexpr = MX_SCALE_BLOCK_K * PRESHUFFLE_FACTOR
        SCALE_BLOCK_N: gl.constexpr = BLOCK_N // PRESHUFFLE_FACTOR
        # unshuffle offsets:
        # .reshape(
        #     (
        #         SCALE_BLOCK_N,
        #         MX_SCALE_BLOCK_K // SCALE_KWIDTH,
        #         N_PRESHUFFLE_FACTOR // 4,
        #         4,
        #         SCALE_KWIDTH,
        #     )
        # )
        # .permute((0, 3, 2, 1, 4))
        # .reshape((BLOCK_N, MX_SCALE_BLOCK_K))
        row = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, DOT_LAYOUT_W_SCALES))
        col = gl.arange(0, MX_SCALE_BLOCK_K, layout=gl.SliceLayout(0, DOT_LAYOUT_W_SCALES))
        d0_w_scales = row // PRESHUFFLE_FACTOR
        d3_w_scales = (row % PRESHUFFLE_FACTOR) // (PRESHUFFLE_FACTOR // 4)
        d2_w_scales = row % (PRESHUFFLE_FACTOR // 4)
        d1_w_scales = col // SCALE_KWIDTH
        d4_w_scales = col % SCALE_KWIDTH
        offs_w_scale = (
            (pid_n * SCALE_BLOCK_N + d0_w_scales.to(index_type))[:, None] * stride_w_mx_n
            + (d1_w_scales.to(index_type)[None, :] * (PRESHUFFLE_FACTOR * SCALE_KWIDTH)
            + d2_w_scales.to(index_type)[:, None] * (4 * SCALE_KWIDTH)
            + d3_w_scales.to(index_type)[:, None] * SCALE_KWIDTH
            + d4_w_scales.to(index_type)[None, :]) * stride_w_mx_k
        )
    else:
        PRESHUFFLE_FACTOR: gl.constexpr = 1
        PACKED_MX_BLOCK: gl.constexpr = MX_SCALE_BLOCK_K
        SCALE_BLOCK_N: gl.constexpr = BLOCK_N
        offs_w_n_scale = pid_n * SCALE_BLOCK_N + gl.arange(0, SCALE_BLOCK_N, layout=gl.SliceLayout(1, DOT_LAYOUT_W_SCALES))
        offs_w_k_scale = gl.arange(0, PACKED_MX_BLOCK, layout=gl.SliceLayout(0, DOT_LAYOUT_W_SCALES))
        offs_w_scale = offs_w_n_scale.to(index_type)[:, None] * stride_w_mx_n + offs_w_k_scale.to(index_type)[None, :] * stride_w_mx_k

    # shared layouts
    if PACKED_BLOCK_K_X <= 256:
        SHARED_LAYOUT_X: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[256, 16]], [PACKED_BLOCK_M_X, PACKED_BLOCK_K_X], [1, 0]
        )
    else:
        SHARED_LAYOUT_X: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[PACKED_BLOCK_K_X, 16]], [PACKED_BLOCK_M_X, PACKED_BLOCK_K_X], [1, 0]
        )
    if PACKED_BLOCK_K_W <= 256:
        SHARED_LAYOUT_W: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[256, 16]], [PACKED_BLOCK_N_W, PACKED_BLOCK_K_W], [1, 0]
        )
    else:
        SHARED_LAYOUT_W: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[PACKED_BLOCK_K_W, 16]], [PACKED_BLOCK_N_W, PACKED_BLOCK_K_W], [1, 0]
        )

    if GatherIndx is None:
        x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=X,
            shape=(M, K // X_K_DIVISOR),
            strides=(stride_x_m, stride_x_k),
            block_shape=(PACKED_BLOCK_M_X, PACKED_BLOCK_K_X),
            layout=SHARED_LAYOUT_X,
        )
    else:
        x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=X,
            shape=(num_tokens, K // X_K_DIVISOR),
            strides=(stride_x_m, stride_x_k),
            block_shape=(PACKED_BLOCK_M_X, PACKED_BLOCK_K_X),
            layout=SHARED_LAYOUT_X,
        )
    w_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=W,
        shape=(N, K // W_K_DIVISOR),
        strides=(stride_w_n, stride_w_k),
        block_shape=(PACKED_BLOCK_N_W, PACKED_BLOCK_K_W),
        layout=SHARED_LAYOUT_W,
    )

    x_buffer = gl.allocate_shared_memory(
        x_desc.dtype, shape=[NUM_BUFFERS] + x_desc.block_shape, layout=x_desc.layout
    )
    w_buffer = gl.allocate_shared_memory(
        w_desc.dtype, shape=[NUM_BUFFERS] + w_desc.block_shape, layout=w_desc.layout
    )

    load_idx = 0
    wmma_idx = 0
    scale_idx = 0

    # prologue: fill NUM_BUFFERS-1 LDS slots via TDM
    for _ in gl.static_range(NUM_BUFFERS - 1):
        if GatherIndx is None:
            gl.amd.gfx1250.tdm.async_load(
                x_desc,
                [offs_x_m.to(index_type), load_idx * PACKED_BLOCK_K_X],
                x_buffer.index(load_idx % NUM_BUFFERS),
            )
        else:
            gl.amd.gfx1250.tdm.async_gather(
                x_desc,
                offs_x_m.to(index_type),
                load_idx * PACKED_BLOCK_K_X,
                x_buffer.index(load_idx % NUM_BUFFERS),
            )
        gl.amd.gfx1250.tdm.async_load(
            w_desc,
            [offs_w_n.to(index_type), load_idx * PACKED_BLOCK_K_W],
            w_buffer.index(load_idx % NUM_BUFFERS),
        )
        load_idx += 1

    # preload tile 0 from LDS into registers
    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * NUM_LOADS_IN_BATCH)
    cur_x = x_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=DOT_LAYOUT_X)
    cur_w = (
        w_buffer.index(wmma_idx % NUM_BUFFERS)
        .permute((1, 0))
        .load(layout=DOT_LAYOUT_W)
    )
    wmma_idx += 1
    cur_x_scales = gl.amd.gfx1250.buffer_load(XMxScale, offs_x_scale)
    cur_w_scales = gl.amd.gfx1250.buffer_load(WMxScale, offs_w_scale)
    scale_idx += 1

    # main loop: perform wmma and fill LDS with next tile
    num_k_iter = gl.cdiv(K, BLOCK_K)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    for col in range(num_k_iter - (NUM_BUFFERS - 1)):
        # issue wmma
        acc = gl.amd.gfx1250.wmma_scaled(cur_x, cur_x_scales, "e2m1", cur_w, cur_w_scales, "e2m1", acc)

        # fill next tile to LDS
        if GatherIndx is None:
            gl.amd.gfx1250.tdm.async_load(
                x_desc,
                [offs_x_m.to(index_type), load_idx * PACKED_BLOCK_K_X],
                x_buffer.index(load_idx % NUM_BUFFERS),
            )
        else:
            gl.amd.gfx1250.tdm.async_gather(
                x_desc,
                offs_x_m.to(index_type),
                load_idx * PACKED_BLOCK_K_X,
                x_buffer.index(load_idx % NUM_BUFFERS),
            )
        gl.amd.gfx1250.tdm.async_load(
            w_desc,
            [offs_w_n.to(index_type), load_idx * PACKED_BLOCK_K_W],
            w_buffer.index(load_idx % NUM_BUFFERS),
        )
        load_idx += 1

        # wait for next tile to be filled
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * NUM_LOADS_IN_BATCH)

        # load next tile from LDS into registers
        next_x = x_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=DOT_LAYOUT_X)
        next_w = (
            w_buffer.index(wmma_idx % NUM_BUFFERS)
            .permute((1, 0))
            .load(layout=DOT_LAYOUT_W)
        )
        wmma_idx += 1
        next_x_scales = gl.amd.gfx1250.buffer_load(XMxScale, offs_x_scale + scale_idx * MX_SCALE_BLOCK_K * stride_x_mx_k)
        next_w_scales = gl.amd.gfx1250.buffer_load(WMxScale, offs_w_scale + scale_idx * PACKED_MX_BLOCK * stride_w_mx_k)
        scale_idx += 1

        # prepare next iteration
        cur_x = next_x
        cur_w = next_w
        cur_x_scales = next_x_scales
        cur_w_scales = next_w_scales

    # epilogue: drain remaining tiles
    for k_ep in gl.static_range(NUM_BUFFERS - 2):
        # issue wmma
        acc = gl.amd.gfx1250.wmma_scaled(cur_x, cur_x_scales, "e2m1", cur_w, cur_w_scales, "e2m1", acc)

        # wait for next tile to be filled
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 3 - k_ep) * NUM_LOADS_IN_BATCH)

        # load next tile from LDS into registers
        next_x = x_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=DOT_LAYOUT_X)
        next_w = (
            w_buffer.index(wmma_idx % NUM_BUFFERS)
            .permute((1, 0))
            .load(layout=DOT_LAYOUT_W)
        )
        wmma_idx += 1
        next_x_scales = gl.amd.gfx1250.buffer_load(XMxScale, offs_x_scale + scale_idx * MX_SCALE_BLOCK_K * stride_x_mx_k)
        next_w_scales = gl.amd.gfx1250.buffer_load(WMxScale, offs_w_scale + scale_idx * PACKED_MX_BLOCK * stride_w_mx_k)
        scale_idx += 1

        # prepare next iteration
        cur_x = next_x
        cur_w = next_w
        cur_x_scales = next_x_scales
        cur_w_scales = next_w_scales

    # issue last wmma
    acc = gl.amd.gfx1250.wmma_scaled(cur_x, cur_x_scales, "e2m1", cur_w, cur_w_scales, "e2m1", acc)

    # bias
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

    # apply activation function
    if APPLY_SWIGLU and SPLIT_K == 1:
        out = _swiglu(acc, alpha, limit, ADD_RESIDUAL)
        out = gl.convert_layout(out, WMMA_LAYOUT)
        gl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
        offs_y_n = OUT_BLOCK_N * pid_n + gl.arange(
            0, OUT_BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
        )
        mask_n = offs_y_n < yN
    else:
        gl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )
        out = acc

    # apply gammas
    if Gammas is not None:
        gammas = gl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        out *= gammas[:, None]

    # write-back
    Y += start_m * stride_y_m
    offs_y_m = offs_m
    offs_y = (
        offs_y_m.to(index_type)[:, None] * stride_y_m
        + offs_y_n.to(index_type)[None, :] * stride_y_n
    )
    mask = mask_m[:, None] & mask_n[None, :]
    out = out.to(gl.bfloat16)
    gl.amd.gfx1250.buffer_store(out, Y, offs_y, mask=mask)
