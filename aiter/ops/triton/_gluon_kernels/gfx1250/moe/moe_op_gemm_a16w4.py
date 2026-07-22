import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd, pid_grid
from aiter.ops.triton._triton_kernels.moe.activations import _swiglu
from aiter.ops.triton._triton_kernels.moe.launch_metadata import (
    matmul_launch_metadata,
)


@gluon.jit
def unswizzle_mx_scale_gfx1250(
    scale, BLOCK_N, MX_SCALE_BLOCK_K, PRESHUFFLE_FACTOR, SCALE_KWIDTH, MX_PACK_DIVISOR
):
    # Invert the host-side preshuffle: the loaded tile packs (k0, n1, k1) along the
    # contiguous dim; reshape + permute reassembles the logical compact scale
    # (BLOCK_N, MX_SCALE_BLOCK_K), one byte per 32-elem group.
    scale = (
        scale.reshape(
            (
                BLOCK_N // PRESHUFFLE_FACTOR,
                MX_SCALE_BLOCK_K // SCALE_KWIDTH,
                PRESHUFFLE_FACTOR,
                SCALE_KWIDTH,
            )
        )
        .permute((0, 2, 1, 3))
        .reshape((BLOCK_N, MX_SCALE_BLOCK_K))
    )

    return scale


@gluon.jit
def _expand_mx_scale_k(scale, BLOCK_N: gl.constexpr, MX_SCALE_BLOCK_K: gl.constexpr):
    # scaled_upcast wants one e8m0 scale per unpacked element (BLOCK_N, BLOCK_K), not
    # the compact per-32-group scale. Broadcast each group scale MX_PACK_DIVISOR (=32)
    # times along K (stride-0) so out[n, g*32 + r] == scale[n, g].
    MX_PACK_DIVISOR: gl.constexpr = 32
    s = scale.reshape(BLOCK_N, MX_SCALE_BLOCK_K, 1)
    tgt = gl.full(
        (BLOCK_N, MX_SCALE_BLOCK_K, MX_PACK_DIVISOR), 0, gl.uint8, layout=s.type.layout
    )
    s, _ = gl.broadcast(s, tgt)
    return s.reshape(BLOCK_N, MX_SCALE_BLOCK_K * MX_PACK_DIVISOR)


@gluon.jit
def _tdm_load_tile(
    x_desc,
    w_desc,
    ws_desc,
    x_slot,
    w_slot,
    ws_slot,
    ki,
    GatherIndx,
    offs_x_m,
    offs_x_m_scalar,
    off_w_n,
    off_w_n_scale,
    BLOCK_K: gl.constexpr,
    PACKED_BLOCK_K_W: gl.constexpr,
    PACKED_MX_BLOCK: gl.constexpr,
):
    # Issue the 3 TDM async loads (X, packed-fp4 W, e8m0 scale) for K-tile `ki`.
    if GatherIndx is None:
        gl.amd.gfx1250.tdm.async_load(x_desc, [offs_x_m_scalar, ki * BLOCK_K], x_slot)
    else:
        # gl.amd.gfx1250.tdm.async_gather(x_desc, offs_x_m, ki * BLOCK_K, x_slot)
        x_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            x_desc, add_offsets=[0, ki * BLOCK_K], clamp_bounds=True
        )
        gl.amd.gfx1250.tdm.async_gather(x_desc, offs_x_m, x_slot)
    gl.amd.gfx1250.tdm.async_load(w_desc, [off_w_n, ki * PACKED_BLOCK_K_W], w_slot)
    gl.amd.gfx1250.tdm.async_load(
        ws_desc, [off_w_n_scale, ki * PACKED_MX_BLOCK], ws_slot
    )


@gluon.jit
def _preload_tile(
    w_slot,
    ws_slot,
    L_IN_W: gl.constexpr,
    L_SCALE_W: gl.constexpr,
    COMPACT_SCALE_LAYOUT: gl.constexpr,
    SWIZZLE_MX_SCALE: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    MX_SCALE_BLOCK_K: gl.constexpr,
    MX_PACK_DIVISOR: gl.constexpr,
    PRESHUFFLE_FACTOR: gl.constexpr,
    SCALE_KWIDTH: gl.constexpr,
    SCALE_REG_SHARE: gl.constexpr,
    SCALE_SEL: gl.constexpr,
):
    # LDS -> register bf16 W operand for one K-tile (fp4 unpack + scale applied).
    # Flip NO_MUL_UPCAST to route the compact scale through the software
    # scaled_upcast instead of the hardware v_cvt_scale_pk8 (see the elif below).
    NO_MUL_UPCAST: gl.constexpr = False
    w_packed = w_slot.permute([1, 0]).load(layout=L_IN_W)
    if SWIZZLE_MX_SCALE == "GFX1250_SCALE":
        ws_buffer_slice = unswizzle_mx_scale_gfx1250(
            ws_slot,
            BLOCK_N,
            MX_SCALE_BLOCK_K,
            PRESHUFFLE_FACTOR,
            SCALE_KWIDTH,
            MX_PACK_DIVISOR,
        )
        w_scale = ws_buffer_slice.load(layout=COMPACT_SCALE_LAYOUT)
        w_scale = _expand_mx_scale_k(w_scale, BLOCK_N, MX_SCALE_BLOCK_K)
        w_scale = gl.convert_layout(w_scale.trans(1, 0), layout=L_SCALE_W)
        # Software fp4->bf16 upcast with per-element expanded scale.
        w_kn = gl.amd.gfx1250.scaled_upcast(w_packed, w_scale, gl.bfloat16, axis=0)
    elif NO_MUL_UPCAST:
        # Reference no-mul path: software scaled_upcast on the COMPACT scale (x32
        # expand -> per-element scale -> software fp4->bf16). Correct for kWidth 8
        # and 16, but ~45% slower as a drop-in at the current kWidth=16 config: the
        # inner-loop x32 expansion + trans/convert_layout below are work the
        # hardware scale_upcast avoids. Disabled (NO_MUL_UPCAST=False); kept for
        # reference. The tuned kWidth=8 variant that drops the convert_layout lives
        # in git fa8e11e4b.
        w_scale = ws_slot.load(layout=COMPACT_SCALE_LAYOUT)
        w_scale = _expand_mx_scale_k(w_scale, BLOCK_N, MX_SCALE_BLOCK_K)
        w_scale = gl.convert_layout(w_scale.trans(1, 0), layout=L_SCALE_W)
        w_kn = gl.amd.gfx1250.scaled_upcast(w_packed, w_scale, gl.bfloat16, axis=0)
    else:
        # Hardware v_cvt_scale_pk8 upcast consuming the COMPACT e8m0 scale directly
        # -- no x32 expansion, no software upcast. k_scale = SCALE_REG_SHARE (=16):
        # each 32-group e8m0 covers DUP = 32/16 = 2 consecutive 16-blocks, duplicated
        # along K. Scale layout (L_SCALE_W) derived by stripping the k_scale
        # register-K identity. Raw e8m0 byte replicated across a uint32 so
        # scale_sel byte routing is a no-op.
        KW: gl.constexpr = SCALE_REG_SHARE
        DUP: gl.constexpr = MX_PACK_DIVISOR // KW
        SK: gl.constexpr = BLOCK_K // KW
        _o3 = gl.full((BLOCK_K, BLOCK_N), 0, gl.uint8, layout=L_SCALE_W).reshape(
            SK, KW, BLOCK_N
        )
        SCALE_L: gl.constexpr = gl.SliceLayout(1, _o3.type.layout)
        _sc3 = gl.full((SK, BLOCK_N), 0, gl.uint8, layout=SCALE_L).reshape(
            MX_SCALE_BLOCK_K, DUP, BLOCK_N
        )
        L_SC3: gl.constexpr = _sc3.type.layout
        e8 = ws_slot.permute([1, 0]).load(layout=gl.SliceLayout(1, L_SC3))
        e8 = gl.expand_dims(e8, 1)
        e8, _ = gl.broadcast(e8, _sc3)
        e8 = e8.reshape(SK, BLOCK_N)
        # Replicate the e8m0 byte across all 4 bytes of a uint32 (multiply by
        # 0x01010101; val <= 255 so no carry) so scale_sel byte routing is a no-op.
        s = e8.to(gl.uint32)
        w_scale = s * 0x01010101
        w_kn = gl.amd.gfx1250.scale_upcast(
            w_packed, w_scale, axis=0, scale_sel=SCALE_SEL, elem_type=gl.bfloat16
        )
    return w_kn


@gluon.jit
def _moe_gemm_a16w4_gluon_impl(
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
    WMxScale,  # E8M0 compact scale (one byte per 32 values along K)
    stride_w_mx_e,
    stride_w_mx_n,
    stride_w_mx_k,
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
    # Pass None; compact e8m0 scale. GFX1250_SCALE branch hangs the ROCm loader;
    # CDNA4_SCALE unsupported -- use the Triton kernel for swizzled scales.
    SWIZZLE_MX_SCALE: gl.constexpr,
    EVEN_K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
    num_warps: gl.constexpr,
    UPCAST_INDICES: gl.constexpr = False,
):
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

    # unpack expert data
    expt_data = gl.load(ExptData + pid_m)
    if XCD_SWIZZLE == 1 and expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = gl.load(ExptHist + expt_id)
    start_m = gl.load(ExptOffs + expt_id)
    # Keep block/expert/pid indices int32 so TDM tile offsets stay 32-bit; apply
    # .to(index_type) only at the int64 base-pointer arithmetic sites below.

    # X / gather offsets
    offs_x_m_scalar = BLOCK_M * block_id
    if GatherIndx is None:
        X += start_m.to(index_type) * stride_x_m
        offs_x_m = offs_x_m_scalar  # unused in non-gather path
    else:
        if GatherIndx.dtype.element_ty == gl.uint16:
            IDX_LAYOUT: gl.constexpr = gl.SliceLayout(
                0, gl.BlockedLayout([1, 16], [32, 1], [1, num_warps], [0, 1])
            )
            # num_tokens is a Python scalar; the later gl.where casts it to uint16.
            oob_idx = num_tokens
        else:
            gl.static_assert(
                GatherIndx.dtype.element_ty == gl.int32,
                "Gather index datatype should be uint16 or int32",
            )
            IDX_LAYOUT: gl.constexpr = gl.SliceLayout(
                0, gl.BlockedLayout([1, 8], [32, 1], [1, num_warps], [0, 1])
            )
            oob_idx = num_tokens

        offs_x_m = BLOCK_M * block_id + gl.arange(0, BLOCK_M, layout=IDX_LAYOUT)
        mask_idx = offs_x_m < M
        offs_x_m = offs_x_m % M
        GatherIndx += start_m
        offs_x_m = gl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
        offs_x_m = gl.where(mask_idx, offs_x_m, oob_idx)

    W_K_DIVISOR: gl.constexpr = 2  # fp4: two values packed per uint8 along K
    W_N_DIVISOR: gl.constexpr = 1
    PACKED_BLOCK_K_W: gl.constexpr = BLOCK_K // W_K_DIVISOR
    PACKED_BLOCK_N_W: gl.constexpr = BLOCK_N // W_N_DIVISOR
    MX_SCALE_BLOCK_K: gl.constexpr = BLOCK_K // MX_PACK_DIVISOR

    off_w_n = pid_n * PACKED_BLOCK_N_W

    W += expt_id.to(index_type) * stride_w_e
    WMxScale += expt_id.to(index_type) * stride_w_mx_e
    if SWIZZLE_MX_SCALE == "GFX1250_SCALE":
        gl.static_assert(stride_w_mx_k is not None)
        gl.static_assert(stride_w_mx_n is not None)
        PRESHUFFLE_FACTOR: gl.constexpr = 32
        PACKED_MX_BLOCK: gl.constexpr = MX_SCALE_BLOCK_K * PRESHUFFLE_FACTOR
        SCALE_BLOCK_N: gl.constexpr = BLOCK_N // PRESHUFFLE_FACTOR
        SCALE_KWIDTH: gl.constexpr = 8
    else:
        PRESHUFFLE_FACTOR: gl.constexpr = 1
        PACKED_MX_BLOCK: gl.constexpr = MX_SCALE_BLOCK_K
        SCALE_BLOCK_N: gl.constexpr = BLOCK_N
        # Unused on the compact path, but _preload_tile takes it unconditionally.
        SCALE_KWIDTH: gl.constexpr = 8

    # Scale tile offsets are in units of the scale descriptor's own blocking
    # (SCALE_BLOCK_N, PACKED_MX_BLOCK) -- NOT the weight's BLOCK_N / BLOCK_K.
    off_w_n_scale = pid_n * SCALE_BLOCK_N

    # WMMA layout for plain bf16 x bf16 (gluon wmma_scaled has no bf16 lhs, so a16w4
    # upcasts fp4->bf16 then uses plain WMMA). warp_bases are in 16x16 instr tiles:
    # [1, 0] steps a 16-row M-tile, [0, 1] a 16-col N-tile.
    if BLOCK_M == 16:
        # Decode: BLOCK_M is a single 16-row M-tile, so warps along M would
        # recompute rows and re-read the same W N-slice. Put every warp along N.
        # Valid since the decode config guarantees BLOCK_N >= num_warps * 16.
        gl.static_assert(
            BLOCK_N >= num_warps * 16,
            "decode warp-along-N layout requires BLOCK_N >= num_warps * 16",
        )
        if num_warps == 4:
            WARP_BASES: gl.constexpr = [[0, 1], [0, 2]]
        else:
            WARP_BASES: gl.constexpr = [[0, 1], [0, 2], [0, 4]]
    elif num_warps == 4:
        WARP_BASES: gl.constexpr = [[0, 1], [1, 0]]
    else:
        WARP_BASES: gl.constexpr = [[0, 1], [1, 0], [2, 0]]

    INSTR_K: gl.constexpr = 32
    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=WARP_BASES,
        reg_bases=[],
        instr_shape=[16, 16, INSTR_K],
    )
    K_PER_INSR: gl.constexpr = 16
    K_WIDTH: gl.constexpr = min(16, BLOCK_K // INSTR_K * K_PER_INSR)
    # scale_upcast k_scale = min(K_WIDTH, 32); K_WIDTH=16 is block16 (OPSEL 4).
    # Replicated scale bytes make the exact byte immaterial.
    SCALE_SEL: gl.constexpr = 4 if min(K_WIDTH, MX_PACK_DIVISOR) == 16 else 0
    DOT_LAYOUT_X: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=WMMA_LAYOUT, k_width=K_WIDTH
    )
    # scaled_upcast operand-aligned layouts: fp4 input k_width=K_WIDTH//2, scale
    # k_width=K_WIDTH. k_width doubles on unpack so the bf16 output matches the
    # WMMA B dot-operand layout directly -> WMMA B operand needs NO convert_layout
    # (saves 128 cross-lane v_permlanes/tile). Scale sits on the lanes the HW
    # broadcast reads.
    L_IN_W: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=K_WIDTH // 2
    )
    L_SCALE_W: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=K_WIDTH
    )

    # Parametric blocked layout for the compact (BLOCK_N, MX_SCALE_BLOCK_K) scale
    # tile (shape-agnostic, so BLOCK_K is tunable). _expand_mx_scale_k operates
    # on the logical values and the result is convert_layout'd to the upcast
    # output layout, so the exact blocked layout here only needs to tile the shape.
    COMPACT_SCALE_LAYOUT: gl.constexpr = gl.BlockedLayout(
        [1, 1], [8, 4], [num_warps, 1], [1, 0]
    )

    SHARED_LAYOUT_X: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    SHARED_LAYOUT_W: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[PACKED_BLOCK_K_W, 16]], [BLOCK_N, PACKED_BLOCK_K_W], [1, 0]
    )
    SHARED_LAYOUT_W_SCALES: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 16]],
        [SCALE_BLOCK_N, PACKED_MX_BLOCK],
        [1, 0],
    )
    SHARED_LAYOUT_Y: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[OUT_BLOCK_N, 8]], [BLOCK_M, OUT_BLOCK_N], [1, 0]
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
        shape=(N // PRESHUFFLE_FACTOR, tl.cdiv(K, MX_PACK_DIVISOR) * PRESHUFFLE_FACTOR),
        strides=(stride_w_mx_n, stride_w_mx_k),
        block_shape=(SCALE_BLOCK_N, PACKED_MX_BLOCK),
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

    num_k_iter = tl.cdiv(K, BLOCK_K)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    if NUM_BUFFERS == 1:
        # Non-pipelined baseline: load one K tile, wait for all TDM ops, consume.
        for ki in range(num_k_iter):
            _tdm_load_tile(
                x_desc,
                w_desc,
                ws_desc,
                x_buffer.index(0),
                w_buffer.index(0),
                ws_buffer.index(0),
                ki,
                GatherIndx,
                offs_x_m,
                offs_x_m_scalar,
                off_w_n,
                off_w_n_scale,
                BLOCK_K,
                PACKED_BLOCK_K_W,
                PACKED_MX_BLOCK,
            )
            gl.amd.gfx1250.tdm.async_wait(0)
            x_tile = x_buffer.index(0).load(layout=DOT_LAYOUT_X)
            w_kn = _preload_tile(
                w_buffer.index(0),
                ws_buffer.index(0),
                L_IN_W,
                L_SCALE_W,
                COMPACT_SCALE_LAYOUT,
                SWIZZLE_MX_SCALE,
                BLOCK_N,
                BLOCK_K,
                MX_SCALE_BLOCK_K,
                MX_PACK_DIVISOR,
                PRESHUFFLE_FACTOR,
                SCALE_KWIDTH,
                min(K_WIDTH, MX_PACK_DIVISOR),
                SCALE_SEL,
            )
            acc = gl.amd.gfx1250.wmma(x_tile, w_kn, acc)
    else:
        # ============================ STAGE 2 ============================
        # LDS-prefetch pipeline (NUM_BUFFERS>=2, needs num_k_iter >= NUM_BUFFERS).
        # Prefetch the tile NUM_BUFFERS-1 ahead into a different LDS slot to overlap
        # TDM latency with compute; operands load from LDS within each iter (not
        # carried in registers -- the bf16 weight+scale are large and would spill).
        #
        # NOTE(perf): does NOT beat stage-1 for minimax-m3 prefill: X is ~78% of the
        # 87 KB LDS footprint, so double-buffering at BLOCK_K=256 collapses occupancy
        # (14.9 vs 7.56 ms). Best pipelined ~8.96 ms at BLOCK_K=128. Kept for reference.
        #
        # NOTE(correctness): verified at the default BLOCK_K=256. The two barriers
        # below guard cross-wave RAW/WAR hazards on the shared slots that Membar does
        # not cover for TDM async ops. Fails at BLOCK_K=128 -- do not use BLOCK_K<256.
        for j in gl.static_range(NUM_BUFFERS - 1):
            _tdm_load_tile(
                x_desc,
                w_desc,
                ws_desc,
                x_buffer.index(j),
                w_buffer.index(j),
                ws_buffer.index(j),
                j,
                GatherIndx,
                offs_x_m,
                offs_x_m_scalar,
                off_w_n,
                off_w_n_scale,
                BLOCK_K,
                PACKED_BLOCK_K_W,
                PACKED_MX_BLOCK,
            )

        main_iters = num_k_iter - (NUM_BUFFERS - 1)
        for ki in range(main_iters):
            gl.amd.gfx1250.tdm.async_wait(max(NUM_BUFFERS - 2, 0) * NUM_TDM_OPS)
            gl.barrier()
            prefetch_ki = ki + NUM_BUFFERS - 1
            _tdm_load_tile(
                x_desc,
                w_desc,
                ws_desc,
                x_buffer.index(prefetch_ki % NUM_BUFFERS),
                w_buffer.index(prefetch_ki % NUM_BUFFERS),
                ws_buffer.index(prefetch_ki % NUM_BUFFERS),
                prefetch_ki,
                GatherIndx,
                offs_x_m,
                offs_x_m_scalar,
                off_w_n,
                off_w_n_scale,
                BLOCK_K,
                PACKED_BLOCK_K_W,
                PACKED_MX_BLOCK,
            )
            cur_slot = ki % NUM_BUFFERS
            x_tile = x_buffer.index(cur_slot).load(layout=DOT_LAYOUT_X)
            w_kn = _preload_tile(
                w_buffer.index(cur_slot),
                ws_buffer.index(cur_slot),
                L_IN_W,
                L_SCALE_W,
                COMPACT_SCALE_LAYOUT,
                SWIZZLE_MX_SCALE,
                BLOCK_N,
                BLOCK_K,
                MX_SCALE_BLOCK_K,
                MX_PACK_DIVISOR,
                PRESHUFFLE_FACTOR,
                SCALE_KWIDTH,
                min(K_WIDTH, MX_PACK_DIVISOR),
                SCALE_SEL,
            )
            acc = gl.amd.gfx1250.wmma(x_tile, w_kn, acc)

        # Epilogue: drain the last NUM_BUFFERS-1 already-prefetched tiles.
        for i in gl.static_range(NUM_BUFFERS - 1):
            gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * NUM_TDM_OPS)
            gl.barrier()
            cur_slot = (main_iters + i) % NUM_BUFFERS
            x_tile = x_buffer.index(cur_slot).load(layout=DOT_LAYOUT_X)
            w_kn = _preload_tile(
                w_buffer.index(cur_slot),
                ws_buffer.index(cur_slot),
                L_IN_W,
                L_SCALE_W,
                COMPACT_SCALE_LAYOUT,
                SWIZZLE_MX_SCALE,
                BLOCK_N,
                BLOCK_K,
                MX_SCALE_BLOCK_K,
                MX_PACK_DIVISOR,
                PRESHUFFLE_FACTOR,
                SCALE_KWIDTH,
                min(K_WIDTH, MX_PACK_DIVISOR),
                SCALE_SEL,
            )
            acc = gl.amd.gfx1250.wmma(x_tile, w_kn, acc)

    # bias / activation / write-back
    if B is not None:
        BPtrs = B + expt_id.to(index_type) * stride_b_e
        SHARED_LAYOUT_BIAS: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        bias_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=BPtrs,
            shape=(1, N),
            strides=(N, 1),
            block_shape=(1, BLOCK_N),
            layout=SHARED_LAYOUT_BIAS,
        )
        bias_buffer = gl.allocate_shared_memory(
            bias_desc.dtype, shape=[1, BLOCK_N], layout=bias_desc.layout
        )
        gl.amd.gfx1250.tdm.async_load(bias_desc, [0, pid_n * BLOCK_N], bias_buffer)
        gl.amd.gfx1250.tdm.async_wait(0)
        bias = bias_buffer.reshape((BLOCK_N,)).load(
            layout=gl.SliceLayout(0, WMMA_LAYOUT)
        )
        acc = acc + bias[None, :]

    if APPLY_SWIGLU:
        out = _swiglu(acc, alpha, limit, ADD_RESIDUAL=ADD_RESIDUAL)
        tl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
    else:
        tl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )
        out = acc

    if Gammas is not None:
        offs_m = BLOCK_M * block_id + gl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        gammas = gl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        out *= gammas[:, None]

    out = out.to(gl.bfloat16)

    # TDM Store: accumulator -> shared memory -> global memory
    Y += start_m.to(index_type) * stride_y_m
    y_buffer = gl.allocate_shared_memory(
        Y.type.element_ty,
        shape=[BLOCK_M, OUT_BLOCK_N],
        layout=SHARED_LAYOUT_Y,
    )
    y_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=Y,
        shape=(M, yN),
        strides=(stride_y_m, stride_y_n),
        block_shape=(BLOCK_M, OUT_BLOCK_N),
        layout=SHARED_LAYOUT_Y,
    )
    y_buffer.store(out)
    gl.amd.gfx1250.tdm.async_store(
        y_desc, [block_id * BLOCK_M, pid_n * OUT_BLOCK_N], y_buffer
    )
    gl.amd.gfx1250.tdm.async_wait(0)


@gluon.jit(launch_metadata=matmul_launch_metadata)
def _moe_gemm_a16w4_gluon_stage1(
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
    WMxScale,  # E8M0 compact scale (one byte per 32 values along K)
    stride_w_mx_e,
    stride_w_mx_n,
    stride_w_mx_k,
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
    # Pass None; compact e8m0 scale. GFX1250_SCALE branch hangs the ROCm loader;
    # CDNA4_SCALE unsupported -- use the Triton kernel for swizzled scales.
    SWIZZLE_MX_SCALE: gl.constexpr,
    EVEN_K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
    num_warps: gl.constexpr,
    UPCAST_INDICES: gl.constexpr = False,
):
    # Single-buffer (stage-1) entry point; distinct name for profiling/dispatch.
    _moe_gemm_a16w4_gluon_impl(
        Y=Y,
        stride_y_k=stride_y_k,
        stride_y_m=stride_y_m,
        stride_y_n=stride_y_n,
        X=X,
        stride_x_m=stride_x_m,
        stride_x_k=stride_x_k,
        W=W,
        stride_w_e=stride_w_e,
        stride_w_k=stride_w_k,
        stride_w_n=stride_w_n,
        WMxScale=WMxScale,
        stride_w_mx_e=stride_w_mx_e,
        stride_w_mx_n=stride_w_mx_n,
        stride_w_mx_k=stride_w_mx_k,
        B=B,
        stride_b_e=stride_b_e,
        Gammas=Gammas,
        num_tokens=num_tokens,
        N=N,
        K=K,
        GatherIndx=GatherIndx,
        ExptHist=ExptHist,
        ExptOffs=ExptOffs,
        ExptOffsSum=ExptOffsSum,
        ExptData=ExptData,
        grid_m=grid_m,
        grid_n=grid_n,
        APPLY_SWIGLU=APPLY_SWIGLU,
        alpha=alpha,
        limit=limit,
        ACTIVATION_REDUCTION_N=ACTIVATION_REDUCTION_N,
        ADD_RESIDUAL=ADD_RESIDUAL,
        N_EXPTS_ACT=N_EXPTS_ACT,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        XCD_SWIZZLE=XCD_SWIZZLE,
        NUM_BUFFERS=1,
        SWIZZLE_MX_SCALE=SWIZZLE_MX_SCALE,
        EVEN_K=EVEN_K,
        SPLIT_K=SPLIT_K,
        W_CACHE_MODIFIER=W_CACHE_MODIFIER,
        num_warps=num_warps,
        UPCAST_INDICES=UPCAST_INDICES,
    )


@gluon.jit(launch_metadata=matmul_launch_metadata)
def _moe_gemm_a16w4_gluon_stage2(
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
    WMxScale,  # E8M0 compact scale (one byte per 32 values along K)
    stride_w_mx_e,
    stride_w_mx_n,
    stride_w_mx_k,
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
    # Pass None; compact e8m0 scale. GFX1250_SCALE branch hangs the ROCm loader;
    # CDNA4_SCALE unsupported -- use the Triton kernel for swizzled scales.
    SWIZZLE_MX_SCALE: gl.constexpr,
    EVEN_K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
    num_warps: gl.constexpr,
    UPCAST_INDICES: gl.constexpr = False,
):
    # Double-buffer (stage-2) LDS-prefetch entry point.
    _moe_gemm_a16w4_gluon_impl(
        Y=Y,
        stride_y_k=stride_y_k,
        stride_y_m=stride_y_m,
        stride_y_n=stride_y_n,
        X=X,
        stride_x_m=stride_x_m,
        stride_x_k=stride_x_k,
        W=W,
        stride_w_e=stride_w_e,
        stride_w_k=stride_w_k,
        stride_w_n=stride_w_n,
        WMxScale=WMxScale,
        stride_w_mx_e=stride_w_mx_e,
        stride_w_mx_n=stride_w_mx_n,
        stride_w_mx_k=stride_w_mx_k,
        B=B,
        stride_b_e=stride_b_e,
        Gammas=Gammas,
        num_tokens=num_tokens,
        N=N,
        K=K,
        GatherIndx=GatherIndx,
        ExptHist=ExptHist,
        ExptOffs=ExptOffs,
        ExptOffsSum=ExptOffsSum,
        ExptData=ExptData,
        grid_m=grid_m,
        grid_n=grid_n,
        APPLY_SWIGLU=APPLY_SWIGLU,
        alpha=alpha,
        limit=limit,
        ACTIVATION_REDUCTION_N=ACTIVATION_REDUCTION_N,
        ADD_RESIDUAL=ADD_RESIDUAL,
        N_EXPTS_ACT=N_EXPTS_ACT,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        XCD_SWIZZLE=XCD_SWIZZLE,
        NUM_BUFFERS=NUM_BUFFERS,
        SWIZZLE_MX_SCALE=SWIZZLE_MX_SCALE,
        EVEN_K=EVEN_K,
        SPLIT_K=SPLIT_K,
        W_CACHE_MODIFIER=W_CACHE_MODIFIER,
        num_warps=num_warps,
        UPCAST_INDICES=UPCAST_INDICES,
    )


@gluon.jit(launch_metadata=matmul_launch_metadata)
def _moe_gemm_a16w4_gluon_stage3(
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
    WMxScale,  # E8M0 compact scale (one byte per 32 values along K)
    stride_w_mx_e,
    stride_w_mx_n,
    stride_w_mx_k,
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
    # Pass None; compact e8m0 scale. GFX1250_SCALE branch hangs the ROCm loader;
    # CDNA4_SCALE unsupported -- use the Triton kernel for swizzled scales.
    SWIZZLE_MX_SCALE: gl.constexpr,
    EVEN_K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
    num_warps: gl.constexpr,
    UPCAST_INDICES: gl.constexpr = False,
):
    # Triple-buffer (stage-3) entry point: stage-2 pipeline with NUM_BUFFERS=3.
    _moe_gemm_a16w4_gluon_impl(
        Y=Y,
        stride_y_k=stride_y_k,
        stride_y_m=stride_y_m,
        stride_y_n=stride_y_n,
        X=X,
        stride_x_m=stride_x_m,
        stride_x_k=stride_x_k,
        W=W,
        stride_w_e=stride_w_e,
        stride_w_k=stride_w_k,
        stride_w_n=stride_w_n,
        WMxScale=WMxScale,
        stride_w_mx_e=stride_w_mx_e,
        stride_w_mx_n=stride_w_mx_n,
        stride_w_mx_k=stride_w_mx_k,
        B=B,
        stride_b_e=stride_b_e,
        Gammas=Gammas,
        num_tokens=num_tokens,
        N=N,
        K=K,
        GatherIndx=GatherIndx,
        ExptHist=ExptHist,
        ExptOffs=ExptOffs,
        ExptOffsSum=ExptOffsSum,
        ExptData=ExptData,
        grid_m=grid_m,
        grid_n=grid_n,
        APPLY_SWIGLU=APPLY_SWIGLU,
        alpha=alpha,
        limit=limit,
        ACTIVATION_REDUCTION_N=ACTIVATION_REDUCTION_N,
        ADD_RESIDUAL=ADD_RESIDUAL,
        N_EXPTS_ACT=N_EXPTS_ACT,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        XCD_SWIZZLE=XCD_SWIZZLE,
        NUM_BUFFERS=NUM_BUFFERS,
        SWIZZLE_MX_SCALE=SWIZZLE_MX_SCALE,
        EVEN_K=EVEN_K,
        SPLIT_K=SPLIT_K,
        W_CACHE_MODIFIER=W_CACHE_MODIFIER,
        num_warps=num_warps,
        UPCAST_INDICES=UPCAST_INDICES,
    )
