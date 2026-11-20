import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx1250.attention.pa_decode_sparse import (
    _cga_bcast,
    _cga_bcast_1d,
    _cga_bcast_3d,
    _cga_shard0,
    _v4_2buff_fetch_slots,
    _v4_2buff_tile,
    _v4_dequant_chunk_mx,
    _v4_load_q_regs,
    _v4_pv_bf16,
    _v4_qk_mx_split,
    _v4_qk_mx_split_regs,
    _v4_store_acc,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.constexpr_function
def _slot_layout(block_k, num_warps, split, cga):
    per = block_k // split
    return gl.DistributedLinearLayout(
        reg_bases=[[1 << i] for i in range(per.bit_length() - 1)],
        lane_bases=[[0]] * 5,
        warp_bases=[
            [per << i if 1 << i < split else 0]
            for i in range(num_warps.bit_length() - 1)
        ],
        block_bases=cga,
        shape=[block_k],
    )


@gluon.jit
def _v4_dequant_lora_mx(
    kv_smem,
    mxs_smem,
    bf16_smem,
    rope_smem,
    rope_layout: gl.constexpr,
    NOPE_DIM: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    BLOCK_K: gl.constexpr,
    chunk2: gl.constexpr,
    chunk3: gl.constexpr,
    SC_SPAN: gl.constexpr,
    DQ0: gl.constexpr,
    DQ1: gl.constexpr,
    DQ2: gl.constexpr,
    ROPE_FP8: gl.constexpr,
):
    _v4_dequant_chunk_mx(
        kv_smem, mxs_smem, bf16_smem, 0, DQ0, BLOCK_K, chunk2, chunk3, SC_SPAN
    )
    _v4_dequant_chunk_mx(
        kv_smem, mxs_smem, bf16_smem, DQ0, DQ1, BLOCK_K, chunk2, chunk3, SC_SPAN
    )
    _v4_dequant_chunk_mx(
        kv_smem, mxs_smem, bf16_smem, DQ0 + DQ1, DQ2, BLOCK_K, chunk2, chunk3, SC_SPAN
    )
    if not ROPE_FP8:
        bf16_smem.slice(NOPE_DIM, ROPE_DIM, dim=1).store(rope_smem.load(rope_layout))


@gluon.jit
def _v4_qk_rope(
    scores,
    q_r,
    q_r_sc,
    kr_t,
    scale_buf,
    kr_sc_layout: gl.constexpr,
    bf16_wmma_layout: gl.constexpr,
    qk_wmma_layout: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SC_COL: gl.constexpr,
    ROPE_FP8: gl.constexpr,
):
    if ROPE_FP8:
        kr_sc = scale_buf.slice(SC_COL, 4, dim=1).load(kr_sc_layout)
        scores = gl.amd.gfx1250.wmma_scaled(
            q_r, q_r_sc, "e4m3", kr_t, kr_sc, "e4m3", scores
        )
    else:
        scores_r = gl.amd.gfx1250.wmma(
            q_r,
            kr_t,
            gl.zeros([BLOCK_H, BLOCK_K], dtype=gl.float32, layout=bf16_wmma_layout),
        )
        scores = scores + gl.convert_layout(scores_r, qk_wmma_layout)
    return scores


@gluon.jit
def _v4_qk_dq_step(
    acc,
    q_c,
    qs_c,
    kv_buf,
    scale_buf,
    bf16_smem,
    dot_k_layout: gl.constexpr,
    k_sc_layout: gl.constexpr,
    kbl: gl.constexpr,
    kbl3: gl.constexpr,
    C: gl.constexpr,
    BLOCK_K: gl.constexpr,
    MXK: gl.constexpr,
    SC_SPAN: gl.constexpr,
):
    NSC: gl.constexpr = MXK // SC_SPAN
    k_bl = kv_buf.slice(C * MXK, MXK, dim=1).load(kbl)
    k_op = gl.convert_layout(gl.permute(k_bl, 1, 0), dot_k_layout, assert_trivial=True)
    acc = gl.amd.gfx1250.wmma_scaled(
        q_c,
        qs_c,
        "e4m3",
        k_op,
        scale_buf.slice(C * NSC, NSC, dim=1).load(k_sc_layout),
        "e4m3",
        acc,
    )
    e = scale_buf.slice(C * NSC, NSC, dim=1).load(gl.SliceLayout(2, kbl3))
    r3 = gl.convert_layout(gl.reshape(k_bl, [BLOCK_K, NSC, SC_SPAN]), kbl3)
    sc3, r3 = gl.broadcast(e[:, :, None], r3)
    bf16_smem.slice(C * MXK, MXK, dim=1).store(
        gl.reshape(gl.amd.gfx1250.scaled_upcast(r3, sc3, gl.bfloat16), [BLOCK_K, MXK])
    )
    return acc


@gluon.jit
def _v4_2buff_gather(
    main_kv_ptr,
    main_u8_ptr,
    main_rope_ptr,
    extra_kv_ptr,
    extra_u8_ptr,
    extra_rope_ptr,
    main_rows,
    extra_rows,
    is_main,
    row,
    kv_desc,
    rope_desc,
    mxs_desc,
    kv_buf,
    rope_buf,
    sc_buf,
    kv_packed_shared: gl.constexpr,
    rope_shared: gl.constexpr,
    mxs_shared: gl.constexpr,
    kv_stride_n: gl.constexpr,
    kvr_stride_n: gl.constexpr,
    NOPE_DIM: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    REAL_MX: gl.constexpr,
    NUM_MX: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_D: gl.constexpr,
    HAS_EXTRA: gl.constexpr,
    ROPE_W: gl.constexpr,
    SC_OFF: gl.constexpr,
    FUSED_REC: gl.constexpr = False,
):
    if FUSED_REC:
        gl.amd.gfx1250.tdm.async_gather(kv_desc, row, kv_buf)
        gl.amd.gfx1250.tdm.async_gather(rope_desc, row, rope_buf)
    elif HAS_EXTRA:
        rows = gl.where(is_main, main_rows, extra_rows)
        tile_kv_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=gl.where(is_main, main_kv_ptr, extra_kv_ptr),
            shape=[rows, NOPE_DIM],
            strides=[kv_stride_n, 1],
            block_shape=[BLOCK_K, BLOCK_D],
            layout=kv_packed_shared,
        )
        tile_rope_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=gl.where(is_main, main_rope_ptr, extra_rope_ptr),
            shape=[rows, ROPE_DIM],
            strides=[kvr_stride_n, 1],
            block_shape=[BLOCK_K, ROPE_W],
            layout=rope_shared,
        )
        tile_mxs_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=gl.where(is_main, main_u8_ptr, extra_u8_ptr) + SC_OFF,
            shape=[rows, REAL_MX],
            strides=[kv_stride_n, 1],
            block_shape=[BLOCK_K, NUM_MX],
            layout=mxs_shared,
        )
        gl.amd.gfx1250.tdm.async_gather(tile_kv_desc, row, kv_buf)
        gl.amd.gfx1250.tdm.async_gather(tile_rope_desc, row, rope_buf)
        gl.amd.gfx1250.tdm.async_gather(tile_mxs_desc, row, sc_buf)
    else:
        gl.amd.gfx1250.tdm.async_gather(kv_desc, row, kv_buf)
        gl.amd.gfx1250.tdm.async_gather(rope_desc, row, rope_buf)
        gl.amd.gfx1250.tdm.async_gather(mxs_desc, row, sc_buf)


_pa_decode_sparse_fp8_repr = make_kernel_repr(
    "_pa_decode_sparse_fp8",
    [
        "BLOCK_H",
        "BLOCK_D",
        "BLOCK_K",
        "H",
        "D",
        "KV_SPLITS",
        "HAS_EXTRA",
    ],
)


@gluon.jit(repr=_pa_decode_sparse_fp8_repr)
def _pa_decode_sparse_fp8(
    q_ptr,
    q_u8_ptr,
    q_rope_ptr,
    unified_kv_ptr,
    kv_u8_ptr,
    kv_rope_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    extra_kv_ptr,
    extra_kv_u8_ptr,
    extra_rope_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    m_partial_ptr,
    l_partial_ptr,
    acc_partial_ptr,
    attn_sink_ptr,
    out_ptr,
    total_pages,
    extra_total_pages,
    main_blk_rows,
    extra_blk_rows,
    main_slots,
    extra_slots,
    q_stride_t: gl.constexpr,
    q_stride_h: gl.constexpr,
    qr_stride_t: gl.constexpr,
    qr_stride_h: gl.constexpr,
    kv_stride_n: gl.constexpr,
    kvr_stride_n: gl.constexpr,
    mp_stride_t: gl.constexpr,
    mp_stride_k: gl.constexpr,
    mp_stride_h: gl.constexpr,
    lp_stride_t: gl.constexpr,
    lp_stride_k: gl.constexpr,
    lp_stride_h: gl.constexpr,
    ap_stride_t: gl.constexpr,
    ap_stride_k: gl.constexpr,
    ap_stride_h: gl.constexpr,
    ap_stride_d: gl.constexpr,
    out_stride_t: gl.constexpr,
    out_stride_h: gl.constexpr,
    out_stride_d: gl.constexpr,
    H: gl.constexpr,
    D: gl.constexpr,
    KV_SPLITS: gl.constexpr,
    softmax_scale: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_D: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NOPE_DIM: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    GROUP_SIZE: gl.constexpr,
    Q_IN_VGPR: gl.constexpr,
    HAS_INVALID: gl.constexpr,
    Q_TDM: gl.constexpr,
    CTAS_H: gl.constexpr,
    MAIN_BLOCK_SIZE: gl.constexpr,
    EXTRA_BLOCK_SIZE: gl.constexpr,
    HAS_EXTRA: gl.constexpr,
    TDM_PARTIALS: gl.constexpr,
    ROPE_FP8: gl.constexpr,
    num_warps: gl.constexpr,
    lse_ptr=None,
    HAS_LSE: gl.constexpr = False,
    FUSED_REC: gl.constexpr = False,
    TAIL_MASK_ONLY: gl.constexpr = False,
    SOLO: gl.constexpr = False,
):
    WARP_SIZE: gl.constexpr = 32
    CGA_H: gl.constexpr = _cga_shard0(CTAS_H)
    CGA_B: gl.constexpr = _cga_bcast(CGA_H)
    CGA_B1: gl.constexpr = _cga_bcast_1d(CGA_H)
    LOG2E: gl.constexpr = 1.4426950408889634
    NUM_MX_BLOCKS: gl.constexpr = BLOCK_D // 32
    ROPE_W: gl.constexpr = 128 if ROPE_FP8 else ROPE_DIM
    ROPE_PAD: gl.constexpr = 16 if ROPE_FP8 else 8
    MX_SLOTS: gl.constexpr = NUM_MX_BLOCKS * (2 if ROPE_FP8 else 1)
    SC_OFF: gl.constexpr = NOPE_DIM + (ROPE_DIM if ROPE_FP8 else 0)
    REAL_MX_BLOCKS: gl.constexpr = SC_OFF // 32

    if num_warps == 1:
        qk_warp_bases: gl.constexpr = []
        pv_warp_bases: gl.constexpr = []
    elif num_warps == 2:
        qk_warp_bases: gl.constexpr = [[1, 0]]
        pv_warp_bases: gl.constexpr = [[1, 0]]
    elif num_warps == 4:
        qk_warp_bases: gl.constexpr = [[1, 0], [2, 0]]
        pv_warp_bases: gl.constexpr = [[0, 1], [0, 2]]
    else:
        qk_warp_bases: gl.constexpr = [[1, 0], [2, 0], [4, 0]]
        pv_warp_bases: gl.constexpr = [[0, 1], [0, 2], [0, 4]]

    QK_WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        instr_shape=[16, 16, 128],
        warp_bases=qk_warp_bases,
        cga_layout=CGA_H,
    )
    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=QK_WMMA_LAYOUT, k_width=16
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=QK_WMMA_LAYOUT, k_width=16
    )
    q_scale_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_q_layout, [BLOCK_H, NUM_MX_BLOCKS], 32
    )
    k_scale_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_k_layout, [BLOCK_K, NUM_MX_BLOCKS], 32
    )
    MXK: gl.constexpr = 128
    NSC_CHUNK: gl.constexpr = MXK // 32
    q_sc_chunk_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_q_layout, [BLOCK_H, NSC_CHUNK], 32
    )
    k_sc_chunk_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_k_layout, [BLOCK_K, NSC_CHUNK], 32
    )
    BF16_WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        instr_shape=[16, 16, 32],
        warp_bases=qk_warp_bases,
        cga_layout=CGA_H,
    )
    if ROPE_FP8:
        dot_qr_layout: gl.constexpr = dot_q_layout
        dot_kr_layout: gl.constexpr = dot_k_layout
    else:
        dot_qr_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=0, parent=BF16_WMMA_LAYOUT, k_width=8
        )
        dot_kr_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=1, parent=BF16_WMMA_LAYOUT, k_width=8
        )
    PV_WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        instr_shape=[16, 16, 32],
        warp_bases=pv_warp_bases,
        cga_layout=CGA_H,
    )
    dot_p_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=PV_WMMA_LAYOUT, k_width=8
    )
    dot_v_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=PV_WMMA_LAYOUT, k_width=8
    )
    valid_col_mma: gl.constexpr = gl.SliceLayout(0, QK_WMMA_LAYOUT)

    QDQ_D_THREADS: gl.constexpr = BLOCK_D // GROUP_SIZE
    QDQ_BLOCKED_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, GROUP_SIZE],
        threads_per_warp=[WARP_SIZE // QDQ_D_THREADS, QDQ_D_THREADS],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
        cga_layout=CGA_H,
    )
    MXS_BLOCKED_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[WARP_SIZE // NUM_MX_BLOCKS, NUM_MX_BLOCKS],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
        cga_layout=CGA_H,
    )
    GRP_D_THREADS: gl.constexpr = GROUP_SIZE // 8
    ROPE_D_THREADS: gl.constexpr = ROPE_DIM // 8
    ROPE_BLOCKED_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[WARP_SIZE // ROPE_D_THREADS, ROPE_D_THREADS],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
        cga_layout=CGA_H,
    )
    KVR_BLOCKED_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[WARP_SIZE // ROPE_D_THREADS, ROPE_D_THREADS],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
        cga_layout=CGA_B,
    )
    SC_SPAN: gl.constexpr = 32
    CGA_B3: gl.constexpr = _cga_bcast_3d(CGA_H)
    if BLOCK_K >= WARP_SIZE:
        DQ_TPW_K: gl.constexpr = WARP_SIZE
    else:
        DQ_TPW_K: gl.constexpr = BLOCK_K
    CHUNK2_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, SC_SPAN],
        threads_per_warp=[DQ_TPW_K, WARP_SIZE // DQ_TPW_K],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
        cga_layout=CGA_B,
    )
    CHUNK3_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, SC_SPAN],
        threads_per_warp=[DQ_TPW_K, WARP_SIZE // DQ_TPW_K, 1],
        warps_per_cta=[1, num_warps, 1],
        order=[2, 1, 0],
        cga_layout=CGA_B3,
    )
    KBL: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 32], [0, 64], [16, 0]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]],
        warp_bases=[],
        block_bases=CGA_B,
        shape=[32, 128],
    )
    KBL3: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 4],
            [0, 0, 8],
            [0, 1, 0],
            [0, 2, 0],
            [16, 0, 0],
        ],
        lane_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 16]],
        warp_bases=[],
        block_bases=CGA_B3,
        shape=[32, 4, 32],
    )
    SLOT_BLOCKED_LAYOUT: gl.constexpr = _slot_layout(
        BLOCK_K, num_warps, num_warps if ROPE_FP8 else 1, CGA_B1
    )

    kv_packed_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_D, 16]], [BLOCK_K, BLOCK_D], [1, 0], CGA_B
    )
    rope_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[ROPE_W, ROPE_PAD]], [BLOCK_K, ROPE_W], [1, 0], CGA_B
    )
    kv_bf16_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_D, 8]], [BLOCK_K, BLOCK_D], [1, 0], CGA_B
    )
    q_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_D, 16]], [BLOCK_H, BLOCK_D], [1, 0], CGA_H
    )
    TAIL_W: gl.constexpr = 128
    tail_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[TAIL_W, 16]], [BLOCK_K, TAIL_W], [1, 0], CGA_B
    )
    OUT_PIECE: gl.constexpr = BLOCK_D // 2
    out_piece_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[OUT_PIECE, 8]], [BLOCK_H, OUT_PIECE], [1, 0], CGA_H
    )
    acc_piece_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[OUT_PIECE, 4]], [BLOCK_H, OUT_PIECE], [1, 0], CGA_H
    )
    qs_shared: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0], cga_layout=CGA_H
    )
    qr_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[ROPE_W, ROPE_PAD]], [BLOCK_H, ROPE_W], [1, 0], CGA_H
    )
    mxs_shared: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0], cga_layout=CGA_B
    )
    slot_shared: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0], cga_layout=CGA_B
    )

    t = gl.program_id(0)
    pid_h = gl.program_id(1)
    pid_k = gl.program_id(2)
    h_off_base = pid_h * BLOCK_H

    kv_start = gl.load(kv_indptr_ptr + t)
    kv_end = gl.load(kv_indptr_ptr + t + 1)

    qk_scale = softmax_scale * LOG2E
    if SOLO:
        gl.static_assert(
            FUSED_REC
            and Q_TDM
            and Q_IN_VGPR
            and KV_SPLITS == 1
            and num_warps == 1
            and BLOCK_H == 32
            and BLOCK_K == 32
            and BLOCK_D == 512
        )
    elif Q_TDM:
        q_smem = gl.allocate_shared_memory(
            q_ptr.dtype.element_ty, [BLOCK_H, BLOCK_D], q_shared
        )
        qs_smem = gl.allocate_shared_memory(gl.uint8, [BLOCK_H, MX_SLOTS], qs_shared)
        qr_smem = gl.allocate_shared_memory(
            q_rope_ptr.dtype.element_ty, [BLOCK_H, ROPE_W], qr_shared
        )
        q_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=q_ptr + t * q_stride_t,
            shape=[H, NOPE_DIM],
            strides=[q_stride_h, 1],
            block_shape=[BLOCK_H, BLOCK_D],
            layout=q_shared,
        )
        qs_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=q_u8_ptr + t * q_stride_t + SC_OFF,
            shape=[H, REAL_MX_BLOCKS],
            strides=[q_stride_h, 1],
            block_shape=[BLOCK_H, MX_SLOTS],
            layout=qs_shared,
        )
        qr_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=q_rope_ptr + t * qr_stride_t,
            shape=[H, ROPE_DIM],
            strides=[qr_stride_h, 1],
            block_shape=[BLOCK_H, ROPE_W],
            layout=qr_shared,
        )
        gl.amd.gfx1250.tdm.async_load(q_desc, [h_off_base, 0], q_smem)
        gl.amd.gfx1250.tdm.async_load(qs_desc, [h_off_base, 0], qs_smem)
        gl.amd.gfx1250.tdm.async_load(qr_desc, [h_off_base, 0], qr_smem)
    elif not Q_TDM:
        h_offs_q = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, QDQ_BLOCKED_LAYOUT))
        d_offs_q = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, QDQ_BLOCKED_LAYOUT))
        h_offs_q_eff = h_off_base + h_offs_q
        q_e4m3 = gl.amd.cdna4.buffer_load(
            ptr=q_ptr + t * q_stride_t,
            offsets=(h_offs_q_eff[:, None] * q_stride_h + d_offs_q[None, :]).to(
                gl.int32
            ),
            mask=(h_offs_q_eff < H)[:, None] & (d_offs_q < NOPE_DIM)[None, :],
            other=0.0,
        )
        mfma_q = gl.convert_layout(q_e4m3, dot_q_layout)

        h_offs_s = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, MXS_BLOCKED_LAYOUT))
        b_offs_s = gl.arange(
            0, NUM_MX_BLOCKS, layout=gl.SliceLayout(0, MXS_BLOCKED_LAYOUT)
        )
        h_offs_s_eff = h_off_base + h_offs_s
        q_exp = gl.amd.cdna4.buffer_load(
            ptr=q_u8_ptr + t * q_stride_t,
            offsets=(
                h_offs_s_eff[:, None] * q_stride_h + NOPE_DIM + b_offs_s[None, :]
            ).to(gl.int32),
            mask=(h_offs_s_eff < H)[:, None] & (b_offs_s < REAL_MX_BLOCKS)[None, :],
            other=0,
        )
        q_scale = gl.convert_layout(q_exp, q_scale_layout)

        h_offs_r = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, ROPE_BLOCKED_LAYOUT))
        d_offs_r = gl.arange(0, ROPE_DIM, layout=gl.SliceLayout(0, ROPE_BLOCKED_LAYOUT))
        h_offs_r_eff = h_off_base + h_offs_r
        q_rope = gl.amd.cdna4.buffer_load(
            ptr=q_rope_ptr + t * qr_stride_t,
            offsets=(h_offs_r_eff[:, None] * qr_stride_h + d_offs_r[None, :]).to(
                gl.int32
            ),
            mask=(h_offs_r_eff < H)[:, None],
            other=0.0,
        )
        mfma_qr = gl.convert_layout(q_rope, dot_qr_layout)

    if not ROPE_FP8:
        qr_sc = 0
    kv_len = kv_end - kv_start
    main_tiles = gl.cdiv(kv_len, BLOCK_K)
    if HAS_EXTRA:
        extra_start = gl.load(extra_indptr_ptr + t)
        extra_len = gl.load(extra_indptr_ptr + t + 1) - extra_start
        num_tiles = main_tiles + gl.cdiv(extra_len, BLOCK_K)
    else:
        extra_start = 0
        extra_len = 0
        num_tiles = main_tiles
    OUT_BLOCKED_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[WARP_SIZE // GRP_D_THREADS, GRP_D_THREADS],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
        cga_layout=CGA_H,
    )
    tiles_per_segment = gl.cdiv(num_tiles, KV_SPLITS)
    if pid_k * tiles_per_segment >= num_tiles:
        if Q_TDM and not SOLO:
            gl.amd.gfx1250.tdm.async_wait(0)
        if KV_SPLITS == 1:
            h_offs_z = h_off_base + gl.arange(
                0, BLOCK_H, layout=gl.SliceLayout(1, OUT_BLOCKED_LAYOUT)
            )
            d_offs_z = gl.arange(
                0, BLOCK_D, layout=gl.SliceLayout(0, OUT_BLOCKED_LAYOUT)
            )
            gl.amd.cdna4.buffer_store(
                gl.zeros(
                    [BLOCK_H, BLOCK_D], out_ptr.dtype.element_ty, OUT_BLOCKED_LAYOUT
                ),
                ptr=out_ptr + t * out_stride_t,
                offsets=(
                    h_offs_z[:, None] * out_stride_h + d_offs_z[None, :] * out_stride_d
                ).to(gl.int32),
                mask=(h_offs_z < H)[:, None],
            )
            if HAS_LSE:
                gl.amd.cdna4.buffer_store(
                    gl.full(
                        [BLOCK_H],
                        float("-inf"),
                        gl.float32,
                        layout=gl.SliceLayout(1, OUT_BLOCKED_LAYOUT),
                    ),
                    ptr=lse_ptr + t * H,
                    offsets=h_offs_z.to(gl.int32),
                    mask=h_offs_z < H,
                )
        return
    tile_start = pid_k * tiles_per_segment
    tile_end = gl.minimum((pid_k + 1) * tiles_per_segment, num_tiles)
    num_iters = tile_end - tile_start

    h_offs_mma_row = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, QK_WMMA_LAYOUT))
    h_offs_mma_row_eff = h_off_base + h_offs_mma_row
    if KV_SPLITS == 1 and not HAS_LSE:
        sink = gl.amd.cdna4.buffer_load(
            ptr=attn_sink_ptr,
            offsets=h_offs_mma_row_eff.to(gl.int32),
            mask=h_offs_mma_row_eff < H,
            other=float("-inf"),
        ).to(gl.float32)
        sink = sink * LOG2E
        m_i = sink
        l_i = gl.exp2(sink - m_i)
    else:
        m_i = gl.full(
            [BLOCK_H],
            float("-inf"),
            gl.float32,
            layout=gl.SliceLayout(1, QK_WMMA_LAYOUT),
        )
        l_i = gl.full(
            [BLOCK_H], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, QK_WMMA_LAYOUT)
        )

    PV0: gl.constexpr = 256
    PV1: gl.constexpr = 128
    PV2: gl.constexpr = NOPE_DIM - PV0 - PV1
    PV_HALF: gl.constexpr = BLOCK_D // 2
    a0 = gl.zeros([BLOCK_H, PV_HALF], gl.float32, layout=PV_WMMA_LAYOUT)
    a1 = gl.zeros([BLOCK_H, PV_HALF], gl.float32, layout=PV_WMMA_LAYOUT)

    NUM_BUFFERS: gl.constexpr = 2
    NUM_STAGE: gl.constexpr = 1 if SOLO else 2
    kv_bufs = gl.allocate_shared_memory(
        unified_kv_ptr.dtype.element_ty,
        [NUM_BUFFERS, BLOCK_K, BLOCK_D],
        kv_packed_shared,
    )
    GATHER_OPS: gl.constexpr = 2 if FUSED_REC else 3
    if FUSED_REC:
        rope_bufs = gl.allocate_shared_memory(
            unified_kv_ptr.dtype.element_ty,
            [NUM_BUFFERS, BLOCK_K, TAIL_W],
            tail_shared,
        )
        scale_bufs = rope_bufs
    else:
        rope_bufs = gl.allocate_shared_memory(
            kv_rope_ptr.dtype.element_ty,
            [NUM_BUFFERS, BLOCK_K, ROPE_W],
            rope_shared,
        )
        scale_bufs = gl.allocate_shared_memory(
            gl.uint8, [NUM_BUFFERS, BLOCK_K, MX_SLOTS], mxs_shared
        )
    kv_bf16 = gl.allocate_shared_memory(
        gl.bfloat16, [NUM_STAGE, BLOCK_K, BLOCK_D], kv_bf16_shared
    )
    NUM_SLOT_BUFFERS: gl.constexpr = 4
    slot_bufs = gl.allocate_shared_memory(
        kv_indices_ptr.dtype.element_ty,
        [NUM_SLOT_BUFFERS, 1, BLOCK_K],
        slot_shared,
    )

    kv_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=unified_kv_ptr,
        shape=[total_pages, NOPE_DIM],
        strides=[kv_stride_n, 1],
        block_shape=[BLOCK_K, BLOCK_D],
        layout=kv_packed_shared,
    )
    mxs_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=kv_u8_ptr + SC_OFF,
        shape=[total_pages, REAL_MX_BLOCKS],
        strides=[kv_stride_n, 1],
        block_shape=[BLOCK_K, MX_SLOTS],
        layout=mxs_shared,
    )
    slot_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=kv_indices_ptr + kv_start,
        shape=[1, gl.maximum(kv_len, 1)],
        strides=[gl.maximum(kv_len, 1), 1],
        block_shape=[1, BLOCK_K],
        layout=slot_shared,
    )
    main_slot_base = kv_indices_ptr + kv_start
    if HAS_EXTRA:
        extra_slot_base = extra_indices_ptr + extra_start
    else:
        extra_slot_base = kv_indices_ptr

    k_offs_mma = gl.arange(0, BLOCK_K, layout=valid_col_mma)
    MASK_TILE: gl.constexpr = HAS_INVALID or HAS_EXTRA

    if FUSED_REC:
        gl.static_assert(ROPE_FP8 and not HAS_EXTRA and ROPE_W == TAIL_W)
        rope_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=unified_kv_ptr + NOPE_DIM,
            shape=[total_pages, TAIL_W],
            strides=[kv_stride_n, 1],
            block_shape=[BLOCK_K, TAIL_W],
            layout=tail_shared,
        )
        rope_cols = (
            gl.arange(0, ROPE_W, layout=gl.SliceLayout(1, dot_kr_layout)) < ROPE_DIM
        )
    else:
        rope_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=kv_rope_ptr,
            shape=[total_pages, ROPE_DIM],
            strides=[kvr_stride_n, 1],
            block_shape=[BLOCK_K, ROPE_W],
            layout=rope_shared,
        )

    if SOLO:
        q_smem = kv_bufs.index(1)
        qt_smem = rope_bufs.index(1)
        q_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=q_ptr + t * q_stride_t,
            shape=[H, NOPE_DIM],
            strides=[q_stride_h, 1],
            block_shape=[BLOCK_K, BLOCK_D],
            layout=kv_packed_shared,
        )
        qt_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=q_ptr + t * q_stride_t + NOPE_DIM,
            shape=[H, TAIL_W],
            strides=[q_stride_h, 1],
            block_shape=[BLOCK_K, TAIL_W],
            layout=tail_shared,
        )
        gl.amd.gfx1250.tdm.async_load(q_desc, [h_off_base, 0], q_smem)
        gl.amd.gfx1250.tdm.async_load(qt_desc, [h_off_base, 0], qt_smem)
        gl.amd.gfx1250.tdm.async_wait(0)
        qs_smem = qt_smem.reinterpret(dtype=gl.uint8).slice(ROPE_DIM, MX_SLOTS, dim=1)
        q_rope_cols = (
            gl.arange(0, ROPE_W, layout=gl.SliceLayout(0, dot_qr_layout)) < ROPE_DIM
        )
        mfma_qr = gl.where(q_rope_cols[None, :], qt_smem.load(dot_qr_layout), 0.0)
        qr_sc = qs_smem.slice(NUM_MX_BLOCKS, 4, dim=1).load(q_sc_chunk_layout)
        qv0, qv1, qv2, qv3, qsv0, qsv1, qsv2, qsv3 = _v4_load_q_regs(
            q_smem, qs_smem, dot_q_layout, q_sc_chunk_layout, BLOCK_D, MXK
        )
    elif Q_TDM:
        gl.amd.gfx1250.tdm.async_wait(0)
        mfma_qr = qr_smem.load(dot_qr_layout)
        if ROPE_FP8:
            qr_sc = qs_smem.slice(NUM_MX_BLOCKS, 4, dim=1).load(q_sc_chunk_layout)
        if Q_IN_VGPR:
            qv0, qv1, qv2, qv3, qsv0, qsv1, qsv2, qsv3 = _v4_load_q_regs(
                q_smem, qs_smem, dot_q_layout, q_sc_chunk_layout, BLOCK_D, MXK
            )
    _v4_2buff_fetch_slots(
        slot_desc,
        main_slot_base,
        extra_slot_base,
        kv_len,
        extra_len,
        tile_start,
        main_tiles,
        slot_bufs.index(0),
        slot_shared,
        BLOCK_K,
        HAS_EXTRA,
    )
    _v4_2buff_fetch_slots(
        slot_desc,
        main_slot_base,
        extra_slot_base,
        kv_len,
        extra_len,
        tile_start + 1,
        main_tiles,
        slot_bufs.index(1),
        slot_shared,
        BLOCK_K,
        HAS_EXTRA,
    )
    gl.amd.gfx1250.tdm.async_wait(1)
    cur_row, cur_is_main, cur_valid = _v4_2buff_tile(
        slot_bufs.index(0),
        tile_start,
        main_tiles,
        kv_len,
        extra_len,
        main_blk_rows,
        extra_blk_rows,
        main_slots,
        extra_slots,
        SLOT_BLOCKED_LAYOUT,
        valid_col_mma,
        BLOCK_K,
        MAIN_BLOCK_SIZE,
        EXTRA_BLOCK_SIZE,
        HAS_INVALID,
        HAS_EXTRA,
    )
    _v4_2buff_gather(
        unified_kv_ptr,
        kv_u8_ptr,
        kv_rope_ptr,
        extra_kv_ptr,
        extra_kv_u8_ptr,
        extra_rope_ptr,
        total_pages,
        extra_total_pages,
        cur_is_main,
        cur_row,
        kv_desc,
        rope_desc,
        mxs_desc,
        kv_bufs.index(0),
        rope_bufs.index(0),
        scale_bufs.index(0),
        kv_packed_shared,
        rope_shared,
        mxs_shared,
        kv_stride_n,
        kvr_stride_n,
        NOPE_DIM,
        ROPE_DIM,
        REAL_MX_BLOCKS,
        MX_SLOTS,
        BLOCK_K,
        BLOCK_D,
        HAS_EXTRA,
        ROPE_W,
        SC_OFF,
        FUSED_REC,
    )

    gl.assume(num_iters >= 1)
    if not SOLO:
        kv_bf16.index(1).store(
            gl.zeros([BLOCK_K, BLOCK_D], gl.bfloat16, layout=QDQ_BLOCKED_LAYOUT)
        )
    s_prev = gl.full(
        [BLOCK_H, BLOCK_K], float("-inf"), gl.float32, layout=QK_WMMA_LAYOUT
    )
    for i in tl.range(0, num_iters):
        if CTAS_H > 1:
            gl.amd.gfx1250.cluster.arrive()
            gl.amd.gfx1250.cluster.wait()
        gl.amd.gfx1250.tdm.async_wait(GATHER_OPS)
        next_row, next_is_main, next_valid = _v4_2buff_tile(
            slot_bufs.index((i + 1) % NUM_SLOT_BUFFERS),
            tile_start + i + 1,
            main_tiles,
            kv_len,
            extra_len,
            main_blk_rows,
            extra_blk_rows,
            main_slots,
            extra_slots,
            SLOT_BLOCKED_LAYOUT,
            valid_col_mma,
            BLOCK_K,
            MAIN_BLOCK_SIZE,
            EXTRA_BLOCK_SIZE,
            HAS_INVALID,
            HAS_EXTRA,
        )
        _v4_2buff_fetch_slots(
            slot_desc,
            main_slot_base,
            extra_slot_base,
            kv_len,
            extra_len,
            tile_start + i + 2,
            main_tiles,
            slot_bufs.index((i + 2) % NUM_SLOT_BUFFERS),
            slot_shared,
            BLOCK_K,
            HAS_EXTRA,
        )
        _v4_2buff_gather(
            unified_kv_ptr,
            kv_u8_ptr,
            kv_rope_ptr,
            extra_kv_ptr,
            extra_kv_u8_ptr,
            extra_rope_ptr,
            total_pages,
            extra_total_pages,
            next_is_main,
            next_row,
            kv_desc,
            rope_desc,
            mxs_desc,
            kv_bufs.index((i + 1) % NUM_BUFFERS),
            rope_bufs.index((i + 1) % NUM_BUFFERS),
            scale_bufs.index((i + 1) % NUM_BUFFERS),
            kv_packed_shared,
            rope_shared,
            mxs_shared,
            kv_stride_n,
            kvr_stride_n,
            NOPE_DIM,
            ROPE_DIM,
            REAL_MX_BLOCKS,
            MX_SLOTS,
            BLOCK_K,
            BLOCK_D,
            HAS_EXTRA,
            ROPE_W,
            SC_OFF,
            FUSED_REC,
        )
        gl.amd.gfx1250.tdm.async_wait(GATHER_OPS + 1)
        if CTAS_H > 1:
            gl.amd.gfx1250.cluster.arrive()
            gl.amd.gfx1250.cluster.wait()

        buf = i % NUM_BUFFERS
        kbuf = kv_bufs.index(buf)
        if FUSED_REC:
            tbuf = rope_bufs.index(buf)
            sbuf = tbuf.reinterpret(dtype=gl.uint8).slice(ROPE_DIM, MX_SLOTS, dim=1)
        else:
            sbuf = scale_bufs.index(buf)
        if SOLO:
            scores = gl.zeros(
                [BLOCK_H, BLOCK_K], dtype=gl.float32, layout=QK_WMMA_LAYOUT
            )
            scores = _v4_qk_dq_step(
                scores,
                qv0,
                qsv0,
                kbuf,
                sbuf,
                kv_bf16.index(0),
                dot_k_layout,
                k_sc_chunk_layout,
                KBL,
                KBL3,
                0,
                BLOCK_K,
                MXK,
                SC_SPAN,
            )
            scores = _v4_qk_dq_step(
                scores,
                qv1,
                qsv1,
                kbuf,
                sbuf,
                kv_bf16.index(0),
                dot_k_layout,
                k_sc_chunk_layout,
                KBL,
                KBL3,
                1,
                BLOCK_K,
                MXK,
                SC_SPAN,
            )
            scores = _v4_qk_dq_step(
                scores,
                qv2,
                qsv2,
                kbuf,
                sbuf,
                kv_bf16.index(0),
                dot_k_layout,
                k_sc_chunk_layout,
                KBL,
                KBL3,
                2,
                BLOCK_K,
                MXK,
                SC_SPAN,
            )
            scores = _v4_qk_dq_step(
                scores,
                qv3,
                qsv3,
                kbuf,
                sbuf,
                kv_bf16.index(0),
                dot_k_layout,
                k_sc_chunk_layout,
                KBL,
                KBL3,
                3,
                BLOCK_K,
                MXK,
                SC_SPAN,
            )
        elif Q_TDM:
            if Q_IN_VGPR:
                scores = _v4_qk_mx_split_regs(
                    qv0,
                    qv1,
                    qv2,
                    qv3,
                    qsv0,
                    qsv1,
                    qsv2,
                    qsv3,
                    kbuf,
                    sbuf,
                    dot_k_layout,
                    k_sc_chunk_layout,
                    QK_WMMA_LAYOUT,
                    BLOCK_H,
                    BLOCK_K,
                    MXK,
                )
            else:
                scores = _v4_qk_mx_split(
                    q_smem,
                    qs_smem,
                    kbuf,
                    sbuf,
                    dot_q_layout,
                    dot_k_layout,
                    q_sc_chunk_layout,
                    k_sc_chunk_layout,
                    QK_WMMA_LAYOUT,
                    BLOCK_H,
                    BLOCK_K,
                    BLOCK_D,
                    MXK,
                )
        else:
            k_e4m3 = kbuf.permute([1, 0]).load(dot_k_layout)
            k_scale = sbuf.load(k_scale_layout)
            scores = gl.amd.gfx1250.wmma_scaled(
                mfma_q,
                q_scale,
                "e4m3",
                k_e4m3,
                k_scale,
                "e4m3",
                gl.zeros([BLOCK_H, BLOCK_K], dtype=gl.float32, layout=QK_WMMA_LAYOUT),
            )
        kr_t = rope_bufs.index(buf).permute([1, 0]).load(dot_kr_layout)
        if FUSED_REC:
            kr_t = gl.where(rope_cols[:, None], kr_t, 0.0)
        scores = qk_scale * _v4_qk_rope(
            scores,
            mfma_qr,
            qr_sc,
            kr_t,
            sbuf,
            k_sc_chunk_layout,
            BF16_WMMA_LAYOUT,
            QK_WMMA_LAYOUT,
            BLOCK_H,
            BLOCK_K,
            NUM_MX_BLOCKS,
            ROPE_FP8,
        )
        if MASK_TILE:
            scores = scores + gl.where(cur_valid, 0.0, float("-inf"))[None, :]
        elif TAIL_MASK_ONLY:
            if (tile_start + i + 1) * BLOCK_K > kv_len:
                in_range = (tile_start + i) * BLOCK_K + k_offs_mma < kv_len
                scores = scores + gl.where(in_range, 0.0, float("-inf"))[None, :]
        else:
            in_range = (tile_start + i) * BLOCK_K + k_offs_mma < kv_len
            scores = scores + gl.where(in_range, 0.0, float("-inf"))[None, :]

        if SOLO:
            s_prev = scores
        m_block = gl.max(s_prev, axis=1)
        m_new = gl.maximum(m_i, m_block)
        m_safe = gl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = gl.exp2(m_i - m_safe)
        p = gl.exp2(s_prev - m_safe[:, None])
        l_i = l_i * alpha + gl.sum(p, axis=1)
        m_i = m_new
        alpha_pv = gl.convert_layout(alpha[:, None], layout=PV_WMMA_LAYOUT)
        a0 = a0 * alpha_pv
        a1 = a1 * alpha_pv
        p_dot = gl.convert_layout(p.to(gl.bfloat16), dot_p_layout)
        a0, a1 = _v4_pv_bf16(
            p_dot,
            kv_bf16.index(0 if SOLO else (i + 1) % 2),
            a0,
            a1,
            dot_v_layout,
            PV_HALF,
        )

        if not SOLO:
            _v4_dequant_lora_mx(
                kbuf,
                sbuf,
                kv_bf16.index(i % 2),
                rope_bufs.index(buf),
                KVR_BLOCKED_LAYOUT,
                NOPE_DIM,
                ROPE_DIM,
                BLOCK_K,
                CHUNK2_LAYOUT,
                CHUNK3_LAYOUT,
                SC_SPAN,
                PV0,
                PV1,
                PV2,
                ROPE_FP8,
            )
            s_prev = scores
        if MASK_TILE:
            cur_valid = next_valid

    gl.amd.gfx1250.tdm.async_wait(0)
    if CTAS_H > 1:
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
    if not SOLO:
        m_block = gl.max(s_prev, axis=1)
        m_new = gl.maximum(m_i, m_block)
        m_safe = gl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = gl.exp2(m_i - m_safe)
        p = gl.exp2(s_prev - m_safe[:, None])
        l_i = l_i * alpha + gl.sum(p, axis=1)
        m_i = m_new
        alpha_pv = gl.convert_layout(alpha[:, None], layout=PV_WMMA_LAYOUT)
        a0 = a0 * alpha_pv
        a1 = a1 * alpha_pv
        p_dot = gl.convert_layout(p.to(gl.bfloat16), dot_p_layout)
        a0, a1 = _v4_pv_bf16(
            p_dot,
            kv_bf16.index((num_iters - 1) % 2),
            a0,
            a1,
            dot_v_layout,
            PV_HALF,
        )

    h_offs_o = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, OUT_BLOCKED_LAYOUT))
    h_offs_o_eff = h_off_base + h_offs_o
    h_mask_o = h_offs_o_eff < H

    if KV_SPLITS == 1:
        if HAS_LSE:
            l_div = gl.where(m_i == float("-inf"), 1.0, l_i)
        else:
            l_div = l_i
        inv = gl.convert_layout(1.0 / l_div[:, None], layout=PV_WMMA_LAYOUT)
        oty: gl.constexpr = out_ptr.dtype.element_ty
        out_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=out_ptr + t * out_stride_t,
            shape=[H, BLOCK_D],
            strides=[out_stride_h, out_stride_d],
            block_shape=[BLOCK_H, OUT_PIECE],
            layout=out_piece_shared,
        )
        o0 = gl.allocate_shared_memory(oty, [BLOCK_H, OUT_PIECE], out_piece_shared)
        o1 = gl.allocate_shared_memory(oty, [BLOCK_H, OUT_PIECE], out_piece_shared)
        o0.store((a0 * inv).to(oty))
        gl.amd.gfx1250.tdm.async_store(out_desc, [h_off_base, 0], o0)
        o1.store((a1 * inv).to(oty))
        gl.amd.gfx1250.tdm.async_store(out_desc, [h_off_base, OUT_PIECE], o1)
        gl.amd.gfx1250.tdm.async_wait(0)
        if HAS_LSE:
            LN2: gl.constexpr = 0.6931471805599453
            h_offs_lse = h_off_base + gl.arange(
                0, BLOCK_H, layout=gl.SliceLayout(1, MXS_BLOCKED_LAYOUT)
            )
            gl.amd.cdna4.buffer_store(
                gl.convert_layout(
                    (m_i + gl.log2(l_i)) * LN2, gl.SliceLayout(1, MXS_BLOCKED_LAYOUT)
                ),
                ptr=lse_ptr + t * H,
                offsets=h_offs_lse.to(gl.int32),
                mask=h_offs_lse < H,
            )
    else:
        h_offs_ml = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, MXS_BLOCKED_LAYOUT))
        h_offs_ml_eff = h_off_base + h_offs_ml
        h_mask_ml = h_offs_ml_eff < H
        m_store = gl.convert_layout(m_i, gl.SliceLayout(1, MXS_BLOCKED_LAYOUT))
        l_store = gl.convert_layout(l_i, gl.SliceLayout(1, MXS_BLOCKED_LAYOUT))
        gl.amd.cdna4.buffer_store(
            m_store,
            ptr=m_partial_ptr + t * mp_stride_t + pid_k * mp_stride_k,
            offsets=(h_offs_ml_eff * mp_stride_h).to(gl.int32),
            mask=h_mask_ml,
        )
        gl.amd.cdna4.buffer_store(
            l_store,
            ptr=l_partial_ptr + t * lp_stride_t + pid_k * lp_stride_k,
            offsets=(h_offs_ml_eff * lp_stride_h).to(gl.int32),
            mask=h_mask_ml,
        )
        ab = acc_partial_ptr + t * ap_stride_t + pid_k * ap_stride_k
        if TDM_PARTIALS:
            acc_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=ab,
                shape=[H, BLOCK_D],
                strides=[ap_stride_h, ap_stride_d],
                block_shape=[BLOCK_H, OUT_PIECE],
                layout=acc_piece_shared,
            )
            p0 = gl.allocate_shared_memory(
                gl.float32, [BLOCK_H, OUT_PIECE], acc_piece_shared
            )
            p1 = gl.allocate_shared_memory(
                gl.float32, [BLOCK_H, OUT_PIECE], acc_piece_shared
            )
            p0.store(a0)
            gl.amd.gfx1250.tdm.async_store(acc_desc, [h_off_base, 0], p0)
            p1.store(a1)
            gl.amd.gfx1250.tdm.async_store(acc_desc, [h_off_base, OUT_PIECE], p1)
            gl.amd.gfx1250.tdm.async_wait(0)
        else:
            _v4_store_acc(
                a0,
                0,
                PV_HALF,
                ab,
                h_offs_o_eff,
                h_mask_o,
                ap_stride_h,
                ap_stride_d,
                OUT_BLOCKED_LAYOUT,
            )
            _v4_store_acc(
                a1,
                PV_HALF,
                PV_HALF,
                ab,
                h_offs_o_eff,
                h_mask_o,
                ap_stride_h,
                ap_stride_d,
                OUT_BLOCKED_LAYOUT,
            )
