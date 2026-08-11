# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Gated Delta Net K5 hidden-state recurrence kernel (@flyc.kernel API).

HIP-aligned fork: same ``mfma_f32_16x16x16bf16_1k`` instruction and warp
partition (BT split-M, K split across waves, V not split) as the hand-tuned
HIP/C++ K5 kernel; writes the public [..., V, K] layout through a [V][K]
transpose buffer + b128 store. Serially over the NT chunks: store the h
snapshot for K6, v_new = u - w @ h, then
h = h * exp(g_last) + k^T @ (v_new * exp(g_last - g_cumsum)).
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

_LOG2E = math.log2(math.e)


def _gview(tensor, base, shape, stride):
    """Buffer-resource view of ``tensor`` rooted at ELEMENT offset ``base``
    (None = origin), replacing the slot's own memref layout. The innermost mode
    is one vector access: slicing it off with ``None`` gives the copy tile."""
    it = fx.get_iter(fx.rocdl.make_buffer_tensor(tensor, max_size=True))
    if base is not None:
        it = fx.add_offset(it, base)
    return fx.Tensor(fx.make_view(it, fx.make_layout(shape, stride)))


def _load_vec(atom, tile, width, numeric):
    """Load ``width`` elements from ``tile``; ``atom`` picks buffer_load (global)
    or ds_read (LDS). fly-promote-regmem-to-vectorssa folds the fragment away."""
    frag = fx.make_rmem_tensor(width, numeric)
    fx.copy(atom, tile, frag)
    vec = frag.load()
    return vec[0] if width == 1 else vec


def _store_vec(atom, tile, value, width, numeric):
    """Inverse of ``_load_vec``: store ``value`` into the coordinate ``tile``."""
    frag = fx.make_rmem_tensor(width, numeric)
    frag.store(fx.Vector.from_elements([value], dtype=numeric) if width == 1 else value)
    fx.copy(atom, frag, tile)


def _make_fast_exp(g_is_log2_scaled: bool):
    """``exp(x)`` via ``exp2``, dropping ``* LOG2E`` when ``g`` is already
    log2(e)-prescaled. Raw ``rocdl.exp2`` (untyped, hence the ir_value hop)
    avoids the denormal rescale guard ``fx.exp2``'s math lowering adds."""
    if g_is_log2_scaled:
        return lambda x: fx.Float32(rocdl.exp2(T.f32, x.ir_value()))
    return lambda x: fx.Float32(rocdl.exp2(T.f32, (x * _LOG2E).ir_value()))


# Truncation keeps outputs bit-identical to the HIP/C++ K5 kernel; RNE is closer
# to the fp32 reference but does NOT bit-match HIP.
_BF16_CONVERT_TRUNC_DEFAULT = True


def _make_bf16_converter(trunc: bool):
    """fp32x4 -> bf16x4 converter: ``trunc=True`` keeps the high 16 bits (HIP
    ``float_to_bf16``), else RNE (``v_cvt_pk_bf16_f32`` on gfx950)."""
    if trunc:
        return lambda v: (
            (v.bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
        )
    return lambda v: v.to(fx.BFloat16)


def compile_chunk_gated_delta_h_mfma16_hip(
    *,
    K: int,
    V: int,
    BT: int = 64,
    BV: int = 32,
    H: int,
    Hg: int,
    USE_G: bool = True,
    USE_GK: bool = False,
    USE_INITIAL_STATE: bool = True,
    STORE_FINAL_STATE: bool = True,
    SAVE_NEW_VALUE: bool = True,
    IS_VARLEN: bool = True,
    WU_CONTIGUOUS: bool = True,
    STATE_DTYPE_BF16: bool = False,
    G_IS_LOG2_SCALED: bool = False,
    USE_STATE_INDICES: bool = False,
    SCHED_GFX942: bool = False,
    G_HEAD_MAJOR: bool = False,
    BF16_CONVERT_TRUNC: bool = _BF16_CONVERT_TRUNC_DEFAULT,
):
    """Compile the GDN K5 kernel into the @flyc.jit launcher below.

    ``STATE_DTYPE_BF16`` puts ``h0`` / ``ht`` in bf16 (promote on load, demote
    on store); the f32 accumulator and the LDS layouts are unchanged.
    """
    # BT=64 is baked into the wave mapping / load batching / BT_STEPS, gated_v
    # alias-reuses h_state panel 1, and the LDS layout is validated at K=V=128.
    assert BT == 64, f"chunk_gated_delta_h_mfma16_hip only supports BT=64, got BT={BT}"
    assert K == 128, f"chunk_gated_delta_h_mfma16_hip only supports K=128, got K={K}"
    assert BV % 16 == 0, f"BV must be a multiple of the MFMA N of 16, got BV={BV}"
    NUM_K_BLOCKS = K // 64

    # ``_gview`` re-describes shape/stride, so only the memref ELEMENT TYPE
    # comes from the tensor: placeholders for unused slots must match dtypes.

    _fast_exp = _make_fast_exp(G_IS_LOG2_SCALED)
    _f32x4_to_bf16x4 = _make_bf16_converter(BF16_CONVERT_TRUNC)

    WARP_SIZE = 64
    NUM_WARPS = 4
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE

    # Each k-step issues a lo/hi MFMA pair, so K_STEP is twice the MFMA K of 16.
    MFMA_N = 16
    K_STEP = 32
    N_REPEAT = BV // MFMA_N

    # w panels, one [BT][64] per 64-K block, written with HIP's
    # ``w_panel_swizzle`` and read by GEMM1 as a plain contiguous-16-K b64.
    LDS_WP_PANEL_ELEMS = BT * 64
    LDS_WP_ELEMS = NUM_K_BLOCKS * LDS_WP_PANEL_ELEMS

    # k panels, one [64 K-rows][BT tokens] per 64-K block, in HIP's rotating-
    # pair swizzle: b128 pair load + b16x2 scatter, so GEMM2's A frag is a b64.
    LDS_KP_PANEL_ELEMS = 64 * BT
    LDS_KP_ELEMS = NUM_K_BLOCKS * LDS_KP_PANEL_ELEMS

    # gated v_new in a shared2 panel; GEMM2 B reads contiguously over BT.
    LDS_GV_ELEMS = (BT // 4) * BV * 4  # 16 bt_groups x BV cols x 4 BT

    # h_state panels (shared2), one [row_block][V] per 64-K block, so GEMM1's B
    # frag is a plain b64. Each cell holds one k_group of 4 K.
    LDS_HP_PANEL_ELEMS = (64 // 4) * BV * 4  # = 64 * BV per 64-K block
    LDS_HP_ELEMS = NUM_K_BLOCKS * LDS_HP_PANEL_ELEMS
    assert LDS_GV_ELEMS == LDS_HP_PANEL_ELEMS, (
        f"gated_v aliases h_state panel 1 and must match it exactly, got "
        f"{LDS_GV_ELEMS} vs {LDS_HP_PANEL_ELEMS}"
    )

    # [V][K] transpose buffer for the h snapshot store: V-major with K innermost
    # so 8 adjacent K are one ds_read_b128 + one 16 B-aligned buffer_store.
    LDS_HT_STRIDE = K
    LDS_HT_ELEMS = BV * LDS_HT_STRIDE

    # ``wp`` / ``hp`` must each stay one Array: their views span both 64-K
    # panels with a panel stride, so the panels have to be contiguous.
    @fx.struct
    class SharedStorage:
        wp: fx.Array[fx.BFloat16, LDS_WP_ELEMS, 16]
        kp: fx.Array[fx.BFloat16, LDS_KP_ELEMS, 16]
        hp: fx.Array[fx.BFloat16, LDS_HP_ELEMS, 16]
        ht: fx.Array[fx.BFloat16, LDS_HT_ELEMS, 16]

    LOAD_VEC_WIDTH = 8  # 8 bf16 = 16 bytes = buffer_load_dwordx4
    THREADS_PER_ROW_64 = 64 // LOAD_VEC_WIDTH  # 8
    ROWS_PER_BATCH_64 = BLOCK_THREADS // THREADS_PER_ROW_64  # 32
    NUM_LOAD_BATCHES_64 = BT // ROWS_PER_BATCH_64  # 2

    K_STEPS_PER_BLOCK = 64 // K_STEP

    @flyc.kernel(name="chunk_gdn_fwd_h_flydsl_mfma16_hip")
    def gdn_h_kernel(
        k_tensor: fx.Tensor,
        v_tensor: fx.Tensor,
        w_tensor: fx.Tensor,
        v_new_tensor: fx.Tensor,
        g_tensor: fx.Tensor,
        gk_tensor: fx.Tensor,
        h_tensor: fx.Tensor,
        h0_tensor: fx.Tensor,
        ht_tensor: fx.Tensor,
        cu_seqlens_tensor: fx.Tensor,
        chunk_offsets_tensor: fx.Tensor,
        state_indices_tensor: fx.Tensor,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        # Only bounds the cu_seqlens / chunk_offsets / state_indices views; the
        # sequence count itself is carried by the grid.y extent.
        N_val: fx.Int32,
    ):
        i_v = fx.block_idx.x
        i_nh = fx.block_idx.y
        i_n = i_nh // H
        i_h = i_nh % H

        cp_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        cp_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        cp_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        cp_bf16x8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)

        # State-pool gather: the slot is ``state_indices[i_n]`` into a
        # [pool_size, H, V, K] pool. Only h0 / ht use it; the snapshot is dense.
        if const_expr(USE_STATE_INDICES):
            si_view = _gview(state_indices_tensor, None, (N_val, 1), (1, 1))
            state_n = _load_vec(cp_i32, fx.slice(si_view, (i_n, None)), 1, fx.Int32)
        else:
            state_n = i_n
        state_nh = state_n * H + i_h

        tid = fx.thread_idx.x
        wid = tid // WARP_SIZE
        lane = tid % WARP_SIZE

        # h0/ht element type follows the compile-time flag. The accesses are 4
        # contiguous K, so the copy atom is sized to match (b64 bf16, b128 f32).
        state_num = fx.BFloat16 if STATE_DTYPE_BF16 else fx.Float32
        cp_state_x4 = fx.make_copy_atom(
            fx.rocdl.BufferCopy64b() if STATE_DTYPE_BF16 else fx.rocdl.BufferCopy128b(),
            state_num,
        )

        if const_expr(IS_VARLEN):
            cu_view = _gview(cu_seqlens_tensor, None, (N_val + 1, 1), (1, 1))
            co_view = _gview(chunk_offsets_tensor, None, (N_val, 1), (1, 1))

        # -- LDS -- every MMA operand panel is a view in the shape the tiled MMA
        # expects (A as (M, K), B as (N, K)), so partition_S computes per-lane
        # addresses once outside the chunk loop. K nests as (4, k_groups,
        # panels): the inner 4 = one k_group = one MFMA operand = one b64, kept
        # contiguous by the base=2 of every swizzle.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        KG_PER_BLOCK = 64 // 4  # k_groups per 64-K panel

        # GEMM1 A -- w panels as (BT, K): [BT][64] row-major + S<4,2,4>, the
        # layout form of HIP's ``w_panel_swizzle``; both panels share one view.
        sW = lds.wp.view(
            fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(4, 2, 4)),
                fx.make_layout(
                    (BT, (4, KG_PER_BLOCK, NUM_K_BLOCKS)),
                    (64, (1, 4, LDS_WP_PANEL_ELEMS)),
                ),
            ),
        )
        # GEMM1 B -- h_state panels (shared2) as (BV, K). No swizzle needed.
        sH = lds.hp.view(
            fx.make_layout(
                (BV, (4, KG_PER_BLOCK, NUM_K_BLOCKS)),
                (4, (1, BV * 4, LDS_HP_PANEL_ELEMS)),
            )
        )
        # GEMM2 B -- gated_v as (BV, BT). ALIASES h_state panel 1 (a separate
        # buffer costs an occupancy step); a WAR barrier orders it after GEMM1.
        sGV = fx.make_view(
            lds.hp.ptr + LDS_HP_PANEL_ELEMS,
            fx.make_layout((BV, (4, BT // 4)), (4, (1, BV * 4))),
        )

        # GEMM2 A -- k panels as (64, BT), one panel being k^T of a 64-K block:
        # [64][BT] row-major + S<2,2,8> + S<4,2,4>, the layout form of HIP's
        # ``k_panel_rotating_pair_addr_bytes``. Two swizzles because the store
        # (K-row bits 3..5) and the MFMA A read (bits 0..3) must both stay
        # conflict-free, folding six row bits into the four usable bits 3..6.
        # Index with fx.slice: partition_S mishandles a NESTED composed layout
        # and silently produces wrong addresses.
        def _k_panel_view(kb, group):
            """k panel ``kb`` as (64, (group, BT/group)): same addresses either
            way -- group=4 is the MFMA A operand, group=2 the store pair."""
            return fx.make_view(
                lds.kp.ptr + kb * LDS_KP_PANEL_ELEMS,
                fx.make_composed_layout(
                    fx.static(fx.SwizzleType.get(4, 2, 4)),
                    fx.make_composed_layout(
                        fx.static(fx.SwizzleType.get(2, 2, 8)),
                        fx.make_layout((64, (group, BT // group)), (BT, (1, group))),
                    ),
                ),
            )

        sK = [_k_panel_view(kb, 4) for kb in range(NUM_K_BLOCKS)]
        sKw = [_k_panel_view(kb, 2) for kb in range(NUM_K_BLOCKS)]
        # [V][K] transpose buffer: [BV][K] row-major + S<4,2,5>, the layout form
        # of the hand-written ``k_group ^ (v & 0xF)`` scatter.
        sHT = lds.ht.view(
            fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(4, 2, 5)),
                fx.make_layout((BV, K // 4, 4), (K, 4, 1)),
            )
        )

        # Every LDS <-> register move here is one k_group (4 bf16 = one b64).
        cp_lds_x4 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)

        def _lds_read_x4(tile):
            return _load_vec(cp_lds_x4, tile, 4, fx.BFloat16)

        def _lds_write_x4(tile, vec4):
            _store_vec(cp_lds_x4, tile, vec4, 4, fx.BFloat16)

        cp_lds_x2 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.BFloat16)

        # Scatter the token pair (2*pc, 2*pc+1) of K-row ``row`` into panel
        # ``kb``, through the swizzled layout the GEMM2 A read uses.
        def _k_panel_store_pair(kb, row, pc, v0, v1):
            _store_vec(
                cp_lds_x2,
                fx.slice(sKw[kb], (row, (None, pc))),
                fx.Vector.from_elements([v0, v1], dtype=fx.BFloat16),
                2,
                fx.BFloat16,
            )

        # 16x16x16 bf16 MFMA (A/B bf16x4 = one b64, C f32x4) tiled over the 4
        # waves along M. Its TV layout is the lane mapping assumed everywhere
        # below: A[wid*16 + l%16][(l//16)*4 + v], B[l%16][(l//16)*4 + v] (zero
        # wave stride), C[wid*16 + (l//16)*4 + v][l%16].
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16, fx.Float32))
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((NUM_WARPS, 1, 1), (1, 0, 0))
        )
        thr_cp_a = fx.make_tiled_copy_A(cp_lds_x4, tiled_mma).get_slice(tid)
        thr_cp_b = fx.make_tiled_copy_B(cp_lds_x4, tiled_mma).get_slice(tid)

        pS_w = thr_cp_a.partition_S(sW)  # GEMM1 A: ((4,1), 1, (4, NUM_K_BLOCKS))
        pS_h = thr_cp_b.partition_S(sH)  # GEMM1 B: ((4,1), N_REPEAT, K_TILES)
        pS_gv = thr_cp_b.partition_S(sGV)  # GEMM2 B: ((4,1), N_REPEAT, BT_TILES)

        # ``retile`` is the copy-side view of the same registers (fx.gemm takes
        # the mma-side one); A's K mode stays nested, the copy/B side is flat.
        K_TILES_PER_BLOCK = 64 // 16
        frag_w = tiled_mma.make_fragment_A(sW)
        frag_h = tiled_mma.make_fragment_B(sH)
        frag_gv = tiled_mma.make_fragment_B(sGV)
        frag_k = [tiled_mma.make_fragment_A(v) for v in sK]
        frag_w_rt = thr_cp_a.retile(frag_w)
        frag_h_rt = thr_cp_b.retile(frag_h)
        frag_gv_rt = thr_cp_b.retile(frag_gv)
        # Accumulators: b_v (GEMM1) over (BT, BV), one h acc per 64-K block.
        # ``fx.Vector(frag.load())`` reshapes NESTED -> flat (4,); keep it.
        frag_bv = fx.make_rmem_tensor(
            fx.tiled_mma_partition_shape(fx.MmaOperand.C, tiled_mma, (BT, BV)),
            fx.Float32,
        )
        frag_h_accs = [
            fx.make_rmem_tensor(
                fx.tiled_mma_partition_shape(fx.MmaOperand.C, tiled_mma, (64, BV)),
                fx.Float32,
            )
            for _ in range(NUM_K_BLOCKS)
        ]

        # load_vec_group is this thread's LOAD_VEC_WIDTH-wide K vector within a
        # 64-K block, i.e. the global-view column coordinate.
        load_row_in_batch = tid // THREADS_PER_ROW_64
        load_vec_group = tid % THREADS_PER_ROW_64
        load_col_group = load_vec_group * (LOAD_VEC_WIDTH // 4)

        if const_expr(IS_VARLEN):
            bos = _load_vec(cp_i32, fx.slice(cu_view, (i_n, None)), 1, fx.Int32)
            eos = _load_vec(cp_i32, fx.slice(cu_view, (i_n + 1, None)), 1, fx.Int32)
            T_local = eos - bos
            NT = (T_local + (BT - 1)) // BT
            boh = _load_vec(cp_i32, fx.slice(co_view, (i_n, None)), 1, fx.Int32)
        else:
            bos = i_n * T_val
            T_local = T_val
            NT = (T_local + (BT - 1)) // BT
            boh = i_n * NT

        # Rows past the sequence end clamp to 0 rather than being masked -- with
        # max_size descriptors that is what keeps w / k / u in range.
        def _clamp_row(row):
            return (row < T_local).select(row, 0)

        # -- Global tensor views -- bases stay i64 (snapshot tensors can exceed
        # 2^31 elements); the per-chunk coordinates fit in i32.
        i_n64 = fx.Int64(i_n)
        i_h64 = fx.Int64(i_h)
        bos64 = fx.Int64(bos)
        boh64 = fx.Int64(boh)
        t_flat64 = fx.Int64(T_flat)

        VEC = LOAD_VEC_WIDTH

        # h: [B, NT, H, V, K] (VK) -- base = (boh*H + i_h) * V * K
        h_view = _gview(
            h_tensor,
            (boh64 * H + i_h64) * (V * K),
            (NT, V, K // VEC, VEC),
            (H * V * K, K, VEC, 1),
        )

        # k: [B, T, Hg, K] -- base = (bos*Hg + i_h//(H//Hg)) * K
        gqa_ratio = H // Hg
        k_view = _gview(
            k_tensor,
            (bos64 * Hg + fx.Int64(i_h // gqa_ratio)) * K,
            (T_local, K // VEC, VEC),
            (Hg * K, VEC, 1),
        )

        if const_expr(WU_CONTIGUOUS):
            if const_expr(IS_VARLEN):
                v_base = (i_h64 * t_flat64 + bos64) * V
                w_base = (i_h64 * t_flat64 + bos64) * K
            else:
                v_base = ((i_n64 * H + i_h64) * t_flat64) * V
                w_base = ((i_n64 * H + i_h64) * t_flat64) * K
            stride_v = V
            stride_w = K
        else:
            v_base = (bos64 * H + i_h64) * V
            w_base = (bos64 * H + i_h64) * K
            stride_v = H * V
            stride_w = H * K
        w_view = _gview(w_tensor, w_base, (T_local, K // VEC, VEC), (stride_w, VEC, 1))
        # u / v_new are accessed one element at a time, so their innermost mode
        # is a single element rather than a vector.
        u_view = _gview(v_tensor, v_base, (T_local, V, 1), (stride_v, 1, 1))

        if const_expr(IS_VARLEN):
            vn_base = (i_h64 * t_flat64 + bos64) * V
        else:
            vn_base = ((i_n64 * H + i_h64) * t_flat64) * V
        vn_view = _gview(v_new_tensor, vn_base, (T_local, V, 1), (V, 1, 1))

        # h0/ht: the [V, K] state slot for this (state, head), K split into the
        # 4 contiguous elements of one state vector access.
        state_slot_base = fx.Int64(state_nh) * (V * K)
        if const_expr(USE_INITIAL_STATE):
            h0_view = _gview(h0_tensor, state_slot_base, (V, K // 4, 4), (K, 4, 1))
        if const_expr(STORE_FINAL_STATE):
            ht_view = _gview(ht_tensor, state_slot_base, (V, K // 4, 4), (K, 4, 1))

        # g strides (g_stride_h, g_stride_t): head-major [B, H, T_flat] is
        # (T_flat, 1), token-major [B, T_flat, H] is (1, H). varlen flattens to
        # B==1 with a global token, so bos folds into the base.
        if const_expr(USE_G):
            if const_expr(G_HEAD_MAJOR):
                g_stride_h = t_flat64
                g_stride_t = 1
            else:
                g_stride_h = 1
                g_stride_t = H
            if const_expr(IS_VARLEN):
                g_sh_base = i_h64 * g_stride_h + bos64 * g_stride_t
            else:
                g_sh_base = i_n64 * H * t_flat64 + i_h64 * g_stride_h
            g_view = _gview(g_tensor, g_sh_base, (T_local, 1), (g_stride_t, 1))

        # gk: [B, T, H, K] token-major; the last token is a coordinate, so the
        # view hoists out of the chunk loop.
        if const_expr(USE_GK):
            gk_view = _gview(
                gk_tensor,
                bos64 * (H * K) + i_h64 * K,
                (T_local, K, 1),
                (H * K, 1, 1),
            )

        lane_n = lane % 16
        lane_m_base = lane // 16
        # The k_group this lane owns in a 64-K panel; the 4 warps cover all 16.
        row_block = wid * 4 + lane_m_base

        # The [V, K] state cell shared by the h0 load and the ht store.
        def _state_coord(kb, slot):
            return i_v * BV + slot * 16 + lane_n, kb * 16 + row_block

        for kb in range_constexpr(NUM_K_BLOCKS):
            frag_h_accs[kb].fill(0.0)

        # h0 is [V, K], so 4 consecutive K are one buffer_load_dwordx4.
        if const_expr(USE_INITIAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for slot in range_constexpr(N_REPEAT):
                    h0_col, h0_kgroup = _state_coord(kb, slot)
                    loaded_vec = _load_vec(
                        cp_state_x4,
                        fx.slice(h0_view, (h0_col, h0_kgroup, None)),
                        4,
                        state_num,
                    )
                    if const_expr(STATE_DTYPE_BF16):
                        loaded_vec = loaded_vec.to(fx.Float32)
                    _acc_cell = frag_h_accs[kb][None, None, slot]
                    _acc_cell.store(fx.Vector(_acc_cell.load()) + loaded_vec)

        # Pipelined chunk loop: the prologue stages chunk 0's w/k, then each
        # iteration prefetches the next chunk's w/k and publishes it at the end.
        GEMM1_PF_SPLIT = 1
        k_vec_group_pf = lane & 7
        k_row_base_pf = k_vec_group_pf * 8
        k_pair_col_pf = wid * 8 + (lane >> 3)
        k_t0_pf = k_pair_col_pf * 2
        k_t1_pf = k_t0_pf + 1

        # Load and store are split so the main loop can issue the loads early;
        # ``row_base=None`` is chunk 0 (absolute row == in-chunk row).
        def _load_w_rows(row_base=None):
            rows = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = batch * ROWS_PER_BATCH_64 + load_row_in_batch
                    abs_row = row if row_base is None else row_base + row
                    wvec = _load_vec(
                        cp_bf16x8,
                        fx.slice(
                            w_view, (_clamp_row(abs_row), kb * 8 + load_vec_group, None)
                        ),
                        LOAD_VEC_WIDTH,
                        fx.BFloat16,
                    )
                    rows.append((kb, row, wvec))
            return rows

        def _store_w_rows(rows):
            for kb, row, wvec in rows:
                _lds_write_x4(
                    fx.slice(sW, (row, (None, load_col_group, kb))),
                    wvec.shuffle(wvec, [0, 1, 2, 3]),
                )
                _lds_write_x4(
                    fx.slice(sW, (row, (None, load_col_group + 1, kb))),
                    wvec.shuffle(wvec, [4, 5, 6, 7]),
                )

        def _load_k_pairs(row_base=None):
            t0 = k_t0_pf if row_base is None else row_base + k_t0_pf
            t1 = k_t1_pf if row_base is None else row_base + k_t1_pf
            safe_t0 = _clamp_row(t0)
            safe_t1 = _clamp_row(t1)
            pairs = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                k_vec_col = kb * 8 + k_vec_group_pf
                kvec_t0 = _load_vec(
                    cp_bf16x8,
                    fx.slice(k_view, (safe_t0, k_vec_col, None)),
                    LOAD_VEC_WIDTH,
                    fx.BFloat16,
                )
                kvec_t1 = _load_vec(
                    cp_bf16x8,
                    fx.slice(k_view, (safe_t1, k_vec_col, None)),
                    LOAD_VEC_WIDTH,
                    fx.BFloat16,
                )
                pairs.append((kb, kvec_t0, kvec_t1))
            return pairs

        def _store_k_pairs(pairs):
            for kb, kvec_t0, kvec_t1 in pairs:
                for i in range_constexpr(LOAD_VEC_WIDTH):
                    _k_panel_store_pair(
                        kb, k_row_base_pf + i, k_pair_col_pf, kvec_t0[i], kvec_t1[i]
                    )

        c_zero = fx.Int64(0)
        c_one = fx.Int64(1)
        nt_idx = fx.Int64(NT)

        # PROLOGUE: stage chunk 0's w/k under a block-uniform NT>0 guard -- an
        # empty varlen sequence has bos == T_flat, so its clamped w/k address
        # would run past the input, and skipping is correct since the loop then
        # runs 0 times. Closures keep views/atoms out of the scf.if state.
        has_work = NT > 0
        if has_work:
            _store_w_rows(_load_w_rows())
            _store_k_pairs(_load_k_pairs())
            gpu.barrier()

        # Loop-carried state: scf.for wants plain values, so each accumulator
        # fragment enters and leaves the body as one flat vector.
        h_accs_init = [frag_h_accs[kb].load() for kb in range_constexpr(NUM_K_BLOCKS)]
        H_ACC_ELEMS = N_REPEAT * 4

        for i_t, state in range(c_zero, nt_idx, c_one, init=h_accs_init):
            for kb in range_constexpr(NUM_K_BLOCKS):
                frag_h_accs[kb].store(fx.Vector(state[kb], (H_ACC_ELEMS,), fx.Float32))
            # ``i_t`` is the raw scf.for induction variable (untyped i64) and
            # must be wrapped before any arithmetic.
            i_t_i32 = fx.Int32(i_t)

            # Absolute token row of MFMA C element ``elem_i``; ``i_t_i32`` is a
            # default arg (ruff B023), only called while tracing this chunk.
            def _bt_abs_row(elem_i, i_t_i32=i_t_i32):
                return i_t_i32 * BT + wid * 16 + lane_m_base * 4 + elem_i

            # Stage h_accs into the h_state panels (GEMM1 B operand) and the
            # [V][K] transpose buffer (b128 HBM store).
            for kb in range_constexpr(NUM_K_BLOCKS):
                for acc_j in range_constexpr(N_REPEAT):
                    acc_bf16 = _f32x4_to_bf16x4(
                        fx.Vector(frag_h_accs[kb][None, None, acc_j].load())
                    )
                    hp_col = acc_j * 16 + lane_n
                    _lds_write_x4(
                        fx.slice(sH, (hp_col, (None, row_block, kb))), acc_bf16
                    )

                    # The sHT swizzle breaks bank conflicts on this scatter
                    # write while keeping a k_group contiguous -> one b64.
                    _lds_write_x4(
                        fx.slice(sHT, (hp_col, kb * 16 + row_block, None)), acc_bf16
                    )

            # w/k for this chunk already in LDS (prologue or prev GEMM2 end).
            gpu.barrier()

            next_chunk_end = (i_t_i32 + 1) * BT
            last_idx_raw = (next_chunk_end < T_local).select(
                next_chunk_end, T_local
            ) - 1

            # Issue every u / g load before GEMM1 so the 64-MFMA chain hides the
            # latency; left alone LLVM sinks them into the middle of GEMM1.
            u_cols = [i_v * BV + idx * 16 + lane_n for idx in range(N_REPEAT)]
            u_prefetch = []  # N_REPEAT x 4 bf16 scalars
            for idx in range_constexpr(N_REPEAT):
                for elem_i in range_constexpr(4):
                    safe_u_row = _clamp_row(_bt_abs_row(elem_i))
                    u_prefetch.append(
                        _load_vec(
                            cp_bf16,
                            fx.slice(u_view, (safe_u_row, u_cols[idx], None)),
                            1,
                            fx.BFloat16,
                        )
                    )

            if const_expr(USE_G):
                g_last = _load_vec(
                    cp_f32, fx.slice(g_view, (last_idx_raw, None)), 1, fx.Float32
                )
                g_row_pf = []
                for elem_i in range_constexpr(4):
                    abs_row = _bt_abs_row(elem_i)
                    in_bounds = abs_row < T_local
                    safe_row = in_bounds.select(abs_row, 0)
                    g_row_pf.append(
                        (
                            _load_vec(
                                cp_f32,
                                fx.slice(g_view, (safe_row, None)),
                                1,
                                fx.Float32,
                            ),
                            in_bounds,
                        )
                    )

            # -- GEMM1: b_v = w @ h_state, with the w_next prefetch interleaved.
            frag_bv.fill(0.0)

            # One 64-K block, one fx.gemm per 16-K tile: the K-tiles stay
            # individually addressable for the w_next split and sched_barrier.
            def _gemm1_kblock(kb):
                for ks in range_constexpr(K_STEPS_PER_BLOCK):
                    for half in range_constexpr(K_STEP // 16):
                        kj = ks * (K_STEP // 16) + half
                        kt = kb * K_TILES_PER_BLOCK + kj
                        fx.copy(
                            cp_lds_x4,
                            pS_w[None, None, (kj, kb)],
                            frag_w_rt[None, None, kt],
                        )
                        fx.copy(
                            cp_lds_x4, pS_h[None, None, kt], frag_h_rt[None, None, kt]
                        )
                        fx.gemm(
                            tiled_mma,
                            frag_bv,
                            frag_w[None, None, (kj, kb)],
                            frag_h[None, None, kt],
                            frag_bv,
                        )
                    # gfx942: keeps LLVM from clustering the per-ks b_frag
                    # ds_read across iterations (LDS port back-pressure).
                    if const_expr(SCHED_GFX942):
                        rocdl.sched_barrier(rocdl.mask_mfma)

            for kb in range_constexpr(GEMM1_PF_SPLIT):
                _gemm1_kblock(kb)

            next_i_t = i_t_i32 + 1
            next_chunk_base = next_i_t * BT
            w_next = _load_w_rows(next_chunk_base)

            # Remaining K-blocks; their MFMA hide the u/g/w_next HBM latency.
            for kb in range_constexpr(GEMM1_PF_SPLIT, NUM_K_BLOCKS):
                _gemm1_kblock(kb)

            # WAR barrier: GEMM1 is done reading the h_state panels, so gated_v
            # may now overwrite panel 1.
            gpu.barrier()

            if const_expr(USE_G):
                exp_g_last = _fast_exp(g_last)
                gate_elems = []
                for elem_i in range_constexpr(4):
                    g_row_val, in_bounds = g_row_pf[elem_i]
                    gate = _fast_exp(g_last - g_row_val)
                    gate_elems.append(in_bounds.select(gate, fx.Float32(0.0)))
                gate_vec = fx.Vector.from_elements(gate_elems, dtype=fx.Float32)
            else:
                # Padding rows must be masked or their v_new reaches GEMM2 and
                # corrupts the state; USE_G gets this free via gate=0.
                mask_elems = []
                for elem_i in range_constexpr(4):
                    in_bounds = _bt_abs_row(elem_i) < T_local
                    mask_elems.append(
                        in_bounds.select(fx.Float32(1.0), fx.Float32(0.0))
                    )
                gate_vec = fx.Vector.from_elements(mask_elems, dtype=fx.Float32)

            for idx in range_constexpr(N_REPEAT):
                bv_val = fx.Vector(frag_bv[None, None, idx].load())
                u_f32_elems = []
                for elem_i in range_constexpr(4):
                    u_f32_elems.append(u_prefetch[idx * 4 + elem_i].to(fx.Float32))
                u_f32 = fx.Vector.from_elements(u_f32_elems, dtype=fx.Float32)
                vn_val = u_f32 - bv_val

                if const_expr(SAVE_NEW_VALUE):
                    vn_bf16 = _f32x4_to_bf16x4(vn_val)
                    for elem_i in range_constexpr(4):
                        vn_bt_row = _bt_abs_row(elem_i)
                        if vn_bt_row < T_local:
                            bf16_v = vn_bf16[elem_i]
                            _store_vec(
                                cp_bf16,
                                fx.slice(vn_view, (vn_bt_row, u_cols[idx], None)),
                                bf16_v,
                                1,
                                fx.BFloat16,
                            )

                gated_val = vn_val * gate_vec
                gv_col = idx * 16 + lane_n
                _lds_write_x4(
                    fx.slice(sGV, (gv_col, (None, row_block))),
                    _f32x4_to_bf16x4(gated_val),
                )

            # Prefetch the next chunk's k; overlaps the barrier + h store below.
            k_next = _load_k_pairs(next_chunk_base)

            if const_expr(USE_G):
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for nr in range_constexpr(N_REPEAT):
                        acc_cell = frag_h_accs[kb][None, None, nr]
                        acc_cell.store(fx.Vector(acc_cell.load()) * exp_g_last)

            # Per-K decay: h[v, k] *= exp(gk_last[k]) at chunk end.
            if const_expr(USE_GK):
                for kb in range_constexpr(NUM_K_BLOCKS):
                    gk_elems = []
                    for elem_i in range_constexpr(4):
                        global_k = kb * 64 + wid * 16 + lane_m_base * 4 + elem_i
                        gk_raw = _load_vec(
                            cp_f32,
                            fx.slice(gk_view, (last_idx_raw, global_k, None)),
                            1,
                            fx.Float32,
                        )
                        gk_elems.append(_fast_exp(gk_raw))
                    gk_vec = fx.Vector.from_elements(gk_elems, dtype=fx.Float32)
                    for nr in range_constexpr(N_REPEAT):
                        acc_cell = frag_h_accs[kb][None, None, nr]
                        acc_cell.store(fx.Vector(acc_cell.load()) * gk_vec)

            gpu.barrier()

            # h snapshot store: the transpose buffer is read-only from here and
            # gated_v is in a different LDS region, so the two do not conflict.
            K_VECS = K // LOAD_VEC_WIDTH
            NUM_HT_VECS = BV * K_VECS
            for vbase in range_constexpr(0, NUM_HT_VECS, BLOCK_THREADS):
                vec_idx = vbase + tid
                kv = vec_idx % K_VECS
                v_loc = vec_idx // K_VECS
                kg_lo = kv * 2
                val_lo = _lds_read_x4(fx.slice(sHT, (v_loc, kg_lo, None)))
                val_hi = _lds_read_x4(fx.slice(sHT, (v_loc, kg_lo + 1, None)))
                vec8 = val_lo.shuffle(val_hi, [0, 1, 2, 3, 4, 5, 6, 7])
                v_global = i_v * BV + v_loc
                _store_vec(
                    cp_bf16x8,
                    fx.slice(h_view, (i_t_i32, v_global, kv, None)),
                    vec8,
                    LOAD_VEC_WIDTH,
                    fx.BFloat16,
                )

            # -- GEMM2: h += k^T @ v_new_gated, each 64-K block its own M tile,
            # k panel view and accumulator. A takes an fx.slice per k-tile.
            BT_STEPS = BT // K_STEP
            for kb in range_constexpr(NUM_K_BLOCKS):
                for bt_s in range_constexpr(BT_STEPS):
                    for half in range_constexpr(K_STEP // 16):
                        kt = bt_s * (K_STEP // 16) + half
                        frag_k[kb][None, None, kt].store(
                            _lds_read_x4(
                                fx.slice(
                                    sK[kb],
                                    (wid * 16 + lane_n, (None, kt * 4 + lane_m_base)),
                                )
                            )
                        )
                        fx.copy(
                            cp_lds_x4,
                            pS_gv[None, None, kt],
                            frag_gv_rt[None, None, kt],
                        )
                        fx.gemm(
                            tiled_mma,
                            frag_h_accs[kb],
                            frag_k[kb][None, None, kt],
                            frag_gv[None, None, kt],
                            frag_h_accs[kb],
                        )

            # Publish the prefetched w_next/k_next to LDS for the next
            # iteration; the barrier ensures GEMM2 is done with the old panels.
            has_next = next_chunk_base < T_local
            if has_next:
                gpu.barrier()
                _store_w_rows(w_next)
                _store_k_pairs(k_next)

            results = yield [
                frag_h_accs[kb].load() for kb in range_constexpr(NUM_K_BLOCKS)
            ]

        for kb in range_constexpr(NUM_K_BLOCKS):
            frag_h_accs[kb].store(fx.Vector(results[kb], (H_ACC_ELEMS,), fx.Float32))

        # -- Epilogue -- acc_val is already f32x4 with element i at K offset i,
        # so the final state stores as one buffer_store_dwordx4.
        if const_expr(STORE_FINAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for slot in range_constexpr(N_REPEAT):
                    acc_val = fx.Vector(frag_h_accs[kb][None, None, slot].load())
                    ht_col, ht_kgroup = _state_coord(kb, slot)
                    if const_expr(STATE_DTYPE_BF16):
                        out_vec = _f32x4_to_bf16x4(acc_val)
                    else:
                        out_vec = acc_val
                    _store_vec(
                        cp_state_x4,
                        fx.slice(ht_view, (ht_col, ht_kgroup, None)),
                        out_vec,
                        4,
                        state_num,
                    )

    # -- Host launcher ------------------------------------------------------
    @flyc.jit
    def launch_gdn_h(
        k_tensor: fx.Tensor,
        v_tensor: fx.Tensor,
        w_tensor: fx.Tensor,
        v_new_tensor: fx.Tensor,
        g_tensor: fx.Tensor,
        gk_tensor: fx.Tensor,
        h_tensor: fx.Tensor,
        h0_tensor: fx.Tensor,
        ht_tensor: fx.Tensor,
        cu_seqlens_tensor: fx.Tensor,
        chunk_offsets_tensor: fx.Tensor,
        state_indices_tensor: fx.Tensor,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
        grid_v: fx.Int32,
        grid_nh: fx.Int32,
        stream: fx.Stream,
    ):
        launcher = gdn_h_kernel(
            k_tensor,
            v_tensor,
            w_tensor,
            v_new_tensor,
            g_tensor,
            gk_tensor,
            h_tensor,
            h0_tensor,
            ht_tensor,
            cu_seqlens_tensor,
            chunk_offsets_tensor,
            state_indices_tensor,
            T_val,
            T_flat,
            N_val,
        )
        launcher.launch(
            grid=(grid_v, grid_nh, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_gdn_h


# NOTE: the host wrapper, BV autotune and kernel cache live in
# ``aiter.ops.flydsl.linear_attention_prefill_kernels`` (no torch/triton here).


__all__ = [
    "compile_chunk_gated_delta_h_mfma16_hip",
]
