# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GDN K5+K6 fused forward — gfx942 (CDNA3 / MI300X) FlyDSL kernel.

Fuses the inter-chunk hidden-state recurrence (K5) with the inter/intra-chunk
output (K6) into a single dispatch, keeping the ``h`` snapshot and gated
``v_new`` resident in LDS instead of round-tripping them through HBM. See
``docs/gdn_k5k6_fusion_plan.md``.

Per chunk t (serial over NT chunks), the fused body computes, in order:
  1. h snapshot -> lds_h  (kept in LDS; NOT written to HBM)
  2. v_new = u - w @ h                         (GEMM1, delta correction)
  3. gated decay + state update:
       v_new *= exp(g_last - g_cumsum)          (scalar g only)
       h      = h * exp(g_last) + k^T @ v_new    (GEMM2)
  4. output:
       o  = q @ h^T                              (GEMM3, inter-chunk)
       A  = tril( (q @ k^T) * exp(g_i - g_j) )   (GEMM4a + gate/mask)
       o  = scale * (o * exp(g)) + scale * (A @ v_new)   (GEMM4b)
     store o -> HBM [T_flat, H, V]

Phase 1: correctness-first. Restricted to NR_SPLIT == 1 (no wave-widened
variants); the [BT, BT] attention matrix ``A`` is staged through a dedicated
LDS scratch buffer between GEMM4a and GEMM4b (the A-operand of GEMM4b is the
transpose of GEMM4a's accumulator layout). ``lds_q`` aliases ``lds_w`` (w is
dead after GEMM1, q is loaded after), so the only new LDS is ``lds_A``.

Unchanged low-level helpers (fast exp/bf16, the MFMA wrapper, the group-major
XOR addressing scheme) are imported from the sibling K5 module so the two stay
in lockstep.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import as_ir_value, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl._mlir.dialects import vector as _vector

from .tensor_shim import GTensor, _to_raw
from .chunk_gated_delta_h_gfx942 import (
    _make_fast_exp,
    _mfma_bf16_16x16x16,
    _to_bf16_fast,
)


def compile_chunk_gated_delta_h_o_gfx942(
    *,
    K: int,
    V: int,
    BT: int = 64,
    BV: int = 32,
    H: int,
    Hg: int,
    SCALE: float,
    USE_G: bool = True,
    USE_GK: bool = False,
    USE_INITIAL_STATE: bool = True,
    STORE_FINAL_STATE: bool = True,
    IS_VARLEN: bool = True,
    WU_CONTIGUOUS: bool = True,
    STATE_DTYPE_BF16: bool = False,
    G_IS_LOG2_SCALED: bool = False,
    NR_SPLIT: int = 1,
):
    """Build the gfx942 fused GDN K5+K6 launcher for one compile-time config.

    Derived from ``compile_chunk_gated_delta_h_gfx942`` (K5). Differences:
      * adds ``q_tensor`` (query) and ``o_tensor`` (output) kernel params,
        plus ``Hg``/``SCALE`` for the K6 output stage;
      * drops the ``h`` HBM snapshot store and the ``v_new`` HBM store
        (both stay resident in LDS / registers);
      * appends GEMM3 (q@h), GEMM4a (q@k^T) + gate/mask, GEMM4b (A@v_new)
        and the ``o`` store.

    NR_SPLIT is fixed to 1 in Phase 1 (see module docstring).
    """
    assert K <= 256
    assert K % 64 == 0
    assert BV % 16 == 0
    assert BV <= 64, (
        f"gfx942 LDS budget caps BV at 64 (got BV={BV})."
    )
    NUM_K_BLOCKS = K // 64

    GRID_V = (V + BV - 1) // BV
    NXCD = 8

    _fast_exp = _make_fast_exp(G_IS_LOG2_SCALED)

    WARP_SIZE = 64
    WMMA_N = 16
    WMMA_K = 16
    N_REPEAT = BV // WMMA_N

    # -- Wave decomposition (NR_SPLIT: the wave-widening axis) --
    # wid_m owns a BT tile (GEMM1) / K tile (GEMM2); wid_n owns a slice of the
    # N_REPEAT (V) axis. NR_SPLIT=1 is the plain 4-wave kernel.
    M_WAVES = BT // 16
    assert N_REPEAT % NR_SPLIT == 0, (
        f"NR_SPLIT={NR_SPLIT} must divide N_REPEAT={N_REPEAT} (=BV/16); "
        f"BV={BV} supports NR_SPLIT in "
        f"{[s for s in (1, 2, 4, 8) if N_REPEAT % s == 0]}"
    )
    N_REPEAT_LOCAL = N_REPEAT // NR_SPLIT
    NUM_WARPS = M_WAVES * NR_SPLIT
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE
    assert BLOCK_THREADS <= 1024, (
        f"BLOCK_THREADS={BLOCK_THREADS} exceeds the gfx942 workgroup limit; "
        f"reduce NR_SPLIT."
    )
    # b_A (GEMM4a) is split across the wid_n waves by key-column tile; each wave
    # writes BT_STEPS_LOCAL of the BT // WMMA_K key-tiles into the shared lds_A.
    assert (BT // WMMA_K) % NR_SPLIT == 0, (
        f"NR_SPLIT={NR_SPLIT} must divide BT_STEPS={BT // WMMA_K} to split b_A "
        f"across the V-split waves"
    )
    BT_STEPS_LOCAL = (BT // WMMA_K) // NR_SPLIT
    NUM_H_ACCS = NUM_K_BLOCKS * N_REPEAT_LOCAL

    # -- Loop-carried gate/u prefetch (as in K5) --
    N_GATE_G = 5 if USE_G else 0  # g_last + 4 g_row
    N_GATE_GK = NUM_K_BLOCKS if USE_GK else 0
    N_U = N_REPEAT_LOCAL * 4
    N_GU = N_GATE_G + N_GATE_GK + N_U

    # -- LDS layout (group-major + XOR; see K5 kernel for the full rationale) --
    assert BT % 4 == 0 and K % 4 == 0
    assert K // 4 >= 16 and BT // 4 >= 16, "group-XOR needs >=16 groups per row"

    LDS_W_NG = K // 4
    LDS_W_ELEMS = BT * K
    LDS_KT_NG = BT // 4
    LDS_KT_ELEMS = K * BT
    LDS_VNT_NG = BT // 4
    LDS_VNT_ELEMS = BV * BT
    LDS_H_NG = K // 4
    LDS_H_ELEMS = BV * K
    # lds_A: [BT, BT] attention matrix, group-major + XOR like the others.
    LDS_A_NG = BT // 4
    LDS_A_ELEMS = BT * BT
    # lds_vn_raw: ungated v_new, transposed [BV, BT] exactly like lds_vnt. The
    # K6 attention output A @ v_new uses the ungated v_new (what K5 writes to
    # HBM before the exp(g_last - g) decay), whereas GEMM2's state update uses
    # the gated copy in lds_vnt -- so both must be kept.
    LDS_VN_RAW_ELEMS = BV * BT

    # LDS budget: the six buffers at BV=64 (K=128) total 72 KiB > 64 KiB. lds_A
    # (BT*BT = 4096 bf16 = 8 KiB) is written by GEMM4a, which runs strictly after
    # GEMM3 -- the last reader of lds_h -- so lds_A can reuse lds_h's storage once
    # a barrier separates the two. lds_A (4096 elems) fits inside lds_h
    # (BV*K = 8192 elems at BV=64). Alias only when needed (BV would otherwise
    # overflow); BV<=32 has room and keeps the buffers distinct (no extra barrier).
    _lds_total_kib = (
        LDS_W_ELEMS + LDS_KT_ELEMS + LDS_VNT_ELEMS + LDS_H_ELEMS
        + LDS_A_ELEMS + LDS_VN_RAW_ELEMS
    ) * 2 / 1024
    # Alias lds_A onto lds_h only when (a) the un-aliased layout would overflow
    # 64 KiB AND (b) lds_A actually fits inside lds_h. The latter holds only for
    # BV >= 32 (lds_h = BV*K; lds_A = BT*BT): at BV=16, lds_h = 2048 < lds_A =
    # 4096, so aliasing is impossible -- but BV=16 also never overflows, so it
    # keeps distinct buffers and no aliasing is needed.
    ALIAS_A_ONTO_H = _lds_total_kib > 64.0 and LDS_A_ELEMS <= LDS_H_ELEMS
    assert not (_lds_total_kib > 64.0 and not ALIAS_A_ONTO_H), (
        f"fused LDS {_lds_total_kib:.0f} KiB > 64 KiB and lds_A ({LDS_A_ELEMS}) "
        f"does not fit lds_h ({LDS_H_ELEMS}) to alias (BV={BV}, K={K})"
    )

    if ALIAS_A_ONTO_H:
        @fx.struct
        class SharedStorage:
            lds_w: fx.Array[fx.BFloat16, LDS_W_ELEMS, 16]
            lds_kt: fx.Array[fx.BFloat16, LDS_KT_ELEMS, 16]
            lds_vnt: fx.Array[fx.BFloat16, LDS_VNT_ELEMS, 16]
            lds_h: fx.Array[fx.BFloat16, LDS_H_ELEMS, 16]
            lds_vn_raw: fx.Array[fx.BFloat16, LDS_VN_RAW_ELEMS, 16]
    else:
        @fx.struct
        class SharedStorage:
            lds_w: fx.Array[fx.BFloat16, LDS_W_ELEMS, 16]
            lds_kt: fx.Array[fx.BFloat16, LDS_KT_ELEMS, 16]
            lds_vnt: fx.Array[fx.BFloat16, LDS_VNT_ELEMS, 16]
            lds_h: fx.Array[fx.BFloat16, LDS_H_ELEMS, 16]
            lds_A: fx.Array[fx.BFloat16, LDS_A_ELEMS, 16]
            lds_vn_raw: fx.Array[fx.BFloat16, LDS_VN_RAW_ELEMS, 16]

    # Cooperative load parameters (bf16x8 = dwordx4) -- identical to K5.
    LOAD_VEC_WIDTH = 8
    THREADS_PER_ROW_64 = 64 // LOAD_VEC_WIDTH
    ROWS_PER_BATCH_64 = BLOCK_THREADS // THREADS_PER_ROW_64
    W_BATCHED = ROWS_PER_BATCH_64 <= BT and BT % ROWS_PER_BATCH_64 == 0
    NUM_LOAD_BATCHES_64 = BT // ROWS_PER_BATCH_64 if W_BATCHED else 0

    W_THREADS_PER_ROW = K // LOAD_VEC_WIDTH
    W_SLOTS = BT * W_THREADS_PER_ROW
    assert W_SLOTS % BLOCK_THREADS == 0
    W_LOADS_PER_THREAD = (
        NUM_K_BLOCKS * NUM_LOAD_BATCHES_64 if W_BATCHED else W_SLOTS // BLOCK_THREADS
    )

    K_STEPS_PER_BLOCK = 64 // WMMA_K  # 4
    BT_STEPS = BT // WMMA_K  # 4

    # -- k store-transpose decomposition (as in K5) --
    K_VEC_WIDTH = min(LOAD_VEC_WIDTH, max(2, (BT // 4) * K // BLOCK_THREADS))
    K_COL_GROUPS = K // K_VEC_WIDTH
    K_ROW_QUADS = BT // 4
    K_XPOSE_SLOTS = K_ROW_QUADS * K_COL_GROUPS
    K_PACKED_XPOSE = K_XPOSE_SLOTS % BLOCK_THREADS == 0
    K_SLOTS_PER_THREAD = K_XPOSE_SLOTS // BLOCK_THREADS if K_PACKED_XPOSE else 0
    K_ROW_QUAD_STRIDE = BLOCK_THREADS // K_COL_GROUPS if K_PACKED_XPOSE else 0

    _kernel_deco_kwargs = (
        {} if BLOCK_THREADS == 256 else {"known_block_size": [BLOCK_THREADS, 1, 1]}
    )

    @flyc.kernel(name="chunk_gdn_fwd_h_o_flydsl_vk", **_kernel_deco_kwargs)
    def gdn_h_o_kernel(
        q_tensor: fx.Tensor,
        k_tensor: fx.Tensor,
        w_tensor: fx.Tensor,
        u_tensor: fx.Tensor,
        g_tensor: fx.Tensor,
        gk_tensor: fx.Tensor,
        o_tensor: fx.Tensor,
        h0_tensor: fx.Tensor,
        ht_tensor: fx.Tensor,
        cu_seqlens_tensor: fx.Tensor,
        chunk_offsets_tensor: fx.Tensor,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
    ):
        # -- Chiplet (XCD) remap (identical to K5) --
        if const_expr(NXCD > 0):
            grid_nh_rt = N_val * fx.Int32(H)
            grid_total = fx.Int32(GRID_V) * grid_nh_rt
            xy = fx.block_idx.x + fx.Int32(GRID_V) * fx.block_idx.y
            xcd = xy % fx.Int32(NXCD)
            cycle = fx.Int32(NXCD) * fx.Int32(GRID_V)
            last_full = (grid_total // cycle) * cycle
            local_id = xy // fx.Int32(NXCD)
            chunk_idx = local_id // fx.Int32(GRID_V)
            pos = local_id % fx.Int32(GRID_V)
            remapped = chunk_idx * cycle + xcd * fx.Int32(GRID_V) + pos
            logical = (xy < last_full).select(remapped, xy)
            i_v = logical % fx.Int32(GRID_V)
            i_nh = logical // fx.Int32(GRID_V)
        else:
            i_v = fx.block_idx.x
            i_nh = fx.block_idx.y

        i_n = i_nh // fx.Int32(H)
        i_h = i_nh % fx.Int32(H)

        tid = fx.thread_idx.x
        wid = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)

        # Wave split. 
        #   wid_m: BT tile (GEMM1) / K tile (GEMM2)
        #   wid_n: this wave's slice of the N_REPEAT (V) axis and of b_A's key-col tiles.
        if const_expr(NR_SPLIT == 1):
            wid_m = wid
        else:
            wid_m = wid % fx.Int32(M_WAVES)
        wid_n = wid // fx.Int32(M_WAVES)

        def _nr_v(nr_local):
            """V-offset (elements) of this wave's local V tile ``nr_local``."""
            if const_expr(NR_SPLIT == 1):
                return fx.Int32(nr_local * 16)
            return wid_n * fx.Int32(N_REPEAT_LOCAL * 16) + fx.Int32(nr_local * 16)

        q_ = GTensor(q_tensor, dtype=T.bf16, shape=(-1,))
        k_ = GTensor(k_tensor, dtype=T.bf16, shape=(-1,))
        w_ = GTensor(w_tensor, dtype=T.bf16, shape=(-1,))
        u_ = GTensor(u_tensor, dtype=T.bf16, shape=(-1,))
        o_ = GTensor(o_tensor, dtype=T.bf16, shape=(-1,))
        g_ = GTensor(g_tensor, dtype=T.f32, shape=(-1,))
        if const_expr(USE_GK):
            gk_ = GTensor(gk_tensor, dtype=T.f32, shape=(-1,))

        state_t = T.bf16 if STATE_DTYPE_BF16 else T.f32
        if const_expr(USE_INITIAL_STATE):
            h0_ = GTensor(h0_tensor, dtype=state_t, shape=(-1,))
        if const_expr(STORE_FINAL_STATE):
            ht_ = GTensor(ht_tensor, dtype=state_t, shape=(-1,))

        if const_expr(IS_VARLEN):
            cu_ = GTensor(cu_seqlens_tensor, dtype=T.i32, shape=(-1,))
            co_ = GTensor(chunk_offsets_tensor, dtype=T.i32, shape=(-1,))

        # -- LDS views --
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_w_ptr = lds.lds_w.ptr
        lds_kt_ptr = lds.lds_kt.ptr
        lds_vnt_ptr = lds.lds_vnt.ptr
        lds_h_ptr = lds.lds_h.ptr
        lds_vn_raw_ptr = lds.lds_vn_raw.ptr
        # lds_A aliases lds_h when LDS is tight (BV=64): GEMM3 (last lds_h reader)
        # runs before GEMM4a (lds_A writer), separated by a barrier below.
        if const_expr(ALIAS_A_ONTO_H):
            lds_A_ptr = lds_h_ptr
        else:
            lds_A_ptr = lds.lds_A.ptr
        # q aliases w: w is dead after GEMM1, q is loaded after GEMM2.
        lds_q_ptr = lds_w_ptr

        # -- Group-major + XOR LDS addressing (identical to K5) --
        def _grp_idx(row, grp, cols, ng):
            mask = (row ^ (row >> fx.Int32(3))) & fx.Int32(ng - 1)
            return row * fx.Int32(cols) + ((grp ^ mask) * fx.Int32(4))

        v4bf16_type = T.vec(4, T.bf16)

        def _lds_h_idx(v_local, k_grp):
            return _grp_idx(v_local, k_grp, K, LDS_H_NG)

        def _lds_w_idx(bt_row, k_grp):
            return _grp_idx(bt_row, k_grp, K, LDS_W_NG)

        def _lds_kt_idx(k_row, bt_grp):
            return _grp_idx(k_row, bt_grp, BT, LDS_KT_NG)

        def _lds_vnt_idx(v_local, bt_grp):
            return _grp_idx(v_local, bt_grp, BT, LDS_VNT_NG)

        def _lds_A_idx(bt_row, bt_grp):
            return _grp_idx(bt_row, bt_grp, BT, LDS_A_NG)

        # -- Cooperative load decomposition (identical to K5) --
        if const_expr(W_BATCHED):
            load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_64)
            load_col_base = (tid % fx.Int32(THREADS_PER_ROW_64)) * fx.Int32(
                LOAD_VEC_WIDTH
            )

            def _w_slot(i_load):
                kb, batch = divmod(i_load, NUM_LOAD_BATCHES_64)
                row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                return row, fx.Int32(kb * 64) + load_col_base

        else:

            def _w_slot(i_load):
                s = fx.Int32(i_load * BLOCK_THREADS) + tid
                row = s // fx.Int32(W_THREADS_PER_ROW)
                col_grp = s % fx.Int32(W_THREADS_PER_ROW)
                return row, col_grp * fx.Int32(LOAD_VEC_WIDTH)

        if const_expr(K_PACKED_XPOSE):
            kx_col_base = (tid % fx.Int32(K_COL_GROUPS)) * fx.Int32(K_VEC_WIDTH)
            kx_row_quad = tid // fx.Int32(K_COL_GROUPS)

        # -- Prologue: compute bos, T_local, NT, boh --
        if const_expr(IS_VARLEN):
            bos = cu_[fx.Int64(i_n)]
            eos = cu_[fx.Int64(i_n) + fx.Int64(1)]
            T_local = eos - bos
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)
        else:
            bos = i_n * T_val
            T_local = T_val
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)

        # -- Base pointer offsets (element counts) --
        gqa_ratio = H // Hg
        i_hg = i_h // fx.Int32(gqa_ratio)
        k_base = (bos * fx.Int32(Hg) + i_hg) * fx.Int32(K)
        stride_k = fx.Int32(Hg * K)
        # q shares k's [B, T, Hg, K] layout.
        q_base = k_base
        stride_q = stride_k

        if const_expr(WU_CONTIGUOUS):
            if const_expr(IS_VARLEN):
                v_base = (i_h * T_flat + bos) * fx.Int32(V)
                w_base = (i_h * T_flat + bos) * fx.Int32(K)
            else:
                v_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(V)
                w_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(K)
            stride_v = fx.Int32(V)
            stride_w = fx.Int32(K)
        else:
            v_base = (bos * fx.Int32(H) + i_h) * fx.Int32(V)
            w_base = (bos * fx.Int32(H) + i_h) * fx.Int32(K)
            stride_v = fx.Int32(H * V)
            stride_w = fx.Int32(H * K)

        # o is token-major [B, T_flat, H, V] (matches Triton K6 output).
        o_base = (bos * fx.Int32(H) + i_h) * fx.Int32(V)
        stride_o = fx.Int32(H * V)

        if const_expr(USE_INITIAL_STATE):
            h0_base = i_nh * fx.Int32(V * K)
        if const_expr(STORE_FINAL_STATE):
            ht_base = i_nh * fx.Int32(V * K)

        # -- MFMA lane mapping for 16x16 tiles --
        lane_n = lane % fx.Int32(16)
        lane_m_base = lane // fx.Int32(16)

        # -- Initialize h accumulators --
        acc_zero = fx.full(4, 0.0, fx.Float32)
        h_accs = []
        for _kb in range_constexpr(NUM_K_BLOCKS):
            for _nr in range_constexpr(N_REPEAT_LOCAL):
                h_accs.append(acc_zero)

        if const_expr(USE_INITIAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT_LOCAL):
                    h0_col = i_v * fx.Int32(BV) + _nr_v(nr) + lane_n
                    h0_row_base = (
                        fx.Int32(kb * 64)
                        + wid_m * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                    )
                    h0_off_base = h0_base + h0_col * fx.Int32(K) + h0_row_base
                    loaded_vec = h0_.vec_load((fx.Int64(h0_off_base),), 4)
                    if const_expr(STATE_DTYPE_BF16):
                        loaded_vec = loaded_vec.extf(T.f32x4)
                    acc_idx = kb * N_REPEAT_LOCAL + nr
                    h_accs[acc_idx] = h_accs[acc_idx] + loaded_vec

        NUM_W_LOADS = W_LOADS_PER_THREAD

        def _load_gate_u(it_i32):
            """Issue chunk ``it_i32``'s g/gk/u loads; return them as a flat list.

            For the fused kernel the g loads additionally include the per-row
            cumsum ``g_row`` values (already loaded by K5) which K6's output
            gate reuses -- no extra loads are needed for GEMM3/GEMM4a's scalar
            gate beyond what K5 already prefetches.
            """
            out = []
            next_end = (it_i32 + fx.Int32(1)) * fx.Int32(BT)
            last_idx = (next_end < T_local).select(next_end, T_local) - fx.Int32(1)
            row_base = (
                it_i32 * fx.Int32(BT)
                + wid_m * fx.Int32(16)
                + lane_m_base * fx.Int32(4)
            )
            if const_expr(USE_G):
                out.append(g_[fx.Int64(i_h * T_flat + (bos + last_idx))])
                for elem_i in range_constexpr(4):
                    abs_row = row_base + fx.Int32(elem_i)
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    out.append(g_[fx.Int64(i_h * T_flat + (bos + safe_row))])
            if const_expr(USE_GK):
                gk_chunk_base = (bos + last_idx) * fx.Int32(H * K) + i_h * fx.Int32(K)
                for kb in range_constexpr(NUM_K_BLOCKS):
                    quad = (
                        gk_chunk_base
                        + fx.Int32(kb * 64)
                        + wid_m * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                    )
                    out.append(gk_.vec_load((fx.Int64(quad),), 4))
            for nr in range_constexpr(N_REPEAT_LOCAL):
                u_col = i_v * fx.Int32(BV) + _nr_v(nr) + lane_n
                for elem_i in range_constexpr(4):
                    abs_row = row_base + fx.Int32(elem_i)
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    u_off = v_base + safe_row * stride_v + u_col
                    out.append(u_.vec_load((fx.Int64(u_off),), 1))
            return out

        # -- Prologue: pre-load first chunk's w + gate/u data --
        i_t0_i32 = fx.Int32(0)
        w_prefetch_init = []
        for i_load in range_constexpr(W_LOADS_PER_THREAD):
            row, col = _w_slot(i_load)
            abs_row = i_t0_i32 * fx.Int32(BT) + row
            safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
            g_off = w_base + safe_row * stride_w + col
            w_prefetch_init.append(w_.vec_load((fx.Int64(g_off),), LOAD_VEC_WIDTH))

        gu_prefetch_init = _load_gate_u(i_t0_i32)

        init_state = (
            [_to_raw(v) for v in h_accs]
            + [_to_raw(v) for v in w_prefetch_init]
            + [_to_raw(v) for v in gu_prefetch_init]
        )
        c_zero = fx.Int64(0)
        c_one = fx.Int64(1)
        nt_idx = fx.Int64(NT)

        for i_t, state in range(c_zero, nt_idx, c_one, init=init_state):
            h_accs_in = list(state[:NUM_H_ACCS])
            w_prefetch_all = list(state[NUM_H_ACCS : NUM_H_ACCS + NUM_W_LOADS])
            gu_prefetch_all = list(state[NUM_H_ACCS + NUM_W_LOADS :])
            i_t_i32 = fx.Int32(i_t)

            # -- w LDS write offsets (group-major [BT][K/4][4] + XOR) --
            w_prefetch_lds_all = []
            for i_load in range_constexpr(W_LOADS_PER_THREAD):
                row, col = _w_slot(i_load)
                grp = col // fx.Int32(4)
                w_prefetch_lds_all.append(
                    (_lds_w_idx(row, grp), _lds_w_idx(row, grp + fx.Int32(1)))
                )

            # -- Store h snapshot to LDS (group-major [BV][K/4][4] + XOR) --
            # Unlike K5, this stays in LDS only; there is no LDS->HBM drain.
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT_LOCAL):
                    acc_idx = kb * N_REPEAT_LOCAL + nr
                    acc_val = h_accs_in[acc_idx]
                    lds_h_v = _nr_v(nr) + lane_n
                    lds_h_g = fx.Int32(kb * 16) + wid_m * fx.Int32(4) + lane_m_base
                    fx.ptr_store(
                        _to_bf16_fast(acc_val, 4),
                        lds_h_ptr + _lds_h_idx(lds_h_v, lds_h_g),
                    )

            gpu.barrier()

            # -- Store prefetched w to LDS (two b64 halves per bf16x8) --
            for i_wp in range_constexpr(NUM_W_LOADS):
                wvec = w_prefetch_all[i_wp]
                off_lo, off_hi = w_prefetch_lds_all[i_wp]
                lo = fx.Vector.from_elements(
                    [wvec[e] for e in range_constexpr(4)], dtype=fx.BFloat16
                )
                hi = fx.Vector.from_elements(
                    [wvec[4 + e] for e in range_constexpr(4)], dtype=fx.BFloat16
                )
                fx.ptr_store(lo, lds_w_ptr + off_lo)
                fx.ptr_store(hi, lds_w_ptr + off_hi)

            gpu.barrier()

            # -- k prefetch (issued now, stored transposed after GEMM1) --
            k_prefetch = []
            k_prefetch_lds_t = []
            if const_expr(K_PACKED_XPOSE):
                for s in range_constexpr(K_SLOTS_PER_THREAD):
                    row_quad = kx_row_quad + fx.Int32(s * K_ROW_QUAD_STRIDE)
                    quad_rows = []
                    for j in range_constexpr(4):
                        row = row_quad * fx.Int32(4) + fx.Int32(j)
                        abs_row = i_t_i32 * fx.Int32(BT) + row
                        safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                        k_off = k_base + safe_row * stride_k + kx_col_base
                        quad_rows.append(k_.vec_load((fx.Int64(k_off),), K_VEC_WIDTH))
                    k_prefetch.append(quad_rows)
                    k_prefetch_lds_t.append(row_quad)
            else:
                for i_load in range_constexpr(W_LOADS_PER_THREAD):
                    row, col = _w_slot(i_load)
                    abs_row = i_t_i32 * fx.Int32(BT) + row
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    k_off = k_base + safe_row * stride_k + col
                    k_prefetch.append(k_.vec_load((fx.Int64(k_off),), LOAD_VEC_WIDTH))
                    k_prefetch_lds_t.append((row, col))

            # -- g / gk / u unpack (prefetched last iter) --
            gu_all = list(gu_prefetch_all)

            def _as_f32(v):
                return fx.Float32(as_ir_value(v))

            if const_expr(USE_G):
                g_last_val = _as_f32(gu_all[0])
                g_row_raw = [_as_f32(v) for v in gu_all[1:5]]
            if const_expr(USE_GK):
                gk_quads = gu_all[N_GATE_G : N_GATE_G + N_GATE_GK]
            u_prefetch = gu_all[N_GATE_G + N_GATE_GK :]

            # -- GEMM1: bv = w @ h  (contraction over K) --
            bv_accs = []
            for _nr in range_constexpr(N_REPEAT_LOCAL):
                bv_accs.append(fx.full(4, 0.0, fx.Float32))

            for kb in range_constexpr(NUM_K_BLOCKS):
                for ks in range_constexpr(K_STEPS_PER_BLOCK):
                    w_row = wid_m * fx.Int32(16) + lane_n
                    w_g = fx.Int32(kb * 16 + ks * (WMMA_K // 4)) + lane_m_base
                    a_frag = fx.ptr_load(
                        lds_w_ptr + _lds_w_idx(w_row, w_g), result_type=v4bf16_type
                    )
                    for nr in range_constexpr(N_REPEAT_LOCAL):
                        h_v = _nr_v(nr) + lane_n
                        h_g = fx.Int32(kb * 16 + ks * (WMMA_K // 4)) + lane_m_base
                        b_frag = fx.ptr_load(
                            lds_h_ptr + _lds_h_idx(h_v, h_g), result_type=v4bf16_type
                        )
                        bv_accs[nr] = _mfma_bf16_16x16x16(a_frag, b_frag, bv_accs[nr])

            # -- v_new = u - bv --
            vn_frags = []
            for nr in range_constexpr(N_REPEAT_LOCAL):
                bv_val = bv_accs[nr]
                u_f32_elems = []
                for elem_i in range_constexpr(4):
                    u_bf16 = fx.BFloat16(u_prefetch[nr * 4 + elem_i])
                    u_f32_elems.append(u_bf16.to(fx.Float32))
                u_f32 = fx.Vector.from_elements(u_f32_elems, dtype=fx.Float32)
                vn_frags.append(u_f32 - bv_val)

            # -- Tail-chunk row mask --
            row_mask_elems = []
            for elem_i in range_constexpr(4):
                bt_row = (
                    i_t_i32 * fx.Int32(BT)
                    + wid_m * fx.Int32(16)
                    + lane_m_base * fx.Int32(4)
                    + fx.Int32(elem_i)
                )
                in_bounds = bt_row < T_local
                row_mask_elems.append(in_bounds.select(fx.Float32(1.0), fx.Float32(0.0)))
            row_mask_vec = fx.Vector.from_elements(row_mask_elems, dtype=fx.Float32)
            for nr in range_constexpr(N_REPEAT_LOCAL):
                vn_frags[nr] = vn_frags[nr] * row_mask_vec

            # -- Snapshot the UNGATED v_new for K6's A @ v_new (GEMM4b) --
            # K6 reads v_new before the exp(g_last - g) decay; keep a copy now,
            # since the gate below mutates vn_frags in place for GEMM2.
            vn_raw_frags = [vn_frags[nr] for nr in range_constexpr(N_REPEAT_LOCAL)]

            # -- Gating (state + v_new) --
            if const_expr(USE_G):
                exp_g_last = _fast_exp(g_last_val)
                gate_elems = []
                for elem_i in range_constexpr(4):
                    abs_row = (
                        i_t_i32 * fx.Int32(BT)
                        + wid_m * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                        + fx.Int32(elem_i)
                    )
                    in_bounds = abs_row < T_local
                    gate = _fast_exp(g_last_val - g_row_raw[elem_i])
                    gate_elems.append(in_bounds.select(gate, fx.Float32(0.0)))
                gate_vec = fx.Vector.from_elements(gate_elems, dtype=fx.Float32)
                for nr in range_constexpr(N_REPEAT_LOCAL):
                    vn_frags[nr] = vn_frags[nr] * gate_vec
                exp_g_last_vec = fx.full(4, fx.Float32(exp_g_last), fx.Float32)
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for nr in range_constexpr(N_REPEAT_LOCAL):
                        acc_idx = kb * N_REPEAT_LOCAL + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * exp_g_last_vec

            if const_expr(USE_GK):
                for kb in range_constexpr(NUM_K_BLOCKS):
                    gk_q = fx.Vector(as_ir_value(gk_quads[kb]))
                    gk_vec = fx.Vector.from_elements(
                        [_fast_exp(gk_q[elem_i]) for elem_i in range_constexpr(4)],
                        dtype=fx.Float32,
                    )
                    for nr in range_constexpr(N_REPEAT_LOCAL):
                        acc_idx = kb * N_REPEAT_LOCAL + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * gk_vec

            # -- Store gated v_new (GEMM2) + ungated v_new (GEMM4b) transposed
            #    as [V, BT] into lds_vnt / lds_vn_raw --
            for nr in range_constexpr(N_REPEAT_LOCAL):
                vnt_v = _nr_v(nr) + lane_n
                vnt_g = wid_m * fx.Int32(4) + lane_m_base
                fx.ptr_store(
                    _to_bf16_fast(vn_frags[nr], 4),
                    lds_vnt_ptr + _lds_vnt_idx(vnt_v, vnt_g),
                )
                fx.ptr_store(
                    _to_bf16_fast(vn_raw_frags[nr], 4),
                    lds_vn_raw_ptr + _lds_vnt_idx(vnt_v, vnt_g),
                )

            # -- Store k transposed as [K, BT] into lds_kt --
            if const_expr(K_PACKED_XPOSE):
                for s in range_constexpr(K_SLOTS_PER_THREAD):
                    quad_rows = k_prefetch[s]
                    row_quad = k_prefetch_lds_t[s]
                    for e in range_constexpr(K_VEC_WIDTH):
                        bt_grp = fx.Vector.from_elements(
                            [quad_rows[j][e] for j in range_constexpr(4)],
                            dtype=fx.BFloat16,
                        )
                        fx.ptr_store(
                            bt_grp,
                            lds_kt_ptr
                            + _lds_kt_idx(kx_col_base + fx.Int32(e), row_quad),
                        )
            else:
                for i_kp in range_constexpr(NUM_W_LOADS):
                    kvec = k_prefetch[i_kp]
                    row, kcol = k_prefetch_lds_t[i_kp]
                    row_g = row // fx.Int32(4)
                    row_e = row % fx.Int32(4)
                    for e in range_constexpr(LOAD_VEC_WIDTH):
                        kt_idx = _lds_kt_idx(kcol + fx.Int32(e), row_g) + row_e
                        fx.ptr_store(kvec[e], lds_kt_ptr + kt_idx)

            gpu.barrier()

            # -- GEMM2: h += k^T @ v_new  (contraction over BT) --
            for kb in range_constexpr(NUM_K_BLOCKS):
                for bt_s in range_constexpr(BT_STEPS):
                    k_m = fx.Int32(kb * 64) + wid_m * fx.Int32(16) + lane_n
                    k_g = fx.Int32(bt_s * (WMMA_K // 4)) + lane_m_base
                    k_a_frag = fx.ptr_load(
                        lds_kt_ptr + _lds_kt_idx(k_m, k_g), result_type=v4bf16_type
                    )
                    for nr in range_constexpr(N_REPEAT_LOCAL):
                        vn_v = _nr_v(nr) + lane_n
                        vn_g = fx.Int32(bt_s * (WMMA_K // 4)) + lane_m_base
                        vn_b_frag = fx.ptr_load(
                            lds_vnt_ptr + _lds_vnt_idx(vn_v, vn_g),
                            result_type=v4bf16_type,
                        )
                        acc_idx = kb * N_REPEAT_LOCAL + nr
                        h_accs_in[acc_idx] = _mfma_bf16_16x16x16(
                            k_a_frag, vn_b_frag, h_accs_in[acc_idx]
                        )

            # =============================================================== #
            # K6 output stage. h[t] is still resident in lds_h (the snapshot,
            # NOT the GEMM2-updated h_accs_in); lds_kt holds k^T; lds_vnt holds
            # gated v_new -- exactly the operands GEMM3/GEMM4 need.
            # =============================================================== #

            # -- Load q into lds_q (aliases lds_w, dead after GEMM1) --
            gpu.barrier()  # protect lds_w readers (GEMM1) before overwrite
            q_prefetch = []
            for i_load in range_constexpr(W_LOADS_PER_THREAD):
                row, col = _w_slot(i_load)
                abs_row = i_t_i32 * fx.Int32(BT) + row
                safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                qoff = q_base + safe_row * stride_q + col
                q_prefetch.append(q_.vec_load((fx.Int64(qoff),), LOAD_VEC_WIDTH))
            for i_qp in range_constexpr(NUM_W_LOADS):
                qvec = q_prefetch[i_qp]
                row, col = _w_slot(i_qp)
                grp = col // fx.Int32(4)
                lo = fx.Vector.from_elements(
                    [qvec[e] for e in range_constexpr(4)], dtype=fx.BFloat16
                )
                hi = fx.Vector.from_elements(
                    [qvec[4 + e] for e in range_constexpr(4)], dtype=fx.BFloat16
                )
                fx.ptr_store(lo, lds_q_ptr + _lds_w_idx(row, grp))
                fx.ptr_store(hi, lds_q_ptr + _lds_w_idx(row, grp + fx.Int32(1)))

            gpu.barrier()

            # -- GEMM3: o = q @ h^T  (contraction over K) --
            # Run FIRST so lds_h is fully consumed before GEMM4a writes lds_A;
            # this is what lets Lever 2 alias lds_A onto the (now-dead) lds_h at
            # BV=64. Same fragment structure as GEMM1: A-frag=q, B-frag=h.
            # Output o_accs[nr][e] = o[m=wid_m*16+lane_m_base*4+e, n=nr*16+lane_n].
            o_accs = []
            for _nr in range_constexpr(N_REPEAT_LOCAL):
                o_accs.append(fx.full(4, 0.0, fx.Float32))
            for kb in range_constexpr(NUM_K_BLOCKS):
                for ks in range_constexpr(K_STEPS_PER_BLOCK):
                    q_row = wid_m * fx.Int32(16) + lane_n
                    q_g = fx.Int32(kb * 16 + ks * (WMMA_K // 4)) + lane_m_base
                    qa_frag = fx.ptr_load(
                        lds_q_ptr + _lds_w_idx(q_row, q_g), result_type=v4bf16_type
                    )
                    for nr in range_constexpr(N_REPEAT_LOCAL):
                        h_v = _nr_v(nr) + lane_n
                        h_g = fx.Int32(kb * 16 + ks * (WMMA_K // 4)) + lane_m_base
                        hb_frag = fx.ptr_load(
                            lds_h_ptr + _lds_h_idx(h_v, h_g), result_type=v4bf16_type
                        )
                        o_accs[nr] = _mfma_bf16_16x16x16(qa_frag, hb_frag, o_accs[nr])

            # When lds_A aliases lds_h, all waves must finish reading lds_h
            # (GEMM1 + GEMM3 above) before GEMM4a overwrites it as lds_A.
            if const_expr(ALIAS_A_ONTO_H):
                gpu.barrier()

            # -- GEMM4a: A = q @ k^T, fused per key-tile (compute -> gate/mask ->
            #    store to lds_A -> free), so at most one f32x4 A tile is live. --
            # A[i,j] = sum_k q[i,k]*k[j,k]. MFMA m=i (query row), n=j (key row),
            # contraction=k. A-op = q[i,k] (lds_q); B-op element e = k[j=lane_n,
            # k=kb*64+ks*16+lane_m_base*4+e]. lds_kt is [K,BT] (BT contiguous),
            # so the 4 contraction elems are one row apart -> 4 scalar reads.
            #
            # Wave split (NR_SPLIT>1): b_A is V-independent, so instead of every
            # wid_n wave recomputing all of A, each wid_n owns BT_STEPS_LOCAL of
            # the BT_STEPS key-column tiles and writes its slice to the shared
            # lds_A. Together the M_WAVES x NR_SPLIT waves fill all of A with no
            # redundant compute; a barrier (below) precedes GEMM4b's full read.
            def _kt_scalar(k_row, bt):
                bt_g = bt // fx.Int32(4)
                bt_e = bt % fx.Int32(4)
                return fx.ptr_load(
                    lds_kt_ptr + _lds_kt_idx(k_row, bt_g) + bt_e,
                    result_type=T.bf16,
                )

            for bt_l in range_constexpr(BT_STEPS_LOCAL):
                a_acc = fx.full(4, 0.0, fx.Float32)
                # runtime key-tile index for this wave (compile-time when
                # NR_SPLIT==1, since wid_n is then identically 0).
                if const_expr(NR_SPLIT == 1):
                    bt_base = fx.Int32(bt_l * 16)
                else:
                    bt_base = (
                        wid_n * fx.Int32(BT_STEPS_LOCAL * 16) + fx.Int32(bt_l * 16)
                    )
                bt_col = bt_base + lane_n  # n = key row j
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for ks in range_constexpr(K_STEPS_PER_BLOCK):
                        q_row = wid_m * fx.Int32(16) + lane_n
                        q_g = fx.Int32(kb * 16 + ks * (WMMA_K // 4)) + lane_m_base
                        qa_frag = fx.ptr_load(
                            lds_q_ptr + _lds_w_idx(q_row, q_g),
                            result_type=v4bf16_type,
                        )
                        kb_frag = fx.Vector.from_elements(
                            [
                                _kt_scalar(
                                    fx.Int32(kb * 64 + ks * 16)
                                    + lane_m_base * fx.Int32(4)
                                    + fx.Int32(e),
                                    bt_col,
                                )
                                for e in range_constexpr(4)
                            ],
                            dtype=fx.BFloat16,
                        )
                        a_acc = _mfma_bf16_16x16x16(qa_frag, kb_frag, a_acc)

                # gate (USE_G only) + causal mask, then store this tile to lds_A.
                # acc element e is the query ROW i = wid_m*16+lane_m_base*4+e;
                # column j = bt_col is fixed per lane.
                col_abs = i_t_i32 * fx.Int32(BT) + bt_col
                if const_expr(USE_G):
                    col_safe = (col_abs < T_local).select(col_abs, fx.Int32(0))
                    g_col = _as_f32(g_[fx.Int64(i_h * T_flat + (bos + col_safe))])
                for e in range_constexpr(4):
                    row_tok = (
                        wid_m * fx.Int32(16) + lane_m_base * fx.Int32(4) + fx.Int32(e)
                    )
                    row_abs = i_t_i32 * fx.Int32(BT) + row_tok
                    causal = (row_tok >= bt_col) & (row_abs < T_local) & (
                        col_abs < T_local
                    )
                    if const_expr(USE_G):
                        gate = _fast_exp(g_row_raw[e] - g_col)
                        a_val = a_acc[e] * causal.select(gate, fx.Float32(0.0))
                    else:
                        # gk path: no output gate; causal mask only.
                        a_val = a_acc[e] * causal.select(
                            fx.Float32(1.0), fx.Float32(0.0)
                        )
                    a_row = row_tok
                    a_g = bt_col // fx.Int32(4)
                    a_e = bt_col % fx.Int32(4)
                    fx.ptr_store(
                        _to_bf16_fast(a_val),
                        lds_A_ptr + _lds_A_idx(a_row, a_g) + a_e,
                    )

            # -- Inter-chunk gate on o: o *= exp(g_row) (per query row) --
            if const_expr(USE_G):
                og_elems = []
                for e in range_constexpr(4):
                    og_elems.append(_fast_exp(g_row_raw[e]))
                og_vec = fx.Vector.from_elements(og_elems, dtype=fx.Float32)
                for nr in range_constexpr(N_REPEAT_LOCAL):
                    o_accs[nr] = o_accs[nr] * og_vec

            gpu.barrier()

            # -- GEMM4b: o += A @ v_new_ungated  (contraction over BT) --
            # A-frag: A[m=query row, contraction=key BT] from lds_A.
            # B-frag: v_new[contraction=key BT, n=V] from lds_vn_raw.
            #   IMPORTANT: the K6 attention term uses the *ungated* v_new (the
            #   value K5 writes to HBM, before the exp(g_last - g) decay), NOT
            #   the gated v_new that GEMM2 consumes. lds_vnt holds the gated
            #   copy; lds_vn_raw holds the ungated copy for exactly this GEMM.
            for bt_s in range_constexpr(BT_STEPS):
                a_m = wid_m * fx.Int32(16) + lane_n
                a_g = fx.Int32(bt_s * (WMMA_K // 4)) + lane_m_base
                a_frag = fx.ptr_load(
                    lds_A_ptr + _lds_A_idx(a_m, a_g), result_type=v4bf16_type
                )
                for nr in range_constexpr(N_REPEAT_LOCAL):
                    vn_v = _nr_v(nr) + lane_n
                    vn_g = fx.Int32(bt_s * (WMMA_K // 4)) + lane_m_base
                    vn_b_frag = fx.ptr_load(
                        lds_vn_raw_ptr + _lds_vnt_idx(vn_v, vn_g),
                        result_type=v4bf16_type,
                    )
                    o_accs[nr] = _mfma_bf16_16x16x16(a_frag, vn_b_frag, o_accs[nr])

            # -- Scale and store o -> HBM [T_flat, H, V] token-major --
            def _emit_o_store(off, value):
                o_[fx.Int64(off)] = value

            scale_vec = fx.full(4, fx.Float32(SCALE), fx.Float32)
            for nr in range_constexpr(N_REPEAT_LOCAL):
                o_scaled = o_accs[nr] * scale_vec
                o_col = i_v * fx.Int32(BV) + _nr_v(nr) + lane_n
                for elem_i in range_constexpr(4):
                    o_bt_row = (
                        i_t_i32 * fx.Int32(BT)
                        + wid_m * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                        + fx.Int32(elem_i)
                    )
                    if (o_bt_row < T_local).ir_value():
                        o_off = o_base + o_bt_row * stride_o + o_col
                        _emit_o_store(o_off, _to_bf16_fast(o_scaled[elem_i]))

            # -- next iteration's w + gate/u prefetch --
            next_i_t_i32 = i_t_i32 + fx.Int32(1)
            w_next_prefetch = []
            for i_load in range_constexpr(W_LOADS_PER_THREAD):
                row, col = _w_slot(i_load)
                abs_row = next_i_t_i32 * fx.Int32(BT) + row
                safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                g_off = w_base + safe_row * stride_w + col
                w_next_prefetch.append(w_.vec_load((fx.Int64(g_off),), LOAD_VEC_WIDTH))

            gu_next_prefetch = _load_gate_u(next_i_t_i32)

            results = (
                yield [_to_raw(v) for v in h_accs_in]
                + [_to_raw(v) for v in w_next_prefetch]
                + [_to_raw(v) for v in gu_next_prefetch]
            )

        h_accs_final = list(results[:NUM_H_ACCS])

        # -- Epilogue: store final state --
        if const_expr(STORE_FINAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT_LOCAL):
                    acc_idx = kb * N_REPEAT_LOCAL + nr
                    acc_val = h_accs_final[acc_idx]
                    ht_col = i_v * fx.Int32(BV) + _nr_v(nr) + lane_n
                    ht_row_base = (
                        fx.Int32(kb * 64)
                        + wid_m * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                    )
                    ht_off_base = ht_base + ht_col * fx.Int32(K) + ht_row_base
                    if const_expr(STATE_DTYPE_BF16):
                        out_vec = _to_bf16_fast(acc_val, 4)
                    else:
                        out_vec = acc_val
                    ht_.vec_store((fx.Int64(ht_off_base),), out_vec, 4)

    # -- Host launcher --
    @flyc.jit
    def launch_gdn_h_o(
        q_tensor: fx.Tensor,
        k_tensor: fx.Tensor,
        w_tensor: fx.Tensor,
        u_tensor: fx.Tensor,
        g_tensor: fx.Tensor,
        gk_tensor: fx.Tensor,
        o_tensor: fx.Tensor,
        h0_tensor: fx.Tensor,
        ht_tensor: fx.Tensor,
        cu_seqlens_tensor: fx.Tensor,
        chunk_offsets_tensor: fx.Tensor,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
        grid_v: fx.Int32,
        grid_nh: fx.Int32,
        stream: fx.Stream,
    ):
        launcher = gdn_h_o_kernel(
            q_tensor,
            k_tensor,
            w_tensor,
            u_tensor,
            g_tensor,
            gk_tensor,
            o_tensor,
            h0_tensor,
            ht_tensor,
            cu_seqlens_tensor,
            chunk_offsets_tensor,
            T_val,
            T_flat,
            N_val,
        )
        launcher.launch(
            grid=(grid_v, grid_nh, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_gdn_h_o


__all__ = [
    "compile_chunk_gated_delta_h_o_gfx942",
]
