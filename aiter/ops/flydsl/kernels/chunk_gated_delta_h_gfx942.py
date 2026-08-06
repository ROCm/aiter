# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GDN K5 inter-chunk state scan — gfx942 (CDNA3 / MI300X) FlyDSL kernel.

For each chunk t (serial over NT chunks):
  1. Store h snapshot for downstream K6
  2. v_new = u - w @ h   (delta correction via MFMA)
  3. Gated decay + state update:
       v_new *= exp(g_last - g_cumsum)
       h = h * exp(g_last) + k^T @ v_new

"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import as_ir_value, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl._mlir.dialects import vector as _vector

from .tensor_shim import GTensor, _to_raw

_LOG2E = math.log2(math.e)  # 1.4426950408889634


def _make_fast_exp(g_is_log2_scaled: bool):
    """Return the ``exp`` helper (see gfx950 kernel for the rationale)."""
    if g_is_log2_scaled:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, x)

    else:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, x * _LOG2E)

    return _fast_exp

def _mfma_bf16_16x16x16(a_bf16x4, b_bf16x4, acc_f32x4):
    """Single ``mfma_f32_16x16x16bf16_1k`` (gfx942 bf16 K=16 MFMA).

    The MFMA fragment ABI:
    * A operand (16x16x16): lane holds bf16x4 with element e = A[m=lane_n, k=grp*4+e],
        where grp = lane_m_base (lane//16).
    * B operand: lane holds bf16x4 with element e = B[k=grp*4+e, n=lane_n].
    * C/D accumulator: lane holds f32x4 with element e = C[m=grp*4+e, n=lane_n].
    
    Operands are bitcast bf16x4 -> vec<4xi16> (the intrinsic's operand type).
    """
    a_i16 = _vector.bitcast(T.vec(4, T.i16), as_ir_value(a_bf16x4))
    b_i16 = _vector.bitcast(T.vec(4, T.i16), as_ir_value(b_bf16x4))
    return rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a_i16, b_i16, acc_f32x4, 0, 0, 0])

def compile_chunk_gated_delta_h_gfx942(
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
):
    """Build the gfx942 GDN K5 launcher for one compile-time configuration.

    Signature matches ``compile_chunk_gated_delta_h`` so ``_get_or_compile`` in
    ``linear_attention_prefill_kernels`` can call either implementation without modification.
    """
    assert K <= 256
    assert K % 64 == 0
    assert BV % 16 == 0
    # gfx942 LDS budget: BV=64 needs 73.5 KiB > 64 KiB/CU. Cap at 32.
    assert BV <= 32, (
        f"gfx942 LDS budget caps BV at 32 (got BV={BV}); "
        "BV=64 overflows the 64 KiB/CU LDS limit at K=128, BT=64."
    )
    NUM_K_BLOCKS = K // 64

    _fast_exp = _make_fast_exp(G_IS_LOG2_SCALED)

    WARP_SIZE = 64
    NUM_WARPS = 4
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE

    WMMA_N = 16
    WMMA_K = 16  # gfx942: K=16
    N_REPEAT = BV // WMMA_N

    NUM_H_ACCS = NUM_K_BLOCKS * N_REPEAT

    # -- LDS layout --
    # lds_w: w tile [BT, K], row-major (A-frag for GEMM1, plain read). Single stage.
    LDS_W_STRIDE = K
    LDS_W_ELEMS = BT * LDS_W_STRIDE

    # lds_k: k tile stored TRANSPOSED as [K, BT] so GEMM2's k A-frag (a run over BT
    # for fixed K) is a contiguous read. Stride = BT + pad.
    LDS_KT_PAD = 4
    LDS_KT_STRIDE = BT + LDS_KT_PAD
    LDS_KT_ELEMS = K * LDS_KT_STRIDE

    # lds_vn: v_new stored TRANSPOSED as [BT, V] -> we need GEMM2 B-frag = run over
    # BT (contraction) at fixed V. Store as [V, BT] so the BT run is contiguous.
    LDS_VNT_PAD = 4
    LDS_VNT_STRIDE = BT + LDS_VNT_PAD
    LDS_VNT_ELEMS = V * LDS_VNT_STRIDE  # V rows (full V, tiled by BV per CTA in cols)

    # lds_h: h snapshot [V, K] used as GEMM1 B-frag = run over K (contraction) at
    # fixed V. Store as [V, K] so the K run is contiguous (already the natural VK
    # order). Stride = K + pad. This tile also feeds the HBM snapshot store.
    LDS_H_PAD = 4
    LDS_H_STRIDE = K + LDS_H_PAD
    LDS_H_ELEMS = BV * LDS_H_STRIDE  # BV rows of V per CTA

    @fx.struct
    class SharedStorage:
        lds_w: fx.Array[fx.BFloat16, LDS_W_ELEMS, 16]
        lds_kt: fx.Array[fx.BFloat16, LDS_KT_ELEMS, 16]
        lds_vnt: fx.Array[fx.BFloat16, LDS_VNT_ELEMS, 16]
        lds_h: fx.Array[fx.BFloat16, LDS_H_ELEMS, 16]

    # Cooperative load parameters (bf16x8 = dwordx4)
    LOAD_VEC_WIDTH = 8
    THREADS_PER_ROW_64 = 64 // LOAD_VEC_WIDTH  # 8
    ROWS_PER_BATCH_64 = BLOCK_THREADS // THREADS_PER_ROW_64  # 32
    NUM_LOAD_BATCHES_64 = BT // ROWS_PER_BATCH_64  # 2

    K_STEPS_PER_BLOCK = 64 // WMMA_K  # 4
    BT_STEPS = BT // WMMA_K  # 4

    @flyc.kernel(name="chunk_gdn_fwd_h_flydsl_vk")
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
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
    ):
        i_v = fx.block_idx.x
        i_nh = fx.block_idx.y
        i_n = i_nh // fx.Int32(H)
        i_h = i_nh % fx.Int32(H)

        tid = fx.thread_idx.x
        wid = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)

        k_ = GTensor(k_tensor, dtype=T.bf16, shape=(-1,))
        v_ = GTensor(v_tensor, dtype=T.bf16, shape=(-1,))
        w_ = GTensor(w_tensor, dtype=T.bf16, shape=(-1,))
        h_ = GTensor(h_tensor, dtype=T.bf16, shape=(-1,))
        g_ = GTensor(g_tensor, dtype=T.f32, shape=(-1,))
        if const_expr(USE_GK):
            gk_ = GTensor(gk_tensor, dtype=T.f32, shape=(-1,))

        vn_ = GTensor(v_new_tensor, dtype=T.bf16, shape=(-1,))
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

        # -- Cooperative load decomposition --
        load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_64)
        load_col_base = (tid % fx.Int32(THREADS_PER_ROW_64)) * fx.Int32(LOAD_VEC_WIDTH)

        # -- Prologue: compute bos, T_local, NT, boh --
        if const_expr(IS_VARLEN):
            bos = cu_[fx.Int64(i_n)]
            eos = cu_[fx.Int64(i_n) + fx.Int64(1)]
            T_local = eos - bos
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)
            boh = co_[fx.Int64(i_n)]
        else:
            bos = i_n * T_val
            T_local = T_val
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)
            boh = i_n * NT

        # -- Base pointer offsets (element counts) --
        h_base = (boh * fx.Int32(H) + i_h) * fx.Int32(V * K)
        stride_h = fx.Int32(H * V * K)

        gqa_ratio = H // Hg
        k_base = (bos * fx.Int32(Hg) + i_h // fx.Int32(gqa_ratio)) * fx.Int32(K)
        stride_k = fx.Int32(Hg * K)

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

        if const_expr(IS_VARLEN):
            vn_base = (i_h * T_flat + bos) * fx.Int32(V)
        else:
            vn_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(V)

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
            for _nr in range_constexpr(N_REPEAT):
                h_accs.append(acc_zero)

        # -- Load initial state if provided --
        # h_accs[kb][nr] element e = h[v = i_v*BV + nr*16 + lane_n,
        #                              k = kb*64 + wid*16 + lane_m_base*4 + e]
        if const_expr(USE_INITIAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT):
                    h0_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_n
                    h0_row_base = (
                        fx.Int32(kb * 64)
                        + wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                    )
                    h0_off_base = h0_base + h0_col * fx.Int32(K) + h0_row_base
                    loaded_vec = h0_.vec_load((fx.Int64(h0_off_base),), 4)
                    if const_expr(STATE_DTYPE_BF16):
                        loaded_vec = loaded_vec.extf(T.f32x4)
                    acc_idx = kb * N_REPEAT + nr
                    h_accs[acc_idx] = h_accs[acc_idx] + loaded_vec

        NUM_W_LOADS = NUM_K_BLOCKS * NUM_LOAD_BATCHES_64

        # -- Prologue: pre-load first chunk's w data --
        i_t0_i32 = fx.Int32(0)
        w_prefetch_init = []
        for kb in range_constexpr(NUM_K_BLOCKS):
            for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                abs_row = i_t0_i32 * fx.Int32(BT) + row
                safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                g_off = w_base + safe_row * stride_w + fx.Int32(kb * 64) + load_col_base
                w_prefetch_init.append(w_.vec_load((fx.Int64(g_off),), LOAD_VEC_WIDTH))

        init_state = [_to_raw(v) for v in h_accs] + [
            _to_raw(v) for v in w_prefetch_init
        ]
        c_zero = fx.Int64(0)
        c_one = fx.Int64(1)
        nt_idx = fx.Int64(NT)

        for i_t, state in range(c_zero, nt_idx, c_one, init=init_state):
            h_accs_in = list(state[:NUM_H_ACCS])
            w_prefetch_all = list(state[NUM_H_ACCS:])
            i_t_i32 = fx.Int32(i_t)

            # -- w LDS write offsets (row-major [BT, K], plain) --
            w_prefetch_lds_all = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    col = fx.Int32(kb * 64) + load_col_base
                    w_prefetch_lds_all.append(row * fx.Int32(LDS_W_STRIDE) + col)

            # -- Store h snapshot to LDS as [V, K] (VK, K contiguous) --
            # h_accs element e = h[v_local = nr*16 + lane_n, k = kb*64 + wid*16 +
            #                      lane_m_base*4 + e].  Store at lds_h[v_local, k].
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + nr
                    acc_val = h_accs_in[acc_idx]
                    lds_h_v = fx.Int32(nr * 16) + lane_n
                    for elem_i in range_constexpr(4):
                        f32_val = acc_val[elem_i]
                        bf16_val = f32_val.to(fx.BFloat16)
                        lds_h_k = (
                            fx.Int32(kb * 64)
                            + wid * fx.Int32(16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        lds_h_idx = lds_h_v * fx.Int32(LDS_H_STRIDE) + lds_h_k
                        fx.ptr_store(bf16_val, lds_h_ptr + fx.Int32(lds_h_idx))

            gpu.barrier()

            # -- LDS -> HBM h snapshot. lds_h is [BV, K] (v_local, k). --
            VK_TOTAL = K * BV
            for vk_base in range_constexpr(0, VK_TOTAL, BLOCK_THREADS):
                linear = fx.Int32(vk_base) + tid
                k_idx = linear % fx.Int32(K)
                v_loc = linear // fx.Int32(K)
                lds_read_idx = v_loc * fx.Int32(LDS_H_STRIDE) + k_idx
                bf16_tile = fx.ptr_load(lds_h_ptr + fx.Int32(lds_read_idx))
                v_global = i_v * fx.Int32(BV) + v_loc
                h_off = h_base + i_t_i32 * stride_h + v_global * fx.Int32(K) + k_idx
                h_[fx.Int64(h_off)] = bf16_tile

            # -- Store prefetched w to LDS (row-major) --
            for i_wp in range_constexpr(NUM_W_LOADS):
                fx.ptr_store(
                    w_prefetch_all[i_wp],
                    lds_w_ptr + fx.Int32(w_prefetch_lds_all[i_wp]),
                )

            gpu.barrier()

            # -- k prefetch (issued now, stored transposed after GEMM1) --
            k_prefetch = []
            k_prefetch_lds_t = []  # transposed store offsets: lds_kt[k, bt]
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    abs_row = i_t_i32 * fx.Int32(BT) + row
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    k_off = (
                        k_base + safe_row * stride_k + fx.Int32(kb * 64) + load_col_base
                    )
                    k_prefetch.append(k_.vec_load((fx.Int64(k_off),), LOAD_VEC_WIDTH))
                    # this vec holds k[row, kb*64 + load_col_base + (0..7)];
                    # store each element transposed to lds_kt[kcol, row].
                    k_prefetch_lds_t.append((row, fx.Int32(kb * 64) + load_col_base))

            # last_idx for gating
            next_chunk_end = (i_t_i32 + fx.Int32(1)) * fx.Int32(BT)
            last_idx_raw = (next_chunk_end < T_local).select(
                next_chunk_end, T_local
            ) - fx.Int32(1)

            # -- g / gk / u prefetch (simple batched, no OPT-VC interleave) --
            if const_expr(USE_G):
                g_last_off = i_h * T_flat + (bos + last_idx_raw)
                g_last_val = g_[fx.Int64(g_last_off)]
                g_row_vals = []
                for elem_i in range_constexpr(4):
                    abs_row = (
                        i_t_i32 * fx.Int32(BT)
                        + wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                        + fx.Int32(elem_i)
                    )
                    in_bounds = abs_row < T_local
                    safe_row = in_bounds.select(abs_row, fx.Int32(0))
                    g_row_off = i_h * T_flat + (bos + safe_row)
                    g_row_vals.append((g_[fx.Int64(g_row_off)], in_bounds))

            if const_expr(USE_GK):
                gk_chunk_base = (bos + last_idx_raw) * fx.Int32(H * K) + i_h * fx.Int32(K)
                gk_last_prefetch = []
                for kb in range_constexpr(NUM_K_BLOCKS):
                    kb_elems = []
                    for elem_i in range_constexpr(4):
                        global_k = (
                            fx.Int32(kb * 64)
                            + wid * fx.Int32(16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        gk_raw = gk_[fx.Int64(gk_chunk_base + global_k)]
                        kb_elems.append(_fast_exp(gk_raw))
                    gk_last_prefetch.append(kb_elems)

            u_prefetch = []
            for nr in range_constexpr(N_REPEAT):
                u_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_n
                for elem_i in range_constexpr(4):
                    u_bt_row_raw = (
                        i_t_i32 * fx.Int32(BT)
                        + wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                        + fx.Int32(elem_i)
                    )
                    safe_u_row = (u_bt_row_raw < T_local).select(
                        u_bt_row_raw, fx.Int32(0)
                    )
                    u_off = v_base + safe_u_row * stride_v + u_col
                    u_prefetch.append(v_.vec_load((fx.Int64(u_off),), 1))

            # -- GEMM1: bv = w @ h  (contraction over K) --
            # A-frag (w): lane holds w[m=BT row, k]; plain read of lds_w.
            # B-frag (h): lane holds h[k, n=V]; read from lds_h[v, k] with the
            #   transposed access = 4 contiguous k for fixed v (since lds_h is
            #   [v, k] with k contiguous, a run over k IS contiguous).
            bv_accs = []
            for _nr in range_constexpr(N_REPEAT):
                bv_accs.append(fx.full(4, 0.0, fx.Float32))

            for kb in range_constexpr(NUM_K_BLOCKS):
                for ks in range_constexpr(K_STEPS_PER_BLOCK):
                    # w A-frag: 4 bf16 K-elems for this lane's BT row.
                    # A[m=BT row=wid*16+lane_n, k=kb*64+ks*16 + lane_m_base*4 + e]
                    w_row = wid * fx.Int32(16) + lane_n
                    w_col = fx.Int32(kb * 64 + ks * WMMA_K) + lane_m_base * fx.Int32(4)
                    a_elems = []
                    for e in range_constexpr(4):
                        a_elems.append(
                            fx.ptr_load(
                                lds_w_ptr
                                + (w_row * fx.Int32(LDS_W_STRIDE) + w_col + fx.Int32(e))
                            )
                        )
                    a_frag = fx.Vector.from_elements(a_elems, dtype=fx.BFloat16)

                    for nr in range_constexpr(N_REPEAT):
                        # h B-frag: B[k=kb*64+ks*16 + lane_m_base*4 + e, n=V=nr*16+lane_n]
                        # lds_h[v, k]: v = nr*16 + lane_n, k run.
                        h_v = fx.Int32(nr * 16) + lane_n
                        h_k = fx.Int32(kb * 64 + ks * WMMA_K) + lane_m_base * fx.Int32(4)
                        b_elems = []
                        for e in range_constexpr(4):
                            b_elems.append(
                                fx.ptr_load(
                                    lds_h_ptr
                                    + (
                                        h_v * fx.Int32(LDS_H_STRIDE)
                                        + h_k
                                        + fx.Int32(e)
                                    )
                                )
                            )
                        b_frag = fx.Vector.from_elements(b_elems, dtype=fx.BFloat16)
                        bv_accs[nr] = _mfma_bf16_16x16x16(a_frag, b_frag, bv_accs[nr])

            # -- v_new = u - bv --
            vn_frags = []
            for nr in range_constexpr(N_REPEAT):
                bv_val = bv_accs[nr]
                u_f32_elems = []
                for elem_i in range_constexpr(4):
                    u_bf16 = fx.BFloat16(u_prefetch[nr * 4 + elem_i])
                    u_f32_elems.append(u_bf16.to(fx.Float32))
                u_f32 = fx.Vector.from_elements(u_f32_elems, dtype=fx.Float32)
                vn_frags.append(u_f32 - bv_val)

            # -- Tail-chunk row mask --
            # On the final chunk, BT rows beyond T_local are padding whose w/u/k
            # loads were clamped to row 0 (garbage). They must be zeroed in v_new
            # before the k^T @ v_new state update, or ``final_state`` is corrupted.
            # The USE_G gate below already zeroes out-of-range rows, but the
            # USE_GK path does no v_new gating -- so mask here unconditionally so
            # both gate ranks are correct. Each lane's f32x4 spans 4 BT rows (one
            # per elem_i); the row is the same across all nr.
            row_mask_elems = []
            for elem_i in range_constexpr(4):
                bt_row = (
                    i_t_i32 * fx.Int32(BT)
                    + wid * fx.Int32(16)
                    + lane_m_base * fx.Int32(4)
                    + fx.Int32(elem_i)
                )
                in_bounds = bt_row < T_local
                row_mask_elems.append(
                    in_bounds.select(fx.Float32(1.0), fx.Float32(0.0))
                )
            row_mask_vec = fx.Vector.from_elements(row_mask_elems, dtype=fx.Float32)
            for nr in range_constexpr(N_REPEAT):
                vn_frags[nr] = vn_frags[nr] * row_mask_vec

            # -- 2b. Store v_new (pre-gating) for output --
            if const_expr(SAVE_NEW_VALUE):

                def _emit_vn_store(off, value):
                    vn_[fx.Int64(off)] = value

                for nr in range_constexpr(N_REPEAT):
                    vn_val = vn_frags[nr]
                    vn_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_n
                    for elem_i in range_constexpr(4):
                        vn_bt_row = (
                            i_t_i32 * fx.Int32(BT)
                            + wid * fx.Int32(16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        if (vn_bt_row < T_local).ir_value():
                            f32_v = vn_val[elem_i]
                            bf16_v = f32_v.to(fx.BFloat16)
                            vn_off = vn_base + vn_bt_row * fx.Int32(V) + vn_col
                            _emit_vn_store(vn_off, bf16_v)

            # -- 3. Gating --
            if const_expr(USE_G):
                exp_g_last = _fast_exp(g_last_val)
                gate_elems = []
                for elem_i in range_constexpr(4):
                    g_row, in_bounds = g_row_vals[elem_i]
                    gate = _fast_exp(g_last_val - g_row)
                    gate_elems.append(in_bounds.select(gate, fx.Float32(0.0)))
                gate_vec = fx.Vector.from_elements(gate_elems, dtype=fx.Float32)
                for nr in range_constexpr(N_REPEAT):
                    vn_frags[nr] = vn_frags[nr] * gate_vec
                exp_g_last_vec = fx.full(4, fx.Float32(exp_g_last), fx.Float32)
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * exp_g_last_vec

            if const_expr(USE_GK):
                for kb in range_constexpr(NUM_K_BLOCKS):
                    gk_vec = fx.Vector.from_elements(
                        [gk_last_prefetch[kb][elem_i] for elem_i in range_constexpr(4)],
                        dtype=fx.Float32,
                    )
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * gk_vec

            # -- 4. State update: h += k^T @ v_new_gated --
            # Store gated v_new transposed as [V, BT] so GEMM2 B-frag (run over
            # BT for fixed V) is contiguous. v_new element e is at
            # BT row = wid*16 + lane_m_base*4 + e, V col = nr*16 + lane_n.
            for nr in range_constexpr(N_REPEAT):
                vn_val = vn_frags[nr]
                vnt_v = fx.Int32(nr * 16) + lane_n
                for elem_i in range_constexpr(4):
                    f32_v = vn_val[elem_i]
                    bf16_v = f32_v.to(fx.BFloat16)
                    vnt_bt = (
                        wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                        + fx.Int32(elem_i)
                    )
                    vnt_idx = vnt_v * fx.Int32(LDS_VNT_STRIDE) + vnt_bt
                    fx.ptr_store(bf16_v, lds_vnt_ptr + fx.Int32(vnt_idx))

            # Store k transposed as [K, BT]. k_prefetch[i] holds k[row, kcol+(0..7)];
            # store each of the 8 elements to lds_kt[kcol+e, row].
            for i_kp in range_constexpr(NUM_W_LOADS):
                kvec = k_prefetch[i_kp]
                row, kcol = k_prefetch_lds_t[i_kp]
                for e in range_constexpr(LOAD_VEC_WIDTH):
                    kt_idx = (kcol + fx.Int32(e)) * fx.Int32(LDS_KT_STRIDE) + row
                    fx.ptr_store(kvec[e], lds_kt_ptr + fx.Int32(kt_idx))

            gpu.barrier()

            # -- next iteration's w prefetch (batched) --
            next_i_t_i32 = i_t_i32 + fx.Int32(1)
            w_next_prefetch = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    abs_row = next_i_t_i32 * fx.Int32(BT) + row
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    g_off = (
                        w_base + safe_row * stride_w + fx.Int32(kb * 64) + load_col_base
                    )
                    w_next_prefetch.append(
                        w_.vec_load((fx.Int64(g_off),), LOAD_VEC_WIDTH)
                    )

            # -- GEMM2: h += k^T @ v_new  (contraction over BT) --
            # A-frag (k): lane holds k[m=V head dim? no] -> k is [BT, K]; we want
            #   k^T = [K, BT] as A so output is [K, V]. A[m=K, contraction=BT].
            #   lds_kt[k, bt]: read 4 contiguous BT for fixed k.
            #   A[m=K=wid*16+lane_n? ] -- see ABI: A[m=lane_n(+row grp), k=grp*4+e].
            #   Here MFMA "m" = K output row, "contraction" = BT.
            for kb in range_constexpr(NUM_K_BLOCKS):
                for bt_s in range_constexpr(BT_STEPS):
                    # A-frag k: m = K row = kb*64 + wid*16 + lane_n,
                    #           contraction bt = bt_s*16 + lane_m_base*4 + e
                    k_m = fx.Int32(kb * 64) + wid * fx.Int32(16) + lane_n
                    k_bt = fx.Int32(bt_s * WMMA_K) + lane_m_base * fx.Int32(4)
                    ka_elems = []
                    for e in range_constexpr(4):
                        ka_elems.append(
                            fx.ptr_load(
                                lds_kt_ptr
                                + (
                                    k_m * fx.Int32(LDS_KT_STRIDE)
                                    + k_bt
                                    + fx.Int32(e)
                                )
                            )
                        )
                    k_a_frag = fx.Vector.from_elements(ka_elems, dtype=fx.BFloat16)

                    for nr in range_constexpr(N_REPEAT):
                        # B-frag v_new: n = V = nr*16 + lane_n,
                        #               contraction bt = bt_s*16 + lane_m_base*4 + e
                        vn_v = fx.Int32(nr * 16) + lane_n
                        vn_bt = fx.Int32(bt_s * WMMA_K) + lane_m_base * fx.Int32(4)
                        vb_elems = []
                        for e in range_constexpr(4):
                            vb_elems.append(
                                fx.ptr_load(
                                    lds_vnt_ptr
                                    + (
                                        vn_v * fx.Int32(LDS_VNT_STRIDE)
                                        + vn_bt
                                        + fx.Int32(e)
                                    )
                                )
                            )
                        vn_b_frag = fx.Vector.from_elements(vb_elems, dtype=fx.BFloat16)
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = _mfma_bf16_16x16x16(
                            k_a_frag, vn_b_frag, h_accs_in[acc_idx]
                        )

            results = yield [_to_raw(v) for v in h_accs_in] + [
                _to_raw(v) for v in w_next_prefetch
            ]

        h_accs_final = list(results[:NUM_H_ACCS])

        # -- Epilogue: store final state --
        if const_expr(STORE_FINAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + nr
                    acc_val = h_accs_final[acc_idx]
                    ht_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_n
                    ht_row_base = (
                        fx.Int32(kb * 64)
                        + wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                    )
                    ht_off_base = ht_base + ht_col * fx.Int32(K) + ht_row_base
                    if const_expr(STATE_DTYPE_BF16):
                        out_vec = acc_val.truncf(T.vec(4, T.bf16))
                    else:
                        out_vec = acc_val
                    ht_.vec_store((fx.Int64(ht_off_base),), out_vec, 4)

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


__all__ = [
    "compile_chunk_gated_delta_h_gfx942",
]
