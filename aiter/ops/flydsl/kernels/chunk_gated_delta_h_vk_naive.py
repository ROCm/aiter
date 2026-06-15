# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Gated Delta Net K5 hidden-state recurrence kernel -- NAIVE / un-pipelined VK fork.

This is a deliberately stripped-down variant of ``chunk_gated_delta_h_vk.py``
with ALL prefetch / software-pipeline scheduling removed, so a trace shows the
raw bottleneck structure before any hand-scheduling is layered back on:

  * NO cross-chunk w prefetch (no init=/yield carry of w).
  * NO OPT-VC g/gk/u emitter queue (g/gk/u are loaded inline at use).
  * NO OPT-K k-prefetch interleave (k is loaded straight to LDS before GEMM1).
  * NO OPT-W next-w interleave into GEMM2.

What is KEPT (these are layout/correctness, not pipeline scheduling):
  * OPT-VWARP layout (wid -> one 16-wide V-tile, full K). VWARP-only file:
    the non-VWARP path is dropped entirely (this fork is BV==64 only).
  * lds_h cooperative-transpose h snapshot store (coalesced HBM write).
  * h_accs recurrence state is still carried across chunks via the dynamic
    ``for ... init=/yield`` -- that is the SSM recurrence, not a prefetch, and
    FlyDSL requires loop-carried values to flow through yield.

For each chunk t (serial over NT chunks):
  1. Store h snapshot for downstream K6 (h_accs -> lds_h -> coalesced HBM).
  2. v_new = u - w @ h   (delta correction via MFMA, h from registers).
  3. Gated decay + state update:
       v_new *= exp(g_last - g_cumsum)
       h = h * exp(g_last) + k^T @ v_new
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .tensor_shim import GTensor, STensor, _to_raw

_LOG2E = math.log2(math.e)  # 1.4426950408889634


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def _make_fast_exp(g_is_log2_scaled: bool):
    if g_is_log2_scaled:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, x)

    else:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, x * _LOG2E)

    return _fast_exp


def _mfma_bf16_16x16x16(a_bf16x4, b_bf16x4, acc_f32x4):
    """Single mfma_f32_16x16x16_bf16 instruction (gfx950 / CDNA bf16 1k form)."""
    a_i16x4 = a_bf16x4.bitcast(fx.Int16)
    b_i16x4 = b_bf16x4.bitcast(fx.Int16)
    return rocdl.mfma_f32_16x16x16bf16_1k(
        T.f32x4, [a_i16x4, b_i16x4, acc_f32x4, 0, 0, 0]
    )


# -- Compile the kernel ---------------------------------------------------


def compile_chunk_gated_delta_h_vk_naive(
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
    """Compile the NAIVE (un-pipelined) GDN K5 VK-fork kernel.

    VWARP-only: requires BV == 64, K % 64 == 0, not USE_GK (the VWARP gate),
    mirroring the conditions under which the optimized VK fork uses VWARP. The
    host wrapper only routes BV==64 to this kernel; other shapes fall back to
    the baseline.
    """
    assert K <= 256
    assert K % 64 == 0
    assert BV % 16 == 0
    NUM_K_BLOCKS = K // 64

    # VWARP gate -- this naive fork is VWARP-only.
    assert BV == 64, "naive VK fork is VWARP-only (BV must be 64)"
    assert not USE_GK, "naive VK fork does not support per-K gates (USE_GK)"

    _fast_exp = _make_fast_exp(G_IS_LOG2_SCALED)

    WARP_SIZE = 64
    NUM_WARPS = 4
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE

    WMMA_N = 16
    WMMA_K = 32  # ds_read_tr16 bt-row span granularity (2 x 16)
    N_REPEAT = BV // WMMA_N  # 4

    NUM_H_ACCS = NUM_K_BLOCKS * N_REPEAT

    K_SUB_PER_BLOCK = 64 // WMMA_N  # 16-wide K sub-tiles per 64 block (=4)
    BT_MTILES = BT // WMMA_N  # 16-row BT M-tiles a warp spans in GEMM1 (=4)

    # -- LDS layout --
    LDS_W_STRIDE = K
    LDS_W_ELEMS = BT * LDS_W_STRIDE
    LDS_W_BYTES = LDS_W_ELEMS * 2

    LDS_K_STRIDE = K
    LDS_K_ELEMS = BT * LDS_K_STRIDE
    LDS_K_BYTES = LDS_K_ELEMS * 2

    LDS_VN_PAD = 4
    LDS_VN_STRIDE = BV + LDS_VN_PAD
    LDS_VN_ELEMS = BT * LDS_VN_STRIDE
    LDS_VN_BYTES = LDS_VN_ELEMS * 2

    LDS_H_PAD = 4
    LDS_H_STRIDE = BV + LDS_H_PAD
    LDS_H_ELEMS = K * LDS_H_STRIDE
    LDS_H_BYTES = LDS_H_ELEMS * 2

    _K5_KERNEL_REVISION = 1  # naive VK fork: no prefetch/pipeline (distinct cache namespace)

    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(
        None,
        arch=GPU_ARCH,
        global_sym_name=f"gdn_h_vk_naive_smem_v{_K5_KERNEL_REVISION}",
    )
    lds_w_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_w_offset + LDS_W_BYTES
    lds_k_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k_offset + LDS_K_BYTES
    lds_vn_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_vn_offset + LDS_VN_BYTES
    lds_h_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_h_offset + LDS_H_BYTES

    # Cooperative load parameters
    LOAD_VEC_WIDTH = 8  # 8 bf16 = 16 bytes = buffer_load_dwordx4
    THREADS_PER_ROW_64 = 64 // LOAD_VEC_WIDTH  # 8
    ROWS_PER_BATCH_64 = BLOCK_THREADS // THREADS_PER_ROW_64  # 32
    NUM_LOAD_BATCHES_64 = BT // ROWS_PER_BATCH_64  # 2

    @flyc.kernel(name="chunk_gdn_fwd_h_flydsl_vk_naive")
    def gdn_h_kernel(
        k_tensor: fx.Pointer,
        v_tensor: fx.Pointer,
        w_tensor: fx.Pointer,
        v_new_tensor: fx.Pointer,
        g_tensor: fx.Pointer,
        gk_tensor: fx.Pointer,
        h_tensor: fx.Pointer,
        h0_tensor: fx.Pointer,
        ht_tensor: fx.Pointer,
        cu_seqlens_tensor: fx.Pointer,
        chunk_offsets_tensor: fx.Pointer,
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
        lds_base_ptr = allocator.get_base()
        lds_w_ptr = SmemPtr(lds_base_ptr, lds_w_offset, T.bf16, shape=(LDS_W_ELEMS,))
        lds_w = STensor(lds_w_ptr, dtype=T.bf16, shape=(LDS_W_ELEMS,))
        lds_k_ptr = SmemPtr(lds_base_ptr, lds_k_offset, T.bf16, shape=(LDS_K_ELEMS,))
        lds_k = STensor(lds_k_ptr, dtype=T.bf16, shape=(LDS_K_ELEMS,))
        lds_vn_ptr = SmemPtr(lds_base_ptr, lds_vn_offset, T.bf16, shape=(LDS_VN_ELEMS,))
        lds_vn = STensor(lds_vn_ptr, dtype=T.bf16, shape=(LDS_VN_ELEMS,))
        lds_h_ptr = SmemPtr(lds_base_ptr, lds_h_offset, T.bf16, shape=(LDS_H_ELEMS,))
        lds_h = STensor(lds_h_ptr, dtype=T.bf16, shape=(LDS_H_ELEMS,))

        # -- Cooperative load decomposition --
        load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_64)
        load_col_base = (tid % fx.Int32(THREADS_PER_ROW_64)) * fx.Int32(LOAD_VEC_WIDTH)

        def _xor_swizzle(row, col):
            return col ^ ((row & fx.Int32(0x7)) << fx.Int32(3))

        def _xor_swizzle_idx(row, col):
            return col ^ ((row & fx.Index(0x7)) << fx.Index(3))

        v8bf16_type = T.vec(8, T.bf16)
        lds_w_memref = lds_w_ptr.get()

        v4bf16_w_type = T.vec(4, T.bf16)

        def _lds_vec_read_w_bf16x4(elem_idx):
            return vector.load_op(v4bf16_w_type, lds_w_memref, [elem_idx])

        v4bf16_type = T.vec(4, T.bf16)

        def _ds_read_tr_bf16x4(lds_byte_offset):
            byte_idx = arith.index_cast(T.index, lds_byte_offset)
            byte_i64 = arith.index_cast(T.i64, byte_idx)
            ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), byte_i64).result
            raw = rocdl.ds_read_tr16_b64(v4bf16_type, ptr).result
            return fx.Vector(raw, (4,), fx.BFloat16)

        tr_k_group = (lane % fx.Int32(16)) // fx.Int32(4)
        tr_col_sub = lane % fx.Int32(4)

        # -- Prologue: bos, T_local, NT, boh --
        if const_expr(IS_VARLEN):
            bos = cu_[fx.Index(i_n)]
            eos = cu_[fx.Index(i_n) + fx.Index(1)]
            T_local = eos - bos
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)
            boh = co_[fx.Index(i_n)]
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

        lane_n = lane % fx.Int32(16)
        lane_m_base = lane // fx.Int32(16)

        wid_idx = fx.Index(wid)
        lane_n_idx = fx.Index(lane_n)
        lane_m_base_idx = fx.Index(lane_m_base)

        # -- Initialize h accumulators --
        acc_zero = fx.full(4, 0.0, fx.Float32)
        h_accs = []
        for _kb in range_constexpr(NUM_K_BLOCKS):
            for _nr in range_constexpr(N_REPEAT):
                h_accs.append(acc_zero)

        # -- Load initial state (VWARP: wid -> V-tile, slot -> K sub-tile) --
        if const_expr(USE_INITIAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for slot in range_constexpr(N_REPEAT):
                    h0_col = i_v * fx.Int32(BV) + wid * fx.Int32(16) + lane_n
                    h0_row_base = (
                        fx.Int32(kb * 64)
                        + fx.Int32(slot * 16)
                        + lane_m_base * fx.Int32(4)
                    )
                    h0_off_base = h0_base + h0_col * fx.Int32(K) + h0_row_base
                    loaded_vec = h0_.vec_load((fx.Index(h0_off_base),), 4)
                    if const_expr(STATE_DTYPE_BF16):
                        loaded_vec = loaded_vec.extf(T.f32x4)
                    acc_idx = kb * N_REPEAT + slot
                    h_accs[acc_idx] = h_accs[acc_idx] + loaded_vec

        NUM_W_LOADS = NUM_K_BLOCKS * NUM_LOAD_BATCHES_64
        NUM_K_LOADS = NUM_K_BLOCKS * NUM_LOAD_BATCHES_64

        c_zero = fx.Index(0)
        c_one = fx.Index(1)
        nt_idx = fx.Index(NT)

        # h_accs is the SSM recurrence state -- carried across chunks via yield.
        init_state = [_to_raw(v) for v in h_accs]

        for i_t, state in range(c_zero, nt_idx, c_one, init=init_state):
            h_accs_in = list(state[:NUM_H_ACCS])
            i_t_i32 = fx.Int32(i_t)

            # ============================================================
            # 1. h snapshot: h_accs -> lds_h ([K][V]) -> coalesced HBM
            # ============================================================
            for kb in range_constexpr(NUM_K_BLOCKS):
                for acc_j in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + acc_j
                    acc_val = h_accs_in[acc_idx]
                    lds_h_col = wid * fx.Int32(16) + lane_n
                    k_tile_base = fx.Int32(kb * 64) + fx.Int32(acc_j * 16)
                    for elem_i in range_constexpr(4):
                        f32_val = acc_val[elem_i]
                        bf16_val = f32_val.to(fx.BFloat16)
                        lds_h_row = (
                            k_tile_base
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        lds_h_idx = lds_h_row * fx.Int32(LDS_H_STRIDE) + lds_h_col
                        lds_h[fx.Index(lds_h_idx)] = bf16_val

            gpu.barrier()

            VK_TOTAL = K * BV
            for vk_base in range_constexpr(0, VK_TOTAL, BLOCK_THREADS):
                linear = fx.Int32(vk_base) + tid
                k_idx = linear % fx.Int32(K)
                v_loc = linear // fx.Int32(K)
                lds_read_idx = k_idx * fx.Int32(LDS_H_STRIDE) + v_loc
                bf16_tile = lds_h[fx.Index(lds_read_idx)]
                v_global = i_v * fx.Int32(BV) + v_loc
                h_off = (
                    h_base + i_t_i32 * stride_h + v_global * fx.Int32(K) + k_idx
                )
                h_[fx.Index(h_off)] = bf16_tile

            gpu.barrier()

            # ============================================================
            # 2. Load w -> lds_w (NO prefetch; load now, use after barrier)
            # ============================================================
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    abs_row = i_t_i32 * fx.Int32(BT) + row
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    w_g_off = (
                        w_base + safe_row * stride_w + fx.Int32(kb * 64) + load_col_base
                    )
                    w_vec = w_.vec_load((fx.Index(w_g_off),), LOAD_VEC_WIDTH)
                    col = fx.Int32(kb * 64) + load_col_base
                    swz_col = _xor_swizzle(row, col)
                    w_lds_off = row * fx.Int32(LDS_W_STRIDE) + swz_col
                    lds_w.vec_store((fx.Index(w_lds_off),), w_vec, LOAD_VEC_WIDTH)

            gpu.barrier()

            # ============================================================
            # 3. Load k -> lds_k (NO prefetch; before GEMM2 use)
            # ============================================================
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    abs_row = i_t_i32 * fx.Int32(BT) + row
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    k_off = (
                        k_base + safe_row * stride_k + fx.Int32(kb * 64) + load_col_base
                    )
                    k_vec = k_.vec_load((fx.Index(k_off),), LOAD_VEC_WIDTH)
                    k_lds_off = (
                        row * fx.Int32(LDS_K_STRIDE) + fx.Int32(kb * 64) + load_col_base
                    )
                    lds_k.vec_store((fx.Index(k_lds_off),), k_vec, LOAD_VEC_WIDTH)

            gpu.barrier()

            # ============================================================
            # 4. GEMM1: b_v = w @ h^T (h from registers). VWARP.
            # ============================================================
            bv_accs = []
            for _i in range_constexpr(N_REPEAT):
                bv_accs.append(fx.full(4, 0.0, fx.Float32))

            for m_bt in range_constexpr(BT_MTILES):
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for slot in range_constexpr(K_SUB_PER_BLOCK):
                        w_lds_row_idx = fx.Index(m_bt * 16) + lane_n_idx
                        w_lds_col_idx = fx.Index(
                            kb * 64 + slot * 16
                        ) + lane_m_base_idx * fx.Index(4)
                        w_lds_col_idx = _xor_swizzle_idx(
                            w_lds_row_idx, w_lds_col_idx
                        )
                        w_lds_idx = (
                            w_lds_row_idx * fx.Index(LDS_W_STRIDE) + w_lds_col_idx
                        )
                        a_frag = _lds_vec_read_w_bf16x4(w_lds_idx)
                        b_frag = h_accs_in[kb * N_REPEAT + slot].to(fx.BFloat16)
                        bv_accs[m_bt] = _mfma_bf16_16x16x16(
                            a_frag, b_frag, bv_accs[m_bt]
                        )

            # ============================================================
            # 5. v_new = u - b_v  (u loaded inline, no prefetch)
            # ============================================================
            u_col = i_v * fx.Int32(BV) + wid * fx.Int32(16) + lane_n
            vn_frags = []
            for m_bt in range_constexpr(BT_MTILES):
                bv_val = bv_accs[m_bt]
                u_f32_elems = []
                for elem_i in range_constexpr(4):
                    u_bt_row_raw = (
                        i_t_i32 * fx.Int32(BT)
                        + fx.Int32(m_bt * 16)
                        + lane_m_base * fx.Int32(4)
                        + fx.Int32(elem_i)
                    )
                    safe_u_row = (u_bt_row_raw < T_local).select(
                        u_bt_row_raw, fx.Int32(0)
                    )
                    u_off = v_base + safe_u_row * stride_v + u_col
                    u_raw = v_[fx.Index(u_off)]
                    u_bf16 = fx.BFloat16(u_raw)
                    u_f32_elems.append(u_bf16.to(fx.Float32))
                u_f32 = vector.from_elements(T.f32x4, u_f32_elems)
                vn_frags.append(u_f32 - bv_val)

            # ============================================================
            # 5b. Store v_new (pre-gating) for output
            # ============================================================
            if const_expr(SAVE_NEW_VALUE):
                def _emit_vn_store(off, value):
                    vn_[fx.Index(off)] = value

                for m_bt in range_constexpr(BT_MTILES):
                    vn_val = vn_frags[m_bt]
                    vn_col = i_v * fx.Int32(BV) + wid * fx.Int32(16) + lane_n
                    bt_tile_base = fx.Int32(m_bt * 16)
                    for elem_i in range_constexpr(4):
                        vn_bt_row = (
                            i_t_i32 * fx.Int32(BT)
                            + bt_tile_base
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        if (vn_bt_row < T_local).ir_value():
                            f32_v = vn_val[elem_i]
                            bf16_v = f32_v.to(fx.BFloat16)
                            vn_off = vn_base + vn_bt_row * fx.Int32(V) + vn_col
                            _emit_vn_store(vn_off, bf16_v)

            # ============================================================
            # 6. Gating (g loaded inline)
            # ============================================================
            next_chunk_end = (i_t_i32 + fx.Int32(1)) * fx.Int32(BT)
            last_idx_raw = (next_chunk_end < T_local).select(
                next_chunk_end, T_local
            ) - fx.Int32(1)

            if const_expr(USE_G):
                g_last_off = i_h * T_flat + (bos + last_idx_raw)
                g_last = g_[fx.Index(g_last_off)]
                exp_g_last = _fast_exp(g_last)

                for m_bt in range_constexpr(BT_MTILES):
                    gate_elems = []
                    for elem_i in range_constexpr(4):
                        abs_row = (
                            i_t_i32 * fx.Int32(BT)
                            + fx.Int32(m_bt * 16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        in_bounds = abs_row < T_local
                        safe_row = in_bounds.select(abs_row, fx.Int32(0))
                        g_row_off = i_h * T_flat + (bos + safe_row)
                        g_row = g_[fx.Index(g_row_off)]
                        gate = _fast_exp(g_last - g_row)
                        gate_elems.append(in_bounds.select(gate, fx.Float32(0.0)))
                    gate_vec = vector.from_elements(T.f32x4, gate_elems)
                    vn_frags[m_bt] = vn_frags[m_bt] * gate_vec

                exp_g_last_s = fx.Float32(exp_g_last)
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * exp_g_last_s

            # ============================================================
            # 7. Store gated v_new -> lds_vn
            # ============================================================
            for m_bt in range_constexpr(N_REPEAT):
                vn_val = vn_frags[m_bt]
                lds_col = wid * fx.Int32(16) + lane_n
                lds_bt_tile = fx.Int32(m_bt * 16)
                for elem_i in range_constexpr(4):
                    f32_v = vn_val[elem_i]
                    bf16_v = f32_v.to(fx.BFloat16)
                    lds_row = (
                        lds_bt_tile + lane_m_base * fx.Int32(4) + fx.Int32(elem_i)
                    )
                    lds_idx = lds_row * fx.Int32(LDS_VN_STRIDE) + lds_col
                    lds_vn[fx.Index(lds_idx)] = bf16_v

            gpu.barrier()

            # ============================================================
            # 8. GEMM2: h += k^T @ v_new_gated. VWARP.
            # ============================================================
            BT_STEPS = BT // WMMA_K
            for kb in range_constexpr(NUM_K_BLOCKS):
                for bt_s in range_constexpr(BT_STEPS):
                    bt_row_tr = (
                        fx.Int32(bt_s * WMMA_K) + lane_m_base * fx.Int32(8) + tr_k_group
                    )
                    vn_v_col = wid * fx.Int32(16) + tr_col_sub * fx.Int32(4)
                    vn_lds_elem = bt_row_tr * fx.Int32(LDS_VN_STRIDE) + vn_v_col
                    vn_lds_byte = vn_lds_elem * fx.Int32(2) + fx.Int32(lds_vn_offset)
                    vn_b_lo = _ds_read_tr_bf16x4(vn_lds_byte)
                    vn_b_hi = _ds_read_tr_bf16x4(
                        vn_lds_byte + fx.Int32(4 * LDS_VN_STRIDE * 2)
                    )

                    for slot in range_constexpr(K_SUB_PER_BLOCK):
                        k_col = fx.Int32(slot * 16) + tr_col_sub * fx.Int32(4)
                        k_lds_elem = (
                            bt_row_tr * fx.Int32(LDS_K_STRIDE)
                            + fx.Int32(kb * 64)
                            + k_col
                        )
                        k_lds_byte = k_lds_elem * fx.Int32(2) + fx.Int32(lds_k_offset)
                        k_a_lo = _ds_read_tr_bf16x4(k_lds_byte)
                        k_a_hi = _ds_read_tr_bf16x4(
                            k_lds_byte + fx.Int32(4 * LDS_K_STRIDE * 2)
                        )
                        acc_idx = kb * N_REPEAT + slot
                        h_accs_in[acc_idx] = _mfma_bf16_16x16x16(
                            k_a_lo, vn_b_lo, h_accs_in[acc_idx]
                        )
                        h_accs_in[acc_idx] = _mfma_bf16_16x16x16(
                            k_a_hi, vn_b_hi, h_accs_in[acc_idx]
                        )

            gpu.barrier()

            results = yield [_to_raw(v) for v in h_accs_in]

        h_accs_final = list(results[:NUM_H_ACCS])

        # -- Epilogue: store final state --
        if const_expr(STORE_FINAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for slot in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + slot
                    acc_val = h_accs_final[acc_idx]
                    ht_col = i_v * fx.Int32(BV) + wid * fx.Int32(16) + lane_n
                    ht_row_base = (
                        fx.Int32(kb * 64)
                        + fx.Int32(slot * 16)
                        + lane_m_base * fx.Int32(4)
                    )
                    ht_off_base = ht_base + ht_col * fx.Int32(K) + ht_row_base
                    if const_expr(STATE_DTYPE_BF16):
                        out_vec = acc_val.truncf(T.vec(4, T.bf16))
                    else:
                        out_vec = acc_val
                    ht_.vec_store((fx.Index(ht_off_base),), out_vec, 4)

    # -- Host launcher ------------------------------------------------------
    @flyc.jit
    def launch_gdn_h(
        k_tensor: fx.Pointer,
        v_tensor: fx.Pointer,
        w_tensor: fx.Pointer,
        v_new_tensor: fx.Pointer,
        g_tensor: fx.Pointer,
        gk_tensor: fx.Pointer,
        h_tensor: fx.Pointer,
        h0_tensor: fx.Pointer,
        ht_tensor: fx.Pointer,
        cu_seqlens_tensor: fx.Pointer,
        chunk_offsets_tensor: fx.Pointer,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
        grid_v: fx.Int32,
        grid_nh: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

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
    "compile_chunk_gated_delta_h_vk_naive",
]
