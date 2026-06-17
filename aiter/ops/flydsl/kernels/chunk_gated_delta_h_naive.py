# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Gated Delta Net K5 hidden-state recurrence kernel -- NAIVE / un-pipelined
BASELINE fork.

This is a deliberately stripped-down twin of ``chunk_gated_delta_h.py``
(the heavily software-pipelined baseline) with ALL prefetch / software-
pipeline scheduling removed, so a trace shows the raw bottleneck structure
before any hand-scheduling is layered back on. It produces IDENTICAL
numerical output to the baseline, on the SAME public VK h-layout
``(B, NT, H, V, K)`` and at ANY BV (this is the baseline layout, not the
BV==64-only VWARP fork).

REMOVED (pipeline scheduling only):
  * Cross-chunk w prefetch: no init=/yield carry of w. w for the current
    chunk is loaded inline (vec_load -> lds_w + barrier) at the top of the
    loop body.
  * OPT-VC g/gk/u emitter queue (extra_load_emitters, SLOT_ASSIGN_CT,
    PROLOGUE_EMITTER_CT, TAIL_EMITTER_CT, _make_emit_*, EXTRAS_PER_SLOT_CT,
    OPT_VC_ENABLED): g_last / g_row / gk / u are loaded DIRECTLY inline at
    the point of use.
  * OPT-K k-prefetch interleave: k is loaded straight to LDS before GEMM1
    (cooperative load + barrier), like a naive impl.
  * OPT-W next-w interleave into GEMM2 (OPT_W_ENABLED): gone with the w
    carry.

KEPT (layout / correctness, NOT pipeline scheduling):
  * The whole MFMA tiling + lane mapping (lane_n, lane_m_base, tr_k_group,
    tr_col_sub, wid/nr layout) -- identical to baseline, mfma_16x16x32_bf16.
  * lds_h cooperative-transpose h snapshot store + LDS->HBM transpose loop
    (per-element store version from the baseline).
  * LDS layout & bank-conflict handling: lds_w XOR swizzle, lds_k/lds_vn/
    lds_h stride padding (LDS_K_PAD / LDS_VN_PAD / LDS_H_PAD via _pad_env).
    LDS_W_STAGES = 1.
  * h_accs recurrence carried via init=/yield (the SSM recurrence -- this
    is NOT a prefetch; FlyDSL requires loop-carried values to flow through
    yield).
  * Gating (USE_G / USE_GK), v_new = u - w@h, h += k^T @ v_new math.
  * STORE_FINAL_STATE / SAVE_NEW_VALUE epilogue.

For each chunk t (serial over NT chunks):
  1. Store h snapshot for downstream K6
  2. v_new = u - w @ h   (delta correction via MFMA)
  3. Gated decay + state update:
       v_new *= exp(g_last - g_cumsum)
       h = h * exp(g_last) + k^T @ v_new
"""

import math
import os

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
_LLVM_GEP_DYNAMIC = -2147483648


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def _make_fast_exp(g_is_log2_scaled: bool):
    """Return the ``exp`` helper for this kernel compile (see baseline)."""
    if g_is_log2_scaled:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, x)

    else:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, x * _LOG2E)

    return _fast_exp


def _mfma_bf16_16x16x32(a_bf16x8, b_bf16x8, acc_f32x4):
    """Single mfma_f32_16x16x32_bf16 instruction."""
    return rocdl.mfma_f32_16x16x32_bf16(
        T.f32x4, [a_bf16x8, b_bf16x8, acc_f32x4, 0, 0, 0]
    )


# -- Compile the kernel ---------------------------------------------------


def compile_chunk_gated_delta_h_naive(
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
    """Compile the NAIVE (un-pipelined) GDN K5 baseline-fork kernel.

    Returns a @flyc.jit function with the SAME signature as the baseline
    ``compile_chunk_gated_delta_h``. Produces identical numerical output;
    only the software-pipeline / prefetch scheduling is removed.
    """
    assert K <= 256
    assert K % 64 == 0
    assert BV % 16 == 0
    NUM_K_BLOCKS = K // 64

    _fast_exp = _make_fast_exp(G_IS_LOG2_SCALED)

    WARP_SIZE = 64
    NUM_WARPS = 4
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE

    WMMA_N = 16
    WMMA_K = 32
    N_REPEAT = BV // WMMA_N

    NUM_H_ACCS = NUM_K_BLOCKS * N_REPEAT

    # -- LDS bank-conflict padding (env-overridable for sweeps) --
    def _pad_env(name, default):
        v = os.environ.get(f"FLYDSL_K5_LDS_{name}_PAD")
        return default if v is None else int(v)

    # -- LDS layout: w and k store all K-blocks to reduce barriers --
    LDS_W_STRIDE = K
    LDS_W_ELEMS_PER_STAGE = BT * LDS_W_STRIDE
    LDS_W_STAGES = 1
    LDS_W_ELEMS = LDS_W_ELEMS_PER_STAGE * LDS_W_STAGES
    LDS_W_BYTES = LDS_W_ELEMS * 2

    # lds_k stride padding to break LDS bank conflict on the transpose read.
    LDS_K_PAD = _pad_env("K", 4)  # 4 bf16 = 8 bytes
    LDS_K_STRIDE = K + LDS_K_PAD
    LDS_K_ELEMS = BT * LDS_K_STRIDE
    LDS_K_BYTES = LDS_K_ELEMS * 2

    # lds_vn stride padding (break 2-way bank conflict, +8 B/row).
    LDS_VN_PAD = _pad_env("VN", 4)  # 4 bf16 = 8 bytes
    LDS_VN_STRIDE = BV + LDS_VN_PAD
    LDS_VN_ELEMS = BT * LDS_VN_STRIDE
    LDS_VN_BYTES = LDS_VN_ELEMS * 2

    # lds_h stride padding (break 2-way bank conflict on ds_read_u16).
    LDS_H_PAD = _pad_env("H", 4)  # 4 bf16 = 8 bytes
    LDS_H_STRIDE = BV + LDS_H_PAD
    LDS_H_ELEMS = K * LDS_H_STRIDE
    LDS_H_BYTES = LDS_H_ELEMS * 2

    # Bump revision so the FlyDSL JIT disk cache invalidates on change.
    _K5_KERNEL_REVISION = 1  # naive baseline fork: no prefetch/pipeline

    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(
        None,
        arch=GPU_ARCH,
        global_sym_name=f"gdn_h_naive_smem_v{_K5_KERNEL_REVISION}",
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

    @flyc.kernel(name="chunk_gdn_fwd_h_flydsl_naive")
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

        # -- XOR swizzle: col ^ ((row & 7) << 3) at 8-element granularity --
        def _xor_swizzle(row, col):
            return col ^ ((row & fx.Int32(0x7)) << fx.Int32(3))

        def _xor_swizzle_idx(row, col):
            return col ^ ((row & fx.Index(0x7)) << fx.Index(3))

        # -- LDS vector read helpers (generates ds_read_b128 for 8xbf16) --
        v8bf16_type = T.vec(8, T.bf16)
        lds_w_memref = lds_w_ptr.get()

        def _lds_vec_read_w_bf16x8(elem_idx):
            return vector.load_op(v8bf16_type, lds_w_memref, [elem_idx])

        # -- ds_read_b64_tr_b16 helper (gfx950) --
        v4bf16_type = T.vec(4, T.bf16)

        def _ds_read_tr_bf16x4(lds_byte_offset):
            byte_idx = arith.index_cast(T.index, lds_byte_offset)
            byte_i64 = arith.index_cast(T.i64, byte_idx)
            ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), byte_i64).result
            raw = rocdl.ds_read_tr16_b64(v4bf16_type, ptr).result
            return fx.Vector(raw, (4,), fx.BFloat16)

        # ds_read_b64_tr_b16 lane decomposition
        tr_k_group = (lane % fx.Int32(16)) // fx.Int32(4)
        tr_col_sub = lane % fx.Int32(4)

        # -- Prologue: compute bos, T_local, NT, boh --
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

        # -- MFMA lane mapping for 16x16 tiles --
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

        # -- Load initial state if provided --
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
                    loaded_vec = h0_.vec_load((fx.Index(h0_off_base),), 4)
                    if const_expr(STATE_DTYPE_BF16):
                        loaded_vec = loaded_vec.extf(T.f32x4)
                    acc_idx = kb * N_REPEAT + nr
                    h_accs[acc_idx] = h_accs[acc_idx] + loaded_vec

        NUM_W_LOADS = NUM_K_BLOCKS * NUM_LOAD_BATCHES_64
        NUM_K_LOADS = NUM_K_BLOCKS * NUM_LOAD_BATCHES_64

        c_zero = fx.Index(0)
        c_one = fx.Index(1)
        nt_idx = fx.Index(NT)

        # h_accs is the SSM recurrence state -- carried across chunks via
        # yield (NOT a prefetch). The w prefetch carry of the baseline is
        # removed; w is loaded inline at the top of the loop body.
        init_state = [_to_raw(v) for v in h_accs]

        for i_t, state in range(c_zero, nt_idx, c_one, init=init_state):
            h_accs_in = list(state[:NUM_H_ACCS])
            i_t_i32 = fx.Int32(i_t)

            # ============================================================
            # 1. Store h snapshot to LDS, then coalesced LDS -> HBM
            # ============================================================
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + nr
                    acc_val = h_accs_in[acc_idx]
                    lds_h_col = fx.Int32(nr * 16) + lane_n

                    for elem_i in range_constexpr(4):
                        f32_val = acc_val[elem_i]
                        bf16_val = f32_val.to(fx.BFloat16)

                        lds_h_row = (
                            fx.Int32(kb * 64)
                            + wid * fx.Int32(16)
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
                h_off = h_base + i_t_i32 * stride_h + v_global * fx.Int32(K) + k_idx
                h_[fx.Index(h_off)] = bf16_tile

            gpu.barrier()

            # ============================================================
            # 2. Load w -> lds_w (inline, no prefetch; XOR swizzled)
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
            # 3. Load k -> lds_k (inline, no prefetch; before GEMM2 use)
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
            # 4. GEMM1: b_v = w @ h^T (h from registers).
            # ============================================================
            bv_accs = []
            for _nr in range_constexpr(N_REPEAT):
                bv_accs.append(fx.full(4, 0.0, fx.Float32))

            K_STEPS_PER_BLOCK = 64 // WMMA_K

            for kb in range_constexpr(NUM_K_BLOCKS):
                for ks in range_constexpr(K_STEPS_PER_BLOCK):
                    w_lds_row_idx = wid_idx * fx.Index(16) + lane_n_idx
                    w_lds_col_idx = fx.Index(
                        kb * 64 + ks * WMMA_K
                    ) + lane_m_base_idx * fx.Index(8)
                    w_lds_col_idx = _xor_swizzle_idx(w_lds_row_idx, w_lds_col_idx)
                    w_lds_idx = w_lds_row_idx * fx.Index(LDS_W_STRIDE) + w_lds_col_idx
                    a_frag = _lds_vec_read_w_bf16x8(w_lds_idx)

                    global_ks = kb * K_STEPS_PER_BLOCK + ks

                    for nr in range_constexpr(N_REPEAT):
                        h_k_row = (
                            fx.Int32(global_ks * WMMA_K)
                            + lane_m_base * fx.Int32(8)
                            + tr_k_group
                        )
                        h_v_col = fx.Int32(nr * 16) + tr_col_sub * fx.Int32(4)
                        h_lds_elem = h_k_row * fx.Int32(LDS_H_STRIDE) + h_v_col
                        h_lds_byte = h_lds_elem * fx.Int32(2) + fx.Int32(lds_h_offset)

                        h_lo = _ds_read_tr_bf16x4(h_lds_byte)
                        h_hi = _ds_read_tr_bf16x4(
                            h_lds_byte + fx.Int32(4 * LDS_H_STRIDE * 2)
                        )
                        b_frag = h_lo.shuffle(h_hi, [0, 1, 2, 3, 4, 5, 6, 7])

                        bv_accs[nr] = _mfma_bf16_16x16x32(a_frag, b_frag, bv_accs[nr])

            # ============================================================
            # 5. v_new = u - b_v  (u loaded inline, no prefetch)
            # ============================================================
            next_chunk_end = (i_t_i32 + fx.Int32(1)) * fx.Int32(BT)
            last_idx_raw = (next_chunk_end < T_local).select(
                next_chunk_end, T_local
            ) - fx.Int32(1)

            vn_frags = []
            for nr in range_constexpr(N_REPEAT):
                bv_val = bv_accs[nr]
                u_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_n
                u_f32_elems = []
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
                    u_raw = v_[fx.Index(u_off)]
                    u_bf16 = fx.BFloat16(u_raw)
                    u_f32_elems.append(u_bf16.to(fx.Float32))
                u_f32 = vector.from_elements(T.f32x4, u_f32_elems)

                vn_frags.append(u_f32 - bv_val)

            # ============================================================
            # 5b. Store v_new (pre-gating) for output
            # ============================================================
            if const_expr(SAVE_NEW_VALUE):
                # Closure wrapper to hide ``vn_`` from FlyDSL's
                # ReplaceIfWithDispatch ast rewriter (see baseline comment).
                def _emit_vn_store(off, value):
                    vn_[fx.Index(off)] = value

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

            # ============================================================
            # 6. Gating (g / gk loaded inline)
            # ============================================================
            if const_expr(USE_G):
                g_last_off = i_h * T_flat + (bos + last_idx_raw)
                g_last = g_[fx.Index(g_last_off)]
                exp_g_last = _fast_exp(g_last)

                gate_elems = []
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
                    g_row = g_[fx.Index(g_row_off)]
                    gate = _fast_exp(g_last - g_row)
                    gate_elems.append(in_bounds.select(gate, fx.Float32(0.0)))
                gate_vec = vector.from_elements(T.f32x4, gate_elems)

                for nr in range_constexpr(N_REPEAT):
                    vn_frags[nr] = vn_frags[nr] * gate_vec

                exp_g_last_vec = fx.full(4, fx.Float32(exp_g_last), fx.Float32)

                for kb in range_constexpr(NUM_K_BLOCKS):
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * exp_g_last_vec

            # Per-K decay: h[v, k] *= exp(gk_last[k]) at chunk end.
            if const_expr(USE_GK):
                gk_chunk_base = (
                    (bos + last_idx_raw) * fx.Int32(H * K) + i_h * fx.Int32(K)
                )
                for kb in range_constexpr(NUM_K_BLOCKS):
                    gk_elems = []
                    for elem_i in range_constexpr(4):
                        global_k = (
                            fx.Int32(kb * 64)
                            + wid * fx.Int32(16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        gk_raw = gk_[fx.Index(gk_chunk_base + global_k)]
                        gk_elems.append(_fast_exp(gk_raw))
                    gk_vec = vector.from_elements(T.f32x4, gk_elems)
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * gk_vec

            # ============================================================
            # 7. Store gated v_new -> lds_vn
            # ============================================================
            for nr in range_constexpr(N_REPEAT):
                vn_val = vn_frags[nr]
                lds_col = fx.Int32(nr * 16) + lane_n
                for elem_i in range_constexpr(4):
                    f32_v = vn_val[elem_i]
                    bf16_v = f32_v.to(fx.BFloat16)
                    lds_row = (
                        wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                        + fx.Int32(elem_i)
                    )
                    lds_idx = lds_row * fx.Int32(LDS_VN_STRIDE) + lds_col
                    lds_vn[fx.Index(lds_idx)] = bf16_v

            gpu.barrier()

            # ============================================================
            # 8. GEMM2: h += k^T @ v_new_gated.
            # ============================================================
            BT_STEPS = BT // WMMA_K
            for kb in range_constexpr(NUM_K_BLOCKS):
                for bt_s in range_constexpr(BT_STEPS):
                    k_col_tr = wid * fx.Int32(16) + tr_col_sub * fx.Int32(4)
                    bt_row_tr = (
                        fx.Int32(bt_s * WMMA_K) + lane_m_base * fx.Int32(8) + tr_k_group
                    )
                    k_lds_elem = (
                        bt_row_tr * fx.Int32(LDS_K_STRIDE)
                        + fx.Int32(kb * 64)
                        + k_col_tr
                    )
                    k_lds_byte = k_lds_elem * fx.Int32(2) + fx.Int32(lds_k_offset)

                    k_lo = _ds_read_tr_bf16x4(k_lds_byte)
                    k_hi = _ds_read_tr_bf16x4(
                        k_lds_byte + fx.Int32(4 * LDS_K_STRIDE * 2)
                    )
                    k_a_frag = k_lo.shuffle(k_hi, [0, 1, 2, 3, 4, 5, 6, 7])

                    for nr in range_constexpr(N_REPEAT):
                        vn_bt_row = (
                            fx.Int32(bt_s * WMMA_K)
                            + lane_m_base * fx.Int32(8)
                            + tr_k_group
                        )
                        vn_v_col = fx.Int32(nr * 16) + tr_col_sub * fx.Int32(4)
                        vn_lds_elem = vn_bt_row * fx.Int32(LDS_VN_STRIDE) + vn_v_col
                        vn_lds_byte = vn_lds_elem * fx.Int32(2) + fx.Int32(
                            lds_vn_offset
                        )

                        vn_lo = _ds_read_tr_bf16x4(vn_lds_byte)
                        vn_hi = _ds_read_tr_bf16x4(
                            vn_lds_byte + fx.Int32(4 * LDS_VN_STRIDE * 2)
                        )
                        vn_b_frag = vn_lo.shuffle(vn_hi, [0, 1, 2, 3, 4, 5, 6, 7])

                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = _mfma_bf16_16x16x32(
                            k_a_frag, vn_b_frag, h_accs_in[acc_idx]
                        )

            gpu.barrier()

            results = yield [_to_raw(v) for v in h_accs_in]

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
    "compile_chunk_gated_delta_h_naive",
]
