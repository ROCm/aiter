# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Gated Delta Net K5 hidden-state recurrence kernel using the @flyc.kernel API.

For each chunk t (serial over NT chunks):
  1. Store h snapshot for downstream K6
  2. v_new = u - w @ h   (delta correction via MFMA)
  3. Gated decay + state update:
       v_new *= exp(g_last - g_cumsum)
       h = h * exp(g_last) + k^T @ v_new
"""

import functools
import math

import torch
import triton

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .tensor_shim import GTensor, STensor, _to_raw

_LOG2E = math.log2(math.e)  # 1.4426950408889634
_LLVM_GEP_DYNAMIC = -2147483648


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def _fast_exp(x):
    """exp(x) via exp2(x * log2(e)), maps to single v_exp_f32 on AMD."""
    return rocdl.exp2(T.f32, x * _LOG2E)


def _mfma_bf16_16x16x32(a_bf16x8, b_bf16x8, acc_f32x4):
    """Single mfma_f32_16x16x32_bf16 instruction."""
    return rocdl.mfma_f32_16x16x32_bf16(
        T.f32x4, [a_bf16x8, b_bf16x8, acc_f32x4, 0, 0, 0]
    )


# -- Utility helpers ------------------------------------------------------


def _prepare_lens(cu_seqlens):
    return cu_seqlens[1:] - cu_seqlens[:-1]


@functools.lru_cache(maxsize=8)
def _prepare_chunk_offsets(cu_seqlens_id, chunk_size, device):
    return None


def prepare_chunk_offsets(cu_seqlens, chunk_size):
    lens = _prepare_lens(cu_seqlens)
    return torch.cat(
        [
            cu_seqlens.new_tensor([0]),
            triton.cdiv(lens, chunk_size),
        ]
    ).cumsum(-1)


# -- Compile the kernel ---------------------------------------------------


def compile_chunk_gated_delta_h(
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
):
    """Compile the GDN K5 kernel.

    Returns a @flyc.jit function:
        launch_fn(k, v, w, v_new, g, gk, h, h0, ht,
                  cu_seqlens, chunk_offsets,
                  T_val, T_flat, N_val, stream)
    """
    assert K <= 256
    assert K % 64 == 0
    assert BV % 16 == 0
    NUM_K_BLOCKS = K // 64

    WARP_SIZE = 64
    NUM_WARPS = 4
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE

    WMMA_N = 16
    WMMA_K = 32
    N_REPEAT = BV // WMMA_N

    NUM_H_ACCS = NUM_K_BLOCKS * N_REPEAT

    # -- LDS layout: w and k store all K-blocks to reduce barriers --
    LDS_W_STRIDE = K
    LDS_W_ELEMS = BT * LDS_W_STRIDE
    LDS_W_BYTES = LDS_W_ELEMS * 2

    LDS_K_STRIDE = K
    LDS_K_ELEMS = BT * LDS_K_STRIDE
    LDS_K_BYTES = LDS_K_ELEMS * 2

    # OPT-D: lds_vn stride padding (break 2-way bank conflict, +8 B/row).
    LDS_VN_PAD = 4  # 4 bf16 = 8 bytes
    LDS_VN_STRIDE = BV + LDS_VN_PAD
    LDS_VN_ELEMS = BT * LDS_VN_STRIDE
    LDS_VN_BYTES = LDS_VN_ELEMS * 2

    # OPT-H: lds_h stride padding (break 2-way bank conflict on ds_read_u16).
    LDS_H_PAD = 4  # 4 bf16 = 8 bytes
    LDS_H_STRIDE = BV + LDS_H_PAD
    LDS_H_ELEMS = K * LDS_H_STRIDE
    LDS_H_BYTES = LDS_H_ELEMS * 2

    # Bump revision so the FlyDSL JIT disk cache (~/.flydsl/cache/) invalidates
    # on revision change (port of FlyDSL commit d4643e0e).
    _K5_KERNEL_REVISION = 2  # OPT-D/H/F/7/4 applied

    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(
        None,
        arch=GPU_ARCH,
        global_sym_name=f"gdn_h_smem_v{_K5_KERNEL_REVISION}",
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

    @flyc.kernel(name="chunk_gdn_fwd_h_opt3")
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
        if const_expr(USE_INITIAL_STATE):
            h0_ = GTensor(h0_tensor, dtype=T.f32, shape=(-1,))
        if const_expr(STORE_FINAL_STATE):
            ht_ = GTensor(ht_tensor, dtype=T.f32, shape=(-1,))

        if const_expr(IS_VARLEN):
            cu_ = GTensor(cu_seqlens_tensor, dtype=T.i32, shape=(-1,))
            co_ = GTensor(chunk_offsets_tensor, dtype=T.i32, shape=(-1,))

        # -- LDS views --
        lds_base_ptr = allocator.get_base()

        # w tile (bf16) -- separate from k
        lds_w_ptr = SmemPtr(lds_base_ptr, lds_w_offset, T.bf16, shape=(LDS_W_ELEMS,))
        lds_w = STensor(lds_w_ptr, dtype=T.bf16, shape=(LDS_W_ELEMS,))

        # k tile (bf16) -- separate from w
        lds_k_ptr = SmemPtr(lds_base_ptr, lds_k_offset, T.bf16, shape=(LDS_K_ELEMS,))
        lds_k = STensor(lds_k_ptr, dtype=T.bf16, shape=(LDS_K_ELEMS,))

        # gated v_new (bf16)
        lds_vn_ptr = SmemPtr(lds_base_ptr, lds_vn_offset, T.bf16, shape=(LDS_VN_ELEMS,))
        lds_vn = STensor(lds_vn_ptr, dtype=T.bf16, shape=(LDS_VN_ELEMS,))

        # h snapshot (bf16)
        lds_h_ptr = SmemPtr(lds_base_ptr, lds_h_offset, T.bf16, shape=(LDS_H_ELEMS,))
        lds_h = STensor(lds_h_ptr, dtype=T.bf16, shape=(LDS_H_ELEMS,))

        # -- Cooperative load decomposition --
        load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_64)
        load_col_base = (tid % fx.Int32(THREADS_PER_ROW_64)) * fx.Int32(LOAD_VEC_WIDTH)

        # -- XOR swizzle: col ^ ((row & 7) << 3) at 8-element granularity for bf16 --
        def _xor_swizzle(row, col):
            return col ^ ((row & fx.Int32(0x7)) << fx.Int32(3))

        def _xor_swizzle_idx(row, col):
            return col ^ ((row & arith.index(0x7)) << arith.index(3))

        # -- LDS vector read helpers (generates ds_read_b128 for 8xbf16) --
        v8bf16_type = T.vec(8, T.bf16)
        lds_w_memref = lds_w_ptr.get()
        lds_k_memref = lds_k_ptr.get()

        def _lds_vec_read_w_bf16x8(elem_idx):
            return vector.load_op(v8bf16_type, lds_w_memref, [elem_idx])

        def _lds_vec_read_k_bf16x8(elem_idx):
            return vector.load_op(v8bf16_type, lds_k_memref, [elem_idx])

        # -- ds_read_b64_tr_b16 helper (gfx950) --
        v4bf16_type = T.vec(4, T.bf16)

        def _ds_read_tr_bf16x4(lds_byte_offset):
            byte_idx = arith.index_cast(T.index, lds_byte_offset)
            byte_i64 = arith.index_cast(T.i64, byte_idx)
            ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), byte_i64).result
            return rocdl.ds_read_tr16_b64(v4bf16_type, ptr).result

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
        # h: [B, NT, H, V, K] (VK) -- base = (boh*H + i_h) * V * K
        h_base = (boh * fx.Int32(H) + i_h) * fx.Int32(V * K)
        stride_h = fx.Int32(H * V * K)

        # k: [B, T, Hg, K] -- base = (bos*Hg + i_h//(H//Hg)) * K
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

        # index-typed versions for LDS addressing
        wid_idx = arith.index_cast(T.index, wid)
        lane_n_idx = arith.index_cast(T.index, lane_n)
        lane_m_base_idx = arith.index_cast(T.index, lane_m_base)

        # -- Initialize h accumulators --
        acc_zero = arith.constant_vector(0.0, T.f32x4)

        # h_accs[kb][nr] = f32x4 accumulator for k-block kb, v-repeat nr
        h_accs = []
        for _kb in range_constexpr(NUM_K_BLOCKS):
            for _nr in range_constexpr(N_REPEAT):
                h_accs.append(acc_zero)

        # -- Load initial state if provided --
        # OPT-F: 4 x scalar f32 load -> 1 x buffer_load_dwordx4 (16 B).
        # h0 is [V, K] so K is innermost; 4 consecutive K positions are
        # contiguous in memory -> a single vec_load(4) covers them.
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
                    acc_idx = kb * N_REPEAT + nr
                    h_accs[acc_idx] = arith.addf(h_accs[acc_idx], loaded_vec)

        # -- Software-pipelined main chunk loop --
        NUM_W_LOADS = NUM_K_BLOCKS * NUM_LOAD_BATCHES_64

        # -- Prologue: pre-load first chunk's w data --
        i_t0_i32 = fx.Int32(0)
        w_prefetch_init = []
        for kb in range_constexpr(NUM_K_BLOCKS):
            for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                abs_row = i_t0_i32 * fx.Int32(BT) + row
                in_bounds = arith.cmpi(arith.CmpIPredicate.slt, abs_row, T_local)
                safe_row = arith.select(in_bounds, abs_row, fx.Int32(0))
                g_off = w_base + safe_row * stride_w + fx.Int32(kb * 64) + load_col_base
                w_prefetch_init.append(w_.vec_load((fx.Index(g_off),), LOAD_VEC_WIDTH))

        init_state = [_to_raw(v) for v in h_accs] + [
            _to_raw(v) for v in w_prefetch_init
        ]
        c_zero = arith.index(0)
        c_one = arith.index(1)
        nt_idx = arith.index_cast(T.index, NT)

        for i_t, state in range(c_zero, nt_idx, c_one, init=init_state):
            h_accs_in = list(state[:NUM_H_ACCS])
            w_prefetch_all = list(state[NUM_H_ACCS:])
            i_t_i32 = arith.index_cast(T.i32, i_t)

            # -- 1. Compute w LDS offsets (w data already prefetched) --
            # OPT-4: XOR swizzle to break 64-way bank conflict on lds_w.
            # Pattern: swz_col = col ^ ((row & 7) << 3) at 8-bf16 granularity
            # matching LOAD_VEC_WIDTH. Read path (W A-frag below) applies the
            # SAME swizzle. lds_k / lds_h are NOT swizzled (ds_read_tr_b16
            # spans 4 rows per instr; a row-dependent XOR would break the HW
            # transpose alignment).
            w_prefetch_lds_all = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    col = fx.Int32(kb * 64) + load_col_base
                    swz_col = _xor_swizzle(row, col)
                    w_prefetch_lds_all.append(row * fx.Int32(LDS_W_STRIDE) + swz_col)

            # -- Store h snapshot to LDS --
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + nr
                    acc_val = h_accs_in[acc_idx]
                    lds_h_col = fx.Int32(nr * 16) + lane_n

                    for elem_i in range_constexpr(4):
                        f32_val = vector.extract(
                            acc_val, static_position=[elem_i], dynamic_position=[]
                        )
                        bf16_val = arith.trunc_f(T.bf16, f32_val)

                        lds_h_row = (
                            fx.Int32(kb * 64)
                            + wid * fx.Int32(16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        # OPT-H: stride is LDS_H_STRIDE = BV + LDS_H_PAD
                        lds_h_idx = lds_h_row * fx.Int32(LDS_H_STRIDE) + lds_h_col
                        lds_h[fx.Index(lds_h_idx)] = bf16_val

            gpu.barrier()

            # OPT-H: LDS -> HBM transpose loop.
            # Iteration count uses VK_TOTAL = K * BV (actual elements, NOT
            # LDS_H_ELEMS which now includes padding). Reading uses the padded
            # LDS_H_STRIDE so we hit the same layout as the writer above.
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

            # -- Store prefetched w to LDS (data already in registers from previous iter/prologue) --
            for i_wp in range_constexpr(NUM_W_LOADS):
                lds_w.vec_store(
                    (fx.Index(w_prefetch_lds_all[i_wp]),),
                    w_prefetch_all[i_wp],
                    LOAD_VEC_WIDTH,
                )

            gpu.barrier()

            # -- 2. Delta correction: b_v = w @ h, then v_new = u - b_v --
            k_prefetch = []
            k_prefetch_lds = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    abs_row = i_t_i32 * fx.Int32(BT) + row
                    in_bounds = arith.cmpi(arith.CmpIPredicate.slt, abs_row, T_local)
                    safe_row = arith.select(in_bounds, abs_row, fx.Int32(0))
                    g_off = (
                        k_base + safe_row * stride_k + fx.Int32(kb * 64) + load_col_base
                    )
                    k_prefetch.append(k_.vec_load((fx.Index(g_off),), LOAD_VEC_WIDTH))
                    k_prefetch_lds.append(
                        row * fx.Int32(LDS_K_STRIDE) + fx.Int32(kb * 64) + load_col_base
                    )

            # Compute last_idx for the current chunk (shared by USE_G / USE_GK)
            if const_expr(USE_G or USE_GK):
                next_chunk_end = (i_t_i32 + fx.Int32(1)) * fx.Int32(BT)
                last_idx_raw = arith.select(
                    arith.cmpi(arith.CmpIPredicate.slt, next_chunk_end, T_local),
                    next_chunk_end,
                    T_local,
                ) - fx.Int32(1)

            # Prefetch g values (overlap with MFMA below)
            if const_expr(USE_G):
                g_last_off = (bos + last_idx_raw) * fx.Int32(H) + i_h
                g_last_prefetch = g_[fx.Index(g_last_off)]

                g_row_prefetch = []
                for elem_i in range_constexpr(4):
                    abs_row = (
                        i_t_i32 * fx.Int32(BT)
                        + wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                        + fx.Int32(elem_i)
                    )
                    in_bounds = arith.cmpi(arith.CmpIPredicate.slt, abs_row, T_local)
                    safe_row = arith.select(in_bounds, abs_row, fx.Int32(0))
                    g_row_off = (bos + safe_row) * fx.Int32(H) + i_h
                    g_row_prefetch.append((g_[fx.Index(g_row_off)], in_bounds))

            # Prefetch gk values for per-K h decay at chunk end.
            # h_accs[kb, nr] holds v4f32 with elements at K = kb*64 + wid*16
            #   + lane_m_base*4 + elem_i  (elem_i in 0..3).
            # gk[(bos + last_idx) * H * K + i_h * K + global_k] is one f32 per K.
            if const_expr(USE_GK):
                gk_chunk_base = (bos + last_idx_raw) * fx.Int32(H * K) + i_h * fx.Int32(
                    K
                )
                gk_last_prefetch = []  # [kb][elem_i] -> exp(gk_last)
                for kb in range_constexpr(NUM_K_BLOCKS):
                    kb_elems = []
                    for elem_i in range_constexpr(4):
                        global_k = (
                            fx.Int32(kb * 64)
                            + wid * fx.Int32(16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        gk_off = gk_chunk_base + global_k
                        kb_elems.append(_fast_exp(gk_[fx.Index(gk_off)]))
                    gk_last_prefetch.append(kb_elems)

            # Prefetch u values (overlap with MFMA below)
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
                    u_row_in_bounds = arith.cmpi(
                        arith.CmpIPredicate.slt, u_bt_row_raw, T_local
                    )
                    safe_u_row = arith.select(
                        u_row_in_bounds, u_bt_row_raw, fx.Int32(0)
                    )
                    u_off = v_base + safe_u_row * stride_v + u_col
                    u_prefetch.append(v_[fx.Index(u_off)])

            bv_accs = []
            for _nr in range_constexpr(N_REPEAT):
                bv_accs.append(arith.constant_vector(0.0, T.f32x4))

            K_STEPS_PER_BLOCK = 64 // WMMA_K

            for kb in range_constexpr(NUM_K_BLOCKS):
                for ks in range_constexpr(K_STEPS_PER_BLOCK):
                    w_lds_row_idx = wid_idx * arith.index(16) + lane_n_idx
                    w_lds_col_idx = arith.index(
                        kb * 64 + ks * WMMA_K
                    ) + lane_m_base_idx * arith.index(8)
                    # OPT-4: apply SAME XOR swizzle as the write side.
                    w_lds_col_idx = _xor_swizzle_idx(w_lds_row_idx, w_lds_col_idx)
                    w_lds_idx = (
                        w_lds_row_idx * arith.index(LDS_W_STRIDE) + w_lds_col_idx
                    )
                    a_frag = _lds_vec_read_w_bf16x8(w_lds_idx)

                    global_ks = kb * K_STEPS_PER_BLOCK + ks

                    for nr in range_constexpr(N_REPEAT):
                        h_k_row = (
                            fx.Int32(global_ks * WMMA_K)
                            + lane_m_base * fx.Int32(8)
                            + tr_k_group
                        )
                        h_v_col = fx.Int32(nr * 16) + tr_col_sub * fx.Int32(4)
                        # OPT-H: stride is LDS_H_STRIDE = BV + LDS_H_PAD
                        h_lds_elem = h_k_row * fx.Int32(LDS_H_STRIDE) + h_v_col
                        h_lds_byte = h_lds_elem * fx.Int32(2) + fx.Int32(lds_h_offset)

                        h_lo = _ds_read_tr_bf16x4(h_lds_byte)
                        h_hi = _ds_read_tr_bf16x4(
                            h_lds_byte + fx.Int32(4 * LDS_H_STRIDE * 2)
                        )
                        b_frag = vector.shuffle(h_lo, h_hi, [0, 1, 2, 3, 4, 5, 6, 7])

                        bv_accs[nr] = _mfma_bf16_16x16x32(a_frag, b_frag, bv_accs[nr])

            # v_new = u - b_v (u values already prefetched)
            vn_frags = []
            for nr in range_constexpr(N_REPEAT):
                bv_val = bv_accs[nr]
                u_f32_elems = []
                for elem_i in range_constexpr(4):
                    u_bf16 = u_prefetch[nr * 4 + elem_i]
                    u_f32_elems.append(arith.extf(T.f32, u_bf16))
                u_f32 = vector.from_elements(T.f32x4, u_f32_elems)

                vn_frags.append(arith.subf(u_f32, bv_val))

            # -- 2b. Store v_new (pre-gating) for output --
            if const_expr(SAVE_NEW_VALUE):
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
                        vn_in_bounds = arith.cmpi(
                            arith.CmpIPredicate.slt, vn_bt_row, T_local
                        )
                        _if_vn = scf.IfOp(vn_in_bounds)
                        with ir.InsertionPoint(_if_vn.then_block):
                            f32_v = vector.extract(
                                vn_val, static_position=[elem_i], dynamic_position=[]
                            )
                            bf16_v = arith.trunc_f(T.bf16, f32_v)
                            vn_off = vn_base + vn_bt_row * fx.Int32(V) + vn_col
                            vn_[fx.Index(vn_off)] = bf16_v
                            scf.YieldOp([])

            # -- 3. Gating -- g values prefetched before MFMA --
            if const_expr(USE_G):
                g_last = g_last_prefetch
                exp_g_last = _fast_exp(g_last)

                gate_vec = arith.constant_vector(0.0, T.f32x4)
                for elem_i in range_constexpr(4):
                    g_row, in_bounds = g_row_prefetch[elem_i]
                    gate = _fast_exp(arith.subf(g_last, g_row))
                    gate_masked = arith.select(
                        in_bounds, gate, arith.constant(0.0, type=T.f32)
                    )
                    gate_vec = vector.insert(
                        gate_masked,
                        gate_vec,
                        static_position=[elem_i],
                        dynamic_position=[],
                    )

                for nr in range_constexpr(N_REPEAT):
                    vn_frags[nr] = arith.mulf(vn_frags[nr], gate_vec)

                exp_g_last_vec = arith.constant_vector(0.0, T.f32x4)
                for ei in range_constexpr(4):
                    exp_g_last_vec = vector.insert(
                        exp_g_last,
                        exp_g_last_vec,
                        static_position=[ei],
                        dynamic_position=[],
                    )

                for kb in range_constexpr(NUM_K_BLOCKS):
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = arith.mulf(
                            h_accs_in[acc_idx], exp_g_last_vec
                        )

            # Per-K decay: h[v, k] *= exp(gk_last[k]) at chunk end.
            # Each lane's v4f32 spans 4 different K positions (one per elem_i),
            # so we build a per-kb gate vector and multiply h_accs accordingly.
            if const_expr(USE_GK):
                for kb in range_constexpr(NUM_K_BLOCKS):
                    gk_vec = arith.constant_vector(0.0, T.f32x4)
                    for elem_i in range_constexpr(4):
                        gk_vec = vector.insert(
                            gk_last_prefetch[kb][elem_i],
                            gk_vec,
                            static_position=[elem_i],
                            dynamic_position=[],
                        )
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = arith.mulf(h_accs_in[acc_idx], gk_vec)

            # -- 4. State update: h += k^T @ v_new_gated --
            BT_STEPS = BT // WMMA_K

            # Store gated v_new + all k K-blocks to LDS in one batch, single barrier
            for nr in range_constexpr(N_REPEAT):
                vn_val = vn_frags[nr]
                lds_col = fx.Int32(nr * 16) + lane_n
                for elem_i in range_constexpr(4):
                    f32_v = vector.extract(
                        vn_val, static_position=[elem_i], dynamic_position=[]
                    )
                    bf16_v = arith.trunc_f(T.bf16, f32_v)
                    lds_row = (
                        wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                        + fx.Int32(elem_i)
                    )
                    lds_idx = lds_row * fx.Int32(LDS_VN_STRIDE) + lds_col
                    lds_vn[fx.Index(lds_idx)] = bf16_v

            for i_kp in range_constexpr(NUM_K_BLOCKS * NUM_LOAD_BATCHES_64):
                lds_k.vec_store(
                    (fx.Index(k_prefetch_lds[i_kp]),), k_prefetch[i_kp], LOAD_VEC_WIDTH
                )

            gpu.barrier()

            # -- Prefetch NEXT iteration's w during state update MFMA --
            next_i_t_i32 = i_t_i32 + fx.Int32(1)
            w_next_prefetch = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    abs_row = next_i_t_i32 * fx.Int32(BT) + row
                    in_bounds = arith.cmpi(arith.CmpIPredicate.slt, abs_row, T_local)
                    safe_row = arith.select(in_bounds, abs_row, fx.Int32(0))
                    g_off = (
                        w_base + safe_row * stride_w + fx.Int32(kb * 64) + load_col_base
                    )
                    w_next_prefetch.append(
                        w_.vec_load((fx.Index(g_off),), LOAD_VEC_WIDTH)
                    )

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
                    k_a_frag = vector.shuffle(k_lo, k_hi, [0, 1, 2, 3, 4, 5, 6, 7])

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
                        # OPT-D: stride is LDS_VN_STRIDE = BV + LDS_VN_PAD
                        vn_hi = _ds_read_tr_bf16x4(
                            vn_lds_byte + fx.Int32(4 * LDS_VN_STRIDE * 2)
                        )
                        vn_b_frag = vector.shuffle(
                            vn_lo, vn_hi, [0, 1, 2, 3, 4, 5, 6, 7]
                        )

                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = _mfma_bf16_16x16x32(
                            k_a_frag, vn_b_frag, h_accs_in[acc_idx]
                        )

            results = yield [_to_raw(v) for v in h_accs_in] + [
                _to_raw(v) for v in w_next_prefetch
            ]

        h_accs_final = list(results[:NUM_H_ACCS])

        # -- Epilogue: store final state --
        # OPT-7: 4 x scalar f32 store -> 1 x buffer_store_dwordx4 (16 B).
        # acc_val is already f32x4 with element i at K offset i -> vec_store
        # directly (no extract + from_elements needed).
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
                    ht_.vec_store((fx.Index(ht_off_base),), acc_val, 4)

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


# -- Python wrapper (matches Triton interface) ----------------------------

_compiled_kernels = {}
_autotune_cache = {}  # (shape_key) -> best BV
_BV_CANDIDATES = [16, 32, 64]
_AUTOTUNE_WARMUP = 5
_AUTOTUNE_ITERS = 25


def _get_or_compile(
    K,
    V,
    BT,
    BV,
    H,
    Hg,
    use_g,
    use_gk,
    use_h0,
    store_fs,
    save_vn,
    is_varlen,
    wu_contig,
):
    cache_key = (
        K,
        V,
        BT,
        BV,
        H,
        Hg,
        use_g,
        use_gk,
        use_h0,
        store_fs,
        save_vn,
        is_varlen,
        wu_contig,
    )
    if cache_key not in _compiled_kernels:
        _compiled_kernels[cache_key] = compile_chunk_gated_delta_h(
            K=K,
            V=V,
            BT=BT,
            BV=BV,
            H=H,
            Hg=Hg,
            USE_G=use_g,
            USE_GK=use_gk,
            USE_INITIAL_STATE=use_h0,
            STORE_FINAL_STATE=store_fs,
            SAVE_NEW_VALUE=save_vn,
            IS_VARLEN=is_varlen,
            WU_CONTIGUOUS=wu_contig,
        )
    return _compiled_kernels[cache_key]


def _launch_kernel(
    launch_fn,
    BV,
    V,
    N,
    H,
    k,
    u,
    w,
    vn_arg,
    g_arg,
    gk_arg,
    h,
    h0_arg,
    ht_arg,
    cu_arg,
    co_arg,
    T,
    T_flat,
    stream,
):
    grid_v = triton.cdiv(V, BV)
    grid_nh = N * H
    launch_fn(
        k,
        u,
        w,
        vn_arg,
        g_arg,
        gk_arg,
        h,
        h0_arg,
        ht_arg,
        cu_arg,
        co_arg,
        T,
        T_flat,
        N,
        grid_v,
        grid_nh,
        stream,
    )


def chunk_gated_delta_rule_fwd_h_flydsl(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """FlyDSL K5 host wrapper.

    Signature is API-compatible with
    ``aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h.chunk_gated_delta_rule_fwd_h_opt_vk``:

    Args:
        k: [B, T, Hg, K] bf16.
        w: [B, H, T_flat, K] bf16, head-major contiguous layout.
        u: [B, H, T_flat, V] bf16, head-major contiguous layout.
        g: [T_total, H] f32 cumulative gate, or None.
        gk: [T_total, H, K] f32 per-K cumulative gate, or None.
        initial_state: [N, H, V, K] f32, or None.
        output_final_state: whether to return the final hidden state.
        chunk_size: chunk size BT (default 64).
        save_new_value: whether to materialize ``v_new``.
        cu_seqlens: [N+1] LongTensor for variable-length batching, or None.

    Returns:
        (h, v_new, final_state) in VK-ordered layout (``[..., V, K]`` on the
        last two dims).

    BV-tile selection is internal; results of the very first call for a given
    shape are cached in module-level ``_autotune_cache``.
    """
    # Layout is fixed to head-major contiguous (matches Triton VK wrapper).
    wu_contiguous = True
    BV = 0  # 0 => autotune (cache hit on subsequent calls)

    B, T, Hg, K = k.shape
    BT = chunk_size

    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]

    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(cu_seqlens) - 1
        lens = cu_seqlens[1:] - cu_seqlens[:-1]
        NT = sum(triton.cdiv(int(seq_len), BT) for seq_len in lens.tolist())
        chunk_offsets = (
            torch.cat(
                [
                    cu_seqlens.new_tensor([0]),
                    triton.cdiv(lens, BT),
                ]
            )
            .cumsum(-1)
            .to(torch.int32)
        )

    assert K <= 256

    h = k.new_empty(B, NT, H, V, K)
    final_state = (
        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    )
    v_new_buf = k.new_empty(B, H, T_flat, V, dtype=u.dtype)
    v_new = v_new_buf if save_new_value else None

    dummy = torch.empty(1, device=k.device, dtype=torch.float32)
    g_arg = g if g is not None else dummy
    gk_arg = gk if gk is not None else dummy
    h0_arg = initial_state if initial_state is not None else dummy
    ht_arg = final_state if final_state is not None else dummy
    vn_arg = v_new_buf
    cu_arg = (
        cu_seqlens.to(torch.int32) if cu_seqlens is not None else dummy.to(torch.int32)
    )
    co_arg = chunk_offsets if chunk_offsets is not None else dummy.to(torch.int32)
    stream = torch.cuda.current_stream()

    use_g = g is not None
    use_gk = gk is not None
    use_h0 = initial_state is not None
    is_varlen = cu_seqlens is not None

    # Resolve BV: explicit > autotune cache > benchmark
    if BV <= 0:
        shape_key = (
            K,
            V,
            BT,
            H,
            Hg,
            T_flat,
            N,
            use_g,
            use_gk,
            use_h0,
            output_final_state,
            save_new_value,
            is_varlen,
            wu_contiguous,
        )

        if shape_key in _autotune_cache:
            BV = _autotune_cache[shape_key]
        else:
            candidates = [bv for bv in _BV_CANDIDATES if bv <= V and V % bv == 0]
            if len(candidates) <= 1:
                BV = candidates[0] if candidates else 16
            else:
                print(f"[K5 autotune] benchmarking BV in {candidates} ...")
                best_bv, best_us = candidates[0], float("inf")
                for bv in candidates:
                    fn = _get_or_compile(
                        K,
                        V,
                        BT,
                        bv,
                        H,
                        Hg,
                        use_g,
                        use_gk,
                        use_h0,
                        output_final_state,
                        save_new_value,
                        is_varlen,
                        wu_contiguous,
                    )
                    for _ in range(_AUTOTUNE_WARMUP):
                        _launch_kernel(
                            fn,
                            bv,
                            V,
                            N,
                            H,
                            k,
                            u,
                            w,
                            vn_arg,
                            g_arg,
                            gk_arg,
                            h,
                            h0_arg,
                            ht_arg,
                            cu_arg,
                            co_arg,
                            T,
                            T_flat,
                            stream,
                        )
                    torch.cuda.synchronize()
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(_AUTOTUNE_ITERS):
                        _launch_kernel(
                            fn,
                            bv,
                            V,
                            N,
                            H,
                            k,
                            u,
                            w,
                            vn_arg,
                            g_arg,
                            gk_arg,
                            h,
                            h0_arg,
                            ht_arg,
                            cu_arg,
                            co_arg,
                            T,
                            T_flat,
                            stream,
                        )
                    e.record()
                    torch.cuda.synchronize()
                    us = s.elapsed_time(e) / _AUTOTUNE_ITERS * 1000
                    print(f"  BV={bv:3d}: {us:.2f} us")
                    if us < best_us:
                        best_us = us
                        best_bv = bv
                BV = best_bv
                print(f"[K5 autotune] best BV={BV} ({best_us:.2f} us)")
            _autotune_cache[shape_key] = BV

    launch_fn = _get_or_compile(
        K,
        V,
        BT,
        BV,
        H,
        Hg,
        use_g,
        use_gk,
        use_h0,
        output_final_state,
        save_new_value,
        is_varlen,
        wu_contiguous,
    )
    _launch_kernel(
        launch_fn,
        BV,
        V,
        N,
        H,
        k,
        u,
        w,
        vn_arg,
        g_arg,
        gk_arg,
        h,
        h0_arg,
        ht_arg,
        cu_arg,
        co_arg,
        T,
        T_flat,
        stream,
    )

    return h, v_new, final_state


__all__ = [
    "compile_chunk_gated_delta_h",
    "chunk_gated_delta_rule_fwd_h_flydsl",
]
