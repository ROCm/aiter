# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Combined Flash Attention kernel for gfx1201 with optimizations:

1. Shape-selected BLOCK_M/BLOCK_N tiles balance KV-loop work and occupancy.
2. rocdl.exp2 (native ISA exp2 intrinsic, bypasses arith lowering).
3. Software-pipelined GEMM2: preload next V pack while current WMMA executes,
   hiding LDS read latency behind matrix compute.
4. Overlapped V global load: pre-issue next iteration's V global loads at end
   of current iteration, so V data is in flight during loop back-edge, barrier,
   and K cooperative load of the next iteration.

Note: V interleaved storage (ds_read_b32) was tested but the element-wise
scatter store overhead negates read savings at BN=32. Row-major V with
software-pipelined scalar reads is faster.

Note: V pre-transpose (scatter store to col-major LDS, vec8 GEMM2 read) was
tested but the extra scalar stores per thread during coop_store_v regressed
versus the row-major layout.

WMMA 16x16x16 register layout (wave32):
  - A/B operand: v8bf16 per lane (lane16 = row/col, klane*8 = K-offset)
  - C/D result: v8f32 per lane, element si = C[klane*8+si][lane16]

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,)
Block:  (256,) -- 8 waves x 32 threads/wave.

Requires: head_dim % 32 == 0, head_dim >= 64.
"""

import math as host_math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import memref as _memref
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from aiter.ops.flydsl.kernels import buffer_ops

from .flash_attn_func_common_gfx1201 import (
    configure_gpu_module,
    flatten_and_mask_scores,
    kv_load_schedule,
    next_kv_tile_start,
    pointer_arg,
    update_online_softmax,
    wrap_pointer_args,
)
from .flash_attn_func_common_gfx1201 import (
    fast_mul as _fmul,
)
from .flash_attn_func_common_gfx1201 import (
    pointer_load as _pointer_load,
)
from .flash_attn_func_common_gfx1201 import (
    pointer_store as _pointer_store,
)
from .flash_attn_func_common_gfx1201 import (
    pointer_to_llvm_ptr as _pointer_to_llvm_ptr,
)
from .kernels_common import dtype_to_elem_type
from .tensor_shim import _run_compiled

_LOG2E = host_math.log2(host_math.e)


def build_flash_attn_func_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    flat_work_group_size=None,
    block_m=None,
    block_n=None,
    tail_mask=False,
    cross_attn=False,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
):
    """Build gfx1201 flash_attn_func (BN=32 + rocdl.exp2 + pipelined GEMM2 + overlapped V load)."""
    gpu_arch = get_hip_arch()

    # ---- WMMA / wave32 constants ----
    WARP_SIZE = 32
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    K_SUB_N = 32
    ROWS_PER_WAVE = WMMA_M

    BLOCK_M = block_m if block_m is not None else 128
    BLOCK_N = block_n if block_n is not None else 32

    assert (
        BLOCK_N % K_SUB_N == 0
    ), f"BLOCK_N ({BLOCK_N}) must be a multiple of K_SUB_N ({K_SUB_N})"
    assert (
        BLOCK_M % ROWS_PER_WAVE == 0
    ), f"BLOCK_M ({BLOCK_M}) must be a multiple of {ROWS_PER_WAVE}"

    N_SUB_TILES = BLOCK_N // K_SUB_N
    NUM_S_ACCS = N_SUB_TILES * 2
    NUM_S_VALS = NUM_S_ACCS * 8

    NUM_WAVES = BLOCK_M // ROWS_PER_WAVE
    if flat_work_group_size is None:
        flat_work_group_size = NUM_WAVES * WARP_SIZE
    BLOCK_SIZE = flat_work_group_size

    BLOCK_N_OUT = BLOCK_N

    NUM_PREFETCH_K = 1
    NUM_PREFETCH_V = 1

    K_STEP_QK = WMMA_K
    K_STEPS_QK = head_dim // K_STEP_QK
    WMMA_LANE_K = 8

    D_CHUNK = WMMA_N
    D_CHUNKS = head_dim // D_CHUNK

    PV_K_STEP = WMMA_K
    PV_K_STEPS = K_SUB_N // PV_K_STEP

    assert BLOCK_M % NUM_WAVES == 0
    assert head_dim % 32 == 0
    assert head_dim >= 64
    assert dtype_str in ("f16", "bf16")

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    TAIL_MASK = tail_mask
    CROSS_ATTN = cross_attn
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    # Descriptive per-variant symbol name (shows up in profiles / ISA dumps).
    _name_flags = (
        f"{'_causal' if causal else ''}"
        f"{'_cross' if cross_attn else ''}"
        f"{'_tail' if tail_mask else ''}"
    )
    KERNEL_NAME = (
        f"flash_attn_func_gfx1201_{dtype_str}"
        f"_h{num_heads}_d{head_dim}_m{BLOCK_M}n{BLOCK_N}{_name_flags}"
    )

    # LDS layout -- K uses padding instead of XOR swizzle; V row-major with padding
    K_STRIDE = HEAD_DIM + 4  # padding to reduce bank conflicts (no swizzle)
    V_STRIDE = HEAD_DIM + 4  # padding to reduce bank conflicts

    ENABLE_LDS_VEC16 = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8
    (
        THREADS_PER_ROW_LOAD,
        ROWS_PER_BATCH_LOAD,
        NUM_BATCHES_KV,
        KV_NEEDS_GUARD,
    ) = kv_load_schedule(BLOCK_SIZE, HEAD_DIM, BLOCK_N, VEC_WIDTH)

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    # Map dtype string to a FlyDSL Numeric class (for Vec.make_type and `.to(...)`).
    # aiter's `dtype_to_elem_type` returns a raw MLIR `ir.Type`; the FlyDSL Vector
    # API requires a Numeric subclass instead. Both forms are kept available.
    _NUMERIC_MAP = {
        "f32": fx.Float32,
        "f16": fx.Float16,
        "bf16": fx.BFloat16,
    }
    elem_numeric_cls = _NUMERIC_MAP[dtype_str]

    # PERF(gfx1201): FlyDSL 0.2.4 and 0.3.1 lower the SharedAllocator pointer
    # path more than 2x slower than this typed memref path on representative
    # attention shapes. Keep this exception until generated ISA is equivalent.
    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="flash_attn_func_gfx1201_smem"
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_func_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        seq_len: fx.Int32,
        seq_len_real: fx.Int32,
        seq_len_kv: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        elem_dtype = elem_numeric_cls
        q_ptr = _pointer_to_llvm_ptr(Q)
        k_ptr = _pointer_to_llvm_ptr(K)
        v_ptr = _pointer_to_llvm_ptr(V)
        o_ptr = _pointer_to_llvm_ptr(O)
        fm_fast = arith.FastMathFlags.fast

        v8f32_type = Vec.make_type(8, fx.Float32)
        v8f16_type = Vec.make_type(8, elem_dtype)
        vxf16_type = Vec.make_type(VEC_WIDTH, elem_dtype)

        def wmma_acc(a_v8, b_v8, c_v8):
            if const_expr(dtype_str == "bf16"):
                a_i16 = Vec(a_v8).bitcast(fx.Int16)
                b_i16 = Vec(b_v8).bitcast(fx.Int16)
                return rocdl.wmma_f32_16x16x16_bf16(
                    v8f32_type, _raw(a_i16), _raw(b_i16), c_v8
                ).result
            return rocdl.wmma_f32_16x16x16_f16(v8f32_type, a_v8, b_v8, c_v8).result

        seq_len_v = fx.Int64(seq_len)
        seq_len_real_v = fx.Int64(seq_len_real)
        if const_expr(CROSS_ATTN):
            seq_len_kv_v = fx.Int64(seq_len_kv)
        else:
            # Self-attn: K/V share Q's sequence length. Aliasing to seq_len_v (not
            # the seq_len_kv arg) leaves the addressing math unchanged, so the
            # self-attn path pays nothing for the cross-attn arg.
            seq_len_kv_v = seq_len_v

        lds_kv = SmemPtr(
            allocator.get_base(),
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()

        block_id = fx.Int64(gpu.block_idx.x)
        tid = fx.Int64(gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16
        klane = lane // 16

        wave_q_offset = wave_id * ROWS_PER_WAVE

        head_idx = block_id % NUM_HEADS
        batch_q_tile_id = block_id // NUM_HEADS
        num_q_tiles = (seq_len_v + BLOCK_M - 1) // BLOCK_M
        q_tile_idx = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * BLOCK_M

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        def global_idx(token_idx, col):
            # Q + O addressing (Q sequence length).
            token = batch_idx * seq_len_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

        def kv_global_idx(token_idx, col):
            # K + V addressing (KV sequence length). For self-attn seq_len_kv_v is
            # seq_len_v, so this is identical to global_idx.
            token = batch_idx * seq_len_kv_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

        def _load_global_half_vec(ptr, base_idx, vec_type):
            gep = buffer_ops.get_element_ptr(
                ptr, fx.Int64(base_idx), elem_type=elem_type
            )
            return _pointer_load(vec_type, gep)

        def _store_global_half(ptr, base_idx, val):
            gep = buffer_ops.get_element_ptr(
                ptr, fx.Int64(base_idx), elem_type=elem_type
            )
            _pointer_store(val, gep)

        def _bitcast_i32(value):
            return fx.Float32(value).bitcast(fx.Int32)

        def _pack_bf16_pair(lo, hi, shift, mask):
            lo_i32 = _bitcast_i32(lo)
            hi_i32 = _bitcast_i32(hi)
            return (hi_i32 & mask) | lo_i32.shrui(shift)

        def bf16_trunc_pack_v8(f32_vals):
            """Pack 8 f32 values into v8bf16 via bitwise truncation (upper 16 bits)."""
            _c16 = fx.Int32(16)
            _cmask = fx.Int32(0xFFFF0000)
            pairs = []
            for j in range_constexpr(4):
                pairs.append(
                    _pack_bf16_pair(f32_vals[j * 2], f32_vals[j * 2 + 1], _c16, _cmask)
                )
            return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype)

        def coop_load_k(tile_start):
            # The loop induction variable is index-typed; normalize it before
            # composing it with the i64 address offsets below. This follows the
            # current FlyDSL API cleanup on main and is a no-op in gfx1201 ISA.
            tile_start = fx.Int64(tile_start)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Int64(BLOCK_N)
                    if row_valid:
                        g_idx = kv_global_idx(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        lds_idx = lds_row * K_STRIDE + load_col_base
                        vec = _load_global_half_vec(k_ptr, g_idx, vxf16_type)
                        Vec(vec).store(lds_kv, [fx.Index(lds_idx)])
                else:
                    g_idx = kv_global_idx(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    lds_idx = lds_row * K_STRIDE + load_col_base
                    vec = _load_global_half_vec(k_ptr, g_idx, vxf16_type)
                    Vec(vec).store(lds_kv, [fx.Index(lds_idx)])

        def _v_store_row_major(v_base, lds_row, vec):
            lds_idx = v_base + lds_row * V_STRIDE + load_col_base
            Vec(vec).store(lds_kv, [fx.Index(lds_idx)])

        def coop_load_v_global(tile_start):
            tile_start = fx.Int64(tile_start)
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(KV_NEEDS_GUARD):
                    # Guard OOB global read: with BLOCK_SIZE>256 the extra load
                    # threads would read past the tile and fault at the last KV
                    # tile (no allocation slack). Wrap into the valid in-tile row
                    # range; the value is discarded by the guarded LDS store.
                    safe_row = load_row_in_batch % fx.Int64(BLOCK_N)
                    row_idx = tile_start + safe_row + row_offset
                else:
                    row_idx = tile_start + load_row_in_batch + row_offset
                g_idx = kv_global_idx(row_idx, load_col_base)
                vecs.append(_load_global_half_vec(v_ptr, g_idx, vxf16_type))
            return vecs

        def coop_store_v_lds(vecs):
            v_base = fx.Int64(LDS_V_BASE)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Int64(BLOCK_N)
                    if row_valid:
                        lds_row = load_row_in_batch + row_offset
                        _v_store_row_major(v_base, lds_row, vecs[batch])
                else:
                    lds_row = load_row_in_batch + row_offset
                    _v_store_row_major(v_base, lds_row, vecs[batch])

        # ---- Q preload ----
        q_row = q_start + wave_q_offset + lane16
        q_row_i32 = fx.Int32(q_row)
        # Use explicit signed-less-than predicate to match baseline ISA
        # (`v_cmp_gt_i64_e64`) and preserve the baseline ISA hash.
        q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, _raw(q_row), _raw(seq_len_v))
        q_row_safe = fx.Int64(q_in_bounds.select(q_row, fx.Int64(0)))
        c_zero_v8f16 = Vec.filled(8, 0.0, elem_dtype)
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Int64(ks * K_STEP_QK) + klane * WMMA_LANE_K
            g_idx = global_idx(q_row_safe, q_col)
            raw = _load_global_half_vec(q_ptr, g_idx, v8f16_type)
            q_b_packs.append(q_in_bounds.select(raw, c_zero_v8f16))

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        _q_end = q_start + BLOCK_M
        if const_expr(CAUSAL):
            kv_upper = (_q_end < seq_len_v).select(_q_end, seq_len_v)
        else:
            kv_upper = seq_len_real_v

        # ---- Opt4: Pre-issue first V global load before loop ----
        _v_vecs_init = coop_load_v_global(fx.Int64(0))

        init_args = [_raw(c_neg_inf), _raw(c_zero_f)]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(_raw(c_zero_v8f32))
        # Carry V prefetch vecs as loop-carried values
        for batch in range_constexpr(NUM_BATCHES_KV):
            init_args.append(_v_vecs_init[batch])

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(
            fx.Int64(0), kv_upper, fx.Int64(BLOCK_N_OUT), init=init_args
        ):
            kv_block_start = fx.Int64(kv_block_start)
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            _v_vecs_prefetch = [
                inner_iter_args[2 + D_CHUNKS + b]
                for b in range_constexpr(NUM_BATCHES_KV)
            ]

            coop_load_k(kv_block_start)
            gpu.barrier()

            # ==== GEMM1: S = K @ Q^T (no swizzle, padding-based) ====
            s_accs = [_raw(c_zero_v8f32) for _ in range(NUM_S_ACCS)]

            for ks in range_constexpr(K_STEPS_QK):
                k_col = fx.Int64(ks * K_STEP_QK) + klane * WMMA_LANE_K

                for st_idx in range_constexpr(N_SUB_TILES):
                    st_base_row = st_idx * K_SUB_N

                    k_row_a = lane16 + fx.Int64(st_base_row)
                    k_lds_a = k_row_a * K_STRIDE + k_col
                    k_pack_a = Vec.load(v8f16_type, lds_kv, [fx.Index(k_lds_a)])

                    k_row_b = lane16 + fx.Int64(st_base_row + 16)
                    k_lds_b = k_row_b * K_STRIDE + k_col
                    k_pack_b = Vec.load(v8f16_type, lds_kv, [fx.Index(k_lds_b)])

                    acc_idx_a = st_idx * 2
                    acc_idx_b = st_idx * 2 + 1
                    s_accs[acc_idx_a] = wmma_acc(
                        k_pack_a, q_b_packs[ks], s_accs[acc_idx_a]
                    )
                    s_accs[acc_idx_b] = wmma_acc(
                        k_pack_b, q_b_packs[ks], s_accs[acc_idx_b]
                    )

            s_raw = flatten_and_mask_scores(
                s_accs,
                kv_block_start,
                klane,
                q_row_i32,
                seq_len_real,
                c_neg_inf,
                num_s_accs=NUM_S_ACCS,
                causal=CAUSAL,
                tail_mask=TAIL_MASK,
            )
            p_vals, m_new_raw, l_new, o_accs = update_online_softmax(
                s_raw,
                m_running,
                l_running,
                o_accs,
                c_sm_scale_log2e,
                c_zero_f,
                num_s_vals=NUM_S_VALS,
                d_chunks=D_CHUNKS,
            )

            # Store V to LDS (row-major, fast vector store)
            coop_store_v_lds(_v_vecs_prefetch)
            gpu.barrier()

            # ==== Build P packs ====
            p_packs_all = []
            for st_idx in range_constexpr(N_SUB_TILES):
                p_packs_st = []
                for pks in range_constexpr(PV_K_STEPS):
                    acc_idx = st_idx * 2 + pks
                    p_base = acc_idx * 8
                    p_slice = [p_vals[p_base + j] for j in range(8)]

                    if const_expr(dtype_str == "bf16"):
                        p_packs_st.append(bf16_trunc_pack_v8(p_slice))
                    else:
                        elem_list = []
                        for j in range_constexpr(8):
                            elem_list.append(fx.Float32(p_slice[j]).to(elem_dtype))
                        p_packs_st.append(Vec.from_elements(elem_list, elem_dtype))
                p_packs_all.append(p_packs_st)

            # ==== GEMM2: O += V^T @ P (software pipelined, row-major V) ====
            # Opt3: Prefetch next V pack while current WMMA executes
            v_base = fx.Int64(LDS_V_BASE)

            def _load_v_rowmajor(st_kv_base_val, pks_val, dc_val, v_base=v_base):
                d_pos = fx.Int64(dc_val * D_CHUNK) + lane16
                v_elems = []
                for k_sub in range_constexpr(8):
                    kv_row = (
                        fx.Int64(st_kv_base_val + pks_val * PV_K_STEP)
                        + klane * WMMA_LANE_K
                        + fx.Int64(k_sub)
                    )
                    v_lds_idx = v_base + kv_row * V_STRIDE + d_pos
                    v_elems.append(_memref.load(lds_kv, [_raw(fx.Index(v_lds_idx))]))
                return Vec.from_elements(v_elems, elem_dtype)

            # Software pipeline: preload first V pack
            cur_v_packs = []
            for st_idx in range_constexpr(N_SUB_TILES):
                cur_v_packs.append(_load_v_rowmajor(st_idx * K_SUB_N, 0, 0))

            for pks in range_constexpr(PV_K_STEPS):
                for dc in range_constexpr(D_CHUNKS):
                    next_dc = dc + 1
                    next_pks = pks
                    if const_expr(next_dc >= D_CHUNKS):
                        next_dc = 0
                        next_pks = pks + 1
                    has_next = const_expr(next_pks < PV_K_STEPS)

                    # Prefetch next V while current WMMA runs
                    next_v_packs = []
                    if const_expr(has_next):
                        for st_idx in range_constexpr(N_SUB_TILES):
                            next_v_packs.append(
                                _load_v_rowmajor(st_idx * K_SUB_N, next_pks, next_dc)
                            )

                    for st_idx in range_constexpr(N_SUB_TILES):
                        o_accs[dc] = wmma_acc(
                            cur_v_packs[st_idx], p_packs_all[st_idx][pks], o_accs[dc]
                        )

                    if const_expr(has_next):
                        cur_v_packs = next_v_packs

            m_running = m_new_raw
            l_running = l_new

            # ---- Opt4: Issue NEXT iteration's V global load ----
            safe_next_kv_start = next_kv_tile_start(
                kv_block_start, kv_upper, fx.Int64(BLOCK_N_OUT), fx.Int64(0)
            )
            _v_vecs_next = coop_load_v_global(safe_next_kv_start)

            _yield_args = [m_running, l_running] + o_accs
            for batch in range_constexpr(NUM_BATCHES_KV):
                _yield_args.append(_v_vecs_next[batch])
            loop_results = yield _yield_args

        # ---- Normalize and store O ----
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        inv_l = arith.divf(_raw(c_one_f), _raw(l_final), fastmath=fm_fast)
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(8)

        if q_in_bounds:
            for dc in range_constexpr(D_CHUNKS):
                o_norm_vec = _fmul(o_finals[dc], inv_l_vec)
                o_trunc = Vec(o_norm_vec).to(elem_dtype)
                d_col = fx.Int64(dc * D_CHUNK) + klane * 8
                o_global = global_idx(q_row, d_col)
                _store_global_half(o_ptr, o_global, o_trunc)

    @flyc.jit
    def launch_flash_attn_func(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_real: fx.Int32,
        seq_len_kv: fx.Int32,
        stream: fx.Stream = fx.Stream(  # noqa: B008  framework idiom: default is evaluated once at import on purpose
            None
        ),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = fx.Int64(batch_size)
        sl_idx = fx.Int64(seq_len)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS

        flash_attn_func_kernel._func.__name__ = KERNEL_NAME
        launcher = flash_attn_func_kernel(Q, K, V, O, seq_len, seq_len_real, seq_len_kv)

        configure_gpu_module(ctx, waves_per_eu, flat_work_group_size, daz)

        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    launch_flash_attn_func.compile_hints = dict(_fmha_compile_hints)

    def _launch(*args, **kwargs):
        args, kwargs = wrap_pointer_args(args, kwargs, range(4), ("Q", "K", "V", "O"))
        stream = kwargs.pop("stream", fx.Stream(None))
        _run_compiled(launch_flash_attn_func, *args, stream)

    def _compile(
        Q,
        K,
        V,
        O,
        batch_size,
        seq_len,
        seq_len_real,
        seq_len_kv,
        stream=None,
    ):
        return flyc.compile(
            launch_flash_attn_func,
            pointer_arg(Q),
            pointer_arg(K),
            pointer_arg(V),
            pointer_arg(O),
            batch_size,
            seq_len,
            seq_len_real,
            seq_len_kv,
            fx.Stream(stream),
        )

    _launch.compile = _compile
    return _launch
