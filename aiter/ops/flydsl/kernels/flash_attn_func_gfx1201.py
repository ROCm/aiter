# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Combined Flash Attention kernel for gfx1201 with optimizations:

1. BLOCK_N=32 (reduced tile, fewer iterations, better occupancy; 121->100ms)
2. rocdl.exp2 (native ISA exp2 intrinsic, bypasses arith lowering)
3. Software-pipelined GEMM2: preload next V pack while current WMMA executes,
   hiding LDS read latency behind matrix compute (100->96ms).
4. Overlapped V global load: pre-issue next iteration's V global loads at end
   of current iteration, so V data is in flight during loop back-edge, barrier,
   and K cooperative load of the next iteration (96->91ms).

Note: V interleaved storage (ds_read_b32) was tested but the element-wise
scatter store overhead negates read savings at BN=32. Row-major V with
software-pipelined scalar reads is faster.

Note: V pre-transpose (scatter store to col-major LDS, vec8 GEMM2 read) was
tested but the 16 scalar stores per thread during coop_store_v add +8.8%
regression vs baseline (102.7ms vs 94.3ms).

WMMA 16x16x16 register layout (wave32):
  - A/B operand: v8bf16 per lane (lane16 = row/col, klane*8 = K-offset)
  - C/D result: v8f32 per lane, element si = C[klane*8+si][lane16]

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,)
Block:  (256,) -- 8 waves x 32 threads/wave.

Requires: head_dim % 32 == 0, head_dim >= 64.
"""

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
    vector,
)
from flydsl.expr.typing import T
from .kernels_common import dtype_to_elem_type
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import (
    memref as _memref,
    scf,
    fly as _fly,
    llvm as _llvm,
    math as math_dialect,
)

KERNEL_NAME = "flash_attn_func_gfx1201_c_exp_a_k_noswizzle_kernel"
_LOG2E = math.log2(math.e)
_LLVM_GEP_DYNAMIC = -2147483648


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def build_flash_attn_func_module_primary(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    flat_work_group_size=None,
    block_m=None,
    block_n=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    path_tag="auto",
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

    PATH_TAG = f"M{BLOCK_M}N{BLOCK_N}_combined"
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
        sm_scale = 1.0 / math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    # LDS layout -- K uses padding instead of XOR swizzle; V row-major with padding
    K_STRIDE = HEAD_DIM + 4  # padding to reduce bank conflicts (no swizzle)
    V_STRIDE = HEAD_DIM + 4  # padding to reduce bank conflicts

    ENABLE_LDS_VEC16 = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"flash_attn_func_gfx1201c_exp_a_smem_{PATH_TAG}",
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_func_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        seq_len: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32
        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        k_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), K)
        v_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), V)
        o_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), O)
        fm_fast = arith.FastMathFlags.fast

        v8f32_type = T.vec(8, compute_type)
        v8f16_type = T.vec(8, elem_type)
        v8i16_type = T.vec(8, T.i16)
        vxf16_type = T.vec(VEC_WIDTH, elem_type)

        def wmma_acc(a_v8, b_v8, c_v8):
            if const_expr(dtype_str == "bf16"):
                a_i16 = vector.bitcast(v8i16_type, a_v8)
                b_i16 = vector.bitcast(v8i16_type, b_v8)
                return rocdl.wmma_f32_16x16x16_bf16(
                    v8f32_type, a_i16, b_i16, c_v8
                ).result
            return rocdl.wmma_f32_16x16x16_f16(v8f32_type, a_v8, b_v8, c_v8).result

        seq_len_v = arith.index_cast(T.index, seq_len)

        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(
            base_ptr,
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()

        block_id = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)

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
            token = batch_idx * seq_len_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

        def _gep_load(base_ptr, elem_idx, vec_type):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(
                _llvm_ptr_ty(),
                base_ptr,
                [idx_i64],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=elem_type,
                noWrapFlags=0,
            )
            return _llvm.LoadOp(vec_type, gep.result).result

        def _gep_store(val, base_ptr, elem_idx):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(
                _llvm_ptr_ty(),
                base_ptr,
                [idx_i64],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=elem_type,
                noWrapFlags=0,
            )
            _llvm.StoreOp(val, gep.result)

        def load_global_f16xN(base_ptr, base_idx):
            return _gep_load(base_ptr, base_idx, vxf16_type)

        def load_global_v8f16(base_ptr, base_idx):
            return _gep_load(base_ptr, base_idx, v8f16_type)

        def bf16_trunc_pack_v8(f32_vals):
            _v4i32 = T.vec(4, T.i32)
            _c16 = arith.constant(16, type=T.i32)
            _cmask = arith.constant(0xFFFF0000, type=T.i32)
            pairs = []
            for j in range_constexpr(4):
                a = arith.ArithValue(f32_vals[j * 2]).bitcast(T.i32)
                b = arith.ArithValue(f32_vals[j * 2 + 1]).bitcast(T.i32)
                p = arith.OrIOp(
                    arith.AndIOp(b, _cmask).result, arith.ShRUIOp(a, _c16).result
                ).result
                pairs.append(p)
            return vector.bitcast(v8f16_type, vector.from_elements(_v4i32, pairs))

        def k_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return arith.index(buf_id * LDS_K_TILE_SIZE)
            return buf_id * arith.index(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id):
            return arith.index(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)

        def coop_load_k(tile_start, buf_id=0):
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult, load_row_in_batch, arith.index(BLOCK_N)
                    )
                    _if_k = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_k.then_block):
                        g_idx = global_idx(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        lds_idx = k_base + lds_row * K_STRIDE + load_col_base
                        vec = load_global_f16xN(k_ptr, g_idx)
                        vector.store(vec, lds_kv, [lds_idx])
                        scf.YieldOp([])
                else:
                    g_idx = global_idx(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    lds_idx = k_base + lds_row * K_STRIDE + load_col_base
                    vec = load_global_f16xN(k_ptr, g_idx)
                    vector.store(vec, lds_kv, [lds_idx])

        def _v_store_row_major(v_base, lds_row, vec):
            lds_idx = v_base + lds_row * V_STRIDE + load_col_base
            vector.store(vec, lds_kv, [lds_idx])

        def coop_load_v_global(tile_start):
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                g_idx = global_idx(row_idx, load_col_base)
                vecs.append(load_global_f16xN(v_ptr, g_idx))
            return vecs

        def coop_store_v_lds(vecs, buf_id=0):
            v_base = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult, load_row_in_batch, arith.index(BLOCK_N)
                    )
                    _if_v = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_v.then_block):
                        lds_row = load_row_in_batch + row_offset
                        _v_store_row_major(v_base, lds_row, vecs[batch])
                        scf.YieldOp([])
                else:
                    lds_row = load_row_in_batch + row_offset
                    _v_store_row_major(v_base, lds_row, vecs[batch])

        # ---- Q preload ----
        q_row = q_start + wave_q_offset + lane16
        q_row_i32 = arith.index_cast(T.i32, q_row)
        q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, q_row, seq_len_v)
        q_row_safe = arith.select(q_in_bounds, q_row, arith.index(0))
        c_zero_v8f16 = arith.constant_vector(0.0, v8f16_type)
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = arith.index(ks * K_STEP_QK) + klane * WMMA_LANE_K
            g_idx = global_idx(q_row_safe, q_col)
            raw = load_global_v8f16(q_ptr, g_idx)
            q_b_packs.append(arith.select(q_in_bounds, raw, c_zero_v8f16))

        # ---- Constants ----
        c_neg_inf = arith.constant(float("-inf"), type=compute_type)
        c_zero_f = arith.constant(0.0, type=compute_type)
        c_one_f = arith.constant(1.0, type=compute_type)
        c_sm_scale_log2e = arith.constant(sm_scale * _LOG2E, type=compute_type)
        c_zero_v8f32 = arith.constant_vector(0.0, v8f32_type)
        width_i32 = arith.constant(WARP_SIZE, type=T.i32)
        shuf_16_i32 = arith.constant(16, type=T.i32)

        def reduction_peer(v_f32):
            return arith.ArithValue(v_f32).shuffle_xor(shuf_16_i32, width_i32)

        _q_end = q_start + BLOCK_M
        if const_expr(CAUSAL):
            kv_upper = arith.MinSIOp(_q_end, seq_len_v).result
        else:
            kv_upper = seq_len_v

        # ---- Opt4: Pre-issue first V global load before loop ----
        _v_vecs_init = coop_load_v_global(arith.index(0))

        init_args = [c_neg_inf, c_zero_f]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v8f32)
        # Carry V prefetch vecs as loop-carried values
        for batch in range_constexpr(NUM_BATCHES_KV):
            init_args.append(_v_vecs_init[batch])

        for kv_block_start, inner_iter_args, loop_results in scf.for_(
            arith.index(0),
            kv_upper,
            arith.index(BLOCK_N_OUT),
            iter_args=init_args,
        ):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            _v_vecs_prefetch = [
                inner_iter_args[2 + D_CHUNKS + b]
                for b in range_constexpr(NUM_BATCHES_KV)
            ]

            coop_load_k(kv_block_start, 0)
            gpu.barrier()
            k_base = k_buf_base(0)

            # ==== GEMM1: S = K @ Q^T (no swizzle, padding-based) ====
            s_accs = [c_zero_v8f32 for _ in range(NUM_S_ACCS)]

            for ks in range_constexpr(K_STEPS_QK):
                k_col = arith.index(ks * K_STEP_QK) + klane * WMMA_LANE_K

                for st_idx in range_constexpr(N_SUB_TILES):
                    st_base_row = st_idx * K_SUB_N

                    k_row_a = lane16 + arith.index(st_base_row)
                    k_lds_a = k_base + k_row_a * K_STRIDE + k_col
                    k_pack_a = vector.load(v8f16_type, lds_kv, [k_lds_a])

                    k_row_b = lane16 + arith.index(st_base_row + 16)
                    k_lds_b = k_base + k_row_b * K_STRIDE + k_col
                    k_pack_b = vector.load(v8f16_type, lds_kv, [k_lds_b])

                    acc_idx_a = st_idx * 2
                    acc_idx_b = st_idx * 2 + 1
                    s_accs[acc_idx_a] = wmma_acc(
                        k_pack_a, q_b_packs[ks], s_accs[acc_idx_a]
                    )
                    s_accs[acc_idx_b] = wmma_acc(
                        k_pack_b, q_b_packs[ks], s_accs[acc_idx_b]
                    )

            # ==== Online softmax ====
            s_raw = []
            for st in range_constexpr(NUM_S_ACCS):
                for r in range_constexpr(8):
                    s_raw.append(
                        vector.extract(
                            s_accs[st], static_position=[r], dynamic_position=[]
                        )
                    )

            if const_expr(CAUSAL):
                kv_start_i32 = arith.index_cast(T.i32, kv_block_start)
                klane_i32 = arith.index_cast(T.i32, klane)
                q_start_i32 = arith.index_cast(T.i32, q_start)
                max_kv_col_i32 = arith.AddIOp(
                    kv_start_i32, arith.constant(BLOCK_N - 1, type=T.i32)
                ).result
                tile_needs_mask = arith.cmpi(
                    arith.CmpIPredicate.ugt, max_kv_col_i32, q_start_i32
                )
                _mask_if = scf.IfOp(
                    tile_needs_mask, [T.f32] * NUM_S_VALS, has_else=True
                )
                with ir.InsertionPoint(_mask_if.then_block):
                    _masked = []
                    for st in range_constexpr(NUM_S_ACCS):
                        st_base = st * 16
                        for r in range_constexpr(8):
                            kv_col = arith.AddIOp(
                                arith.AddIOp(
                                    kv_start_i32,
                                    arith.constant(st_base + r, type=T.i32),
                                ).result,
                                arith.MulIOp(
                                    klane_i32, arith.constant(8, type=T.i32)
                                ).result,
                            ).result
                            is_masked = arith.cmpi(
                                arith.CmpIPredicate.ugt, kv_col, q_row_i32
                            )
                            idx = st * 8 + r
                            _masked.append(
                                arith.select(is_masked, c_neg_inf, s_raw[idx])
                            )
                    scf.YieldOp(_masked)
                with ir.InsertionPoint(_mask_if.else_block):
                    scf.YieldOp(s_raw)
                s_raw = [_mask_if.results[i] for i in range(NUM_S_VALS)]

            _max_fm = {"fastmath": fm_fast}
            local_max = s_raw[0]
            for r in range_constexpr(NUM_S_VALS - 1):
                local_max = arith.MaxNumFOp(local_max, s_raw[r + 1], **_max_fm).result
            peer_max = reduction_peer(local_max)
            row_max = arith.MaxNumFOp(local_max, peer_max, **_max_fm).result
            m_new_raw = arith.MaxNumFOp(m_running, row_max, **_max_fm).result

            # ---- Opt2: rocdl.exp2 ----
            diff_m_raw = arith.SubFOp(m_running, m_new_raw, fastmath=fm_fast).result
            diff_m_scaled = arith.MulFOp(
                diff_m_raw, c_sm_scale_log2e, fastmath=fm_fast
            ).result
            corr = rocdl.exp2(ir.F32Type.get(), diff_m_scaled)

            scaled_max = arith.MulFOp(
                c_sm_scale_log2e, m_new_raw, fastmath=fm_fast
            ).result
            neg_scaled_max = arith.SubFOp(c_zero_f, scaled_max, fastmath=fm_fast).result

            p_vals = []
            local_sum = c_zero_f
            for r in range_constexpr(NUM_S_VALS):
                diff = math_dialect.fma(s_raw[r], c_sm_scale_log2e, neg_scaled_max)
                p = rocdl.exp2(ir.F32Type.get(), diff)
                p_vals.append(p)
                local_sum = arith.AddFOp(local_sum, p, fastmath=fm_fast).result

            peer_sum = reduction_peer(local_sum)
            tile_sum = arith.AddFOp(local_sum, peer_sum, fastmath=fm_fast).result
            l_corr = arith.MulFOp(corr, l_running, fastmath=fm_fast).result
            l_new = arith.AddFOp(l_corr, tile_sum, fastmath=fm_fast).result

            corr_vec = vector.broadcast(v8f32_type, corr)
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = arith.MulFOp(o_accs[dc], corr_vec, fastmath=fm_fast).result

            # Store V to LDS (row-major, fast vector store)
            coop_store_v_lds(_v_vecs_prefetch, 0)
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
                            elem_list.append(arith.trunc_f(elem_type, p_slice[j]))
                        p_packs_st.append(vector.from_elements(v8f16_type, elem_list))
                p_packs_all.append(p_packs_st)

            # ==== GEMM2: O += V^T @ P (software pipelined, row-major V) ====
            # Opt3: Prefetch next V pack while current WMMA executes
            v_base = v_buf_base(0)

            def _load_v_rowmajor(st_kv_base_val, pks_val, dc_val):
                d_pos = arith.index(dc_val * D_CHUNK) + lane16
                v_elems = []
                for k_sub in range_constexpr(8):
                    kv_row = (
                        arith.index(st_kv_base_val + pks_val * PV_K_STEP)
                        + klane * WMMA_LANE_K
                        + arith.index(k_sub)
                    )
                    v_lds_idx = v_base + kv_row * V_STRIDE + d_pos
                    v_elems.append(_memref.load(lds_kv, [v_lds_idx]))
                return vector.from_elements(v8f16_type, v_elems)

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
            next_kv_start = kv_block_start + arith.index(BLOCK_N_OUT)
            _v_vecs_next = coop_load_v_global(next_kv_start)

            _yield_args = [m_running, l_running] + o_accs
            for batch in range_constexpr(NUM_BATCHES_KV):
                _yield_args.append(_v_vecs_next[batch])
            yield _yield_args

        # ---- Normalize and store O ----
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        inv_l = arith.DivFOp(c_one_f, l_final, fastmath=fm_fast).result
        inv_l_vec = vector.broadcast(v8f32_type, inv_l)

        _o_guard = scf.IfOp(q_in_bounds, [], has_else=False)
        with ir.InsertionPoint(_o_guard.then_block):
            for dc in range_constexpr(D_CHUNKS):
                o_norm_vec = arith.MulFOp(
                    o_finals[dc], inv_l_vec, fastmath=fm_fast
                ).result
                o_trunc = arith.truncf(v8f16_type, o_norm_vec)
                d_col = arith.index(dc * D_CHUNK) + klane * 8
                o_global = global_idx(q_row, d_col)
                _gep_store(o_trunc, o_ptr, o_global)
            scf.YieldOp([])

    @flyc.jit
    def launch_flash_attn_func(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = arith.index_cast(T.index, batch_size)
        sl_idx = arith.index_cast(T.index, seq_len)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS

        launcher = flash_attn_func_kernel(Q, K, V, O, seq_len)

        if const_expr(waves_per_eu is not None):
            _wpe = int(waves_per_eu)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32, _wpe
                        )
        if const_expr(flat_work_group_size is not None):
            _fwgs = int(flat_work_group_size)
            if const_expr(_fwgs >= 1):
                flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

        passthrough_entries = []
        if const_expr(daz):
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("denormal-fp-math-f32"),
                        ir.StringAttr.get("preserve-sign,preserve-sign"),
                    ]
                )
            )
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("no-nans-fp-math"),
                        ir.StringAttr.get("true"),
                    ]
                )
            )
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("unsafe-fp-math"),
                        ir.StringAttr.get("true"),
                    ]
                )
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return launch_flash_attn_func(*args, **kwargs)

    def _compile(Q, K, V, O, batch_size, seq_len, stream=None):  # noqa: E741
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return flyc.compile(
                launch_flash_attn_func,
                Q,
                K,
                V,
                O,
                batch_size,
                seq_len,
                fx.Stream(stream),
            )

    _launch.compile = _compile
    return _launch


build_flash_attn_func_module = build_flash_attn_func_module_primary
