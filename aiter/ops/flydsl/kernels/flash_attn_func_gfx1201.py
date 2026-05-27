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

import math as host_math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
)
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import ArithValue, _to_raw as _raw
from .kernels_common import dtype_to_elem_type
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import (
    fly as _fly,
    llvm as _llvm,
    memref as _memref,
)

KERNEL_NAME = "flash_attn_func_gfx1201_c_exp_a_k_noswizzle_kernel"
_LOG2E = host_math.log2(host_math.e)


def _llvm_value(value):
    """Unwrap FlyDSL scalar/vector wrappers for LLVM pointer load ops."""
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _extract_aligned_pointer(tensor) -> ir.Value:
    """Extract the aligned LLVM pointer from a FlyDSL tensor/memref."""
    return _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), _llvm_value(tensor))


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return _llvm.LoadOp(result_type, _llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return _llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


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
        sm_scale = 1.0 / host_math.sqrt(head_dim)

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

    # Map dtype string to a FlyDSL Numeric class (for Vec.make_type and `.to(...)`).
    # aiter's `dtype_to_elem_type` returns a raw MLIR `ir.Type`; the FlyDSL Vector
    # API requires a Numeric subclass instead. Both forms are kept available.
    _NUMERIC_MAP = {
        "f32": fx.Float32,
        "f16": fx.Float16,
        "bf16": fx.BFloat16,
    }
    elem_numeric_cls = _NUMERIC_MAP[dtype_str]

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_func_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        seq_len: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        elem_dtype = elem_numeric_cls
        q_ptr = _extract_aligned_pointer(Q)
        k_ptr = _extract_aligned_pointer(K)
        v_ptr = _extract_aligned_pointer(V)
        o_ptr = _extract_aligned_pointer(O)
        fm_fast = arith.FastMathFlags.fast

        # Local fast-math arithmetic helpers — preserve fastmath flag while using
        # the lowercase op names that accept _raw() unwrapping (PR #462 pattern).
        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

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

        seq_len_v = fx.Index(seq_len)

        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(
            base_ptr,
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)

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

        def load_global_f16xN(base_ptr, base_idx):
            return _load_global_half_vec(base_ptr, base_idx, vxf16_type)

        def load_global_v8f16(base_ptr, base_idx):
            return _load_global_half_vec(base_ptr, base_idx, v8f16_type)

        def _bitcast_i32(value):
            return fx.Int32(ArithValue(value).bitcast(fx.Int32.ir_type))

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
            return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()

        def k_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(buf_id * LDS_K_TILE_SIZE)
            return buf_id * fx.Index(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id):
            return fx.Index(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)

        def coop_load_k(tile_start, buf_id=0):
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        g_idx = global_idx(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        lds_idx = k_base + lds_row * K_STRIDE + load_col_base
                        vec = load_global_f16xN(k_ptr, g_idx)
                        Vec(vec).store(lds_kv, [lds_idx])
                else:
                    g_idx = global_idx(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    lds_idx = k_base + lds_row * K_STRIDE + load_col_base
                    vec = load_global_f16xN(k_ptr, g_idx)
                    Vec(vec).store(lds_kv, [lds_idx])

        def _v_store_row_major(v_base, lds_row, vec):
            lds_idx = v_base + lds_row * V_STRIDE + load_col_base
            Vec(vec).store(lds_kv, [lds_idx])

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
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
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
        # (`v_cmp_gt_i64_e64`). fx.Index defaults to unsigned which would lower
        # to `v_cmp_gt_u64_e64` and cause an ISA hash drift even though both
        # variants are semantically equivalent for non-negative offsets.
        q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, _raw(q_row), _raw(seq_len_v))
        q_row_safe = fx.Index(ArithValue(q_in_bounds).select(q_row, fx.Index(0)))
        c_zero_v8f16 = Vec.filled(8, 0.0, elem_dtype).ir_value()
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Index(ks * K_STEP_QK) + klane * WMMA_LANE_K
            g_idx = global_idx(q_row_safe, q_col)
            raw = load_global_v8f16(q_ptr, g_idx)
            q_b_packs.append(ArithValue(q_in_bounds).select(raw, c_zero_v8f16))

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_16_i32 = fx.Int32(16)

        def reduction_peer(v_f32):
            return fx.Float32(v_f32).shuffle_xor(shuf_16_i32, width_i32)

        _q_end = q_start + BLOCK_M
        if const_expr(CAUSAL):
            kv_upper = fx.Index(
                ArithValue(_q_end < seq_len_v).select(_q_end, seq_len_v)
            )
        else:
            kv_upper = seq_len_v

        # ---- Opt4: Pre-issue first V global load before loop ----
        _v_vecs_init = coop_load_v_global(fx.Index(0))

        init_args = [_raw(c_neg_inf), _raw(c_zero_f)]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(_raw(c_zero_v8f32))
        # Carry V prefetch vecs as loop-carried values
        for batch in range_constexpr(NUM_BATCHES_KV):
            init_args.append(_v_vecs_init[batch])

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(
            0, kv_upper, BLOCK_N_OUT, init=init_args
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
            s_accs = [_raw(c_zero_v8f32) for _ in range(NUM_S_ACCS)]

            for ks in range_constexpr(K_STEPS_QK):
                k_col = fx.Index(ks * K_STEP_QK) + klane * WMMA_LANE_K

                for st_idx in range_constexpr(N_SUB_TILES):
                    st_base_row = st_idx * K_SUB_N

                    k_row_a = lane16 + fx.Index(st_base_row)
                    k_lds_a = k_base + k_row_a * K_STRIDE + k_col
                    k_pack_a = Vec.load(v8f16_type, lds_kv, [k_lds_a])

                    k_row_b = lane16 + fx.Index(st_base_row + 16)
                    k_lds_b = k_base + k_row_b * K_STRIDE + k_col
                    k_pack_b = Vec.load(v8f16_type, lds_kv, [k_lds_b])

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
                    s_raw.append(Vec(s_accs[st])[r])

            if const_expr(CAUSAL):
                kv_start_i32 = fx.Int32(kv_block_start)
                klane_i32 = fx.Int32(klane)
                q_start_i32 = fx.Int32(q_start)
                max_kv_col_i32 = kv_start_i32 + fx.Int32(BLOCK_N - 1)
                tile_needs_mask = max_kv_col_i32 > q_start_i32

                # SSA-style restructure (PR #462 pattern, lines 700-870):
                # FlyDSL's `if` rewriter requires each loop-carried/conditional
                # state variable to be a single MLIR Value, not a list. Unfold
                # `s_raw[0..NUM_S_VALS-1]` into NUM_S_VALS named scalars, then
                # reassign each one inside the `if tile_needs_mask:` branch.
                # NUM_S_VALS == NUM_S_ACCS * 8 == 16 for BLOCK_N=32.
                s_v0 = s_raw[0]
                s_v1 = s_raw[1]
                s_v2 = s_raw[2]
                s_v3 = s_raw[3]
                s_v4 = s_raw[4]
                s_v5 = s_raw[5]
                s_v6 = s_raw[6]
                s_v7 = s_raw[7]
                s_v8 = s_raw[8]
                s_v9 = s_raw[9]
                s_v10 = s_raw[10]
                s_v11 = s_raw[11]
                s_v12 = s_raw[12]
                s_v13 = s_raw[13]
                s_v14 = s_raw[14]
                s_v15 = s_raw[15]
                if tile_needs_mask:
                    klane_off_i32 = klane_i32 * fx.Int32(8)
                    # st=0
                    _b0 = kv_start_i32 + fx.Int32(0) + klane_off_i32
                    s_v0 = ArithValue(_b0 > q_row_i32).select(c_neg_inf, s_v0)
                    _b1 = kv_start_i32 + fx.Int32(1) + klane_off_i32
                    s_v1 = ArithValue(_b1 > q_row_i32).select(c_neg_inf, s_v1)
                    _b2 = kv_start_i32 + fx.Int32(2) + klane_off_i32
                    s_v2 = ArithValue(_b2 > q_row_i32).select(c_neg_inf, s_v2)
                    _b3 = kv_start_i32 + fx.Int32(3) + klane_off_i32
                    s_v3 = ArithValue(_b3 > q_row_i32).select(c_neg_inf, s_v3)
                    _b4 = kv_start_i32 + fx.Int32(4) + klane_off_i32
                    s_v4 = ArithValue(_b4 > q_row_i32).select(c_neg_inf, s_v4)
                    _b5 = kv_start_i32 + fx.Int32(5) + klane_off_i32
                    s_v5 = ArithValue(_b5 > q_row_i32).select(c_neg_inf, s_v5)
                    _b6 = kv_start_i32 + fx.Int32(6) + klane_off_i32
                    s_v6 = ArithValue(_b6 > q_row_i32).select(c_neg_inf, s_v6)
                    _b7 = kv_start_i32 + fx.Int32(7) + klane_off_i32
                    s_v7 = ArithValue(_b7 > q_row_i32).select(c_neg_inf, s_v7)
                    # st=1 (st_base=16)
                    _b8 = kv_start_i32 + fx.Int32(16) + klane_off_i32
                    s_v8 = ArithValue(_b8 > q_row_i32).select(c_neg_inf, s_v8)
                    _b9 = kv_start_i32 + fx.Int32(17) + klane_off_i32
                    s_v9 = ArithValue(_b9 > q_row_i32).select(c_neg_inf, s_v9)
                    _b10 = kv_start_i32 + fx.Int32(18) + klane_off_i32
                    s_v10 = ArithValue(_b10 > q_row_i32).select(c_neg_inf, s_v10)
                    _b11 = kv_start_i32 + fx.Int32(19) + klane_off_i32
                    s_v11 = ArithValue(_b11 > q_row_i32).select(c_neg_inf, s_v11)
                    _b12 = kv_start_i32 + fx.Int32(20) + klane_off_i32
                    s_v12 = ArithValue(_b12 > q_row_i32).select(c_neg_inf, s_v12)
                    _b13 = kv_start_i32 + fx.Int32(21) + klane_off_i32
                    s_v13 = ArithValue(_b13 > q_row_i32).select(c_neg_inf, s_v13)
                    _b14 = kv_start_i32 + fx.Int32(22) + klane_off_i32
                    s_v14 = ArithValue(_b14 > q_row_i32).select(c_neg_inf, s_v14)
                    _b15 = kv_start_i32 + fx.Int32(23) + klane_off_i32
                    s_v15 = ArithValue(_b15 > q_row_i32).select(c_neg_inf, s_v15)
                s_raw = [
                    s_v0,
                    s_v1,
                    s_v2,
                    s_v3,
                    s_v4,
                    s_v5,
                    s_v6,
                    s_v7,
                    s_v8,
                    s_v9,
                    s_v10,
                    s_v11,
                    s_v12,
                    s_v13,
                    s_v14,
                    s_v15,
                ]

            local_max = s_raw[0]
            for r in range_constexpr(NUM_S_VALS - 1):
                local_max = _fmax(local_max, s_raw[r + 1])
            peer_max = reduction_peer(local_max)
            row_max = _fmax(local_max, peer_max)
            m_new_raw = _fmax(m_running, row_max)

            # ---- Opt2: rocdl.exp2 ----
            diff_m_raw = _fsub(m_running, m_new_raw)
            diff_m_scaled = _fmul(diff_m_raw, c_sm_scale_log2e)
            corr = rocdl.exp2(ir.F32Type.get(), _raw(diff_m_scaled))

            scaled_max = _fmul(c_sm_scale_log2e, m_new_raw)
            neg_scaled_max = _fsub(c_zero_f, scaled_max)

            p_vals = []
            local_sum = _raw(c_zero_f)
            for r in range_constexpr(NUM_S_VALS):
                diff = fmath.fma(s_raw[r], _raw(c_sm_scale_log2e), neg_scaled_max)
                p = rocdl.exp2(ir.F32Type.get(), _raw(diff))
                p_vals.append(p)
                local_sum = _fadd(local_sum, p)

            peer_sum = reduction_peer(local_sum)
            tile_sum = _fadd(local_sum, peer_sum)
            l_corr = _fmul(corr, l_running)
            l_new = _fadd(l_corr, tile_sum)

            corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(8).ir_value()
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = _fmul(o_accs[dc], corr_vec)

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
                            elem_list.append(fx.Float32(p_slice[j]).to(elem_dtype))
                        p_packs_st.append(
                            Vec.from_elements(elem_list, elem_dtype).ir_value()
                        )
                p_packs_all.append(p_packs_st)

            # ==== GEMM2: O += V^T @ P (software pipelined, row-major V) ====
            # Opt3: Prefetch next V pack while current WMMA executes
            v_base = v_buf_base(0)

            def _load_v_rowmajor(st_kv_base_val, pks_val, dc_val):
                d_pos = fx.Index(dc_val * D_CHUNK) + lane16
                v_elems = []
                for k_sub in range_constexpr(8):
                    kv_row = (
                        fx.Index(st_kv_base_val + pks_val * PV_K_STEP)
                        + klane * WMMA_LANE_K
                        + fx.Index(k_sub)
                    )
                    v_lds_idx = v_base + kv_row * V_STRIDE + d_pos
                    # Kept as raw memref.load: scalar element load with no
                    # direct Vec equivalent — Vec is for SIMD vectors.
                    v_elems.append(_memref.load(lds_kv, [_raw(v_lds_idx)]))
                return Vec.from_elements(v_elems, elem_dtype).ir_value()

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
            next_kv_start = kv_block_start + fx.Index(BLOCK_N_OUT)
            _v_vecs_next = coop_load_v_global(next_kv_start)

            _yield_args = [m_running, l_running] + o_accs
            for batch in range_constexpr(NUM_BATCHES_KV):
                _yield_args.append(_v_vecs_next[batch])
            loop_results = yield _yield_args

        # ---- Normalize and store O ----
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        inv_l = arith.divf(_raw(c_one_f), _raw(l_final), fastmath=fm_fast)
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(8).ir_value()

        if q_in_bounds:
            for dc in range_constexpr(D_CHUNKS):
                o_norm_vec = _fmul(o_finals[dc], inv_l_vec)
                o_trunc = Vec(o_norm_vec).to(elem_dtype).ir_value()
                d_col = fx.Index(dc * D_CHUNK) + klane * 8
                o_global = global_idx(q_row, d_col)
                _store_global_half(o_ptr, o_global, o_trunc)

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

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
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
