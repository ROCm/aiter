# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FP8-input Flash Attention kernel for gfx1201 (RDNA4).

Q, K, V arrive as fp8 (e4m3) in HBM with per-tensor scales q_scale, k_scale,
v_scale; output O is bf16. Taking fp8 directly (rather than converting bf16->fp8
inside the kernel) halves K/V HBM traffic and does the quantization once upstream
instead of on every q-tile re-stream. q_scale*k_scale folds into the softmax
scale; v_scale folds into the 1/l normalizer.

Performance-relevant choices:

1. Shape-selected BLOCK_M/BLOCK_N tiles balance KV-loop work and occupancy.
2. rocdl.exp2: native ISA exp2 intrinsic, bypasses arith lowering.
3. Software-pipelined GEMM2: preload the next V pack while the current WMMA
   runs, hiding LDS read latency behind matrix compute.
4. Overlapped V global load: pre-issue the next iteration's V global loads at
   the end of the current one, so V is in flight across the loop back-edge,
   barrier, and the next K cooperative load.

Approaches tried and dropped:
  - V interleaved storage (ds_read_b32): the element-wise scatter store overhead
    negates the read savings at BN=32; row-major V with pipelined scalar reads
    is faster.
  - V pre-transpose (scatter store to col-major LDS, vec8 GEMM2 read): the 16
    scalar stores per thread in coop_store_v cost ~8.8% over the row-major
    layout.

WMMA 16x16x16 register layout (wave32):
  - A/B operand: v8bf16 per lane (lane16 = row/col, klane*8 = K-offset)
  - C/D result: v8f32 per lane, element si = C[klane*8+si][lane16]

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,)
Block:  (256,) -- 8 waves x 32 threads/wave.

Requires: head_dim % 32 == 0, head_dim >= 64.
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import (
    memref as _memref,
)
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
)
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

try:
    from flydsl.expr import buffer_ops
except ImportError:
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
        f"flash_attn_func_fp8_gfx1201"
        f"_h{num_heads}_d{head_dim}_m{BLOCK_M}n{BLOCK_N}{_name_flags}"
    )

    # LDS layout -- K is row-major; V is transposed with the KV row contiguous.
    K_STRIDE = HEAD_DIM + 4  # padding to reduce bank conflicts (no swizzle)
    K_STRIDE_I32 = K_STRIDE // 4  # K in i32 units (4 fp8 per i32)

    # The FP8 packing and transposed LDS layout require one 16-byte vector per
    # lane. The BF16 kernel's vec8 diagnostic toggle does not apply here.
    VEC_WIDTH = 16
    (
        THREADS_PER_ROW_LOAD,
        ROWS_PER_BATCH_LOAD,
        NUM_BATCHES_KV,
        KV_NEEDS_GUARD,
    ) = kv_load_schedule(BLOCK_SIZE, HEAD_DIM, BLOCK_N, VEC_WIDTH)

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    # FP8 V is transposed in LDS (V_T[d][kv_row]) so GEMM2 reads contiguous
    # v2i32. All V sizing and addressing is expressed directly in bytes.
    KV_STRIDE_FP8 = BLOCK_N + 4  # fp8 bytes per d-row (kv_row inner + pad)
    KV_STRIDE_I32_FP8 = KV_STRIDE_FP8 // 4  # i32 words per d-row (contiguous kv load)
    V_BYTE_BASE = LDS_K_TOTAL_SIZE  # fp8 V region starts after fp8 K region (bytes)
    LDS_V_TOTAL_BYTES = NUM_PREFETCH_V * HEAD_DIM * KV_STRIDE_FP8
    LDS_TOTAL_BYTES = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_BYTES

    # PERF(gfx1201): Keep typed SmemPtr views and their required fx.Index
    # boundaries. FlyDSL 0.2.4 and 0.3.1 SharedAllocator byte-pointer lowering
    # makes representative FP8 attention shapes 3.7-5.1x slower. Revisit only
    # when the generated ISA and latency are equivalent.
    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"flash_attn_func_fp8_gfx1201c_exp_a_smem_{PATH_TAG}",
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    # FP8 K (1 byte/elem) followed by the transposed FP8 V region.
    allocator.ptr = lds_kv_offset + LDS_TOTAL_BYTES

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
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        seq_len: fx.Int32,
        seq_len_real: fx.Int32,
        seq_len_kv: fx.Int32,
        q_scale_ptr: fx.Pointer,
        k_scale_ptr: fx.Pointer,
        v_scale_ptr: fx.Pointer,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        elem_dtype = elem_numeric_cls
        q_ptr = _pointer_to_llvm_ptr(Q)
        k_ptr = _pointer_to_llvm_ptr(K)
        v_ptr = _pointer_to_llvm_ptr(V)
        o_ptr = _pointer_to_llvm_ptr(O)
        # fp8-input: per-tensor scales arrive as device pointers (the upstream quant
        # op writes amax to device, no host .item() sync). Load one f32 in the prologue.
        _f32_ty = ir.F32Type.get()
        q_scale = _pointer_load(_f32_ty, _pointer_to_llvm_ptr(q_scale_ptr))
        k_scale = _pointer_load(_f32_ty, _pointer_to_llvm_ptr(k_scale_ptr))
        v_scale = _pointer_load(_f32_ty, _pointer_to_llvm_ptr(v_scale_ptr))
        fm_fast = arith.FastMathFlags.fast

        v8f32_type = Vec.make_type(8, fx.Float32)
        v2i32_type = Vec.make_type(2, fx.Int32)
        v4i32_type = Vec.make_type(4, fx.Int32)
        v16i8_type = Vec.make_type(16, fx.Int8)
        _i8_input_ty = ir.IntegerType.get_signless(8)

        def wmma_acc_fp8(k_v2i32_raw, q_pk_pair, c_v8):
            # FP8 WMMA for GEMM1 (S = K @ Q^T). The gfx12 mma_base params are the
            # same for bf16 and fp8, so A/B semantics match the bf16 path: a=K, b=Q.
            # k_v2i32_raw: raw MLIR Value of vector<2xi32> — K fragment → WMMA A
            # q_pk_pair: [i32_val0, i32_val1] — Q fragment → WMMA B
            q_vec = Vec.from_elements(q_pk_pair, fx.Int32).ir_value()
            return rocdl.wmma_f32_16x16x16_fp8_fp8(
                res=v8f32_type, a=k_v2i32_raw, b=q_vec, c=c_v8
            ).result

        seq_len_v = fx.Index(seq_len)
        seq_len_real_v = fx.Index(seq_len_real)
        if const_expr(CROSS_ATTN):
            seq_len_kv_v = fx.Index(seq_len_kv)
        else:
            # Self-attn: K/V share Q's sequence length. Aliasing to seq_len_v (not
            # the seq_len_kv arg) leaves the addressing math unchanged, so the
            # self-attn path pays nothing for the cross-attn arg.
            seq_len_kv_v = seq_len_v

        base_ptr = allocator.get_base()
        # FP8 K region indexed in i32 units (4 fp8/i32, ds_read_b64 for v2i32 loads).
        # Byte offset same as lds_kv_offset; elem_type=i32; shape=LDS_K_TOTAL_SIZE//4 i32 words.
        _i32_mlir_type = ir.IntegerType.get_signless(32)
        lds_k_i32 = SmemPtr(
            base_ptr,
            lds_kv_offset,
            _i32_mlir_type,
            shape=(LDS_K_TOTAL_SIZE // 4,),
        ).get()
        # fp8 V region views (same base_ptr/offset as lds_kv):
        # i8 view → byte-addressable GEMM2 read; i32 view → vectorized convert-store.
        _i8_mlir_type = ir.IntegerType.get_signless(8)
        lds_v_i8 = SmemPtr(
            base_ptr,
            lds_kv_offset,
            _i8_mlir_type,
            shape=(LDS_TOTAL_BYTES,),
        ).get()
        lds_v_i32 = SmemPtr(
            base_ptr,
            lds_kv_offset,
            _i32_mlir_type,
            shape=(LDS_TOTAL_BYTES // 4,),
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
            # Q + O addressing (Q sequence length).
            token = batch_idx * seq_len_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

        def kv_global_idx(token_idx, col):
            # K + V addressing (KV sequence length). For self-attn seq_len_kv_v is
            # seq_len_v, so this is identical to global_idx.
            token = batch_idx * seq_len_kv_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

        def _store_global_half(ptr, base_idx, val):
            gep = buffer_ops.get_element_ptr(
                ptr, fx.Int64(base_idx), elem_type=elem_type
            )
            _pointer_store(val, gep)

        def _load_global_fp8(base_ptr, base_idx, vec_type):
            # fp8-input: base_idx is in element units == byte units (fp8 = 1 byte).
            gep = buffer_ops.get_element_ptr(
                base_ptr, fx.Int64(base_idx), elem_type=_i8_input_ty
            )
            return _pointer_load(vec_type, gep)

        def coop_load_k(tile_start):
            # Load K global (already fp8), store as i32 words in lds_k_i32.
            # LDS index in i32 units: k_base_i32 = buf_id * LDS_K_TILE_SIZE // 4
            # Per-row: lds_row * K_STRIDE_I32 + load_col_base // 4
            # VEC_WIDTH=16 fp8 per thread = 4 i32 → stored at consecutive i32 slots.
            k_base_i32 = fx.Index(0)
            load_col_i32 = load_col_base // fx.Index(4)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        g_idx = kv_global_idx(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        lds_i32_idx = (
                            k_base_i32 + lds_row * fx.Index(K_STRIDE_I32) + load_col_i32
                        )
                        v4 = _load_global_fp8(k_ptr, g_idx, v4i32_type)
                        fp8_i32s = [Vec(v4)[wi] for wi in range(4)]
                        for wi in range_constexpr(4):
                            _memref.store(
                                _raw(fp8_i32s[wi]),
                                lds_k_i32,
                                [_raw(lds_i32_idx + fx.Index(wi))],
                            )
                else:
                    g_idx = kv_global_idx(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    lds_i32_idx = (
                        k_base_i32 + lds_row * fx.Index(K_STRIDE_I32) + load_col_i32
                    )
                    v4 = _load_global_fp8(k_ptr, g_idx, v4i32_type)
                    fp8_i32s = [Vec(v4)[wi] for wi in range(4)]
                    for wi in range_constexpr(4):
                        _memref.store(
                            _raw(fp8_i32s[wi]),
                            lds_k_i32,
                            [_raw(lds_i32_idx + fx.Index(wi))],
                        )

        def _v_store_row_major_fp8(lds_row, v_bytes):
            # fp8-input: V already fp8 bytes (v16i8) — no convert, just scatter-store
            # TRANSPOSED (V_T[d][kv_row]). 16 d-values of this lane land in 16 d-rows
            # at the same kv column (stride KV_STRIDE_FP8). Makes GEMM2 load contiguous.
            for j in range_constexpr(VEC_WIDTH):
                d_col = load_col_base + fx.Index(j)
                byte_idx = (
                    fx.Index(V_BYTE_BASE) + d_col * fx.Index(KV_STRIDE_FP8) + lds_row
                )
                _memref.store(_raw(v_bytes[j]), lds_v_i8, [_raw(byte_idx)])

        def coop_load_v_global(tile_start):
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(KV_NEEDS_GUARD):
                    # Guard OOB global read: with BLOCK_SIZE>256 the extra load
                    # threads would read past the tile and fault at the last KV
                    # tile (no allocation slack). Wrap into the valid in-tile row
                    # range; the value is discarded by the guarded LDS store.
                    safe_row = load_row_in_batch % fx.Index(BLOCK_N)
                    row_idx = tile_start + safe_row + row_offset
                else:
                    row_idx = tile_start + load_row_in_batch + row_offset
                g_idx = kv_global_idx(row_idx, load_col_base)
                vecs.append(_load_global_fp8(v_ptr, g_idx, v16i8_type))
            return vecs

        def coop_store_v_lds(vecs):
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        lds_row = load_row_in_batch + row_offset
                        _v_store_row_major_fp8(lds_row, vecs[batch])
                else:
                    lds_row = load_row_in_batch + row_offset
                    _v_store_row_major_fp8(lds_row, vecs[batch])

        # ---- Q preload ----
        q_row = q_start + wave_q_offset + lane16
        q_row_i32 = fx.Int32(q_row)
        # Use explicit signed-less-than predicate to match baseline ISA
        # (`v_cmp_gt_i64_e64`). fx.Index defaults to unsigned which would lower
        # to `v_cmp_gt_u64_e64` and cause an ISA hash drift even though both
        # variants are semantically equivalent for non-negative offsets.
        q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, _raw(q_row), _raw(seq_len_v))
        q_row_safe = fx.Index(ArithValue(q_in_bounds).select(q_row, fx.Index(0)))

        c_zero_v2i32_vec = Vec.filled(2, 0, fx.Int32).ir_value()
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Index(ks * K_STEP_QK) + klane * WMMA_LANE_K
            g_idx = global_idx(q_row_safe, q_col)
            # fp8-input: Q already fp8 — load 8 fp8 bytes as v2i32 WMMA-B frag direct.
            raw = _load_global_fp8(q_ptr, g_idx, v2i32_type)
            raw_safe = ArithValue(q_in_bounds).select(raw, c_zero_v2i32_vec)
            q_b_packs.append([_raw(Vec(raw_safe)[0]), _raw(Vec(raw_safe)[1])])

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        # fp8-input: fold per-tensor q_scale*k_scale into the softmax log2e scale
        # (S = q_scale*k_scale * (Q_fp8 . K_fp8^T)). Runtime scalar, computed once.
        c_sm_scale_log2e_rt = _fmul(_fmul(q_scale, k_scale), c_sm_scale_log2e)
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        _q_end = q_start + BLOCK_M
        if const_expr(CAUSAL):
            kv_upper = fx.Index(
                ArithValue(_q_end < seq_len_v).select(_q_end, seq_len_v)
            )
        else:
            kv_upper = seq_len_real_v

        # ---- Pre-issue first V global load before the loop ----
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

            coop_load_k(kv_block_start)
            gpu.barrier()

            # ==== GEMM1: S = K @ Q^T (fp8 WMMA) ====
            # K loaded as v2i32 (ds_read_b64) from lds_k_i32; Q as v2i32 from registers.
            # Single-buffered LDS, so k_base_i32 = 0.
            s_accs = [_raw(c_zero_v8f32) for _ in range(NUM_S_ACCS)]
            k_base_i32 = fx.Index(0)

            for ks in range_constexpr(K_STEPS_QK):
                # k_col_i32 in i32 units: ks*K_STEP_QK fp8 elements / 4 fp8 per i32
                # + klane * WMMA_LANE_K fp8 per lane / 4 = klane * 2
                k_col_i32 = fx.Index(ks * K_STEP_QK // 4) + klane * fx.Index(
                    WMMA_LANE_K // 4
                )

                for st_idx in range_constexpr(N_SUB_TILES):
                    st_base_row = st_idx * K_SUB_N

                    k_row_a = lane16 + fx.Index(st_base_row)
                    k_lds_a_i32 = (
                        k_base_i32 + k_row_a * fx.Index(K_STRIDE_I32) + k_col_i32
                    )
                    k_pack_a_v2i32 = Vec.load(v2i32_type, lds_k_i32, [k_lds_a_i32])

                    k_row_b = lane16 + fx.Index(st_base_row + 16)
                    k_lds_b_i32 = (
                        k_base_i32 + k_row_b * fx.Index(K_STRIDE_I32) + k_col_i32
                    )
                    k_pack_b_v2i32 = Vec.load(v2i32_type, lds_k_i32, [k_lds_b_i32])

                    acc_idx_a = st_idx * 2
                    acc_idx_b = st_idx * 2 + 1
                    s_accs[acc_idx_a] = wmma_acc_fp8(
                        Vec(k_pack_a_v2i32).ir_value(),
                        q_b_packs[ks],
                        s_accs[acc_idx_a],
                    )
                    s_accs[acc_idx_b] = wmma_acc_fp8(
                        Vec(k_pack_b_v2i32).ir_value(),
                        q_b_packs[ks],
                        s_accs[acc_idx_b],
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
                c_sm_scale_log2e_rt,
                c_zero_f,
                num_s_vals=NUM_S_VALS,
                d_chunks=D_CHUNKS,
            )

            # Store V transposed as V_T[d][kv_row] for contiguous GEMM2 loads.
            coop_store_v_lds(_v_vecs_prefetch)
            gpu.barrier()

            # ==== Build P packs (fp8, pair list for wmma_acc_fp8) ====
            def _8p_to_pair_fp8(p8):
                _i32ty = ir.IntegerType.get_signless(32)
                _c0 = arith.constant(0, type=_i32ty)
                pk0 = rocdl.cvt_pk_fp8_f32(_i32ty, p8[0], p8[1], _c0, 0)
                pk0 = rocdl.cvt_pk_fp8_f32(_i32ty, p8[2], p8[3], pk0, 1)
                pk1 = rocdl.cvt_pk_fp8_f32(_i32ty, p8[4], p8[5], _c0, 0)
                pk1 = rocdl.cvt_pk_fp8_f32(_i32ty, p8[6], p8[7], pk1, 1)
                return [pk0, pk1]

            p_packs_all = []
            for st_idx in range_constexpr(N_SUB_TILES):
                p_packs_st = []
                for pks in range_constexpr(PV_K_STEPS):
                    acc_idx = st_idx * 2 + pks
                    p_base = acc_idx * 8
                    p_slice = [p_vals[p_base + j] for j in range(8)]
                    p_packs_st.append(_8p_to_pair_fp8(p_slice))
                p_packs_all.append(p_packs_st)

            # ==== GEMM2: O += V^T @ P (software pipelined, row-major V) ====
            # Prefetch the next V pack while the current WMMA executes.

            def _load_v_rowmajor(st_kv_base_val, pks_val, dc_val):
                # Transposed V: 8 kv bytes are contiguous, so one ds_read_b64 (v2i32)
                # mirrors the GEMM1 K load with no per-byte gather. d_pos selects the d-row.
                d_pos = fx.Index(dc_val * D_CHUNK) + lane16
                v_i32_idx = (
                    fx.Index(V_BYTE_BASE // 4)
                    + d_pos * fx.Index(KV_STRIDE_I32_FP8)
                    + fx.Index((st_kv_base_val + pks_val * PV_K_STEP) // 4)
                    + klane * fx.Index(WMMA_LANE_K // 4)
                )
                v_pack = Vec.load(v2i32_type, lds_v_i32, [v_i32_idx])
                return Vec(v_pack).ir_value()

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
                        o_accs[dc] = wmma_acc_fp8(
                            cur_v_packs[st_idx], p_packs_all[st_idx][pks], o_accs[dc]
                        )

                    if const_expr(has_next):
                        cur_v_packs = next_v_packs

            m_running = m_new_raw
            l_running = l_new

            # ---- Issue the next iteration's V global load ----
            safe_next_kv_start = next_kv_tile_start(
                kv_block_start, kv_upper, fx.Index(BLOCK_N_OUT), fx.Index(0)
            )
            _v_vecs_next = coop_load_v_global(safe_next_kv_start)

            _yield_args = [m_running, l_running] + o_accs
            for batch in range_constexpr(NUM_BATCHES_KV):
                _yield_args.append(_v_vecs_next[batch])
            loop_results = yield _yield_args

        # ---- Normalize and store O ----
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        # fp8-input: O = v_scale * (P . V_fp8); fold v_scale into the 1/l normalizer.
        inv_l = arith.divf(_raw(c_one_f), _raw(l_final), fastmath=fm_fast)
        inv_l = _fmul(inv_l, v_scale)
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
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_real: fx.Int32,
        seq_len_kv: fx.Int32,
        q_scale_ptr: fx.Pointer,
        k_scale_ptr: fx.Pointer,
        v_scale_ptr: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS

        flash_attn_func_kernel._func.__name__ = KERNEL_NAME
        launcher = flash_attn_func_kernel(
            Q,
            K,
            V,
            O,
            seq_len,
            seq_len_real,
            seq_len_kv,
            q_scale_ptr,
            k_scale_ptr,
            v_scale_ptr,
        )

        configure_gpu_module(ctx, waves_per_eu, flat_work_group_size, daz)

        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    launch_flash_attn_func.compile_hints = dict(_fmha_compile_hints)

    def _launch(*args, **kwargs):
        args, kwargs = wrap_pointer_args(
            args,
            kwargs,
            (0, 1, 2, 3, 8, 9, 10),
            ("Q", "K", "V", "O", "q_scale_ptr", "k_scale_ptr", "v_scale_ptr"),
        )
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
        q_scale=None,
        k_scale=None,
        v_scale=None,
        stream=None,
    ):
        # scales are device f32 pointers (1-element tensors), not host floats.
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
            pointer_arg(q_scale),
            pointer_arg(k_scale),
            pointer_arg(v_scale),
            fx.Stream(stream),
        )

    _launch.compile = _compile
    return _launch
