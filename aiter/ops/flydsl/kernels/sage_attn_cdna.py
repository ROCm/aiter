# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL Sage Attention kernel for CDNA (gfx942/gfx950).

Implements Int8 Q/K × FP8 V flash attention using MFMA instructions:
  - GEMM1 (Q×K^T): mfma_i32_32x32x32_i8 → Int32 accum → scale to Float32
  - GEMM2 (P×V):   mfma_f32_32x32x16_bf16 → Float32 accum

Architecture: gfx942 (MI300X) and gfx950 (MI350)
  - wave64 (64 threads/wave)
  - MFMA 32x32x32 for Int8, 32x32x16 for BF16 (gfx950 binding)
  - Block: (BLOCK_M // ROWS_PER_WAVE) waves × 64 threads = total threads

Layout: Q/K/V/O are 1D-flattened from BSHD.
Grid:   (batch * num_q_tiles * num_q_heads,)
Block:  (NUM_WAVES * 64,) -- default 4 waves → 256 threads

Supports:
  - Causal masking
  - GQA / MQA (num_q_heads divisible by num_kv_heads)
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
    vector,
)
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import ArithValue, _to_raw as _raw
from .kernels_common import get_warp_size
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import (
    fly as _fly,
    llvm as _llvm,
    memref as _memref,
    vector as _vector,
)

_LOG2E = host_math.log2(host_math.e)


def _llvm_value(value):
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _extract_aligned_pointer(tensor) -> ir.Value:
    return _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), _llvm_value(tensor))


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return _llvm.LoadOp(result_type, _llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return _llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


def build_sage_attn_cdna_module(
    num_q_heads,
    num_kv_heads,
    head_dim,
    causal=False,
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
    """Build FlyDSL sage attention kernel for CDNA (gfx942/gfx950).

    Uses Int8 MFMA for QK^T and BF16 MFMA for PV, matching triton sage v1 precision.
    """
    gpu_arch = get_hip_arch()
    WARP_SIZE = get_warp_size(gpu_arch)  # 64 for CDNA

    # MFMA 32x32 tile dimensions (Stage A: both QK and PV use 32x32)
    MFMA_M = 32
    MFMA_N = 32

    # For Int8 MFMA: mfma_i32_32x32x32_i8 accumulates K=32 per call
    MFMA_K_INT8 = 32
    # For BF16 MFMA: mfma_f32_32x32x16_bf16 accumulates K=16 per call
    MFMA_K_BF16 = 16

    ROWS_PER_WAVE = MFMA_M  # 32 output rows per wave

    BLOCK_M = block_m if block_m is not None else 256
    BLOCK_N = block_n if block_n is not None else 128

    NUM_WAVES = BLOCK_M // ROWS_PER_WAVE  # e.g. 256//16 = 16
    if flat_work_group_size is None:
        flat_work_group_size = NUM_WAVES * WARP_SIZE
    BLOCK_SIZE = flat_work_group_size

    # K steps for GEMM1 (QK): head_dim // MFMA_K_INT8
    K_STEPS_QK = head_dim // MFMA_K_INT8

    # D chunks for GEMM2 (PV): head_dim / MFMA_N output columns per chunk
    D_CHUNK = MFMA_N  # 16 columns per MFMA output fragment
    D_CHUNKS = head_dim // D_CHUNK

    # K steps for GEMM2 per BLOCK_N tile: BLOCK_N // MFMA_K_BF16
    PV_K_STEPS = BLOCK_N // MFMA_K_BF16

    assert head_dim % MFMA_K_INT8 == 0, f"head_dim {head_dim} must be divisible by {MFMA_K_INT8}"
    assert head_dim % MFMA_K_BF16 == 0, f"head_dim {head_dim} must be divisible by {MFMA_K_BF16}"
    assert BLOCK_M % ROWS_PER_WAVE == 0
    assert BLOCK_N % MFMA_K_BF16 == 0

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_Q_HEADS = num_q_heads
    NUM_KV_HEADS = num_kv_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    GROUPS = num_q_heads // num_kv_heads  # GQA group size

    STRIDE_TOKEN = NUM_Q_HEADS * HEAD_DIM
    KV_STRIDE_TOKEN = NUM_KV_HEADS * HEAD_DIM

    # LDS layout for K (Int8) and V (BF16 after cast)
    # Add padding to reduce bank conflicts
    K_STRIDE = HEAD_DIM + 8    # extra 8 bytes padding for Int8 rows
    V_STRIDE = HEAD_DIM + 4    # extra 4 BF16 elements padding for BF16 rows

    # Cooperative load parameters
    VEC_WIDTH_K = 16   # 16 Int8 elements = 16 bytes per thread
    VEC_WIDTH_V = 8    # 8 BF16 elements = 16 bytes per thread
    THREADS_PER_ROW_K = HEAD_DIM // VEC_WIDTH_K
    THREADS_PER_ROW_V = HEAD_DIM // VEC_WIDTH_V
    ROWS_PER_BATCH_K = BLOCK_SIZE // THREADS_PER_ROW_K
    ROWS_PER_BATCH_V = BLOCK_SIZE // THREADS_PER_ROW_V

    if ROWS_PER_BATCH_K >= BLOCK_N:
        NUM_BATCHES_K = 1
        K_NEEDS_GUARD = ROWS_PER_BATCH_K > BLOCK_N
    else:
        NUM_BATCHES_K = BLOCK_N // ROWS_PER_BATCH_K
        K_NEEDS_GUARD = False

    if ROWS_PER_BATCH_V >= BLOCK_N:
        NUM_BATCHES_V = 1
        V_NEEDS_GUARD = ROWS_PER_BATCH_V > BLOCK_N
    else:
        NUM_BATCHES_V = BLOCK_N // ROWS_PER_BATCH_V
        V_NEEDS_GUARD = False

    LDS_K_TILE = BLOCK_N * K_STRIDE
    LDS_V_TILE = BLOCK_N * V_STRIDE
    LDS_V_BASE = LDS_K_TILE

    # Int8 occupies 1 byte; BF16 occupies 2 bytes.
    # Allocate LDS for K in Int8 and V in BF16 (2 bytes each).
    LDS_K_BYTES = BLOCK_N * K_STRIDE * 1   # 1 byte per Int8
    LDS_V_BYTES = BLOCK_N * V_STRIDE * 2   # 2 bytes per BF16
    LDS_TOTAL_BYTES = LDS_K_BYTES + LDS_V_BYTES

    # We store K as i8 and V as bf16. Because the SmemAllocator works in elements
    # and we share one LDS array, we allocate in Int8 units and use byte offsets.
    # K section: LDS_K_BYTES bytes starting at lds_offset
    # V section: LDS_V_BYTES bytes starting at lds_offset + LDS_K_BYTES
    # Total: LDS_TOTAL_BYTES bytes
    LDS_TOTAL_I8_ELEMS = LDS_TOTAL_BYTES  # 1:1 byte mapping for i8 allocator

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"sage_attn_cdna_smem_M{BLOCK_M}N{BLOCK_N}_{path_tag}",
    )
    lds_base_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_base_offset + LDS_TOTAL_I8_ELEMS * 2  # *2 for safety

    # Architecture-specific FP8 type (gfx942: fnuz, gfx950: fn)
    # Deferred: MLIR types must be constructed inside an active MLIR Context,
    # which only exists inside the @flyc.kernel scope.
    _is_gfx950 = "gfx950" in gpu_arch
    _fp8_mlir_type_cls = ir.Float8E4M3FNType if _is_gfx950 else ir.Float8E4M3FNUZType

    # The 32x32 MFMA variants don't have a flydsl.expr.rocdl wrapper, so we
    # call the raw ODS class directly with positional arguments.
    from flydsl._mlir.dialects._rocdl_ops_gen import (
        mfma_i32_32x32x32_i8 as _ods_mfma_i32_32x32x32_i8,
        mfma_f32_32x32x16_bf16 as _ods_mfma_f32_32x32x16_bf16,
    )

    def mfma_i32_k32(result_type, operands):
        a, b, c = operands[0], operands[1], operands[2]
        cbsz = operands[3] if len(operands) > 3 else 0
        abid = operands[4] if len(operands) > 4 else 0
        blgp = operands[5] if len(operands) > 5 else 0
        a_v = a.ir_value() if hasattr(a, "ir_value") and not isinstance(a, ir.Value) else a
        b_v = b.ir_value() if hasattr(b, "ir_value") and not isinstance(b, ir.Value) else b
        c_v = c.ir_value() if hasattr(c, "ir_value") and not isinstance(c, ir.Value) else c
        return _ods_mfma_i32_32x32x32_i8(
            res=result_type, a=a_v, b=b_v, c=c_v, cbsz=cbsz, abid=abid, blgp=blgp,
        ).result

    def mfma_bf16_k16(result_type, operands):
        a, b, c = operands[0], operands[1], operands[2]
        cbsz = operands[3] if len(operands) > 3 else 0
        abid = operands[4] if len(operands) > 4 else 0
        blgp = operands[5] if len(operands) > 5 else 0
        a_v = a.ir_value() if hasattr(a, "ir_value") and not isinstance(a, ir.Value) else a
        b_v = b.ir_value() if hasattr(b, "ir_value") and not isinstance(b, ir.Value) else b
        c_v = c.ir_value() if hasattr(c, "ir_value") and not isinstance(c, ir.Value) else c
        return _ods_mfma_f32_32x32x16_bf16(
            res=result_type, a=a_v, b=b_v, c=c_v, cbsz=cbsz, abid=abid, blgp=blgp,
        ).result

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def sage_attn_kernel(
        Q: fx.Tensor,          # Int8, 1D flattened BSHD
        K: fx.Tensor,          # Int8, 1D flattened BSHD (num_kv_heads)
        V: fx.Tensor,          # FP8, 1D flattened BSHD (num_kv_heads)
        O: fx.Tensor,          # BF16 output, 1D flattened BSHD
        Q_descale: fx.Tensor,  # f32, shape [batch, num_q_heads, num_q_blocks]
        K_descale: fx.Tensor,  # f32, shape [batch, num_kv_heads, num_k_blocks]
        V_descale: fx.Tensor,  # f32, shape [batch, num_kv_heads, head_dim] (per-element)
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        num_q_blocks: fx.Int32,
    ):
        fm_fast = arith.FastMathFlags.fast

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        def _sitofp(v):
            return arith.SIToFPOp(T.f32, _raw(v)).result

        q_ptr = _extract_aligned_pointer(Q)
        k_ptr = _extract_aligned_pointer(K)
        v_ptr = _extract_aligned_pointer(V)
        o_ptr = _extract_aligned_pointer(O)
        qds_ptr = _extract_aligned_pointer(Q_descale)
        kds_ptr = _extract_aligned_pointer(K_descale)
        vds_ptr = _extract_aligned_pointer(V_descale)

        v4i32_type = Vec.make_type(4, fx.Int32)
        v4f32_type = Vec.make_type(4, fx.Float32)
        v8f32_type = Vec.make_type(8, fx.Float32)
        v16i32_type = Vec.make_type(16, fx.Int32)
        v16f32_type = Vec.make_type(16, fx.Float32)
        # MFMA 32x32x32_i8: A/B = v4i32 (=16xi8 per lane), C/D = v16i32 per lane (wave64)
        v8i8_type = Vec.make_type(8, fx.Int8)
        # MFMA 32x32x16_bf16: A/B = v8bf16, C/D = v16f32 per lane (wave64)
        v4bf16_type = Vec.make_type(4, fx.BFloat16)
        v16i8_type = Vec.make_type(16, fx.Int8)
        v8bf16_type = Vec.make_type(8, fx.BFloat16)
        v4i16_type = Vec.make_type(4, fx.Int16)
        v8i16_type = Vec.make_type(8, fx.Int16)

        seq_len_q_v = fx.Index(seq_len_q)
        seq_len_k_v = fx.Index(seq_len_k)

        base_ptr = allocator.get_base()
        # Shared LDS as Int8 view (K stored as i8, V as bf16 in upper half)
        lds = SmemPtr(
            base_ptr,
            lds_base_offset,
            T.i8,
            shape=(LDS_TOTAL_I8_ELEMS,),
        ).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE

        # MFMA 32x32 wave64 layout:
        # lane32 = lane % 32  (row index in A operand, col index in B/output)
        # klane  = lane // 32 (K-block selector, 0 or 1 for wave64)
        lane32 = lane % MFMA_M
        klane = lane // MFMA_M  # 0 or 1 for wave64

        wave_q_offset = wave_id * ROWS_PER_WAVE

        head_q_idx = block_id % NUM_Q_HEADS
        batch_q_tile_id = block_id // NUM_Q_HEADS
        num_q_tiles = (seq_len_q_v + BLOCK_M - 1) // BLOCK_M
        q_tile_idx = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * BLOCK_M

        # KV head for this Q head (GQA)
        head_kv_idx = head_q_idx // GROUPS

        load_row_k_batch = tid // THREADS_PER_ROW_K
        load_lane_k = tid % THREADS_PER_ROW_K
        load_col_k_base = load_lane_k * VEC_WIDTH_K

        load_row_v_batch = tid // THREADS_PER_ROW_V
        load_lane_v = tid % THREADS_PER_ROW_V
        load_col_v_base = load_lane_v * VEC_WIDTH_V

        def q_global_idx(token_idx, col):
            token = batch_idx * seq_len_q_v + token_idx
            return token * STRIDE_TOKEN + head_q_idx * HEAD_DIM + col

        def kv_global_idx(token_idx, col):
            token = batch_idx * seq_len_k_v + token_idx
            return token * KV_STRIDE_TOKEN + head_kv_idx * HEAD_DIM + col

        def _load_ptr_i8_vec16(ptr, base_idx):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.i8)
            return _pointer_load(v16i8_type, gep)

        def _load_ptr_i8_vec8(ptr, base_idx):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.i8)
            return _pointer_load(v8i8_type, gep)

        def _load_ptr_i64(ptr, base_idx):
            """Load 8 contiguous bytes from `ptr+base_idx` as a scalar i64
            (8xi8 packed). Used for MFMA i8 operands which require packed i64.
            """
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.i8)
            return _pointer_load(T.i64, gep)

        def _load_ptr_f8_vec8(ptr, base_idx):
            # FP8 is loaded as raw bytes and bitcast; the actual ExtFOp uses the
            # arch-specific FP8 type (FNUZ on gfx942, FN on gfx950).
            return _load_ptr_i8_vec8(ptr, base_idx)

        def _store_ptr_bf16(ptr, base_idx, val):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.bf16)
            _pointer_store(val, gep)

        def _load_ptr_f32(ptr, base_idx):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.f32)
            return _pointer_load(T.f32, gep)

        # ---- Preload Q to registers (Int8, v4i32 = 16xi8 packs for MFMA) ----
        # Each wave owns ROWS_PER_WAVE=32 Q rows.
        # For mfma_i32_32x32x32_i8: A operand is vector<4xi32> = 16 i8 per lane.
        # lane32 = q_row within the 32-row tile (0..31)
        # klane (0,1) selects the 16-byte half of the 32-wide K dimension.
        # We preload K_STEPS_QK packs per lane (each = 16 i8 = v4i32).
        q_row = q_start + wave_q_offset + lane32
        q_row_i32 = fx.Int32(q_row)
        q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, _raw(q_row), _raw(seq_len_q_v))
        q_row_safe = fx.Index(ArithValue(q_in_bounds).select(q_row, fx.Index(0)))

        _zero_v4i32_ir = Vec.filled(4, 0, fx.Int32).ir_value()
        q_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            # klane in {0,1} picks the 16-byte K sub-pack (K=k_block*16 + 0..15).
            q_col = fx.Index(ks * MFMA_K_INT8) + klane * 16
            g_idx = q_global_idx(q_row_safe, q_col)
            v16i8 = _load_ptr_i8_vec16(q_ptr, g_idx)
            v4i32 = vector.bitcast(v4i32_type, v16i8)
            q_packs.append(ArithValue(q_in_bounds).select(v4i32, _zero_v4i32_ir))

        # ---- Descale factors ----
        # q_descale: [batch, num_q_heads, num_q_blocks], index = [batch, head_q, q_tile]
        # k_descale: [batch, num_kv_heads, num_k_blocks]
        # v_descale: [batch, num_kv_heads] (per head, broadcast over k-blocks)
        num_k_blocks_per_head = (seq_len_k_v + BLOCK_N - 1) // BLOCK_N

        q_descale_base = (
            batch_idx * NUM_Q_HEADS * fx.Index(num_q_blocks)
            + head_q_idx * fx.Index(num_q_blocks)
            + q_tile_idx
        )
        q_ds = fx.Float32(_load_ptr_f32(qds_ptr, q_descale_base))

        v_descale_base = (batch_idx * NUM_KV_HEADS + head_kv_idx) * HEAD_DIM

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        c_zero_v16i32 = Vec.filled(16, 0, fx.Int32)

        # Warp shuffle for reduction: wave64 has 64 lanes; with 32x32 MFMA the
        # only cross-lane partner of a (klane, lane32) is the lane at the same
        # lane32 with the other klane (XOR distance = 32).
        shuf32_i32 = fx.Int32(32)
        width_i32 = fx.Int32(WARP_SIZE)

        def reduction_peer32(v_f32):
            return fx.Float32(v_f32).shuffle_xor(shuf32_i32, width_i32)

        def row_max_reduce(local_max):
            """Reduce max across the 2 klane halves within wave64."""
            return _fmax(local_max, reduction_peer32(local_max))

        def row_sum_reduce(local_sum):
            """Reduce sum across the 2 klane halves within wave64."""
            return _fadd(local_sum, reduction_peer32(local_sum))

        _q_end = q_start + BLOCK_M
        if const_expr(CAUSAL):
            kv_upper = fx.Index(ArithValue(_q_end < seq_len_k_v).select(_q_end, seq_len_k_v))
        else:
            kv_upper = seq_len_k_v

        # ---- Cooperative load helpers for K (Int8) ----
        def coop_load_k(tile_start):
            for batch in range_constexpr(NUM_BATCHES_K):
                row_offset = batch * ROWS_PER_BATCH_K
                row_idx_raw = tile_start + load_row_k_batch + row_offset
                # Clamp OOB rows to safe row 0 (we mask the resulting scores
                # to -inf before softmax, so the K values are discarded).
                kv_in_bounds = ArithValue(row_idx_raw < seq_len_k_v)
                row_idx = fx.Index(kv_in_bounds.select(row_idx_raw, fx.Index(0)))
                if const_expr(K_NEEDS_GUARD):
                    row_valid = load_row_k_batch < fx.Index(BLOCK_N)
                    do_load = row_valid
                else:
                    do_load = True
                if do_load:
                    g_idx = kv_global_idx(row_idx, load_col_k_base)
                    lds_row = load_row_k_batch + row_offset
                    lds_idx = lds_row * K_STRIDE + load_col_k_base
                    vec = _load_ptr_i8_vec16(k_ptr, g_idx)
                    Vec(vec).store(lds, [lds_idx])

        # V is stored in the upper half of LDS as BF16.
        # LDS is i8 view; V BF16 elements start at byte LDS_K_BYTES.
        # We use a bf16 view offset: each BF16 is 2 bytes → byte_offset // 2 for bf16 index.
        # But we work entirely in byte offsets for i8 LDS array.
        LDS_V_BYTE_BASE = LDS_K_BYTES  # byte offset of V section

        _fp8_mlir_type = _fp8_mlir_type_cls.get()

        # Packed FP8→F32 conversion: rocdl.cvt.pk.f32.fp8 unpacks 2 of the 4
        # FP8 bytes in an i32 src into vector<2xf32> per call (wordSel False/True).
        # 8 FP8 bytes → 2 i32 words → 4 packed cvt ops → 8 f32. This is ~50 %
        # fewer cvt instructions than the per-byte variant.
        _cvt_pk_f32_fp8 = rocdl.CvtPkF32Fp8Op
        v2f32_type = Vec.make_type(2, fx.Float32)
        v2i32_type = Vec.make_type(2, fx.Int32)

        def coop_load_v(tile_start):
            """Load V from global (FP8) → convert to BF16 → single vec16 LDS store.

            Per cooperative-load batch:
              1. v8i8 global load (8 FP8 bytes per thread)
              2. Reinterpret as v2i32, run 4 packed CvtPkF32Fp8 ops → 8 f32
              3. Truncate to v8bf16, bitcast to v16i8
              4. One 16-byte LDS store (vs 16 single-byte stores previously)

            For rows beyond seq_len_k we clamp the load address to row 0 and
            zero the resulting v8bf16 to prevent 0 * NaN = NaN in the PV MFMA.
            """
            zero_v8bf16 = Vec.from_elements(
                [arith.constant(0.0, type=T.bf16) for _ in range_constexpr(8)],
                fx.BFloat16,
            ).ir_value()
            for batch in range_constexpr(NUM_BATCHES_V):
                row_offset = batch * ROWS_PER_BATCH_V
                row_idx_raw = tile_start + load_row_v_batch + row_offset
                kv_in_bounds = ArithValue(row_idx_raw < seq_len_k_v)
                row_idx = fx.Index(kv_in_bounds.select(row_idx_raw, fx.Index(0)))
                if const_expr(V_NEEDS_GUARD):
                    row_valid = load_row_v_batch < fx.Index(BLOCK_N)
                    do_load = row_valid
                else:
                    do_load = True
                if do_load:
                    g_idx = kv_global_idx(row_idx, load_col_v_base)
                    raw_v = _load_ptr_f8_vec8(v_ptr, g_idx)  # v8i8

                    # Reinterpret 8 FP8 bytes as 2 packed i32 words.
                    pair_i32 = vector.bitcast(v2i32_type, raw_v)
                    bf16_elems = []
                    for word in range_constexpr(2):
                        w_i32 = vector.extract(
                            pair_i32, static_position=[word], dynamic_position=[]
                        )
                        # Lo pair (FP8 bytes 0,1) and hi pair (bytes 2,3).
                        pk_lo = _cvt_pk_f32_fp8(v2f32_type, w_i32, False).res
                        pk_hi = _cvt_pk_f32_fp8(v2f32_type, w_i32, True).res
                        for half, pk in ((0, pk_lo), (1, pk_hi)):
                            for elem in range_constexpr(2):
                                f32_e = vector.extract(
                                    pk, static_position=[elem], dynamic_position=[]
                                )
                                bf16_elems.append(arith.TruncFOp(T.bf16, f32_e).result)

                    v8bf16 = Vec.from_elements(bf16_elems, fx.BFloat16).ir_value()
                    # Force the whole vector to zero for OOB rows.
                    v8bf16 = ArithValue(kv_in_bounds).select(v8bf16, zero_v8bf16)
                    # Bitcast to v16i8 so we can store into the i8-typed LDS view.
                    v16i8_val = vector.bitcast(v16i8_type, v8bf16)

                    lds_row = load_row_v_batch + row_offset
                    base_off = (
                        LDS_V_BYTE_BASE
                        + (lds_row * V_STRIDE + load_col_v_base) * 2
                    )
                    Vec(v16i8_val).store(lds, [base_off])

        def load_k_frag(kv_block_row, ks):
            """Load v4i32 (=16xi8) K fragment from LDS for mfma_i32_32x32x32_i8.

            kv_block_row: row within BLOCK_N tile (0..BLOCK_N-1)
            ks: K-step index (selects 32-wide K chunk; klane picks the 16-byte half)
            """
            k_col = fx.Index(ks * MFMA_K_INT8) + klane * 16
            lds_idx = fx.Index(kv_block_row * K_STRIDE) + k_col
            v16i8 = Vec.load(v16i8_type, lds, [lds_idx])
            return vector.bitcast(v4i32_type, v16i8)

        v2i8_type = Vec.make_type(2, fx.Int8)

        def load_v_frag_bf16(pks, dc):
            """Load v8bf16 V fragment from LDS for mfma_f32_32x32x16_bf16.

            For lane (klane, lane32) at PV step `pks` and D chunk `dc`, A operand
            needs V[kv_row, d] for d=dc*32+lane32 (fixed) and 8 contiguous
            kv_row = pks*16 + klane*8 + 0..7 (strided V_STRIDE in LDS).

            Returns a v8bf16 register (bitcast to v8i16 by caller for the MFMA).
            """
            d_col = dc * MFMA_M + lane32
            kv_row_base = fx.Index(pks * MFMA_K_BF16) + klane * 8
            elems = []
            for kk in range_constexpr(8):
                row = kv_row_base + kk
                byte_off = LDS_V_BYTE_BASE + (row * V_STRIDE + d_col) * 2
                v2i8_val = Vec.load(v2i8_type, lds, [byte_off])
                v1bf16 = vector.bitcast(Vec.make_type(1, fx.BFloat16), v2i8_val)
                elems.append(vector.extract(v1bf16, static_position=[0], dynamic_position=[]))
            return Vec.from_elements(elems, fx.BFloat16).ir_value()

        def f32_to_bf16_trunc(f32_raw):
            """Bitwise f32 → bf16 truncation (upper 16 bits)."""
            i32_val = arith.BitcastOp(T.i32, f32_raw).result
            i16_val = arith.TruncIOp(
                T.i16,
                arith.ShRUIOp(i32_val, arith.constant(16, type=T.i32)).result,
            ).result
            return arith.BitcastOp(T.bf16, i16_val).result

        # ---- Main loop: iterate over KV tiles ----
        init_args = [_raw(c_neg_inf), _raw(c_zero_f)]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(_raw(c_zero_v16f32))

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(0, kv_upper, BLOCK_N, init=init_args):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]

            # Load K tile (Int8) to LDS
            coop_load_k(kv_block_start)
            gpu.barrier()

            # ==== GEMM1: S_i32 = K_i8 @ Q_i8^T ====
            # s_accs[subtile]: v16i32 per lane (MFMA 32x32 output, wave64).
            # Each wave owns ROWS_PER_WAVE=32 Q rows. MFMA computes 32 Q rows × 32 KV rows.
            # With BLOCK_N=128, we need BLOCK_N/MFMA_N = 4 row-subtiles.
            N_SUBTILES = BLOCK_N // MFMA_N

            s_accs = [_raw(c_zero_v16i32) for _ in range(N_SUBTILES)]

            for ks in range_constexpr(K_STEPS_QK):
                for st in range_constexpr(N_SUBTILES):
                    # K operand A: lane has 16 i8 at row m=lane32+st*32, K=ks*32+klane*16..+15
                    kv_row = lane32 + st * MFMA_N
                    k_frag = load_k_frag(kv_row, ks)
                    s_accs[st] = mfma_i32_k32(
                        v16i32_type, [k_frag, q_packs[ks], s_accs[st], 0, 0, 0]
                    )

            # Scale Int32 → Float32: S_f32 = S_i32 * q_descale[q_tile] * k_descale[kv_tile]
            kv_tile_idx = kv_block_start // BLOCK_N
            k_descale_base = (
                batch_idx * NUM_KV_HEADS * num_k_blocks_per_head
                + head_kv_idx * num_k_blocks_per_head
                + kv_tile_idx
            )
            k_ds = fx.Float32(_load_ptr_f32(kds_ptr, k_descale_base))
            qk_scale = _fmul(q_ds, k_ds)

            # 32x32 MFMA output layout: lane (klane, lane32) holds 16 elements
            # per subtile. Output indexing for vector index i ∈ 0..15:
            #   m = (i // 4) * 8 + klane * 4 + (i % 4)   (kv_row within subtile)
            #   n = lane32                                (q_row within wave; constant per lane)
            # Note: A=K (M=kv_row), B=Q (N=q_row). So m here is kv_row.
            ELEMS_PER_TILE = 16
            s_f32 = []
            for st in range_constexpr(N_SUBTILES):
                for elem in range_constexpr(ELEMS_PER_TILE):
                    i32_elem = Vec(s_accs[st])[elem]
                    f32_elem = _sitofp(i32_elem)
                    s_f32.append(_fmul(f32_elem, qk_scale))

            # ==== Causal masking ====
            NUM_S_VALS = N_SUBTILES * ELEMS_PER_TILE

            # Per-element masking: out-of-range KV columns (kv_col >= seq_len_k)
            # and (if causal) kv_col > q_row get -inf.
            # For each (st, elem): kv_col = kv_block_start + st*32 + (elem//4)*8 + klane*4 + (elem%4)
            kv_start_i32 = fx.Int32(kv_block_start)
            klane_i32 = fx.Int32(klane)
            klane_off_i32 = klane_i32 * fx.Int32(4)
            seq_len_k_i32 = fx.Int32(seq_len_k_v)
            s_named = list(s_f32)
            for st in range_constexpr(N_SUBTILES):
                for elem in range_constexpr(ELEMS_PER_TILE):
                    idx = st * ELEMS_PER_TILE + elem
                    msub = elem // 4
                    erem = elem % 4
                    kv_col_i32 = (
                        kv_start_i32
                        + fx.Int32(st * MFMA_N)
                        + fx.Int32(msub * 8)
                        + klane_off_i32
                        + fx.Int32(erem)
                    )
                    out_of_range = ArithValue(kv_col_i32 >= seq_len_k_i32)
                    if const_expr(CAUSAL):
                        out_of_range = out_of_range | ArithValue(kv_col_i32 > q_row_i32)
                    s_named[idx] = out_of_range.select(c_neg_inf, s_named[idx])
            s_f32 = s_named

            # ==== Online softmax ====
            local_max = s_f32[0]
            for r in range_constexpr(NUM_S_VALS - 1):
                local_max = _fmax(local_max, s_f32[r + 1])
            row_max = row_max_reduce(local_max)
            m_new = _fmax(m_running, row_max)

            # Correction factor for previous accumulator.
            # NOTE: s_f32 already contains (Q @ K.T) * sm_scale * log2(e) because
            # sage_quant absorbs sm_scale_log2e into q_scale. Do NOT multiply by
            # sm_scale_log2e again here.
            diff_m = _fsub(m_running, m_new)
            corr = rocdl.exp2(ir.F32Type.get(), _raw(diff_m))

            # Compute P values directly: p = exp2(s_f32 - m_new)
            neg_max = _fsub(c_zero_f, m_new)
            p_vals = []
            local_sum = _raw(c_zero_f)
            for r in range_constexpr(NUM_S_VALS):
                diff = _fadd(s_f32[r], neg_max)
                p = rocdl.exp2(ir.F32Type.get(), _raw(diff))
                p_vals.append(p)
                local_sum = _fadd(local_sum, p)

            tile_sum = row_sum_reduce(local_sum)
            l_new = _fadd(_fmul(corr, l_running), tile_sum)

            # Rescale O accumulators (each is v16f32 per lane).
            corr_vec16 = Vec.from_elements([corr], fx.Float32).broadcast_to(16).ir_value()
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = _fmul(o_accs[dc], corr_vec16)

            # Load V tile (FP8 → BF16) to LDS
            coop_load_v(kv_block_start)
            gpu.barrier()

            # ==== GEMM2: O += V @ P  (mfma_f32_32x32x16_bf16) ====
            # A=V (v8bf16/lane: row m=lane32 of V_descaled[d, kv]; K=k_block*8+0..7
            #     in kv dim). Loaded by load_v_frag_bf16(pks, dc).
            # B=P (v8bf16/lane: column n=lane32 of P[kv, q]; K=k_block*8+0..7).
            # Output: v16f32/lane covering 32 d positions × 32 q rows; same layout
            # as QK output (m=(i//4)*8 + klane*4 + (i%4) for d index, n=lane32 for
            # q index).
            #
            # P-handoff bridge: QK output gives lane (klane, lane32) the P values
            # at q_row=lane32 and kv_rows m=(i//4)*8 + klane*4 + (i%4) within
            # subtile st. The PV B operand needs kv_rows = pks*16 + klane*8 + 0..7
            # contiguous in K. For each pks, half of the 8 needed values live in
            # this lane's QK accumulator (i indices base..base+3 for own klane)
            # and the other half live in the partner lane (same lane32, other
            # klane) at the SAME i indices. shuffle_xor(width=64) by 32 swaps
            # them.
            #
            # base index: pks_in_sub = pks % 2, base = klane*4 + pks_in_sub*8
            #   (klane=0 own holds m={0..3 lo, 8..11 lo, 16..19 lo, 24..27 lo} at
            #    i=0..3 / 4..7 / 8..11 / 12..15 of the v16 in subtile st;
            #    so base is the i-offset where the 4 needed own elements start.)
            # K layout in B operand:
            #   klane=0 needs K=0..7  → [own[base..+3], partner[base..+3]]
            #   klane=1 needs K=8..15 → [partner[base..+3], own[base..+3]]
            klane_is_zero = ArithValue(klane == fx.Index(0))
            for pks in range_constexpr(PV_K_STEPS):
                subtile_idx = pks // 2  # Python int (range_constexpr yields ints)
                pks_in_sub = pks % 2
                # For each j in 0..3, our_keep[j] and to_partner[j] are
                # selected from p_vals (extracted at constexpr indices) by
                # klane. Then to_partner is sent to the partner via
                # shuffle_xor(32); we receive from_partner.
                our_keep = []
                from_partner = []
                for j in range_constexpr(4):
                    flat_lo = subtile_idx * ELEMS_PER_TILE + pks_in_sub * 8 + j
                    flat_hi = subtile_idx * ELEMS_PER_TILE + pks_in_sub * 8 + 4 + j
                    p_lo = p_vals[flat_lo]
                    p_hi = p_vals[flat_hi]
                    our = klane_is_zero.select(p_lo, p_hi)
                    to_par = klane_is_zero.select(p_hi, p_lo)
                    par = _raw(reduction_peer32(to_par))
                    our_keep.append(_raw(our))
                    from_partner.append(par)
                # Build v8bf16: klane=0 uses [our, partner] for K=0..7;
                # klane=1 uses [partner, our] for K=8..15.
                p_bf16_elems = []
                for j in range_constexpr(4):
                    own_e = f32_to_bf16_trunc(our_keep[j])
                    par_e = f32_to_bf16_trunc(from_partner[j])
                    p_bf16_elems.append(klane_is_zero.select(own_e, par_e))
                for j in range_constexpr(4):
                    own_e = f32_to_bf16_trunc(our_keep[j])
                    par_e = f32_to_bf16_trunc(from_partner[j])
                    p_bf16_elems.append(klane_is_zero.select(par_e, own_e))
                p_pack_v8bf16 = Vec.from_elements(p_bf16_elems, fx.BFloat16).ir_value()

                for dc in range_constexpr(D_CHUNKS):
                    v_frag = load_v_frag_bf16(pks, dc)
                    o_accs[dc] = mfma_bf16_k16(
                        v16f32_type, [v_frag, p_pack_v8bf16, o_accs[dc], 0, 0, 0]
                    )

            m_running = m_new
            l_running = l_new

            _yield_args = [m_running, l_running] + o_accs
            loop_results = yield _yield_args

        # ---- Normalize and store O ----
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        inv_l = arith.divf(_raw(c_one_f), _raw(l_final), fastmath=fm_fast)

        if q_in_bounds:
            # MFMA 32x32 C/D layout (wave64): lane (klane, lane32) holds 16 f32
            # outputs covering 16 distinct M positions (d positions in PV output)
            # at one fixed N position = lane32 (= q_row within wave).
            # Vector index i ∈ 0..15:
            #   d_offset = (i // 4) * 8 + klane * 4 + (i % 4)
            # Each group of 4 consecutive i (i.e. msub=0..3) gives 4 contiguous
            # d positions, so we can do 4 v4bf16 stores per dc instead of 16
            # scalar stores.
            for dc in range_constexpr(D_CHUNKS):
                for msub in range_constexpr(4):
                    d_col_base = fx.Index(dc * MFMA_M) + fx.Index(msub * 8) + klane * 4
                    bf16_elems = []
                    for erem in range_constexpr(4):
                        i_pos = msub * 4 + erem
                        v_ds = fx.Float32(_load_ptr_f32(
                            vds_ptr,
                            v_descale_base
                            + fx.Index(dc * MFMA_M)
                            + fx.Index(msub * 8)
                            + klane * 4
                            + erem,
                        ))
                        o_elem = vector.extract(
                            o_finals[dc], static_position=[i_pos], dynamic_position=[]
                        )
                        scale = arith.mulf(_raw(inv_l), _raw(v_ds), fastmath=fm_fast)
                        o_norm = arith.mulf(o_elem, scale, fastmath=fm_fast)
                        bf16_elems.append(f32_to_bf16_trunc(o_norm))
                    o_vec = Vec.from_elements(bf16_elems, fx.BFloat16).ir_value()
                    o_global = q_global_idx(q_row, d_col_base)
                    _store_ptr_bf16(o_ptr, o_global, o_vec)

    @flyc.jit
    def launch_sage_attn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        Q_descale: fx.Tensor,
        K_descale: fx.Tensor,
        V_descale: fx.Tensor,
        batch_size: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        num_q_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = fx.Index(batch_size)
        slq_idx = fx.Index(seq_len_q)
        num_q_tiles = (slq_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_Q_HEADS

        launcher = sage_attn_kernel(
            Q, K, V, O, Q_descale, K_descale, V_descale,
            seq_len_q, seq_len_k, num_q_blocks,
        )

        if const_expr(waves_per_eu is not None):
            _wpe = int(waves_per_eu)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, _wpe)

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
                ir.ArrayAttr.get([
                    ir.StringAttr.get("denormal-fp-math-f32"),
                    ir.StringAttr.get("preserve-sign,preserve-sign"),
                ])
            )
            passthrough_entries.append(
                ir.ArrayAttr.get([
                    ir.StringAttr.get("no-nans-fp-math"),
                    ir.StringAttr.get("true"),
                ])
            )
            passthrough_entries.append(
                ir.ArrayAttr.get([
                    ir.StringAttr.get("unsafe-fp-math"),
                    ir.StringAttr.get("true"),
                ])
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    _compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_compile_hints):
            return launch_sage_attn(*args, **kwargs)

    def _compile(Q, K, V, O, Q_descale, K_descale, V_descale,
                 batch_size, seq_len_q, seq_len_k, num_q_blocks, stream=None):
        with CompilationContext.compile_hints(_compile_hints):
            return flyc.compile(
                launch_sage_attn,
                Q, K, V, O, Q_descale, K_descale, V_descale,
                batch_size, seq_len_q, seq_len_k, num_q_blocks,
                fx.Stream(stream),
            )

    _launch.compile = _compile
    return _launch
