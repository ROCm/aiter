# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL Sage Attention kernel for CDNA (gfx942/gfx950).

Implements Int8 Q/K × FP8 V flash attention using MFMA instructions:
  - GEMM1 (Q×K^T): mfma_i32_16x16x32i8 → Int32 accum → scale to Float32
  - GEMM2 (P×V):   mfma_f32_16x16x16bf16_1k → Float32 accum

Architecture: gfx942 (MI300X) and gfx950 (MI350)
  - wave64 (64 threads/wave)
  - MFMA 16x16x32 for Int8, 16x16x16 for BF16
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

    # MFMA 16x16 tile dimensions
    MFMA_M = 16
    MFMA_N = 16

    # For Int8 MFMA: mfma_i32_16x16x32i8 accumulates K=32 per call
    MFMA_K_INT8 = 32
    # For BF16 MFMA: mfma_f32_16x16x16bf16_1k accumulates K=16 per call
    MFMA_K_BF16 = 16

    ROWS_PER_WAVE = MFMA_M  # 16 output rows per wave

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
    _is_gfx950 = "gfx950" in gpu_arch
    _fp8_mlir_type = T.f8e4m3fn if _is_gfx950 else T.f8e4m3fnuz

    mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
        rocdl, "mfma_i32_16x16x32_i8", None
    )
    assert mfma_i32_k32 is not None, "INT8 K32 MFMA not found"

    mfma_bf16_k16 = getattr(rocdl, "mfma_f32_16x16x16bf16_1k", None) or getattr(
        rocdl, "mfma_f32_16x16x16_bf16_1k", None
    )
    assert mfma_bf16_k16 is not None, "BF16 K16 MFMA not found"

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
        # MFMA 16x16x32_i8: A/B = v8i8, C/D = v4i32 per lane (wave64)
        v8i8_type = Vec.make_type(8, fx.Int8)
        # MFMA 16x16x16_bf16_1k: A/B = v4bf16, C/D = v4f32 per lane (wave64)
        v4bf16_type = Vec.make_type(4, fx.BFloat16)
        v16i8_type = Vec.make_type(16, fx.Int8)
        v8bf16_type = Vec.make_type(8, fx.BFloat16)

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

        # MFMA 16x16 wave64 layout:
        # lane[0:3]  = row within 16 (lane % 16)
        # lane[4]    = k-group (lane // 16, gives 0 or 1 for k-step within MFMA_K)
        lane16 = lane % MFMA_M
        klane = lane // MFMA_M  # 0..3 for wave64

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
            return _pointer_load(Vec.make_type(16, fx.Int8).ir_type, gep)

        def _load_ptr_f8_vec8(ptr, base_idx):
            # FP8 is loaded as raw bytes and bitcast; FlyDSL uses T.f8 or T.fp8e4m3
            # Represent FP8 as i8 for load, then convert to BF16
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.i8)
            return _pointer_load(Vec.make_type(8, fx.Int8).ir_type, gep)

        def _store_ptr_bf16(ptr, base_idx, val):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.bf16)
            _pointer_store(val, gep)

        def _load_ptr_f32(ptr, base_idx):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.f32)
            return _pointer_load(T.f32, gep)

        # ---- Preload Q to registers (Int8, v8i8 packs for MFMA) ----
        # Each wave owns ROWS_PER_WAVE=16 Q rows.
        # For mfma_i32_16x16x32i8: A operand is v8i8 per lane.
        # lane[0:3] = q_row within the 16-row tile
        # Lane k-group selects the 8-byte half of the 32-wide K dimension.
        # We preload K_STEPS_QK * 2 packs per lane (2 halves of K=32 per step).
        q_row = q_start + wave_q_offset + lane16
        q_row_i32 = fx.Int32(q_row)
        q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, _raw(q_row), _raw(seq_len_q_v))
        q_row_safe = fx.Index(ArithValue(q_in_bounds).select(q_row, fx.Index(0)))

        c_zero_v8i8 = Vec.filled(8, 0, fx.Int8).ir_value()
        q_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            # Each MFMA_K_INT8=32 step: klane selects 8-byte sub-pack (0 or 1)
            q_col = fx.Index(ks * MFMA_K_INT8) + klane * 8
            g_idx = q_global_idx(q_row_safe, q_col)
            raw = _load_ptr_i8_vec16(q_ptr, g_idx)
            # Extract the 8-byte half for this klane
            half = Vec(raw).slice(klane * 8, 8).ir_value()
            q_packs.append(ArithValue(q_in_bounds).select(half, c_zero_v8i8))

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
        c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)
        c_zero_v4i32 = Vec.filled(4, 0, fx.Int32)

        # Warp shuffle for reduction: wave64 has 64 lanes; reduce across 4 klane groups
        shuf16_i32 = fx.Int32(16)
        shuf32_i32 = fx.Int32(32)
        width_i32 = fx.Int32(WARP_SIZE)

        def reduction_peer16(v_f32):
            return fx.Float32(v_f32).shuffle_xor(shuf16_i32, width_i32)

        def reduction_peer32(v_f32):
            return fx.Float32(v_f32).shuffle_xor(shuf32_i32, width_i32)

        def row_max_reduce(local_max):
            """Reduce max across all 4 klane groups within wave64."""
            m1 = _fmax(local_max, reduction_peer16(local_max))
            return _fmax(m1, reduction_peer32(m1))

        def row_sum_reduce(local_sum):
            """Reduce sum across all 4 klane groups within wave64."""
            s1 = _fadd(local_sum, reduction_peer16(local_sum))
            return _fadd(s1, reduction_peer32(s1))

        _q_end = q_start + BLOCK_M
        if const_expr(CAUSAL):
            kv_upper = fx.Index(ArithValue(_q_end < seq_len_k_v).select(_q_end, seq_len_k_v))
        else:
            kv_upper = seq_len_k_v

        # ---- Cooperative load helpers for K (Int8) ----
        def coop_load_k(tile_start):
            for batch in range_constexpr(NUM_BATCHES_K):
                row_offset = batch * ROWS_PER_BATCH_K
                row_idx = tile_start + load_row_k_batch + row_offset
                if const_expr(K_NEEDS_GUARD):
                    row_valid = load_row_k_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        g_idx = kv_global_idx(row_idx, load_col_k_base)
                        lds_row = load_row_k_batch + row_offset
                        # Store 16 Int8 elements at byte offset lds_row*K_STRIDE + col
                        lds_idx = fx.Index(lds_row * K_STRIDE + load_col_k_base)
                        vec = _load_ptr_i8_vec16(k_ptr, g_idx)
                        Vec(vec).store(lds, [lds_idx])
                else:
                    g_idx = kv_global_idx(row_idx, load_col_k_base)
                    lds_row = load_row_k_batch + row_offset
                    lds_idx = fx.Index(lds_row * K_STRIDE + load_col_k_base)
                    vec = _load_ptr_i8_vec16(k_ptr, g_idx)
                    Vec(vec).store(lds, [lds_idx])

        # V is stored in the upper half of LDS as BF16.
        # LDS is i8 view; V BF16 elements start at byte LDS_K_BYTES.
        # We use a bf16 view offset: each BF16 is 2 bytes → byte_offset // 2 for bf16 index.
        # But we work entirely in byte offsets for i8 LDS array.
        LDS_V_BYTE_BASE = LDS_K_BYTES  # byte offset of V section

        def _fp8_i8_to_bf16(raw_i8):
            """Convert a raw Int8 FP8 value to BF16 via float32."""
            fp8_val = arith.BitcastOp(_fp8_mlir_type, _raw(raw_i8)).result
            f32_val = arith.ExtFOp(T.f32, fp8_val).result
            return arith.TruncFOp(T.bf16, f32_val).result

        def coop_load_v(tile_start):
            """Load V from global memory (FP8) → convert to BF16 → store to LDS."""
            for batch in range_constexpr(NUM_BATCHES_V):
                row_offset = batch * ROWS_PER_BATCH_V
                row_idx = tile_start + load_row_v_batch + row_offset
                if const_expr(V_NEEDS_GUARD):
                    row_valid = load_row_v_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        g_idx = kv_global_idx(row_idx, load_col_v_base)
                        raw_v = _load_ptr_f8_vec8(v_ptr, g_idx)
                        lds_row = load_row_v_batch + row_offset
                        # BF16 byte offset: V_BYTE_BASE + lds_row * V_STRIDE * 2 + col * 2
                        # Stored as individual bf16 elements in i8 LDS view
                        for elem in range_constexpr(8):
                            raw_elem = Vec(raw_v)[elem]
                            bf16_elem = _fp8_i8_to_bf16(raw_elem)
                            byte_off = (
                                LDS_V_BYTE_BASE
                                + (lds_row * V_STRIDE + load_col_v_base + elem) * 2
                            )
                            # Store via memref.store into i8 LDS (reinterpret as bf16)
                            # Use bitcast at element level via i16
                            bf16_as_i16 = arith.BitcastOp(T.i16, bf16_elem).result
                            bf16_lo = arith.TruncIOp(T.i8, bf16_as_i16).result
                            bf16_hi = arith.TruncIOp(
                                T.i8,
                                arith.ShRUIOp(bf16_as_i16, arith.constant(T.i16, 8)).result,
                            ).result
                            _memref.store(bf16_lo, lds, [_raw(fx.Index(LDS_V_BYTE_BASE + (lds_row * V_STRIDE + load_col_v_base + elem) * 2))])
                            _memref.store(bf16_hi, lds, [_raw(fx.Index(LDS_V_BYTE_BASE + (lds_row * V_STRIDE + load_col_v_base + elem) * 2 + 1))])
                else:
                    g_idx = kv_global_idx(row_idx, load_col_v_base)
                    raw_v = _load_ptr_f8_vec8(v_ptr, g_idx)
                    lds_row = load_row_v_batch + row_offset
                    for elem in range_constexpr(8):
                        raw_elem = Vec(raw_v)[elem]
                        bf16_elem = _fp8_i8_to_bf16(raw_elem)
                        bf16_as_i16 = arith.BitcastOp(T.i16, bf16_elem).result
                        bf16_lo = arith.TruncIOp(T.i8, bf16_as_i16).result
                        bf16_hi = arith.TruncIOp(
                            T.i8,
                            arith.ShRUIOp(bf16_as_i16, arith.constant(T.i16, 8)).result,
                        ).result
                        _memref.store(bf16_lo, lds, [_raw(fx.Index(LDS_V_BYTE_BASE + (lds_row * V_STRIDE + load_col_v_base + elem) * 2))])
                        _memref.store(bf16_hi, lds, [_raw(fx.Index(LDS_V_BYTE_BASE + (lds_row * V_STRIDE + load_col_v_base + elem) * 2 + 1))])

        def load_k_frag(kv_block_row, ks):
            """Load v8i8 K fragment from LDS for mfma_i32_16x16x32i8.

            kv_block_row: row within BLOCK_N tile (0..BLOCK_N-1)
            ks: K-step index (selects 32-wide K chunk)
            """
            k_col = fx.Index(ks * MFMA_K_INT8) + klane * 8
            lds_idx = fx.Index(kv_block_row * K_STRIDE) + k_col
            return Vec.load(v8i8_type, lds, [lds_idx])

        def load_v_frag_bf16(kv_row, d_chunk):
            """Load v4bf16 V fragment from LDS for mfma_f32_16x16x16bf16_1k.

            kv_row: row within BLOCK_N tile
            d_chunk: D output chunk index (selects HEAD_DIM/16 column group)
            """
            d_col = d_chunk * D_CHUNK + lane16
            # Byte offset in LDS for this BF16 element
            byte_off = LDS_V_BYTE_BASE + (kv_row * V_STRIDE + d_col) * 2
            # Load as i16 (2 bytes) and bitcast to bf16
            byte0 = fx.Int8(_memref.load(lds, [_raw(fx.Index(byte_off))]))
            byte1 = fx.Int8(_memref.load(lds, [_raw(fx.Index(byte_off + 1))]))
            b0_i16 = arith.ExtUIOp(T.i16, _raw(byte0)).result
            b1_i16 = arith.ExtUIOp(T.i16, _raw(byte1)).result
            i16_val = arith.OrIOp(b0_i16, arith.ShLIOp(b1_i16, arith.constant(T.i16, 8)).result).result
            return arith.BitcastOp(T.bf16, i16_val).result

        def bf16_trunc_pack_v4(f32_vals):
            """Pack 4 f32 values into v4bf16 via bitwise truncation (upper 16 bits)."""
            results = []
            for j in range_constexpr(4):
                f32_raw = _raw(f32_vals[j])
                i32_val = arith.BitcastOp(T.i32, f32_raw).result
                i16_val = arith.TruncIOp(T.i16, arith.ShRUIOp(i32_val, arith.constant(T.i32, 16)).result).result
                bf16_val = arith.BitcastOp(T.bf16, i16_val).result
                results.append(bf16_val)
            return Vec.from_elements(results, fx.BFloat16).ir_value()

        # ---- Main loop: iterate over KV tiles ----
        init_args = [_raw(c_neg_inf), _raw(c_zero_f)]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(_raw(c_zero_v4f32))

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(0, kv_upper, BLOCK_N, init=init_args):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]

            # Load K tile (Int8) to LDS
            coop_load_k(kv_block_start)
            gpu.barrier()

            # ==== GEMM1: S_i32 = K_i8 @ Q_i8^T ====
            # s_accs[row_within_tile]: v4i32 per lane (MFMA 16x16 output, wave64)
            # Each wave owns ROWS_PER_WAVE=16 Q rows. MFMA computes 16 Q rows × 16 KV rows.
            # With BLOCK_N=128, we need BLOCK_N/MFMA_N=8 row-subtiles.
            # Each subtile produces one v4i32 accumulator per lane.
            N_SUBTILES = BLOCK_N // MFMA_N

            s_accs = [_raw(c_zero_v4i32) for _ in range(N_SUBTILES)]

            for ks in range_constexpr(K_STEPS_QK):
                for st in range_constexpr(N_SUBTILES):
                    kv_row = lane16 + st * MFMA_N
                    k_frag = load_k_frag(kv_row, ks)
                    # mfma_i32_16x16x32i8(res_type, [A_v8i8, B_v8i8, C_v4i32, 0, 0, 0])
                    s_accs[st] = mfma_i32_k32(
                        v4i32_type, [k_frag, q_packs[ks], s_accs[st], 0, 0, 0]
                    ).result

            # Scale Int32 → Float32: S_f32 = S_i32 * q_descale[q_tile] * k_descale[kv_tile]
            kv_tile_idx = kv_block_start // BLOCK_N
            k_descale_base = (
                batch_idx * NUM_KV_HEADS * num_k_blocks_per_head
                + head_kv_idx * num_k_blocks_per_head
                + kv_tile_idx
            )
            k_ds = fx.Float32(_load_ptr_f32(kds_ptr, k_descale_base))
            qk_scale = _fmul(q_ds, k_ds)

            s_f32 = []
            for st in range_constexpr(N_SUBTILES):
                for lane_r in range_constexpr(4):
                    i32_elem = Vec(s_accs[st])[lane_r]
                    f32_elem = _sitofp(i32_elem)
                    s_f32.append(_fmul(f32_elem, qk_scale))

            # ==== Causal masking ====
            NUM_S_VALS = N_SUBTILES * 4

            if const_expr(CAUSAL):
                kv_start_i32 = fx.Int32(kv_block_start)
                klane_i32 = fx.Int32(klane)
                q_start_i32 = fx.Int32(q_start)
                max_kv_col_i32 = kv_start_i32 + fx.Int32(BLOCK_N - 1)
                tile_needs_mask = max_kv_col_i32 > q_start_i32

                # Unfold s_f32 into named scalars for SSA-safe conditional rewriting
                s_named = list(s_f32)
                if tile_needs_mask:
                    klane_off_i32 = klane_i32 * fx.Int32(4)
                    for st in range_constexpr(N_SUBTILES):
                        for lane_r in range_constexpr(4):
                            idx = st * 4 + lane_r
                            kv_col_i32 = (
                                kv_start_i32
                                + fx.Int32(st * MFMA_N)
                                + klane_off_i32
                                + fx.Int32(lane_r)
                            )
                            s_named[idx] = ArithValue(kv_col_i32 > q_row_i32).select(
                                c_neg_inf, s_named[idx]
                            )
                s_f32 = s_named

            # ==== Online softmax ====
            local_max = s_f32[0]
            for r in range_constexpr(NUM_S_VALS - 1):
                local_max = _fmax(local_max, s_f32[r + 1])
            row_max = row_max_reduce(local_max)
            m_new = _fmax(m_running, row_max)

            # Correction factor for previous accumulator
            diff_m = _fsub(m_running, m_new)
            diff_m_scaled = _fmul(diff_m, c_sm_scale_log2e)
            corr = rocdl.exp2(ir.F32Type.get(), _raw(diff_m_scaled))

            # Compute P values (exp2 of scaled scores)
            scaled_max = _fmul(c_sm_scale_log2e, m_new)
            neg_scaled_max = _fsub(c_zero_f, scaled_max)
            p_vals = []
            local_sum = _raw(c_zero_f)
            for r in range_constexpr(NUM_S_VALS):
                diff = fmath.fma(s_f32[r], _raw(c_sm_scale_log2e), neg_scaled_max)
                p = rocdl.exp2(ir.F32Type.get(), _raw(diff))
                p_vals.append(p)
                local_sum = _fadd(local_sum, p)

            tile_sum = row_sum_reduce(local_sum)
            l_new = _fadd(_fmul(corr, l_running), tile_sum)

            # Rescale O accumulators
            corr_vec4 = Vec.from_elements([corr], fx.Float32).broadcast_to(4).ir_value()
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = _fmul(o_accs[dc], corr_vec4)

            # Load V tile (FP8 → BF16) to LDS
            coop_load_v(kv_block_start)
            gpu.barrier()

            # ==== GEMM2: O += V_bf16^T @ P_bf16 ====
            # Build P packs (v4bf16) for each MFMA_K_BF16=16 sub-step
            # P is [N_SUBTILES * 4] f32 values; pack them into v4bf16 for MFMA B operand
            # Each PV_K_STEPS step covers 16 KV rows (MFMA_K_BF16)
            for pks in range_constexpr(PV_K_STEPS):
                # P pack for this 16-row k-step: p_vals[pks*4 : pks*4+4] per lane
                # In wave64 layout: klane selects which 4 of the 16 KV rows this lane owns
                p_base = pks * MFMA_N + klane * 4  # = pks*16 + klane*4
                p_slice = [p_vals[p_base + j] for j in range(4)]
                p_pack_bf16 = bf16_trunc_pack_v4(p_slice)

                for dc in range_constexpr(D_CHUNKS):
                    # Load V fragment: v4bf16 for the (pks, dc) tile
                    v_elems = []
                    for k_sub in range_constexpr(4):
                        kv_row_v = pks * MFMA_K_BF16 + klane * 4 + k_sub
                        v_elem = load_v_frag_bf16(kv_row_v, dc)
                        v_elems.append(v_elem)
                    v_frag = Vec.from_elements(v_elems, fx.BFloat16).ir_value()

                    # mfma_f32_16x16x16bf16_1k(res_type, [A_v4bf16, B_v4bf16, C_v4f32, 0, 0, 0])
                    o_accs[dc] = mfma_bf16_k16(
                        v4f32_type, [v_frag, p_pack_bf16, o_accs[dc], 0, 0, 0]
                    ).result

            m_running = m_new
            l_running = l_new

            _yield_args = [m_running, l_running] + o_accs
            loop_results = yield _yield_args

        # ---- Normalize and store O ----
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        inv_l = arith.divf(_raw(c_one_f), _raw(l_final), fastmath=fm_fast)

        if q_in_bounds:
            for dc in range_constexpr(D_CHUNKS):
                d_col = fx.Index(dc * D_CHUNK) + lane16
                v_ds_dc = fx.Float32(_load_ptr_f32(vds_ptr, v_descale_base + fx.Index(dc * D_CHUNK) + lane16))
                inv_l_vds_dc = _fmul(inv_l, v_ds_dc)
                inv_l_vec4_dc = Vec.from_elements([inv_l_vds_dc], fx.Float32).broadcast_to(4).ir_value()
                o_norm_vec = _fmul(o_finals[dc], inv_l_vec4_dc)
                # Truncate f32 → bf16 and store
                o_bf16 = Vec(o_norm_vec).to(fx.BFloat16).ir_value()
                o_global = q_global_idx(q_row, d_col)
                _store_ptr_bf16(o_ptr, o_global, o_bf16)

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
