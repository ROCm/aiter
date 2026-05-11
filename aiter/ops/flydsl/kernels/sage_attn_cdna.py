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
    scf as _scf,
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
    # For BF16 MFMA: mfma_f32_32x32x16_bf16 accumulates K=16 per call (legacy/unused)
    MFMA_K_BF16 = 16
    # For FP8 K=64 MFMA: mfma_scale_f32_32x32x64_f8f6f4 accumulates K=64 per call.
    MFMA_K_FP8 = 64

    # FP8 (gfx950 e4m3fn) max representable magnitude. Used to scale P into [0, 1]
    # → [0, FP8_MAX] before fp8 cast. Since P = exp2(s - m_new) ∈ [0, 1] by
    # construction (per-row max of P is exactly 1.0), p_amax = 1.0.
    FP8_MAX = 448.0
    INV_FP8_MAX = 1.0 / FP8_MAX

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

    # K steps for GEMM2 per BLOCK_N tile: BLOCK_N // MFMA_K_FP8
    PV_K_STEPS = BLOCK_N // MFMA_K_FP8

    assert head_dim % MFMA_K_INT8 == 0, f"head_dim {head_dim} must be divisible by {MFMA_K_INT8}"
    assert BLOCK_M % ROWS_PER_WAVE == 0
    assert BLOCK_N % MFMA_K_FP8 == 0

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_Q_HEADS = num_q_heads
    NUM_KV_HEADS = num_kv_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    GROUPS = num_q_heads // num_kv_heads  # GQA group size

    STRIDE_TOKEN = NUM_Q_HEADS * HEAD_DIM
    KV_STRIDE_TOKEN = NUM_KV_HEADS * HEAD_DIM

    # LDS layout
    # K: row-major BLOCK_N × HEAD_DIM (Int8, 1 byte/elem). +8 byte pad per row.
    # V on gfx942: COLUMN-MAJOR HEAD_DIM × BLOCK_N (FP8 raw bytes, 1 byte/elem).
    #    +8 byte pad per V "row" (= V "column" in the original BLOCK_N×HEAD_DIM
    #    view). Storing V transposed makes per-lane PV reads contiguous in K
    #    (32 contiguous bytes at fixed d), enabling 2× ds_read_b128 instead of
    #    32× ds_read_u8.
    # V on gfx950 (Stage IV-c): ROW-MAJOR BLOCK_N × HEAD_DIM (matches global
    #    layout). Cooperative load is 1× ds_write_b128 per row instead of 16×
    #    scattered ds_write_b8. PV per-lane gather uses ds_read_tr8_b64 (CDNA4-
    #    only; performs a 16-lane in-instruction transpose) to deliver the
    #    32 K-contiguous bytes per lane that the f8f6f4 MFMA expects.
    _is_gfx950_kv = "gfx950" in gpu_arch
    K_STRIDE = HEAD_DIM + 16    # 16 B pad: HEAD_DIM=128 = 32 banks × 4 B,
                                #   so naive +0/+8 hits bank conflicts on
                                #   ds_read paths. +16 shifts each row by 4
                                #   banks → conflict-free (matches V_STRIDE).
    if _is_gfx950_kv:
        # Pad row stride by 16 bytes to break LDS bank conflicts on the
        # ds_read_tr8_b64 path. HEAD_DIM=128 = exact multiple of 32 banks ×
        # 4 B/bank, so consecutive rows hit the same bank set; adding 16 B
        # shifts each row by 4 banks.
        V_STRIDE = HEAD_DIM + 16
    else:
        # Pad V "row" stride to a 16-byte multiple so loads can issue as
        # ds_read_b128. BLOCK_N=128 + 16 = 144 bytes per row.
        V_STRIDE = BLOCK_N + 16

    # Cooperative load parameters
    VEC_WIDTH_K = 16   # 16 Int8 elements = 16 bytes per thread
    # V coop load (gfx942): each thread reads 16 contiguous FP8 bytes from one
    # global V row (16 d-positions) and SCATTERS them as 16 separate bytes
    # into column-major LDS at 16 different LDS rows (the d-positions), same
    # LDS column (the kv_row).
    # V coop load (gfx950): each thread reads 16 contiguous FP8 bytes and
    # WRITES them contiguously into row-major LDS as one ds_write_b128.
    VEC_WIDTH_V = 16   # 16 FP8 bytes per thread (one global vec load)
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

    # V LDS section.
    # gfx942 (column-major): HEAD_DIM rows × V_STRIDE bytes per row.
    # gfx950 (row-major):    BLOCK_N rows × V_STRIDE (=HEAD_DIM) bytes per row.
    LDS_K_BYTES = BLOCK_N * K_STRIDE * 1   # 1 byte per Int8
    if _is_gfx950_kv:
        LDS_V_BYTES = BLOCK_N * V_STRIDE * 1   # row-major: BLOCK_N × HEAD_DIM
    else:
        LDS_V_BYTES = HEAD_DIM * V_STRIDE * 1  # column-major: HEAD_DIM × V_STRIDE
    # Stage III: 2-stage software pipeline. Allocate two ping-pong buffers for
    # K and V so the next KV block's loads can be issued while the current
    # block's MFMAs are still executing.
    # Layout: [K0 | K1 | V0 | V1]. K and V groups are kept contiguous so each
    # typed view (lds, lds_v) covers all of its buffers — that lets the LLVM
    # backend keep the alignment hint that lowers V reads to ds_read_b128.
    NUM_PIPE_STAGES = 2
    LDS_K_TOTAL_BYTES = LDS_K_BYTES * NUM_PIPE_STAGES
    LDS_V_TOTAL_BYTES = LDS_V_BYTES * NUM_PIPE_STAGES
    LDS_TOTAL_BYTES = LDS_K_TOTAL_BYTES + LDS_V_TOTAL_BYTES

    # We store K as i8 and V as bf16. Because the SmemAllocator works in elements
    # and we share one LDS array, we allocate in Int8 units and use byte offsets.
    # K section: LDS_K_TOTAL_BYTES bytes starting at lds_offset
    # V section: LDS_V_TOTAL_BYTES bytes starting at lds_offset + LDS_K_TOTAL_BYTES
    LDS_TOTAL_I8_ELEMS = LDS_TOTAL_BYTES  # 1:1 byte mapping for i8 allocator

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"sage_attn_cdna_smem_M{BLOCK_M}N{BLOCK_N}_{path_tag}",
    )
    lds_base_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_base_offset + LDS_TOTAL_I8_ELEMS

    # Architecture-specific FP8 type (gfx942: fnuz, gfx950: fn)
    # Deferred: MLIR types must be constructed inside an active MLIR Context,
    # which only exists inside the @flyc.kernel scope.
    _is_gfx950 = "gfx950" in gpu_arch
    _fp8_mlir_type_cls = ir.Float8E4M3FNType if _is_gfx950 else ir.Float8E4M3FNUZType

    # The 32x32 MFMA variants don't have a flydsl.expr.rocdl wrapper, so we
    # call the raw ODS class directly with positional arguments.
    from flydsl._mlir.dialects._rocdl_ops_gen import (
        mfma_i32_32x32x32_i8 as _ods_mfma_i32_32x32x32_i8,
        mfma_scale_f32_32x32x64_f8f6f4 as _ods_mfma_scale_f32_32x32x64_f8f6f4,
        ds_read_tr8_b64 as _ods_ds_read_tr8_b64,
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

    def mfma_fp8_k64(result_type, a, b, c, scale_a, scale_b):
        """rocdl.mfma.scale.f32.32x32x64.f8f6f4 (fp8 * fp8) wrapper.

        a, b: vector<8xi32> per lane (= 32 fp8 bytes per lane).
        c: vector<16xf32> per lane (accumulator; same layout as 32x32x16_bf16).
        scale_a, scale_b: i32 (e8m0). Pass 127 for "no scaling" (1.0 multiplier).
        opselA=opselB=0 selects the FP8 path.
        """
        a_v = a.ir_value() if hasattr(a, "ir_value") and not isinstance(a, ir.Value) else a
        b_v = b.ir_value() if hasattr(b, "ir_value") and not isinstance(b, ir.Value) else b
        c_v = c.ir_value() if hasattr(c, "ir_value") and not isinstance(c, ir.Value) else c
        return _ods_mfma_scale_f32_32x32x64_f8f6f4(
            res=result_type,
            a=a_v, b=b_v, c=c_v,
            cbsz=0, blgp=0,
            opselA=0, scaleA=scale_a,
            opselB=0, scaleB=scale_b,
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
        batch_size: fx.Int32,
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

        # Buffer descriptors for K/V global loads.
        # Tight num_records_bytes = B*S*Hk*HEAD_DIM lets the hardware return
        # 0 for OOB lanes past tensor end, so we can drop the row-clamp +
        # V zero-fill selects in coop_load_k/v. Score masking already pins
        # P=0 for KV columns past seq_len_k, so even inner-batch overflow
        # rows that fall into a neighboring batch's bytes contribute 0 to
        # the PV MFMA (0 * any-fp8 = 0; sage_quant outputs are NaN-free).
        bs_v_for_rsrc = fx.Index(batch_size)
        slk_v_for_rsrc = fx.Index(seq_len_k)
        kv_total_bytes = bs_v_for_rsrc * slk_v_for_rsrc * fx.Index(NUM_KV_HEADS * HEAD_DIM)
        k_rsrc = buffer_ops.create_buffer_resource(
            K, max_size=False, num_records_bytes=kv_total_bytes,
        )
        v_rsrc = buffer_ops.create_buffer_resource(
            V, max_size=False, num_records_bytes=kv_total_bytes,
        )

        v4i32_type = Vec.make_type(4, fx.Int32)
        v4f32_type = Vec.make_type(4, fx.Float32)
        v8f32_type = Vec.make_type(8, fx.Float32)
        v16i32_type = Vec.make_type(16, fx.Int32)
        v16f32_type = Vec.make_type(16, fx.Float32)
        # MFMA 32x32x32_i8: A/B = v4i32 (=16xi8 per lane), C/D = v16i32 per lane (wave64)
        v8i8_type = Vec.make_type(8, fx.Int8)
        # MFMA 32x32x64_f8f6f4: A/B = v8i32 (=32xi8 per lane), C/D = v16f32 per lane (wave64)
        v8i32_type = Vec.make_type(8, fx.Int32)
        v32i8_type = Vec.make_type(32, fx.Int8)
        v4bf16_type = Vec.make_type(4, fx.BFloat16)
        v16i8_type = Vec.make_type(16, fx.Int8)
        v8bf16_type = Vec.make_type(8, fx.BFloat16)
        v4i16_type = Vec.make_type(4, fx.Int16)
        v8i16_type = Vec.make_type(8, fx.Int16)

        seq_len_q_v = fx.Index(seq_len_q)
        seq_len_k_v = fx.Index(seq_len_k)

        base_ptr = allocator.get_base()
        # Shared LDS as Int8 view: covers BOTH K ping-pong buffers (K0 | K1).
        # Buffer index is a runtime value; per-buffer LDS index = base_idx +
        # buf_idx * LDS_K_BYTES.
        lds = SmemPtr(
            base_ptr,
            lds_base_offset,
            T.i8,
            shape=(LDS_K_TOTAL_BYTES,),
        ).get()
        # Separate, statically-shaped view for V — gives the LLVM backend a
        # tight bound that helps prove 16-byte alignment for vector loads, so
        # they can lower to ds_read_b128 instead of 32× ds_read_u8. Covers
        # BOTH V ping-pong buffers (V0 | V1).
        lds_v = SmemPtr(
            base_ptr,
            lds_base_offset + LDS_K_TOTAL_BYTES,
            T.i8,
            shape=(LDS_V_TOTAL_BYTES,),
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
        # buf_off: byte offset into the K-side LDS that selects the ping-pong
        # buffer. 0 = buffer 0, LDS_K_BYTES = buffer 1.
        def coop_load_k(tile_start, buf_off):
            for batch in range_constexpr(NUM_BATCHES_K):
                row_offset = batch * ROWS_PER_BATCH_K
                row_idx_raw = tile_start + load_row_k_batch + row_offset
                # Tight buffer descriptor (num_records_bytes=B*S*Hk*HEAD_DIM)
                # makes hardware return 0 for OOB lanes past tensor end. For
                # inner-batch tail rows that overflow into a neighbouring
                # batch the score mask still pins P=0 for those columns, so
                # the PV contribution is 0 regardless. No row-clamp select
                # needed.
                if const_expr(K_NEEDS_GUARD):
                    row_valid = load_row_k_batch < fx.Index(BLOCK_N)
                    do_load = row_valid
                else:
                    do_load = True
                if do_load:
                    g_idx = kv_global_idx(row_idx_raw, load_col_k_base)
                    lds_row = load_row_k_batch + row_offset
                    lds_idx = buf_off + lds_row * K_STRIDE + load_col_k_base
                    # buffer_load_dwordx4: g_idx is in i8 elements (=bytes); divide
                    # by 4 to get the dword index buffer_load expects when dtype=i32.
                    g_dword_i32 = arith.unwrap(
                        arith.index_cast(T.i32, _raw(g_idx >> fx.Index(2)))
                    )
                    v4i32 = buffer_ops.buffer_load(
                        k_rsrc, g_dword_i32, vec_width=4, dtype=T.i32
                    )
                    vec = vector.bitcast(v16i8_type, v4i32)
                    Vec(vec).store(lds, [lds_idx])

        # V is stored in the upper half of LDS, COLUMN-MAJOR (transposed):
        # the LDS V section is HEAD_DIM rows × V_STRIDE bytes per row, where
        # row index = d (0..HEAD_DIM-1) and the first BLOCK_N bytes of each row
        # hold consecutive kv positions (column = kv_row).
        # Storing V transposed makes per-lane PV reads be 32 contiguous bytes
        # (one ds_read_b256 = 2 ds_read_b128) at fixed d, kv_row=k_start..+31.
        LDS_V_BYTE_BASE = LDS_K_BYTES  # byte offset of V section

        def coop_load_v(tile_start, buf_off):
            """Load V from global (FP8 raw bytes) into LDS.

            gfx942 (column-major LDS): each thread reads VEC_WIDTH_V=16
              contiguous FP8 bytes from one global V row and SCATTERS them as
              16 individual byte writes to 16 different LDS rows.
            gfx950 (row-major LDS): each thread reads VEC_WIDTH_V=16 contiguous
              FP8 bytes and writes them contiguously as one ds_write_b128 —
              same shape as global, no transpose. Reduces ~2048 ds_write_b8
              transactions per KV iter down to ~128 ds_write_b128 transactions.

            Tight buffer descriptor (num_records_bytes=B*S*Hk*HEAD_DIM) makes
            hardware return 0 for OOB lanes past tensor end. For inner-batch
            tail rows that overflow into a neighbouring batch the score mask
            pins P=0 for those columns, so 0 * (whatever V byte) = 0 in the
            PV MFMA. sage_quant outputs are NaN-free, so no NaN propagation.

            buf_off: byte offset into lds_v selecting the ping-pong V buffer.
            """
            for batch in range_constexpr(NUM_BATCHES_V):
                row_offset = batch * ROWS_PER_BATCH_V
                row_idx_raw = tile_start + load_row_v_batch + row_offset
                if const_expr(V_NEEDS_GUARD):
                    row_valid = load_row_v_batch < fx.Index(BLOCK_N)
                    do_load = row_valid
                else:
                    do_load = True
                if do_load:
                    g_idx = kv_global_idx(row_idx_raw, load_col_v_base)
                    g_dword_i32 = arith.unwrap(
                        arith.index_cast(T.i32, _raw(g_idx >> fx.Index(2)))
                    )
                    v4i32 = buffer_ops.buffer_load(
                        v_rsrc, g_dword_i32, vec_width=4, dtype=T.i32
                    )
                    raw_v = vector.bitcast(v16i8_type, v4i32)

                    if const_expr(_is_gfx950):
                        # Row-major: LDS[k_v * V_STRIDE + d]. Each thread writes
                        # its 16 contiguous d-bytes for one k_v row as a single
                        # ds_write_b128.
                        lds_row = load_row_v_batch + row_offset  # = k_v
                        v_off = buf_off + lds_row * V_STRIDE + load_col_v_base
                        Vec(raw_v).store(lds_v, [v_off])
                    else:
                        # Column-major: LDS[d * V_STRIDE + k_v]. Scatter 16
                        # individual ds_write_u8 to 16 different LDS rows.
                        lds_col = load_row_v_batch + row_offset
                        for di in range_constexpr(VEC_WIDTH_V):
                            d_idx = load_col_v_base + di
                            v_off = buf_off + d_idx * V_STRIDE + lds_col
                            b_i8 = vector.extract(
                                raw_v, static_position=[di], dynamic_position=[]
                            )
                            Vec.from_elements([b_i8], fx.Int8).store(lds_v, [v_off])

        def load_k_frag(kv_block_row, ks, buf_off):
            """Load v4i32 (=16xi8) K fragment from LDS for mfma_i32_32x32x32_i8.

            kv_block_row: row within BLOCK_N tile (0..BLOCK_N-1)
            ks: K-step index (selects 32-wide K chunk; klane picks the 16-byte half)
            buf_off: ping-pong K-buffer byte offset.
            """
            k_col = fx.Index(ks * MFMA_K_INT8) + klane * 16
            lds_idx = buf_off + fx.Index(kv_block_row * K_STRIDE) + k_col
            v16i8 = Vec.load(v16i8_type, lds, [lds_idx])
            return vector.bitcast(v4i32_type, v16i8)

        def load_v_frag_fp8(pks, dc, buf_off, iter_lane_addr_i32):
            """Load v8i32 (=32xi8 fp8) V fragment from LDS for mfma_scale_f32_32x32x64.

            For lane (klane, lane32) at PV step `pks` and D chunk `dc`, the A
            operand needs V[d=dc*32+lane32, k=pks*64+klane*32 .. +31].

            gfx942 path (column-major LDS, V_STRIDE = BLOCK_N+16):
              LDS layout has d as the row index and k_v as the column index, so
              the 32 K-bytes per lane are 32 LDS-contiguous bytes. Issue as
              two 16-byte vector loads (compiler lowers to 2× ds_read_b128).

            gfx950 path (row-major LDS, V_STRIDE = HEAD_DIM):
              Use 4× ds_read_tr8_b64 (CDNA4-only). Each instruction is a
              SIMD-64 op composed of 4 independent 16-lane sub-ops; within
              each 16-lane group the instruction does an in-tile transpose so
              that physical lane t outputs 8 K-bytes at fixed d=base_d + (t%16).

              Per-lane LDS address (verified by /tmp/probe_ds_read_tr8_b64.py
              against the Triton emitter at MemoryOpToLLVM.cpp:140-330):
                g       = lane // 16    # which 16-lane sub-group
                i       = lane %  16    # output position within sub-group
                base_kv = pks*64 + (g//2)*32 + k_idx*8     # k_idx ∈ {0,1,2,3}
                base_d  = dc*32  + (g%2)*16
                addr    = (base_kv + i//2) * V_STRIDE + (base_d + (i%2)*8)
              4 calls (k_idx 0..3) per lane gather 32 contiguous K-bytes.

              `iter_lane_addr_i32` (gfx950): precomputed per-iter, per-lane
              i32 LDS address holding (lds_v_base + cur_v_off + lane-invariant
              part). The (pks, dc, k_idx)-dependent part is added per call as
              a compile-time literal so the AMDGPU backend folds it into
              `ds_read offset:LIT`, eliminating per-call `v_or_b32`.
            """
            if const_expr(_is_gfx950):
                lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
                v2i32_type = Vec.make_type(2, fx.Int32)

                words = []
                for k_idx in range_constexpr(4):
                    # Compile-time-constant byte offset for this (pks, dc, k_idx).
                    lit_bytes = (pks * MFMA_K_FP8 + k_idx * 8) * V_STRIDE \
                                + dc * MFMA_M
                    addr_i32 = arith.AddIOp(
                        iter_lane_addr_i32,
                        arith.constant(lit_bytes, type=T.i32),
                    ).result
                    ptr_val = _llvm.inttoptr(lds_ptr_ty, addr_i32)
                    v2i32 = _ods_ds_read_tr8_b64(
                        res=v2i32_type, ptr=ptr_val
                    ).result
                    words.append(v2i32)

                # Pack 4× v2i32 → v8i32 (= 32 fp8 bytes per lane).
                elems = []
                for w_idx in range_constexpr(4):
                    for sub in range_constexpr(2):
                        elems.append(vector.extract(
                            words[w_idx], static_position=[sub], dynamic_position=[]
                        ))
                return Vec.from_elements(elems, fx.Int32).ir_value()

            # gfx942 column-major path (unchanged).
            d_col = dc * MFMA_M + lane32
            kv_k_start = fx.Index(pks * MFMA_K_FP8) + klane * 32
            v_off = buf_off + d_col * V_STRIDE + kv_k_start
            lo_v16i8 = Vec.load(v16i8_type, lds_v, [v_off])
            hi_v16i8 = Vec.load(v16i8_type, lds_v, [v_off + 16])
            lo_v4i32 = vector.bitcast(v4i32_type, lo_v16i8)
            hi_v4i32 = vector.bitcast(v4i32_type, hi_v16i8)
            elems = []
            for w in range_constexpr(4):
                elems.append(vector.extract(lo_v4i32, static_position=[w], dynamic_position=[]))
            for w in range_constexpr(4):
                elems.append(vector.extract(hi_v4i32, static_position=[w], dynamic_position=[]))
            return Vec.from_elements(elems, fx.Int32).ir_value()

        def f32_to_bf16_trunc(f32_raw):
            """Bitwise f32 → bf16 truncation (upper 16 bits)."""
            i32_val = arith.BitcastOp(T.i32, f32_raw).result
            i16_val = arith.TruncIOp(
                T.i16,
                arith.ShRUIOp(i32_val, arith.constant(16, type=T.i32)).result,
            ).result
            return arith.BitcastOp(T.bf16, i16_val).result

        # ---- Main loop: iterate over KV tiles ----
        # Stage III: 2-stage software pipeline.
        #   Prologue: prefetch KV block 0 into buffer 0.
        #   Iter i:   barrier; if not last, prefetch block (i+1) into buf (i+1)%2;
        #             then QK MFMA + softmax + PV MFMA on buf (i%2).
        # Buffer parity is carried as a loop-state i32 (0 or 1) to keep the LDS
        # offset on the fast scalar-broadcast path (no integer division by
        # BLOCK_N inside the hot loop).
        K_BUF1_OFF = fx.Index(LDS_K_BYTES)
        V_BUF1_OFF = fx.Index(LDS_V_BYTES)
        ZERO_INDEX = fx.Index(0)

        def _buf_off(buf_idx_i32, stride_index):
            """buf_idx ∈ {0,1} → byte offset in {0, stride}, returned as fx.Index."""
            is_one = ArithValue(fx.Int32(buf_idx_i32) == fx.Int32(1))
            return fx.Index(is_one.select(stride_index, ZERO_INDEX))

        # Per-lane V LDS base address (gfx950 fast path): hoist the lane-
        # invariant component once so per-iter ds_reads only need to ADD
        # cur_v_off and a compile-time literal for (pks, dc, k_idx). This
        # eliminates the per-ds_read `v_or_b32 vXX, s4, vYY` pattern that
        # accounted for ~33 instr/iter vs Triton.
        if const_expr(_is_gfx950):
            from flydsl.expr.utils.arith import ArithValue as _AV
            _g = lane // fx.Index(16)
            _i = lane % fx.Index(16)
            _klane_g = _g // fx.Index(2)
            _d_group = _g % fx.Index(2)
            _row_off_within = _i // fx.Index(2)
            _col_extra_within = (_i % fx.Index(2)) * fx.Index(8)
            _per_lane_v_inv = (_klane_g * fx.Index(32) + _row_off_within) \
                              * fx.Index(V_STRIDE) \
                              + _d_group * fx.Index(16) + _col_extra_within
            _lds_v_base = _memref.extract_aligned_pointer_as_index(
                _llvm_value(lds_v)
            )
            _v_lane_base_index = _AV(_lds_v_base) + _per_lane_v_inv
            v_lane_base_i32 = arith.unwrap(
                arith.index_cast(T.i32, _v_lane_base_index)
            )

        # ===== Straight QK→softmax→PV per iter (Attack #3 2026-05-11) =====
        # Cross-iter softmax pipelining was tried and added scf.for yield
        # pressure for p_words+corr (16 i32 + 1 f32 across the boundary)
        # without a measured win at long-S vs Triton. Triton uses straight
        # QK→softmax→PV per iter and wins. Restoring that simpler structure
        # to (a) shrink loop-carried state by ~17 vregs and (b) let the JIT
        # scheduler interleave softmax VALU into PV MFMA shadow naturally
        # within a single iter rather than across the yield boundary.
        N_SUBTILES = BLOCK_N // MFMA_N
        ELEMS_PER_TILE = 16
        NUM_S_VALS = N_SUBTILES * ELEMS_PER_TILE

        # Hoist constants used by the softmax helper from inside-loop scope.
        klane_i32 = fx.Int32(klane)
        klane_off_i32 = klane_i32 * fx.Int32(4)
        seq_len_k_i32 = fx.Int32(seq_len_k_v)

        def _emit_qk_softmax_pquant(kv_block_start_arg, k_buf_off_arg,
                                    m_in, l_in):
            """Emit QK MFMA + scale + mask + softmax + p-quant for the KV tile
            at kv_block_start_arg, using K loaded into LDS at k_buf_off_arg.

            Returns (m_new, l_new, corr, p_words_2d) — IR values.
            """
            # 1) QK MFMA
            s_accs_loc = [_raw(c_zero_v16i32) for _ in range(N_SUBTILES)]
            for ks in range_constexpr(K_STEPS_QK):
                for st in range_constexpr(N_SUBTILES):
                    kv_row_loc = lane32 + st * MFMA_N
                    k_frag = load_k_frag(kv_row_loc, ks, k_buf_off_arg)
                    s_accs_loc[st] = mfma_i32_k32(
                        v16i32_type,
                        [k_frag, q_packs[ks], s_accs_loc[st], 0, 0, 0],
                    )

            # 2) Scale (with descale-index clamp for the last-iter OOB tile)
            kv_tile_idx_loc = kv_block_start_arg // BLOCK_N
            max_kv_tile = num_k_blocks_per_head - fx.Index(1)
            kv_tile_safe = fx.Index(
                ArithValue(kv_tile_idx_loc < num_k_blocks_per_head).select(
                    kv_tile_idx_loc, max_kv_tile
                )
            )
            k_descale_base_loc = (
                batch_idx * NUM_KV_HEADS * num_k_blocks_per_head
                + head_kv_idx * num_k_blocks_per_head
                + kv_tile_safe
            )
            k_ds_loc = fx.Float32(_load_ptr_f32(kds_ptr, k_descale_base_loc))
            qk_scale_loc = _fmul(q_ds, k_ds_loc)
            qk_scale_v16_loc = (
                Vec.from_elements([qk_scale_loc], fx.Float32).broadcast_to(16)
            )
            s_f32_loc = []
            for st in range_constexpr(N_SUBTILES):
                s_i32_vec = Vec(s_accs_loc[st])
                s_f32_vec = s_i32_vec.to(fx.Float32)
                s_scaled_vec = Vec(
                    arith.mulf(s_f32_vec, qk_scale_v16_loc, fastmath=fm_fast)
                )
                for elem in range_constexpr(ELEMS_PER_TILE):
                    s_f32_loc.append(s_scaled_vec[elem])

            # 3) Mask
            kv_start_i32_loc = fx.Int32(
                arith.unwrap(arith.index_cast(T.i32, _raw(kv_block_start_arg)))
            )
            if const_expr(CAUSAL):
                s_named = list(s_f32_loc)
                for st in range_constexpr(N_SUBTILES):
                    for elem in range_constexpr(ELEMS_PER_TILE):
                        idx = st * ELEMS_PER_TILE + elem
                        msub = elem // 4
                        erem = elem % 4
                        kv_col_i32 = (
                            kv_start_i32_loc
                            + fx.Int32(st * MFMA_N)
                            + fx.Int32(msub * 8)
                            + klane_off_i32
                            + fx.Int32(erem)
                        )
                        out_of_range = ArithValue(kv_col_i32 >= seq_len_k_i32)
                        out_of_range = out_of_range | ArithValue(kv_col_i32 > q_row_i32)
                        s_named[idx] = out_of_range.select(c_neg_inf, s_named[idx])
                s_f32_loc = s_named
            else:
                tile_end = kv_block_start_arg + fx.Index(BLOCK_N)
                tile_oob_av = ArithValue(tile_end > seq_len_k_v)
                cond_i1 = _raw(tile_oob_av)
                f32_ty = ir.F32Type.get()
                result_types = [f32_ty] * NUM_S_VALS
                if_op = _scf.IfOp(
                    cond_i1, result_types, has_else=True,
                    loc=ir.Location.unknown(),
                )
                with ir.InsertionPoint(if_op.then_block):
                    _llvm.inline_asm(
                        None, [], "", "", has_side_effects=True,
                    )
                    masked = []
                    for st in range_constexpr(N_SUBTILES):
                        for elem in range_constexpr(ELEMS_PER_TILE):
                            idx = st * ELEMS_PER_TILE + elem
                            msub = elem // 4
                            erem = elem % 4
                            kv_col_i32 = (
                                kv_start_i32_loc
                                + fx.Int32(st * MFMA_N)
                                + fx.Int32(msub * 8)
                                + klane_off_i32
                                + fx.Int32(erem)
                            )
                            out_of_range = ArithValue(kv_col_i32 >= seq_len_k_i32)
                            masked.append(_raw(out_of_range.select(c_neg_inf, s_f32_loc[idx])))
                    _scf.YieldOp(masked)
                with ir.InsertionPoint(if_op.else_block):
                    _scf.YieldOp([_raw(v) for v in s_f32_loc])
                s_f32_loc = list(if_op.results)

            # 4) Softmax
            local_max_l = s_f32_loc[0]
            for r in range_constexpr(NUM_S_VALS - 1):
                local_max_l = _fmax(local_max_l, s_f32_loc[r + 1])
            row_max_l = row_max_reduce(local_max_l)
            m_new_l = _fmax(m_in, row_max_l)
            diff_m_l = _fsub(m_in, m_new_l)
            corr_l = rocdl.exp2(ir.F32Type.get(), _raw(diff_m_l))
            neg_max_l = _fsub(c_zero_f, m_new_l)
            p_vals_l = []
            local_sum_l = _raw(c_zero_f)
            for r in range_constexpr(NUM_S_VALS):
                diff = _fadd(s_f32_loc[r], neg_max_l)
                p = rocdl.exp2(ir.F32Type.get(), _raw(diff))
                p_vals_l.append(p)
                local_sum_l = _fadd(local_sum_l, p)
            tile_sum_l = row_sum_reduce(local_sum_l)
            l_new_l = _fadd(_fmul(corr_l, l_in), tile_sum_l)

            # 5) P-quant → p_words[st][w] : i32
            c0_i32_l = arith.constant(0, type=T.i32)
            p_words_l = []
            for st in range_constexpr(N_SUBTILES):
                sub = []
                for w in range_constexpr(4):
                    s0 = _raw(p_vals_l[st * ELEMS_PER_TILE + w * 4 + 0])
                    s1 = _raw(p_vals_l[st * ELEMS_PER_TILE + w * 4 + 1])
                    s2 = _raw(p_vals_l[st * ELEMS_PER_TILE + w * 4 + 2])
                    s3 = _raw(p_vals_l[st * ELEMS_PER_TILE + w * 4 + 3])
                    packed = c0_i32_l
                    packed = rocdl.cvt_pk_fp8_f32(T.i32, s0, s1, packed, 0)
                    packed = rocdl.cvt_pk_fp8_f32(T.i32, s2, s3, packed, 1)
                    sub.append(packed)
                p_words_l.append(sub)
            return m_new_l, l_new_l, corr_l, p_words_l

        # ---- Prologue ----
        # Prefetch buf 0 (K[0]/V[0]) AND buf 1 (K[1]/V[1] — harmless if num_iters<2;
        # coop loaders clamp OOB rows to row 0). Then barrier; iter 0 computes
        # softmax[0] from buf 0 in-line.
        coop_load_k(ZERO_INDEX, ZERO_INDEX)
        coop_load_v(ZERO_INDEX, ZERO_INDEX)
        coop_load_k(fx.Index(BLOCK_N), K_BUF1_OFF)
        coop_load_v(fx.Index(BLOCK_N), V_BUF1_OFF)
        gpu.barrier()

        # i32 0 — current buffer index for iter 0 (buf 0 holds K[0]/V[0]).
        c0_i32_init = arith.constant(0, type=T.i32)

        # Iter args layout (Attack #3 — no cross-iter softmax):
        #   [cur_buf (i32), m (f32), l (f32), *o_accs (D_CHUNKS × v16f32)]
        # Drops `corr` and `p_words` (N_SUBTILES * 4 i32) from the prior
        # cross-iter scheme — those values are now consumed within the same
        # iter that produces them and need not cross the scf.for yield.
        init_args = [c0_i32_init, _raw(c_neg_inf), _raw(c_zero_f)]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(_raw(c_zero_v16f32))

        _OFF_CUR_BUF = 0
        _OFF_M = 1
        _OFF_L = 2
        _OFF_O_ACCS = 3

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(0, kv_upper, BLOCK_N, init=init_args):
            cur_buf_i32 = inner_iter_args[_OFF_CUR_BUF]
            m_running = inner_iter_args[_OFF_M]
            l_running = inner_iter_args[_OFF_L]
            o_accs = [inner_iter_args[_OFF_O_ACCS + i] for i in range_constexpr(D_CHUNKS)]

            # Buffer offsets
            cur_k_off = _buf_off(cur_buf_i32, K_BUF1_OFF)
            cur_v_off = _buf_off(cur_buf_i32, V_BUF1_OFF)
            if const_expr(_is_gfx950):
                cur_v_off_i32 = arith.unwrap(arith.index_cast(T.i32, cur_v_off))
                v_iter_lane_addr_i32 = arith.AddIOp(
                    v_lane_base_i32, cur_v_off_i32
                ).result
            else:
                v_iter_lane_addr_i32 = None
            next_buf_i32 = arith.XOrIOp(
                _raw(cur_buf_i32), arith.constant(1, type=T.i32)
            ).result

            # ==== Compute softmax[k] using cur_buf K ====
            m_new, l_new, corr, p_words = _emit_qk_softmax_pquant(
                kv_block_start, cur_k_off, m_running, l_running
            )

            # ==== Apply correction factor to o_accs (in-iter) ====
            corr_vec16 = (
                Vec.from_elements([corr], fx.Float32).broadcast_to(16).ir_value()
            )
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = _fmul(o_accs[dc], corr_vec16)

            # ==== GEMM2: PV[k] using p_words[k] (just computed) and cur_buf V ====
            from flydsl._mlir.dialects._rocdl_ops_gen import permlane32_swap as _permlane32_swap_op
            _struct_ty_2xi32 = ir.Type.parse("!llvm.struct<(i32, i32)>")
            for pks in range_constexpr(PV_K_STEPS):
                v8_elems = []
                for w in range_constexpr(4):
                    a_w = _raw(p_words[pks * 2][w])
                    b_w = _raw(p_words[pks * 2 + 1][w])
                    swapped = _permlane32_swap_op(
                        _struct_ty_2xi32, old=a_w, src=b_w,
                        fi=False, bound_control=True,
                    )
                    lo_word = _llvm.extractvalue(T.i32, swapped, [0])
                    hi_word = _llvm.extractvalue(T.i32, swapped, [1])
                    v8_elems.append(lo_word)
                    v8_elems.append(hi_word)
                p_pack_v8i32 = Vec.from_elements(v8_elems, fx.Int32).ir_value()
                scale_127 = arith.constant(127, type=T.i32)
                for dc in range_constexpr(D_CHUNKS):
                    v_frag = load_v_frag_fp8(pks, dc, cur_v_off, v_iter_lane_addr_i32)
                    o_accs[dc] = mfma_fp8_k64(
                        v16f32_type, v_frag, p_pack_v8i32, o_accs[dc],
                        scale_127, scale_127,
                    )

            # Barrier: PV V reads of cur_buf done, then prefetch K[k+2]/V[k+2]
            # into cur_buf (overwriting K[k]/V[k]). Next iter reads next_buf
            # (untouched here) so a single barrier suffices — vs the prior
            # scheme's two barriers per iter (the second was needed because
            # next iter's softmax read this-iter's prefetch target).
            gpu.barrier()

            kv_block_after_next = kv_block_start + fx.Index(2 * BLOCK_N)
            coop_load_k(kv_block_after_next, cur_k_off)
            coop_load_v(kv_block_after_next, cur_v_off)

            # ==== Yield ====
            _yield_args = [next_buf_i32, _raw(m_new), _raw(l_new)]
            for dc in range_constexpr(D_CHUNKS):
                _yield_args.append(o_accs[dc])
            loop_results = yield _yield_args

        # ---- Normalize and store O ----
        # loop_results layout: [cur_buf, m, l, *o_accs (D_CHUNKS)]
        l_final = loop_results[_OFF_L]
        o_finals = [loop_results[_OFF_O_ACCS + dc] for dc in range_constexpr(D_CHUNKS)]

        inv_l = arith.divf(_raw(c_one_f), _raw(l_final), fastmath=fm_fast)
        # No FP8_MAX compensation needed: P-quant casts P∈[0,1] directly without
        # the FP8_MAX pre-scale (matches Triton).
        inv_l_fp8 = _raw(inv_l)

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
                        scale = arith.mulf(_raw(inv_l_fp8), _raw(v_ds), fastmath=fm_fast)
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
            batch_size, seq_len_q, seq_len_k, num_q_blocks,
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
        "llvm_options": {"enable-post-misched": True, "lsr-drop-solution": True},
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
