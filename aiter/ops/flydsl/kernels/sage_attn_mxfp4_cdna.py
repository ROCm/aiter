# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL Sage Attention MXFP4 kernel for CDNA gfx950.

Drop-in replacement for the Triton ``fav3_sage_mxfp4_func`` (FlashAttention-v3
forward with packed-FP4 Q/K and FP8 V).

  - GEMM1 (Q×K^T): mfma_scale_f32_32x32x64_f8f6f4 in FP4 mode (cbsz=4, blgp=4).
                   Probe-validated: K=32 elements per call (one MXFP4 scale
                   group of 32). Per lane: lower 16 bytes of v8i32 are active
                   = 32 nibbles. opselA selects ONE byte (e8m0) of i32 scaleA
                   → that scale applies to the entire K=32 of the call. To
                   support per-32-group scaling for head_dim=128, we issue 4
                   MFMA calls per (q-tile, n-subtile), one per scale group.
  - GEMM2 (P×V):   mfma_scale_f32_32x32x64_f8f6f4 in FP8 mode (cbsz=0, blgp=0,
                   scales=127). Identical to sage v1 PV path.

MXFP4 vs INT8 sage v1: same QK MFMA chain length, half the Q/K bandwidth.

Architecture: gfx950 only (FP4 MFMA is gfx950-exclusive).
Layout: Q/K stored as uint8 packed FP4 (head_dim_bytes = head_dim/2).
        V stored as FP8 (head_dim bytes per row).
        Q_descale / K_descale: uint8 e8m0, shape [B, S, H, D//32].
        V_descale: f32, shape [B, Hkv, D] (per-channel).

Grid:   (batch * num_q_tiles * num_q_heads,)
Block:  (NUM_WAVES * 64,) -- default 8 waves → 512 threads
Supports: causal masking, GQA/MQA, optional bias (delta_s for q-smoothing).
"""

import math as host_math
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


def build_sage_attn_mxfp4_cdna_module(
    num_q_heads,
    num_kv_heads,
    head_dim,
    causal=False,
    use_bias=False,
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
    """Build FlyDSL MXFP4 sage attention kernel for CDNA gfx950.

    Drop-in for the Triton ``fav3_sage_mxfp4_func``. FP4 MFMA for QK^T,
    FP8 MFMA for PV. Per-32-group scales applied inside the QK MFMAs.
    """
    gpu_arch = get_hip_arch()
    if "gfx950" not in gpu_arch:
        raise ValueError(
            f"sage_attn_mxfp4 requires gfx950 (FP4 MFMA), got {gpu_arch}"
        )
    if head_dim != 128:
        raise ValueError(
            f"sage_attn_mxfp4 currently restricted to head_dim=128, got {head_dim}"
        )
    WARP_SIZE = get_warp_size(gpu_arch)  # 64 for CDNA

    # MFMA 32x32 tile dimensions (matches sage v1 wave layout)
    MFMA_M = 32
    MFMA_N = 32

    # FP4 MFMA: mfma_scale_f32_32x32x64_f8f6f4 with cbsz=4, blgp=4 processes
    # K=32 elements per call (one MXFP4 scale group). Per lane: 16 active
    # bytes = 32 nibbles. Other 16 bytes of v8i32 zeroed.
    MFMA_K_FP4 = 32  # one scale group per call
    # PV: same MFMA in FP8 mode — K=64 elements per call.
    MFMA_K_FP8 = 64

    ROWS_PER_WAVE = MFMA_M  # 32 output rows per wave

    BLOCK_M = block_m if block_m is not None else 256
    BLOCK_N = block_n if block_n is not None else 128

    NUM_WAVES = BLOCK_M // ROWS_PER_WAVE
    if flat_work_group_size is None:
        flat_work_group_size = NUM_WAVES * WARP_SIZE
    BLOCK_SIZE = flat_work_group_size

    # K steps for GEMM1 (QK): one per scale group of 32 along head_dim.
    K_STEPS_QK = head_dim // MFMA_K_FP4

    # FP4 packs 2 elements per byte. Storage byte stride is head_dim / 2.
    HEAD_DIM_BYTES = head_dim // 2

    # D chunks for GEMM2 (PV): head_dim / MFMA_N output columns per chunk
    D_CHUNK = MFMA_N
    D_CHUNKS = head_dim // D_CHUNK

    # K steps for GEMM2 per BLOCK_N tile: BLOCK_N // MFMA_K_FP8
    PV_K_STEPS = BLOCK_N // MFMA_K_FP8

    assert (
        head_dim % MFMA_K_FP4 == 0
    ), f"head_dim {head_dim} must be divisible by {MFMA_K_FP4}"
    assert BLOCK_M % ROWS_PER_WAVE == 0
    assert BLOCK_N % MFMA_K_FP8 == 0

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_Q_HEADS = num_q_heads
    NUM_KV_HEADS = num_kv_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    USE_BIAS = use_bias
    GROUPS = num_q_heads // num_kv_heads  # GQA group size
    NUM_SCALE_GROUPS = head_dim // 32  # = K_STEPS_QK; one e8m0 per 32-elem group

    # FP4 storage: head_dim/2 bytes per (token, head). V (FP8) keeps head_dim bytes.
    Q_STRIDE_TOKEN = NUM_Q_HEADS * HEAD_DIM_BYTES
    K_STRIDE_TOKEN = NUM_KV_HEADS * HEAD_DIM_BYTES
    V_STRIDE_TOKEN = NUM_KV_HEADS * HEAD_DIM

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
    _is_gfx950_kv = True  # gfx950-only path (FP4 MFMA)
    # K LDS row stride: FP4 packs 2 elem/byte, so HEAD_DIM_BYTES = HEAD_DIM/2
    # bytes per row. Add 16 B pad to break LDS bank conflicts.
    K_STRIDE = HEAD_DIM_BYTES + 16
    # V stays FP8 (1 byte per elem). Same V LDS layout as v1 gfx950 path.
    V_STRIDE = HEAD_DIM + 16

    # Cooperative load parameters
    # K is FP4 packed: each row is HEAD_DIM_BYTES = HEAD_DIM/2 bytes.
    VEC_WIDTH_K = 16  # 16 packed FP4 bytes per thread = 32 nibbles = 32 elements
    # V coop load (gfx942): each thread reads 16 contiguous FP8 bytes from one
    # global V row (16 d-positions) and SCATTERS them as 16 separate bytes
    # into column-major LDS at 16 different LDS rows (the d-positions), same
    # LDS column (the kv_row).
    # V coop load (gfx950): each thread reads 16 contiguous FP8 bytes and
    # WRITES them contiguously into row-major LDS as one ds_write_b128.
    VEC_WIDTH_V = 16  # 16 FP8 bytes per thread (one global vec load)
    THREADS_PER_ROW_K = HEAD_DIM_BYTES // VEC_WIDTH_K  # FP4: 64B per row / 16B
    THREADS_PER_ROW_V = HEAD_DIM // VEC_WIDTH_V        # FP8: 128B per row / 16B
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

    # V LDS section: row-major BLOCK_N × V_STRIDE (gfx950 only).
    # K LDS: HEAD_DIM_BYTES (= HEAD_DIM/2) bytes per FP4 row + pad.
    LDS_K_BYTES = BLOCK_N * K_STRIDE * 1
    LDS_V_BYTES = BLOCK_N * V_STRIDE * 1
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
        global_sym_name=f"sage_attn_mxfp4_cdna_smem_M{BLOCK_M}N{BLOCK_N}_{path_tag}",
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
        mfma_scale_f32_32x32x64_f8f6f4 as _ods_mfma_scale_f32_32x32x64_f8f6f4,
        ds_read_tr8_b64 as _ods_ds_read_tr8_b64,
    )

    def mfma_fp4_scaled(result_type, a, b, c, scale_a_i32, scale_b_i32,
                        opsel_a, opsel_b):
        """rocdl.mfma.scale.f32.32x32x64.f8f6f4 in FP4 mode (cbsz=4, blgp=4).

        a, b: vector<8xi32> per lane. Lower 16 bytes = 32 nibbles = 32 active
              FP4 elements per lane (one MXFP4 scale group of 32). Upper 16
              bytes must be zero.
        scale_a_i32, scale_b_i32: i32 holding 4 packed e8m0 bytes. opselA /
              opselB ∈ {0,1,2,3} selects which byte to use as the per-call
              e8m0 scale.
        """
        a_v = a.ir_value() if hasattr(a, "ir_value") and not isinstance(a, ir.Value) else a
        b_v = b.ir_value() if hasattr(b, "ir_value") and not isinstance(b, ir.Value) else b
        c_v = c.ir_value() if hasattr(c, "ir_value") and not isinstance(c, ir.Value) else c
        return _ods_mfma_scale_f32_32x32x64_f8f6f4(
            res=result_type, a=a_v, b=b_v, c=c_v,
            cbsz=4, blgp=4,
            opselA=opsel_a, scaleA=scale_a_i32,
            opselB=opsel_b, scaleB=scale_b_i32,
        ).result

    def mfma_fp8_k64(result_type, a, b, c, scale_a, scale_b):
        """rocdl.mfma.scale.f32.32x32x64.f8f6f4 (fp8 * fp8) wrapper.

        a, b: vector<8xi32> per lane (= 32 fp8 bytes per lane).
        c: vector<16xf32> per lane (accumulator; same layout as 32x32x16_bf16).
        scale_a, scale_b: i32 (e8m0). Pass 127 for "no scaling" (1.0 multiplier).
        opselA=opselB=0 selects the FP8 path.
        """
        a_v = (
            a.ir_value()
            if hasattr(a, "ir_value") and not isinstance(a, ir.Value)
            else a
        )
        b_v = (
            b.ir_value()
            if hasattr(b, "ir_value") and not isinstance(b, ir.Value)
            else b
        )
        c_v = (
            c.ir_value()
            if hasattr(c, "ir_value") and not isinstance(c, ir.Value)
            else c
        )
        return _ods_mfma_scale_f32_32x32x64_f8f6f4(
            res=result_type,
            a=a_v,
            b=b_v,
            c=c_v,
            cbsz=0,
            blgp=0,
            opselA=0,
            scaleA=scale_a,
            opselB=0,
            scaleB=scale_b,
        ).result

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def sage_attn_kernel(
        Q: fx.Tensor,  # uint8 packed FP4, 1D flattened BSHD (HEAD_DIM_BYTES per head)
        K: fx.Tensor,  # uint8 packed FP4, 1D flattened BSHD (num_kv_heads)
        V: fx.Tensor,  # FP8, 1D flattened BSHD (num_kv_heads)
        O: fx.Tensor,  # BF16 output, 1D flattened BSHD  # noqa: E741
        Q_descale: fx.Tensor,  # uint8 e8m0, shape [batch, S, num_q_heads, D//32]
        K_descale: fx.Tensor,  # uint8 e8m0, shape [batch, S, num_kv_heads, D//32]
        V_descale: fx.Tensor,  # f32, shape [batch, num_kv_heads, head_dim] (per-element)
        Bias: fx.Tensor,  # f32, shape [B, Hq, Q_NUM_BLKS, S_k] when USE_BIAS, else dummy 1B
        batch_size: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
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
        # K is FP4: HEAD_DIM_BYTES per (token, head). V is FP8: HEAD_DIM bytes.
        k_total_bytes = (
            bs_v_for_rsrc * slk_v_for_rsrc * fx.Index(NUM_KV_HEADS * HEAD_DIM_BYTES)
        )
        v_total_bytes = (
            bs_v_for_rsrc * slk_v_for_rsrc * fx.Index(NUM_KV_HEADS * HEAD_DIM)
        )
        k_rsrc = buffer_ops.create_buffer_resource(
            K, max_size=False, num_records_bytes=k_total_bytes,
        )
        v_rsrc = buffer_ops.create_buffer_resource(
            V, max_size=False, num_records_bytes=v_total_bytes,
        )
        # K_descale buffer (uint8 e8m0), shape [B, S, Hkv, D//32].
        k_descale_total_bytes = (
            bs_v_for_rsrc * slk_v_for_rsrc
            * fx.Index(NUM_KV_HEADS * NUM_SCALE_GROUPS)
        )
        k_descale_rsrc = buffer_ops.create_buffer_resource(
            K_descale, max_size=False, num_records_bytes=k_descale_total_bytes,
        )

        v4i32_type = Vec.make_type(4, fx.Int32)
        v16i32_type = Vec.make_type(16, fx.Int32)
        v16f32_type = Vec.make_type(16, fx.Float32)
        # MFMA 32x32x32_i8: A/B = v4i32 (=16xi8 per lane), C/D = v16i32 per lane (wave64)
        v8i8_type = Vec.make_type(8, fx.Int8)
        v16i8_type = Vec.make_type(16, fx.Int8)

        seq_len_q_v = fx.Index(seq_len_q)
        seq_len_k_v = fx.Index(seq_len_k)

        # Bias buffer (f32), shape [B, Hq, Q_NUM_BLKS, S_k]. Q_NUM_BLKS uses
        # BLOCK_M (the kernel's q-tile dim, matching Triton's BLKQ=BLOCK_M).
        # Only created when USE_BIAS — otherwise the wrapper passes a dummy
        # 1-byte tensor and the buffer-load code below is never emitted.
        if const_expr(USE_BIAS):
            num_q_tiles_idx_b = (
                (seq_len_q_v + fx.Index(BLOCK_M - 1)) // fx.Index(BLOCK_M)
            )
            bias_total_bytes = (
                bs_v_for_rsrc
                * fx.Index(NUM_Q_HEADS)
                * num_q_tiles_idx_b
                * seq_len_k_v
                * fx.Index(4)
            )
            bias_rsrc = buffer_ops.create_buffer_resource(
                Bias, max_size=False, num_records_bytes=bias_total_bytes,
            )

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
            """Q is FP4 packed: token stride = NUM_Q_HEADS * HEAD_DIM_BYTES."""
            token = batch_idx * seq_len_q_v + token_idx
            return token * Q_STRIDE_TOKEN + head_q_idx * HEAD_DIM_BYTES + col

        def k_global_idx(token_idx, col):
            """K is FP4 packed."""
            token = batch_idx * seq_len_k_v + token_idx
            return token * K_STRIDE_TOKEN + head_kv_idx * HEAD_DIM_BYTES + col

        def v_global_idx(token_idx, col):
            """V is FP8."""
            token = batch_idx * seq_len_k_v + token_idx
            return token * V_STRIDE_TOKEN + head_kv_idx * HEAD_DIM + col

        def o_global_idx(token_idx, col):
            """O is bf16. col is in element units."""
            token = batch_idx * seq_len_q_v + token_idx
            # bf16 output: NUM_Q_HEADS * HEAD_DIM elements per token row
            return token * (NUM_Q_HEADS * HEAD_DIM) + head_q_idx * HEAD_DIM + col

        def k_descale_global_idx(token_idx, scale_group):
            """K_descale shape [B, S, Hkv, D//32], 1 byte per scale group."""
            token = batch_idx * seq_len_k_v + token_idx
            return (token * NUM_KV_HEADS + head_kv_idx) * NUM_SCALE_GROUPS + scale_group

        def q_descale_global_idx(token_idx, scale_group):
            """Q_descale shape [B, S, Hq, D//32], 1 byte per scale group."""
            token = batch_idx * seq_len_q_v + token_idx
            return (token * NUM_Q_HEADS + head_q_idx) * NUM_SCALE_GROUPS + scale_group

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

        # ---- Preload Q to registers (FP4 packed) ----
        # Each wave owns ROWS_PER_WAVE=32 Q rows.
        # For FP4 32x32x64: each MFMA call processes K=32 elements (one scale
        # group). Per lane = 8 bytes = 16 nibbles for K=group*32 + klane*16..+15.
        # We load K_STEPS_QK packs per lane, each = 8 bytes (v2i32 wrapped as
        # v8i32 with upper bytes zeroed at MFMA-call time).
        v2i32_type = Vec.make_type(2, fx.Int32)
        v8i32_type = Vec.make_type(8, fx.Int32)
        v8i8_type_l = Vec.make_type(8, fx.Int8)

        def _load_ptr_i8_vec8_local(ptr, base_idx):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.i8)
            return _pointer_load(v8i8_type_l, gep)

        q_row = q_start + wave_q_offset + lane32
        q_row_i32 = fx.Int32(q_row)
        q_in_bounds = arith.cmpi(
            arith.CmpIPredicate.slt, _raw(q_row), _raw(seq_len_q_v)
        )
        q_row_safe = fx.Index(ArithValue(q_in_bounds).select(q_row, fx.Index(0)))

        _zero_v2i32_ir = Vec.filled(2, 0, fx.Int32).ir_value()
        q_packs = []  # one v2i32 per scale group (= 8 bytes = 16 K-nibbles per lane)
        for ks in range_constexpr(K_STEPS_QK):
            # ks = scale group index (0..3 for head_dim=128).
            # K-byte range: ks*16 + klane*8 .. + 7 (8 bytes per lane).
            q_col_byte = fx.Index(ks * 16) + klane * 8
            g_idx = q_global_idx(q_row_safe, q_col_byte)
            v8i8 = _load_ptr_i8_vec8_local(q_ptr, g_idx)
            v2i32 = vector.bitcast(v2i32_type, v8i8)
            q_packs.append(ArithValue(q_in_bounds).select(v2i32, _zero_v2i32_ir))

        # ---- Q-scale (e8m0) preload: 4 bytes per lane = 1 i32 per (q_row) ----
        # q_descale shape: [B, S, Hq, D//32 = 4]. One i32 per q_row covers all
        # 4 scale groups at once. Load once per kernel and reuse every kv iter.
        # opselA selects which byte (= which scale group) the MFMA uses.
        # Address: q_descale_global_idx(q_row_safe, 0) gives byte 0 (group 0).
        # Load 4 contiguous bytes as a single i32.
        q_scale_base_byte = q_descale_global_idx(q_row_safe, fx.Index(0))
        q_scale_gep = buffer_ops.get_element_ptr(
            qds_ptr, fx.Int64(q_scale_base_byte), elem_type=T.i8
        )
        q_scale_i32 = _pointer_load(T.i32, q_scale_gep)
        # Out-of-bounds q_rows → use 0x7F7F7F7F (×1.0 scale, harmless given
        # q_in_bounds gate elsewhere)
        _scale_one_i32 = arith.constant(0x7F7F7F7F, type=T.i32)
        q_scale_i32 = ArithValue(q_in_bounds).select(
            q_scale_i32, _scale_one_i32
        )

        # v_descale per-channel f32, indexed [B, Hkv, D]
        v_descale_base = (batch_idx * NUM_KV_HEADS + head_kv_idx) * HEAD_DIM

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
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
            kv_upper = fx.Index(
                ArithValue(_q_end < seq_len_k_v).select(_q_end, seq_len_k_v)
            )
        else:
            kv_upper = seq_len_k_v

        # ---- Tile-split: count fully-unmasked tiles vs masked-tail tiles. ----
        # Body loop processes tiles where every (Q-row, K-col) pair is valid:
        #   - K-col never exceeds seq_len_k (no seq-len tail)
        #   - K-col never exceeds q_row (no causal diagonal)
        # Tail loop covers remaining tiles up to kv_upper and runs the existing
        # masked path. Mirrors Triton's _sage_fwd_no_mask / _sage_fwd_mask split.
        #
        # n_full = min(in_range_full_blocks, fully_below_diag_blocks if causal)
        #   in_range_full_blocks = seq_len_k // BLOCK_N
        #   fully_below_diag_blocks = q_start // BLOCK_N (BLOCK_M >= BLOCK_N)
        BLOCK_N_IDX = fx.Index(BLOCK_N)
        in_range_full_blocks = seq_len_k_v // BLOCK_N_IDX
        if const_expr(CAUSAL):
            fully_below_diag_blocks = q_start // BLOCK_N_IDX
            n_full_blocks_idx = fx.Index(
                ArithValue(fully_below_diag_blocks < in_range_full_blocks).select(
                    fully_below_diag_blocks, in_range_full_blocks
                )
            )
        else:
            n_full_blocks_idx = in_range_full_blocks
        kv_body_end = n_full_blocks_idx * BLOCK_N_IDX

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
                    g_idx = k_global_idx(row_idx_raw, load_col_k_base)
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
                    g_idx = v_global_idx(row_idx_raw, load_col_v_base)
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

        def load_k_frag_fp4(kv_block_row, ks, buf_off):
            """Load v2i32 (=8xi8 = 16 nibbles) K fragment for FP4 MFMA.

            kv_block_row: row within BLOCK_N tile (0..BLOCK_N-1)
            ks: scale-group index (0..K_STEPS_QK-1); klane picks the 8-byte half.
            buf_off: ping-pong K-buffer byte offset.
            """
            k_col_byte = fx.Index(ks * 16) + klane * 8
            lds_idx = buf_off + fx.Index(kv_block_row * K_STRIDE) + k_col_byte
            v8i8 = Vec.load(v8i8_type_l, lds, [lds_idx])
            return vector.bitcast(v2i32_type, v8i8)

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
                    lit_bytes = (pks * MFMA_K_FP8 + k_idx * 8) * V_STRIDE + dc * MFMA_M
                    addr_i32 = arith.AddIOp(
                        iter_lane_addr_i32,
                        arith.constant(lit_bytes, type=T.i32),
                    ).result
                    ptr_val = _llvm.inttoptr(lds_ptr_ty, addr_i32)
                    v2i32 = _ods_ds_read_tr8_b64(res=v2i32_type, ptr=ptr_val).result
                    words.append(v2i32)

                # Pack 4× v2i32 → v8i32 (= 32 fp8 bytes per lane).
                elems = []
                for w_idx in range_constexpr(4):
                    for sub in range_constexpr(2):
                        elems.append(
                            vector.extract(
                                words[w_idx], static_position=[sub], dynamic_position=[]
                            )
                        )
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
                elems.append(
                    vector.extract(lo_v4i32, static_position=[w], dynamic_position=[])
                )
            for w in range_constexpr(4):
                elems.append(
                    vector.extract(hi_v4i32, static_position=[w], dynamic_position=[])
                )
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
            _per_lane_v_inv = (
                (_klane_g * fx.Index(32) + _row_off_within) * fx.Index(V_STRIDE)
                + _d_group * fx.Index(16)
                + _col_extra_within
            )
            _lds_v_base = _memref.extract_aligned_pointer_as_index(_llvm_value(lds_v))
            _v_lane_base_index = _AV(_lds_v_base) + _per_lane_v_inv
            v_lane_base_i32 = arith.unwrap(arith.index_cast(T.i32, _v_lane_base_index))

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

        def _emit_qk_softmax_pquant(
            kv_block_start_arg, k_buf_off_arg, m_in, l_in, mask_mode="full"
        ):
            """Emit QK FP4 MFMA + (mask) + softmax + p-quant for the KV tile
            at kv_block_start_arg.

            mask_mode (Python-level constant — controls IR generation):
              - "none" : no causal mask, no seq-len-tile-end mask, no bias-OOB
                         bounds-check. Used by the body loop where the caller
                         guarantees every (Q-row, K-col) is in range.
              - "full" : all masking active (current behavior). Used by the
                         tail loop, which covers the diagonal/seq-len boundary.

            FP4 MFMA semantics (probe-validated 2026-05-13):
              - mfma_scale_f32_32x32x64_f8f6f4 with cbsz=4, blgp=4 processes
                K=64 elements per call, with ONE e8m0 scale (selected by
                opsel{A,B} from 4 bytes of i32 scale{A,B}).
              - For per-32-group MXFP4 scaling, we issue K_STEPS_QK calls per
                (q-tile, n-subtile), each with K=32 active (one scale group).
                Per lane: lower 8 bytes of v8i32 hold 16 nibbles = 16 K-elems
                for klane=0, klane=1 holds the next 16 K-elems → K=32 active.
                Upper 24 bytes are zero → no extra K-contribution.
              - opselA=ks, opselB=ks selects the matching scale byte for both
                operands. q_scale_i32 was preloaded at kernel start;
                k_scale_i32[st] is loaded per-st inside this function.
              - **NO post-MFMA scale multiply** (scales applied inside MFMA),
                vs sage v1 which multiplied by qk_scale = q_ds * k_ds.

            Returns (m_new, l_new, corr, p_words_2d) — IR values.
            """
            v8i32_t = Vec.make_type(8, fx.Int32)
            _zero_i32 = arith.constant(0, type=T.i32)

            def _v2_to_v8(v2):
                """Pad v2i32 → v8i32 by zero-extending upper 6 i32 elements."""
                e0 = vector.extract(v2, static_position=[0], dynamic_position=[])
                e1 = vector.extract(v2, static_position=[1], dynamic_position=[])
                return Vec.from_elements(
                    [e0, e1, _zero_i32, _zero_i32, _zero_i32, _zero_i32, _zero_i32, _zero_i32],
                    fx.Int32,
                ).ir_value()

            # ---- Load per-(kv_row) k_scale i32 packs, one per N-subtile ----
            # k_descale shape [B, S, Hkv, D//32 = 4]. Per lane: kv_row =
            # kv_block_start + lane32 + st*32. Load 4 e8m0 bytes into i32.
            _scale_one_i32_l = arith.constant(0x7F7F7F7F, type=T.i32)
            k_scale_i32_per_st = []
            for st in range_constexpr(N_SUBTILES):
                kv_row_for_lane = kv_block_start_arg + lane32 + fx.Index(st * MFMA_N)
                kv_in_bounds_l = arith.cmpi(
                    arith.CmpIPredicate.slt,
                    _raw(kv_row_for_lane),
                    _raw(seq_len_k_v),
                )
                kv_row_safe_l = fx.Index(
                    ArithValue(kv_in_bounds_l).select(
                        kv_row_for_lane, fx.Index(0)
                    )
                )
                k_scale_byte_idx = k_descale_global_idx(kv_row_safe_l, fx.Index(0))
                # Load 4 bytes as i32 via buffer_load with i32 dtype
                k_scale_dword_idx = arith.unwrap(
                    arith.index_cast(T.i32, _raw(k_scale_byte_idx >> fx.Index(2)))
                )
                k_scale_i32_lane = buffer_ops.buffer_load(
                    k_descale_rsrc, k_scale_dword_idx, vec_width=1, dtype=T.i32,
                )
                # Out-of-bounds → use ×1.0 (0x7F7F7F7F). Because of buffer
                # descriptor's tight num_records_bytes, OOB returns 0; we want
                # 0x7F instead so the score stays valid (then masked via mask).
                k_scale_i32_lane = ArithValue(kv_in_bounds_l).select(
                    k_scale_i32_lane, _scale_one_i32_l
                )
                k_scale_i32_per_st.append(k_scale_i32_lane)

            # ---- 1) QK FP4 MFMA: K_STEPS_QK calls per (st), each one scale group ----
            # MFMA operand order: A=K, B=Q (mirrors sage v1 INT8 convention).
            # Output S[m=K-row, n=Q-row]: lane32=Q-row, vector elems span K-rows.
            # This matches the downstream mask/softmax/P-quant code that
            # encodes the K-col index via (msub, klane, erem) of the vector
            # elements and reduces per-Q-row max across klane (XOR 32).
            s_accs_loc = [_raw(c_zero_v16f32) for _ in range(N_SUBTILES)]
            for ks in range_constexpr(K_STEPS_QK):
                # Pack q_packs[ks] (v2i32 = 8 active bytes per lane) into v8i32
                # with upper 24 bytes zero.
                q_v8 = _v2_to_v8(q_packs[ks])
                for st in range_constexpr(N_SUBTILES):
                    kv_row_loc = lane32 + st * MFMA_N
                    k_v2 = load_k_frag_fp4(kv_row_loc, ks, k_buf_off_arg)
                    k_v8 = _v2_to_v8(k_v2)
                    # A=K (M=K-row=lane32), B=Q (N=Q-row=lane32). Scale per
                    # operand: scaleA = K-row scale, scaleB = Q-row scale.
                    s_accs_loc[st] = mfma_fp4_scaled(
                        v16f32_type,
                        k_v8, q_v8, s_accs_loc[st],
                        k_scale_i32_per_st[st], q_scale_i32,
                        ks, ks,
                    )

            # 2) Score post-processing.
            # The MFMA already applies the per-32-group dequantization scales
            # via scaleA/scaleB. sm_scale * log2(e) is baked into Q by the
            # Triton rotation_smooth_qk quantizer (see sage_quant_mxfp4 →
            # rotation_smooth_qk(sm_scale=sm_scale*1.4426950408889634)).
            # No post-MFMA multiply is needed — Triton's reference also
            # skips it (see _attn_fwd_inner_full at ~_triton_kernels/.../
            # fav3_sage_attention_mxfp4.py:225 — straight exp2(qk - max)).
            s_f32_loc = []
            for st in range_constexpr(N_SUBTILES):
                s_vec = Vec(s_accs_loc[st])
                for elem in range_constexpr(ELEMS_PER_TILE):
                    s_f32_loc.append(s_vec[elem])

            # 2b) Bias add (q_smoothing path: delta_s = q_mean × K_rot).
            # Triton applies bias[None, :] to qk before softmax (per K-col,
            # broadcast across Q rows). Bias shape: [B, Hq, Q_NUM_BLKS, S_k].
            # Per lane (klane, lane32), each vector elem (msub, erem) of st-th
            # subtile maps to K-col = kv_block_start + st*32 + msub*8 +
            # klane*4 + erem.
            #
            # Per-element OOB bounds-check (kv_col<seq_len_k) gated on CAUSAL.
            # Empirically (measured 2026-05-13) the cmpi+select pair shifts
            # LLVM's instruction schedule in opposite directions for causal
            # vs non-causal kernels:
            #   - non-causal q_smooth=True (single loop, mask_mode='full' on
            #     last tile): the predicate helps LLVM schedule the bias
            #     buffer_load latency. Dropping it: S=32768 fwd 2.48x → 1.64x
            #     Triton (-34%). KEEP the check.
            #   - causal q_smooth=True (body+tail split, body mask_mode='none'
            #     covers most iters): the predicate is provably redundant
            #     in the body and apparently blocks better scheduling.
            #     Dropping it: S=16384 caus 1.67x → 2.58x Triton (+54%).
            #     DROP the check.
            if const_expr(USE_BIAS):
                # Bias base byte offset for (B, Hq, Q_NUM_BLKS, *):
                #   ((batch * NUM_Q_HEADS + head_q) * num_q_tiles + q_tile) * S_k * 4
                num_q_tiles_loc = (
                    (seq_len_q_v + fx.Index(BLOCK_M - 1)) // fx.Index(BLOCK_M)
                )
                bias_base_dword = (
                    (batch_idx * fx.Index(NUM_Q_HEADS) + head_q_idx)
                    * num_q_tiles_loc + q_tile_idx
                ) * seq_len_k_v
                s_bias = []
                for st in range_constexpr(N_SUBTILES):
                    for elem in range_constexpr(ELEMS_PER_TILE):
                        idx = st * ELEMS_PER_TILE + elem
                        msub = elem // 4
                        erem = elem % 4
                        kv_col_idx = (
                            kv_block_start_arg
                            + fx.Index(st * MFMA_N)
                            + fx.Index(msub * 8)
                            + klane * 4
                            + fx.Index(erem)
                        )
                        if const_expr(not CAUSAL):
                            # Non-causal: keep the bounds-check (helps LLVM scheduling)
                            kv_in_bias_bounds = ArithValue(
                                arith.cmpi(
                                    arith.CmpIPredicate.slt,
                                    _raw(kv_col_idx),
                                    _raw(seq_len_k_v),
                                )
                            )
                            kv_col_safe = fx.Index(
                                kv_in_bias_bounds.select(kv_col_idx, fx.Index(0))
                            )
                            bias_dword_idx = arith.unwrap(
                                arith.index_cast(
                                    T.i32, _raw(bias_base_dword + kv_col_safe)
                                )
                            )
                            bias_val = buffer_ops.buffer_load(
                                bias_rsrc, bias_dword_idx,
                                vec_width=1, dtype=T.f32,
                            )
                            bias_val_safe = kv_in_bias_bounds.select(
                                bias_val, fx.Float32(0.0)
                            )
                            s_bias.append(_fadd(s_f32_loc[idx], bias_val_safe))
                        else:
                            # Causal: drop the bounds-check. In body the bound
                            # is provably true; in tail the score mask kills
                            # OOB cols via -inf, so any garbage bias added is
                            # harmless. Skipping the check unblocks better
                            # LLVM scheduling for the causal kernel.
                            bias_dword_idx = arith.unwrap(
                                arith.index_cast(
                                    T.i32, _raw(bias_base_dword + kv_col_idx)
                                )
                            )
                            bias_val = buffer_ops.buffer_load(
                                bias_rsrc, bias_dword_idx,
                                vec_width=1, dtype=T.f32,
                            )
                            s_bias.append(_fadd(s_f32_loc[idx], bias_val))
                s_f32_loc = s_bias

            # 3) Mask — only emitted in mask_mode='full'.
            # In mask_mode='none' the body loop guarantees every (Q-row, K-col)
            # is in range (no causal diagonal, no seq-len overflow), so the
            # entire mask code path is statically removed.
            if const_expr(mask_mode == "full"):
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
                        cond_i1,
                        result_types,
                        has_else=True,
                        loc=ir.Location.unknown(),
                    )
                    with ir.InsertionPoint(if_op.then_block):
                        _llvm.inline_asm(
                            None,
                            [],
                            "",
                            "",
                            has_side_effects=True,
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
                                masked.append(
                                    _raw(out_of_range.select(c_neg_inf, s_f32_loc[idx]))
                                )
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

        # Hoist module imports outside the loop body for reuse.
        from flydsl._mlir.dialects._rocdl_ops_gen import (
            permlane32_swap as _permlane32_swap_op,
        )

        _struct_ty_2xi32 = ir.Type.parse("!llvm.struct<(i32, i32)>")

        def _emit_iter(kv_block_start, inner_iter_args, mask_mode):
            """Emit one KV-tile iteration: QK MFMA + softmax + PV MFMA + prefetch.

            mask_mode is a Python-level constant ('none' or 'full') passed
            through to _emit_qk_softmax_pquant. Returns the yield_args list.
            """
            cur_buf_i32 = inner_iter_args[_OFF_CUR_BUF]
            m_running = inner_iter_args[_OFF_M]
            l_running = inner_iter_args[_OFF_L]
            o_accs = [
                inner_iter_args[_OFF_O_ACCS + i] for i in range_constexpr(D_CHUNKS)
            ]

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

            # ==== PRE_LOAD_V (q_smooth=False only): hoist all V fragments
            # before the QK MFMA chain so 32 ds_read_tr8_b64 issue up-front
            # and their LDS latency is hidden behind the long QK MFMA chain.
            # Mirrors Triton's PRE_LOAD_V=True path
            # (fav3_sage_attention_mxfp4.py:171, 196-202). Trades ~64 vgpr/lane
            # of register pressure for better ILP.
            #
            # Disabled when USE_BIAS=True (q_smooth=True): the bias path
            # already adds significant register pressure (per-element bias
            # loads and adds across 64 vector elements), so pre-loading V
            # spills VGPRs and regresses q_smooth=True forward shapes by
            # 10-16% (measured 2026-05-13). The post-load V fetch path is
            # kept for q_smooth=True since LLVM's scheduler already pipelines
            # ds_reads with PV MFMA effectively when registers are tight.
            if const_expr(not USE_BIAS):
                v_frags_pre = [[None] * D_CHUNKS for _ in range(PV_K_STEPS)]
                for pks in range_constexpr(PV_K_STEPS):
                    for dc in range_constexpr(D_CHUNKS):
                        v_frags_pre[pks][dc] = load_v_frag_fp8(
                            pks, dc, cur_v_off, v_iter_lane_addr_i32
                        )

            # ==== Compute softmax[k] using cur_buf K ====
            m_new, l_new, corr, p_words = _emit_qk_softmax_pquant(
                kv_block_start, cur_k_off, m_running, l_running, mask_mode=mask_mode
            )

            # ==== Apply correction factor to o_accs (in-iter) ====
            corr_vec16 = (
                Vec.from_elements([corr], fx.Float32).broadcast_to(16).ir_value()
            )
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = _fmul(o_accs[dc], corr_vec16)

            # ==== GEMM2: PV[k] using p_words[k] and V (pre-loaded if available) ====
            for pks in range_constexpr(PV_K_STEPS):
                v8_elems = []
                for w in range_constexpr(4):
                    a_w = _raw(p_words[pks * 2][w])
                    b_w = _raw(p_words[pks * 2 + 1][w])
                    swapped = _permlane32_swap_op(
                        _struct_ty_2xi32,
                        old=a_w,
                        src=b_w,
                        fi=False,
                        bound_control=True,
                    )
                    lo_word = _llvm.extractvalue(T.i32, swapped, [0])
                    hi_word = _llvm.extractvalue(T.i32, swapped, [1])
                    v8_elems.append(lo_word)
                    v8_elems.append(hi_word)
                p_pack_v8i32 = Vec.from_elements(v8_elems, fx.Int32).ir_value()
                scale_127 = arith.constant(127, type=T.i32)
                for dc in range_constexpr(D_CHUNKS):
                    if const_expr(not USE_BIAS):
                        v_frag = v_frags_pre[pks][dc]
                    else:
                        v_frag = load_v_frag_fp8(
                            pks, dc, cur_v_off, v_iter_lane_addr_i32
                        )
                    o_accs[dc] = mfma_fp8_k64(
                        v16f32_type,
                        v_frag,
                        p_pack_v8i32,
                        o_accs[dc],
                        scale_127,
                        scale_127,
                    )

            # Barrier + prefetch K[k+2]/V[k+2] into cur_buf (overwriting K[k]/V[k]).
            gpu.barrier()

            kv_block_after_next = kv_block_start + fx.Index(2 * BLOCK_N)
            coop_load_k(kv_block_after_next, cur_k_off)
            coop_load_v(kv_block_after_next, cur_v_off)

            # ==== Yield ====
            _yield_args = [next_buf_i32, _raw(m_new), _raw(l_new)]
            for dc in range_constexpr(D_CHUNKS):
                _yield_args.append(o_accs[dc])
            return _yield_args

        # Causal: emit body+tail split. Body covers tiles fully below the
        # diagonal (no mask, no scf.if), tail covers the diagonal stripe with
        # the full mask path. Big causal win.
        # Non-causal: keep a single loop with mask_mode='full' (the seq-len
        # tile-end mask is a runtime scf.if predicate that's false on every
        # interior tile, so it costs ~one branch per iter — emitting a body+
        # tail split here introduced a small fwd regression on H24 large-S
        # shapes via duplicated loop bodies and iter_args plumbing).
        if const_expr(CAUSAL):
            body_results = init_args
            for kv_block_start, inner_iter_args in range(
                0, kv_body_end, BLOCK_N, init=init_args
            ):
                body_results = yield _emit_iter(
                    kv_block_start, inner_iter_args, mask_mode="none"
                )
            loop_results = body_results
            for kv_block_start, inner_iter_args in range(
                kv_body_end, kv_upper, BLOCK_N, init=body_results
            ):
                loop_results = yield _emit_iter(
                    kv_block_start, inner_iter_args, mask_mode="full"
                )
        else:
            loop_results = init_args
            for kv_block_start, inner_iter_args in range(
                0, kv_upper, BLOCK_N, init=init_args
            ):
                loop_results = yield _emit_iter(
                    kv_block_start, inner_iter_args, mask_mode="full"
                )

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
                        v_ds = fx.Float32(
                            _load_ptr_f32(
                                vds_ptr,
                                v_descale_base
                                + fx.Index(dc * MFMA_M)
                                + fx.Index(msub * 8)
                                + klane * 4
                                + erem,
                            )
                        )
                        o_elem = vector.extract(
                            o_finals[dc], static_position=[i_pos], dynamic_position=[]
                        )
                        scale = arith.mulf(
                            _raw(inv_l_fp8), _raw(v_ds), fastmath=fm_fast
                        )
                        o_norm = arith.mulf(o_elem, scale, fastmath=fm_fast)
                        bf16_elems.append(f32_to_bf16_trunc(o_norm))
                    o_vec = Vec.from_elements(bf16_elems, fx.BFloat16).ir_value()
                    o_global = o_global_idx(q_row, d_col_base)
                    _store_ptr_bf16(o_ptr, o_global, o_vec)

    @flyc.jit
    def launch_sage_attn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        Q_descale: fx.Tensor,
        K_descale: fx.Tensor,
        V_descale: fx.Tensor,
        Bias: fx.Tensor,
        batch_size: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
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
            Q,
            K,
            V,
            O,
            Q_descale,
            K_descale,
            V_descale,
            Bias,
            batch_size,
            seq_len_q,
            seq_len_k,
        )

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

    _compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {"enable-post-misched": True, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_compile_hints):
            return launch_sage_attn(*args, **kwargs)

    def _compile(
        Q,
        K,
        V,
        O,  # noqa: E741
        Q_descale,
        K_descale,
        V_descale,
        batch_size,
        seq_len_q,
        seq_len_k,
        stream=None,
    ):
        with CompilationContext.compile_hints(_compile_hints):
            return flyc.compile(
                launch_sage_attn,
                Q,
                K,
                V,
                O,
                Q_descale,
                K_descale,
                V_descale,
                batch_size,
                seq_len_q,
                seq_len_k,
                fx.Stream(stream),
            )

    _launch.compile = _compile
    return _launch
