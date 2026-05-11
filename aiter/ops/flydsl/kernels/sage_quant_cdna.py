# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL kernels for SageAttention V1 quantization (CDNA gfx942/gfx950).

This module exposes two builders:

* ``build_sage_quant_v_module`` — V FP8 (e4m3fn) per-channel quant only.
  One-launch standalone kernel. Used by Stage 0 of the FlyDSL sage_quant
  port; kept for reference / fallback.

* ``build_sage_quant_fused_module`` — fused single-launch kernel that
  performs Q INT8 + K INT8 + V FP8 quant in one shader, dispatching
  per-block on bid (matches Triton's ``sage_quant_kernel`` structure at
  ``aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py:847``).
  This is the path the production wrapper uses to beat Triton end-to-end:
  one launch instead of three, eliminating the ~10 us-per-launch Python
  overhead that dominated Stage 0's per-call cost.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, range_constexpr, const_expr, gpu
from flydsl.expr.typing import T, Int32, Float32, Vector as Vec
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf, rocdl
from flydsl._mlir.dialects import scf as _scf
from flydsl.expr import buffer_ops
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch


BLOCK_THREADS = 256
WARP_SIZE = 64
NUM_WAVES = BLOCK_THREADS // WARP_SIZE  # 4


# =========================================================================
# Standalone V FP8 kernel (Stage 0 — kept as reference / fallback)
# =========================================================================

def build_sage_quant_v_module(
    head_dim: int,
    blk_k: int,
    num_kv_heads: int,
):
    """Return a JIT launcher for V FP8 per-channel quantization (BSHD).

    Parameters
    ----------
    head_dim : int
        Last-dim size; must be divisible by VEC_V (8).
    blk_k : int
        KV block size (matches attention BLOCK_N — 128 or 256).
    num_kv_heads : int
        Number of KV heads (compile-time so the pid decode is cheap).
    """
    assert head_dim % 8 == 0, f"head_dim={head_dim} must be divisible by 8"
    VEC_V = 8
    THREADS_PER_ROW = head_dim // VEC_V
    assert BLOCK_THREADS % THREADS_PER_ROW == 0, (
        f"BLOCK_THREADS={BLOCK_THREADS} not divisible by THREADS_PER_ROW={THREADS_PER_ROW}"
    )
    ROWS_PER_PASS = BLOCK_THREADS // THREADS_PER_ROW
    assert blk_k % ROWS_PER_PASS == 0, (
        f"blk_k={blk_k} must be divisible by ROWS_PER_PASS={ROWS_PER_PASS}"
    )
    PASSES = blk_k // ROWS_PER_PASS

    elem_bytes_in = 2  # bf16

    @flyc.kernel
    def sage_quant_v_kernel(
        V_in: fx.Tensor,
        V_out: fx.Tensor,
        V_scale: fx.Tensor,
        seq_len: Int32,
        num_blocks_k: Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32

        c0_i32 = arith.constant(0, type=i32)
        c2_i32 = arith.constant(2, type=i32)
        c4_i32 = arith.constant(4, type=i32)
        c_threads_per_row = arith.constant(THREADS_PER_ROW, type=i32)
        c_blk_k = arith.constant(blk_k, type=i32)
        c_head_dim = arith.constant(head_dim, type=i32)
        c_kv_heads = arith.constant(num_kv_heads, type=i32)
        c_vec = arith.constant(VEC_V, type=i32)
        c_elem_in = arith.constant(elem_bytes_in, type=i32)

        bid_i32 = ArithValue(bid)
        tid_i32 = ArithValue(tid)

        num_blocks_k_i32 = ArithValue(num_blocks_k)
        blk_n_id = bid_i32 % num_blocks_k_i32
        bh_id = bid_i32 // num_blocks_k_i32
        h_id = bh_id % c_kv_heads
        b_id = bh_id // c_kv_heads

        row_base = tid_i32 // c_threads_per_row
        col_d = (tid_i32 % c_threads_per_row) * c_vec

        in_rsrc = buffer_ops.create_buffer_resource(V_in, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(V_out, max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(V_scale, max_size=True)

        scale_dw_base = (b_id * c_kv_heads + h_id) * c_head_dim + col_d
        v4f32_ty = T.vec(4, f32)

        scale_lo_raw = buffer_ops.buffer_load(
            scale_rsrc, scale_dw_base, vec_width=4, dtype=i32
        )
        scale_lo = vector.bitcast(v4f32_ty, scale_lo_raw)
        scale_hi_raw = buffer_ops.buffer_load(
            scale_rsrc, scale_dw_base + c4_i32, vec_width=4, dtype=i32
        )
        scale_hi = vector.bitcast(v4f32_ty, scale_hi_raw)

        v_inv_scales = []
        for vi in range_constexpr(4):
            s = vector.extract(scale_lo, static_position=[vi], dynamic_position=[])
            inv_s = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [s], [], [])
            v_inv_scales.append(inv_s)
        for vi in range_constexpr(4):
            s = vector.extract(scale_hi, static_position=[vi], dynamic_position=[])
            inv_s = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [s], [], [])
            v_inv_scales.append(inv_s)

        seq_len_i32 = ArithValue(seq_len)

        vec_bf16_ty = T.vec(VEC_V, T.bf16)
        vec_f32_ty = T.vec(VEC_V, f32)

        for p in range_constexpr(PASSES):
            row_in_blk = row_base + arith.constant(p * ROWS_PER_PASS, type=i32)
            global_row = blk_n_id * c_blk_k + row_in_blk

            row_in_range = arith.cmpi(CmpIPredicate.ult, global_row, seq_len_i32)
            _if_row = scf.IfOp(row_in_range)
            with ir.InsertionPoint(_if_row.then_block):
                row_elem_off = (
                    (b_id * seq_len_i32 + global_row) * c_kv_heads + h_id
                ) * c_head_dim + col_d
                in_dw_off = row_elem_off >> arith.constant(1, type=i32)
                raw = buffer_ops.buffer_load(
                    in_rsrc, in_dw_off, vec_width=4, dtype=i32
                )
                bf16_v = vector.bitcast(vec_bf16_ty, raw)
                f32_v = bf16_v.extf(vec_f32_ty)

                scaled_vals = []
                for vi in range_constexpr(VEC_V):
                    x = vector.extract(
                        f32_v, static_position=[vi], dynamic_position=[]
                    )
                    scaled_vals.append(x * v_inv_scales[vi])

                out_byte_off = row_elem_off
                for _wg in range_constexpr(VEC_V // 4):
                    _b = _wg * 4
                    packed_w = c0_i32
                    packed_w = rocdl.cvt_pk_fp8_f32(
                        i32,
                        scaled_vals[_b],
                        scaled_vals[_b + 1],
                        packed_w,
                        0,
                    )
                    packed_w = rocdl.cvt_pk_fp8_f32(
                        i32,
                        scaled_vals[_b + 2],
                        scaled_vals[_b + 3],
                        packed_w,
                        1,
                    )
                    word_off = out_byte_off + arith.constant(_wg * 4, type=i32)
                    buffer_ops.buffer_store(
                        packed_w,
                        out_rsrc,
                        word_off,
                        offset_is_bytes=True,
                    )

                scf.YieldOp([])

    @flyc.jit
    def launch_sage_quant_v(
        V_in: fx.Tensor,
        V_out: fx.Tensor,
        V_scale: fx.Tensor,
        seq_len: fx.Int32,
        num_blocks_k: fx.Int32,
        grid_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        idx_grid = arith.index_cast(T.index, grid_size)
        launcher = sage_quant_v_kernel(V_in, V_out, V_scale, seq_len, num_blocks_k)
        launcher.launch(
            grid=(idx_grid, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_sage_quant_v


# =========================================================================
# Fused Q+K+V single-launch kernel
# =========================================================================

def build_sage_quant_fused_module(
    head_dim: int,
    blk_q: int,
    blk_k: int,
    num_q_heads: int,
    num_kv_heads: int,
):
    """Build a single fused kernel doing Q INT8 + K INT8 + V FP8 quant.

    Launch grid = q_count + k_count + v_count where:
      q_count = B * num_q_heads  * num_q_blocks
      k_count = B * num_kv_heads * num_k_blocks
      v_count = B * num_kv_heads * num_k_blocks  (V uses the K block size)

    Per-block dispatch:
      bid in [0, q_count)                            -> Q INT8 quant
      bid in [q_count, q_count + k_count)            -> K INT8 quant
      bid in [q_count + k_count, q_count + 2*k_count)-> V FP8 quant

    BSHD layout only. sm_scale * log2(e) is applied to Q before the amax
    so the int8 dequant during attention gives Q*K * sm_scale*log2e
    directly (matching Triton sage_quant; the FlyDSL attention kernel
    consumes pre-baked scales).
    """
    assert head_dim % 8 == 0
    assert blk_q in (128, 256), f"unexpected BLK_Q={blk_q}"
    assert blk_k in (64, 128, 256), f"unexpected BLK_K={blk_k}"

    VEC = 8
    THREADS_PER_ROW = head_dim // VEC               # 16 for head_dim=128
    assert BLOCK_THREADS % THREADS_PER_ROW == 0
    ROWS_PER_PASS = BLOCK_THREADS // THREADS_PER_ROW  # 16
    assert blk_q % ROWS_PER_PASS == 0
    assert blk_k % ROWS_PER_PASS == 0
    PASSES_Q = blk_q // ROWS_PER_PASS
    PASSES_K = blk_k // ROWS_PER_PASS
    PASSES_V = PASSES_K

    INT8_MAX_F = 127.0
    INV_INT8_MAX = 1.0 / INT8_MAX_F

    # LDS scratch: NUM_WAVES f32 slots for the cross-wave amax reduction +
    # broadcast slot for inv_scale. We reuse slot 0 for the broadcast after
    # the reduction.
    LDS_F32_SLOTS = NUM_WAVES  # 4
    LDS_BYTES = LDS_F32_SLOTS * 4

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=(
            f"sage_quant_fused_smem_h{head_dim}_q{blk_q}_k{blk_k}"
            f"_hq{num_q_heads}_hk{num_kv_heads}"
        ),
    )
    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + LDS_BYTES

    @flyc.kernel
    def sage_quant_fused_kernel(
        Q_in: fx.Tensor,    # bf16 [B, S_q, H_q, D]
        Q_out: fx.Tensor,   # int8 [B, S_q, H_q, D]
        Q_scale: fx.Tensor, # f32  [B, H_q, NUM_BLKS_Q]
        K_in: fx.Tensor,    # bf16 [B, S_k, H_k, D]  (already smoothed)
        K_out: fx.Tensor,   # int8 [B, S_k, H_k, D]
        K_scale: fx.Tensor, # f32  [B, H_k, NUM_BLKS_K]
        V_in: fx.Tensor,    # bf16 [B, S_k, H_k, D]
        V_out: fx.Tensor,   # fp8  [B, S_k, H_k, D]
        V_scale: fx.Tensor, # f32  [B, H_k, D]
        seq_len_q: Int32,
        seq_len_k: Int32,
        num_blocks_q: Int32,
        num_blocks_k: Int32,
        q_task_count: Int32,
        k_task_count: Int32,
        sm_scale_log2e_bits: Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32
        i8 = T.i8

        bid_i32 = ArithValue(bid)
        tid_i32 = ArithValue(tid)

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c4_i32 = arith.constant(4, type=i32)
        c8_i32 = arith.constant(8, type=i32)
        c16_i32 = arith.constant(16, type=i32)
        c24_i32 = arith.constant(24, type=i32)
        c32_i32 = arith.constant(32, type=i32)
        c0xFF_i32 = arith.constant(0xFF, type=i32)
        c_warp_size_i32 = arith.constant(WARP_SIZE, type=i32)
        c_num_waves_i32 = arith.constant(NUM_WAVES, type=i32)
        c_threads_per_row = arith.constant(THREADS_PER_ROW, type=i32)
        c_head_dim = arith.constant(head_dim, type=i32)
        c_q_heads = arith.constant(num_q_heads, type=i32)
        c_kv_heads = arith.constant(num_kv_heads, type=i32)
        c_blk_q = arith.constant(blk_q, type=i32)
        c_blk_k = arith.constant(blk_k, type=i32)
        c_vec = arith.constant(VEC, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)
        c_p_half = arith.constant(0.5, type=f32)
        c_n_half = arith.constant(-0.5, type=f32)
        c_inv_int8_max = arith.constant(INV_INT8_MAX, type=f32)
        c_int8_max_pos = arith.constant(127, type=i32)
        c_int8_max_neg = arith.constant(-127, type=i32)

        # Per-thread coordinates in a (BLK, D) tile
        row_base = tid_i32 // c_threads_per_row
        col_d = (tid_i32 % c_threads_per_row) * c_vec
        wave_id = tid_i32 // c_warp_size_i32
        lane = tid_i32 % c_warp_size_i32

        # LDS scratch (4 f32 = 16 bytes, NUM_WAVES slots; slot 0 doubles as
        # the inv_scale broadcast cell after the reduction)
        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_offset, T.f32, shape=(LDS_F32_SLOTS,)).get()

        # ---------- buffer resources (allocate once, reused per branch) ----------
        q_in_rsrc = buffer_ops.create_buffer_resource(Q_in, max_size=True)
        q_out_rsrc = buffer_ops.create_buffer_resource(Q_out, max_size=True)
        q_scale_rsrc = buffer_ops.create_buffer_resource(Q_scale, max_size=True)
        k_in_rsrc = buffer_ops.create_buffer_resource(K_in, max_size=True)
        k_out_rsrc = buffer_ops.create_buffer_resource(K_out, max_size=True)
        k_scale_rsrc = buffer_ops.create_buffer_resource(K_scale, max_size=True)
        v_in_rsrc = buffer_ops.create_buffer_resource(V_in, max_size=True)
        v_out_rsrc = buffer_ops.create_buffer_resource(V_out, max_size=True)
        v_scale_rsrc = buffer_ops.create_buffer_resource(V_scale, max_size=True)

        seq_len_q_i32 = ArithValue(seq_len_q)
        seq_len_k_i32 = ArithValue(seq_len_k)
        num_blocks_q_i32 = ArithValue(num_blocks_q)
        num_blocks_k_i32 = ArithValue(num_blocks_k)
        q_task_count_i32 = ArithValue(q_task_count)
        k_task_count_i32 = ArithValue(k_task_count)
        # sm_scale_log2e arrives as int32 (bit pattern of f32) for the fast
        # CallState slot path; reinterpret it back to f32 here.
        sm_scale_log2e_v = arith.bitcast(f32, ArithValue(sm_scale_log2e_bits))

        qk_task_count_i32 = q_task_count_i32 + k_task_count_i32

        vec_bf16_ty = T.vec(VEC, T.bf16)
        vec_f32_ty = T.vec(VEC, f32)
        v4f32_ty = T.vec(4, f32)

        SHUFFLE_DISTS = [32, 16, 8, 4, 2, 1]

        # ---------- helper: cross-wave amax reduction + broadcast inv_scale ----------
        def cross_wave_amax_to_inv_scale(local_max_v, store_scale_to_global, scale_off):
            """Reduce per-thread local_max across all 256 threads in the block,
            compute scale = block_amax / 127, write inv_scale to LDS[0],
            barrier, return the broadcast inv_scale read by every thread.

            ``store_scale_to_global`` is a callable that, given the block_max,
            stores scale=block_max/127 to the global scale tensor (Q_scale or
            K_scale). It runs only on lane 0 of wave 0. ``scale_off`` is the
            i32 element offset into the scale tensor.
            """
            local_max_av = ArithValue(local_max_v)

            # Wave-shuffle reduce within each wave
            wave_max = local_max_av
            for sh in SHUFFLE_DISTS:
                off_i32 = arith.constant(sh, type=i32)
                peer = wave_max.shuffle_xor(off_i32, c_warp_size_i32)
                wave_max = arith.maximumf(wave_max, peer)

            # Lane 0 of each wave writes its wave_max to LDS[wave_id]
            is_lane0 = arith.cmpi(CmpIPredicate.eq, lane, c0_i32)
            _if_lane0 = scf.IfOp(is_lane0)
            with ir.InsertionPoint(_if_lane0.then_block):
                Vec.from_elements([wave_max], fx.Float32).store(
                    lds, [arith.index_cast(T.index, wave_id)]
                )
                scf.YieldOp([])

            gpu.barrier()

            # Wave 0, lane 0 reads NUM_WAVES values, reduces, computes
            # inv_scale, stores it to LDS[0] for broadcast, and writes the
            # final scale to global Q/K scale tensor.
            is_w0 = arith.cmpi(CmpIPredicate.eq, wave_id, c0_i32)
            is_w0l0 = arith.andi(is_w0, is_lane0)
            _if_w0l0 = scf.IfOp(is_w0l0)
            with ir.InsertionPoint(_if_w0l0.then_block):
                block_max = c0_f32
                for w in range_constexpr(NUM_WAVES):
                    wval_vec = Vec.load(
                        T.vec(1, f32), lds,
                        [arith.constant(w, index=True)],
                    )
                    wval = vector.extract(
                        wval_vec, static_position=[0], dynamic_position=[]
                    )
                    block_max = arith.maximumf(block_max, wval)

                # scale = block_max / 127
                scale = block_max * c_inv_int8_max
                # inv_scale = 127 / block_max  (use rcp for speed; if block_max
                # is 0, scale is 0 and we'll divide by zero — handle by
                # forcing inv_scale to 0 in that case so the quantized value
                # collapses to 0).
                is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, block_max, c0_f32)
                inv_scale_raw = llvm.call_intrinsic(
                    f32, "llvm.amdgcn.rcp.f32", [scale], [], []
                )
                inv_scale = arith.select(is_zero, c0_f32, inv_scale_raw)

                Vec.from_elements([inv_scale], fx.Float32).store(
                    lds, [arith.constant(0, index=True)]
                )

                # Write scale to global Q/K_scale[scale_off] (1 f32)
                store_scale_to_global(scale)
                scf.YieldOp([])

            gpu.barrier()

            inv_vec = Vec.load(
                T.vec(1, f32), lds, [arith.constant(0, index=True)]
            )
            return vector.extract(
                inv_vec, static_position=[0], dynamic_position=[]
            )

        # ---------- generic Q/K INT8 body ----------
        def do_qk_quant(
            local_pid,
            in_rsrc, out_rsrc, scale_rsrc,
            BLK, c_blk, num_heads, c_num_heads, num_blks_i32,
            seq_len_i32,
            PASSES,
            apply_sm_scale,
        ):
            blk_idx = local_pid % num_blks_i32
            bh_id = local_pid // num_blks_i32
            h_id = bh_id % c_num_heads
            b_id = bh_id // c_num_heads

            # Stage 1: load + amax
            cached_vals = []  # PASSES x VEC list of f32 ArithValues
            local_max = c0_f32

            for p in range_constexpr(PASSES):
                row_in_blk = row_base + arith.constant(p * ROWS_PER_PASS, type=i32)
                global_row = blk_idx * c_blk + row_in_blk
                row_in_range = arith.cmpi(
                    CmpIPredicate.ult, global_row, seq_len_i32
                )

                # bf16 element offset
                row_elem_off = (
                    (b_id * seq_len_i32 + global_row) * c_num_heads + h_id
                ) * c_head_dim + col_d
                in_dw_off = row_elem_off >> c1_i32  # bf16: 2 bytes/elem -> dword off = elem/2

                raw = buffer_ops.buffer_load(
                    in_rsrc, in_dw_off, vec_width=4, dtype=i32
                )
                bf16_v = vector.bitcast(vec_bf16_ty, raw)
                f32_v = bf16_v.extf(vec_f32_ty)

                pass_vals = []
                for vi in range_constexpr(VEC):
                    x = vector.extract(
                        f32_v, static_position=[vi], dynamic_position=[]
                    )
                    if const_expr(apply_sm_scale):
                        x = x * sm_scale_log2e_v
                    # Mask OOB rows so amax/quant are unaffected
                    x_masked = arith.select(row_in_range, x, c0_f32)
                    pass_vals.append(x_masked)
                    abs_x = llvm.call_intrinsic(
                        f32, "llvm.fabs.f32", [x_masked], [], []
                    )
                    local_max = arith.maximumf(local_max, abs_x)

                cached_vals.append(pass_vals)

            # Stage 2: cross-wave reduce + broadcast inv_scale
            # scale_off = (b * H + h) * NUM_BLKS + blk_idx (f32 element index)
            scale_off = (b_id * c_num_heads + h_id) * num_blks_i32 + blk_idx

            def store_scale(scale_v):
                # buffer_store of f32 -> use vec_width=1, offset_is_bytes=False
                # (offset is dword index since f32 is 4 bytes / dword).
                buffer_ops.buffer_store(
                    scale_v,
                    scale_rsrc,
                    scale_off,
                    offset_is_bytes=False,
                )

            inv_scale = cross_wave_amax_to_inv_scale(
                local_max, store_scale, scale_off
            )

            # Stage 3: quantize and store (int8)
            for p in range_constexpr(PASSES):
                row_in_blk = row_base + arith.constant(p * ROWS_PER_PASS, type=i32)
                global_row = blk_idx * c_blk + row_in_blk
                row_in_range = arith.cmpi(
                    CmpIPredicate.ult, global_row, seq_len_i32
                )
                _if_store = scf.IfOp(row_in_range)
                with ir.InsertionPoint(_if_store.then_block):
                    row_elem_off = (
                        (b_id * seq_len_i32 + global_row) * c_num_heads + h_id
                    ) * c_head_dim + col_d
                    # int8: byte offset = element offset

                    int8_vals_i32 = []
                    for vi in range_constexpr(VEC):
                        xs = cached_vals[p][vi] * inv_scale
                        # Round-half-away-from-zero: trunc(x + 0.5*sign(x))
                        is_pos = arith.cmpf(
                            arith.CmpFPredicate.OGE, xs, c0_f32
                        )
                        signed_half = arith.select(is_pos, c_p_half, c_n_half)
                        rounded = xs + signed_half
                        as_i32 = arith.fptosi(i32, rounded)
                        # Clamp to [-127, 127]
                        clamped = arith.maxsi(
                            arith.minsi(as_i32, c_int8_max_pos),
                            c_int8_max_neg,
                        )
                        int8_vals_i32.append(clamped)

                    # Pack 4 i8 (taken from LSB) per i32 word
                    for _wg in range_constexpr(VEC // 4):
                        _b = _wg * 4
                        b0 = int8_vals_i32[_b] & c0xFF_i32
                        b1 = (int8_vals_i32[_b + 1] & c0xFF_i32) << c8_i32
                        b2 = (int8_vals_i32[_b + 2] & c0xFF_i32) << c16_i32
                        b3 = (int8_vals_i32[_b + 3] & c0xFF_i32) << c24_i32
                        packed = b0 | b1 | b2 | b3
                        word_off = row_elem_off + arith.constant(
                            _wg * 4, type=i32
                        )
                        buffer_ops.buffer_store(
                            packed,
                            out_rsrc,
                            word_off,
                            offset_is_bytes=True,
                        )
                    scf.YieldOp([])

        # ---------- V FP8 body (same as standalone V kernel) ----------
        def do_v_quant(local_pid):
            blk_n_id = local_pid % num_blocks_k_i32
            bh_id = local_pid // num_blocks_k_i32
            h_id = bh_id % c_kv_heads
            b_id = bh_id // c_kv_heads

            # v_scale[b, h, col_d:col_d+8] - constant across BLK_K rows
            scale_dw_base = (b_id * c_kv_heads + h_id) * c_head_dim + col_d
            scale_lo_raw = buffer_ops.buffer_load(
                v_scale_rsrc, scale_dw_base, vec_width=4, dtype=i32
            )
            scale_lo = vector.bitcast(v4f32_ty, scale_lo_raw)
            scale_hi_raw = buffer_ops.buffer_load(
                v_scale_rsrc, scale_dw_base + c4_i32, vec_width=4, dtype=i32
            )
            scale_hi = vector.bitcast(v4f32_ty, scale_hi_raw)

            v_inv_scales = []
            for vi in range_constexpr(4):
                s = vector.extract(
                    scale_lo, static_position=[vi], dynamic_position=[]
                )
                inv_s = llvm.call_intrinsic(
                    f32, "llvm.amdgcn.rcp.f32", [s], [], []
                )
                v_inv_scales.append(inv_s)
            for vi in range_constexpr(4):
                s = vector.extract(
                    scale_hi, static_position=[vi], dynamic_position=[]
                )
                inv_s = llvm.call_intrinsic(
                    f32, "llvm.amdgcn.rcp.f32", [s], [], []
                )
                v_inv_scales.append(inv_s)

            for p in range_constexpr(PASSES_V):
                row_in_blk = row_base + arith.constant(p * ROWS_PER_PASS, type=i32)
                global_row = blk_n_id * c_blk_k + row_in_blk
                row_in_range = arith.cmpi(
                    CmpIPredicate.ult, global_row, seq_len_k_i32
                )
                _if_row = scf.IfOp(row_in_range)
                with ir.InsertionPoint(_if_row.then_block):
                    row_elem_off = (
                        (b_id * seq_len_k_i32 + global_row) * c_kv_heads + h_id
                    ) * c_head_dim + col_d
                    in_dw_off = row_elem_off >> c1_i32
                    raw = buffer_ops.buffer_load(
                        v_in_rsrc, in_dw_off, vec_width=4, dtype=i32
                    )
                    bf16_v = vector.bitcast(vec_bf16_ty, raw)
                    f32_v = bf16_v.extf(vec_f32_ty)

                    scaled_vals = []
                    for vi in range_constexpr(VEC):
                        x = vector.extract(
                            f32_v, static_position=[vi], dynamic_position=[]
                        )
                        scaled_vals.append(x * v_inv_scales[vi])

                    out_byte_off = row_elem_off
                    for _wg in range_constexpr(VEC // 4):
                        _b = _wg * 4
                        packed_w = c0_i32
                        packed_w = rocdl.cvt_pk_fp8_f32(
                            i32,
                            scaled_vals[_b],
                            scaled_vals[_b + 1],
                            packed_w,
                            0,
                        )
                        packed_w = rocdl.cvt_pk_fp8_f32(
                            i32,
                            scaled_vals[_b + 2],
                            scaled_vals[_b + 3],
                            packed_w,
                            1,
                        )
                        word_off = out_byte_off + arith.constant(
                            _wg * 4, type=i32
                        )
                        buffer_ops.buffer_store(
                            packed_w,
                            v_out_rsrc,
                            word_off,
                            offset_is_bytes=True,
                        )
                    scf.YieldOp([])

        # ---------- Branch dispatch on bid ----------
        is_q = arith.cmpi(CmpIPredicate.ult, bid_i32, q_task_count_i32)
        _if_q = scf.IfOp(is_q, has_else=True)
        with ir.InsertionPoint(_if_q.then_block):
            do_qk_quant(
                bid_i32,
                q_in_rsrc, q_out_rsrc, q_scale_rsrc,
                blk_q, c_blk_q, num_q_heads, c_q_heads, num_blocks_q_i32,
                seq_len_q_i32,
                PASSES_Q,
                apply_sm_scale=True,
            )
            scf.YieldOp([])
        with ir.InsertionPoint(_if_q.else_block):
            local_pid_kv = bid_i32 - q_task_count_i32
            is_k = arith.cmpi(CmpIPredicate.ult, local_pid_kv, k_task_count_i32)
            _if_k = scf.IfOp(is_k, has_else=True)
            with ir.InsertionPoint(_if_k.then_block):
                do_qk_quant(
                    local_pid_kv,
                    k_in_rsrc, k_out_rsrc, k_scale_rsrc,
                    blk_k, c_blk_k, num_kv_heads, c_kv_heads, num_blocks_k_i32,
                    seq_len_k_i32,
                    PASSES_K,
                    apply_sm_scale=False,
                )
                scf.YieldOp([])
            with ir.InsertionPoint(_if_k.else_block):
                v_pid = local_pid_kv - k_task_count_i32
                do_v_quant(v_pid)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_sage_quant_fused(
        Q_in: fx.Tensor,
        Q_out: fx.Tensor,
        Q_scale: fx.Tensor,
        K_in: fx.Tensor,
        K_out: fx.Tensor,
        K_scale: fx.Tensor,
        V_in: fx.Tensor,
        V_out: fx.Tensor,
        V_scale: fx.Tensor,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        num_blocks_q: fx.Int32,
        num_blocks_k: fx.Int32,
        q_task_count: fx.Int32,
        k_task_count: fx.Int32,
        sm_scale_log2e_bits: fx.Int32,
        grid_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        idx_grid = arith.index_cast(T.index, grid_size)
        launcher = sage_quant_fused_kernel(
            Q_in, Q_out, Q_scale,
            K_in, K_out, K_scale,
            V_in, V_out, V_scale,
            seq_len_q, seq_len_k,
            num_blocks_q, num_blocks_k,
            q_task_count, k_task_count,
            sm_scale_log2e_bits,
        )
        launcher.launch(
            grid=(idx_grid, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_sage_quant_fused
