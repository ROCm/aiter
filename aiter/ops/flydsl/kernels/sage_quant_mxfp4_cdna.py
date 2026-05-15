# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL kernels for SageAttention V2 MXFP4 quantization (CDNA gfx950).

Builds a single fused kernel that performs Q rotate+MXFP4 + K rotate+MXFP4 +
V FP8 quantization in one launch. Mirrors the layout of Triton's
sage_quant_mxfp4 (sage_attention_quant_wrappers.py:107-196):

    q_fp4:    uint8 [B, S_q, Hq,  D//2]
    q_d:      uint8 [B, S_q, Hq,  D//32]   e8m0 (one byte per 32-elem group)
    k_fp4:    uint8 [B, S_k, Hk,  D//2]
    k_d:      uint8 [B, S_k, Hk,  D//32]
    v_fp8:    fp8e4m3 [B, S_k, Hk, D]
    v_scale:  f32   [B, Hk, D]              per-channel (precomputed outside)

Layout notes (head_dim=128, BLOCK_THREADS=256, VEC=8):

  * THREADS_PER_ROW = D/VEC = 16. Each thread holds 8 contiguous bf16
    elements of one row. ROWS_PER_PASS = 16, PASSES = BLK/16.
  * Hadamard rotation (BLOCK_R=128) inline butterfly: stages 0,1,2 are
    intra-thread (within the 8-elem register vector); stages 3,4,5,6 use
    shuffle_xor across the row's 16-lane span (offsets 1,2,4,8).
  * Per-32-elem MXFP4 amax: each thread computes intra-amax over its 8
    elems; then 2 stages of shuffle_xor (offsets 1,2 within the 4-lane
    sub-group that holds the 32-elem scale group). Result replicated to
    all 4 lanes of the group.
  * e8m0 RNE-to-power-of-2 mirrors Triton _compute_mx_quant_and_scale_rne
    (sage_attention_quant.py:41-66) byte-exactly.
  * FP4 E2M1 packing mirrors Triton _compute_mx_quant_and_scale_rne
    (sage_attention_quant.py:92-122).
  * Descale store: pack 4 per-group e8m0 bytes (held in lanes 0,4,8,12 of
    the row) into 1 i32 via OR-reduce across the row's 16 lanes (4
    shuffle_xor OR stages), then lane 0 of row writes 1 dword.

K-mean subtraction: handled out-of-kernel by torch (matches Triton).

q_smoothing: NOT supported in this kernel (caller must fall back to
Triton quant for q_smooth=True).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, range_constexpr, const_expr, gpu
from flydsl.expr.typing import T, Int32, Vector as Vec
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf, rocdl
from flydsl.expr import buffer_ops
from flydsl.runtime.device import get_rocm_arch as get_hip_arch  # noqa: F401

BLOCK_THREADS = 256
WARP_SIZE = 64
NUM_WAVES = BLOCK_THREADS // WARP_SIZE  # 4


def build_sage_quant_mxfp4_module(
    head_dim: int,
    blk_q: int,
    blk_k: int,
    num_q_heads: int,
    num_kv_heads: int,
    subtract_k_mean: bool = True,
):
    """Build a fused single-launch kernel for MXFP4 sage quantization.

    Grid layout: bid in [0, q_count + k_count + v_count) where
        q_count = B * Hq * num_blocks_q
        k_count = B * Hk * num_blocks_k
        v_count = B * Hk * num_blocks_k    (V uses BLK_K rows per block)

    Per-bid dispatch:
        bid < q_count                         -> Q rotate + MXFP4 quant
        bid in [q_count, q_count+k_count)     -> K rotate + MXFP4 quant
        bid in [q_count+k_count, ...+v_count) -> V FP8 quant
    """
    assert head_dim == 128, f"only head_dim=128 supported, got {head_dim}"
    assert blk_q in (128, 256), f"unexpected BLK_Q={blk_q}"
    assert blk_k in (64, 128, 256), f"unexpected BLK_K={blk_k}"

    VEC = 8                       # bf16 elements per thread
    THREADS_PER_ROW = head_dim // VEC  # 16
    assert BLOCK_THREADS % THREADS_PER_ROW == 0
    ROWS_PER_PASS = BLOCK_THREADS // THREADS_PER_ROW  # 16
    assert blk_q % ROWS_PER_PASS == 0
    assert blk_k % ROWS_PER_PASS == 0
    PASSES_Q = blk_q // ROWS_PER_PASS
    PASSES_K = blk_k // ROWS_PER_PASS
    PASSES_V = PASSES_K

    NUM_SCALE_GROUPS = head_dim // 32   # 4
    GROUP_LANES = 32 // VEC              # 4 lanes per scale group

    # Hadamard normalization 1/sqrt(128)
    import math
    HADAMARD_NORM = 1.0 / math.sqrt(head_dim)

    @flyc.kernel
    def sage_quant_mxfp4_kernel(
        Q_in: fx.Tensor,        # bf16 [B, S_q, Hq, D]
        Q_fp4: fx.Tensor,       # u8   [B, S_q, Hq, D//2]    packed FP4
        Q_descale: fx.Tensor,   # u8   [B, S_q, Hq, D//32]   e8m0
        K_in: fx.Tensor,        # bf16 [B, S_k, Hk, D]       (RAW)
        K_fp4: fx.Tensor,       # u8   [B, S_k, Hk, D//2]
        K_descale: fx.Tensor,   # u8   [B, S_k, Hk, D//32]
        K_mean: fx.Tensor,      # f32  [B, Hk, D]            subtracted from K rows
        V_in: fx.Tensor,        # bf16 [B, S_k, Hk, D]
        V_fp8: fx.Tensor,       # fp8  [B, S_k, Hk, D]
        V_scale: fx.Tensor,     # f32  [B, Hk, D]            per-channel (precomputed)
        seq_len_q: Int32,
        seq_len_k: Int32,
        num_blocks_q: Int32,
        num_blocks_k: Int32,
        q_task_count: Int32,
        k_task_count: Int32,
        sm_scale_log2e_bits: Int32,   # f32 bits of sm_scale * log2(e)
        hadamard_norm_bits: Int32,    # f32 bits of 1/sqrt(D)
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32

        bid_i32 = ArithValue(bid)
        tid_i32 = ArithValue(tid)

        # Constants ---------------------------------------------------------
        c0_i32  = arith.constant(0,    type=i32)
        c1_i32  = arith.constant(1,    type=i32)
        c2_i32  = arith.constant(2,    type=i32)
        c4_i32  = arith.constant(4,    type=i32)
        c8_i32  = arith.constant(8,    type=i32)
        c16_i32 = arith.constant(16,   type=i32)
        c24_i32 = arith.constant(24,   type=i32)
        c0xFF_i32 = arith.constant(0xFF,    type=i32)
        c0x7FFFFF_i32 = arith.constant(0x7FFFFF, type=i32)
        c0x400000_i32 = arith.constant(0x400000, type=i32)
        c0x80000000_i32 = arith.constant(-0x80000000, type=i32)  # sign bit
        c0x7F800000_i32 = arith.constant(0x7F800000, type=i32)
        c127_i32 = arith.constant(127, type=i32)
        c126_i32 = arith.constant(126, type=i32)
        c254_i32 = arith.constant(254, type=i32)
        c7_i32   = arith.constant(7,   type=i32)
        c21_i32  = arith.constant(21,  type=i32)
        c23_i32  = arith.constant(23,  type=i32)
        c28_i32  = arith.constant(28,  type=i32)

        c_warp_size_i32  = arith.constant(WARP_SIZE,       type=i32)
        c_threads_per_row = arith.constant(THREADS_PER_ROW, type=i32)
        c_head_dim       = arith.constant(head_dim,        type=i32)
        c_q_heads        = arith.constant(num_q_heads,     type=i32)
        c_kv_heads       = arith.constant(num_kv_heads,    type=i32)
        c_blk_q          = arith.constant(blk_q,           type=i32)
        c_blk_k          = arith.constant(blk_k,           type=i32)
        c_vec            = arith.constant(VEC,             type=i32)
        c_d_div_32       = arith.constant(NUM_SCALE_GROUPS, type=i32)
        c0_f32           = arith.constant(0.0,             type=f32)

        # Hadamard butterfly stage offsets (cross-lane partner offsets)
        c_lane_off_1 = arith.constant(1, type=i32)
        c_lane_off_2 = arith.constant(2, type=i32)
        c_lane_off_4 = arith.constant(4, type=i32)
        c_lane_off_8 = arith.constant(8, type=i32)

        # ------------------------------------------------------------------
        # Per-thread coordinates within the (BLK, D) tile
        # ------------------------------------------------------------------
        # row_base:    which row of the 16 rows-per-pass this thread handles
        # col_d:       starting column (in elements) for this thread
        # lane_in_row: 0..15, identifies the thread's column-segment
        # group_idx:   0..3,  which scale group this thread is in (lane_in_row//4)
        row_base    = tid_i32 // c_threads_per_row
        col_d       = (tid_i32 % c_threads_per_row) * c_vec
        lane_in_row = tid_i32 % c_threads_per_row
        group_idx   = lane_in_row >> c2_i32  # //4

        seq_len_q_i32     = ArithValue(seq_len_q)
        seq_len_k_i32     = ArithValue(seq_len_k)
        num_blocks_q_i32  = ArithValue(num_blocks_q)
        num_blocks_k_i32  = ArithValue(num_blocks_k)
        q_task_count_i32  = ArithValue(q_task_count)
        k_task_count_i32  = ArithValue(k_task_count)
        sm_scale_log2e_v  = arith.bitcast(f32, ArithValue(sm_scale_log2e_bits))
        hadamard_norm_v   = arith.bitcast(f32, ArithValue(hadamard_norm_bits))

        vec_bf16_ty = T.vec(VEC, T.bf16)
        vec_f32_ty  = T.vec(VEC, f32)
        v4f32_ty    = T.vec(4, f32)

        # ------------------------------------------------------------------
        # Buffer resources
        # ------------------------------------------------------------------
        q_in_rsrc      = buffer_ops.create_buffer_resource(Q_in,      max_size=True)
        q_fp4_rsrc     = buffer_ops.create_buffer_resource(Q_fp4,     max_size=True)
        q_descale_rsrc = buffer_ops.create_buffer_resource(Q_descale, max_size=True)
        k_in_rsrc      = buffer_ops.create_buffer_resource(K_in,      max_size=True)
        k_fp4_rsrc     = buffer_ops.create_buffer_resource(K_fp4,     max_size=True)
        k_descale_rsrc = buffer_ops.create_buffer_resource(K_descale, max_size=True)
        k_mean_rsrc    = buffer_ops.create_buffer_resource(K_mean,    max_size=True)
        v_in_rsrc      = buffer_ops.create_buffer_resource(V_in,      max_size=True)
        v_fp8_rsrc     = buffer_ops.create_buffer_resource(V_fp8,     max_size=True)
        v_scale_rsrc   = buffer_ops.create_buffer_resource(V_scale,   max_size=True)

        # ==================================================================
        # Helper: per-row inline FWHT (Walsh-Hadamard) for BLOCK_R=128
        # ==================================================================
        def hadamard_rotate_row(v):
            """Apply 7-stage in-place WHT butterfly to the per-thread vector.

            ``v`` is a list of 8 f32 ArithValues holding the thread's 8
            consecutive elements of one row. Returns the rotated list.
            """
            # ----- Stages 0..2: intra-thread butterflies -----
            # Stage 0: distance 1 (pairs of consecutive elements)
            new_v = list(v)
            for i in (0, 2, 4, 6):
                a = new_v[i]
                b = new_v[i + 1]
                new_v[i]     = a + b
                new_v[i + 1] = a - b
            # Stage 1: distance 2
            v = list(new_v)
            for i in (0, 1, 4, 5):
                a = v[i]
                b = v[i + 2]
                v[i]     = a + b
                v[i + 2] = a - b
            # Stage 2: distance 4
            new_v = list(v)
            for i in (0, 1, 2, 3):
                a = new_v[i]
                b = new_v[i + 4]
                new_v[i]     = a + b
                new_v[i + 4] = a - b

            # ----- Stages 3..6: cross-lane butterflies -----
            # Stage k (k=3..6): partner = my_lane XOR (1<<(k-3)) within row's
            # 16-lane span. Sign: if (lane_in_row & lane_offset) != 0 then
            # subtract (we are the "high" partner), else add.
            for lane_off_const in (c_lane_off_1, c_lane_off_2,
                                    c_lane_off_4, c_lane_off_8):
                # is_high = (lane_in_row & lane_off) != 0
                masked = arith.AndIOp(lane_in_row, lane_off_const).result
                is_high = arith.cmpi(CmpIPredicate.ne, masked, c0_i32)

                rotated = []
                for i in (0, 1, 2, 3, 4, 5, 6, 7):
                    self_v = ArithValue(new_v[i])
                    peer = self_v.shuffle_xor(lane_off_const, c_warp_size_i32)
                    add_v = self_v + peer
                    sub_v = peer - self_v   # peer - self when "I am high"
                    chosen = arith.select(is_high, sub_v, add_v)
                    rotated.append(chosen)
                new_v = rotated

            # Normalize by 1/sqrt(128)
            return [ArithValue(x) * hadamard_norm_v for x in new_v]

        # ==================================================================
        # Helper: per-32-elem amax via 2-stage shuffle_xor in the 4-lane group
        # ==================================================================
        def per_group_amax(v):
            """Returns the per-(row, group) amax for this thread.

            All 4 lanes in the same scale group end up with the same value
            (replicated). Each thread first reduces over its 8 elements
            (intra-thread), then 2 shuffle_xor stages (offsets 1, 2) inside
            the 4-lane group.
            """
            # Intra-thread amax
            local = c0_f32
            for i in (0, 1, 2, 3, 4, 5, 6, 7):
                abs_x = llvm.call_intrinsic(f32, "llvm.fabs.f32", [v[i]], [], [])
                local = arith.maximumf(local, abs_x)
            local_av = ArithValue(local)
            for off_const in (c_lane_off_1, c_lane_off_2):
                peer = local_av.shuffle_xor(off_const, c_warp_size_i32)
                local_av = arith.maximumf(local_av, peer)
            return local_av

        # ==================================================================
        # Helper: e8m0 RNE-to-power-of-2 (mirror Triton lines 41-66)
        # Returns (scale_exponent_i32 in [0,254], quant_scale_f32 = 1/dequant)
        # ==================================================================
        def amax_to_e8m0_and_quant_scale(amax_f32):
            max_bits   = arith.bitcast(i32, amax_f32)
            exponent   = arith.AndIOp(
                arith.ShRUIOp(max_bits, c23_i32).result, c0xFF_i32
            ).result
            mantissa   = arith.AndIOp(max_bits, c0x7FFFFF_i32).result

            # should_round_up = (mant > 0x400000) | ((mant == 0x400000) & (exp & 1 == 1))
            mant_gt = arith.cmpi(CmpIPredicate.ugt, mantissa, c0x400000_i32)
            mant_eq = arith.cmpi(CmpIPredicate.eq,  mantissa, c0x400000_i32)
            exp_lsb = arith.AndIOp(exponent, c1_i32).result
            exp_odd = arith.cmpi(CmpIPredicate.eq, exp_lsb, c1_i32)
            tie_up  = arith.andi(mant_eq, exp_odd)
            should_up = arith.ori(mant_gt, tie_up)

            rounded_exp = arith.select(should_up,
                                        arith.AddIOp(exponent, c1_i32).result,
                                        exponent)
            # scale_exp = clamp(rounded_exp - 2, 0, 254)
            scale_exp_signed = arith.subi(rounded_exp, c2_i32)
            # Triton uses tl.maximum(.,0) then tl.minimum(.,254). For signed:
            # if scale_exp_signed < 0 -> 0
            neg = arith.cmpi(CmpIPredicate.slt, scale_exp_signed, c0_i32)
            scale_exp = arith.select(neg, c0_i32, scale_exp_signed)
            too_big = arith.cmpi(CmpIPredicate.sgt, scale_exp, c254_i32)
            scale_exp = arith.select(too_big, c254_i32, scale_exp)

            # dequant_scale = bitcast((scale_exp << 23) & 0x7F800000) -> f32
            dequant_bits = arith.AndIOp(
                arith.ShLIOp(scale_exp, c23_i32).result, c0x7F800000_i32
            ).result
            dequant_scale = arith.bitcast(f32, dequant_bits)

            # quant_scale = (dequant == 0) ? 0 : 1/dequant
            is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, dequant_scale, c0_f32)
            inv_raw = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32",
                                            [dequant_scale], [], [])
            quant_scale = arith.select(is_zero, c0_f32, inv_raw)

            return scale_exp, quant_scale

        # ==================================================================
        # Helper: f32 -> FP4 E2M1 nibble (mirror Triton lines 92-116)
        # ==================================================================
        def f32_to_e2m1_nibble(x_f32):
            x_bits = arith.bitcast(i32, x_f32)
            sign     = arith.AndIOp(x_bits, c0x80000000_i32).result
            exponent = arith.AndIOp(
                arith.ShRUIOp(x_bits, c23_i32).result, c0xFF_i32
            ).result
            mantissa = arith.AndIOp(x_bits, c0x7FFFFF_i32).result

            # adjusted_exponents = E8_BIAS - (exp + 1) = 126 - exp
            adj_exp = arith.subi(c126_i32, exponent)
            # subnormal mantissa = (0x400000 | (mant >> 1)) >> adj_exp
            mant_or = arith.OrIOp(c0x400000_i32,
                                    arith.ShRUIOp(mantissa, c1_i32).result).result
            sub_mant = arith.ShRUIOp(mant_or, adj_exp).result
            is_sub = arith.cmpi(CmpIPredicate.ult, exponent, c127_i32)
            mantissa = arith.select(is_sub, sub_mant, mantissa)

            # exponent = max(exp, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)
            # E8_BIAS - E2_BIAS = 126
            exp_clipped = arith.select(
                arith.cmpi(CmpIPredicate.ult, exponent, c126_i32),
                c126_i32,
                exponent,
            )
            exp_unbiased = arith.subi(exp_clipped, c126_i32)

            # e2m1_tmp = min(((exp_unbiased << 2) | (mant >> 21)) + 1) >> 1, 0x7)
            shifted = arith.OrIOp(
                arith.ShLIOp(exp_unbiased, c2_i32).result,
                arith.ShRUIOp(mantissa, c21_i32).result,
            ).result
            plus1 = arith.AddIOp(shifted, c1_i32).result
            shifted_back = arith.ShRUIOp(plus1, c1_i32).result
            sat = arith.cmpi(CmpIPredicate.ugt, shifted_back, c7_i32)
            e2m1_tmp = arith.select(sat, c7_i32, shifted_back)

            # nibble = (sign >> 28) | e2m1_tmp
            sign_bit = arith.ShRUIOp(sign, c28_i32).result   # bit 31 -> bit 3
            nibble = arith.OrIOp(sign_bit, e2m1_tmp).result
            return arith.AndIOp(nibble, arith.constant(0xF, type=i32)).result

        # ==================================================================
        # Q/K rotate-quant body
        # ==================================================================
        def do_qk_rotate_quant(local_pid, in_rsrc, fp4_rsrc, descale_rsrc,
                                BLK, c_blk, c_num_heads, num_blks_i32,
                                seq_len_i32, PASSES, apply_sm_scale,
                                subtract_k_mean=False):
            blk_idx = local_pid % num_blks_i32
            bh_id   = local_pid // num_blks_i32
            h_id    = bh_id % c_num_heads
            b_id    = bh_id // c_num_heads

            # Load K_mean[b, h, col_d:col_d+VEC] once per block (8 f32 values).
            if const_expr(subtract_k_mean):
                k_mean_off = (b_id * c_num_heads + h_id) * c_head_dim + col_d
                kmean_lo_raw = buffer_ops.buffer_load(
                    k_mean_rsrc, k_mean_off, vec_width=4, dtype=i32
                )
                kmean_hi_raw = buffer_ops.buffer_load(
                    k_mean_rsrc, k_mean_off + c4_i32, vec_width=4, dtype=i32
                )
                kmean_lo = vector.bitcast(v4f32_ty, kmean_lo_raw)
                kmean_hi = vector.bitcast(v4f32_ty, kmean_hi_raw)
                k_means = []
                for vi in (0, 1, 2, 3):
                    k_means.append(
                        vector.extract(kmean_lo, static_position=[vi],
                                        dynamic_position=[])
                    )
                for vi in (0, 1, 2, 3):
                    k_means.append(
                        vector.extract(kmean_hi, static_position=[vi],
                                        dynamic_position=[])
                    )

            for p in range_constexpr(PASSES):
                row_in_blk = row_base + arith.constant(p * ROWS_PER_PASS, type=i32)
                global_row = blk_idx * c_blk + row_in_blk
                row_in_range = arith.cmpi(CmpIPredicate.ult, global_row, seq_len_i32)

                # ---- Stage 1: load 8 bf16 elements -----
                row_elem_off = (
                    (b_id * seq_len_i32 + global_row) * c_num_heads + h_id
                ) * c_head_dim + col_d
                in_dw_off = arith.ShRUIOp(row_elem_off, c1_i32).result  # bf16 elem -> dword

                # OOB rows: clamp to 0 to avoid large-offset miss; values masked later.
                safe_off = arith.select(row_in_range, in_dw_off, c0_i32)
                raw = buffer_ops.buffer_load(in_rsrc, safe_off, vec_width=4, dtype=i32)
                bf16_v = vector.bitcast(vec_bf16_ty, raw)
                f32_v  = bf16_v.extf(vec_f32_ty)

                v_list = []
                for vi in range_constexpr(VEC):
                    x = vector.extract(f32_v, static_position=[vi], dynamic_position=[])
                    if const_expr(subtract_k_mean):
                        x = x - k_means[vi]
                    v_list.append(x)

                # ---- Stage 2: Hadamard rotation -----
                rotated = hadamard_rotate_row(v_list)

                # Optional sm_scale * log2(e) (Q only)
                if const_expr(apply_sm_scale):
                    rotated = [ArithValue(x) * sm_scale_log2e_v for x in rotated]

                # Mask OOB rows (so amax/quant produce zeros)
                rotated_masked = [arith.select(row_in_range, x, c0_f32) for x in rotated]

                # ---- Stage 3: per-32-elem amax -----
                amax = per_group_amax(rotated_masked)

                # ---- Stage 4: e8m0 + quant_scale -----
                scale_exp, quant_scale = amax_to_e8m0_and_quant_scale(amax)

                # ---- Stage 5: quantize each elem to FP4 nibble -----
                nibbles = []
                for vi in range_constexpr(VEC):
                    qv = ArithValue(rotated_masked[vi]) * ArithValue(quant_scale)
                    nibbles.append(f32_to_e2m1_nibble(qv))

                # ---- Stage 6: pack into 4 bytes (1 dword) -----
                # byte i = nibbles[2i] | (nibbles[2i+1] << 4)
                bytes4 = []
                for i in (0, 1, 2, 3):
                    lo = nibbles[2 * i]
                    hi = arith.ShLIOp(nibbles[2 * i + 1], c4_i32).result
                    bytes4.append(arith.OrIOp(lo, hi).result)
                # pack 4 bytes into 1 i32: b0 | (b1<<8) | (b2<<16) | (b3<<24)
                w0 = bytes4[0]
                w1 = arith.ShLIOp(bytes4[1], c8_i32).result
                w2 = arith.ShLIOp(bytes4[2], c16_i32).result
                w3 = arith.ShLIOp(bytes4[3], c24_i32).result
                packed = arith.OrIOp(
                    arith.OrIOp(w0, w1).result,
                    arith.OrIOp(w2, w3).result,
                ).result

                # Output byte offset: each row holds D//2 = 64 packed bytes;
                # this thread writes 4 bytes at byte offset col_d/2.
                out_byte_row = (
                    (b_id * seq_len_i32 + global_row) * c_num_heads + h_id
                ) * arith.constant(head_dim // 2, type=i32)
                out_byte_off = out_byte_row + arith.ShRUIOp(col_d, c1_i32).result

                # ---- Stage 7: descale i32 packing via OR-reduce across row -----
                # Only the lane_in_row % 4 == 0 lane in each group has the real
                # e8m0; mask others to 0 to avoid double-counting in the OR.
                # Then shift by group_idx * 8 and OR-reduce across 16 lanes.
                grp_first_lane = arith.cmpi(CmpIPredicate.eq,
                                              arith.AndIOp(lane_in_row,
                                                            arith.constant(3, type=i32)
                                                            ).result,
                                              c0_i32)
                shift_amt = arith.ShLIOp(group_idx, arith.constant(3, type=i32)).result  # *8
                shifted   = arith.ShLIOp(scale_exp, shift_amt).result
                masked_shifted = arith.select(grp_first_lane, shifted, c0_i32)

                # OR-reduce across the row's 16 lanes (4 stages: 1,2,4,8)
                or_acc = ArithValue(masked_shifted)
                for off_const in (c_lane_off_1, c_lane_off_2,
                                    c_lane_off_4, c_lane_off_8):
                    peer = or_acc.shuffle_xor(off_const, c_warp_size_i32)
                    or_acc = arith.OrIOp(or_acc, peer).result
                    or_acc = ArithValue(or_acc)

                # Conditional store (only valid rows)
                _if_store = scf.IfOp(row_in_range)
                with ir.InsertionPoint(_if_store.then_block):
                    # Write the 4 packed FP4 bytes (1 dword)
                    buffer_ops.buffer_store(
                        packed, fp4_rsrc, out_byte_off, offset_is_bytes=True,
                    )
                    # Lane 0 of row writes the descale dword
                    is_row_lane0 = arith.cmpi(CmpIPredicate.eq, lane_in_row, c0_i32)
                    _if_dscl = scf.IfOp(is_row_lane0)
                    with ir.InsertionPoint(_if_dscl.then_block):
                        descale_byte_off = (
                            (b_id * seq_len_i32 + global_row) * c_num_heads + h_id
                        ) * c_d_div_32
                        buffer_ops.buffer_store(
                            or_acc, descale_rsrc, descale_byte_off,
                            offset_is_bytes=True,
                        )
                        scf.YieldOp([])
                    scf.YieldOp([])

        # ==================================================================
        # V FP8 quant body (per-channel scale loaded once per block)
        # ==================================================================
        def do_v_quant(local_pid):
            blk_n_id = local_pid % num_blocks_k_i32
            bh_id    = local_pid // num_blocks_k_i32
            h_id     = bh_id % c_kv_heads
            b_id     = bh_id // c_kv_heads

            scale_dw_base = (b_id * c_kv_heads + h_id) * c_head_dim + col_d
            scale_lo_raw = buffer_ops.buffer_load(v_scale_rsrc, scale_dw_base,
                                                    vec_width=4, dtype=i32)
            scale_lo = vector.bitcast(v4f32_ty, scale_lo_raw)
            scale_hi_raw = buffer_ops.buffer_load(v_scale_rsrc,
                                                    scale_dw_base + c4_i32,
                                                    vec_width=4, dtype=i32)
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

            for p in range_constexpr(PASSES_V):
                row_in_blk = row_base + arith.constant(p * ROWS_PER_PASS, type=i32)
                global_row = blk_n_id * c_blk_k + row_in_blk
                row_in_range = arith.cmpi(CmpIPredicate.ult, global_row,
                                            seq_len_k_i32)
                _if_row = scf.IfOp(row_in_range)
                with ir.InsertionPoint(_if_row.then_block):
                    row_elem_off = (
                        (b_id * seq_len_k_i32 + global_row) * c_kv_heads + h_id
                    ) * c_head_dim + col_d
                    in_dw_off = arith.ShRUIOp(row_elem_off, c1_i32).result
                    raw = buffer_ops.buffer_load(v_in_rsrc, in_dw_off,
                                                    vec_width=4, dtype=i32)
                    bf16_v = vector.bitcast(vec_bf16_ty, raw)
                    f32_v  = bf16_v.extf(vec_f32_ty)
                    scaled_vals = []
                    for vi in range_constexpr(VEC):
                        x = vector.extract(f32_v, static_position=[vi],
                                            dynamic_position=[])
                        scaled_vals.append(x * v_inv_scales[vi])

                    out_byte_off = row_elem_off  # 1 byte/elem for FP8
                    for _wg in range_constexpr(VEC // 4):
                        _b = _wg * 4
                        packed_w = c0_i32
                        packed_w = rocdl.cvt_pk_fp8_f32(
                            i32, scaled_vals[_b], scaled_vals[_b + 1],
                            packed_w, 0,
                        )
                        packed_w = rocdl.cvt_pk_fp8_f32(
                            i32, scaled_vals[_b + 2], scaled_vals[_b + 3],
                            packed_w, 1,
                        )
                        word_off = out_byte_off + arith.constant(_wg * 4, type=i32)
                        buffer_ops.buffer_store(
                            packed_w, v_fp8_rsrc, word_off, offset_is_bytes=True,
                        )
                    scf.YieldOp([])

        # ==================================================================
        # bid dispatch
        # ==================================================================
        is_q = arith.cmpi(CmpIPredicate.ult, bid_i32, q_task_count_i32)
        _if_q = scf.IfOp(is_q, has_else=True)
        with ir.InsertionPoint(_if_q.then_block):
            do_qk_rotate_quant(
                bid_i32, q_in_rsrc, q_fp4_rsrc, q_descale_rsrc,
                blk_q, c_blk_q, c_q_heads, num_blocks_q_i32,
                seq_len_q_i32, PASSES_Q, apply_sm_scale=True,
            )
            scf.YieldOp([])
        with ir.InsertionPoint(_if_q.else_block):
            local_pid_kv = bid_i32 - q_task_count_i32
            is_k = arith.cmpi(CmpIPredicate.ult, local_pid_kv, k_task_count_i32)
            _if_k = scf.IfOp(is_k, has_else=True)
            with ir.InsertionPoint(_if_k.then_block):
                do_qk_rotate_quant(
                    local_pid_kv, k_in_rsrc, k_fp4_rsrc, k_descale_rsrc,
                    blk_k, c_blk_k, c_kv_heads, num_blocks_k_i32,
                    seq_len_k_i32, PASSES_K, apply_sm_scale=False,
                    subtract_k_mean=subtract_k_mean,
                )
                scf.YieldOp([])
            with ir.InsertionPoint(_if_k.else_block):
                v_pid = local_pid_kv - k_task_count_i32
                do_v_quant(v_pid)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_sage_quant_mxfp4(
        Q_in: fx.Tensor,
        Q_fp4: fx.Tensor,
        Q_descale: fx.Tensor,
        K_in: fx.Tensor,
        K_fp4: fx.Tensor,
        K_descale: fx.Tensor,
        K_mean: fx.Tensor,
        V_in: fx.Tensor,
        V_fp8: fx.Tensor,
        V_scale: fx.Tensor,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        num_blocks_q: fx.Int32,
        num_blocks_k: fx.Int32,
        q_task_count: fx.Int32,
        k_task_count: fx.Int32,
        sm_scale_log2e_bits: fx.Int32,
        hadamard_norm_bits: fx.Int32,
        grid_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        idx_grid = arith.index_cast(T.index, grid_size)
        launcher = sage_quant_mxfp4_kernel(
            Q_in, Q_fp4, Q_descale,
            K_in, K_fp4, K_descale, K_mean,
            V_in, V_fp8, V_scale,
            seq_len_q, seq_len_k,
            num_blocks_q, num_blocks_k,
            q_task_count, k_task_count,
            sm_scale_log2e_bits, hadamard_norm_bits,
        )
        launcher.launch(
            grid=(idx_grid, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    # Stash compile-time constants for the wrapper to read (debug / asserts).
    launch_sage_quant_mxfp4._meta = dict(
        head_dim=head_dim, blk_q=blk_q, blk_k=blk_k,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        hadamard_norm=HADAMARD_NORM,
    )
    return launch_sage_quant_mxfp4
