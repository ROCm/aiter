# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused silu_and_mul + MXFP4 quantization + sorted-scale write kernel (FlyDSL).

Designed for split-K MOE stage1 post-processing:

  input   : tmp_out  (token_num * topk, inter_dim * 2) bf16
  sorted  : sorted_token_ids (sorted_len,) i32 -- packed (token<<0 | slot<<24)
            num_valid_ids    (1,) i32
  output  : out_fp4          raw byte buffer -- FP4x2 packed, row stride = inter_dim//2
            out_scale_sorted raw byte buffer -- tiled E8M0 scale (same layout as moe_mxfp4_sort)

Grid:  (num_sorted_rows, 1, 1)  -- one workgroup per sorted row (including blockM padding).
Block: (BLOCK_THREADS, 1, 1)

Each workgroup:
  1. Loads sorted_token_ids[bid] -> (token_id, slot_id) -> row = token_id * topk + slot_id
  2. If bid < num_valid_ids (valid row):
     a. Reads gate = tmp_out[row, 0:inter_dim], up = tmp_out[row, inter_dim:2*inter_dim]
     b. Computes silu(gate) * up in f32
     c. Per-1x32 MXFP4 quant -> writes packed FP4 + E8M0 scale in tiled layout
  3. If bid >= num_valid_ids (blockM padding row):
     a. Writes zero FP4 bytes to out_fp4
     b. Writes zero E8M0 scale to out_scale_sorted (keeps tiled layout consistent)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import buffer_ops

BLOCK_THREADS = 256
WARP_SIZE = 64


def build_silu_and_mul_fq_module(inter_dim: int, topk: int):
    """Return a JIT launcher for fused silu_and_mul + mxfp4 quant + scale sort.

    Parameters
    ----------
    inter_dim : int
        Output columns of stage1 (after activation). Input has inter_dim*2 cols.
        Must be divisible by 32 (MXFP4 block size).
    topk : int
        Number of expert slots per token.
    """
    assert inter_dim % 32 == 0, f"inter_dim={inter_dim} must be divisible by 32"

    scale_cols = inter_dim // 32
    ELEMS_PER_THREAD = (inter_dim + BLOCK_THREADS - 1) // BLOCK_THREADS
    # VEC: number of f32 elements each thread handles; must be even for FP4 packing
    VEC = max(ELEMS_PER_THREAD, 2)
    if VEC % 2 != 0:
        VEC += 1
    assert 32 % VEC == 0, f"VEC={VEC} must divide 32 evenly"
    # threads that actually participate in a 32-element quant group
    THREADS_PER_QUANT_BLK = 32 // VEC
    # shuffle distances for intra-group reduction
    SHUFFLE_DISTS = []
    d = 1
    while d < THREADS_PER_QUANT_BLK:
        SHUFFLE_DISTS.append(d)
        d *= 2

    elem_bytes_bf16 = 2

    @flyc.kernel
    def silu_and_mul_fq_kernel(
        x: fx.Tensor,  # (token_num*topk, inter_dim*2) bf16
        out_fp4: fx.Tensor,  # raw byte buffer for packed FP4 output
        out_scale_sorted: fx.Tensor,  # raw byte buffer for sorted E8M0 scales
        sorted_ids: fx.Tensor,  # (sorted_len,) i32
        num_valid_ids: fx.Tensor,  # (1,) i32
        token_num: Int32,  # host scalar
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c2_i32 = arith.constant(2, type=i32)
        c3_i32 = arith.constant(3, type=i32)
        c4_i32 = arith.constant(4, type=i32)
        c5_i32 = arith.constant(5, type=i32)
        c7_i32 = arith.constant(7, type=i32)
        c15_i32 = arith.constant(15, type=i32)
        c21_i32 = arith.constant(21, type=i32)
        c23_i32 = arith.constant(23, type=i32)
        c28_i32 = arith.constant(28, type=i32)
        c31_i32 = arith.constant(31, type=i32)
        c32_i32 = arith.constant(32, type=i32)
        c64_i32 = arith.constant(64, type=i32)
        c126_i32 = arith.constant(126, type=i32)
        c127_i32 = arith.constant(127, type=i32)
        c254_i32 = arith.constant(254, type=i32)
        c256_i32 = arith.constant(256, type=i32)
        c0xFF_i32 = arith.constant(0xFF, type=i32)
        c0x200000_i32 = arith.constant(0x200000, type=i32)
        c0xFF800000_i32 = arith.constant(0xFF800000, type=i32)
        c0x400000_i32 = arith.constant(0x400000, type=i32)
        c0x7FFFFF_i32 = arith.constant(0x7FFFFF, type=i32)
        c0x80000000_i32 = arith.constant(0x80000000, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)
        c1_f32 = arith.constant(1.0, type=f32)

        scale_cols_i32 = arith.constant(scale_cols, type=i32)
        inter_dim_i32 = arith.constant(inter_dim, type=i32)
        topk_i32 = arith.constant(topk, type=i32)
        n32_sort = scale_cols_i32 * c32_i32

        # Buffer resources
        in_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_fp4, max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(out_scale_sorted, max_size=True)
        tid_rsrc = buffer_ops.create_buffer_resource(sorted_ids, max_size=True)
        nv_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)

        num_valid = buffer_ops.buffer_load(nv_rsrc, c0_i32, vec_width=1, dtype=i32)
        token_num_i32 = ArithValue(token_num)
        bid_i32 = ArithValue(bid)

        row_in_range = arith.cmpi(CmpIPredicate.ult, bid_i32, num_valid)
        fused_tid_val = buffer_ops.buffer_load(
            tid_rsrc, bid_i32, vec_width=1, dtype=i32
        )
        mask24 = arith.constant(0xFFFFFF, type=i32)
        token_id = fused_tid_val & mask24
        slot_id = ArithValue(fused_tid_val) >> arith.constant(24, type=i32)
        t_ok = arith.cmpi(CmpIPredicate.ult, token_id, token_num_i32)
        s_ok = arith.cmpi(CmpIPredicate.ult, slot_id, topk_i32)
        is_valid = arith.andi(row_in_range, arith.andi(t_ok, s_ok))

        def _f32_to_e2m1(qx_f32):
            """Convert a scaled f32 value to fp4 (e2m1) 4-bit integer."""
            qx = qx_f32.bitcast(i32)
            s = qx & c0x80000000_i32
            e = (qx >> c23_i32) & c0xFF_i32
            m = qx & c0x7FFFFF_i32
            adj_exp = arith.maxsi(c126_i32 - e, c0_i32)
            m_denorm = (c0x400000_i32 | (m >> c1_i32)) >> adj_exp
            is_denorm = arith.cmpi(CmpIPredicate.ult, e, c127_i32)
            m = arith.select(is_denorm, m_denorm, m)
            e = arith.maxsi(e - c126_i32, c0_i32)
            combined = (e << c2_i32) | (m >> c21_i32)
            rounded = (combined + c1_i32) >> c1_i32
            e2m1 = arith.minui(rounded, c7_i32)
            return (s >> c28_i32) | e2m1

        thread_id = ArithValue(tid)

        COLS_PER_ITER = BLOCK_THREADS * VEC

        for iter_idx in range_constexpr(
            (inter_dim + COLS_PER_ITER - 1) // COLS_PER_ITER
        ):
            col0 = thread_id * arith.constant(VEC, type=i32) + arith.constant(
                iter_idx * COLS_PER_ITER, type=i32
            )

            col_valid = arith.cmpi(CmpIPredicate.ult, col0, inter_dim_i32)
            _if_col = scf.IfOp(col_valid)
            with ir.InsertionPoint(_if_col.then_block):

                _if_valid = scf.IfOp(is_valid, has_else=True)
                with ir.InsertionPoint(_if_valid.then_block):
                    in_row = token_id * topk_i32 + slot_id
                    # FP4 output in token order: row = token_id * topk + slot_id
                    out_row_byte_base = in_row * arith.constant(
                        inter_dim // 2, type=i32
                    )
                    fp4_byte_off = out_row_byte_base + (col0 >> c1_i32)
                    in_row_byte_base = in_row * arith.constant(
                        inter_dim * 2 * elem_bytes_bf16, type=i32
                    )
                    up_byte_offset = arith.constant(
                        inter_dim * elem_bytes_bf16, type=i32
                    )

                    gate_byte = in_row_byte_base + col0 * arith.constant(
                        elem_bytes_bf16, type=i32
                    )
                    up_byte = gate_byte + up_byte_offset
                    gate_dw = gate_byte >> c2_i32
                    up_dw = up_byte >> c2_i32
                    vec_dw = VEC * elem_bytes_bf16 // 4

                    gate_raw = buffer_ops.buffer_load(
                        in_rsrc, gate_dw, vec_width=vec_dw, dtype=i32
                    )
                    up_raw = buffer_ops.buffer_load(
                        in_rsrc, up_dw, vec_width=vec_dw, dtype=i32
                    )

                    vec_bf16_ty = T.vec(VEC, T.bf16)
                    vec_f32_ty = T.vec(VEC, f32)
                    if vec_dw == 1:
                        vec1_i32_ty = T.vec(1, i32)
                        gate_vec = vector.from_elements(vec1_i32_ty, [gate_raw])
                        up_vec = vector.from_elements(vec1_i32_ty, [up_raw])
                        gate_bf16 = vector.bitcast(vec_bf16_ty, gate_vec)
                        up_bf16 = vector.bitcast(vec_bf16_ty, up_vec)
                    else:
                        gate_bf16 = vector.bitcast(vec_bf16_ty, gate_raw)
                        up_bf16 = vector.bitcast(vec_bf16_ty, up_raw)
                    gate_f32 = gate_bf16.extf(vec_f32_ty)
                    up_f32 = up_bf16.extf(vec_f32_ty)

                    neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                    act_vals = []
                    for vi in range_constexpr(VEC):
                        g = vector.extract(
                            gate_f32, static_position=[vi], dynamic_position=[]
                        )
                        u = vector.extract(
                            up_f32, static_position=[vi], dynamic_position=[]
                        )
                        t = g * neg_log2e
                        emu = llvm.call_intrinsic(
                            f32, "llvm.amdgcn.exp2.f32", [t], [], []
                        )
                        den = c1_f32 + emu
                        sig = llvm.call_intrinsic(
                            f32, "llvm.amdgcn.rcp.f32", [den], [], []
                        )
                        act_vals.append(g * sig * u)

                    local_max = c0_f32
                    for vi in range_constexpr(VEC):
                        abs_v = llvm.call_intrinsic(
                            f32, "llvm.fabs.f32", [act_vals[vi]], [], []
                        )
                        local_max = arith.maximumf(local_max, abs_v)

                    for sh_dist in SHUFFLE_DISTS:
                        off = arith.constant(sh_dist, type=i32)
                        peer = local_max.shuffle_xor(off, c64_i32)
                        local_max = arith.maximumf(local_max, peer)

                    max_i32_v = local_max.bitcast(i32)
                    max_rounded = (max_i32_v + c0x200000_i32) & c0xFF800000_i32
                    exp_field = max_rounded >> c23_i32
                    e8m0_biased = arith.maxsi(exp_field - c2_i32, c0_i32)

                    quant_exp = c254_i32 - e8m0_biased
                    quant_scale = (quant_exp << c23_i32).bitcast(f32)

                    fp4_vals = []
                    for vi in range_constexpr(VEC):
                        scaled_v = act_vals[vi] * quant_scale
                        fp4_vals.append(_f32_to_e2m1(scaled_v))

                    packed_i32 = fp4_vals[0] | (fp4_vals[1] << c4_i32)
                    for k in range_constexpr(1, VEC // 2):
                        byte_k = fp4_vals[2 * k] | (fp4_vals[2 * k + 1] << c4_i32)
                        packed_i32 = packed_i32 | (
                            byte_k << arith.constant(k * 8, type=i32)
                        )

                    _pack_bytes = VEC // 2
                    if _pack_bytes == 1:
                        store_val = arith.TruncIOp(T.i8, packed_i32)
                        buffer_ops.buffer_store(
                            store_val, out_rsrc, fp4_byte_off, offset_is_bytes=True
                        )
                    elif _pack_bytes == 2:
                        store_val = arith.TruncIOp(T.i16, packed_i32)
                        buffer_ops.buffer_store(
                            store_val, out_rsrc, fp4_byte_off, offset_is_bytes=True
                        )
                    else:
                        buffer_ops.buffer_store(
                            packed_i32, out_rsrc, fp4_byte_off, offset_is_bytes=True
                        )

                    lane_in_blk = col0 & c31_i32
                    _if_sw = scf.IfOp(arith.cmpi(CmpIPredicate.eq, lane_in_blk, c0_i32))
                    with ir.InsertionPoint(_if_sw.then_block):
                        row_s = bid_i32
                        col_s = col0 >> c5_i32
                        d0 = row_s >> c5_i32
                        d1 = (row_s >> c4_i32) & c1_i32
                        d2 = row_s & c15_i32
                        d3 = col_s >> c3_i32
                        d4 = (col_s >> c2_i32) & c1_i32
                        d5 = col_s & c3_i32
                        s_byte_off = (
                            d0 * n32_sort
                            + d3 * c256_i32
                            + d5 * c64_i32
                            + d2 * c4_i32
                            + d4 * c2_i32
                            + d1
                        )
                        e8m0_i8 = arith.TruncIOp(T.i8, e8m0_biased)
                        buffer_ops.buffer_store(
                            e8m0_i8, scale_rsrc, s_byte_off, offset_is_bytes=True
                        )
                        scf.YieldOp([])
                    scf.YieldOp([])

                with ir.InsertionPoint(_if_valid.else_block):
                    # Padding row: skip FP4 write (stage2 gather-loads by token_id,
                    # so padding rows are never read). Only write zero scale.
                    lane_in_blk_p = col0 & c31_i32
                    _if_sw_p = scf.IfOp(
                        arith.cmpi(CmpIPredicate.eq, lane_in_blk_p, c0_i32)
                    )
                    with ir.InsertionPoint(_if_sw_p.then_block):
                        row_s_p = bid_i32
                        col_s_p = col0 >> c5_i32
                        d0_p = row_s_p >> c5_i32
                        d1_p = (row_s_p >> c4_i32) & c1_i32
                        d2_p = row_s_p & c15_i32
                        d3_p = col_s_p >> c3_i32
                        d4_p = (col_s_p >> c2_i32) & c1_i32
                        d5_p = col_s_p & c3_i32
                        s_byte_off_p = (
                            d0_p * n32_sort
                            + d3_p * c256_i32
                            + d5_p * c64_i32
                            + d2_p * c4_i32
                            + d4_p * c2_i32
                            + d1_p
                        )
                        c0_i8 = arith.TruncIOp(T.i8, c0_i32)
                        buffer_ops.buffer_store(
                            c0_i8, scale_rsrc, s_byte_off_p, offset_is_bytes=True
                        )
                        scf.YieldOp([])
                    scf.YieldOp([])
                scf.YieldOp([])

    @flyc.jit
    def launch_silu_and_mul_fq(
        x: fx.Tensor,
        out_fp4: fx.Tensor,
        out_scale_sorted: fx.Tensor,
        sorted_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        token_num: fx.Int32,
        num_sorted_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        idx_rows = arith.index_cast(T.index, num_sorted_rows)
        launcher = silu_and_mul_fq_kernel(
            x, out_fp4, out_scale_sorted, sorted_ids, num_valid_ids, token_num
        )
        launcher.launch(
            grid=(idx_rows, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_silu_and_mul_fq
