# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused swiglu_and_mul kernel for interleaved (N0, 2, NLane) layout (FlyDSL).

Input layout (from cktile split_k with a16w4 interleave preshuffle):
  For each row of inter_dim*2 columns, data is arranged as:
    [gate_block0(NLane), up_block0(NLane), gate_block1(NLane), up_block1(NLane), ...]
  where NLane = 16, N0 = inter_dim // NLane.

Computes per-element:
    out = clamp(gate, max=limit) * sigmoid(alpha * clamp(gate, max=limit))
          * (clamp(up, -limit, limit) + 1)

Grid:  (rows, 1, 1)
Block: (BLOCK_THREADS, 1, 1)

Each thread processes 2 consecutive output bf16 elements (= 1 dword)
per iteration, avoiding read-modify-write races.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, range_constexpr
from flydsl.expr.typing import Int32, T

from aiter.ops.flydsl.kernels import buffer_ops

BLOCK_THREADS = 256
NLANE = 16
ALPHA = 1.702
LIMIT = 7.0


def _swiglu_scalar(g, u, limit_f32, neg_limit_f32, neg_alpha_log2e, c1_f32):
    """Compute one swiglu element in f32."""
    f32 = T.f32
    g_clamped = arith.minimumf(g, limit_f32)
    u_clamped = arith.maximumf(arith.minimumf(u, limit_f32), neg_limit_f32)

    t = g_clamped * neg_alpha_log2e
    emu = fx.rocdl.exp2(f32, t)
    den = c1_f32 + emu
    sig = fx.rocdl.rcp(f32, den)
    return g_clamped * sig * (u_clamped + c1_f32)


def build_swiglu_and_mul_module(inter_dim: int):
    """Return a JIT launcher for fused swiglu_and_mul on interleaved input.

    Parameters
    ----------
    inter_dim : int
        Output columns (after activation). Input has inter_dim*2 cols
        in interleaved (N0, 2, NLane) layout.
    """
    elem_bytes = 2  # bf16
    assert inter_dim % NLANE == 0
    out_dwords = inter_dim // 2
    DWORDS_PER_ITER = BLOCK_THREADS

    @flyc.kernel
    def swiglu_and_mul_kernel(
        x: fx.Tensor,  # (rows, inter_dim*2) bf16, interleaved layout
        out: fx.Tensor,  # (rows, inter_dim)   bf16
        num_rows: Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32 = T.f32

        c1_f32 = arith.constant(1.0, type=f32)
        limit_f32 = arith.constant(LIMIT, type=f32)
        neg_limit_f32 = arith.constant(-LIMIT, type=f32)
        neg_alpha_log2e = arith.constant(-ALPHA * 1.4426950408889634, type=f32)

        if bid < num_rows:
            in_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)

            in_row_dw_base = bid * (inter_dim * 2 * elem_bytes // 4)
            out_row_dw_base = bid * (inter_dim * elem_bytes // 4)

            for iter_idx in range_constexpr(
                (out_dwords + DWORDS_PER_ITER - 1) // DWORDS_PER_ITER
            ):
                dw_idx = tid + iter_idx * DWORDS_PER_ITER

                if dw_idx < out_dwords:
                    # 2 output cols per dword: col_lo = dw_idx*2, col_hi = dw_idx*2+1
                    col_lo = dw_idx * 2
                    col_hi = col_lo + 1

                    results = []
                    for col in [col_lo, col_hi]:
                        # Interleave addressing:
                        #   out_col -> block = out_col / NLane, lane = out_col % NLane
                        #   gate col = block * 2 * NLane + lane
                        #   up   col = block * 2 * NLane + NLane + lane
                        block = col // NLANE
                        lane = col - block * NLANE
                        gate_col = block * (2 * NLANE) + lane
                        up_col = gate_col + NLANE

                        # byte offset -> dword offset for buffer_load
                        gate_byte_off = gate_col * elem_bytes
                        up_byte_off = up_col * elem_bytes
                        gate_dw_off = in_row_dw_base + (gate_byte_off >> 2)
                        up_dw_off = in_row_dw_base + (up_byte_off >> 2)

                        gate_raw_dw = fx.Int32(
                            buffer_ops.buffer_load(
                                in_rsrc, gate_dw_off, vec_width=1, dtype=T.i32
                            )
                        )
                        up_raw_dw = fx.Int32(
                            buffer_ops.buffer_load(
                                in_rsrc, up_dw_off, vec_width=1, dtype=T.i32
                            )
                        )

                        # Extract correct bf16 half from dword
                        gate_is_hi = (gate_byte_off >> 1) & 1
                        up_is_hi = (up_byte_off >> 1) & 1

                        gate_shifted = gate_raw_dw >> (gate_is_hi * 16)
                        up_shifted = up_raw_dw >> (up_is_hi * 16)

                        # Place bf16 in upper 16 bits of f32 (bf16 -> f32 bitcast)
                        g_f32 = arith.bitcast(
                            f32, ((gate_shifted & 0xFFFF) << 16).ir_value()
                        )
                        u_f32 = arith.bitcast(
                            f32, ((up_shifted & 0xFFFF) << 16).ir_value()
                        )

                        r = _swiglu_scalar(
                            g_f32,
                            u_f32,
                            limit_f32,
                            neg_limit_f32,
                            neg_alpha_log2e,
                            c1_f32,
                        )
                        r_bf16 = arith.trunc_f(T.bf16, r)
                        r_i16 = arith.bitcast(T.i16, r_bf16)
                        results.append(fx.Int32(arith.extui(T.i32, r_i16)))

                    # Pack 2 bf16 results into one dword: lo | (hi << 16)
                    packed = results[0] | (results[1] << 16)
                    out_dw = out_row_dw_base + dw_idx
                    buffer_ops.buffer_store(packed, out_rsrc, out_dw)

    @flyc.jit
    def launch_swiglu_and_mul(
        x: fx.Tensor,
        out: fx.Tensor,
        num_rows: fx.Int32,
        stream: fx.Stream,
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        idx_rows = fx.Int64(num_rows)
        launcher = swiglu_and_mul_kernel(x, out, num_rows)
        launcher.launch(
            grid=(idx_rows, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_swiglu_and_mul
