# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL Flash Attention Backward Preprocess kernel.

Computes per-row dot product:
    delta[b, h, m] = sum_d( O[b, h, m, d] * dO[b, h, m, d] )

This is Pass 1 of the backward, needed before the main dK/dV/dQ kernel.
See: Flash Attention backward pass, equation: dS = P * (dP - delta).

Grid:  (seqlen_q, batch, num_heads)
Block: (BLOCK_THREADS, 1, 1)

Each workgroup owns one (batch, head, seq_row) triple.
Threads split over head_dim: each thread accumulates HEAD_DIM/BLOCK_THREADS
elements, then the warp reduces to a single scalar stored to delta.

Assumptions for this initial implementation:
  - HEAD_DIM is a multiple of BLOCK_THREADS
  - BLOCK_THREADS <= WARP_SIZE (one warp per block — no cross-warp LDS needed)
  - stride_ok and stride_dok are both 1 (contiguous D dimension, typical BSHD layout)
  - O and dO are bf16 or fp16
  - delta output is fp32
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, buffer_ops
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl._mlir import ir
from flydsl._mlir.dialects import gpu, scf

from .kernels_common import get_warp_size


def build_fmha_bwd_preprocess_module(
    head_dim: int,
    dtype: str = "bf16",  # "bf16" or "fp16"
):
    """Return a JIT launcher for the FMHA backward preprocess kernel.

    Parameters
    ----------
    head_dim : int
        Head dimension D (e.g. 128). Must be a multiple of WARP_SIZE.
    dtype : str
        Input dtype of O and dO tensors ("bf16" or "fp16").
    """
    WARP_SIZE = get_warp_size()
    BLOCK_THREADS = WARP_SIZE  # one warp per block, no cross-warp reduction needed

    assert head_dim % BLOCK_THREADS == 0, (
        f"head_dim ({head_dim}) must be a multiple of BLOCK_THREADS ({BLOCK_THREADS})"
    )
    assert dtype in ("bf16", "fp16"), f"unsupported dtype: {dtype!r}"

    ELEM_BYTES = 2  # bf16 / fp16 are 2 bytes each
    ELEMS_PER_THREAD = head_dim // BLOCK_THREADS  # elements each thread covers

    @flyc.kernel
    def fmha_bwd_preprocess_kernel(
        o_ptr: fx.Tensor,      # forward output   [..., head_dim]
        do_ptr: fx.Tensor,     # output gradient  [..., head_dim]
        delta_ptr: fx.Tensor,  # output: rowsum(O * dO), fp32
        # O strides (in elements)
        stride_ob: Int32,
        stride_oh: Int32,
        stride_om: Int32,
        stride_ok: Int32,
        # dO strides (in elements) — may differ from O if layouts differ
        stride_dob: Int32,
        stride_doh: Int32,
        stride_dom: Int32,
        stride_dok: Int32,
        # delta strides (in elements)
        stride_deltab: Int32,
        stride_deltah: Int32,
        stride_deltam: Int32,
        seqlen_q: Int32,
    ):
        seq_idx   = ArithValue(fx.block_idx.x)
        batch_idx = ArithValue(fx.block_idx.y)
        head_idx  = ArithValue(fx.block_idx.z)
        tid       = ArithValue(fx.thread_idx.x)

        i32 = T.i32
        f32 = T.f32

        seqlen_q_v = ArithValue(seqlen_q)

        # Guard: skip workgroups beyond seqlen_q
        valid_row = arith.cmpi(CmpIPredicate.ult, seq_idx, seqlen_q_v)
        _if_valid = scf.IfOp(valid_row)
        with ir.InsertionPoint(_if_valid.then_block):
            o_rsrc  = buffer_ops.create_buffer_resource(o_ptr,  max_size=True)
            do_rsrc = buffer_ops.create_buffer_resource(do_ptr, max_size=True)

            stride_ob_v  = ArithValue(stride_ob)
            stride_oh_v  = ArithValue(stride_oh)
            stride_om_v  = ArithValue(stride_om)
            stride_ok_v  = ArithValue(stride_ok)
            stride_dob_v = ArithValue(stride_dob)
            stride_doh_v = ArithValue(stride_doh)
            stride_dom_v = ArithValue(stride_dom)
            stride_dok_v = ArithValue(stride_dok)

            c_elem_bytes = arith.constant(ELEM_BYTES, type=i32)
            c_16         = arith.constant(16, type=i32)
            c_mask16     = arith.constant(0xFFFF, type=i32)
            c_1_i32      = arith.constant(1, type=i32)

            # Base element offsets for this (batch, head, row)
            o_elem_base  = (batch_idx * stride_ob_v
                            + head_idx * stride_oh_v
                            + seq_idx  * stride_om_v)
            do_elem_base = (batch_idx * stride_dob_v
                            + head_idx * stride_doh_v
                            + seq_idx  * stride_dom_v)

            acc = arith.constant(0.0, type=f32)

            # Each thread covers ELEMS_PER_THREAD elements, stride = BLOCK_THREADS
            for iter_idx in range_constexpr(ELEMS_PER_THREAD):
                col = tid + arith.constant(iter_idx * BLOCK_THREADS, type=i32)

                # Byte offsets for O and dO elements
                # stride_ok / stride_dok expected to be 1 (contiguous D)
                o_byte_off  = (o_elem_base  + col * stride_ok_v)  * c_elem_bytes
                do_byte_off = (do_elem_base + col * stride_dok_v) * c_elem_bytes

                # Dword offset and hi/lo half selection
                o_dw_off  = o_byte_off  >> arith.constant(2, type=i32)
                do_dw_off = do_byte_off >> arith.constant(2, type=i32)

                # is_hi: 1 if the bf16 sits in the upper 16 bits of the dword
                o_is_hi  = (o_byte_off  >> c_1_i32) & c_1_i32
                do_is_hi = (do_byte_off >> c_1_i32) & c_1_i32

                o_raw  = buffer_ops.buffer_load(o_rsrc,  o_dw_off,  vec_width=1, dtype=i32)
                do_raw = buffer_ops.buffer_load(do_rsrc, do_dw_off, vec_width=1, dtype=i32)

                # Extract bf16 into f32 by placing the 16-bit mantissa in the
                # upper half of a 32-bit word (bf16 and f32 share the same exponent)
                o_shifted  = (o_raw  >> (o_is_hi  * c_16)) & c_mask16
                do_shifted = (do_raw >> (do_is_hi * c_16)) & c_mask16

                o_f32  = arith.bitcast(f32, o_shifted  << c_16)
                do_f32 = arith.bitcast(f32, do_shifted << c_16)

                prod = arith.mulf(o_f32, do_f32)
                acc  = arith.addf(acc, prod)

            # Intra-warp reduction (sum across all WARP_SIZE lanes via xor shuffle).
            # Python for-loop over constant offsets — FlyDSL unrolls it at compile time.
            # (A while-loop would become scf.while, making `sh` an SSA value that
            # arith.constant cannot accept.)
            width_i32 = arith.constant(WARP_SIZE, type=i32)
            for sh in [WARP_SIZE >> i for i in range(1, WARP_SIZE.bit_length())]:
                off  = arith.constant(sh, type=i32)
                peer = ArithValue(
                    gpu.ShuffleOp(acc, off, width_i32, mode="xor").shuffleResult
                )
                acc = arith.addf(acc, peer)

            # Lane 0 writes the reduced delta value (f32 → dword store)
            is_lane0 = arith.cmpi(CmpIPredicate.eq, tid, arith.constant(0, type=i32))
            _if_lane0 = scf.IfOp(is_lane0)
            with ir.InsertionPoint(_if_lane0.then_block):
                delta_rsrc = buffer_ops.create_buffer_resource(delta_ptr, max_size=True)

                stride_deltab_v = ArithValue(stride_deltab)
                stride_deltah_v = ArithValue(stride_deltah)
                stride_deltam_v = ArithValue(stride_deltam)

                # delta is f32: 4 bytes per element, stride in elements → dword offset directly
                delta_dw_off = (batch_idx * stride_deltab_v
                                + head_idx * stride_deltah_v
                                + seq_idx  * stride_deltam_v)

                acc_i32 = arith.bitcast(i32, acc)
                buffer_ops.buffer_store(acc_i32, delta_rsrc, delta_dw_off)
                scf.YieldOp([])

            scf.YieldOp([])

    @flyc.jit
    def launch_fmha_bwd_preprocess(
        o: fx.Tensor,
        do: fx.Tensor,
        delta: fx.Tensor,
        stride_ob:  fx.Int32,
        stride_oh:  fx.Int32,
        stride_om:  fx.Int32,
        stride_ok:  fx.Int32,
        stride_dob: fx.Int32,
        stride_doh: fx.Int32,
        stride_dom: fx.Int32,
        stride_dok: fx.Int32,
        stride_deltab: fx.Int32,
        stride_deltah: fx.Int32,
        stride_deltam: fx.Int32,
        seqlen_q: fx.Int32,
        batch:     fx.Int32,
        num_heads: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        sq   = arith.index_cast(T.index, seqlen_q)
        b    = arith.index_cast(T.index, batch)
        h    = arith.index_cast(T.index, num_heads)

        launcher = fmha_bwd_preprocess_kernel(
            o, do, delta,
            stride_ob, stride_oh, stride_om, stride_ok,
            stride_dob, stride_doh, stride_dom, stride_dok,
            stride_deltab, stride_deltah, stride_deltam,
            seqlen_q,
        )
        launcher.launch(
            grid=(sq, b, h),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fmha_bwd_preprocess
