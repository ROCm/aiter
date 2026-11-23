# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Merge the KV-split partials of the fused QSA sparse decode.

Partials, per (row, kv head, split, head-in-group): ``part_m`` is the row max in
log2 units, ``part_l`` the softmax denominator at that max, and ``part_o`` the
split's normalized output. One CTA merges one (row, query head); thread ``d``
owns output dim ``d``. The split count is a compile-time constant so every
partial load is issued up front instead of one dependent load per loop trip.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import gpu, range_constexpr

from .sparse_decode_attn import HEAD_DIM


@functools.cache
def build_sparse_decode_reduce(*, group_size: int, num_splits: int):
    """Compile-time specialization of the split-merge kernel and its launcher."""
    G = group_size
    NS = num_splits

    def _gptr(base, elem_ty, align):
        ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Global, align)
        return fx.recast_iter(ty, base)

    @flyc.kernel(known_block_size=(HEAD_DIM, 1, 1))
    def sparse_decode_reduce_kernel(
        part_m_ptr: fx.Pointer,
        part_l_ptr: fx.Pointer,
        part_o_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        num_kv_heads: fx.Int32,
    ):
        d = fx.Int32(gpu.thread_id("x"))
        row = fx.Int32(gpu.block_id("x"))
        q_head = fx.Int32(gpu.block_id("y"))
        kvh = q_head // G
        h = q_head % G
        part_m = _gptr(part_m_ptr, fx.Float32, 4)
        part_l = _gptr(part_l_ptr, fx.Float32, 4)
        part_o = _gptr(part_o_ptr, fx.Float32, 4)
        out = _gptr(out_ptr, fx.BFloat16, 2)
        first = (row * num_kv_heads + kvh) * (NS * G) + h
        neg_inf = fx.Float32(float("-inf"))
        zero_f = fx.Float32(0.0)

        parts = [first + sp * G for sp in range_constexpr(NS)]
        m_split = [fx.ptr_load(fx.add_offset(part_m, p)) for p in parts]
        l_split = [fx.ptr_load(fx.add_offset(part_l, p)) for p in parts]
        o_split = [
            fx.ptr_load(fx.add_offset(part_o, fx.Int64(p) * HEAD_DIM + fx.Int64(d)))
            for p in parts
        ]
        m_max = m_split[0]
        for sp in range_constexpr(1, NS):
            m_max = fx.maxnumf(m_max, m_split[sp])
        m_safe = (m_max > neg_inf).select(m_max, zero_f)
        o_sum = zero_f
        l_sum = zero_f
        for sp in range_constexpr(NS):
            weight = l_split[sp] * fx.exp2(m_split[sp] - m_safe, fastmath="fast")
            o_sum = o_sum + weight * o_split[sp]
            l_sum = l_sum + weight
        o_val = (l_sum > zero_f).select(o_sum / l_sum, zero_f)
        num_q_heads = num_kv_heads * G
        fx.ptr_store(
            o_val.to(fx.BFloat16),
            fx.add_offset(out, (row * num_q_heads + q_head) * HEAD_DIM + d),
        )

    @flyc.jit
    def launch(
        part_m_ptr: fx.Pointer,
        part_l_ptr: fx.Pointer,
        part_o_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        num_rows: fx.Int32,
        num_kv_heads: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        with CompilationContext.compile_hints({"fastmath": "contract"}):
            sparse_decode_reduce_kernel(
                part_m_ptr,
                part_l_ptr,
                part_o_ptr,
                out_ptr,
                num_kv_heads,
            ).launch(
                grid=(num_rows, num_kv_heads * G, 1),
                block=(HEAD_DIM, 1, 1),
                stream=stream,
            )

    return launch
