# SPDX-License-Identifier: MIT
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Split-K combine pass and workspace sizing."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.fmha_gfx950.pipeline import (
    _cu_load,
    _make_ws_rsrc,
    _store_lse,
)


class DualwaveSplitKCombineContext:
    """Shared per-kernel state for the split-K combine pass."""

    def __init__(
        self,
        traits_or_ctx,
        O=None,
        WS=None,
        batch_size=None,
        seq_len=None,
        stride_o_n=None,
        CuSeqQ=None,
        LSE=None,
        lse_stride_h=None,
    ):
        if isinstance(traits_or_ctx, DualwaveSplitKCombineContext):
            self.__dict__.update(traits_or_ctx.__dict__)
            self.ctx_ref = getattr(traits_or_ctx, "ctx_ref", traits_or_ctx)
            return

        self.ctx_ref = self
        self.traits = traits_or_ctx
        self.O = O
        self.WS = WS
        self.CuSeqQ = CuSeqQ
        self.LSE = LSE
        self.lse_stride_h = lse_stride_h
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.stride_o_n = stride_o_n

    def init_types_and_constants(self):
        self.elem_dtype = fx.BFloat16  # fp8 in, bf16 out
        self.fm_fast = fx.arith.FastMathFlags.fast
        self.c_zero_f = fx.Float32(0.0)
        self.c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)

    def init_runtime_indices(self):
        self.seq_len_v = fx.Index(self.seq_len)
        if const_expr(self.traits.RETURN_LSE):
            self.lse_stride_h_v = fx.Index(self.lse_stride_h)
        self.stride_o_n_v = fx.Index(self.stride_o_n)
        self.batch_size_v = fx.Index(self.batch_size)

    def init_thread_mapping(self, combine_rows_per_block, combine_lanes_per_row):
        traits = self.traits
        self.tid = fx.Index(gpu.thread_idx.x)
        self.blk = fx.Index(gpu.block_idx.x)
        self.batch_idx = fx.Index(gpu.block_idx.y)
        self.col = (self.tid % combine_lanes_per_row) * 4
        rows_per_batch = self.seq_len_v * traits.NUM_HEADS_Q
        row_raw = self.blk * combine_rows_per_block + self.tid // combine_lanes_per_row
        threads_in_use = combine_rows_per_block * combine_lanes_per_row
        self.row = (self.tid < threads_in_use).select(row_raw, rows_per_batch)
        self.row_valid = self.row < rows_per_batch
        self.q_head_idx = self.row // self.seq_len_v
        self.seq_idx = self.row % self.seq_len_v

    def init_workspace(self):
        traits = self.traits
        z_total = self.batch_size_v * traits.NUM_KV_SPLITS
        self.ws_opart_per_split_elems = (
            traits.NUM_HEADS_Q * self.seq_len_v * traits.HEAD_DIM_V // 2
        )
        self.ws_ml_per_split_elems = traits.NUM_HEADS_Q * self.seq_len_v
        self.ws_opart_per_split_bytes = self.ws_opart_per_split_elems * 4
        self.ws_ml_per_split_bytes = self.ws_ml_per_split_elems * 4
        self.ws_mrow_abs_bytes = z_total * self.ws_opart_per_split_bytes
        self.ws_lrow_abs_bytes = (
            self.ws_mrow_abs_bytes + z_total * self.ws_ml_per_split_bytes
        )
        self.local_ml_idx = self.q_head_idx * self.seq_len_v + self.seq_idx
        self.local_o_base = (
            (self.q_head_idx * self.seq_len_v + self.seq_idx) * traits.HEAD_DIM_V // 2
        )
        self.ws_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(self.WS)))

    def init_descriptors(self):
        if const_expr(self.traits.VARLEN):
            _cuq_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(self.CuSeqQ), fx.make_layout(1, 1)
            )
            _cu_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            _cu_v1i32 = Vec.make_type(1, fx.Int32)
            q_tok_base = _cu_load(_cuq_div, self.batch_idx, _cu_atom, _cu_v1i32)
            q_tok_end = _cu_load(_cuq_div, self.batch_idx + 1, _cu_atom, _cu_v1i32)
            self.q_tok_base = q_tok_base
            self.seqlen_q_b = q_tok_end - q_tok_base
            batch_byte_off = q_tok_base * self.stride_o_n_v * 2
            nrec_bytes = (q_tok_end - q_tok_base) * self.stride_o_n_v * 2
        else:
            per_batch_elems = self.seq_len_v * self.stride_o_n_v
            batch_byte_off = self.batch_idx * per_batch_elems * 2
            nrec_bytes = per_batch_elems * 2
        self.o_nrec_bytes = nrec_bytes
        self.o_rsrc = buffer_ops.create_buffer_resource_from_addr(
            as_mlir_value(
                fx.Int64(fx.ptrtoint(fx.get_iter(self.O))) + fx.Int64(batch_byte_off)
            ),
            num_records_bytes=as_mlir_value(fx.Int64(nrec_bytes)),
        )
        self.load_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)

    def workspace_resource(self, byte_offset, nrec_bytes):
        return _make_ws_rsrc(self.ws_base_i64, byte_offset, nrec_bytes)

    def split_z(self, split_i):
        return self.batch_idx * self.traits.NUM_KV_SPLITS + split_i

    def opart_resource(self, split_z):
        return self.workspace_resource(
            split_z * self.ws_opart_per_split_bytes, self.ws_opart_per_split_bytes
        )

    def mrow_resource(self, split_z):
        return self.workspace_resource(
            self.ws_mrow_abs_bytes + split_z * self.ws_ml_per_split_bytes,
            self.ws_ml_per_split_bytes,
        )

    def lrow_resource(self, split_z):
        return self.workspace_resource(
            self.ws_lrow_abs_bytes + split_z * self.ws_ml_per_split_bytes,
            self.ws_ml_per_split_bytes,
        )


class DualwaveSplitKCombineHelper(DualwaveSplitKCombineContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_ml_rows(self):
        m_s = []
        l_s = []
        for i in range_constexpr(self.traits.NUM_KV_SPLITS):
            split_z_i = self.split_z(i)
            m_f32 = buffer_ops.buffer_load(
                self.mrow_resource(split_z_i),
                as_mlir_value(fx.Int32(self.local_ml_idx)),
                vec_width=1,
                dtype=T.f32,
            )
            l_f32 = buffer_ops.buffer_load(
                self.lrow_resource(split_z_i),
                as_mlir_value(fx.Int32(self.local_ml_idx)),
                vec_width=1,
                dtype=T.f32,
            )
            m_s.append(m_f32)
            l_s.append(l_f32)
        return m_s, l_s

    def reduce_m_max(self, m_s):
        m_max = m_s[0]
        for i in range_constexpr(self.traits.NUM_KV_SPLITS - 1):
            m_max = fx.maxnumf(m_max, m_s[i + 1])
        return m_max

    def init_accumulators(self):
        return as_mlir_value(self.c_zero_v4f32), as_mlir_value(self.c_zero_f)

    def accumulate_split(self, acc, den, split_i, m_i, l_i, m_max):
        orsrc_i = self.opart_resource(self.split_z(split_i))
        local_o_idx_i = self.local_o_base + self.col // 2

        @flyc.jit
        def _accum_split(acc, den):
            if fx.Float32(l_i) > fx.Float32(0.0):
                w = rocdl.exp2(T.f32, as_mlir_value(m_i - m_max))
                wl = w * l_i
                den = den + wl
                o2_raw = buffer_ops.buffer_load(
                    orsrc_i,
                    as_mlir_value(fx.Int32(local_o_idx_i)),
                    vec_width=2,
                    dtype=T.i32,
                )
                o2_i32 = ir.Value(o2_raw)
                o4 = Vec(o2_i32, (2,), fx.Int32).bitcast(self.elem_dtype).to(fx.Float32)
                w4 = Vec.from_elements([fx.Float32(wl)], fx.Float32).broadcast_to(4)
                acc = acc + w4 * o4
            return acc, den

        return _accum_split(acc, den)

    def accumulate_splits(self, m_s, l_s, m_max):
        acc, den = self.init_accumulators()
        for i in range_constexpr(self.traits.NUM_KV_SPLITS):
            acc, den = self.accumulate_split(acc, den, i, m_s[i], l_s[i], m_max)
        return acc, den

    def pack_output(self, acc, den):
        inv_rcp = rocdl.rcp(T.f32, den)
        inv = (fx.Float32(den) > self.c_zero_f).select(inv_rcp, self.c_zero_f)
        inv4 = Vec.from_elements([fx.Float32(inv)], fx.Float32).broadcast_to(4)
        out4 = Vec(acc * inv4, (4,), fx.Float32)
        lo = rocdl.cvt_pk_bf16_f32(out4[0], out4[1])
        hi = rocdl.cvt_pk_bf16_f32(out4[2], out4[3])
        return Vec.from_elements([fx.Int32(lo), fx.Int32(hi)], fx.Int32)

    def store_lse(self, m_max, den):
        # row_valid only bounds seq_idx by max_seqlen_q; varlen batches are shorter.
        in_range = self.row_valid
        if const_expr(self.traits.VARLEN):
            in_range = in_range & (self.seq_idx < self.seqlen_q_b)
        _store_lse(self, self.seq_idx, m_max, den, in_range, self.col == 0)

    def store_output(self, o_pack):
        o_global = (
            self.seq_idx * self.stride_o_n_v
            + self.q_head_idx * self.traits.HEAD_DIM_V
            + self.col
        )
        # Out-of-range rows aim past num_records, which the buffer drops.
        o_off = self.row_valid.select(o_global * 2, self.o_nrec_bytes)
        buffer_ops.buffer_store(
            o_pack.ir_value(),
            self.o_rsrc,
            as_mlir_value(fx.Int32(o_off)),
            offset_is_bytes=True,
        )


def dualwave_splitk_workspace_elems(
    batch_size, num_heads, seq_len, num_kv_splits, head_dim=128
):
    """fp32 elements needed for the split-K workspace: O_partial + Mrow + Lrow.

    O_partial is stored as kernel-native 16-bit (bf16/fp16), two columns per
    fp32 slot; Mrow/Lrow stay fp32.
    """
    rows = batch_size * num_kv_splits * num_heads * seq_len
    return rows * (head_dim // 2) + 2 * rows
