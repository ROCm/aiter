# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Normalize PA output and store packed partition partials."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from ..tensor_shim import buf_base_i64, buf_copy_store, ptr_buf_tensor
from ..utils import rcp_f32
from .traits import LOG2E, MFMA_MNK


class PaDecodeEpilogue:
    def __init__(self, ctx):
        self.ctx = ctx
        self.traits = ctx.traits

    def init_outputs(self):
        self.pmax = fx.recast_iter(fx.Float32, self.ctx.pmax_ptr)
        self.psum = fx.recast_iter(fx.Float32, self.ctx.psum_ptr)
        self.pout = fx.recast_iter(self.traits.Q_DTYPE, self.ctx.pout_ptr)
        if const_expr(self.traits.buffer_plan_output):
            # Widen before multiplying: packed storage can exceed 2 GiB.
            if const_expr(self.traits.batch_first_plan_grid):
                pout_slot = fx.Int64(self.ctx.kv_h) * fx.Int64(
                    gpu.grid_dim.x
                ) * fx.Int64(gpu.grid_dim.z) + fx.Int64(self.ctx.part)
            else:
                pout_slot = fx.Int64(self.ctx.kv_h) * fx.Int64(
                    gpu.grid_dim.x
                ) + fx.Int64(self.ctx.part)
            pout_slot_bytes = self.traits.TOTAL_ROWS * self.traits.head_dim * 2
            pout_base = buf_base_i64(self.ctx.pout_ptr) + pout_slot * fx.Int64(
                pout_slot_bytes
            )
            self.pout_buffer = ptr_buf_tensor(
                pout_base,
                self.traits.Q_DTYPE,
                n=self.traits.TOTAL_ROWS * self.traits.head_dim,
                unit_elems=self.traits.OP_ELEMS,
                # BF16 views may be only 2-byte aligned, even for 8-byte stores.
                unit_stride=1,
                num_records_bytes=pout_slot_bytes,
            )

    def _emit(self, o_norm, sub, global_row):
        if const_expr(self.traits.buffer_plan_output):
            # Row guards keep the full 8-byte store within this slot.
            pout_offset = global_row * self.traits.head_dim + sub * self.traits.OP_ELEMS
            buf_copy_store(
                self.pout_buffer,
                pout_offset,
                o_norm,
                elem=self.traits.Q_DTYPE,
                unit_elems=self.traits.OP_ELEMS,
                cache_modifier=0,
            )
        else:
            base = self.ctx.partial_slot * self.traits.TOTAL_ROWS + global_row
            pout_offset = base * self.traits.head_dim + sub * self.traits.OP_ELEMS
            fx.ptr_store(o_norm, fx.add_offset(self.pout, pout_offset))

    @flyc.jit
    def store(self, o_final):
        # Keep pointer writes as SSA operands of the row/warp guards.
        pmax = self.pmax
        psum = self.psum
        # Each lane stores four head-dim values for one query row.
        inv_fp8 = fx.Float32(1.0 / self.traits.FP8_MAX)
        for m in range_constexpr(self.traits.M_TILES):
            # Flat (mtp, gqa) query-row for this lane.
            row = m * MFMA_MNK + self.ctx.lane16
            global_row = self.ctx.query_begin * self.traits.query_group_size + row
            l_row = o_final[self.traits.l_slot(m)]
            safe_l = (l_row > self.ctx.ZERO_F).select(l_row, fx.Float32(1.0))
            inv_l = fx.Float32(rcp_f32(safe_l))
            if const_expr(self.traits.per_token_kv):
                o_scale = inv_l
            else:
                o_scale = inv_l * (self.ctx.v_scale_f * inv_fp8)
            o_scale_b = fx.Vector.from_elements(
                [o_scale], dtype=fx.Float32
            ).broadcast_to(self.traits.OP_ELEMS)

            for vh in range_constexpr(self.traits.VHE_CHUNKS):
                o_slot = self.traits.o_slot(m, vh)
                o_norm = (o_final[o_slot] * o_scale_b).to(self.traits.Q_DTYPE)
                head_base = (
                    vh * (self.traits.NWARP * MFMA_MNK)
                    + self.ctx.warp * MFMA_MNK
                    + self.ctx.rgroup * self.traits.OP_ELEMS
                )
                sub = head_base // self.traits.OP_ELEMS
                if row < self.traits.CTA_ROWS:
                    self._emit(o_norm, sub, global_row)

            if self.ctx.warp == 0 and self.ctx.rgroup == 0:
                base = self.ctx.partial_slot * self.traits.TOTAL_ROWS + global_row
                if row < self.traits.CTA_ROWS:
                    # The shared reducer expects maxima in natural-log units.
                    pmax[base] = o_final[self.traits.m_slot(m)] * fx.Float32(
                        1.0 / LOG2E
                    )
                    psum[base] = l_row
