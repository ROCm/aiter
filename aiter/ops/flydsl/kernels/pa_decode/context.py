# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Per-CTA operands, thread ownership and partition bounds for PA decode.

Initialization stays explicit so the pipeline controls the order of global loads.
Loop-carried softmax and output values are passed as SSA values, not stored here.
"""

import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr

from .traits import LOG2E, MFMA_MNK, WAVE


class PaDecodeContext:
    def __init__(
        self,
        traits,
        shared_storage,
        pmax_ptr,
        psum_ptr,
        pout_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        key_scale_ptr,
        value_scale_ptr,
        max_blocks_per_seq,
        stride_ks_block,
        stride_ks_head,
        stride_q_row,
        stride_q_head,
        num_sequences,
        planned_seq,
        planned_start,
        planned_end,
        planned_context,
    ):
        self.traits = traits
        self.shared_storage = shared_storage
        self.pmax_ptr = pmax_ptr
        self.psum_ptr = psum_ptr
        self.pout_ptr = pout_ptr
        self.query_ptr = query_ptr
        self.key_cache_ptr = key_cache_ptr
        self.value_cache_ptr = value_cache_ptr
        self.block_tables_ptr = block_tables_ptr
        self.key_scale_ptr = key_scale_ptr
        self.value_scale_ptr = value_scale_ptr
        self.max_blocks_per_seq = max_blocks_per_seq
        self.stride_ks_block = stride_ks_block
        self.stride_ks_head = stride_ks_head
        self.stride_q_row = stride_q_row
        self.stride_q_head = stride_q_head
        self.num_sequences = num_sequences
        self.planned_seq = planned_seq
        self.planned_start = planned_start
        self.planned_end = planned_end
        self.planned_context = planned_context

    def init_thread_mapping(self):
        """Resolve the physical grid to a sequence, KV head and partial slot."""
        self.tid = fx.Int32(gpu.thread_id("x"))
        self.warp = self.tid // WAVE  # 0..NWARP-1
        self.lane = self.tid - self.warp * WAVE  # 0..63
        self.seq = self.planned_seq
        self.kv_query = fx.Int32(gpu.block_id("y"))
        self.kv_h = self.kv_query // self.traits.query_splits
        self.query_begin = (
            self.kv_query % self.traits.query_splits
        ) * self.traits.QUERIES_PER_CTA
        self.n_kv = fx.Int32(gpu.grid_dim.y) // self.traits.query_splits
        if const_expr(self.traits.batch_first_plan_grid):
            self.part = fx.Int32(
                fx.Uint32(gpu.block_id("x")) * fx.Uint32(gpu.grid_dim.z)
                + fx.Uint32(gpu.block_id("z"))
            )
            # Physical x groups packed slots, not sequences.
            capacity = fx.Int32(fx.Uint32(gpu.grid_dim.x) * fx.Uint32(gpu.grid_dim.z))
            self.partial_slot = self.kv_h * capacity + self.part
        else:
            self.part = fx.Int32(gpu.block_id("x"))
            self.partial_slot = self.kv_h * fx.Int32(gpu.grid_dim.x) + self.part

    def init_query_mapping(self):
        self.rgroup = (
            self.lane // MFMA_MNK
        )  # 0..3: quarter-wave (paired with warp -> query row)
        self.lane16 = (
            self.lane - self.rgroup * MFMA_MNK
        )  # 0..15: this row's head-dim chunk index
        self.qh_local = (
            self.warp * 4 + self.rgroup
        )  # 0..15: this thread's query row within an M-tile

    def init_context_length(self):
        self.context_len = self.planned_context

    def init_partition_bounds(self):
        self.num_pages = (
            self.context_len + self.traits.block_size - 1
        ) // self.traits.block_size  # pages this sequence really owns
        self.part_start = self.planned_start
        self.part_end = self.planned_end

    def init_scales(self, key_scale, value_scale):
        if const_expr(self.traits.per_token_kv):
            self.scale_qk = fx.Float32(self.traits.softmax_scale * LOG2E)
        else:
            self.scale_qk = fx.Float32(self.traits.softmax_scale * LOG2E) * fx.Float32(
                key_scale
            )
            self.v_scale_f = fx.Float32(value_scale)
        self.NEG_INF = fx.Float32(float("-inf"))
        self.ZERO_F = fx.Float32(0.0)
        # Finite or -inf scores permit nnan's bare max instructions.
        # Do not set ninf: -inf is the mask sentinel.
        self.fm_nnan = arith.FastMathFlags.nnan

    def init_causal_offsets(self):
        # Distance to the newest query keeps causal bounds tile-relative.
        if const_expr(self.traits.QUERIES_PER_CTA == 1):
            self.causal_offset = [
                self.traits.query_length - 1 - self.query_begin
                for _m in range_constexpr(self.traits.M_TILES)
            ]
        else:
            self.causal_offset = [
                self.traits.query_length
                - 1
                - self.query_begin
                - (m * MFMA_MNK + self.lane16) // self.traits.query_group_size
                for m in range_constexpr(self.traits.M_TILES)
            ]
            if const_expr(self.traits.CTA_ROWS % MFMA_MNK != 0):
                # Padded rows still need nonnegative causal offsets.
                self.causal_offset = [
                    (offset > 0).select(offset, 0) for offset in self.causal_offset
                ]
