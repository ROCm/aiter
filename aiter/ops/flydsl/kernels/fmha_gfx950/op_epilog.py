# SPDX-License-Identifier: MIT
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Epilogue: output packing and the dense / split-K partial stores."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import as_ir_value as as_mlir_value
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.fmha_gfx950.pipeline import (
    DualwaveFp8KernelContext,
    _store_lse,
)


class DualwaveFp8StoreHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _pack_out_pair(self, a, b):
        """Pack two f32 outputs into one i32 (a -> lo16, b -> hi16).

        bf16 and f16 are both 2 bytes with the same lo/hi ordering, so only the
        conversion differs; the lane-swap and store code downstream is shared.
        """
        if const_expr(self.traits.OUT_F16):
            # FlyDSL 0.5.0 toward_zero cast (unchanged from 0.4.0) emits scalar v_cvt_f16_f32 (not RTZ), changing f16 outputs by 1 ulp.
            v2 = rocdl.cvt_pkrtz(
                Vec.make_type(2, fx.Float16), fx.as_ir_value(a), fx.as_ir_value(b)
            )
            return fx.as_ir_value(Vec(v2, (2,), fx.Float16).bitcast(fx.Int32)[0])
        return fx.as_ir_value(
            Vec.from_elements([a, b], fx.BFloat16).bitcast(fx.Int32)[0]
        )

    def _o_pack_2dw(self, v_o, dc, store_group):
        r_base = store_group * 4
        lo = self._pack_out_pair(Vec(v_o[dc])[r_base], Vec(v_o[dc])[r_base + 1])
        hi = self._pack_out_pair(Vec(v_o[dc])[r_base + 2], Vec(v_o[dc])[r_base + 3])
        return lo, hi

    def _swap_half_partner(self, dw):
        pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
        # Raw rocdl op: stable fx.gpu.shuffle_xor returns one value per call, so replacing
        # the dual-output swap needs extra selects/moves (+~10% main-loop VALU on shuffled
        # BN64/paged, 5-6% slower prefill on gfx950). No stable dual-output swap at FlyDSL 0.5.0.
        swapped = rocdl.permlane32_swap(
            pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False
        )
        lo_res = llvm.extractvalue(T.i32, swapped, [0])
        hi_res = llvm.extractvalue(T.i32, swapped, [1])
        return (self.lane_div_32 != 0).select(lo_res, hi_res)

    def _packed_o_128_dwords(self, v_o, dc, g):
        is_hi_half = self.lane_div_32 != 0
        d0_a, d1_a = self._o_pack_2dw(v_o, dc, 2 * g)
        d0_b, d1_b = self._o_pack_2dw(v_o, dc, 2 * g + 1)
        y0_a, y1_a = self._swap_half_partner(d0_a), self._swap_half_partner(d1_a)
        y0_b, y1_b = self._swap_half_partner(d0_b), self._swap_half_partner(d1_b)
        w0 = is_hi_half.select(y0_b, as_mlir_value(d0_a))
        w1 = is_hi_half.select(y1_b, as_mlir_value(d1_a))
        w2 = is_hi_half.select(as_mlir_value(d0_b), y0_a)
        w3 = is_hi_half.select(as_mlir_value(d1_b), y1_a)
        return w0, w1, w2, w3

    def _packed_o_128_vec(self, v_o, dc, g):
        return Vec.from_elements(self._packed_o_128_dwords(v_o, dc, g), fx.Int32)

    def _row_has_keys(self, q_row):
        has_keys = self.seqlen_kv_v > 0
        if const_expr(self.traits.CAUSAL):
            has_keys = has_keys & (fx.Int32(q_row) + self.delta_i32 >= fx.Int32(0))
        return has_keys

    def _o_stage_layout(self):
        """Byte offset of (packed row, 16-byte chunk) in one O stage half: 256-byte rows, chunk XOR (row % 16).

        A ds_write_b128 phase (16 rows, one chunk) and a ds_read_b128 phase (one row, 16 chunks) both hit all 64
        banks. The swizzle keys on the packed row, so it is the same for every group.
        """
        traits = self.traits
        half_rows = traits.NUM_WAVES // 2 * traits.ROWS_PER_WAVE
        assert half_rows % 16 == 0, half_rows
        return fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(4, 4, 4)),
            fx.make_layout(
                (half_rows, traits.HEAD_DIM_V * 2 // 16),
                (traits.HEAD_DIM_V * 2, 16),
            ),
        )

    def _store_final_o_staged(self, v_o, q_row):
        """Stage the wave's 32 packed rows through LDS, then store 4 packed rows per buffer_store_dwordx4.

        Group % 4 == 0, so packed rows 4i..4i+3 are 4 heads of one token: 1 KB contiguous in O, against 32 rows x
        32 bytes per store in the direct epilogue.
        """
        traits = self.traits
        o_stage = self._o_stage_layout()
        # Waves [0, NUM_WAVES/2) stage their rows in the half overlaying kv, the rest in the half overlaying vt.
        # Rows are relative to the half; BLOCK_M/2 is a multiple of 16, so the row % 16 swizzle is unchanged.
        is_hi = self.wave_id_uni >= traits.NUM_WAVES // 2
        # ptrtoint/select/inttoptr: FlyDSL 0.5.0 has no pointer select. Epilogue only; the hot loop keeps its scopes.
        p_i8 = fx.PointerType.get(fx.Int8.ir_type, fx.AddressSpace.Shared, alignment=16)
        lo_addr = fx.Int32(fx.ptrtoint(self.lds.o_lo.ptr))
        hi_addr = fx.Int32(fx.ptrtoint(self.lds.o_hi.ptr))
        lds_i8 = fx.inttoptr(p_i8, is_hi.select(hi_addr, lo_addr))
        wave_row0 = (
            fx.Int32(self.wave_id_uni)
            - is_hi.select(fx.Int32(traits.NUM_WAVES // 2), fx.Int32(0))
        ) * fx.Int32(traits.ROWS_PER_WAVE)
        has_keys = self._row_has_keys(q_row)
        packs = [
            [self._packed_o_128_vec(v_o, dc, g) for g in range_constexpr(2)]
            for dc in range_constexpr(traits.D_CHUNKS)
        ]
        # Each stage half is the `o` arm of a union overlaying one ring buffer (kv or vt), which other waves may
        # still be reading for the last tile; the ring is dead only after this waitcnt + barrier. vmcnt(0) drains
        # this wave's LDS DMAs (tracked by vmcnt, not lgkmcnt): an empty segment issues tile 0's K/V DMA and runs
        # no loop, so nothing else guarantees it landed before the stage overwrites it. The loop path issues no
        # prefetch on its last tile, so the wait is already satisfied there.
        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
        # Raw rocdl op: keeps the scheduler from moving LDS accesses across the barrier; FlyDSL 0.5.0 exports no sched_barrier wrapper.
        rocdl.sched_barrier(0)
        fx.gpu.barrier()
        rocdl.sched_barrier(0)
        w_row = wave_row0 + fx.Int32(self.lane_mod_32)
        for dc in range_constexpr(traits.D_CHUNKS):
            for g in range_constexpr(2):
                o_pack = packs[dc][g]
                # Minimum pipeline padding can execute an all-masked row;
                # the public empty-window result is zero, not softmax NaN.
                o_pack = Vec.from_elements(
                    [
                        has_keys.select(o_pack[i], fx.Int32(0))
                        for i in range_constexpr(4)
                    ],
                    fx.Int32,
                )
                chunk = fx.Int32(dc * 4 + 2 * g) + fx.Int32(self.lane_div_32)
                off = fx.Int32(fx.get_scalar(fx.crd2idx((w_row, chunk), o_stage)))
                fx.ptr_store(o_pack, fx.add_offset(lds_i8, off))
        # Each wave reads back only its own rows.
        rocdl.s_waitcnt(lgkmcnt=0)
        # Raw rocdl op (see above): no sched_barrier wrapper in FlyDSL 0.5.0.
        rocdl.sched_barrier(0)
        # Read map: lane = 16-byte chunk c of the row + 16 * row-in-store hl0, so one dwordx4 store covers 4 rows.
        read_crd = fx.idx2crd(fx.Int32(self.lane), fx.make_layout((16, 4), (1, 16)))
        c = fx.Int32(read_crd[0].unpack())
        hl0 = fx.Int32(read_crd[1].unpack())
        vals = []
        for i in range_constexpr(traits.ROWS_PER_WAVE // 4):
            r_row = wave_row0 + fx.Int32(4 * i) + hl0
            off = fx.Int32(fx.get_scalar(fx.crd2idx((r_row, c), o_stage)))
            vals.append(
                fx.ptr_load(
                    fx.add_offset(lds_i8, off), result_type=Vec.make_type(4, fx.Int32)
                )
            )
        G = traits.GQA_GROUP_SIZE
        # Packed rows are (head-in-group, token), head fastest. G % 4 == 0, so rows 4i..4i+3 share the token of
        # row 4i and hl0 adds to its head with no carry; the row-4i coordinate is compile-time.
        packed = fx.make_layout((G, traits.ROWS_PER_WAVE // G), (1, G))
        tok0 = self.q_start + self.wave_id_uni * (traits.ROWS_PER_WAVE // G)
        for i in range_constexpr(traits.ROWS_PER_WAVE // 4):
            head_tok = fx.idx2crd(4 * i, packed)
            head = fx.Index(hl0) + fx.Index(head_tok[0].unpack())
            o_global = (
                (self.q_tok_base + tok0 + fx.Index(head_tok[1].unpack()))
                * self.stride_o_n_v
                + (self.q_head_idx + head) * traits.HEAD_DIM_V
                + fx.Index(c) * 8
            )
            self.buffer_store_128(Vec(vals[i]), o_global)

    def store_final_o(self, v_o, q_row, m_row=None, l_row=None):
        if const_expr(self.traits.O_STAGED):
            self._store_final_o_staged(v_o, q_row)
            return
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for g in range_constexpr(2):
                o_pack = self._packed_o_128_vec(v_o, dc, g)
                if const_expr(self.traits.PAGED):
                    has_keys = self.seqlen_kv_v > 0
                    if const_expr(self.traits.CAUSAL):
                        has_keys = has_keys & (
                            fx.Int32(q_row) + self.delta_i32 >= fx.Int32(0)
                        )
                    # Minimum pipeline padding can execute an all-masked row;
                    # the public empty-window result is zero, not softmax NaN.
                    o_pack = Vec.from_elements(
                        [
                            has_keys.select(o_pack[i], fx.Int32(0))
                            for i in range_constexpr(4)
                        ],
                        fx.Int32,
                    )
                d_col = (dc * self.traits.D_CHUNK) + (2 * g + self.lane_div_32) * 8
                o_global = self.global_idx_o(q_row, d_col)
                self.buffer_store_128(o_pack, o_global)
        if const_expr(self.traits.RETURN_LSE):
            _store_lse(
                self,
                q_row,
                fx.Float32(m_row) * self.c_logit_scale,
                l_row,
                q_row < self.seqlen_q_v,
                self.lane < 32,
            )

    def store_splitk_partial_o(self, v_o, m_row, l_row, q_row):
        m_row = fx.Float32(m_row) * self.c_logit_scale
        split_z = self.batch_idx * self.traits.NUM_KV_SPLITS + self.split_idx
        o_part_row_base = (
            (split_z * self.traits.NUM_HEADS_Q + self.q_head_idx) * self.seq_len_v
            + q_row
        ) * (self.traits.HEAD_DIM_V // 2)
        grid_z = fx.Index(gpu.grid_dim.z)
        mrow_base = (
            grid_z
            * self.traits.NUM_HEADS_Q
            * self.seq_len_v
            * (self.traits.HEAD_DIM_V // 2)
        )
        lrow_base = mrow_base + grid_z * self.traits.NUM_HEADS_Q * self.seq_len_v
        ml_row_idx = (
            split_z * self.traits.NUM_HEADS_Q + self.q_head_idx
        ) * self.seq_len_v + q_row

        @flyc.jit
        def _store_splitk_partial_if_qrow():
            if q_row < self.seq_len_v:
                for dc in range_constexpr(self.traits.D_CHUNKS):
                    for g in range_constexpr(2):
                        dw_col = (
                            dc * (self.traits.D_CHUNK // 2)
                            + (2 * g + self.lane_div_32) * 4
                        )
                        self.ws_store_quad_i32(
                            self._packed_o_128_dwords(v_o, dc, g),
                            o_part_row_base + dw_col,
                        )
                if self.lane < 32:
                    self.ws_store_f32(m_row, mrow_base + ml_row_idx)
                    self.ws_store_f32(l_row, lrow_base + ml_row_idx)

        _store_splitk_partial_if_qrow()

    def store_empty_split(self):
        @flyc.jit
        def _store_empty_split():
            if self.max_num_tiles < self.split_t0 + 4:
                q_row_e = self.q_start + self.wave_q_offset + self.lane_mod_32
                split_z_e = self.batch_idx * self.traits.NUM_KV_SPLITS + self.split_idx
                o_row_base_e = (
                    (split_z_e * self.traits.NUM_HEADS_Q + self.q_head_idx)
                    * self.seq_len_v
                    + q_row_e
                ) * (self.traits.HEAD_DIM_V // 2)
                grid_z_e = fx.Index(gpu.grid_dim.z)
                mrow_base_e = (
                    grid_z_e
                    * self.traits.NUM_HEADS_Q
                    * self.seq_len_v
                    * (self.traits.HEAD_DIM_V // 2)
                )
                lrow_base_e = (
                    mrow_base_e + grid_z_e * self.traits.NUM_HEADS_Q * self.seq_len_v
                )
                ml_row_e = (
                    split_z_e * self.traits.NUM_HEADS_Q + self.q_head_idx
                ) * self.seq_len_v + q_row_e
                if q_row_e < self.seq_len_v:
                    c_zero_i = fx.Int32(0)
                    for dc in range_constexpr(self.traits.D_CHUNKS):
                        for g in range_constexpr(2):
                            dw_col = (
                                dc * (self.traits.D_CHUNK // 2)
                                + (2 * g + self.lane_div_32) * 4
                            )
                            self.ws_store_quad_i32(
                                [c_zero_i, c_zero_i, c_zero_i, c_zero_i],
                                o_row_base_e + dw_col,
                            )
                    if self.lane < 32:
                        self.ws_store_f32(fx.Float32(-1e30), mrow_base_e + ml_row_e)
                        self.ws_store_f32(self.c_zero_f, lrow_base_e + ml_row_e)

        _store_empty_split()
