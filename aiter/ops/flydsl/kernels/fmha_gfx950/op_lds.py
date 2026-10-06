# SPDX-License-Identifier: MIT
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""LDS data movement: Q load, K/V global->LDS, LDS->VGPR."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.fmha_gfx950.pipeline import (
    DualwaveFp8KernelContext,
    _ds_read_tr8_b64_imm,
)


class DualwaveFp8QLoader(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def stage_q_to_lds(self):
        traits = self.traits
        chunks_per_row = traits.HEAD_DIM // 16  # 16-byte DMA chunks per Q row
        total_chunks = (traits.BLOCK_M * traits.HEAD_DIM) // 16
        for p in range_constexpr(total_chunks // traits.BLOCK_SIZE):
            c = self.tid + (p * traits.BLOCK_SIZE)
            row = c // chunks_per_row
            dchunk = c % chunks_per_row
            src_elem = self.q_gmem_elem_offset + row * self.stride_q_n_v + dchunk * 16
            if const_expr(traits.GQA_PACK_M):
                # The descriptor stays uniform; only the byte offset gathers heads.
                src_elem = (
                    self.q_gmem_elem_offset
                    + (row // traits.GQA_GROUP_SIZE) * self.stride_q_n_v
                    + (row % traits.GQA_GROUP_SIZE) * traits.HEAD_DIM
                    + dchunk * 16
                )
            lds_addr = self.lds_q_base_idx + c * 16
            self.buffer_load_lds_128(self.q_div, lds_addr, src_elem, 0)


class DualwaveFp8PageIdLoader(DualwaveFp8KernelContext):
    def load_block_table_to_lds(self):
        traits = self.traits

        @flyc.jit
        def _run():
            segment_tiles = self.split_t_end - self.split_t0
            staged_tiles = (segment_tiles > traits.NUM_PREFETCH_K).select(
                segment_tiles, fx.Index(traits.NUM_PREFETCH_K)
            )
            num_kv_tiles = (self.seqlen_kv_v + traits.BLOCK_N - 1) // traits.BLOCK_N
            for pas in range_constexpr(traits.PAGED_BT_LDS_SIZE // traits.BLOCK_SIZE):
                local_tile = self.tid + pas * traits.BLOCK_SIZE
                if local_tile < staged_tiles:
                    # Keep byte-addressed stores to preserve LDS address materialization
                    # across the nested guards.
                    dst = buffer_ops.get_element_ptr(
                        self.lds_bt_base_ptr,
                        byte_offset=_raw(fx.Int32(local_tile * 4)),
                        elem_type=T.i8,
                    )
                    # Empty segments and rounded pipeline tiles still resolve to
                    # an in-bounds page before descriptor bounds zero their data.
                    llvm.StoreOp(_raw(fx.Int32(0)), dst)
                    tile = fx.Index(self.split_t0) + local_tile
                    if (local_tile < segment_tiles) & (tile < num_kv_tiles):
                        row = self.batch_idx * self.block_table_stride_v + tile
                        reg = fx.make_rmem_tensor(1, fx.Int32)
                        fx.copy(
                            self.bt_atom,
                            fx.slice(self.bt_div, (None, fx.Int32(row))),
                            reg,
                        )
                        page = Vec(fx.memref_load_vec(reg))[0]
                        llvm.StoreOp(_raw(page), dst)

        _run()

    def begin_page_ids(self, tiles):
        handles = []
        for tile in tiles:
            last = fx.Index(self.split_t_end) - 1
            clamped = (tile < last).select(tile, last)
            first = fx.Index(self.split_t0)
            clamped = (clamped > first).select(clamped, first)
            bt = self.lds.bt.view(fx.make_layout(self.traits.PAGED_BT_LDS_SIZE, 1))
            handles.append(fx.memref_load(bt, fx.Int32(clamped - first)))
        return handles

    def end_page_ids(self, handles):
        rocdl.s_waitcnt(lgkmcnt=0)
        return [fx.Int64(fx.Int32(rocdl.readfirstlane(T.i32, h))) for h in handles]


class DualwaveFp8KvGmemToLdsLoader(DualwaveFp8PageIdLoader):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_k(self, tile_start, buf_id, page_id=None):
        """DMA one K tile into LDS, one pass per head-dim band.

        A band's LDS line is this wave's n-rows of `chunk` bytes, row-contiguous,
        so the QK read indexes it as (band, row, 64-byte slice). The 64-byte tail
        band at head_dim 192 runs on the low 32 lanes and moves no padding.
        """
        traits = self.traits
        eb = traits.ELEM_BYTES
        k_div = self.k_div
        if const_expr(traits.PAGED):
            k_div = self.kv_page_div("k", page_id, tile_start)
        k_lds_byte_base = self.lds_kv_base_idx + self.k_buf_base(buf_id) * eb
        if const_expr(traits.BODY_VARIANT == "conventional_bn64"):
            self._load_k_bn64(k_div, tile_start, k_lds_byte_base)
        else:
            self._load_k_bands(k_div, tile_start, k_lds_byte_base)

    def _load_k_bn64(self, k_div, tile_start, k_lds_byte_base):
        traits = self.traits
        for pas in range_constexpr(2):
            stripe = self.wave_id_uni + pas * traits.NUM_WAVES
            n_in_tile = (
                (self.lane_in_warp // 8) * 8 + self.wave_id + pas * traits.NUM_WAVES
            )
            global_d = (self.lane_in_warp % 8) * traits.VEC_KV
            src_elem = (
                self.kv_gmem_elem_offset + n_in_tile * self.stride_kv_n_v + global_d
            )
            if const_expr(traits.K_SHUFFLED):
                src_elem = (
                    self.kv_gmem_elem_offset
                    + global_d // 16 * traits.BLOCK_N * 16
                    + n_in_tile * 16
                )
                # A prefix descriptor bound cannot mask the shuffled logical tail.
                valid = n_in_tile + tile_start < self.seqlen_kv_v
                src_elem = valid.select(src_elem, fx.Index(0x7FFFFFF0))
            lds_addr = k_lds_byte_base + stripe * traits.K_BAND_LINE_STRIDE[0]
            self.buffer_load_lds_128(k_div, lds_addr, src_elem, 0)

    def _load_k_bands(self, k_div, tile_start, k_lds_byte_base):
        traits = self.traits
        eb = traits.ELEM_BYTES
        rows_per_wave = -(-traits.BLOCK_N // traits.NUM_WAVES)
        for d in range_constexpr(self.NUM_DMA_K):
            lanes_per_row = traits.K_BAND_CHUNK[d] // traits.VEC_KV
            slots = rows_per_wave * lanes_per_row
            band_base = (
                k_lds_byte_base
                + traits.K_BAND_BASE[d] * eb
                + self.wave_id_uni * (traits.K_BAND_LINE_STRIDE[d] * eb)
            )
            for pas in range_constexpr(-(-slots // traits.WARP_SIZE)):
                slot = self.lane_in_warp + (pas * traits.WARP_SIZE)
                n_in_tile = (slot // lanes_per_row) * traits.NUM_WAVES + self.wave_id
                global_d = (
                    slot % lanes_per_row
                ) * traits.VEC_KV + traits.K_BAND_GLOBAL_D[d]
                src_elem = (
                    self.kv_gmem_elem_offset + n_in_tile * self.stride_kv_n_v + global_d
                )
                if const_expr(traits.K_SHUFFLED):
                    layout = fx.make_layout(
                        (traits.HEAD_DIM // 16, traits.BLOCK_N, 16),
                        (traits.BLOCK_N * 16, 16, 1),
                    )
                    src_elem = self.kv_gmem_elem_offset + fx.Index(
                        fx.get_scalar(
                            fx.crd2idx(
                                (fx.Int32(global_d // 16), fx.Int32(n_in_tile), 0),
                                layout,
                            )
                        )
                    )
                    # Shuffled tokens are strided across the page, so a prefix
                    # descriptor bound cannot mask the logical K tail.
                    valid = n_in_tile + tile_start < self.seqlen_kv_v
                    src_elem = valid.select(src_elem, fx.Index(0x7FFFFFF0))
                lds_addr = band_base + fx.Index(
                    pas * traits.WARP_SIZE * traits.VEC_KV * eb
                )
                active = min(slots - pas * traits.WARP_SIZE, traits.WARP_SIZE)
                if const_expr(active == traits.WARP_SIZE):
                    if const_expr(traits.PAGED):
                        self.buffer_load_lds_128(k_div, lds_addr, src_elem, 0)
                    else:
                        self.buffer_load_lds_128(
                            self.k_div,
                            lds_addr,
                            src_elem,
                            tile_start * self.stride_kv_n_v,
                        )
                else:
                    self._load_k_band_partial_wave(
                        lds_addr, src_elem, tile_start, active
                    )

    def _load_k_band_partial_wave(self, lds_addr, src_elem, tile_start, active_lanes):
        soffset = tile_start * self.stride_kv_n_v
        k_div = self.k_div

        @flyc.jit
        def _run():
            if self.lane_in_warp < active_lanes:
                self.buffer_load_lds_128(k_div, lds_addr, src_elem, soffset)

        _run()

    def load_v(self, tile_start, buf_id, page_id=None):
        if const_expr(self.traits.V_SHUFFLED):
            self._stage_v_fp8_coalesced(tile_start, buf_id, page_id)
        else:
            self._stage_v_fp8_block_dma(tile_start, buf_id, page_id)

    def _stage_v_fp8_coalesced(self, tile_start, buf_id, page_id):
        traits = self.traits
        v_div = self.kv_page_div("v", page_id, tile_start)
        aligned_base = ((self.lds_vt_base_idx + 127) // 128) * 128
        buf_off = buf_id * traits.BLOCK_N * traits.HEAD_DIM_V
        num_dma = traits.BLOCK_N * traits.HEAD_DIM_V // 1024
        for pas in range_constexpr(num_dma // traits.NUM_WAVES):
            src_elem = (
                self.v_gmem_elem_offset + (self.tid + pas * traits.BLOCK_SIZE) * 16
            )
            lds_addr = (
                aligned_base
                + buf_off
                + (self.wave_id_uni + pas * traits.NUM_WAVES) * 1024
            )
            self.buffer_load_lds_128(v_div, lds_addr, src_elem, 0)

    def _stage_v_fp8_block_dma(self, tile_start, buf_id, page_id=None):
        traits = self.traits
        v_div = self.v_div
        if const_expr(traits.PAGED):
            v_div = self.kv_page_div("v", page_id, tile_start)
        nbands = traits.HEAD_DIM_V // 16
        v_tile_bytes = (traits.BLOCK_N // 8) * nbands * 128
        buf_off = buf_id * v_tile_bytes
        aligned_base = ((self.lds_vt_base_idx + 127) // 128) * 128
        # The tile is BLOCK_N * nbands 16-byte slots, and one DMA instruction moves a
        # whole wave of them. Hand out instructions, not row-groups: a wave's LDS
        # destination is then always a full WARP_SIZE*16 span, so nothing has to be
        # masked off inside a wave. buffer_load...lds strides the LDS write by lane
        # regardless of exec, so an intra-wave mask would still write past the span.
        per_dma = traits.WARP_SIZE * traits.VEC_KV * traits.ELEM_BYTES
        slots_per_group = 8 * nbands
        num_dma = (traits.BLOCK_N * nbands * 16) // per_dma
        passes = -(-num_dma // traits.NUM_WAVES)
        for pas in range_constexpr(passes):
            dma_id = self.wave_id_uni + (pas * traits.NUM_WAVES)
            slot = dma_id * traits.WARP_SIZE + self.lane
            lds_addr = aligned_base + fx.Index(buf_off) + dma_id * per_dma
            grp = slot // slots_per_group
            rem = slot % slots_per_group
            dest_n = fx.Int32(grp * 8 + rem % 8)
            w16 = dest_n % fx.Int32(16)
            c_add = (w16 >= fx.Int32(4)) & (w16 < fx.Int32(8))
            c_sub = (w16 >= fx.Int32(8)) & (w16 < fx.Int32(12))
            n = (
                dest_n
                + c_add.select(fx.Int32(4), fx.Int32(0))
                - c_sub.select(fx.Int32(4), fx.Int32(0))
            )
            d_block = rem // 8
            src_elem = (
                self.v_gmem_elem_offset + fx.Index(n) * self.stride_v_n_v + d_block * 16
            )
            if const_expr(num_dma % traits.NUM_WAVES == 0 or pas < passes - 1):
                if const_expr(traits.PAGED):
                    self.buffer_load_lds_128(v_div, lds_addr, src_elem, 0)
                else:
                    self.buffer_load_lds_128(
                        self.v_div, lds_addr, src_elem, tile_start * self.stride_v_n_v
                    )
            else:
                self._load_v_group_if_in_tile(
                    lds_addr, src_elem, tile_start, dma_id, num_dma
                )

    def _load_v_group_if_in_tile(self, lds_addr, src_elem, tile_start, grp, groups):
        soffset = tile_start * self.stride_v_n_v
        v_div = self.v_div

        @flyc.jit
        def _run():
            if grp < groups:
                self.buffer_load_lds_128(v_div, lds_addr, src_elem, soffset)

        _run()


class DualwaveFp8KvLdsToVgprLoader(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_k(self, buf_id):
        # Read K in the wide 32x32x64 QK operand layout (32 contiguous head-dim/lane,
        # two N-strips, two head-dim halves).
        traits = self.traits
        k_base = self.k_buf_base(buf_id)
        d_base = self.lane_div_32 * 32
        n_lo = self.lane_mod_32
        n_hi = self.lane_mod_32 + 32

        rows_per_line = (
            8 if traits.BODY_VARIANT == "conventional_bn64" else traits.NUM_WAVES
        )

        def _read_strip(key):
            out = []
            for ws in range_constexpr(traits.HEAD_DIM // 64):
                b = traits.K_WS_BAND[ws]
                line = (key % rows_per_line) * traits.K_BAND_LINE_STRIDE[b]
                row = line + (key // rows_per_line) * traits.K_BAND_CHUNK[b]
                addr = (
                    k_base + traits.K_BAND_BASE[b] + row + traits.K_WS_OFF[ws] + d_base
                )
                out.append(self.read_i32x8_lds(self.lds_kv_base_ptr, addr))
            return out

        return (_read_strip(n_lo), _read_strip(n_hi))

    def load_v(self, buf_id, tile_start=None):
        if const_expr(
            self.traits.BODY_VARIANT == "conventional_bn64" and self.traits.V_SHUFFLED
        ):
            packs = self._load_v_fp8_bn64(buf_id, tile_start)
        elif const_expr(self.traits.V_SHUFFLED):
            packs = self._load_v_fp8_coalesced(buf_id, tile_start)
        else:
            packs = self._load_v_fp8_block(buf_id)
        return packs

    def _load_v_fp8_bn64(self, buf_id, tile_start):
        traits = self.traits
        full_tile = fx.Int64(tile_start) + traits.BLOCK_N <= fx.Int64(self.seqlen_kv_v)

        @flyc.jit
        def _run():
            packs = [[fx.Int64(0).ir_value()] * traits.D_CHUNKS for _ in range(4)]
            if full_tile:
                packs = self._load_v_fp8_coalesced(buf_id, tile_start, mask_tail=False)
            else:
                packs = self._load_v_fp8_coalesced(buf_id, tile_start)
            return packs

        return _run()

    def _load_v_fp8_coalesced(self, buf_id, tile_start, mask_tail=True):
        traits = self.traits
        aligned_base = ((self.lds_vt_base_idx + 127) // 128) * 128
        tile_base = aligned_base + buf_id * traits.BLOCK_N * traits.HEAD_DIM_V
        d_lane = fx.Int32(self.lane % 32)
        lane_half = fx.Int32(self.lane // 32)
        masks = []
        if const_expr(mask_tail):
            for half in range_constexpr(2):
                for word in range_constexpr(4):
                    token = fx.Int32(tile_start) + lane_half * 16 + half * 32 + word * 4
                    count = self.seqlen_kv_i32 - token
                    count = (count > 0).select(count, fx.Int32(0))
                    count = (count < 4).select(count, fx.Int32(4))
                    # Shift a 32-bit all-ones word; avoid shifting by the word width.
                    shift = (count > 0).select((4 - count) * 8, fx.Int32(24))
                    mask = fx.Int32(-1).shrui(shift)
                    masks.append((count > 0).select(mask, fx.Int32(0)))
        packs = [[None] * traits.D_CHUNKS for _ in range(4)]
        for dc in range_constexpr(traits.D_CHUNKS):
            d = fx.Int32(32 * dc) + d_lane
            for half in range_constexpr(2):
                byte_off = fx.Int32(
                    tile_base + (lane_half + 2 * half) * traits.HEAD_DIM_V * 16 + d * 16
                )
                # The typed LDS view splits this aligned b128 load into read2_b32
                # pairs and adds address arithmetic; keep the explicit alignment.
                ptr = buffer_ops.get_element_ptr(
                    self.lds_vt_base_ptr,
                    byte_offset=_raw(byte_off - fx.Int32(self.lds_vt_base_idx)),
                    elem_type=T.i8,
                )
                words = Vec(
                    llvm.LoadOp(Vec.make_type(4, fx.Int32), ptr, alignment=16).result
                )
                if const_expr(mask_tail):
                    # Integer masks remove invalid bytes without sanitizing valid NaNs.
                    words = words & Vec.from_elements(
                        masks[4 * half : 4 * half + 4], fx.Int32
                    )
                for pair in range_constexpr(2):
                    packs[2 * half + pair][dc] = (
                        Vec.from_elements(
                            [fx.Int32(words[2 * pair]), fx.Int32(words[2 * pair + 1])],
                            fx.Int32,
                        )
                        .bitcast(fx.Int64)[0]
                        .ir_value()
                    )
        return packs

    def _load_v_fp8_block(self, buf_id):
        traits = self.traits
        v_tile_bytes = (traits.BLOCK_N // 8) * (traits.HEAD_DIM_V // 16) * 128
        buf_off = buf_id * v_tile_bytes
        nbands = traits.HEAD_DIM_V // 16
        rh = (self.lane % 32) // 16
        l16 = self.lane % 16
        lane_hi = self.lane // 32
        aligned_base = ((self.lds_vt_base_idx + 127) // 128) * 128
        base = fx.Int32(
            aligned_base + buf_off + rh * 128 + l16 * 8 + lane_hi * (nbands * 128)
        )

        def _tr8(imm):
            r = _ds_read_tr8_b64_imm(self.v2i32_type, base, imm)
            return Vec(r).bitcast(fx.Int64)[0].ir_value()

        packs = [[None] * traits.D_CHUNKS for _ in range(4)]
        for dc in range_constexpr(traits.D_CHUNKS):
            for ks in range_constexpr(4):
                imm0 = (2 * ks * nbands + dc * 2) * 128
                packs[ks][dc] = _tr8(imm0)
        return packs
