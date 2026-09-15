# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.fmha_gfx950.common import load as _load
from aiter.ops.flydsl.kernels.fmha_gfx950.common import store as _store
from aiter.ops.flydsl.kernels.fmha_gfx950.paged_pipeline import (
    PAGED_FP8_BUFFER_LIMIT_BYTES,
    DualwaveFp8KernelContext,
    _ds_read_tr8_b64_imm,
    _sigma_k_tile_n,
    _vec_k_dma_oct_idx,
)


def _page1_k_page_ids(page_ids):
    """Convert token-major page IDs to K's sigma lanes without another lookup."""
    # Swap lane bits 2/3: move lanes 4..7 forward and 8..11 backward by four.
    value = fx.Int32(page_ids).ir_value()
    swapped = rocdl.update_dpp(T.i32, value, value, 0x104, 15, 2, False)
    result = rocdl.update_dpp(T.i32, swapped, value, 0x114, 15, 4, False)
    return fx.Int64(fx.Int32(result))


def _transpose_v_fp8_16x16(source, lane, stages=4):
    """Transpose byte quads (two stages) or 16-lane tiles (four), with full EXEC."""
    assert stages in (2, 4)
    words = [fx.Uint32(Vec(source)[i]) for i in range_constexpr(4)]
    lane = fx.Int32(lane)
    for bit in range_constexpr(stages):
        lane_bit = (lane & (1 << bit)) != 0
        if const_expr(bit < 2):
            # Select alternating bytes, then halfwords, from self/peer words.
            low_selector = 0x06020400 if bit == 0 else 0x05040100
            high_selector = 0x03070105 if bit == 0 else 0x03020706
            selector = lane_bit.select(fx.Int32(high_selector), fx.Int32(low_selector))
        updated = []
        for word in range_constexpr(4):
            peer_word = word if bit < 2 else word ^ (1 << (bit - 2))
            peer_value = words[peer_word].ir_value()
            if const_expr(bit < 2):
                dpp_ctrl = 0xB1 if bit == 0 else 0x4E
                peer = rocdl.update_dpp(
                    T.i32, peer_value, peer_value, dpp_ctrl, 15, 15, False
                )
                updated.append(fx.Uint32(rocdl.perm_b32(peer, words[word], selector)))
            else:
                peer = fx.Uint32(
                    rocdl.ds_swizzle(
                        T.i32, peer_value, fx.Int32(31 | (1 << (bit + 10))).ir_value()
                    )
                )
                if const_expr(word & (1 << (bit - 2))):
                    updated.append(lane_bit.select(words[word], peer))
                else:
                    updated.append(lane_bit.select(peer, words[word]))
        words = updated
    return Vec.from_elements([fx.Int32(word) for word in words], fx.Int32)


class DualwaveFp8KvGmemToLdsLoader(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_k(self, tile_start, buf_id, page_id=None):
        """DMA K using the vectorized page layout or dense head-dimension bands.

        Dense bands are row-contiguous within each wave. Their D192 tail
        uses the low 32 lanes; paged tiles retain the physical D-group mapping.
        """
        traits = self.traits
        if const_expr(traits.PAGED and traits.K_LDS_PAGE_GROUPED):
            self._load_k_page16(tile_start, buf_id, page_id=page_id)
            return
        eb = traits.ELEM_BYTES
        k_lds_byte_base = self.lds_kv_base_idx + self.k_buf_base(buf_id) * eb
        if const_expr(traits.KV_VECTORIZED):
            src_div = None
            if const_expr(traits.PAGE_SIZE == 1 and page_id is not None):
                page_id = _page1_k_page_ids(page_id)
            if const_expr(traits.PAGE_SIZE >= traits.BLOCK_N):
                if page_id is None:
                    page_id = self.load_page_id(tile_start)
                src_div = self.make_page_view(self.k_base_iter, page_id)

            def _load_vectorized():
                for d in range_constexpr(self.NUM_DMA_K):
                    if const_expr(traits.PAGE_SIZE < traits.BLOCK_N):
                        oct_idx = _vec_k_dma_oct_idx(
                            traits, d, self.wave_id_uni, self.lane_in_warp
                        )
                        token = fx.Int64(tile_start) + _sigma_k_tile_n(
                            oct_idx % traits.BLOCK_N
                        )
                        dim_group = oct_idx // traits.BLOCK_N
                        physical_page = (
                            self.load_page_id(token, uniform=False)
                            if page_id is None
                            else page_id
                        )
                        src_elem = (
                            physical_page * self.k_page_bytes
                            + self.kv_head_idx * traits.HEAD_DIM * traits.PAGE_SIZE
                            + (dim_group * traits.PAGE_SIZE + token % traits.PAGE_SIZE)
                            * traits.KV_VEC_SIZE
                        )
                        valid = (token < self.seqlen_kv_v) & (
                            dim_group * traits.KV_VEC_SIZE < traits.HEAD_DIM
                        )
                        if const_expr(traits.CACHE_BUFFERED):
                            offset = valid.select(
                                fx.Int32(src_elem),
                                fx.Int32(PAGED_FP8_BUFFER_LIMIT_BYTES),
                            )
                            dst_byte = (
                                k_lds_byte_base
                                + (self.wave_id_uni * traits.SMEM_D_RPT + d)
                                * traits.WARP_SIZE
                                * traits.KV_VEC_SIZE
                            )
                            dst_byte = rocdl.readfirstlane(
                                T.i32, fx.Int32(dst_byte).ir_value()
                            )
                            self.buffer_load_lds_128(self.k_div, dst_byte, offset, 0)
                        else:
                            source = self.global_load_fp8x16(
                                self.k_base_iter, src_elem, valid
                            )
                            dst_byte = (
                                self.k_buf_base(buf_id) + oct_idx * traits.KV_VEC_SIZE
                            )
                            dst = fx.slice(
                                self.k_lds_i32_tiles, (None, fx.Uint32(dst_byte) // 16)
                            )
                            _store(fx.get_iter(dst), fx.Vector(source))
                    else:
                        oct_idx = _vec_k_dma_oct_idx(
                            traits, d, self.wave_id_uni, self.lane_in_warp
                        )
                        token = _sigma_k_tile_n(oct_idx % traits.BLOCK_N)
                        if const_expr(traits.PAGE_SIZE != traits.BLOCK_N):
                            token = fx.Int64(tile_start) % traits.PAGE_SIZE + token
                        dim_group = oct_idx // traits.BLOCK_N
                        src_elem = (
                            self.kv_head_idx * traits.HEAD_DIM * traits.PAGE_SIZE
                            + (dim_group * traits.PAGE_SIZE + token)
                            * traits.KV_VEC_SIZE
                        )
                    if const_expr(traits.PAGE_SIZE >= traits.BLOCK_N):
                        lds_addr = (
                            k_lds_byte_base
                            + (self.wave_id_uni * traits.SMEM_D_RPT + d)
                            * traits.WARP_SIZE
                            * traits.KV_VEC_SIZE
                        )
                        lds_addr = rocdl.readfirstlane(
                            T.i32, fx.Int32(lds_addr).ir_value()
                        )
                        self.buffer_load_lds_128(src_div, lds_addr, src_elem, 0)

            if const_expr(traits.HEAD_DIM_V == 192):

                @flyc.jit
                def _load_compact():
                    copy_waves = (traits.BLOCK_N * traits.HEAD_DIM) // (
                        self.NUM_DMA_K * traits.WARP_SIZE * traits.KV_VEC_SIZE
                    )
                    if self.wave_id_uni < fx.Int64(copy_waves):
                        _load_vectorized()

                _load_compact()
            else:
                _load_vectorized()
            return

        rows_per_wave = -(-traits.BLOCK_N // traits.NUM_WAVES)
        for d in range_constexpr(self.NUM_DMA_K):
            lanes_per_row = traits.K_BAND_CHUNK[d] // traits.VEC_KV
            slots = rows_per_wave * lanes_per_row
            band_base = (
                k_lds_byte_base
                + fx.Index(traits.K_BAND_BASE[d] * eb)
                + self.wave_id_uni * (traits.K_BAND_LINE_STRIDE[d] * eb)
            )
            for pas in range_constexpr(-(-slots // traits.WARP_SIZE)):
                slot = self.lane_in_warp + fx.Index(pas * traits.WARP_SIZE)
                n_in_tile = (slot // lanes_per_row) * traits.NUM_WAVES + self.wave_id
                global_d = (
                    slot % lanes_per_row
                ) * traits.VEC_KV + traits.K_BAND_GLOBAL_D[d]
                src_elem = (
                    self.kv_gmem_elem_offset + n_in_tile * self.stride_kv_n_v + global_d
                )
                lds_addr = band_base + fx.Index(
                    pas * traits.WARP_SIZE * traits.VEC_KV * eb
                )
                active = min(slots - pas * traits.WARP_SIZE, traits.WARP_SIZE)
                if const_expr(active == traits.WARP_SIZE):
                    self.buffer_load_lds_128(
                        self.k_div, lds_addr, src_elem, tile_start * self.stride_kv_n_v
                    )
                else:
                    self._load_k_band_partial_wave(
                        lds_addr, src_elem, tile_start, active
                    )

    def _load_k_page16(self, tile_start, buf_id, page_id=None):
        """Each wave copies four D-groups from one physical page directly to LDS."""
        traits = self.traits
        page_in_tile = self.wave_id_uni % fx.Int64(4)
        if page_id is None:
            page_id = self.load_page_id(
                fx.Int64(tile_start) + page_in_tile * fx.Int64(16)
            )
        if const_expr(traits.CACHE_BUFFERED):
            # The byte bound makes the uniform displacement fit i32; the
            # shared resource retains the full 64-bit cache base pointer.
            src_div = self.k_div
            page_offset = fx.Int32(page_id * self.k_page_bytes)
        else:
            src_div = self.make_page_view(self.k_base_iter, page_id)
            page_offset = fx.Int32(0)
        num_dma = traits.HEAD_DIM // traits.KV_VEC_SIZE
        for part in range_constexpr(
            (num_dma + traits.NUM_WAVES - 1) // traits.NUM_WAVES
        ):
            dma_id = self.wave_id_uni + part * traits.NUM_WAVES

            def _copy_page_groups(dma_id):
                d_group = (dma_id // fx.Int64(4)) * fx.Int64(
                    4
                ) + self.lane_in_warp // fx.Int64(16)
                token = _sigma_k_tile_n(self.lane_in_warp % fx.Int64(16))
                src_elem = (
                    self.kv_head_idx * traits.HEAD_DIM * traits.PAGE_SIZE
                    + (d_group * traits.PAGE_SIZE + token) * traits.KV_VEC_SIZE
                )
                dst_byte = (
                    self.lds_kv_base_idx
                    + self.k_buf_base(buf_id)
                    + dma_id * fx.Int64(1024)
                )
                dst_byte = rocdl.readfirstlane(T.i32, fx.Int32(dst_byte).ir_value())
                self.buffer_load_lds_128(src_div, dst_byte, src_elem, page_offset)

            if const_expr(
                num_dma % traits.NUM_WAVES == 0 or part < num_dma // traits.NUM_WAVES
            ):
                _copy_page_groups(dma_id)
            else:

                @flyc.jit
                def _copy_tail_waves(dma_id):
                    if dma_id < fx.Int64(num_dma):
                        _copy_page_groups(dma_id)

                _copy_tail_waves(dma_id)

    def _load_k_band_partial_wave(self, lds_addr, src_elem, tile_start, active_lanes):
        soffset = tile_start * self.stride_kv_n_v
        k_div = self.k_div

        @flyc.jit
        def _run():
            if self.lane_in_warp < fx.Index(active_lanes):
                self.buffer_load_lds_128(k_div, lds_addr, src_elem, soffset)

        _run()

    def load_v(self, tile_start, buf_id, page_id=None, *, mask_padding=True):
        if const_expr(self.traits.PAGED):
            self._stage_v_fp8_vectorized_bankpad(
                tile_start, buf_id, page_id=page_id, mask_padding=mask_padding
            )
        else:
            self._stage_v_fp8_block_dma(tile_start, buf_id)

    def _stage_v_fp8_block_dma(self, tile_start, buf_id):
        traits = self.traits
        nbands = traits.HEAD_DIM_V // 16
        v_tile_bytes = (traits.BLOCK_N // 8) * nbands * 128
        buf_off = buf_id * v_tile_bytes
        aligned_base = (
            (self.lds_vt_base_idx + fx.Index(127)) // fx.Index(128)
        ) * fx.Index(128)
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
            dma_id = self.wave_id_uni + fx.Index(pas * traits.NUM_WAVES)
            slot = dma_id * fx.Index(traits.WARP_SIZE) + self.lane
            lds_addr = aligned_base + fx.Index(buf_off) + dma_id * fx.Index(per_dma)
            grp = slot // fx.Index(slots_per_group)
            rem = slot % fx.Index(slots_per_group)
            dest_n = fx.Int32(grp * fx.Index(8) + rem % fx.Index(8))
            w16 = dest_n % fx.Int32(16)
            c_add = (w16 >= fx.Int32(4)) & (w16 < fx.Int32(8))
            c_sub = (w16 >= fx.Int32(8)) & (w16 < fx.Int32(12))
            n = (
                dest_n
                + c_add.select(fx.Int32(4), fx.Int32(0))
                - c_sub.select(fx.Int32(4), fx.Int32(0))
            )
            d_block = rem // fx.Index(8)
            src_elem = (
                self.v_gmem_elem_offset
                + fx.Index(n) * self.stride_v_n_v
                + d_block * fx.Index(16)
            )
            if const_expr(num_dma % traits.NUM_WAVES == 0 or pas < passes - 1):
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
            if grp < fx.Index(groups):
                self.buffer_load_lds_128(v_div, lds_addr, src_elem, soffset)

        _run()

    def _permute_v_fp8_vectorized(self, src_i32x4):
        src_words = Vec(src_i32x4, (4,), fx.Int32)
        # permlane16_swap returns an LLVM pair; unpack it at this intrinsic boundary.
        pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
        pair_lo = rocdl.permlane16_swap(
            pair_ty,
            src_words[0].ir_value(),
            src_words[1].ir_value(),
            False,
            False,
        )
        pair_hi = rocdl.permlane16_swap(
            pair_ty,
            src_words[2].ir_value(),
            src_words[3].ir_value(),
            False,
            False,
        )
        return Vec.from_elements(
            [
                fx.Int32(llvm.extractvalue(T.i32, pair_lo, [0])),
                fx.Int32(llvm.extractvalue(T.i32, pair_lo, [1])),
                fx.Int32(llvm.extractvalue(T.i32, pair_hi, [0])),
                fx.Int32(llvm.extractvalue(T.i32, pair_hi, [1])),
            ],
            fx.Int32,
        )

    def _stage_v_fp8_vectorized_bankpad(
        self, tile_start, buf_id, page_id=None, *, mask_padding=True
    ):
        """Stage vectorized paged V in the bank-padded direct-FP8 layout."""
        if const_expr(self.traits.K_LDS_PAGE_GROUPED and self.traits.CACHE_BUFFERED):
            self._stage_v_fp8_page16_segments(
                tile_start, buf_id, page_id=page_id, mask_padding=mask_padding
            )
            return
        source = self.load_v_source(tile_start, page_id=page_id)
        self.store_v_source(source, buf_id, tile_start, mask_padding=mask_padding)

    def _mask_v_fp8_group(self, source, token_start):
        """Zero inactive tokens before PV: a zero probability cannot mask a NaN V."""
        packed = Vec(source)
        full_words = fx.Int32(self.seqlen_kv_v) >> 2
        tail_bits = (fx.Uint32(self.seqlen_kv_v) & 3) * 8
        tail_mask = (fx.Uint32(1) << tail_bits) - 1
        group_word = fx.Int32(fx.Int64(token_start) >> 2)
        words = []
        for word in range_constexpr(packed.numel):
            index = group_word + word % 4
            last_word = (index == full_words).select(tail_mask, fx.Uint32(0))
            mask = (index < full_words).select(fx.Uint32(0xFFFFFFFF), last_word)
            words.append(fx.Int32(fx.Uint32(packed[word]) & mask))
        return Vec.from_elements(words, fx.Int32).ir_value()

    def _page16_v_source(self, tile_start, page_id):
        """Coordinates and bounded resource for the wave-grouped V producer."""
        traits = self.traits
        token = fx.Int64(tile_start) + (self.wave_id_uni % fx.Int64(4)) * fx.Int64(16)
        d_col = (self.wave_id // fx.Int64(4)) * fx.Int64(64) + self.lane_in_warp
        if page_id is None:
            page_id = self.load_page_id(token)
        src_div = self.make_page_view(self.v_base_iter, page_id, is_value=True)
        src_elem = (
            self.kv_head_idx * traits.PAGE_SIZE * traits.HEAD_DIM_V
            + d_col * traits.KV_VEC_SIZE
        )
        return token, d_col, src_div, src_elem

    def _stage_v_fp8_page16_segments(
        self, tile_start, buf_id, page_id=None, *, mask_padding=True
    ):
        """Consume each V192 segment before loading the next segment's words."""
        traits = self.traits
        token, _, src_div, src_elem = self._page16_v_source(tile_start, page_id)
        src_elem = (token < self.seqlen_kv_v).select(
            src_elem, fx.Int64(self.v_page_bytes)
        )
        prefix = self.buffer_load_fp8x16(src_div, src_elem)
        if const_expr(mask_padding):
            prefix = self._mask_v_fp8_group(prefix, token)
        self._store_v_fp8_page16(prefix, buf_id, include_tail=False)

        @flyc.jit
        def _stage_tail():
            if self.wave_id_uni < fx.Int64(4):
                tail = self.buffer_load_fp8x16(
                    src_div, src_elem + fx.Int64(traits.FP8_V_H1 * traits.KV_VEC_SIZE)
                )
                if const_expr(mask_padding):
                    tail = self._mask_v_fp8_group(tail, token)
                self._store_v_fp8_page16(
                    tail, buf_id, segment_offset=traits.FP8_V_H1, include_tail=False
                )

        _stage_tail()

    def _store_v_fp8_lds(self, data, byte_offset):
        dst = fx.slice(self.v_lds_i32_tiles, (None, fx.Uint32(byte_offset) // 16))
        _store(fx.get_iter(dst), data)

    def load_v_source(self, tile_start, page_id=None):
        """Prefetch a native V packet; store_v_source consumes its selected layout."""
        traits = self.traits
        if const_expr(traits.PAGE_SIZE < traits.BLOCK_N):
            return self._load_v_fp8_small_pages_source(tile_start, page_id=page_id)
        if page_id is None:
            page_id = self.load_page_id(tile_start)
        src_div = self.make_page_view(self.v_base_iter, page_id, is_value=True)
        token_groups16 = traits.BLOCK_N // traits.KV_VEC_SIZE
        # permlane16 requires a full EXEC mask. The first segment has exactly
        # one vector per CTA thread, so no per-lane bounds branch is needed.
        prefix_dim = traits.FP8_V_H1 if traits.FP8_PV_SEGMENTED else traits.HEAD_DIM_V
        assert prefix_dim * token_groups16 == traits.BLOCK_SIZE
        n_group16 = self.lane_in_warp // fx.Int64(16)
        d_col = self.wave_id * fx.Int64(16) + self.lane_in_warp % fx.Int64(16)
        src_elem = (
            self.kv_head_idx * traits.PAGE_SIZE * traits.HEAD_DIM_V
            + n_group16 * traits.HEAD_DIM_V * traits.KV_VEC_SIZE
            + d_col * traits.KV_VEC_SIZE
        )
        if const_expr(traits.PAGE_SIZE != traits.BLOCK_N):
            src_elem += (fx.Int64(tile_start) % traits.PAGE_SIZE) * traits.HEAD_DIM_V
        source = self.buffer_load_fp8x16(src_div, src_elem)
        if const_expr(traits.FP8_PV_SEGMENTED):
            tail = self.buffer_load_fp8x16(
                src_div, src_elem + fx.Int64(traits.FP8_V_H1 * traits.KV_VEC_SIZE)
            )
            source = Vec(source).shuffle(Vec(tail), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
        return source

    def _load_v_fp8_small_pages_source(self, tile_start, page_id=None):
        """Load native small-page V in the selected lane- or wave-grouped layout."""
        traits = self.traits
        if const_expr(traits.PAGE_SIZE == 1):
            return self._load_v_fp8_page1_source(tile_start, page_id=page_id)
        if const_expr(traits.V_LOAD_LAYOUT == "lane_groups"):
            token = fx.Int64(tile_start) + (
                self.lane_in_warp // fx.Int64(16)
            ) * fx.Int64(16)
            d_col = self.wave_id * fx.Int64(16) + self.lane_in_warp % fx.Int64(16)
            if page_id is None:
                page_id = self.load_page_id(token, uniform=False)
            src_elem = (
                page_id * self.v_page_bytes
                + self.kv_head_idx * traits.PAGE_SIZE * traits.HEAD_DIM_V
                + d_col * traits.KV_VEC_SIZE
            )
            return self.global_load_fp8x16(
                self.v_base_iter, src_elem, token < self.seqlen_kv_v, is_value=True
            )
        token, d_col, src_div, src_elem = self._page16_v_source(tile_start, page_id)
        valid = token < self.seqlen_kv_v
        src_elem = valid.select(src_elem, fx.Int64(self.v_page_bytes))
        source = self.buffer_load_fp8x16(src_div, src_elem)
        if const_expr(traits.FP8_PV_SEGMENTED):
            tail_elem = (valid & (d_col + traits.FP8_V_H1 < traits.HEAD_DIM_V)).select(
                src_elem + fx.Int64(traits.FP8_V_H1 * traits.KV_VEC_SIZE),
                fx.Int64(self.v_page_bytes),
            )
            tail = self.buffer_load_fp8x16(src_div, tail_elem)
            source = Vec(source).shuffle(Vec(tail), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
        return source

    def _load_v_fp8_page1_source(self, tile_start, page_id=None):
        """Load token-major vectors for the selected full/scattered V transpose."""
        traits = self.traits
        token = fx.Int64(tile_start) + self.lane_in_warp
        d_base = self.wave_id * fx.Int64(16)
        if page_id is None:
            page_id = self.load_page_id(token, uniform=False)
        byte_offset = (
            page_id * self.v_page_bytes + self.kv_head_idx * traits.HEAD_DIM_V + d_base
        )
        valid = token < self.seqlen_kv_v
        prefix = self.global_load_fp8x16(
            self.v_base_iter, byte_offset, valid, is_value=True
        )
        stages = 2 if traits.V_LOAD_LAYOUT == "token_words" else 4
        source = _transpose_v_fp8_16x16(prefix, self.lane_in_warp, stages=stages)
        if const_expr(traits.FP8_PV_SEGMENTED):
            tail = self.global_load_fp8x16(
                self.v_base_iter,
                byte_offset + traits.FP8_V_H1,
                valid & (d_base + traits.FP8_V_H1 < traits.HEAD_DIM_V),
                is_value=True,
            )
            tail_source = _transpose_v_fp8_16x16(tail, self.lane_in_warp, stages=2)
            source = source.shuffle(tail_source, [0, 1, 2, 3, 4, 5, 6, 7])
        return source.ir_value()

    def store_v_source(self, src_i32x4, buf_id, tile_start, *, mask_padding=True):
        """Store a load_v_source packet in the common MFMA byte order."""
        if const_expr(self.traits.V_LOAD_LAYOUT == "token_words"):
            self._store_v_fp8_page1(src_i32x4, buf_id)
            return
        page16_scatter = self.traits.V_LOAD_LAYOUT == "page_waves"

        if const_expr(mask_padding and self.traits.PAGE_SIZE != 1):
            group = self.wave_id_uni % 4 if page16_scatter else self.lane_in_warp // 16
            token = fx.Int64(tile_start) + fx.Int64(group) * 16
            src_i32x4 = self._mask_v_fp8_group(src_i32x4, token)
        if const_expr(page16_scatter):
            self._store_v_fp8_page16(src_i32x4, buf_id)
            return
        if const_expr(self.traits.FP8_PV_SEGMENTED):
            source = Vec(src_i32x4)
            prefix = source.shuffle(source, [0, 1, 2, 3]).ir_value()
            tail = source.shuffle(source, [4, 5, 6, 7]).ir_value()
            self._store_v_fp8_vectorized_bankpad_segment(prefix, buf_id, 0)

            @flyc.jit
            def _store_tail():
                if self.wave_id_uni < fx.Int64(
                    self.traits.FP8_V_H2 // self.traits.KV_VEC_SIZE
                ):
                    self._store_v_fp8_vectorized_bankpad_segment(
                        tail, buf_id, self.traits.FP8_V_H1
                    )

            _store_tail()
        else:
            self._store_v_fp8_vectorized_bankpad_segment(src_i32x4, buf_id, 0)

    def _store_v_fp8_page1(self, source, buf_id):
        """Complete the byte transpose by scattering four-token words to D rows."""
        traits = self.traits
        words = Vec(source)
        lane = self.lane_in_warp
        # Retained V token-bit order is [0, 1, 4, 3, 5, 2]; low bits stay in each word.
        token_word = (
            ((lane & 16) >> 2) + ((lane & 32) >> 1) + ((lane & 4) << 3) + (lane & 8)
        )
        aligned_base = (
            (self.lds_vt_base_idx + fx.Int64(127)) // fx.Int64(128)
        ) * fx.Int64(128)
        tile_base = (
            aligned_base
            - self.lds_vt_base_idx
            + buf_id * (traits.HEAD_DIM_V * traits.FP8_V_ROW_STRIDE)
        )
        d_col = self.wave_id * fx.Int64(16) + (lane & 3)

        def _store_segment(first_word, d_offset):
            for word in range_constexpr(4):
                row = d_col + fx.Int64(d_offset + word * 4)
                token_offset = token_word
                if const_expr(traits.HEAD_DIM_V == 192):
                    token_offset = token_offset ^ (
                        ((row // fx.Int64(4)) % fx.Int64(4)) * fx.Int64(16)
                    )
                dst = tile_base + row * traits.FP8_V_ROW_STRIDE + token_offset
                ptr = fx.add_offset(
                    fx.get_iter(self.v_lds_i32_tiles), fx.Uint32(dst) // 4
                )
                _store(ptr, fx.Int32(words[first_word + word]))

        _store_segment(0, 0)
        if const_expr(traits.FP8_PV_SEGMENTED):

            @flyc.jit
            def _store_tail():
                if self.wave_id_uni < fx.Int64(traits.FP8_V_H2 // traits.KV_VEC_SIZE):
                    _store_segment(4, traits.FP8_V_H1)

            _store_tail()

    def _store_v_fp8_page16(self, source, buf_id, segment_offset=0, include_tail=True):
        """Scatter native page vectors into the retained MFMA byte order."""
        traits = self.traits
        words = Vec(source)
        page = self.wave_id % fx.Int64(4)
        group_offset = (page % fx.Int64(2)) * fx.Int64(4) + (
            page // fx.Int64(2)
        ) * fx.Int64(16)
        aligned_base = (
            (self.lds_vt_base_idx + fx.Int64(127)) // fx.Int64(128)
        ) * fx.Int64(128)
        tile_base = (
            aligned_base
            - self.lds_vt_base_idx
            + buf_id * (traits.HEAD_DIM_V * traits.FP8_V_ROW_STRIDE)
        )
        d_col = (self.wave_id // fx.Int64(4)) * fx.Int64(64) + self.lane_in_warp

        def _store_segment(first_word, d_offset):
            row = d_col + fx.Int64(d_offset)
            row_base = tile_base + row * traits.FP8_V_ROW_STRIDE
            for word in range_constexpr(4):
                token_offset = group_offset + (word % 2) * 32 + (word // 2) * 8
                if const_expr(traits.HEAD_DIM_V == 192):
                    token_offset = token_offset ^ (
                        ((row // fx.Int64(4)) % fx.Int64(4)) * fx.Int64(16)
                    )
                ptr = fx.add_offset(
                    fx.get_iter(self.v_lds_i32_tiles),
                    fx.Uint32(row_base + token_offset) // 4,
                )
                _store(ptr, fx.Int32(words[first_word + word]))

        _store_segment(0, segment_offset)
        if const_expr(traits.FP8_PV_SEGMENTED and include_tail):

            @flyc.jit
            def _store_tail_pages():
                if self.wave_id_uni < fx.Int64(4):
                    _store_segment(4, traits.FP8_V_H1)

            _store_tail_pages()

    def _store_v_fp8_vectorized_bankpad_segment(self, src_i32x4, buf_id, d_offset):
        traits = self.traits
        row_stride = traits.FP8_V_ROW_STRIDE
        v_tile_bytes = traits.HEAD_DIM_V * row_stride
        aligned_base = (
            (self.lds_vt_base_idx + fx.Int64(127)) // fx.Int64(128)
        ) * fx.Int64(128)
        dst_base = aligned_base + fx.Int64(buf_id * v_tile_bytes)
        n_group16 = self.lane_in_warp // fx.Int64(16)
        d_col = (
            fx.Int64(d_offset)
            + self.wave_id * fx.Int64(16)
            + self.lane_in_warp % fx.Int64(16)
        )
        reordered = self._permute_v_fp8_vectorized(src_i32x4)
        dest_group16 = (n_group16 % fx.Int64(2)) * fx.Int64(2) + n_group16 // fx.Int64(
            2
        )
        if const_expr(traits.HEAD_DIM_V == 192):
            # XOR 16-byte groups instead of padding each row in the six-slot ring.
            dest_group16 = dest_group16 ^ ((d_col // fx.Int64(4)) % fx.Int64(4))
        dst_byte = dst_base + d_col * fx.Int64(row_stride) + dest_group16 * fx.Int64(16)
        self._store_v_fp8_lds(reordered, dst_byte - self.lds_vt_base_idx)


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

        if const_expr(traits.KV_VECTORIZED):

            def _read_vec_key(key):
                key_sigma = _sigma_k_tile_n(key)
                packs = []
                for ws in range_constexpr(traits.HEAD_DIM // 64):
                    dg0 = ws * 4 + self.lane_div_32 * 2
                    if const_expr(traits.PAGED and traits.K_LDS_PAGE_GROUPED):
                        # [D-group/4, page, D-group%4, token%16, byte]. The
                        # offset delta from page-64 layout is divisible by 256,
                        # preserving the gfx950 LDS bank index.
                        oct_idx = (
                            (ws * 4 + key_sigma // 16) * 64
                            + self.lane_div_32 * 32
                            + key_sigma % 16
                        )
                        off0 = k_base + oct_idx * traits.KV_VEC_SIZE
                        off1 = off0 + 16 * traits.KV_VEC_SIZE
                    else:
                        off0 = (
                            k_base
                            + (dg0 * traits.BLOCK_N + key_sigma) * traits.KV_VEC_SIZE
                        ) * traits.ELEM_BYTES
                        off1 = (
                            k_base
                            + ((dg0 + 1) * traits.BLOCK_N + key_sigma)
                            * traits.KV_VEC_SIZE
                        ) * traits.ELEM_BYTES
                    lo = Vec(self.read_i32x4_lds(off0), (4,), fx.Int32)
                    hi = Vec(self.read_i32x4_lds(off1), (4,), fx.Int32)
                    packs.append(lo.shuffle(hi, [0, 1, 2, 3, 4, 5, 6, 7]).ir_value())
                return packs

            return (_read_vec_key(n_lo), _read_vec_key(n_hi))

        rows_per_line = traits.NUM_WAVES

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

    def load_v(self, buf_id):
        if const_expr(self.traits.PAGED):
            return self._load_v_fp8_vectorized_bankpad(buf_id)
        return self._load_v_fp8_block(buf_id)

    def _load_v_fp8_vectorized_bankpad(self, buf_id):
        """Read the D-major V tile into the FP8 MFMA A fragment."""
        packs = [[None] * self.traits.D_CHUNKS for _ in range(4)]
        for dc in range_constexpr(self.traits.D_CHUNKS):
            fragment = Vec(self.load_v_fragment(buf_id, dc)).bitcast(fx.Int64)
            for ks in range_constexpr(4):
                packs[ks][dc] = fragment[ks].ir_value()
        return packs

    def load_v_fragment(self, buf_id, dc):
        """Read one 32-channel V fragment in the retained MFMA byte order."""
        traits = self.traits
        row_stride = traits.FP8_V_ROW_STRIDE
        v_tile_bytes = traits.HEAD_DIM_V * row_stride
        aligned_base = (
            (self.lds_vt_base_idx + fx.Int64(127)) // fx.Int64(128)
        ) * fx.Int64(128)
        tile_base = aligned_base + fx.Int64(buf_id * v_tile_bytes)
        token_base = self.lane_div_32 * fx.Int64(32)
        d_row = fx.Int64(dc * traits.D_CHUNK) + self.lane_mod_32
        row_base = tile_base + d_row * fx.Int64(row_stride)
        halves = []
        for half in range_constexpr(2):
            token_offset = token_base + fx.Int64(half * 16)
            if const_expr(traits.HEAD_DIM_V == 192):
                token_offset = token_offset ^ (
                    ((d_row // fx.Int64(4)) % fx.Int64(4)) * fx.Int64(16)
                )
            view = fx.slice(
                self.v_lds_i32_tiles,
                (None, fx.Uint32(row_base + token_offset - self.lds_vt_base_idx) // 16),
            )
            halves.append(_load(fx.get_iter(view), dtype=fx.Int32, count=4))
        return halves[0].shuffle(halves[1], [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

    def _load_v_fp8_block(self, buf_id):
        traits = self.traits
        v_tile_bytes = (traits.BLOCK_N // 8) * (traits.HEAD_DIM_V // 16) * 128
        buf_off = buf_id * v_tile_bytes
        nbands = traits.HEAD_DIM_V // 16
        rh = (self.lane % fx.Index(32)) // fx.Index(16)
        l16 = self.lane % fx.Index(16)
        lane_hi = self.lane // fx.Index(32)
        aligned_base = (
            (self.lds_vt_base_idx + fx.Index(127)) // fx.Index(128)
        ) * fx.Index(128)
        base = fx.Int32(
            aligned_base
            + buf_off
            + rh * fx.Index(128)
            + l16 * fx.Index(8)
            + lane_hi * fx.Index(nbands * 128)
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
