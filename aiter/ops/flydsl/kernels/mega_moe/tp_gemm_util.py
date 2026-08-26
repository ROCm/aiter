# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A-operand loaders that gather rows by token id instead of reading them contiguously.

MegaMoE's EP dispatch physically permutes rows into expert-major order as a side
effect of pushing them, so its loaders read 32 contiguous rows starting at a tile
base. TP's all-gather produces a dense rank-major buffer and expresses the
permutation as an index list, so the loaders have to gather.

These are copies of gemm_util.ATileLoader / AScaleLoader with the address
computation changed and nothing else. gemm_util.py is frozen; see section 6 of
docs/superpowers/specs/2026-08-26-tp-moe-stage1-fused-p2p-design.md.

FOUR CONSTRAINTS THAT ARE NOT VISIBLE IN THE API AND FAIL SILENTLY:

1. ``load_step`` MUST issue zero VMEM instructions. gemm1.py:152-156 has a
   hardcoded ``wait_lds_barrier(NUM_ACC_N * _PACK + NUM_B_SCALE)`` that proves
   the A prefetch DMAs retired by counting the VMEM issued after them. That count
   assumes the A-scale loader reads LDS, not memory. Move the token lookup into
   load_step and the barrier releases early, giving half-written A tiles and
   silently wrong numbers.
2. ``load_step`` must return EXACTLY ``m_repeat // 2`` scalar i32 values.
   gemm1.py:97-104 unpacks the loop-carried state by that count.
3. Anything depending on ``tile_row_base`` belongs in ``for_tile`` / ``stage``,
   never ``__init__``: __init__ runs outside the persistent tile loop.
4. The LDS layout is unchanged. AS2RLoader.load_operand reads LDS by tile-local
   slot and its XOR swizzle must match store() bit for bit.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from .gemm_util import _PACK, _buffer_load, _make_buffer


def _clamp_to_pad(tok, pad_row):
    """min(tok, pad_row) for i32. fx.Int32 has no ``.min``; select is the fx idiom."""
    pad = fx.Int32(pad_row)
    return (tok < pad).select(tok, pad)


class TPATileLoader:
    """Dense rank-major A rows gathered by token id, gmem->reg->LDS."""

    def __init__(
        self,
        *,
        row_bytes,
        sort_block_m,
        k_step_bytes,
        total_threads,
        swizzle=False,
        x_tensor=None,
        tok_rsrc=None,
        pad_row=None,
        total_rows=None,
        async_copy=False,
    ):
        assert x_tensor is not None and tok_rsrc is not None
        assert pad_row is not None and total_rows is not None
        # num_records is a 32-bit byte count, so the whole A tensor must fit.
        assert total_rows * row_bytes < (1 << 32), (
            f"A tensor {total_rows * row_bytes} bytes exceeds the 32-bit buffer "
            "num_records; a per-row descriptor is impossible because buffer "
            "resources are wave-uniform (SGPR) and token ids are per-lane"
        )
        self._sort_block_m = sort_block_m
        self._k_step_bytes = k_step_bytes
        self._total_threads = total_threads
        self._swizzle = swizzle
        self._row_bytes = row_bytes
        self._tx = fx.thread_idx.x
        self._wave = self._tx // 64
        self._x_tensor = x_tensor
        self._tok_rsrc = tok_rsrc
        self._pad_row = int(pad_row)
        self._async_copy = bool(async_copy)
        # One wide descriptor over the whole A tensor, built once.
        self._rsrc = _make_buffer(
            x_tensor,
            fx.Int32,
            4,
            max_size=False,
            num_records_bytes=total_rows * row_bytes,
        )
        if const_expr(self._async_copy):
            assert total_threads % 64 == 0
            assert (sort_block_m * 16) % total_threads == 0
            assert row_bytes % 16 == 0 and k_step_bytes % 16 == 0
            self._dma_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopyLDS128b(),
                128,
            )
            self._dma = fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        fx.get_iter(x_tensor),
                        fx.make_layout(total_rows * row_bytes, 1),
                    )
                ),
                max_size=False,
                num_records_bytes=total_rows * row_bytes,
            )
            self._tile_dma = fx.logical_divide(self._dma, fx.make_layout(1, 1))

    def _row_of(self, tile_row_base_i32, row_i32):
        """sorted slot -> dense A row. Padding slots clamp to the zeroed pad row."""
        tok = _buffer_load(self._tok_rsrc, tile_row_base_i32 + row_i32, fx.Int32)
        tok = tok & fx.Int32(0x00FFFFFF)
        return _clamp_to_pad(tok, self._pad_row)

    def for_tile(self, tile_row_base_i32):
        """Precompute this tile's per-thread global byte offsets and LDS slots."""
        chunks_per_row = self._k_step_bytes // 16
        row_stride_i32 = self._k_step_bytes // 4
        total_chunks = self._sort_block_m * chunks_per_row
        self._chunks = []
        for c in range_constexpr(0, total_chunks, self._total_threads):
            lin = fx.Int32(c) + fx.Int32(self._tx)
            row = lin // fx.Int32(chunks_per_row)
            chunk = lin - row * fx.Int32(chunks_per_row)
            row_byte = self._row_of(tile_row_base_i32, row) * fx.Int32(self._row_bytes)
            if const_expr(self._swizzle):
                col_i32 = chunk * fx.Int32(4)
                swz = row * fx.Int32(row_stride_i32) + (
                    col_i32 ^ ((row & fx.Int32(15)) << fx.Int32(2))
                )
                lds_byte = swz * fx.Int32(4)
            else:
                lds_byte = lin * fx.Int32(16)
            self._chunks.append((lds_byte, row_byte + chunk * fx.Int32(16)))
        if const_expr(self._async_copy):
            # Hoisted out of the K loop on purpose: row/token are K-independent,
            # and re-reading the token table per K step would both cost VMEM and
            # perturb the hardcoded vmcnt in gemm1.py:152-156.
            self._dma_rows = []
            total = self._sort_block_m * 16
            for round_base in range_constexpr(0, total, self._total_threads):
                physical = fx.Int32(round_base) + fx.Int32(self._tx)
                row = physical // fx.Int32(16)
                physical_chunk = physical - row * fx.Int32(16)
                if const_expr(self._swizzle):
                    logical_chunk = physical_chunk ^ (row & fx.Int32(15))
                else:
                    logical_chunk = physical_chunk
                src_row_byte = self._row_of(tile_row_base_i32, row) * fx.Int32(
                    self._row_bytes
                )
                self._dma_rows.append((round_base, src_row_byte, logical_chunk))

    def load_regs(self, k_step_byte_off):
        """Read this K-step's 16-byte chunks gmem->reg. Only the K offset varies."""
        koff = fx.Int32(k_step_byte_off)
        regs = []
        for lds_byte, chunk_base in self._chunks:
            group = (chunk_base + koff) // fx.Int32(16)
            regs.append((lds_byte, _buffer_load(self._rsrc, group, fx.Int32, 4)))
        return regs

    def store(self, lds_dst, regs, base_i32=0):
        """Scatter loaded chunks into LDS. Byte-identical to ATileLoader.store."""
        base_bytes = fx.Int32(base_i32) * fx.Int32(4)
        for lds_byte, v in regs:
            dst = fx.make_view(
                fx.add_offset(
                    fx.recast_iter(fx.Int32, lds_dst.ptr),
                    (base_bytes + lds_byte) // fx.Int32(4),
                ),
                fx.make_layout(4, 1),
            )
            fragment = fx.make_rmem_tensor(4, fx.Int32)
            fragment.store(Vec(v))
            fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32), fragment, dst)

    def prefetch_to_lds(self, k_step_byte_off, lds_dst, base_i32=0):
        """Direct global->LDS copies using the token ids cached by for_tile."""
        koff = fx.Int32(k_step_byte_off)
        base_bytes = fx.Int32(base_i32) * fx.Int32(4)
        lds_f8 = fx.recast_iter(fx.Float8E4M3FN, lds_dst.ptr)
        for round_base, src_row_byte, logical_chunk in self._dma_rows:
            src_byte = src_row_byte + koff + logical_chunk * fx.Int32(16)
            src = fx.slice(self._tile_dma, (None, src_byte))
            wave_base = base_bytes + fx.Int32((round_base + self._wave * 64) * 16)
            dst = fx.make_view(fx.add_offset(lds_f8, wave_base), fx.make_layout(1, 1))
            fx.copy(self._dma_atom, src, dst)


class TPAScaleLoader:
    """Per-1x32 E8M0 A scales gathered by token id, staged to LDS once per tile."""

    def __init__(
        self,
        *,
        scale_rsrc,
        m_repeat,
        model_dim,
        sort_block_m,
        total_threads,
        tok_rsrc,
        pad_row,
    ):
        # stage() decomposes into (row, chunk-in-row), so a row must be a whole
        # number of 16-byte chunks. The original was a flat linear copy and only
        # needed sort_block_m*n_scale % 16 == 0.
        assert (model_dim // 32) % 16 == 0, (
            f"model_dim={model_dim} gives n_scale={model_dim // 32}, which is not "
            "a multiple of 16; the gathered stage() needs whole chunks per row"
        )
        self._rsrc = scale_rsrc
        self._n_scale = model_dim // 32
        self._lane = fx.thread_idx.x % 64
        self._n_groups = m_repeat // _PACK
        self._sort_block_m = sort_block_m
        self._total_threads = total_threads
        self._tx = fx.thread_idx.x
        self._tok_rsrc = tok_rsrc
        self._pad_row = int(pad_row)

    def stage(self, lds_ascale, tile_row_base_i32):
        """Gather this tile's e8m0 rows into LDS as a [sort_block_m, n_scale] block.

        Row-decomposed, unlike the contiguous original: source rows are scattered
        but the LDS destination is identical, so everything downstream is unchanged.
        """
        chunks_per_row = self._n_scale // 16
        total_chunks = self._sort_block_m * chunks_per_row

        @flyc.jit
        def copy_chunk(lin: fx.Int32):
            if lin < fx.Int32(total_chunks):
                row = lin // fx.Int32(chunks_per_row)
                cir = lin - row * fx.Int32(chunks_per_row)
                tok = _buffer_load(self._tok_rsrc, tile_row_base_i32 + row, fx.Int32)
                tok = _clamp_to_pad(tok & fx.Int32(0x00FFFFFF), self._pad_row)
                src_group = (
                    tok * fx.Int32(self._n_scale) + cir * fx.Int32(16)
                ) // fx.Int32(16)
                v = _buffer_load(self._rsrc, src_group, fx.Int32, 4)
                dst = fx.make_view(
                    fx.add_offset(
                        fx.recast_iter(fx.Int32, lds_ascale.ptr), lin * fx.Int32(4)
                    ),
                    fx.make_layout(4, 1),
                )
                fragment = fx.make_rmem_tensor(4, fx.Int32)
                fragment.store(v)
                fx.copy(
                    fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32), fragment, dst
                )

        for c in range_constexpr(0, total_chunks, self._total_threads):
            # 32*14 = 448 chunks does not divide a 256/512-thread CTA, hence the
            # predicate inside copy_chunk. Same as the original.
            copy_chunk(fx.Int32(c) + fx.Int32(self._tx))

    def load_step(self, lds_ascale, kstep_i32):
        """One packed i32 per pack-group, read from LDS. MUST issue zero VMEM."""
        lane_row = fx.Int32(self._lane % 16)
        col0 = kstep_i32 * fx.Int32(8) + fx.Int32(self._lane // 16)
        out = []
        for g in range_constexpr(self._n_groups):
            r0 = fx.Int32(g * 32) + lane_row
            r1 = r0 + fx.Int32(16)
            b = []
            for ksub in range_constexpr(_PACK):
                for rr in (r0, r1):
                    b.append(
                        self._read_scale_lds(lds_ascale, rr, col0 + fx.Int32(ksub * 4))
                    )
            out.append(
                b[0]
                | (b[1] << fx.Int32(8))
                | (b[2] << fx.Int32(16))
                | (b[3] << fx.Int32(24))
            )
        return out

    def _read_scale_lds(self, lds_ascale, row_i32, col_i32):
        off = row_i32 * fx.Int32(self._n_scale) + col_i32
        ptr = fx.recast_iter(
            fx.Uint8, fx.add_offset(lds_ascale.ptr, fx.make_int_tuple(off))
        )
        v = fx.make_view(ptr, fx.make_layout(1, 1)).load()
        return Vec(v, dtype=fx.Uint8)[0].to(fx.Int32)
