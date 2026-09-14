# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Gather A / A-scale rows from a dense TP buffer by moe_sorting token ids.

EP MegaMoE ``ATileLoader`` reads expert-packed contiguous rows. TP activations
stay rank-concatenated; GEMM1 tiles index them through ``sorted_token_ids``.
"""

from __future__ import annotations

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from .gemm_util import AScaleLoader, _buffer_load, _make_buffer

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOKEN_ID_MASK = 0x00FFFFFF


def _register_loader_source_dir():
    """FlyDSL disk cache hashes EXTRA_SOURCE_DIRS, not imported class bodies."""
    try:
        from flydsl.compiler.jit_function import EXTRA_SOURCE_DIRS
    except ImportError:
        EXTRA_SOURCE_DIRS = None
    if EXTRA_SOURCE_DIRS is not None and _HERE not in EXTRA_SOURCE_DIRS:
        EXTRA_SOURCE_DIRS.append(_HERE)
    cur = os.environ.get("FLYDSL_EXTRA_SOURCE_DIRS", "")
    parts = [p for p in cur.split(":") if p]
    if _HERE not in parts:
        parts.append(_HERE)
        os.environ["FLYDSL_EXTRA_SOURCE_DIRS"] = ":".join(parts)


_register_loader_source_dir()


def _decode_token(fused_i32, n_tokens):
    token = fused_i32 & fx.Int32(_TOKEN_ID_MASK)
    return (token < n_tokens).select(token, n_tokens)


class TokenATileLoader:
    """Dense ``[n_tokens+1, H]`` FP8 rows gathered by fused sorted token ids."""

    def __init__(
        self,
        *,
        row_bytes,
        sort_block_m,
        k_step_bytes,
        total_threads,
        swizzle=False,
        x_tensor=None,
        sorted_ids=None,
        tokens=None,
        async_copy=False,
    ):
        assert x_tensor is not None
        assert sorted_ids is not None
        assert tokens is not None
        assert not async_copy, "TokenATileLoader Step 1 does not implement async_a_copy"
        self._sort_block_m = sort_block_m
        self._k_step_bytes = k_step_bytes
        self._total_threads = total_threads
        self._swizzle = swizzle
        self._row_bytes = row_bytes
        self._tx = fx.thread_idx.x
        self._x_tensor = x_tensor
        self._sorted_rsrc = _make_buffer(sorted_ids, fx.Int32)
        self._tokens = tokens
        self._async_copy = False

    def for_tile(self, tile_row_base_i32):
        """Resolve this tile's token ids and LDS swizzle; dense A buffer is full-width."""
        self._x_rsrc = _make_buffer(
            self._x_tensor,
            fx.Int32,
            4,
            max_size=False,
            num_records_bytes=(self._tokens + fx.Int32(1)) * fx.Int32(self._row_bytes),
        )
        chunks_per_row = self._k_step_bytes // 16
        row_stride_i32 = self._k_step_bytes // 4
        total_chunks = self._sort_block_m * chunks_per_row
        self._chunks = []
        for c in range_constexpr(0, total_chunks, self._total_threads):
            lin = fx.Int32(c) + fx.Int32(self._tx)
            row = lin // fx.Int32(chunks_per_row)
            chunk = lin % fx.Int32(chunks_per_row)
            if const_expr(self._swizzle):
                col_i32 = chunk * fx.Int32(4)
                swz = row * fx.Int32(row_stride_i32) + (
                    col_i32 ^ ((row & fx.Int32(15)) << fx.Int32(2))
                )
                lds_byte = swz * fx.Int32(4)
            else:
                lds_byte = lin * fx.Int32(16)
            fused = _buffer_load(self._sorted_rsrc, tile_row_base_i32 + row, fx.Int32)
            token = _decode_token(fused, self._tokens)
            self._chunks.append((lds_byte, token, chunk))

    def load_regs(self, k_step_byte_off):
        """Gather this K-step's 16-B chunks from dense rows ``token * H + k_off``."""
        koff = fx.Int32(k_step_byte_off)
        regs = []
        for lds_byte, token, chunk in self._chunks:
            src_byte = token * fx.Int32(self._row_bytes) + koff + chunk * fx.Int32(16)
            group = src_byte // fx.Int32(16)
            regs.append((lds_byte, _buffer_load(self._x_rsrc, group, fx.Int32, 4)))
        return regs

    def store(self, lds_dst, regs, base_i32=0):
        """Scatter loaded chunks into LDS via ds_write (same swizzle as ATileLoader)."""
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


class TokenAScaleLoader(AScaleLoader):
    """Stage per-token E8M0 scales from dense ``[n_tokens+1, H/32]`` into LDS."""

    def __init__(
        self,
        *,
        scale_tensor,
        m_repeat,
        model_dim,
        sort_block_m,
        total_threads,
        sorted_ids,
        tokens,
    ):
        n_scale = model_dim // 32
        scale_rsrc = _make_buffer(
            scale_tensor,
            fx.Int32,
            max_size=False,
            num_records_bytes=(tokens + fx.Int32(1)) * fx.Int32(n_scale),
        )
        super().__init__(
            scale_rsrc=scale_rsrc,
            m_repeat=m_repeat,
            model_dim=model_dim,
            sort_block_m=sort_block_m,
            total_threads=total_threads,
        )
        self._sorted_rsrc = _make_buffer(sorted_ids, fx.Int32)
        self._tokens = tokens

    def stage(self, lds_ascale, tile_row_base_i32):
        """Copy this tile's e8m0 rows into LDS, one i32 at a time (row-aligned)."""
        i32_per_row = self._n_scale // 4
        assert self._n_scale % 4 == 0
        n_i32 = self._sort_block_m * i32_per_row

        @flyc.jit
        def copy_i32(lin: fx.Int32):
            if lin < fx.Int32(n_i32):
                row = lin // fx.Int32(i32_per_row)
                col = lin % fx.Int32(i32_per_row)
                fused = _buffer_load(
                    self._sorted_rsrc, tile_row_base_i32 + row, fx.Int32
                )
                token = _decode_token(fused, self._tokens)
                src = token * fx.Int32(self._n_scale) + col * fx.Int32(4)
                v = _buffer_load(self._rsrc, src // fx.Int32(4), fx.Int32)
                ptr = fx.add_offset(fx.recast_iter(fx.Int32, lds_ascale.ptr), lin)
                fx.ptr_store(Vec.from_elements([v], fx.Int32), ptr)

        for c in range_constexpr(0, n_i32, self._total_threads):
            copy_i32(fx.Int32(c) + fx.Int32(self._tx))
