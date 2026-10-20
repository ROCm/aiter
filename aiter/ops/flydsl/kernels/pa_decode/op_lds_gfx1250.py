# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors


"""LDS access, FP8 conversion and Q/K/V staging for gfx1250 PA decode."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from ..kernels_common import create_llvm_ptr
from ..tensor_shim import _to_raw, buf_base_i64
from ..utils import rcp_f32


class Gfx1250Memory:
    def __init__(self, ctx):
        self.ctx = ctx
        self.t = ctx.traits

    def allocate(self):
        storage = fx.SharedAllocator().allocate(self.ctx.shared_storage).peek()
        self.base = fx.recast_iter(fx.Uint8, storage.buf.ptr)

    def pointer(self, offset, dtype):
        return fx.recast_iter(
            fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Shared, dtype.width // 8),
            fx.add_offset(self.base, offset),
        )

    def load(self, offset, dtype, n=1):
        return fx.Vector(
            fx.ptr_load(
                self.pointer(offset, dtype), result_type=fx.Vector.make_type(n, dtype)
            )
        )

    def store(self, offset, value, dtype):
        fx.ptr_store(value, self.pointer(offset, dtype))

    def fp8(self, values):
        # Unit scale preserves the explicitly normalized Q/P values. The gfx1250
        # instruction converts eight FP32 values into two packed dwords.
        return fx.Vector(
            fx.rocdl.cvt_scalef32_pk8_fp8_f32(
                T.i32x2, _to_raw(values), _to_raw(fx.Float32(1.0))
            )
        )

    def operand(self, offset, stride, k, lane16, half):
        # Each half-wave owns alternating 16-byte chunks of the contraction.
        words = []
        for j in range_constexpr(k * self.t.elem_bytes // 32):
            frag = self.load(offset + lane16 * stride + half * 16 + j * 32, fx.Int32, 4)
            words.extend([frag[i] for i in range_constexpr(4)])
        result = fx.Vector.from_elements(words, dtype=fx.Int32)
        return result.bitcast(fx.BFloat16) if self.t.bf16 else result

    def packed_operand(self, offset, rows, k, lane16, half):
        # WMMA distributes alternating 16-byte contraction chunks across the
        # two half-waves. Read them directly from the packed LDS layout.
        words = []
        for j in range_constexpr(k // 32):
            frag = self.load(
                offset + (half + 2 * j) * rows * 16 + lane16 * 16, fx.Int32, 4
            )
            words.extend([frag[i] for i in range_constexpr(4)])
        return fx.Vector.from_elements(words, dtype=fx.Int32)

    def async_copy(self, global_base, global_offset, shared_offset):
        src = create_llvm_ptr(global_base + fx.Int64(global_offset), address_space=1)
        dst = create_llvm_ptr(
            fx.Int64(fx.ptrtoint(self.base)) + fx.Int64(shared_offset), address_space=3
        )
        fx.rocdl.global_load_async_to_lds_b128(src, dst, 0)

    def tdm_copy(
        self, ptr, offset, shared_offset, shape, strides, extents, width, row_stride
    ):
        """Gather a packed cache slice into padded LDS using one wave."""
        packed_strides = []
        stride = 1
        for size in reversed(shape):
            packed_strides.insert(0, stride)
            stride *= size
        layout = fx.make_layout(shape, tuple(packed_strides))
        lds_strides = tuple(
            s // width * row_stride if s >= width else s for s in packed_strides
        )
        src = fx.Tensor(
            fx.make_view(
                fx.add_offset(fx.recast_iter(self.t.kv_dtype, ptr), offset),
                layout,
            )
        )
        atom = fx.rocdl.make_tdm_atom(
            src,
            extents,
            strides=strides,
            num_warps=1,
            pad_interval=width if row_stride != width else 0,
            pad_amount=row_stride - width,
            early_timeout=True,
        )
        dst = fx.Tensor(
            fx.make_view(
                self.pointer(shared_offset, self.t.kv_dtype),
                fx.make_layout(shape, lds_strides),
            )
        )
        fx.copy_atom_call(atom, src, dst)

    @flyc.jit
    def stage_tdm(self, token_base, kv_off, full_tile=False, cached_page=None):
        if const_expr(self.t.packed_lds):
            self.stage_packed_tdm(token_base, kv_off, full_tile, cached_page)
        else:
            self.stage_padded_tdm(token_base, kv_off, full_tile)

    @flyc.jit
    def stage_packed_tdm(self, token_base, kv_off, full_tile, cached_page=None):
        ctx, t = self.ctx, self.t
        if const_expr(t.TOKENS == 128):
            self.stage_page_tdm(token_base, kv_off, full_tile, cached_page)
            return
        table = fx.recast_iter(fx.Int32, ctx.block_tables_ptr)
        # Keep the cache's contiguous 16-byte chunks in LDS. TDM then gathers
        # whole spans without transposing them or inserting row padding.
        safe = (
            token_base
            if full_tile
            else (token_base < ctx.planned_context).select(token_base, fx.Int32(0))
        )
        page = fx.Int32(
            table[
                ctx.planned_seq * ctx.max_blocks_per_seq
                + fx.Uint32(safe) // t.block_size
            ]
        )
        off = (fx.Int64(page) * ctx.n_kv + ctx.kv_h) * t.head_dim * t.block_size
        pt = fx.Uint32(safe) % t.block_size
        valid = fx.max(
            fx.Int32(0),
            fx.min(fx.Int32(16), ctx.planned_context - token_base - ctx.warp * 16),
        )
        self.tdm_copy(
            ctx.key_cache_ptr,
            off + (pt + ctx.warp * 16) * 16,
            kv_off + ctx.warp * t.head_dim * 16,
            (t.head_dim // 16, 256) if full_tile else (t.head_dim // 16, 16, 16),
            (t.block_size * 16, 1) if full_tile else (t.block_size * 16, 16, 1),
            (None, None) if full_tile else (None, valid, None),
            16,
            16,
        )
        self.tdm_copy(
            ctx.value_cache_ptr,
            off + (pt // 16 + ctx.warp) * t.head_dim * 16,
            kv_off + t.k_bytes + ctx.warp * t.head_dim * 16,
            (t.head_dim * 16,) if full_tile else (t.head_dim, 16),
            (1,) if full_tile else (16, 1),
            (None,) if full_tile else (None, valid),
            16,
            16,
        )

    @flyc.jit
    def stage_page_tdm(self, token_base, kv_off, full_tile, cached_page=None):
        """Stage a packed 128-token slice, splitting K by dimension and V by token."""
        ctx, t = self.ctx, self.t
        table = fx.recast_iter(fx.Int32, ctx.block_tables_ptr)
        safe = (
            token_base
            if full_tile
            else (token_base < ctx.planned_context).select(token_base, fx.Int32(0))
        )
        if const_expr(cached_page is not None):
            page = cached_page
        else:
            page = fx.Int32(
                table[
                    ctx.planned_seq * ctx.max_blocks_per_seq
                    + fx.Uint32(safe) // t.block_size
                ]
            )
        off = (fx.Int64(page) * ctx.n_kv + ctx.kv_h) * t.head_dim * t.block_size
        pt = fx.Uint32(safe) % t.block_size
        valid = fx.max(
            fx.Int32(0), fx.min(fx.Int32(128), ctx.planned_context - token_base)
        )
        self.tdm_copy(
            ctx.key_cache_ptr,
            off + ctx.warp * (t.head_dim // 64) * t.block_size * 16 + pt * 16,
            kv_off + ctx.warp * (t.head_dim // 64) * 2048,
            (t.head_dim // 64, 2048) if full_tile else (t.head_dim // 64, 128, 16),
            (t.block_size * 16, 1) if full_tile else (t.block_size * 16, 16, 1),
            (None, None) if full_tile else (None, valid, None),
            16,
            16,
        )
        if const_expr(full_tile):
            self.tdm_copy(
                ctx.value_cache_ptr,
                off + (pt // 16 + ctx.warp * 2) * t.head_dim * 16,
                kv_off + t.k_bytes + ctx.warp * t.head_dim * 32,
                (t.head_dim * 32,),
                (1,),
                (None,),
                16,
                16,
            )
        else:
            # Exact token extents zero-fill invalid V bytes, including NaNs.
            for chunk in range_constexpr(2):
                count = fx.max(
                    fx.Int32(0),
                    fx.min(fx.Int32(16), valid - ctx.warp * 32 - chunk * 16),
                )
                self.tdm_copy(
                    ctx.value_cache_ptr,
                    off + (pt // 16 + ctx.warp * 2 + chunk) * t.head_dim * 16,
                    kv_off
                    + t.k_bytes
                    + ctx.warp * t.head_dim * 32
                    + chunk * t.head_dim * 16,
                    (t.head_dim, 16),
                    (16, 1),
                    (None, count),
                    16,
                    16,
                )

    def page_k_operand(self, offset, token, lane16, half):
        words = []
        for j in range_constexpr(4):
            frag = self.load(
                offset + (half + 2 * j) * 128 * 16 + (token + lane16) * 16, fx.Int32, 4
            )
            words.extend([frag[i] for i in range_constexpr(4)])
        return fx.Vector.from_elements(words, dtype=fx.Int32)

    @flyc.jit
    def stage_padded_tdm(self, token_base, kv_off, full_tile):
        ctx, t = self.ctx, self.t
        table = fx.recast_iter(fx.Int32, ctx.block_tables_ptr)
        # Explicit wave slices keep the hardware padding local to each copy.
        # Collective rank-3 atoms cannot apply row padding to their wave offset
        # with the installed lowering, so each wave owns a single-wave atom.
        absolute = token_base + ctx.warp * 16
        safe_token = (
            absolute
            if full_tile
            else (absolute < ctx.planned_context).select(absolute, fx.Int32(0))
        )
        page = fx.Int32(
            table[ctx.planned_seq * ctx.max_blocks_per_seq + safe_token // t.block_size]
        )
        off = (fx.Int64(page) * ctx.n_kv + ctx.kv_h) * t.head_dim * t.block_size
        page_token = safe_token % t.block_size
        valid = fx.max(
            fx.Int32(0), fx.min(fx.Int32(16), ctx.planned_context - absolute)
        )
        self.tdm_copy(
            ctx.key_cache_ptr,
            off + page_token * t.chunk,
            kv_off + ctx.warp * 16 * t.q_stride,
            (16, t.head_dim // t.chunk, t.chunk),
            (t.chunk, t.block_size * t.chunk, 1),
            (None if full_tile else valid, None, None),
            t.head_dim,
            t.q_stride // t.elem_bytes,
        )
        if const_expr(t.block_size == 16):
            v_tokens, v_rows = 16, t.head_dim
            v_dst = ctx.warp * 16 * t.elem_bytes
            v_d = fx.Int32(0)
        else:
            v_tokens, v_rows = t.TOKENS, t.head_dim // t.NWARP
            v_d = ctx.warp * v_rows
            v_dst = v_d * t.v_stride
            safe_token = (token_base < ctx.planned_context).select(
                token_base, fx.Int32(0)
            )
            page = fx.Int32(
                table[
                    ctx.planned_seq * ctx.max_blocks_per_seq
                    + safe_token // t.block_size
                ]
            )
            off = (fx.Int64(page) * ctx.n_kv + ctx.kv_h) * t.head_dim * t.block_size
            page_token = safe_token % t.block_size
            valid = fx.max(
                fx.Int32(0),
                fx.min(fx.Int32(v_tokens), ctx.planned_context - token_base),
            )
        if const_expr(t.trans_v):
            self.tdm_copy(
                ctx.value_cache_ptr,
                off + (page_token // t.chunk) * t.head_dim * t.chunk + v_d * t.chunk,
                kv_off + t.k_bytes + v_dst,
                (v_rows, v_tokens // t.chunk, t.chunk),
                (t.chunk, t.head_dim * t.chunk, 1),
                (None, None if full_tile else (valid + t.chunk - 1) // t.chunk, None),
                v_tokens,
                t.v_stride // t.elem_bytes,
            )
        else:
            self.tdm_copy(
                ctx.value_cache_ptr,
                off + v_d * t.block_size + page_token,
                kv_off + t.k_bytes + v_dst,
                (v_rows, v_tokens),
                (t.block_size, 1),
                (None, None if full_tile else valid),
                v_tokens,
                t.v_stride // t.elem_bytes,
            )

    @flyc.jit
    def stage_vector_dma(self, token_base, kv_off):
        ctx, t = self.ctx, self.t
        table = fx.recast_iter(fx.Int32, ctx.block_tables_ptr)
        key_base = buf_base_i64(ctx.key_cache_ptr)
        value_base = buf_base_i64(ctx.value_cache_ptr)
        # K cache has contiguous 16-byte head-dimension chunks.
        for i in range_constexpr(t.TOKENS * t.head_dim // t.chunk // 128):
            chunk = ctx.tid + i * 128
            token = chunk // (t.head_dim // t.chunk)
            d = (chunk % (t.head_dim // t.chunk)) * t.chunk
            absolute = token_base + token
            safe_token = (absolute < ctx.planned_context).select(absolute, fx.Int32(0))
            page = fx.Int32(
                table[
                    ctx.planned_seq * ctx.max_blocks_per_seq
                    + safe_token // t.block_size
                ]
            )
            off = (
                (fx.Int64(page) * ctx.n_kv + ctx.kv_h) * t.head_dim * t.block_size
                + fx.Int64(d // t.chunk) * t.block_size * t.chunk
                + (safe_token % t.block_size) * t.chunk
            )
            self.async_copy(
                key_base,
                off * t.elem_bytes,
                kv_off + token * t.q_stride + d * t.elem_bytes,
            )

        # Both supported V layouts have contiguous 16-byte token chunks.
        for i in range_constexpr(t.head_dim * (t.TOKENS // t.chunk) // 128):
            chunk = ctx.tid + i * 128
            d = chunk // (t.TOKENS // t.chunk)
            token = (chunk % (t.TOKENS // t.chunk)) * t.chunk
            absolute = token_base + token
            safe_token = (absolute < ctx.planned_context).select(absolute, fx.Int32(0))
            page = fx.Int32(
                table[
                    ctx.planned_seq * ctx.max_blocks_per_seq
                    + safe_token // t.block_size
                ]
            )
            page_token = safe_token % t.block_size
            off = (fx.Int64(page) * ctx.n_kv + ctx.kv_h) * t.head_dim * t.block_size
            if const_expr(t.trans_v):
                off = off + (page_token // t.chunk) * t.head_dim * t.chunk + d * t.chunk
            else:
                off = off + d * t.block_size + page_token
            self.async_copy(
                value_base,
                off * t.elem_bytes,
                kv_off + t.k_bytes + d * t.v_stride + token * t.elem_bytes,
            )

    @flyc.jit
    def stage_kv(self, token_base, buffer):
        # Constant full-tile extents keep the hot TDM descriptor path scalar.
        # The tail retains clamped extents and safe owned-page addresses.
        if token_base + self.t.TOKENS <= self.ctx.planned_context:
            self.stage_kv_impl(token_base, buffer, True)
        else:
            self.stage_kv_impl(token_base, buffer, False)

    @flyc.jit
    def stage_kv_impl(self, token_base, buffer, full_tile):
        ctx, t = self.ctx, self.t
        kv_off = t.q_bytes + buffer * t.kv_stride
        table = fx.recast_iter(fx.Int32, ctx.block_tables_ptr)
        page = fx.Int32(0)
        if const_expr(t.async_page):
            safe = (
                token_base
                if full_tile
                else (token_base < ctx.planned_context).select(token_base, fx.Int32(0))
            )
            page = fx.Int32(
                table[ctx.planned_seq * ctx.max_blocks_per_seq + safe // t.block_size]
            )
        if const_expr(t.use_tdm):
            self.stage_tdm(
                token_base, kv_off, full_tile, page if t.async_page else None
            )
        else:
            self.stage_vector_dma(token_base, kv_off)

        if const_expr(t.async_page and full_tile and t.per_token_kv):
            # One wave copies four FP32 scales per lane directly into LDS.
            # Waiting on ASYNC_CNT at consumption avoids the load/store drain
            # that otherwise serializes K/V staging with scale staging.
            index = fx.Int64(page) * ctx.stride_ks_block + ctx.kv_h * ctx.stride_ks_head
            if ctx.tid < t.WAVE:
                self.async_copy(
                    fx.Int64(fx.ptrtoint(ctx.key_scale_ptr)),
                    (index + ctx.tid * 4) * 4,
                    kv_off + t.k_bytes + t.v_bytes + ctx.tid * 16,
                )
                self.async_copy(
                    fx.Int64(fx.ptrtoint(ctx.value_scale_ptr)),
                    (index + ctx.tid * 4) * 4,
                    kv_off + t.k_bytes + t.v_bytes + t.TOKENS * 4 + ctx.tid * 16,
                )
        else:
            if const_expr(t.per_token_kv):
                ks = fx.recast_iter(fx.Float32, ctx.key_scale_ptr)
                vs = fx.recast_iter(fx.Float32, ctx.value_scale_ptr)
                if ctx.tid < t.TOKENS:
                    absolute = token_base + ctx.tid
                    safe_token = (
                        absolute
                        if full_tile
                        else (absolute < ctx.planned_context).select(
                            absolute, fx.Int32(0)
                        )
                    )
                    if const_expr(t.block_size >= t.TOKENS):
                        page_start = (
                            token_base
                            if full_tile
                            else (token_base < ctx.planned_context).select(
                                token_base, fx.Int32(0)
                            )
                        )
                        if const_expr(not t.async_page):
                            page = fx.Int32(
                                table[
                                    ctx.planned_seq * ctx.max_blocks_per_seq
                                    + page_start // t.block_size
                                ]
                            )
                        scale_idx = (
                            fx.Int64(page) * ctx.stride_ks_block
                            + ctx.kv_h * ctx.stride_ks_head
                            + page_start % t.block_size
                            + ctx.tid
                        )
                    else:
                        page = fx.Int32(
                            table[
                                ctx.planned_seq * ctx.max_blocks_per_seq
                                + safe_token // t.block_size
                            ]
                        )
                        scale_idx = (
                            fx.Int64(page) * ctx.stride_ks_block
                            + ctx.kv_h * ctx.stride_ks_head
                            + safe_token % t.block_size
                        )
                    kscale = fx.Float32(ks[scale_idx])
                    vscale = fx.Float32(vs[scale_idx])
                    if const_expr(not full_tile):
                        kscale = (absolute < ctx.planned_context).select(
                            kscale, fx.Float32(0.0)
                        )
                    # Normalize over the MTP window union, independently of the row.
                    visible = (
                        fx.Int32(1) if full_tile else absolute < ctx.planned_context
                    )
                    if const_expr(t.sliding_window > 0):
                        visible = visible & (
                            absolute
                            >= ctx.planned_context
                            - t.query_length
                            + 1
                            - t.sliding_window
                        )
                    vscale = visible.select(vscale, fx.Float32(0.0))
                    self.store(
                        kv_off + t.k_bytes + t.v_bytes + ctx.tid * 4, kscale, fx.Float32
                    )
                    self.store(
                        kv_off + t.k_bytes + t.v_bytes + (t.TOKENS + ctx.tid) * 4,
                        vscale,
                        fx.Float32,
                    )

    @flyc.jit
    def stage_query(self, first_tile=0):
        ctx, t = self.ctx, self.t
        query = fx.recast_iter(t.q_dtype, ctx.query_ptr)
        # Eight adjacent lanes stage each row, normalizing only for FP8.
        row, shard = ctx.tid // 8, ctx.tid % 8
        for m in range_constexpr(t.q_tiles):
            flat = (first_tile + m) * 16 + row
            shared_row = m * 16 + row
            units = []
            for j in range_constexpr(t.head_dim // 64):
                d = shard * (t.head_dim // 8) + j * 8
                qoff = (
                    (
                        fx.Int64(ctx.planned_seq) * t.query_length
                        + ctx.query_begin
                        + flat // t.query_group_size
                    )
                    * ctx.stride_q_row
                    + (ctx.kv_h * t.query_group_size + flat % t.query_group_size)
                    * ctx.stride_q_head
                    + d
                )
                values = fx.Vector.filled(8, 0.0, fx.Float32)
                if flat < t.rows:
                    values = fx.Vector(
                        fx.ptr_load(
                            fx.add_offset(query, qoff),
                            result_type=fx.Vector.make_type(8, t.q_dtype),
                        )
                    ).to(fx.Float32)
                units.append(values)
            if const_expr(not t.bf16):
                absmax = fx.Float32(0.0)
                for j in range_constexpr(len(units)):
                    absmax = fx.maxnumf(
                        absmax, fmath.absf(units[j]).reduce(ReductionOp.MAX)
                    )
                for shift in (4, 2, 1):
                    absmax = fx.maxnumf(absmax, absmax.shuffle_xor(shift, t.WAVE))
                scale = absmax * fx.Float32(1.0 / t.FP8_MAX)
                inv = fx.Float32(rcp_f32(fx.maxnumf(scale, fx.Float32(1e-20))))
            for j in range_constexpr(len(units)):
                if const_expr(t.bf16):
                    packed = units[j].to(fx.BFloat16)
                else:
                    packed = self.fp8(
                        units[j]
                        * fx.Vector.from_elements(
                            [fx.Float32(inv)], dtype=fx.Float32
                        ).broadcast_to(8)
                    )
                self.store(
                    shared_row * t.q_stride
                    + (shard * (t.head_dim // 8) + j * 8) * t.elem_bytes,
                    packed,
                    fx.BFloat16 if t.bf16 else fx.Int32,
                )
            if const_expr(not t.bf16):  # noqa: SIM102
                if shard == 0:
                    # Temporary query scales live in the unused Q row padding.
                    self.store(shared_row * t.q_stride + t.head_dim, scale, fx.Float32)
