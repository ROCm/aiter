# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors

"""Wave32 FP8/BF16 WMMA backend for planned paged decode on gfx1250.

Four waves split a 64-token compute tile within each 256-token plan tile.
K and V move from the packed paged cache to alternating LDS buffers via TDM
for per-token scales and power-of-two D128+, with vector DMA for other cases.
FP8 D128/page64+ transposed caches with one query tile retain their packed LDS
layout, avoiding
TDM transposition and padding; WMMA operands assemble directly from chunks.
FP8 Q and P use native eight-element conversion; BF16 uses K32 WMMA.
BF16 TDM is enabled for power-of-two D64–512, with vector DMA otherwise.
Softmax statistics and accumulation stay FP32.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import ReductionOp, T

from ..kernels_common import create_llvm_ptr
from ..tensor_shim import _to_raw, buf_base_i64
from ..utils import rcp_f32
from . import implementation_cache_tag
from .traits import KV_COMPUTE_BLOCK, LOG2E

_CACHE = {}


@dataclass(frozen=True)
class Gfx1250Traits:
    head_dim: int
    query_group_size: int
    block_size: int
    softmax_scale: float
    query_dtype: str
    per_token_kv: bool
    query_length: int
    trans_v: bool
    sliding_window: int
    compute_dtype: str = "fp8"

    WAVE = 32
    NWARP = 4
    TOKENS = 64
    FP8_MAX = 448.0

    @property
    def bf16(self):
        return self.compute_dtype == "bf16"

    @property
    def elem_bytes(self):
        return 2 if self.bf16 else 1

    @property
    def chunk(self):
        return 16 // self.elem_bytes

    @property
    def kv_dtype(self):
        return fx.BFloat16 if self.bf16 else fx.Uint8

    @property
    def rows(self):
        return self.query_length * self.query_group_size

    @property
    def m_tiles(self):
        return (self.rows + 15) // 16

    @property
    def q_stride(self):
        return self.head_dim * self.elem_bytes + 16

    @property
    def q_tiles(self):
        # The widest BF16 MTP tile fits by staging one query tile at a time.
        full = self.m_tiles * 16 * self.q_stride
        return (
            1 if full + self.kv_stride + self.p_bytes + 576 > 327680 else self.m_tiles
        )

    @property
    def q_bytes(self):
        return self.q_tiles * 16 * self.q_stride

    @property
    def packed_lds(self):
        return (
            not self.bf16
            and self.per_token_kv
            and self.head_dim == 128
            and self.block_size >= 64
            and self.trans_v
            and self.m_tiles == 1
        )

    @property
    def k_bytes(self):
        return self.TOKENS * (self.head_dim if self.packed_lds else self.q_stride)

    @property
    def v_stride(self):
        return self.TOKENS * self.elem_bytes + 16

    @property
    def v_bytes(self):
        return self.head_dim * (self.TOKENS if self.packed_lds else self.v_stride)

    @property
    def kv_stride(self):
        return self.k_bytes + self.v_bytes + 2 * self.TOKENS * 4

    @property
    def p_bytes(self):
        return 16 * self.v_stride

    @property
    def buffers(self):
        return (
            2 if self.q_bytes + 2 * self.kv_stride + self.p_bytes + 576 <= 327680 else 1
        )

    @property
    def p_offset(self):
        return self.q_bytes + self.buffers * self.kv_stride

    @property
    def stats_offset(self):
        return self.p_offset + self.p_bytes

    @property
    def total_bytes(self):
        return self.stats_offset + 576

    @property
    def use_tdm(self):
        # FP8 per-token D128+ benefits from scalar address generation; D64
        # and scalar scales measured faster with vector DMA. BF16 descriptors
        # fit D64–512. Row padding requires a power-of-two interval <= 1024 bytes.
        return (
            (self.bf16 or (self.per_token_kv and self.head_dim >= 128))
            and self.head_dim * self.elem_bytes <= 1024
            and self.head_dim & (self.head_dim - 1) == 0
        )

    @property
    def q_dtype(self):
        return fx.BFloat16 if self.query_dtype == "bf16" else fx.Float16


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
    def stage_tdm(self, token_base, kv_off, full_tile=False):
        if const_expr(self.t.packed_lds):
            self.stage_packed_tdm(token_base, kv_off, full_tile)
        else:
            self.stage_padded_tdm(token_base, kv_off, full_tile)

    @flyc.jit
    def stage_packed_tdm(self, token_base, kv_off, full_tile):
        ctx, t = self.ctx, self.t
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
            kv_off + ctx.warp * 2048,
            (8, 16, 16),
            (t.block_size * 16, 16, 1),
            (None, None if full_tile else valid, None),
            16,
            16,
        )
        self.tdm_copy(
            ctx.value_cache_ptr,
            off + (pt // 16 + ctx.warp) * 128 * 16,
            kv_off + t.k_bytes + ctx.warp * 2048,
            (128, 16),
            (16, 1),
            (None, None if full_tile else valid),
            16,
            16,
        )

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
        if const_expr(t.use_tdm):
            self.stage_tdm(token_base, kv_off, full_tile)
        else:
            self.stage_vector_dma(token_base, kv_off)

        if const_expr(t.per_token_kv):
            ks = fx.recast_iter(fx.Float32, ctx.key_scale_ptr)
            vs = fx.recast_iter(fx.Float32, ctx.value_scale_ptr)
            if ctx.tid < t.TOKENS:
                absolute = token_base + ctx.tid
                safe_token = (
                    absolute
                    if full_tile
                    else (absolute < ctx.planned_context).select(absolute, fx.Int32(0))
                )
                if const_expr(t.block_size >= t.TOKENS):
                    page_start = (
                        token_base
                        if full_tile
                        else (token_base < ctx.planned_context).select(
                            token_base, fx.Int32(0)
                        )
                    )
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
                visible = fx.Int32(1) if full_tile else absolute < ctx.planned_context
                if const_expr(t.sliding_window > 0):
                    visible = visible & (
                        absolute
                        >= ctx.planned_context - t.query_length + 1 - t.sliding_window
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


class Gfx1250Gemm:
    @staticmethod
    def mma(a, b, acc, k, bf16=False):
        if const_expr(bf16):
            result = fx.rocdl.wmma_f32_16x16x32_bf16(
                fx.Vector.make_type(8, fx.Float32),
                _to_raw(a),
                _to_raw(b),
                _to_raw(acc),
                modC=ir.Attribute.parse("#rocdl<wmma_c_modifier none>"),
                reuseA=False,
                reuseB=False,
            ).result
        elif const_expr(k == 128):
            result = fx.rocdl.wmma_f32_16x16x128_fp8_fp8(
                fx.Vector.make_type(8, fx.Float32), a, b, acc
            )
        else:
            result = fx.rocdl.wmma_f32_16x16x64_fp8_fp8(
                fx.Vector.make_type(8, fx.Float32),
                _to_raw(a),
                _to_raw(b),
                _to_raw(acc),
                modC=ir.Attribute.parse("#rocdl<wmma_c_modifier none>"),
                reuseA=False,
                reuseB=False,
            ).result
        return fx.Vector(result)


class Gfx1250Pipeline:
    def __init__(self, ctx):
        self.ctx, self.t = ctx, ctx.traits
        self.mem = Gfx1250Memory(ctx)

    @flyc.jit
    def run(self):
        ctx, t = self.ctx, self.t
        ctx.tid = fx.Int32(gpu.thread_id("x"))
        ctx.warp, ctx.lane = ctx.tid // t.WAVE, ctx.tid % t.WAVE
        ctx.lane16, ctx.half = ctx.lane % 16, ctx.lane // 16
        ctx.kv_h = fx.Int32(gpu.block_id("y"))
        ctx.n_kv = fx.Int32(gpu.grid_dim.y)
        self.mem.allocate()
        if const_expr(t.q_tiles == t.m_tiles):
            self.mem.stage_query()
        begin, end = (
            ctx.planned_start * KV_COMPUTE_BLOCK,
            ctx.planned_end * KV_COMPUTE_BLOCK,
        )
        self.mem.stage_kv(begin, fx.Int32(0))
        # Per-M-tile state: maximum, denominator, then wave-owned D fragments.
        vh_count = t.head_dim // 64
        initial = []
        for m in range_constexpr(t.m_tiles):
            initial.extend([fx.Float32(float("-inf")), fx.Float32(0.0)])
            initial.extend(
                [
                    fx.Vector.filled(8, 0.0, fx.Float32)
                    for vh in range_constexpr(vh_count)
                ]
            )
        state_stride = 2 + vh_count
        resident_q = not t.bf16 and t.head_dim == 128 and t.m_tiles == 1
        if const_expr(resident_q):
            gpu.barrier()
            q_operand = self.mem.operand(0, t.q_stride, 128, ctx.lane16, ctx.half)
            q_scale = fx.Float32(
                self.mem.load(ctx.lane16 * t.q_stride + t.head_dim, fx.Float32)[0]
            )
        for token_i, state in range(begin, end, t.TOKENS, init=initial):
            token_base = fx.Int32(token_i)
            buffer = ((token_base - begin) // t.TOKENS) % t.buffers
            if const_expr(t.use_tdm):
                tdm_ops.tensor_wait(0)
            else:
                fx.rocdl.s_wait_asynccnt(0)
            gpu.barrier()
            kv_off = t.q_bytes + buffer * t.kv_stride
            if const_expr(t.buffers == 2):  # noqa: SIM102
                if token_base + t.TOKENS < end:
                    self.mem.stage_kv(token_base + t.TOKENS, 1 - buffer)
            vnorm = fx.Float32(1.0)
            vfactor = fx.Float32(1.0 if t.bf16 else 1.0 / t.FP8_MAX)
            if const_expr(t.per_token_kv):
                # Every wave reduces the same 64 scales, avoiding a CTA barrier.
                scale_base = kv_off + t.k_bytes + t.v_bytes + t.TOKENS * 4
                vmax = fx.maxnumf(
                    fx.Float32(self.mem.load(scale_base + ctx.lane * 4, fx.Float32)[0]),
                    fx.Float32(
                        self.mem.load(scale_base + (ctx.lane + 32) * 4, fx.Float32)[0]
                    ),
                )
                for shift in (16, 8, 4, 2, 1):
                    vmax = fx.maxnumf(vmax, vmax.shuffle_xor(shift, t.WAVE))
                vfactor = vmax * fx.Float32(1.0 / t.FP8_MAX)
                vnorm = fx.Float32(rcp_f32(fx.maxnumf(vfactor, fx.Float32(1e-20))))
            elif const_expr(not t.bf16):
                vs = fx.recast_iter(fx.Float32, ctx.value_scale_ptr)
                vfactor = fx.Float32(vs[0]) * fx.Float32(1.0 / t.FP8_MAX)
            next_state = []
            for m in range_constexpr(t.m_tiles):
                if const_expr(t.q_tiles != t.m_tiles):
                    self.mem.stage_query(m)
                    gpu.barrier()
                q_tile = m if t.q_tiles == t.m_tiles else 0
                row = m * 16 + ctx.lane16
                qscale = fx.Float32(1.0)
                if const_expr(resident_q):
                    qscale = q_scale
                elif const_expr(not t.bf16):
                    qscale = fx.Float32(
                        self.mem.load(
                            (q_tile * 16 + ctx.lane16) * t.q_stride + t.head_dim,
                            fx.Float32,
                        )[0]
                    )
                scale = qscale * fx.Float32(t.softmax_scale * LOG2E)
                if const_expr(not t.per_token_kv and not t.bf16):
                    ks = fx.recast_iter(fx.Float32, ctx.key_scale_ptr)
                    scale = scale * fx.Float32(ks[0])
                scores = fx.Vector.filled(8, 0.0, fx.Float32)
                qk_k = 32 if t.bf16 else (128 if t.head_dim % 128 == 0 else 64)
                for d in range_constexpr(t.head_dim // qk_k):
                    if const_expr(t.packed_lds):
                        a = self.mem.packed_operand(
                            kv_off + ctx.warp * 2048, 16, qk_k, ctx.lane16, ctx.half
                        )
                    else:
                        a = self.mem.operand(
                            kv_off
                            + ctx.warp * 16 * t.q_stride
                            + d * qk_k * t.elem_bytes,
                            t.q_stride,
                            qk_k,
                            ctx.lane16,
                            ctx.half,
                        )
                    if const_expr(resident_q):
                        b = q_operand
                    else:
                        b = self.mem.operand(
                            q_tile * 16 * t.q_stride + d * qk_k * t.elem_bytes,
                            t.q_stride,
                            qk_k,
                            ctx.lane16,
                            ctx.half,
                        )
                    scores = Gfx1250Gemm.mma(a, b, scores, qk_k, t.bf16)
                upper = (
                    ctx.planned_context - t.query_length + row // t.query_group_size + 1
                )
                if const_expr(t.per_token_kv):
                    scale_token = ctx.warp * 16 + ctx.half * 8
                    kscales = self.mem.load(
                        kv_off + t.k_bytes + t.v_bytes + scale_token * 4,
                        fx.Float32,
                        8,
                    )
                    vscales = self.mem.load(
                        kv_off + t.k_bytes + t.v_bytes + (t.TOKENS + scale_token) * 4,
                        fx.Float32,
                        8,
                    )
                masked = []
                for r in range_constexpr(8):
                    token = ctx.warp * 16 + ctx.half * 8 + r
                    value = scores[r] * scale
                    if const_expr(t.per_token_kv):
                        value = value * fx.Float32(kscales[r])
                    visible = (token_base + token < upper) & (row < t.rows)
                    if const_expr(t.sliding_window > 0):
                        visible = visible & (
                            token_base + token >= upper - t.sliding_window
                        )
                    masked.append(visible.select(value, fx.Float32(float("-inf"))))
                scores = fx.Vector.from_elements(masked, dtype=fx.Float32)
                local_max = scores.reduce(ReductionOp.MAX)
                local_max = fx.maxnumf(local_max, local_max.shuffle_xor(16, t.WAVE))
                if ctx.half == 0:
                    self.mem.store(
                        t.stats_offset + (ctx.lane16 * t.NWARP + ctx.warp) * 4,
                        local_max,
                        fx.Float32,
                    )
                gpu.barrier()
                tile_max = self.mem.load(
                    t.stats_offset + ctx.lane16 * t.NWARP * 4, fx.Float32, t.NWARP
                ).reduce(ReductionOp.MAX)
                old_max = state[m * state_stride]
                new_max = fx.maxnumf(old_max, tile_max)
                safe_max = (new_max > fx.Float32(float("-inf"))).select(
                    new_max, fx.Float32(0.0)
                )
                # Approximate exponentials avoid the library underflow fixup.
                # Keep infinity semantics for masked rows and empty history.
                corr = fx.Float32(fx.exp2(old_max - safe_max, fastmath="afn"))
                probs = fx.Vector(
                    fx.exp2(
                        scores
                        - fx.Vector.from_elements(
                            [fx.Float32(safe_max)], dtype=fx.Float32
                        ).broadcast_to(8),
                        fastmath="afn",
                    )
                )
                local_sum = probs.reduce(ReductionOp.ADD)
                local_sum = local_sum + local_sum.shuffle_xor(16, t.WAVE)
                pscale = []
                for r in range_constexpr(8):
                    if const_expr(t.per_token_kv):
                        vs = fx.Float32(vscales[r])
                        pscale.append(probs[r] * vs * vnorm)
                    else:
                        pscale.append(
                            probs[r] * fx.Float32(1.0 if t.bf16 else t.FP8_MAX)
                        )
                pvalues = fx.Vector.from_elements(pscale, dtype=fx.Float32)
                p = pvalues.to(fx.BFloat16) if t.bf16 else self.mem.fp8(pvalues)
                self.mem.store(
                    t.p_offset
                    + ctx.lane16 * t.v_stride
                    + (ctx.warp * 16 + ctx.half * 8) * t.elem_bytes,
                    p,
                    fx.BFloat16 if t.bf16 else fx.Int32,
                )
                if ctx.half == 0:
                    self.mem.store(
                        t.stats_offset + 256 + (ctx.lane16 * t.NWARP + ctx.warp) * 4,
                        local_sum,
                        fx.Float32,
                    )
                # Prefetch V operands before waiting for cross-wave P stores.
                pv_k = 32 if t.bf16 else 64
                if const_expr(t.packed_lds):
                    v_operands = [
                        [
                            self.mem.packed_operand(
                                kv_off + t.k_bytes + (vh * 64 + ctx.warp * 16) * 16,
                                128,
                                64,
                                ctx.lane16,
                                ctx.half,
                            )
                        ]
                        for vh in range_constexpr(vh_count)
                    ]
                else:
                    v_operands = [
                        [
                            self.mem.operand(
                                kv_off
                                + t.k_bytes
                                + (vh * 64 + ctx.warp * 16) * t.v_stride
                                + d * pv_k * t.elem_bytes,
                                t.v_stride,
                                pv_k,
                                ctx.lane16,
                                ctx.half,
                            )
                            for d in range_constexpr(t.TOKENS // pv_k)
                        ]
                        for vh in range_constexpr(vh_count)
                    ]
                gpu.barrier()
                denom = state[m * state_stride + 1] * corr + self.mem.load(
                    t.stats_offset + 256 + ctx.lane16 * t.NWARP * 4, fx.Float32, t.NWARP
                ).reduce(ReductionOp.ADD)
                next_state.extend([new_max, denom])
                p_operands = [
                    self.mem.operand(
                        t.p_offset + d * pv_k * t.elem_bytes,
                        t.v_stride,
                        pv_k,
                        ctx.lane16,
                        ctx.half,
                    )
                    for d in range_constexpr(t.TOKENS // pv_k)
                ]
                for vh in range_constexpr(vh_count):
                    out = fx.Vector.filled(8, 0.0, fx.Float32)
                    for d in range_constexpr(t.TOKENS // pv_k):
                        out = Gfx1250Gemm.mma(
                            v_operands[vh][d], p_operands[d], out, pv_k, t.bf16
                        )
                    next_state.append(
                        state[m * state_stride + 2 + vh]
                        * fx.Vector.from_elements(
                            [fx.Float32(corr)], dtype=fx.Float32
                        ).broadcast_to(8)
                        + out
                        * fx.Vector.from_elements(
                            [fx.Float32(vfactor)], dtype=fx.Float32
                        ).broadcast_to(8)
                    )
                # The next iteration's barrier retires the final M-tile reads
                # before a double buffer is reused. Intra-iteration reuse and
                # the single-buffer staging path still need an explicit barrier.
                if const_expr(m + 1 < t.m_tiles or t.buffers == 1):
                    gpu.barrier()
            if const_expr(t.buffers == 1):  # noqa: SIM102
                if token_base + t.TOKENS < end:
                    self.mem.stage_kv(token_base + t.TOKENS, fx.Int32(0))
            final = yield next_state
        pout = fx.recast_iter(t.q_dtype, ctx.pout_ptr)
        pmax = fx.recast_iter(fx.Float32, ctx.pmax_ptr)
        psum = fx.recast_iter(fx.Float32, ctx.psum_ptr)
        slot = fx.Int64(ctx.kv_h) * fx.Int64(gpu.grid_dim.x) + fx.Int64(
            gpu.block_id("x")
        )
        for m in range_constexpr(t.m_tiles):
            row = m * 16 + ctx.lane16
            denom = final[m * state_stride + 1]
            inv = fx.Float32(rcp_f32((denom > 0.0).select(denom, fx.Float32(1.0))))
            if row < t.rows:
                for vh in range_constexpr(vh_count):
                    d = vh * 64 + ctx.warp * 16 + ctx.half * 8
                    value = (
                        final[m * state_stride + 2 + vh]
                        * fx.Vector.from_elements(
                            [fx.Float32(inv)], dtype=fx.Float32
                        ).broadcast_to(8)
                    ).to(t.q_dtype)
                    fx.ptr_store(
                        value,
                        fx.add_offset(pout, (slot * t.rows + row) * t.head_dim + d),
                    )
                if ctx.warp == 0 and ctx.half == 0:
                    pmax[slot * t.rows + row] = final[m * state_stride] * fx.Float32(
                        1.0 / LOG2E
                    )
                    psum[slot * t.rows + row] = denom


def compile_gfx1250_pa_decode(
    *,
    head_dim,
    query_group_size,
    block_size,
    softmax_scale,
    query_dtype,
    per_token_kv,
    query_length,
    trans_v,
    sliding_window,
    compute_dtype="fp8",
):
    t = Gfx1250Traits(
        head_dim,
        query_group_size,
        block_size,
        head_dim**-0.5 if softmax_scale is None else softmax_scale,
        query_dtype,
        per_token_kv,
        query_length,
        trans_v,
        sliding_window,
        compute_dtype,
    )
    if compute_dtype not in ("fp8", "bf16"):
        raise ValueError(f"Unsupported gfx1250 PA compute dtype: {compute_dtype}")
    if t.bf16 and per_token_kv:
        raise ValueError("BF16 PA decode does not use quantization scales")
    if t.total_bytes > 327680:
        raise NotImplementedError(
            "gfx1250 PA decode query/KV tiles exceed LDS capacity"
        )
    if t in _CACHE:
        return _CACHE[t]
    tag = ("gfx1250", tuple(t.__dict__.values()), implementation_cache_tag())

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, t.total_bytes // 4, 16]

    # Reuse the argument context and packed task contract of the CDNA backend.
    from .context import PaDecodeContext

    @flyc.kernel(known_block_size=(128, 1, 1))
    def kernel(
        pmax: fx.Pointer,
        psum: fx.Pointer,
        pout: fx.Pointer,
        query: fx.Pointer,
        key: fx.Pointer,
        value: fx.Pointer,
        table: fx.Pointer,
        ks: fx.Pointer,
        vs: fx.Pointer,
        max_blocks: fx.Int32,
        sk_block: fx.Int32,
        sk_head: fx.Int32,
        sq_row: fx.Int32,
        sq_head: fx.Int32,
        work: fx.Pointer,
        sequences: fx.Int32,
    ):
        _ = tag
        task = fx.ptr_load(
            fx.add_offset(fx.recast_iter(fx.Int32, work), gpu.block_id("x") * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
        if fx.Int32(task[1]) < fx.Int32(task[2]):
            ctx = PaDecodeContext(
                t,
                SharedStorage,
                pmax,
                psum,
                pout,
                query,
                key,
                value,
                table,
                ks,
                vs,
                max_blocks,
                sk_block,
                sk_head,
                sq_row,
                sq_head,
                sequences,
                fx.Int32(task[0]),
                fx.Int32(task[1]),
                fx.Int32(task[2]),
                fx.Int32(task[3]),
            )
            Gfx1250Pipeline(ctx).run()

    @flyc.jit
    def launch(
        pmax: fx.Pointer,
        psum: fx.Pointer,
        pout: fx.Pointer,
        query: fx.Pointer,
        key: fx.Pointer,
        value: fx.Pointer,
        table: fx.Pointer,
        ks: fx.Pointer,
        vs: fx.Pointer,
        max_blocks: fx.Int32,
        sequences: fx.Int32,
        kv_heads: fx.Int32,
        sk_block: fx.Int32,
        sk_head: fx.Int32,
        sq_row: fx.Int32,
        sq_head: fx.Int32,
        work: fx.Pointer,
        capacity: fx.Int32,
        stream: fx.Stream,
    ):
        _ = tag
        kernel(
            pmax,
            psum,
            pout,
            query,
            key,
            value,
            table,
            ks,
            vs,
            max_blocks,
            sk_block,
            sk_head,
            sq_row,
            sq_head,
            work,
            sequences,
        ).launch(grid=(capacity, kv_heads, 1), block=(128, 1, 1), stream=stream)

    compiled = {"launch": launch, "kernel": kernel}
    return _CACHE.setdefault(t, compiled)
