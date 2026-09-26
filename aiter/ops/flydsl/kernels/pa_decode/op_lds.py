# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""LDS views, query quantization and paged K/V/scale loads."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.protocol import dsl_size_of
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp

from .. import dpp_utils
from ..tensor_shim import ptr_buf_tensor
from ..utils import rcp_f32
from .traits import MFMA_MNK


class PaDecodeLds:
    """Typed views over the Q/P alias and the remaining shared scratch."""

    def __init__(self, traits, storage_type):
        self.traits = traits
        self.storage_type = storage_type

    def allocate(self):
        # Keep an ir.Value pointer for use inside scf control flow;
        # the Python SharedStorage handle cannot cross that boundary.
        lds = fx.SharedAllocator().allocate(self.storage_type).peek()
        self.base = fx.recast_iter(fx.Uint8, lds.buf.ptr)  # byte-addressed base

    def pointer(self, byte_off, elem_ty):
        # Restore element alignment lost by the byte-base recast.
        p = fx.add_offset(self.base, fx.make_int_tuple(byte_off))
        ptr_ty = fx.PointerType.get(
            elem_ty.ir_type, fx.AddressSpace.Shared, dsl_size_of(elem_ty)
        )
        return fx.recast_iter(ptr_ty, p)

    def load(self, byte_off, elem_ty, n):
        return fx.ptr_load(
            self.pointer(byte_off, elem_ty), result_type=fx.Vector.make_type(n, elem_ty)
        )

    def store(self, byte_off, elem_ty, vec):
        fx.ptr_store(vec, self.pointer(byte_off, elem_ty))

    def row_offset(self, byte_off, m_idx, width, elem_ty):
        return byte_off + m_idx * (width * dsl_size_of(elem_ty))

    def load_scalar(self, byte_off, m_idx):
        return self.load(
            self.row_offset(byte_off, m_idx, 1, fx.Float32), fx.Float32, 1
        )[0]

    def store_scalar(self, byte_off, m_idx, val):
        self.store(
            self.row_offset(byte_off, m_idx, 1, fx.Float32),
            fx.Float32,
            fx.Vector.from_elements([val], dtype=fx.Float32),
        )

    def store_wave(self, base_off, row, w, val):
        off = base_off + (row * self.traits.NWARP_PAD + w) * 4
        self.store(off, fx.Float32, fx.Vector.from_elements([val], dtype=fx.Float32))

    def load_wave(self, base_off, row):
        off = base_off + row * (self.traits.NWARP_PAD * 4)
        return self.load(off, fx.Float32, self.traits.NWARP)

    def store_words(self, byte_off, words):
        self.store(byte_off, fx.Int32, words)


class PaDecodeQueryLoader:
    """Load Q early, normalize to FP8 and retain its MFMA operands."""

    def __init__(self, ctx, lds, gemm):
        self.ctx = ctx
        self.traits = ctx.traits
        self.lds = lds
        self.gemm = gemm

    def init_loaders(self):
        _q_copy_op = (
            fx.rocdl.BufferCopy128b()
            if self.traits.QLOAD_UNIT == 8
            else fx.rocdl.BufferCopy64b()
        )
        self.q_buf = ptr_buf_tensor(self.ctx.query_ptr, self.traits.Q_DTYPE)
        self.q_tiled = fx.logical_divide(self.q_buf, fx.make_layout(1, 1))
        self.q_copy_atom = fx.make_copy_atom(_q_copy_op, self.traits.Q_DTYPE)
        self.q_reg = fx.make_rmem_tensor(
            fx.make_layout(self.traits.QLOAD_UNIT, 1), self.traits.Q_DTYPE
        )

    def load_chunk(self, elem_idx):
        fx.copy(self.q_copy_atom, fx.slice(self.q_tiled, (None, elem_idx)), self.q_reg)
        return fx.Vector(fx.memref_load_vec(self.q_reg))

    def load_row(self, qi, gs_head):
        qh0 = self.ctx.kv_h * self.traits.query_group_size + gs_head
        row_byte0 = (
            (self.ctx.seq * self.traits.query_length + self.ctx.query_begin + qi)
            * self.ctx.stride_q_row
            + qh0 * self.ctx.stride_q_head
        ) * 2  # 16-bit float = 2B/elem
        base_elem = (
            row_byte0 + self.ctx.lane16 * (self.traits.QCHUNK * 2)
        ) // 2  # byte offset -> element index
        return [
            self.load_chunk(base_elem + u * self.traits.QLOAD_UNIT)
            for u in range_constexpr(self.traits.N_QLOADS)
        ]

    def prefetch(self):
        # Issue independent Q loads ahead of K/page/scale prefetch.
        q_units_prefetched = None
        if const_expr(self.traits.MTP4_FUSED):
            q_units_prefetched = []
            for m in range_constexpr(self.traits.M_TILES):
                flat_idx = m * MFMA_MNK + self.ctx.qh_local
                qi = flat_idx // self.traits.query_group_size
                gs_head = flat_idx - qi * self.traits.query_group_size
                q_units_prefetched.append(self.load_row(qi, gs_head))
        return q_units_prefetched

    def local_absmax(self, q_unit):
        if const_expr(self.traits.Q_ABSMAX_F32):
            return fmath.absf(q_unit.to(fx.Float32)).reduce(ReductionOp.MAX)
        else:
            return fmath.absf(q_unit).reduce(ReductionOp.MAX).to(fx.Float32)

    @flyc.jit
    def quantize_row(self, m, q_row_off, q_units):
        """Each M-tile owns disjoint rows; only scale publication needs a lane guard."""
        if const_expr(self.traits.SCALAR_FP8_DECODE):
            for u in range_constexpr(self.traits.N_QLOADS):
                self.lds.store_words(
                    q_row_off
                    + self.ctx.qh_local * self.traits.head_dim
                    + self.ctx.lane16 * self.traits.QCHUNK
                    + u * self.traits.QLOAD_UNIT,
                    self.gemm.fp8_words(q_units[u].to(fx.Float32)),
                )
        else:
            absmax = self.local_absmax(q_units[0])
            for u in range_constexpr(1, self.traits.N_QLOADS):
                absmax = fx.maxnumf(
                    absmax,
                    self.local_absmax(q_units[u]),
                )
            for sh in (8, 4, 2, 1):
                absmax = fx.maxnumf(absmax, dpp_utils.dpp_xor_f32(absmax, sh))

            q_scale = absmax * fx.Float32(1.0 / self.traits.FP8_MAX)
            inv = fx.Float32(rcp_f32(fx.maxnumf(q_scale, fx.Float32(1e-20))))
            inv_b = fx.Vector.from_elements([inv], dtype=fx.Float32).broadcast_to(
                self.traits.QLOAD_UNIT
            )

            for u in range_constexpr(self.traits.N_QLOADS):
                q_scaled_unit = q_units[u].to(fx.Float32) * inv_b
                self.lds.store_words(
                    q_row_off
                    + self.ctx.qh_local * self.traits.head_dim
                    + self.ctx.lane16 * self.traits.QCHUNK
                    + u * self.traits.QLOAD_UNIT,
                    self.gemm.fp8_words(q_scaled_unit),
                )
            if self.ctx.lane16 == 0:
                # Transposed [qh][m] enables one vector read across M-tiles.
                self.lds.store_scalar(
                    self.traits.sQscale_off,
                    self.ctx.qh_local * self.traits.M_TILES + m,
                    q_scale,
                )

    @flyc.jit
    def stage(self, q_units_prefetched):
        for m in range_constexpr(self.traits.M_TILES):
            flat_idx = m * MFMA_MNK + self.ctx.qh_local
            qi = flat_idx // self.traits.query_group_size
            gs_head = flat_idx - qi * self.traits.query_group_size
            q_row_off = m * MFMA_MNK * self.traits.head_dim
            # Only the final M-tile can need a runtime row guard.
            if const_expr((m + 1) * MFMA_MNK <= self.traits.CTA_ROWS):
                q_units = (
                    q_units_prefetched[m]
                    if const_expr(self.traits.MTP4_FUSED)
                    else self.load_row(qi, gs_head)
                )
                self.quantize_row(m, q_row_off, q_units)
            elif flat_idx < self.traits.CTA_ROWS:
                self.quantize_row(m, q_row_off, self.load_row(qi, gs_head))
            else:
                self.lds.store_words(
                    q_row_off
                    + self.ctx.qh_local * self.traits.head_dim
                    + self.ctx.lane16 * self.traits.QCHUNK,
                    fx.Vector.filled(self.traits.QCHUNK // 4, 0, fx.Int32),
                )
                if (
                    const_expr(not self.traits.SCALAR_FP8_DECODE)
                    and self.ctx.lane16 == 0
                ):
                    self.lds.store_scalar(
                        self.traits.sQscale_off,
                        self.ctx.qh_local * self.traits.M_TILES + m,
                        self.ctx.ZERO_F,
                    )

    def load_operands(self):
        # Match the head-dim permutation of PaDecodeKVLoader.load_k_chunk.
        q_ops_all = []
        for m in range_constexpr(self.traits.M_TILES):
            q_row_off = m * MFMA_MNK * self.traits.head_dim
            for qkhe in range_constexpr(self.traits.QKHE_LOOP):
                he_idx = qkhe * self.traits.RGROUP_QUARTERS + self.ctx.rgroup
                chunk = self.lds.load(
                    q_row_off
                    + self.ctx.lane16 * self.traits.head_dim
                    + he_idx * self.traits.QK_CHUNK_ELEMS,
                    fx.Int64,
                    2,
                )
                q_ops_all.extend([chunk[0], chunk[1]])
        return q_ops_all


class PaDecodeKVLoader:
    """Bounded page-table reads, wide cache addressing and scale publication."""

    def __init__(self, ctx, lds):
        self.ctx = ctx
        self.traits = ctx.traits
        self.lds = lds
        self.key_scale = None
        self.value_scale = None

    def _make_raw_flat_loader(self, tensor_ptr, elem_ty, reg_width, extent):
        # Stream dense KV; retain cache locality for sliding windows.
        copy_op = (
            fx.rocdl.BufferCopy128b(
                cache_modifier=2 if const_expr(self.traits.sliding_window == 0) else 0
            )
            if const_expr(self.traits.BUFFER_KV)
            else fx.UniversalCopy128b()
        )
        copy_atom = fx.make_copy_atom(copy_op, elem_ty)
        reg = fx.make_rmem_tensor(fx.make_layout(reg_width, 1), elem_ty)
        flat = (
            ptr_buf_tensor(tensor_ptr, elem_ty)
            if const_expr(self.traits.BUFFER_KV)
            else fx.Tensor(
                fx.make_view(
                    fx.recast_iter(elem_ty, tensor_ptr), fx.make_layout(extent, 1)
                )
            )
        )
        tiled = fx.logical_divide(flat, fx.make_layout(1, 1))

        def _load(elem_idx):
            fx.copy(copy_atom, fx.slice(tiled, (None, elem_idx)), reg)
            return fx.Vector(fx.memref_load_vec(reg))

        return _load

    def init_loaders(self):
        self._k_load_fp8x16 = self._make_raw_flat_loader(
            self.ctx.key_cache_ptr, self.traits.FP8, 16, self.traits.KV_EXTENT
        )
        self._v_load_fp8x16 = self._make_raw_flat_loader(
            self.ctx.value_cache_ptr, self.traits.FP8, 16, self.traits.KV_EXTENT
        )

    def kv_address(self, phys, page_elems, rest):
        if const_expr(self.traits.BUFFER_KV):
            return fx.Uint32(phys) * fx.Uint32(page_elems) + fx.Uint32(rest)
        # Widen before the page product reaches 2^31 FP8 elements.
        if const_expr(self.traits.wide_kv_addressing):
            return fx.Int64(phys) * fx.Int64(page_elems) + fx.Int64(rest)
        return phys * page_elems + rest

    def load_k_vector(self, byte_off):
        return self._k_load_fp8x16(byte_off).bitcast(fx.Int64)

    def load_v_vector(self, byte_off):
        return self._v_load_fp8x16(byte_off).bitcast(fx.Int64)

    def init_page_table(self):
        # A partial compute tile may read past block_tables; bounded loads
        # return page 0 for those masked tail tokens instead of faulting.
        bt_num_records_bytes = (
            fx.Int64(self.ctx.num_sequences) * fx.Int64(self.ctx.max_blocks_per_seq) * 4
        )
        # Wide loads must preserve row starts that are only int32-aligned.
        self.bt_buf = ptr_buf_tensor(
            self.ctx.block_tables_ptr,
            fx.Int32,
            unit_elems=self.traits.PAGES_PER_CHUNK,
            unit_stride=1,
            num_records_bytes=bt_num_records_bytes,
        )
        if const_expr(not self.traits.per_token_kv):
            key_scale_buf = ptr_buf_tensor(self.ctx.key_scale_ptr, fx.Float32)
            value_scale_buf = ptr_buf_tensor(self.ctx.value_scale_ptr, fx.Float32)
            self.key_scale = fx.Float32(key_scale_buf[0])
            self.value_scale = fx.Float32(value_scale_buf[0])

    def init_scale_loaders(self):
        if const_expr(self.traits.per_token_kv):
            scale_load_width = (
                self.traits.NCHUNK
                if self.traits.block_size >= 64 and not self.traits.UNIQUE_SCALE_STAGING
                else 1
            )
            scale_copy_op = (
                fx.rocdl.BufferCopy32b()
                if scale_load_width == 1
                else fx.rocdl.BufferCopy128b()
            )
            self.scale_copy_atom = fx.make_copy_atom(scale_copy_op, fx.Float32)
            k_scale_buf = ptr_buf_tensor(self.ctx.key_scale_ptr, fx.Float32)
            v_scale_buf = ptr_buf_tensor(self.ctx.value_scale_ptr, fx.Float32)
            self.k_scale_tiled = fx.logical_divide(k_scale_buf, fx.make_layout(1, 1))
            self.v_scale_tiled = fx.logical_divide(v_scale_buf, fx.make_layout(1, 1))
            self.k_scale_reg = fx.make_rmem_tensor(
                fx.make_layout(scale_load_width, 1), fx.Float32
            )
            self.v_scale_reg = fx.make_rmem_tensor(
                fx.make_layout(scale_load_width, 1), fx.Float32
            )

    def load_k_scale(self, elem_idx):
        fx.copy(
            self.scale_copy_atom,
            fx.slice(self.k_scale_tiled, (None, elem_idx)),
            self.k_scale_reg,
        )
        return fx.Vector(fx.memref_load_vec(self.k_scale_reg))

    def load_v_scale(self, elem_idx):
        fx.copy(
            self.scale_copy_atom,
            fx.slice(self.v_scale_tiled, (None, elem_idx)),
            self.v_scale_reg,
        )
        return fx.Vector(fx.memref_load_vec(self.v_scale_reg))

    def load_pages(self, page, vec_width=1):
        # Even an in-bounds table entry can contain a stale page ID beyond
        # this context. Pin those pages to block 0 before loading K/V.
        element_offset = self.ctx.seq * self.ctx.max_blocks_per_seq + page
        if const_expr(vec_width == 1):
            result = self.bt_buf[element_offset]
            return (page < self.ctx.num_pages).select(fx.Int32(result), fx.Int32(0))
        bt_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        frag = fx.make_fragment_like(fx.slice(self.bt_buf, (0, None)))
        fx.copy(
            bt_copy_atom,
            fx.slice(self.bt_buf, (element_offset, None)),
            frag,
        )
        loaded = fx.Vector(fx.memref_load_vec(frag))
        return fx.Vector.from_elements(
            [
                (page + i < self.ctx.num_pages).select(fx.Int32(loaded[i]), fx.Int32(0))
                for i in range_constexpr(vec_width)
            ],
            dtype=fx.Int32,
        )

    @flyc.jit
    def stage_v_pages(self, phys_vec):
        if self.ctx.lane == 0:
            self.lds.store(
                self.traits.sVPage_off
                + self.ctx.warp * (self.traits.PAGES_PER_CHUNK * 4),
                fx.Int32,
                phys_vec,
            )

    def fetch_v_pages(self, tt_i32):
        base_page = (
            tt_i32 * self.traits.TILE_TOK // self.traits.block_size
        )  # tile start is page-aligned
        fetched = self.load_pages(
            base_page
            + (self.ctx.warp * self.traits.TOK_PER_WARP) // self.traits.block_size,
            self.traits.PAGES_PER_CHUNK,
        )
        fetched_vec = (
            fx.Vector.from_elements([fx.Int32(fetched)], dtype=fx.Int32)
            if const_expr(self.traits.PAGES_PER_CHUNK == 1)
            else fx.Vector(fetched)
        )
        self.stage_v_pages(fetched_vec)
        return fetched_vec

    def read_v_pages(self):
        off = self.traits.sVPage_off + self.ctx.rgroup * (
            self.traits.PAGES_PER_CHUNK * 4
        )
        return self.lds.load(off, fx.Int32, self.traits.PAGES_PER_CHUNK)

    def read_k_pages(self):
        off = self.traits.sVPage_off + self.ctx.warp * (self.traits.PAGES_PER_CHUNK * 4)
        return self.lds.load(off, fx.Int32, self.traits.PAGES_PER_CHUNK)

    def scale_buffer_offset(self, tt_val):
        if const_expr(self.traits.per_token_kv and not self.traits.single_tile_plan):
            return (tt_val & fx.Int32(1)) * self.traits.KV_BUF_STRIDE
        return 0

    def stage_scales(self, phys_vec, buf_off=0):
        if const_expr(self.traits.block_size >= 64):
            phys = fx.Int32(phys_vec[0])
            chunk_tok = (
                self.ctx.lane
                if const_expr(self.traits.UNIQUE_SCALE_STAGING)
                else self.ctx.lane16 * self.traits.NCHUNK
            )
            page_tok = (
                self.ctx.warp * self.traits.TOK_PER_WARP
            ) % self.traits.block_size + chunk_tok
            scale_idx = (
                phys * self.ctx.stride_ks_block
                + self.ctx.kv_h * self.ctx.stride_ks_head
                + page_tok
            )
            k_scale_vec = self.load_k_scale(scale_idx)
            v_scale_vec = self.load_v_scale(scale_idx)
            slot = (
                self.ctx.warp * self.traits.TOK_PER_WARP + chunk_tok
            ) * self.traits.f32
            self.lds.store(
                self.traits.sKScale_off + buf_off + slot, fx.Float32, k_scale_vec
            )
            self.lds.store(
                self.traits.sVScale_off + buf_off + slot, fx.Float32, v_scale_vec
            )
        else:
            # Each rgroup stages its own page16 sub-block.
            phys = fx.Int32(fx.Vector(phys_vec)[self.ctx.rgroup])
            scale_idx = (
                phys * self.ctx.stride_ks_block
                + self.ctx.kv_h * self.ctx.stride_ks_head
                + self.ctx.lane16
            )
            k_scale_scalar = fx.Float32(self.load_k_scale(scale_idx)[0])
            v_scale_scalar = fx.Float32(self.load_v_scale(scale_idx)[0])
            fx.rocdl.sched_barrier(fx.rocdl.mask_vmem_rd)
            slot = (
                self.ctx.warp * self.traits.TOK_PER_WARP
                + self.ctx.rgroup * MFMA_MNK
                + self.ctx.lane16
            ) * self.traits.f32
            self.lds.store(
                self.traits.sKScale_off + buf_off + slot,
                fx.Float32,
                fx.Vector.from_elements([k_scale_scalar], dtype=fx.Float32),
            )
            self.lds.store(
                self.traits.sVScale_off + buf_off + slot,
                fx.Float32,
                fx.Vector.from_elements([v_scale_scalar], dtype=fx.Float32),
            )

    def load_scale(self, base_off, a, buf_off=0):
        slot = (
            self.ctx.warp * self.traits.TOK_PER_WARP
            + a * MFMA_MNK
            + self.ctx.rgroup * 4
        ) * self.traits.f32
        return self.lds.load(base_off + buf_off + slot, fx.Float32, 4)

    def load_kv_scales(self, a, buf_off=0):
        return self.load_scale(self.traits.sKScale_off, a, buf_off), self.load_scale(
            self.traits.sVScale_off, a, buf_off
        )

    def load_k_chunk(self, phys, a):
        # K token = warp*TOK_PER_WARP + a*MFMA_MNK + lane16, matching the
        # softmax mask and probability-word scatter.
        within_page_tok = (
            self.ctx.warp * self.traits.TOK_PER_WARP + a * MFMA_MNK + self.ctx.lane16
        ) % self.traits.block_size
        ops = []
        for qkhe in range_constexpr(self.traits.QKHE_LOOP):
            he_idx = qkhe * self.traits.RGROUP_QUARTERS + self.ctx.rgroup
            base = self.kv_address(
                phys,
                self.ctx.n_kv
                * (
                    self.traits.QCHUNK
                    * self.traits.block_size
                    * self.traits.QK_CHUNK_ELEMS
                ),
                (
                    (self.ctx.kv_h * self.traits.QCHUNK + he_idx)
                    * self.traits.block_size
                    + within_page_tok
                )
                * self.traits.QK_CHUNK_ELEMS,
            )
            w = self.load_k_vector(
                base
            )  # head[he_idx*16 : +16] -> two K32 operand packs
            if const_expr(self.traits.block_size == 16):
                # Overlap page16 gathers.
                fx.rocdl.sched_barrier(fx.rocdl.mask_vmem_rd)
            ops.extend([w[0], w[1]])
        return ops  # N_SUBCHUNKS i64 operands

    def load_k_from_pages(self, phys_vec):
        flat = []
        for a in range_constexpr(self.traits.NCHUNK):
            phys = fx.Int32(phys_vec[(a * MFMA_MNK) // self.traits.block_size])
            flat.extend(self.load_k_chunk(phys, a))
        if const_expr(self.traits.head_dim == 64):
            fx.rocdl.sched_vmem(len(flat) // 2)

        return fx.Vector.from_elements(flat, dtype=fx.Int64)

    def load_k(self, tt_i32):
        base_page = (
            tt_i32 * self.traits.TILE_TOK // self.traits.block_size
        )  # tile start is page-aligned
        fetched = self.load_pages(
            base_page
            + (self.ctx.warp * self.traits.TOK_PER_WARP) // self.traits.block_size,
            self.traits.PAGES_PER_CHUNK,
        )
        phys_vec = (
            fx.Vector.from_elements([fx.Int32(fetched)], dtype=fx.Int32)
            if const_expr(self.traits.PAGES_PER_CHUNK == 1)
            else fx.Vector(fetched)
        )
        return self.load_k_from_pages(phys_vec), phys_vec

    def load_v(self, phys_row, vh):
        head_group = ((vh * self.traits.VHE_SIZE) // 16) + self.ctx.warp
        head_element = head_group * 16 + self.ctx.lane16
        ops = []
        for sub in range_constexpr(self.traits.PAGES_PER_CHUNK):
            for step in range_constexpr(self.traits.STEPS_PER_CHUNK):
                # PV's token chunk is owned by rgroup after the LDS transpose.
                page_step = (
                    (self.ctx.rgroup * self.traits.TOK_PER_WARP)
                    % self.traits.block_size
                ) // 16 + step
                if const_expr(self.traits.trans_v):
                    base = self.kv_address(
                        phys_row[sub],
                        self.ctx.n_kv
                        * (self.traits.STEPS_PER_PAGE * self.traits.head_dim * 16),
                        (
                            (self.ctx.kv_h * self.traits.STEPS_PER_PAGE + page_step)
                            * self.traits.head_dim
                            + head_element
                        )
                        * 16,
                    )
                else:
                    base = self.kv_address(
                        phys_row[sub],
                        self.ctx.n_kv * (self.traits.head_dim * self.traits.block_size),
                        (self.ctx.kv_h * self.traits.head_dim + head_element)
                        * self.traits.block_size
                        + page_step * 16,
                    )
                w = self.load_v_vector(base)
                if const_expr(self.traits.block_size == 16):
                    fx.rocdl.sched_barrier(fx.rocdl.mask_vmem_rd)
                ops.extend([w[0], w[1]])
        if const_expr(self.traits.head_dim == 64):
            fx.rocdl.sched_vmem(len(ops) // 2)
        return ops  # NVOPS i64, the 64-token contiguous run for this head

    @flyc.jit
    def prefetch_first(self):
        # Empty partitions must not read K/V or block_tables.
        k_pf0 = fx.Vector.filled(
            self.traits.NCHUNK * self.traits.N_SUBCHUNKS, 0, fx.Int64
        )
        if self.ctx.part_start < self.ctx.part_end:
            k_pf0, phys_vec0 = self.load_k(self.ctx.part_start)
            if const_expr(self.traits.REUSE_KV_PAGES):
                # Reuse K page IDs for V's LDS broadcast.
                self.stage_v_pages(phys_vec0)
            else:
                self.fetch_v_pages(self.ctx.part_start)
            if const_expr(self.traits.per_token_kv):
                self.stage_scales(
                    phys_vec0, self.scale_buffer_offset(fx.Int32(self.ctx.part_start))
                )
        elif self.ctx.lane == 0:
            self.lds.store(
                self.traits.sVPage_off
                + self.ctx.warp * (self.traits.PAGES_PER_CHUNK * 4),
                fx.Int32,
                fx.Vector.filled(self.traits.PAGES_PER_CHUNK, 0, fx.Int32),
            )
        return k_pf0
