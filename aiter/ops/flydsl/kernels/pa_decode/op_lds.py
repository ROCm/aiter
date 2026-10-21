# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""LDS views, query staging and paged K/V/scale loads."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.protocol import dsl_size_of
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from .. import dpp_utils
from ..kernels_common import create_llvm_ptr
from ..tensor_shim import _to_raw, buf_base_i64, ptr_buf_tensor
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
    """Load Q early and stage native BF16 or normalized FP8 MFMA operands."""

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
        inv_b = fx.full(self.traits.QLOAD_UNIT, inv, fx.Float32)

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
    def stage_row(self, m, q_row_off, q_units):
        if const_expr(self.traits.is_bf16_kv):
            for u in range_constexpr(self.traits.N_QLOADS):
                elem_off = (
                    self.ctx.qh_local * self.traits.head_dim
                    + self.ctx.lane16 * self.traits.QCHUNK
                    + u * self.traits.QLOAD_UNIT
                )
                self.lds.store(q_row_off + elem_off * 2, fx.BFloat16, q_units[u])
        else:
            self.quantize_row(m, q_row_off, q_units)

    @flyc.jit
    def stage(self, q_units_prefetched):
        for m in range_constexpr(self.traits.M_TILES):
            flat_idx = m * MFMA_MNK + self.ctx.qh_local
            qi = flat_idx // self.traits.query_group_size
            gs_head = flat_idx - qi * self.traits.query_group_size
            q_row_off = m * MFMA_MNK * self.traits.head_dim * self.traits.QP_ELEM_BYTES
            # Only the final M-tile can need a runtime row guard.
            if const_expr((m + 1) * MFMA_MNK <= self.traits.CTA_ROWS):
                q_units = (
                    q_units_prefetched[m]
                    if const_expr(self.traits.MTP4_FUSED)
                    else self.load_row(qi, gs_head)
                )
                self.stage_row(m, q_row_off, q_units)
            elif flat_idx < self.traits.CTA_ROWS:
                self.stage_row(m, q_row_off, self.load_row(qi, gs_head))
            else:
                if const_expr(self.traits.is_bf16_kv):
                    for u in range_constexpr(self.traits.N_QLOADS):
                        elem_off = (
                            self.ctx.qh_local * self.traits.head_dim
                            + self.ctx.lane16 * self.traits.QCHUNK
                            + u * self.traits.QLOAD_UNIT
                        )
                        self.lds.store(
                            q_row_off + elem_off * 2,
                            fx.BFloat16,
                            fx.full(self.traits.QLOAD_UNIT, 0.0, fx.BFloat16),
                        )
                else:
                    self.lds.store_words(
                        q_row_off
                        + self.ctx.qh_local * self.traits.head_dim
                        + self.ctx.lane16 * self.traits.QCHUNK,
                        fx.full(self.traits.QCHUNK // 4, 0, fx.Int32),
                    )
                    if self.ctx.lane16 == 0:
                        self.lds.store_scalar(
                            self.traits.sQscale_off,
                            self.ctx.qh_local * self.traits.M_TILES + m,
                            self.ctx.ZERO_F,
                        )

    def load_operands(self):
        # Match the head-dim permutation of PaDecodeKVLoader.load_k_chunk.
        q_ops_all = []
        for m in range_constexpr(self.traits.M_TILES):
            q_row_off = m * MFMA_MNK * self.traits.head_dim * self.traits.QP_ELEM_BYTES
            for qkhe in range_constexpr(self.traits.QKHE_LOOP):
                he_idx = qkhe * self.traits.RGROUP_QUARTERS + self.ctx.rgroup
                chunk = self.lds.load(
                    q_row_off
                    + (
                        self.ctx.lane16 * self.traits.head_dim
                        + he_idx * self.traits.QK_CHUNK_ELEMS
                    )
                    * self.traits.QP_ELEM_BYTES,
                    fx.Int64,
                    4 if self.traits.is_bf16_kv else 2,
                )
                for qkr in range_constexpr(4 if self.traits.is_bf16_kv else 2):
                    q_ops_all.append(chunk[qkr])
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

    def _make_nt_k_loader(self, tensor_ptr):
        # K on the pipe is a flat global_load_dwordx4 with !nontemporal.
        # V stays on the cached copy.
        base = buf_base_i64(tensor_ptr)
        ptr_ty = fx.PointerType.get(
            fx.Int32.ir_type,
            address_space=fx.AddressSpace.Global,
            alignment=16,
        )

        def _load(elem_idx):
            # FP8 elements are one byte, so the element index is the byte offset.
            # inttoptr, not add_offset: the GEP form reschedules the K loads and
            # splits their vmcnt waits (q4 loop: one wait becomes seven).
            ptr = fx.Pointer(fx.inttoptr(ptr_ty, base + fx.Int64(elem_idx)))
            return fx.generic_load(ptr, dtype=fx.Int32, count=4, nontemporal=True)

        return _load

    def init_loaders(self):
        if const_expr(self.traits.D256_PIPE):
            self._k_load_vector = self._make_nt_k_loader(self.ctx.key_cache_ptr)
        else:
            self._k_load_vector = self._make_raw_flat_loader(
                self.ctx.key_cache_ptr,
                self.traits.KV_DTYPE,
                self.traits.KV_VECTOR_WIDTH,
                self.traits.KV_EXTENT,
            )
        self._v_load_vector = self._make_raw_flat_loader(
            self.ctx.value_cache_ptr,
            self.traits.KV_DTYPE,
            self.traits.KV_VECTOR_WIDTH,
            self.traits.KV_EXTENT,
        )

    def kv_address(self, phys, page_elems, rest):
        if const_expr(self.traits.BUFFER_KV):
            return fx.Uint32(phys) * fx.Uint32(page_elems) + fx.Uint32(rest)
        # Widen before the page product reaches 2^31 FP8 elements.
        if const_expr(self.traits.wide_kv_addressing):
            return fx.Int64(phys) * fx.Int64(page_elems) + fx.Int64(rest)
        return phys * page_elems + rest

    def load_k_vector(self, elem_off):
        return self._k_load_vector(elem_off).bitcast(fx.Int64)

    def load_v_vector(self, elem_off):
        return self._v_load_vector(elem_off).bitcast(fx.Int64)

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
        self.bt_scalar = (
            self.bt_buf
            if const_expr(self.traits.PAGES_PER_CHUNK == 1)
            else ptr_buf_tensor(
                self.ctx.block_tables_ptr,
                fx.Int32,
                num_records_bytes=bt_num_records_bytes,
            )
        )
        if const_expr(not self.traits.per_token_kv and not self.traits.is_bf16_kv):
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

    def _table_page(self, page):
        element_offset = self.ctx.seq * self.ctx.max_blocks_per_seq + page
        result = self.bt_scalar[element_offset]
        return (page < self.ctx.num_pages).select(fx.Int32(result), fx.Int32(0))

    @flyc.jit
    def _staged_page(self, page):
        """Page id from LDS, or the block table when the index is outside it.

        A task can own more pages than ``max_context_length`` staged. The
        table load stays in the other branch so an in-window page does not
        wait on it.
        """
        rel = page - self.ctx.part_start * (
            self.traits.TILE_TOK // self.traits.block_size
        )
        page_id = fx.Int32(0)
        if (rel >= 0) & (rel < self.traits.stage_page_capacity):
            page_id = fx.Int32(
                self.lds.load(self.traits.sPages_off + rel * 4, fx.Int32, 1)[0]
            )
        else:
            page_id = self._table_page(page)
        return page_id

    def load_pages(self, page, vec_width=1):
        # Even an in-bounds table entry can contain a stale page ID beyond
        # this context. Pin those pages to block 0 before loading K/V.
        # Staged ids were pinned the same way when they were stored.
        if const_expr(self.traits.stage_page_capacity):
            if const_expr(vec_width == 1):
                return self._staged_page(page)
            return fx.Vector.from_elements(
                [self._staged_page(page + i) for i in range_constexpr(vec_width)],
                dtype=fx.Int32,
            )
        if const_expr(vec_width == 1):
            return self._table_page(page)
        element_offset = self.ctx.seq * self.ctx.max_blocks_per_seq + page
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
            fx.Vector.from_elements([fetched], dtype=fx.Int32)
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
            if const_expr(self.traits.is_bf16_kv):
                for qkr in range_constexpr(2):
                    he8_idx = he_idx * 2 + qkr
                    base = self.kv_address(
                        phys,
                        self.ctx.n_kv * self.traits.head_dim * self.traits.block_size,
                        (
                            (self.ctx.kv_h * (self.traits.head_dim // 8) + he8_idx)
                            * self.traits.block_size
                            + within_page_tok
                        )
                        * 8,
                    )
                    w = self.load_k_vector(base)
                    ops.extend([w[0], w[1]])
            else:
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
                w = self.load_k_vector(base)
                ops.extend([w[0], w[1]])
            if const_expr(self.traits.block_size == 16):
                # Overlap page16 gathers.
                fx.rocdl.sched_barrier(fx.rocdl.mask_vmem_rd)
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
            fx.Vector.from_elements([fetched], dtype=fx.Int32)
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
                ) // self.traits.KV_VECTOR_WIDTH + step
                if const_expr(self.traits.trans_v):
                    base = self.kv_address(
                        phys_row[sub],
                        self.ctx.n_kv
                        * (
                            self.traits.STEPS_PER_PAGE
                            * self.traits.head_dim
                            * self.traits.KV_VECTOR_WIDTH
                        ),
                        (
                            (self.ctx.kv_h * self.traits.STEPS_PER_PAGE + page_step)
                            * self.traits.head_dim
                            + head_element
                        )
                        * self.traits.KV_VECTOR_WIDTH,
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
    def stage_page_ids(self):
        """Stage this task's page ids in LDS so the tile loop never reads the table."""
        traits = self.traits
        ppt = traits.TILE_TOK // traits.block_size
        first_page = self.ctx.part_start * ppt
        n_part_pages = (self.ctx.part_end - self.ctx.part_start) * ppt
        rounds = traits.stage_page_capacity // traits.BLOCK_THREADS
        for r in range_constexpr(rounds):
            idx = self.ctx.tid + r * traits.BLOCK_THREADS
            if idx < n_part_pages:
                pg = first_page + idx
                raw = fx.Int32(
                    self.bt_scalar[self.ctx.seq * self.ctx.max_blocks_per_seq + pg]
                )
                val = (pg < self.ctx.num_pages).select(raw, fx.Int32(0))
                self.lds.store(
                    traits.sPages_off + idx * 4,
                    fx.Int32,
                    fx.Vector.from_elements([val], dtype=fx.Int32),
                )

    @flyc.jit
    def prefetch_first(self):
        # Empty partitions must not read K/V or block_tables.
        k_pf0 = fx.full(self.traits.NCHUNK * self.traits.N_SUBCHUNKS, 0, fx.Int64)
        if self.ctx.part_start < self.ctx.part_end:
            if const_expr(self.traits.D256_PIPE and self.traits.per_token_kv):
                # Put scales ahead of K on vmcnt, so waiting for their LDS
                # writes does not also drain the first K prefetch.
                phys_vec0 = self.fetch_v_pages(self.ctx.part_start)
                self.stage_scales(
                    phys_vec0, self.scale_buffer_offset(fx.Int32(self.ctx.part_start))
                )
                k_pf0 = self.load_k_from_pages(phys_vec0)
            else:
                k_pf0, phys_vec0 = self.load_k(self.ctx.part_start)
                if const_expr(self.traits.REUSE_KV_PAGES):
                    # Reuse K page IDs for V's LDS broadcast.
                    self.stage_v_pages(phys_vec0)
                else:
                    self.fetch_v_pages(self.ctx.part_start)
                if const_expr(self.traits.per_token_kv):
                    self.stage_scales(
                        phys_vec0,
                        self.scale_buffer_offset(fx.Int32(self.ctx.part_start)),
                    )
        elif self.ctx.lane == 0:
            self.lds.store(
                self.traits.sVPage_off
                + self.ctx.warp * (self.traits.PAGES_PER_CHUNK * 4),
                fx.Int32,
                fx.full(self.traits.PAGES_PER_CHUNK, 0, fx.Int32),
            )
        return k_pf0


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
        absolute = token_base + (ctx.warp * 16 if t.block_size == 16 else 0)
        safe = (
            absolute
            if full_tile
            else (absolute < ctx.planned_context).select(absolute, fx.Int32(0))
        )
        page = fx.Int32(
            table[
                ctx.planned_seq * ctx.max_blocks_per_seq
                + fx.Uint32(safe) // t.block_size
            ]
        )
        off = (fx.Int64(page) * ctx.n_kv + ctx.kv_h) * t.head_dim * t.block_size
        pt = fx.Uint32(safe) % t.block_size
        wave_token = pt + (ctx.warp * 16 if t.block_size != 16 else 0)
        valid = fx.max(
            fx.Int32(0),
            fx.min(fx.Int32(16), ctx.planned_context - token_base - ctx.warp * 16),
        )
        self.tdm_copy(
            ctx.key_cache_ptr,
            off + wave_token * 16,
            kv_off + ctx.warp * t.head_dim * 16,
            (t.head_dim // 16, 256) if full_tile else (t.head_dim // 16, 16, 16),
            (t.block_size * 16, 1) if full_tile else (t.block_size * 16, 16, 1),
            (None, None) if full_tile else (None, valid, None),
            16,
            16,
        )
        self.tdm_copy(
            ctx.value_cache_ptr,
            off + (wave_token // 16) * t.head_dim * 16,
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

    def bf16_k_operand(self, offset, token, dim, lane16, half):
        # Retain vector-8 cache packing: each half-wave reads alternate chunks.
        words = []
        for j in range_constexpr(2):
            frag = self.load(
                offset + ((dim // 8 + half + 2 * j) * 128 + token + lane16) * 16,
                fx.Int32,
                4,
            )
            words.extend([frag[i] for i in range_constexpr(4)])
        return fx.Vector.from_elements(words, dtype=fx.Int32).bitcast(fx.BFloat16)

    def bf16_v_operand(self, offset, dim, token, lane16, half):
        words = []
        for j in range_constexpr(2):
            frag = self.load(
                offset
                + (token // 8 + half + 2 * j) * self.t.head_dim * 16
                + (dim + lane16) * 16,
                fx.Int32,
                4,
            )
            words.extend([frag[i] for i in range_constexpr(4)])
        return fx.Vector.from_elements(words, dtype=fx.Int32).bitcast(fx.BFloat16)

    @flyc.jit
    def stage_bf16_page(self, token_base, buffer):
        """Gather a BF16 page without transposition; zero-fill its unwritten tail."""
        ctx, t = self.ctx, self.t
        safe = (token_base < ctx.planned_context).select(token_base, fx.Int32(0))
        table = fx.recast_iter(fx.Int32, ctx.block_tables_ptr)
        page = fx.Int32(table[ctx.planned_seq * ctx.max_blocks_per_seq + safe // 128])
        off = (fx.Int64(page) * ctx.n_kv + ctx.kv_h) * t.head_dim * 128
        kv_off = t.q_bytes + buffer * t.kv_stride
        valid = fx.max(
            fx.Int32(0), fx.min(fx.Int32(128), ctx.planned_context - token_base)
        )
        self.tdm_copy(
            ctx.key_cache_ptr,
            off + ctx.warp * (t.head_dim // 32) * 1024,
            kv_off + ctx.warp * (t.head_dim // 32) * 2048,
            (t.head_dim // 32, 128, 8),
            (1024, 8, 1),
            (None, valid, None),
            8,
            8,
        )
        if token_base + 128 <= ctx.planned_context:
            self.tdm_copy(
                ctx.value_cache_ptr,
                off + ctx.warp * t.head_dim * 32,
                kv_off + t.k_bytes + ctx.warp * t.head_dim * 64,
                (32 * t.head_dim,),
                (1,),
                (None,),
                8,
                8,
            )
        else:
            # Token extents apply inside each eight-token V chunk.
            for chunk in range_constexpr(4):
                count = fx.max(
                    fx.Int32(0), fx.min(fx.Int32(8), valid - ctx.warp * 32 - chunk * 8)
                )
                self.tdm_copy(
                    ctx.value_cache_ptr,
                    off + (ctx.warp * 4 + chunk) * t.head_dim * 8,
                    kv_off + t.k_bytes + (ctx.warp * 4 + chunk) * t.head_dim * 16,
                    (t.head_dim, 8),
                    (8, 1),
                    (None, count),
                    8,
                    8,
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
