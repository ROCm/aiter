# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from ..kernels_common import LOG2E as _LOG2E
from .common import (
    _buffer_load_128,
    _buffer_load_lds_128,
    _buffer_store_128,
    _cu_load,
)
from .common import load as _load

PAGED_FP8_BLOCK_M = 256
PAGED_FP8_BUFFER_LIMIT_BYTES = (1 << 31) - 16


def _tree_reduce(vals, binop):
    items = list(vals)
    while len(items) > 1:
        nxt = [binop(items[i], items[i + 1]) for i in range(0, len(items) - 1, 2)]
        if len(items) % 2 == 1:
            nxt.append(items[-1])
        items = nxt
    return items[0]


def _bitcast_i32(value):
    return fx.Float32(value).bitcast(fx.Int32).ir_value()


def _bitcast_f32(value):
    return fx.Int32(value).bitcast(fx.Float32).ir_value()


def _reduction_pair(v_f32):
    v_i32 = _bitcast_i32(v_f32)
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_ty, v_i32, v_i32, False, True)
    lhs_i32 = llvm.extractvalue(T.i32, swapped, [0])
    rhs_i32 = llvm.extractvalue(T.i32, swapped, [1])
    return _bitcast_f32(lhs_i32), _bitcast_f32(rhs_i32)


def _score_pair_to_lists(v_s):
    s_lo, s_hi = v_s
    return (
        [Vec(s_lo)[r] for r in range_constexpr(16)],
        [Vec(s_hi)[r] for r in range_constexpr(16)],
    )


def _score_lists_to_vecs(v_s_lists):
    s_lo, s_hi = v_s_lists
    return (
        Vec.from_elements([as_mlir_value(v) for v in s_lo], fx.Float32).ir_value(),
        Vec.from_elements([as_mlir_value(v) for v in s_hi], fx.Float32).ir_value(),
    )


def _lane_pair_reduce(v, reducer, fm_fast):
    lhs, rhs = _reduction_pair(v)
    return reducer(lhs, rhs, fm_fast)


def _score_pair_sum(v_s, zero_f, fm_fast):
    s_lo, s_hi = _score_lists_to_vecs(v_s)
    tile = Vec(s_lo) + Vec(s_hi)
    reducer = lambda a, b, _fm: a + b
    return _lane_pair_reduce(
        tile.reduce("add", init_val=zero_f, fastmath=fm_fast), reducer, fm_fast
    )


def _scale_sub_score_pair(v_s, row_max_raw, scale, zero_f, bias=None):
    """Fuse softmax scaling, row-max subtraction and FP8 probability headroom.

    Returns ``scale * (v_s - row_max_raw) + bias`` per element via a single FMA
    (``fma(s, scale, bias - scale*row_max_raw)``), so the fp8 QK MMA can emit raw
    (un-scaled) logits and reduce_max can run in the raw domain (scale > 0 is
    order-preserving). Replaces the separate post-QK scale multiply + subtract.
    ``-inf`` masked lanes stay ``-inf`` (scale > 0), matching the un-fused path.

    ``bias`` lands in the FMA's addend, so a caller needing ``exp2`` to produce
    ``2**bias * P`` pays nothing -- see ``DualwaveFp8SoftmaxHelper.sub_m``.
    """
    s_lo, s_hi = v_s
    neg_scaled_max = zero_f - scale * row_max_raw
    if bias is not None:
        # exp2 lands on 2**bias * P instead of P, at no extra instruction: the
        # FMA's addend absorbs it.
        neg_scaled_max = neg_scaled_max + bias

    # Explicit nearest-even rounding retains unpacked FP32 FMA in gfx950 codegen.
    def _center(values):
        centered = []
        for i in range_constexpr(16):
            centered.append(
                fx.fma(
                    Vec(values)[i],
                    scale,
                    neg_scaled_max,
                    fastmath=fx.arith.FastMathFlags.none,
                    roundingmode=fx.RoundingMode.to_nearest_even,
                )
            )
        return Vec.from_elements(centered, fx.Float32)

    lo, hi = _center(s_lo), _center(s_hi)
    return as_mlir_value(lo), as_mlir_value(hi)


def _exp2_score_slice(v_s, start):
    if const_expr(start == 0):
        s_lo = [Vec(v_s[0])[r] for r in range_constexpr(16)]
        lo_partial = []
        for r in range_constexpr(16):
            lo_partial.append(rocdl.exp2(T.f32, as_mlir_value(s_lo[r])))
        return Vec.from_elements(lo_partial, fx.Float32).ir_value(), v_s[1]

    lo_partial = [Vec(v_s[0])[r] for r in range_constexpr(16)]
    hi_full = []
    for r in range_constexpr(16):
        hi_full.append(rocdl.exp2(T.f32, as_mlir_value(Vec(v_s[1])[r])))
    return lo_partial, hi_full


def _safe_l_inv(l_row, zero_f):
    l_inv = rocdl.rcp(T.f32, as_mlir_value(l_row))
    return (fx.Float32(l_row) > zero_f).select(l_inv, zero_f)


def _scale_o_accs(v_o, scale_scalar, traits):
    scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
    for dc in range_constexpr(traits.D_CHUNKS):
        v_o[dc] = Vec(v_o[dc]) * scale_vec


def _vec_k_dma_oct_idx(traits, d, wave_id_uni, lane_in_warp):
    """Flat octet index for this wave/lane's d-th DMA slot in vectorized K layout."""
    return (
        wave_id_uni * (traits.WARP_SIZE * traits.SMEM_D_RPT)
        + d * traits.WARP_SIZE
        + lane_in_warp
    )


def _sigma_k_tile_n(ni):
    """Sigma permutation applied to K tile-n during vectorized DMA (bit-shuffle)."""
    return (ni & 3) | ((ni & 8) >> 1) | ((ni & 4) << 1) | (ni & ~15)


def _init_dualwave_thread_mapping(ctx):
    """Map packed requests and GQA heads onto paged-attention workgroups."""
    traits = ctx.traits
    batch_interleave_group = traits.BATCH_INTERLEAVE_GROUP
    if const_expr(batch_interleave_group > 1):
        # Keep a bounded batch group in grid X so the final causal q-block
        # occupies more CUs without abandoning K/V locality across all batches.
        linear_head_batch = fx.Index(gpu.block_idx.x)
        if const_expr(traits.PAIRED_PAGE_IDS):
            ctx.h_idx = linear_head_batch // batch_interleave_group
            ctx.batch_idx = (
                fx.Int64(gpu.block_idx.z) * batch_interleave_group
                + linear_head_batch % batch_interleave_group
            )
        else:
            ctx.h_idx = linear_head_batch % traits.NUM_HEADS_Q
            batch_in_group = linear_head_batch // traits.NUM_HEADS_Q
            ctx.batch_idx = (
                fx.Int64(gpu.block_idx.z) * batch_interleave_group + batch_in_group
            )
        ctx.q_block_idx = fx.Index(gpu.block_idx.y)
    else:
        ctx.h_idx = fx.Index(gpu.block_idx.x)
        ctx.q_block_idx = fx.Index(gpu.block_idx.y)
    if const_expr(batch_interleave_group <= 1):
        ctx.batch_idx = fx.Int64(gpu.block_idx.z)
    ctx.tid = fx.Index(gpu.thread_idx.x)

    ctx.wave_id = ctx.tid // traits.WARP_SIZE
    ctx.lane = ctx.tid % traits.WARP_SIZE
    ctx.lane_mod_32 = ctx.lane % 32
    ctx.lane_div_32 = ctx.lane // 32

    _tid_i32 = fx.Int32(ctx.tid)
    _wave_id_uni_i32 = rocdl.readfirstlane(
        T.i32,
        (_tid_i32 // fx.Int32(traits.WARP_SIZE)).ir_value(),
    )
    ctx.wave_id_uni = fx.Index(_wave_id_uni_i32)

    ctx.wave_q_offset = ctx.wave_id * traits.ROWS_PER_WAVE
    ctx.q_start = ctx.q_block_idx * traits.BLOCK_M

    ctx.h_kv_idx = ctx.h_idx % traits.NUM_HEADS_KV
    ctx.group_id = ctx.h_idx // traits.NUM_HEADS_KV
    ctx.q_head_idx = ctx.h_kv_idx * traits.GQA_GROUP_SIZE + ctx.group_id
    ctx.kv_head_idx = ctx.h_kv_idx


def _init_dualwave_q_row(ctx):
    """Set q_row / q_row_i32 / q_start_pos_i32 on a dualwave-style context."""
    traits = ctx.traits
    ctx.q_row_in_block = ctx.wave_q_offset + ctx.lane_mod_32
    ctx.q_start_pos_i32 = fx.Int32(ctx.q_start + ctx.wave_id_uni * traits.ROWS_PER_WAVE)
    ctx.q_row = ctx.q_start + ctx.q_row_in_block
    ctx.q_row_i32 = fx.Int32(ctx.q_row)


@dataclass(frozen=True)
class PagedDualwaveSwpFp8Traits:
    """Layouts for native paged FP8 QK/PV with register-resident Q."""

    HEAD_DIM: int
    HEAD_DIM_V: int
    D_CHUNKS: int
    NUM_HEADS_Q: int
    NUM_HEADS_KV: int
    GQA_GROUP_SIZE: int
    WAVES_PER_EU: int
    DAZ: bool
    DUALWAVE_SWP_LAZY_RESCALE: bool
    FP8_PV_SEGMENTED: bool
    FP8_V_H1: int
    FP8_V_H2: int
    DEFAULT_STRIDE_Q_N: int
    SMEM_D_RPT: int
    NUM_PREFETCH_K: int
    DUALWAVE_SWP_KV_PER_BUFFER: int
    LDS_KV_TOTAL_SIZE: int
    DUALWAVE_SWP_K_BUF_BASE: tuple[int, ...]
    VT_BF16_TOTAL: int
    FP8_V_ROW_STRIDE: int
    DUALWAVE_SWP_RESCALE_THRESHOLD: float
    PAIRED_PAGE_IDS: bool = True
    BATCH_INTERLEAVE_GROUP: int = 1
    BLOCK_M: int = PAGED_FP8_BLOCK_M
    BLOCK_N: int = 64
    WARP_SIZE: int = 64
    NUM_WAVES: int = 8
    BLOCK_SIZE: int = 512
    ROWS_PER_WAVE: int = 32
    D_CHUNK: int = 32
    PV_K_STEPS: int = 2
    PAGE_SIZE: int = 64
    KV_CACHE_LAYOUT: str = "vectorized"
    CACHE_BUFFERED: bool = False
    CSR_PAGE_TABLE: bool = False
    HAS_LAST_PAGE_LENS: bool = False
    GUARD_OUTPUT_ROWS: bool = True
    DMA_BYTES: int = 16
    ELEM_BYTES: int = 1
    OUT_ELEM_BYTES: int = 2
    KV_VEC_SIZE: int = 16
    SCHED_MFMA_MASK: int = 0x008
    SCHED_DS_READ_MASK: int = 0x100

    @property
    def cache_tag(self):
        return (
            "paged_fp8_native",
            self.NUM_HEADS_Q,
            self.NUM_HEADS_KV,
            self.HEAD_DIM,
            self.HEAD_DIM_V,
            self.WAVES_PER_EU,
            self.DAZ,
            self.DUALWAVE_SWP_LAZY_RESCALE,
            self.DUALWAVE_SWP_RESCALE_THRESHOLD,
            True,  # Packed varlen; retain the existing cache-tag prefix.
            self.NUM_PREFETCH_K,
            self.FP8_V_ROW_STRIDE,
            self.BATCH_INTERLEAVE_GROUP,
            self.PAIRED_PAGE_IDS,
            self.PAGE_SIZE,
            self.KV_CACHE_LAYOUT,
            self.CACHE_BUFFERED,
            self.CSR_PAGE_TABLE,
            self.HAS_LAST_PAGE_LENS,
            self.GUARD_OUTPUT_ROWS,
        )

    @property
    def K_LDS_PAGE_GROUPED(self):
        return self.PAGE_SIZE == 16 and self.HEAD_DIM_V == 192

    @property
    def V_LOAD_LAYOUT(self):
        """Producer word order; the matching store and byte mask must agree."""
        if self.PAGE_SIZE == 1:
            return (
                "token_transpose"
                if (self.HEAD_DIM, self.HEAD_DIM_V) == (192, 128)
                else "token_words"
            )
        if (
            self.PAGE_SIZE == 16
            and self.HEAD_DIM == 192
            and (self.HEAD_DIM_V == 192 or not self.CACHE_BUFFERED)
        ):
            return "page_waves"
        return "lane_groups"

    @property
    def SHARED_PAGE_ID_LAYOUT(self):
        if self.PAGE_SIZE == 1 and self.HEAD_DIM_V == 192:
            return "tokens"
        if self.PAGE_SIZE == 16 and self.CACHE_BUFFERED:
            return self.V_LOAD_LAYOUT
        return None


def _make_paged_dualwave_swp_fp8_traits(
    num_heads,
    num_kv_heads,
    head_dim,
    value_head_dim,
    rescale_threshold,
    waves_per_eu=2,
    daz=True,
    dualwave_swp_lazy_rescale=True,
    batch_interleave_group=1,
    paired_page_ids=True,
    page_size=64,
    kv_cache_layout="vectorized",
    cache_buffered=False,
    metadata_mode="block_table",
    has_last_page_lens=False,
    guard_output_rows=True,
):
    """Build layouts after the dedicated builder validates the paged contract."""
    block_n = 64
    smem_d_rpt = (head_dim + 127) // 128
    # Retain padded K-slot spacing; V192 below compacts the unused D256 tail.
    smem_k_tile_elems = 8 * smem_d_rpt * 1040
    compact_v192 = value_head_dim == 192
    if compact_v192:
        # Six K/V slots fit in 160 KiB only when K omits its unused D256 tail.
        smem_k_tile_elems = block_n * head_dim
    num_prefetch_k = 8 if head_dim == 128 else 6
    slot_elems = smem_k_tile_elems
    # The builder accepts V128/V192: one 128-channel prefix and an optional V64 tail.
    fp8_v_h1, fp8_v_h2 = 128, value_head_dim - 128
    fp8_v_row_stride = block_n if compact_v192 else block_n + 16
    fp8_v_tile_bytes = value_head_dim * fp8_v_row_stride

    return PagedDualwaveSwpFp8Traits(
        HEAD_DIM=head_dim,
        HEAD_DIM_V=value_head_dim,
        D_CHUNKS=value_head_dim // 32,
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=num_heads // num_kv_heads,
        WAVES_PER_EU=waves_per_eu,
        DAZ=bool(daz),
        DUALWAVE_SWP_LAZY_RESCALE=bool(dualwave_swp_lazy_rescale),
        DUALWAVE_SWP_RESCALE_THRESHOLD=rescale_threshold,
        FP8_PV_SEGMENTED=fp8_v_h2 > 0,
        FP8_V_H1=fp8_v_h1,
        FP8_V_H2=fp8_v_h2,
        PAIRED_PAGE_IDS=bool(paired_page_ids),
        PAGE_SIZE=page_size,
        KV_CACHE_LAYOUT=kv_cache_layout,
        CACHE_BUFFERED=bool(cache_buffered),
        CSR_PAGE_TABLE=metadata_mode == "csr",
        HAS_LAST_PAGE_LENS=bool(has_last_page_lens),
        GUARD_OUTPUT_ROWS=bool(guard_output_rows),
        DEFAULT_STRIDE_Q_N=num_heads * head_dim,
        SMEM_D_RPT=smem_d_rpt,
        NUM_PREFETCH_K=num_prefetch_k,
        DUALWAVE_SWP_KV_PER_BUFFER=slot_elems,
        LDS_KV_TOTAL_SIZE=num_prefetch_k * slot_elems,
        DUALWAVE_SWP_K_BUF_BASE=tuple(i * slot_elems for i in range(num_prefetch_k)),
        VT_BF16_TOTAL=num_prefetch_k * (fp8_v_tile_bytes // 2) + 128,
        FP8_V_ROW_STRIDE=fp8_v_row_stride,
        BATCH_INTERLEAVE_GROUP=int(batch_interleave_group),
    )


class DualwaveFp8KernelContext:
    """Shared per-kernel state for the gfx950 dualwave fp8 attention helpers.

    Q/K/V remain byte-typed for DMA. Descales apply to accumulated QK scores
    and the final P*V output; the ``vt`` allocation holds packed FP8 V bytes.
    """

    def __init__(
        self,
        traits_or_ctx,
        Q=None,
        K=None,
        V=None,
        O=None,
        CuSeqQ=None,
        KvMetadata=None,
        LastPageLens=None,
        QDescale=None,
        KDescale=None,
        VDescale=None,
        stride_q_n=None,
        softmax_scale=None,
        stride_o_n=None,
        BlockTable=None,
        block_table_stride=None,
    ):
        if isinstance(traits_or_ctx, DualwaveFp8KernelContext):
            self.__dict__.update(traits_or_ctx.__dict__)
            self.ctx_ref = getattr(traits_or_ctx, "ctx_ref", traits_or_ctx)
            return
        self.ctx_ref = self
        self.traits = traits_or_ctx
        self.Q = Q
        self.K = K
        self.V = V
        self.O = O
        self.CuSeqQ = CuSeqQ
        self.KvMetadata = KvMetadata
        self.LastPageLens = LastPageLens
        self.QDescale = QDescale
        self.KDescale = KDescale
        self.VDescale = VDescale
        self.stride_q_n = stride_q_n
        self.stride_o_n = stride_q_n if stride_o_n is None else stride_o_n
        self.softmax_scale = softmax_scale
        self.BlockTable = BlockTable
        self.block_table_stride = block_table_stride

    def init_types_and_constants(self):
        traits = self.traits
        self.fm_fast = fx.arith.FastMathFlags.fast
        self.v4i32_type = Vec.make_type(4, fx.Int32)
        self.NUM_DMA_K = traits.SMEM_D_RPT
        self.c_neg_inf = fx.Float32(float("-inf"))
        self.c_neg_floor = fx.Float32(-3.0e38)
        self.c_zero_f = fx.Float32(0.0)
        self.c_rescale_thr_f = fx.Float32(traits.DUALWAVE_SWP_RESCALE_THRESHOLD)
        self.c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)

    def init_runtime_indices(self):
        self.stride_q_n_v = fx.Index(self.stride_q_n)
        self.stride_o_n_v = fx.Int64(self.stride_o_n)

    def init_lds(self, shared_storage):
        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.kv.ptr))
        self.lds_vt_base_idx = fx.Index(fx.ptrtoint(lds.vt.ptr))
        # The 16-byte tile stride preserves wide LDS loads; i64-element
        # views can instead lower to slower paired 64-bit reads.
        self.k_lds_i32_tiles = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, lds.kv.ptr),
                fx.make_layout(self.traits.LDS_KV_TOTAL_SIZE // 4, 1),
            ),
            fx.make_layout(4, 1),
        )
        self.v_lds_i32_tiles = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, lds.vt.ptr),
                fx.make_layout(self.traits.VT_BF16_TOTAL // 2, 1),
            ),
            fx.make_layout(4, 1),
        )

    def init_thread_mapping(self):
        _init_dualwave_thread_mapping(self)

    def init_dma_thread_offsets(self):
        self.lane_in_warp = self.tid % self.traits.WARP_SIZE

    def init_sequence_lengths(self):
        traits = self.traits
        _cuq_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(self.CuSeqQ), fx.make_layout(1, 1)
        )
        _cuk_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(self.KvMetadata), fx.make_layout(1, 1)
        )
        _cu_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        _cu_v1i32 = Vec.make_type(1, fx.Int32)
        self.q_tok_base = _cu_load(_cuq_div, self.batch_idx, _cu_atom, _cu_v1i32)
        self.q_tok_end = _cu_load(_cuq_div, self.batch_idx + 1, _cu_atom, _cu_v1i32)
        self.kv_tok_base = fx.Index(0)
        if const_expr(traits.CSR_PAGE_TABLE):
            self.page_base = fx.Int64(
                _cu_load(_cuk_div, self.batch_idx, _cu_atom, _cu_v1i32)
            )
            page_end = fx.Int64(
                _cu_load(_cuk_div, self.batch_idx + 1, _cu_atom, _cu_v1i32)
            )
            # CSR prefixes and per-request token lengths obey the int32
            # metadata ABI. Keep their proven width before widening any
            # physical address; otherwise LLVM carries 64-bit loop bounds.
            self.request_page_count = fx.Int64(
                fx.Int32(page_end) - fx.Int32(self.page_base)
            )
            if const_expr(traits.HAS_LAST_PAGE_LENS):
                last_div = fx.logical_divide(
                    fx.rocdl.make_buffer_tensor(self.LastPageLens),
                    fx.make_layout(1, 1),
                )
                last = fx.Int64(_cu_load(last_div, self.batch_idx, _cu_atom, _cu_v1i32))
                length = (self.request_page_count - 1) * traits.PAGE_SIZE + last
                self.kv_tok_end = fx.Index(
                    fx.Int32((self.request_page_count > 0).select(length, fx.Int64(0)))
                )
            else:
                self.kv_tok_end = fx.Index(
                    fx.Int32(self.request_page_count * traits.PAGE_SIZE)
                )
        else:
            self.kv_tok_end = _cu_load(_cuk_div, self.batch_idx, _cu_atom, _cu_v1i32)
        self.seqlen_q_v = self.q_tok_end - self.q_tok_base
        self.seqlen_kv_v = self.kv_tok_end - self.kv_tok_base
        self.seqlen_kv_i32 = fx.Int32(self.seqlen_kv_v)
        self.delta_i32 = fx.Int32(self.seqlen_kv_i32 - fx.Int32(self.seqlen_q_v))

    def init_varlen_causal_lpt_order(self):
        """Reverse active query blocks without making padded workgroups valid."""
        num_q_blocks = (
            self.seqlen_q_v + self.traits.BLOCK_M - 1
        ) // self.traits.BLOCK_M
        active_q_block = self.q_block_idx < num_q_blocks
        reversed_q_block = num_q_blocks - 1 - self.q_block_idx
        self.q_block_idx = active_q_block.select(reversed_q_block, self.q_block_idx)
        self.q_start = self.q_block_idx * self.traits.BLOCK_M

    def init_descriptors(self):
        traits = self.traits
        eb = traits.ELEM_BYTES
        q_nrec_bytes = as_mlir_value(self.q_tok_end * self.stride_q_n_v * eb)
        o_nrec_bytes = as_mlir_value(
            self.q_tok_end * self.stride_o_n_v * traits.OUT_ELEM_BYTES
        )

        def _make_buf_div(tensor, nrec_bytes):
            # fp8 Q/K/V buffer views are i8-typed so DMA and register loads share one
            # byte view.
            bt = fx.rocdl.make_buffer_tensor(tensor, num_records_bytes=nrec_bytes)
            it = fx.get_iter(bt)
            i8_ptr_ty = fx.PointerType.get(
                elem_ty=fx.Int8.ir_type,
                address_space=fx.PointerType(it.type).address_space,
                alignment=fx.PointerType(it.type).alignment,
            )
            bt = fx.Tensor(
                fx.make_view(fx.recast_iter(i8_ptr_ty, it), fx.get_layout(bt))
            )
            return fx.logical_divide(bt, fx.make_layout(1, 1))

        self.q_div = _make_buf_div(self.Q, q_nrec_bytes)
        self.o_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(self.O, num_records_bytes=o_nrec_bytes),
            fx.make_layout(1, 1),
        )
        # K/V are one-byte OCP FP8 pages.  Rebase one descriptor per tile
        # with 64-bit pointer arithmetic so serving-sized pools beyond the
        # 4-GiB boundary retain the exact physical page address.
        self.k_base_iter = fx.recast_iter(fx.Int8, fx.get_iter(self.K))
        self.v_base_iter = fx.recast_iter(fx.Int8, fx.get_iter(self.V))
        self.k_page_bytes = traits.PAGE_SIZE * traits.NUM_HEADS_KV * traits.HEAD_DIM
        self.v_page_bytes = traits.PAGE_SIZE * traits.NUM_HEADS_KV * traits.HEAD_DIM_V
        self.init_page_table()
        self.k_div = None
        self.v_div = None
        if const_expr(traits.CACHE_BUFFERED):

            def _cache_div(tensor, base_iter):
                num_bytes = fx.Int64(fx.get_scalar(fx.cosize(fx.get_layout(tensor))))
                flat = fx.make_view(base_iter, fx.make_layout(num_bytes, 1))
                # make_buffer_tensor defaults to an unbounded resource;
                # supply the exact byte count for zero-filling OOB copies.
                bounded = fx.rocdl.make_buffer_tensor(
                    flat, num_records_bytes=num_bytes.ir_value()
                )
                return fx.logical_divide(bounded, fx.make_layout(1, 1))

            self.k_div = _cache_div(self.K, self.k_base_iter)
            self.v_div = _cache_div(self.V, self.v_base_iter)

    def init_atoms_and_lds_ptrs(self):
        traits = self.traits
        self.load_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.store_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.o_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        # fp8 global->LDS DMA uses i8 destination typing; K/V LDS reads are byte-addressed.
        self.lds_ptr_ty = fx.PointerType.get(fx.Int8.ir_type, 2, traits.DMA_BYTES)

    def _load_scale_scalar(self, tensor):
        _div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(tensor), fx.make_layout(1, 1)
        )
        _atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        _v = fly.copy_atom_call_ssa(
            [Vec.make_type(1, fx.Float32)], _atom, fx.slice(_div, (None, fx.Int32(0)))
        )
        return Vec(_v, (1,), fx.Float32)[0]

    def init_descale(self):
        c_sm_scale_log2e = self.softmax_scale * fx.Float32(_LOG2E)
        _qd = self._load_scale_scalar(self.QDescale)
        _kd = self._load_scale_scalar(self.KDescale)
        # fp8 feeds raw Q/K into the MFMA, so q/k descale * softmax scale multiplies
        # the fp32 logits after QK.
        self.c_logit_scale = c_sm_scale_log2e * (_qd * _kd)

    def init_tile_bounds(self):
        traits = self.traits
        kv_tile_size = traits.BLOCK_N
        num_kv_tiles = (self.seqlen_kv_v + kv_tile_size - 1) // kv_tile_size
        self.num_kv_tiles = num_kv_tiles
        # The padded Q block can extend past INT32_MAX even when both
        # sequence lengths fit int32. Clamp before narrowing the bound.
        causal_end_raw = (
            fx.Int64(self.q_start) + traits.BLOCK_M + fx.Int64(self.delta_i32)
        )
        causal_end_i32 = fx.Int32(
            fx.min(fx.max(causal_end_raw, fx.Int64(0)), fx.Int64((1 << 31) - 1))
        )
        causal_num_tiles = (fx.Index(causal_end_i32) + kv_tile_size - 1) // kv_tile_size
        max_num_tiles = fx.Index(
            (causal_num_tiles < num_kv_tiles).select(causal_num_tiles, num_kv_tiles)
        )
        # Pipeline needs an EVEN tile count >= 4; extra tiles read 0 (num_records) and are masked.
        max_num_tiles = ((max_num_tiles + fx.Index(1)) // fx.Index(2)) * fx.Index(2)
        max_num_tiles = fx.Index(
            (max_num_tiles < fx.Index(4)).select(fx.Index(4), max_num_tiles)
        )
        self.max_num_tiles = max_num_tiles
        split_t0 = 0
        split_t_end = max_num_tiles
        active = self.q_start < self.seqlen_q_v
        in_mask = causal_end_raw > fx.Int64(0)
        active = active & in_mask
        split_t_end = fx.Index(active.select(split_t_end, split_t0))

        self.split_t0 = split_t0
        self.split_t_end = split_t_end

    def init_q_row(self):
        _init_dualwave_q_row(self)

    def k_buf_base(self, buf_id):
        traits = self.traits
        if const_expr(isinstance(buf_id, int)):
            return traits.DUALWAVE_SWP_K_BUF_BASE[buf_id]
        return buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER

    def buffer_load_128(self, elem_index):
        return _buffer_load_128(
            elem_index, self.load_atom_128, self.q_div, self.v4i32_type
        )

    def buffer_load_fp8x16(self, src_div, elem_index):
        return fly.copy_atom_call_ssa(
            [self.v4i32_type],
            self.load_atom_128,
            fx.slice(src_div, (None, fx.Int32(elem_index))),
        )

    def global_load_fp8x16(self, base_iter, byte_offset, valid, *, is_value=False):
        """Load a native FP8 vector through a bounded cache or a 64-bit pointer."""
        if const_expr(self.traits.CACHE_BUFFERED):
            offset = valid.select(
                fx.Int32(byte_offset), fx.Int32(PAGED_FP8_BUFFER_LIMIT_BYTES)
            )
            return self.buffer_load_fp8x16(
                self.v_div if is_value else self.k_div, offset
            )
        # Do not issue an out-of-allocation load before selecting zero padding.
        safe_offset = valid.select(fx.Int64(byte_offset), fx.Int64(0))
        loaded = _load(fx.add_offset(base_iter, safe_offset), dtype=fx.Int32, count=4)
        return valid.select(loaded, fx.Vector.filled(4, 0, fx.Int32)).ir_value()

    def buffer_load_lds_128(self, src_div, lds_byte_addr, src_elem, soffset_elems):
        _buffer_load_lds_128(
            src_div,
            lds_byte_addr,
            src_elem,
            soffset_elems,
            _dma_atom=self.dma_atom,
            _lds_ptr_ty=self.lds_ptr_ty,
        )

    def buffer_store_128(self, pack_i32_vec, elem_index):
        _buffer_store_128(
            pack_i32_vec,
            elem_index,
            self.o_store_reg_128,
            self.store_atom_128,
            self.o_div,
        )

    def global_idx_q(self, token_idx, col):
        return (
            (self.q_tok_base + token_idx) * self.stride_q_n_v
            + self.q_head_idx * self.traits.HEAD_DIM
            + col
        )

    def global_idx_o(self, token_idx, col):
        return (
            (self.q_tok_base + token_idx) * self.stride_o_n_v
            + self.q_head_idx * self.traits.HEAD_DIM_V
            + col
        )

    def init_page_table(self):
        self.page_i32_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        self.page_v1i32 = Vec.make_type(1, fx.Int32)
        self.page_i32x2_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
        self.page_v2i32 = Vec.make_type(2, fx.Int32)
        if const_expr(self.traits.CSR_PAGE_TABLE):
            # A request-local resource both enforces the CSR boundary and
            # removes the row-base add from each lane's hot-loop lookup.
            request = fx.make_view(
                fx.add_offset(fx.get_iter(self.BlockTable), self.page_base),
                fx.make_layout(self.request_page_count, 1),
            )
            resource = fx.rocdl.make_buffer_tensor(
                request, num_records_bytes=(self.request_page_count * 4).ir_value()
            )
            self.page_indices_div = fx.logical_divide(resource, fx.make_layout(1, 1))
            self.page_base = fx.Int64(0)
        else:
            self.page_indices_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(self.BlockTable), fx.make_layout(1, 1)
            )
            self.page_base = fx.Int64(self.batch_idx) * fx.Int64(
                self.block_table_stride
            )
        if const_expr(self.traits.PAIRED_PAGE_IDS):
            if const_expr(not self.traits.CSR_PAGE_TABLE):
                request = fx.make_view(
                    fx.add_offset(fx.get_iter(self.BlockTable), self.page_base),
                    fx.make_layout(fx.Int64(self.block_table_stride), 1),
                )
            num_pages = (
                self.seqlen_kv_v + self.traits.PAGE_SIZE - 1
            ) // self.traits.PAGE_SIZE
            if const_expr(self.traits.CSR_PAGE_TABLE):
                num_pages = fx.min(
                    fx.Int64(num_pages), fx.Int64(self.request_page_count)
                )
            bounded = fx.rocdl.make_buffer_tensor(
                request, num_records_bytes=(num_pages * 4).ir_value()
            )
            self.page_pair_indices_div = fx.logical_divide(
                bounded, fx.make_layout(1, 1)
            )

    def load_page_id(self, tile_start, *, uniform=True):
        """Load a physical page ID, zero-filling lookups past this request's pages."""
        local_page = fx.Int64(tile_start) // self.traits.PAGE_SIZE
        page_idx = self.page_base + local_page
        if const_expr(self.traits.PAGE_SIZE == self.traits.BLOCK_N):
            num_pages = self.num_kv_tiles
        else:
            num_pages = (
                self.seqlen_kv_v + self.traits.PAGE_SIZE - 1
            ) // self.traits.PAGE_SIZE
        valid = local_page < num_pages
        safe_idx = valid.select(page_idx, 0)
        v = fly.copy_atom_call_ssa(
            [self.page_v1i32],
            self.page_i32_atom,
            fx.slice(self.page_indices_div, (None, fx.Int32(safe_idx))),
        )
        page = valid.select(Vec(v, (1,), fx.Int32)[0], fx.Int32(0))
        if uniform:
            return fx.Int64(rocdl.readfirstlane(T.i32, page.ir_value()))
        return fx.Int64(page)

    def load_page_id_pair(self, tile_start):
        """Return reusable page IDs for a compute-tile pair, or defer to producers."""
        if const_expr(self.traits.PAGE_SIZE < self.traits.BLOCK_N):
            layout = self.traits.SHARED_PAGE_ID_LAYOUT
            if const_expr(layout is None):
                return None, None
            if const_expr(layout == "tokens"):
                token = fx.Int64(tile_start) + self.lane_in_warp
            elif const_expr(layout == "page_waves"):
                token = fx.Int64(tile_start) + (
                    self.wave_id_uni % fx.Int64(4)
                ) * fx.Int64(16)
            else:
                token = fx.Int64(tile_start) + (
                    self.lane_in_warp // fx.Int64(16)
                ) * fx.Int64(16)
            uniform = layout == "page_waves"
            return self.load_page_id(token, uniform=uniform), self.load_page_id(
                token + 64, uniform=uniform
            )
        if const_expr(not self.traits.PAIRED_PAGE_IDS):
            return self.load_page_id(tile_start), self.load_page_id(
                tile_start + self.traits.BLOCK_N
            )
        local_page = fx.Int32(fx.Int64(tile_start) // self.traits.PAGE_SIZE)
        # gfx950 zero-fills each out-of-range dword, including an odd pair tail.
        v = fly.copy_atom_call_ssa(
            [self.page_v2i32],
            self.page_i32x2_atom,
            fx.slice(self.page_pair_indices_div, (None, local_page)),
        )
        pages = Vec(v, (2,), fx.Int32)
        return (
            fx.Int64(rocdl.readfirstlane(T.i32, pages[0].ir_value())),
            fx.Int64(rocdl.readfirstlane(T.i32, pages[1].ir_value())),
        )

    def make_page_view(self, tensor_iter, page_id, *, is_value=False):
        page_bytes = self.v_page_bytes if is_value else self.k_page_bytes
        # Rebase in 64 bits before constructing the bounded per-page resource.
        page_ptr = fx.add_offset(tensor_iter, page_id * page_bytes)
        page = fx.make_view(page_ptr, fx.make_layout(page_bytes, 1))
        return fx.logical_divide(
            fx.rocdl.make_buffer_tensor(page, num_records_bytes=page_bytes),
            fx.make_layout(1, 1),
        )

    def read_i32x4_lds(self, byte_row):
        tile = fx.slice(self.k_lds_i32_tiles, (None, fx.Uint32(byte_row) // 16))
        return _load(fx.get_iter(tile), dtype=fx.Int32, count=4).ir_value()

    def preserve_accumulators(self, v_o):
        # Preserve FP expression boundaries without inline-assembly pins.
        return [llvm.intr_arithmetic_fence(fx.as_ir_value(acc)) for acc in v_o]
