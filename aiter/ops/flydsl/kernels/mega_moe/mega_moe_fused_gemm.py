# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""GEMM1/GEMM2 tiles, buffer helpers and launch configs shared by the MegaMoEV2 small- and large-input kernels."""

from bisect import bisect_left
from dataclasses import dataclass
from functools import cache
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import (
    Float32,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec

from ..mxfp4_gemm_common import _lds_swizzle_mask as lds_swizzle_mask
from ..mxfp4_gemm_common import (
    flat_buffer_view,
    global_typed_ptr,
    kBS_stride_k0_dw,
    kStages,
    lds_dma_atom_128,
    lds_dma_dst,
    lds_swizzle_mask_f8,
    lds_vec_load,
)
from ..tensor_shim import _fx_value_elem, buf_copy_load, buf_copy_store, ptr_buf_tensor
from .gemm2 import mma_one_j, scale_mma_atoms, scale_view
from .mega_moe_config import (
    LARGE_M_MTPRS,
    REFERENCE_EXPERTS_PER_RANK,
    _fixed_dispatch_cu,
)

TOKEN_BUCKETS = (
    1,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
)
# The small-input single kernel keeps fixed slots up to this MTPR (past it the main path is faster).
SMALL_M_FIXED_SLOT_MAX_MTPR = 512


@dataclass(frozen=True, slots=True)
class Stage1Config:
    sort_block_m: int
    tile_n: int
    num_waves: int
    grid_mult: int
    num_dispatch_cu: int
    mfma_amajor: bool
    async_a_copy: bool
    use_tile_resource: bool
    b_nt: int
    waves_per_eu_hint: int = 2
    tile_k: int = 256
    pipe_weights: bool = True
    swizzle_a: bool = True
    work_shards: int = 8
    payload_chunk_rows: int = 0


@dataclass(frozen=True, slots=True)
class Stage2Config:
    block_m: int
    block_n: int
    persist: bool
    persist_cu: int
    use_nt: bool
    persist_strided: bool = False
    skew_cu: int = 0
    block_k: int = 256
    b_hoist: bool = True
    ascale_prefetch: bool = True
    spatial_partition: int = 402
    bf16_lds: bool = False


@dataclass(frozen=True, slots=True)
class MegaMoEConfig:
    stage1: Stage1Config
    stage2: Stage2Config
    p2p_quant: str

    def __post_init__(self):
        sbm = self.stage1.sort_block_m
        bm = self.stage2.block_m
        if bm > sbm or sbm % bm:
            raise ValueError(
                f"Stage2 block_m={bm} must divide Stage1 sort_block_m={sbm}"
            )
        if self.p2p_quant not in ("none", "fp8_blockwise_1x32"):
            raise ValueError(f"unsupported p2p_quant={self.p2p_quant!r}")
        if self.p2p_quant != "none" and self.stage2.bf16_lds:
            raise ValueError("FP8 P2P requires Stage2 bf16_lds=False")


def nearest_token_bucket(tokens: int) -> int:
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    index = bisect_left(TOKEN_BUCKETS, tokens)
    if index == 0:
        return TOKEN_BUCKETS[0]
    if index == len(TOKEN_BUCKETS):
        return TOKEN_BUCKETS[-1]
    lower, upper = TOKEN_BUCKETS[index - 1], TOKEN_BUCKETS[index]
    return upper if upper - tokens <= tokens - lower else lower


def _select_fixed_stage1(bucket: int) -> Stage1Config:
    grid_mult = max(1, bucket // 4) if bucket <= 16 else 3
    return Stage1Config(
        sort_block_m=32,
        tile_n=256 if bucket <= 8 else 128,
        num_waves=4,
        grid_mult=grid_mult,
        num_dispatch_cu=_fixed_dispatch_cu(bucket),
        mfma_amajor=False,
        async_a_copy=False,
        use_tile_resource=bucket <= 16,
        b_nt=0 if bucket == 1 else 3,
        waves_per_eu_hint=1 if bucket == 16 else 2,
    )


def _select_fixed_stage2(bucket: int, model_dim: int) -> Stage2Config:
    block_n = 256 if bucket in (1, 4, 64) or bucket >= 1024 else 128
    if model_dim < 4096:
        block_n = 128
    persist = bucket >= 128
    return Stage2Config(
        block_m=32,
        block_n=block_n,
        persist=persist,
        persist_cu=128 if bucket == 256 else 240 if persist else 0,
        use_nt=bucket <= 128,
        persist_strided=512 <= bucket <= 1024,
    )


def _large_config(model_dim: int, inter_dim: int) -> MegaMoEConfig:
    stage1 = Stage1Config(
        sort_block_m=128,
        tile_n=512 if inter_dim >= 2048 else 256,
        num_waves=8,
        grid_mult=1,
        num_dispatch_cu=96,
        mfma_amajor=True,
        async_a_copy=True,
        use_tile_resource=True,
        b_nt=0,
        work_shards=4,
        payload_chunk_rows=384,
    )
    stage2 = Stage2Config(
        block_m=64,
        block_n=128 if model_dim < 4096 else 256,
        persist=True,
        persist_cu=240,
        use_nt=False,
        skew_cu=96,
    )
    return MegaMoEConfig(stage1=stage1, stage2=stage2, p2p_quant="fp8_blockwise_1x32")


@cache
def _select_config(
    bucket: int, mtpr: int, model_dim: int, inter_dim: int
) -> MegaMoEConfig:
    if mtpr in LARGE_M_MTPRS:
        return _large_config(model_dim, inter_dim)
    stage1 = _select_fixed_stage1(bucket)
    stage2 = _select_fixed_stage2(bucket, model_dim)
    return MegaMoEConfig(stage1=stage1, stage2=stage2, p2p_quant="none")


def select_mega_moe_config(
    tokens: int,
    mtpr: int,
    *,
    experts_per_rank: int = REFERENCE_EXPERTS_PER_RANK,
    model_dim: int = 7168,
    inter_dim: int = 3072,
) -> MegaMoEConfig:
    """Small-input (fixed-slot, MTPR <= 512) or large-input (MTPR in LARGE_M_MTPRS) config for ``tokens`` per rank."""
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(f"mtpr={mtpr} must be a positive power of two")
    if mtpr > SMALL_M_FIXED_SLOT_MAX_MTPR and mtpr not in LARGE_M_MTPRS:
        raise ValueError(f"no small- or large-input config for mtpr={mtpr}")
    if tokens > mtpr:
        raise ValueError(f"tokens={tokens} exceeds mtpr={mtpr}")
    if experts_per_rank <= 0 or experts_per_rank > 64:
        raise ValueError(f"experts_per_rank must be in [1, 64], got {experts_per_rank}")
    if model_dim <= 0 or inter_dim <= 0:
        raise ValueError(f"invalid model shape {model_dim}x{inter_dim}")
    return _select_config(
        nearest_token_bucket(tokens),
        mtpr,
        model_dim,
        inter_dim,
    )


class AddrBuffer(NamedTuple):
    """A raw device address and an optional hardware OOB bound in bytes."""

    addr: object
    num_records_bytes: object = None


def _addr_view(buf, elem, width):
    # unit_stride=1: a wide access may start at any element offset.
    return ptr_buf_tensor(
        buf.addr,
        elem,
        unit_elems=width,
        unit_stride=1,
        num_records_bytes=buf.num_records_bytes,
    )


def load_at(buf, index, elem=fx.Int32, width=1, cache_modifier=0, mask=None):
    """Load ``width`` x ``elem`` at element ``index`` of ``buf``.

    A false ``mask`` moves the access past the buffer bound.
    """
    if mask is not None:
        index = fx.Boolean(mask).select(
            index, fx.Int32(0x7FFFFFFF // (elem.width // 8))
        )
    return buf_copy_load(
        _addr_view(buf, elem, width), index, elem, width, cache_modifier
    )


def store_at(value, buf, index, cache_modifier=0, offset_is_bytes=False):
    """Store ``value`` (scalar or vector) at element ``index`` of ``buf``."""
    elem, width = _fx_value_elem(value)
    if offset_is_bytes:
        index = fx.Int32(index) // fx.Int32(elem.width // 8)
    buf_copy_store(
        _addr_view(buf, elem, width), index, value, elem, width, cache_modifier
    )


def _make_buffer(tensor, elem_ty, width=1, *, max_size=True, num_records_bytes=None):
    alignment = max(1, elem_ty.width * width // 8)
    ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Global, alignment)
    base = fx.inttoptr(ptr_ty, fx.Int64(fx.ptrtoint(fx.get_iter(tensor))))
    view = fx.Tensor(fx.make_view(base, fx.make_layout((width, 1), (1, width))))
    return fx.rocdl.make_buffer_tensor(
        view, max_size=max_size, num_records_bytes=num_records_bytes
    )


def _make_buffer_from_addr(addr, elem_ty, width=1, *, num_records_bytes=None):
    alignment = max(1, elem_ty.width * width // 8)
    ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Global, alignment)
    base = fx.inttoptr(ptr_ty, fx.Int64(addr))
    view = fx.Tensor(fx.make_view(base, fx.make_layout((width, 1), (1, width))))
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)


def _buffer_load(buffer, group_index, elem_ty, width=1, cache_modifier=0):
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(elem_ty.width * width, cache_modifier), elem_ty
    )
    fragment = fx.make_rmem_tensor(width, elem_ty)
    fx.copy(atom, fx.slice(buffer, (None, group_index)), fragment)
    value = Vec(fragment.load())
    return value[0] if width == 1 else value


def _buffer_store(buffer, group_index, value, elem_ty, width=1, cache_modifier=0):
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(elem_ty.width * width, cache_modifier), elem_ty
    )
    fragment = fx.make_rmem_tensor(width, elem_ty)
    fragment.store(Vec.from_elements([value], elem_ty) if width == 1 else Vec(value))
    fx.copy(atom, fragment, fx.slice(buffer, (None, group_index)))


def issue_a_load_lds_dt(
    arg_aq,
    s_aq_base,
    slot,
    kt,
    m_row,
    wave,
    lane,
    is_f8,
    KH_TILE_A,
    K_BYTES,
    BM=32,
    valid_rows=None,
    NW=4,
):
    """Load one A tile through a tile-local descriptor so allocations may span the 4 GiB buffer ABI."""
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4) / 16 (fp8)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4) / 4 (fp8)
    a_lane_row = lane // lanes_per_row
    rows_per_wave = BM // NW  # rows each wave loads (NW4: BM32 8, BM64 16; NW8: BM64 8)
    # BM16 fp4: partial-wave round-robin (waves 2,3 re-load, harmless); BM>=32 byte-identical per-wave blocks.
    partial_wave_gather = rows_per_wave < rows_per_call
    if const_expr(partial_wave_gather):
        n_gather_calls = BM // rows_per_call
        gather_base_row = (wave % fx.Int32(n_gather_calls)) * rows_per_call
        n_row_groups = 1
    else:
        gather_base_row = wave * rows_per_wave
        n_row_groups = rows_per_wave // rows_per_call
    lane_col = (lane % lanes_per_row) * 16
    base_i32 = s_aq_base
    atom = lds_dma_atom_128()
    tile_base = fx.Int64(arg_aq) + fx.Int64(m_row) * fx.Int64(K_BYTES)
    src = flat_buffer_view(
        tile_base,
        None,
        T.i32,
        align=16,
        elem_bytes=4,
        fold=False,
        num_records_bytes=fx.Int64(BM if valid_rows is None else valid_rows)
        * fx.Int64(K_BYTES),
    )
    for g in range_constexpr(n_row_groups):
        lds_row = gather_base_row + g * rows_per_call
        mask = (
            lds_swizzle_mask_f8(lds_row + a_lane_row, KH_TILE_A)
            if const_expr(is_f8)
            else lds_swizzle_mask(lds_row + a_lane_row)
        )
        tile_row = lds_row + a_lane_row
        voffset = (lane_col ^ mask) + tile_row * K_BYTES
        off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
        v_e = (voffset + kt * KH_TILE_A) // 4  # per-lane i32-elem index
        fx.copy(
            atom, src[v_e, None], lds_dma_dst(base_i32, off, elem_ty=T.i32, align=16)
        )


@flyc.jit
def gemm2_compute_v2(
    lds_base_i32,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_aq,
    i32_max_m_blocks,
    bx_i32,
    lane,
    wave,
    i32_inter,
    i32_hidden,
    i32_kpad,
    i32_npad,
    *,
    BM,
    BN=256,
    BK=256,
    use_nt,
    INTER_MAX,
    aStages,
    a_dtype,
    has_pad=False,
    SBM=None,
    g2_bhoist=True,
    g2_ascale_pf=True,
    expert_offset=0,
    valid_rows=None,
    NW=4,
    early_b_load=False,
):
    """Run the GEMM2 K-loop and return accumulators for the selected epilogue."""
    # SBM is the sort padding unit; BM is the compute tile and must divide SBM.
    if SBM is None:
        SBM = BM
    kMChunks = BM // 16  # 16-row MFMA row-groups
    kHalves = BK // 128  # 16x16x128 MFMA K-steps per K-tile
    tilesPerScaleChunk = 256 // BK  # K-tiles sharing one 256-K E8M0 word
    numAccN = (BN // NW) // 16  # 16-column MFMA subblocks per wave
    nPairs = max(1, numAccN // 2)  # one B-scale per two 16-column subblocks
    # BM16: single 16-row block owning a 32-row scale chunk (chunk==m_block_idx, rg0-only).
    is_bm16 = BM < 32
    rg_off = 0
    kScaleSubBlocks = max(1, kMChunks // 2)
    is_f8_a = a_dtype == "fp8"  # only the A path differs
    a_pack = 1 if is_f8_a else 2
    KH_TILE_A = BK // a_pack
    slot_bytes = BM * KH_TILE_A
    # Contraction K = inter_dim runtime (i32_inter); INTER_MAX caps compile-time view/fragment bounds.
    K_rt = fx.Int32(i32_inter)
    K_BYTES = K_rt // fx.Int32(a_pack)  # A row stride bytes (runtime)
    kc_rt = K_rt // fx.Int32(256)  # (K//32)//4//2
    K_TILES_RT = K_rt // fx.Int32(BK)  # runtime K-tile trip count
    kAS_per_chunk_dw = kc_rt * fx.Int32(64)
    kBS_stride_n0_dw = kc_rt * fx.Int32(64)
    # N_OUT = model_dim/hidden is the gemm2 output N dim; runtime via i32_hidden (no K-loop dependency).
    N_OUT_rt = fx.Int32(i32_hidden)
    kbs_per_expert_dw = (
        N_OUT_rt // fx.Int32(32)
    ) * kBS_stride_n0_dw  # (N_OUT//16//2)*stride
    num_n_blocks = N_OUT_rt // fx.Int32(BN)
    KH4 = K_rt // fx.Int32(8)  # i32 col stride (= K_HALF//4)
    K_SCALE_CHUNKS_MAX = INTER_MAX // 256

    # Padded shapes mask weight tiles beyond the real K/N extents.
    N_real = None
    if const_expr(has_pad):
        K_real = K_rt - fx.Int32(i32_kpad)
        halves_real = (K_real + fx.Int32(127)) // fx.Int32(128)
        N_real = N_OUT_rt - fx.Int32(i32_npad)

    # Map each compute block to its SBM-padded expert metadata row.
    m_block_idx = bx_i32 // num_n_blocks
    n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    eids_ptr = global_typed_ptr(arg_eids, T.i32)
    if const_expr(SBM == BM):
        e = rocdl.readfirstlane(T.i32, eids_ptr[m_block_idx])
        m_row = m_block_idx * BM
    else:
        m_row = m_block_idx * BM
        e = rocdl.readfirstlane(T.i32, eids_ptr[m_row // fx.Int32(SBM)])
    if const_expr(expert_offset != 0):
        e = e - fx.Int32(expert_offset)

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    s_aq_base = lds_base_i32
    mma_atoms = scale_mma_atoms(a_dtype)

    # A activation: global->LDS DMA (issue_a_load_lds), then LDS->reg ds-read (issue_a_ds_read).
    A_NDW = (
        8 if is_f8_a else 4
    )  # fp8 packs two 128-K halves -> i32<8:1>; fp4 -> i32<4:1>
    a_frags = [
        [fx.make_rmem_tensor(A_NDW, Int32) for _ in range_constexpr(kHalves)]
        for _ in range_constexpr(kMChunks)
    ]

    def issue_a_load_lds(slot, kt):
        issue_a_load_lds_dt(
            arg_aq,
            s_aq_base,
            slot,
            kt,
            m_row,
            wave,
            lane,
            is_f8_a,
            KH_TILE_A,
            K_BYTES,
            BM=BM,
            valid_rows=valid_rows,
            NW=NW,
        )

    def issue_a_ds_read(slot):
        # A ds-read for one slot into a_frags: fp8 -> i32<8:1> (two 128-K halves), fp4 -> i32<4:1>.
        for k in range_constexpr(kHalves):
            for i in range_constexpr(kMChunks):
                lds_row = lane_mod_16 + i * 16
                row_off = fx.Int32(slot * slot_bytes) + lds_row * KH_TILE_A
                if const_expr(is_f8_a):
                    mask = lds_swizzle_mask_f8(lane_mod_16, KH_TILE_A)
                    col0 = lane_div_16 * 16 + k * 128
                    col_lo = col0 ^ mask
                    col_hi = (col0 + 64) ^ mask
                    lo = Vec(
                        lds_vec_load(
                            s_aq_base,
                            row_off + col_lo,
                            Vec.make_type(2, fx.Int64),
                            fx.Int64,
                            align=16,
                        )
                    )
                    hi = Vec(
                        lds_vec_load(
                            s_aq_base,
                            row_off + col_hi,
                            Vec.make_type(2, fx.Int64),
                            fx.Int64,
                            align=16,
                        )
                    )
                    a64 = Vec.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int64)
                    a_frags[i][k].store(a64.bitcast(fx.Int32))
                else:
                    mask = lds_swizzle_mask(lane_mod_16)
                    lds_col = (lane_div_16 * 16 + k * 64) ^ mask
                    vec = lds_vec_load(
                        s_aq_base,
                        row_off + lds_col,
                        Vec.make_type(4, fx.Int32),
                        fx.Int32,
                        align=16,
                    )
                    a_frags[i][k].store(Vec(vec))

    # The shared e8m0 scale layout bounds each A-scale view to its remaining bytes.
    sc_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

    asc_per_mb = fx.Int32(kScaleSubBlocks) * kAS_per_chunk_dw * fx.Int32(4)
    asc_num = fx.Int64(i32_max_m_blocks) * fx.Int64(asc_per_mb)
    scale_chunk0 = m_block_idx if const_expr(is_bm16) else m_row // 32

    def make_ascale_view(sub):
        base_dw = (scale_chunk0 + fx.Int32(sub)) * kAS_per_chunk_dw
        nrec = asc_num - fx.Int64(base_dw) * fx.Int64(4)
        return scale_view(
            arg_ascale,
            base_dw,
            K_SCALE_CHUNKS_MAX,
            k0_stride_dw=64,
            num_records_bytes=nrec,
        )

    ascale_views = [make_ascale_view(sub) for sub in range_constexpr(kScaleSubBlocks)]
    sc_frag_tmpl = ascale_views[0][0, 0, 0, None]  # i32<1:1> (one e8m0 word)

    def load_a_scale_tile(kt):
        # One i32 A-scale register per 32-row chunk (kScaleSubBlocks).
        chunk_kt = (
            kt
            if const_expr(tilesPerScaleChunk == 1)
            else kt // fx.Int32(tilesPerScaleChunk)
        )
        out = []
        for sub in range_constexpr(kScaleSubBlocks):
            saf = fx.make_fragment_like(sc_frag_tmpl)
            fx.copy(
                sc_copy_atom,
                ascale_views[sub][lane_div_16, lane_mod_16, chunk_kt, None],
                saf,
            )
            out.append(Vec(saf.load())[0])
        return out

    # Stream B weights and scales through registers so use_nt reaches the ISA cache policy.
    bq_rsrc = AddrBuffer(arg_bq)

    bq_base_dw = [
        rocdl.readfirstlane(
            T.i32,
            (e * N_OUT_rt + n_block_idx * BN + wave * (BN // NW) + j * 16) * KH4,
        )
        for j in range_constexpr(numAccN)
    ]

    mni_base = n_block_idx * (BN // 16 // 2) + wave * (BN // (16 * NW) // 2)
    bscale_views = [
        scale_view(
            arg_bscale,
            e * kbs_per_expert_dw + (mni_base + mw) * kBS_stride_n0_dw,
            K_SCALE_CHUNKS_MAX,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(nPairs)
    ]

    frag_tmpl = fx.make_rmem_tensor(4, Int32)
    # B-scale word template shares the A-scale layout (sc_frag_tmpl).

    def issue_b_load_into(bqf, bsf, kt_rt):
        # Issue B-weight + B-scale vmem loads for K-tile kt_rt into the given (per-stage) fragments.
        for j in range_constexpr(numAccN):
            for half in range_constexpr(kHalves):
                bq_off_dw = (
                    bq_base_dw[j]
                    + lane_div_16 * fx.Int32(64)
                    + lane_mod_16 * fx.Int32(4)
                    + kt_rt * fx.Int32(kHalves * 256)
                    + fx.Int32(half * 256)
                )
                load_mask = None
                if const_expr(has_pad):
                    col = n_block_idx * BN + wave * (BN // NW) + j * 16
                    load_mask = (col < N_real) & (
                        kt_rt * fx.Int32(kHalves) + fx.Int32(half) < halves_real
                    )
                bq_vec = load_at(
                    bq_rsrc, bq_off_dw, fx.Int32, 4, 2 if use_nt else 0, mask=load_mask
                )
                bqf[j][half].store(Vec(bq_vec))
        chunk_kt = (
            kt_rt
            if const_expr(tilesPerScaleChunk == 1)
            else kt_rt // fx.Int32(tilesPerScaleChunk)
        )
        for mw in range_constexpr(nPairs):
            fx.copy(
                sc_copy_atom,
                bscale_views[mw][lane_div_16, lane_mod_16, chunk_kt, None],
                bsf[mw],
            )

    def stream_b_tile(kt_rt):
        # Fresh per-iter fragments (B streamed, not register-resident) then issue_b_load_into.
        bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
            for _ in range_constexpr(numAccN)
        ]
        bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        issue_b_load_into(bqf, bsf, kt_rt)
        return bqf, bsf

    # Scaled-MFMA clusters over the loaded A / B / scale fragments.
    def shift_scale_word(scale, kt_rt):
        if const_expr(tilesPerScaleChunk == 1):
            return scale
        scale_shift = (kt_rt % fx.Int32(tilesPerScaleChunk)) * fx.Int32(16)
        return fx.Int32(scale).shrui(scale_shift)

    def mfma_cluster(bqf, bsf, sa, kt_rt):
        # opsel (no gate/up split): mni=J//2, in_b=J%2; sa is a per-32-row-chunk list.
        sa = [
            shift_scale_word(sa[sub], kt_rt) for sub in range_constexpr(kScaleSubBlocks)
        ]
        sb_words = [
            shift_scale_word(Vec(bsf[mni].load())[0], kt_rt)
            for mni in range_constexpr(nPairs)
        ]
        for J in range_constexpr(numAccN):
            mni, in_b = J // 2, J % 2
            sb = sb_words[mni]
            if const_expr(is_bm16):
                mma_one_j(
                    J,
                    in_b,
                    sa[0],
                    sb,
                    bqf,
                    a_frags,
                    c_frags,
                    mma_atoms,
                    i0=0,
                    single_rg=True,
                    rg_off=rg_off,
                    k_halves=kHalves,
                )
                continue
            for sub in range_constexpr(kScaleSubBlocks):
                mma_one_j(
                    J,
                    in_b,
                    sa[sub],
                    sb,
                    bqf,
                    a_frags,
                    c_frags,
                    mma_atoms,
                    i0=2 * sub,
                    k_halves=kHalves,
                )

    # C accumulator: register fragments, zeroed then accumulated in place; (un)packed to K-loop carry.
    zero4 = Vec.filled(4, 0.0, Float32)
    c_frags = [
        [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(numAccN)]
        for _ in range_constexpr(kMChunks)
    ]
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(numAccN):
            c_frags[i][J].store(zero4)

    def load_c_carry():
        return [c_frags[i][J].load() for i in range(kMChunks) for J in range(numAccN)]

    def store_c_carry(state):
        n = 0
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(numAccN):
                c_frags[i][J].store(state[n])
                n += 1
        return n

    if const_expr(BM == 64 and BN // NW == 64):
        # BM64 with a 64-column per-wave slab (NW4 BN256 / NW8 BN512) uses the 1-stage B path.
        for kt_iv, state in range(
            fx.Int32(0),
            K_TILES_RT,
            fx.Int32(1),
            init=load_c_carry(),
        ):
            store_c_carry(state)
            kt_rt = fx.Int32(kt_iv)
            if const_expr(early_b_load):
                # Issue this K-tile's B and A-scale loads before the barrier and the next A-tile DMA:
                # vmcnt retires in issue order, so waiting for them leaves that DMA in flight.
                bqf, bsf = stream_b_tile(kt_rt)
                sa = load_a_scale_tile(kt_rt)
            gpu.barrier()
            issue_a_ds_read(kt_rt % fx.Int32(aStages))
            nxt = kt_rt + fx.Int32(kStages)
            if nxt < K_TILES_RT:
                issue_a_load_lds(nxt % fx.Int32(aStages), nxt)
            if const_expr(not early_b_load):
                bqf, bsf = stream_b_tile(kt_rt)
                sa = load_a_scale_tile(kt_rt)
            mfma_cluster(bqf, bsf, sa, kt_rt)
            results = yield load_c_carry()
        store_c_carry(results)
    else:
        # 2-stage B pipeline: consume carried "current" B, prefetch next tile into the same fragments via scf.for state.
        cur_bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
            for _ in range_constexpr(numAccN)
        ]
        cur_bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        nxt_bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
            for _ in range_constexpr(numAccN)
        ]
        nxt_bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        # g2_ascale_pf: carry the A-scale through scf.for state, same rotating-buffer model as B.
        cur_saf = nxt_saf = None
        if const_expr(g2_ascale_pf):
            cur_saf = [
                fx.make_fragment_like(sc_frag_tmpl)
                for _ in range_constexpr(kScaleSubBlocks)
            ]
            nxt_saf = [
                fx.make_fragment_like(sc_frag_tmpl)
                for _ in range_constexpr(kScaleSubBlocks)
            ]

        def load_b_carry():
            # Flat CURRENT (to-consume) B-weight, B-scale, then (opt) A-scale values.
            out = []
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    out.append(cur_bqf[j][half].load())
            for mw in range_constexpr(nPairs):
                out.append(cur_bsf[mw].load())
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    out.append(cur_saf[sub].load())
            return out

        def store_b_carry(state, base):
            n = base
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    cur_bqf[j][half].store(state[n])
                    n += 1
            for mw in range_constexpr(nPairs):
                cur_bsf[mw].store(state[n])
                n += 1
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    cur_saf[sub].store(state[n])
                    n += 1
            return n

        def rotate_b_carry():
            # Yield the PREFETCHED (next-tile) values -> become "current" next iteration.
            out = []
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    out.append(nxt_bqf[j][half].load())
            for mw in range_constexpr(nPairs):
                out.append(nxt_bsf[mw].load())
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    out.append(nxt_saf[sub].load())
            return out

        def issue_a_scale_load_into(saf, kt_rt):
            # A-scale vmem load(s) for K-tile kt_rt into the given (per-stage) fragment(s).
            sa = load_a_scale_tile(kt_rt)
            for sub in range_constexpr(kScaleSubBlocks):
                saf[sub].store(Vec.filled(1, sa[sub], Int32))

        def load_carry():
            return load_c_carry() + load_b_carry()

        def store_carry(state):
            base = store_c_carry(state)
            store_b_carry(state, base)

        def yield_carry():
            return load_c_carry() + rotate_b_carry()

        # Prologue: prefetch tile 0's B/B-scale into "current" (VALUES enter via init=load_carry()).
        issue_b_load_into(cur_bqf, cur_bsf, fx.Int32(0))
        if const_expr(g2_ascale_pf):
            issue_a_scale_load_into(cur_saf, fx.Int32(0))
        rocdl.sched_barrier(0)

        def prefetch_next_b(kt_rt):
            # Prefetch NEXT tile's B; if none, copy current through (rotate_b_carry state, unused after loop).
            nxt_b = kt_rt + fx.Int32(1)
            if nxt_b < K_TILES_RT:
                issue_b_load_into(nxt_bqf, nxt_bsf, nxt_b)
                if const_expr(g2_ascale_pf):
                    issue_a_scale_load_into(nxt_saf, nxt_b)
            else:
                for j in range_constexpr(numAccN):
                    for half in range_constexpr(kHalves):
                        nxt_bqf[j][half].store(cur_bqf[j][half].load())
                for mw in range_constexpr(nPairs):
                    nxt_bsf[mw].store(cur_bsf[mw].load())
                if const_expr(g2_ascale_pf):
                    for sub in range_constexpr(kScaleSubBlocks):
                        nxt_saf[sub].store(cur_saf[sub].load())

        for kt_iv, state in range(
            fx.Int32(0),
            K_TILES_RT,
            fx.Int32(1),
            init=load_carry(),
        ):
            store_carry(state)
            kt_rt = fx.Int32(kt_iv)
            if const_expr(g2_bhoist):
                prefetch_next_b(kt_rt)
            gpu.barrier()
            issue_a_ds_read(kt_rt % fx.Int32(aStages))
            nxt_a = kt_rt + fx.Int32(kStages)
            if nxt_a < K_TILES_RT:
                issue_a_load_lds(nxt_a % fx.Int32(aStages), nxt_a)
            # A-scale from the prefetch carry (g2_ascale_pf) or loaded synchronously here.
            if const_expr(g2_ascale_pf):
                sa = [
                    Vec(cur_saf[sub].load())[0]
                    for sub in range_constexpr(kScaleSubBlocks)
                ]
            else:
                sa = load_a_scale_tile(kt_rt)
            if const_expr(not g2_bhoist):
                prefetch_next_b(kt_rt)
            # Fence the MFMA chain from the B vmem loads (next-tile loads ride ahead of compute).
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(cur_bqf, cur_bsf, sa, kt_rt)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            results = yield yield_carry()
        store_carry(results)

    # Load the C fragments (fp8/fp4 unified onto the same fx.gemm path) and hand them to the epilog.
    accm_vecs = [
        [c_frags[i][J].load() for J in range(numAccN)] for i in range(kMChunks)
    ]
    return accm_vecs, m_row, n_block_idx, N_OUT_rt
